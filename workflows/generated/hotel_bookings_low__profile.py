import pandas as pd
import numpy as np
import re
from datetime import datetime
import difflib

INPUT_PATH = r"benchmark\datasets\hotel_bookings\noisy_low.csv"
OUTPUT_PATH = r"results\cleaned_datasets\hotel_bookings\noisy_low__profile.csv"

def extract_numeric(value):
    if pd.isna(value):
        return np.nan
    s = str(value).replace(",", ".")
    s = s.replace("O", "0").replace("o", "0")
    match = re.search(r"-?\d+\.?\d*", s)
    return float(match.group()) if match else np.nan

def harmonize_category(value, valid_values):
    if pd.isna(value):
        return value
    s = str(value).strip()
    if s in valid_values:
        return s
    for v in valid_values:
        if v.lower() == s.lower():
            return v
    match = difflib.get_close_matches(s, valid_values, n=1, cutoff=0.6)
    return match[0] if match else value

def parse_date(value):
    if pd.isna(value):
        return value
    date_formats = [
        "%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%B %d, %Y",
        "%d-%b-%Y", "%Y/%m/%d"
    ]
    for fmt in date_formats:
        try:
            return datetime.strptime(str(value), fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    try:
        if str(value).isdigit() and len(str(value)) in (9, 10):
            return datetime.fromtimestamp(int(value)).strftime("%Y-%m-%d")
    except (ValueError, OSError):
        pass
    return str(value)

def clean_hotel_bookings():
    df = pd.read_csv(INPUT_PATH)

    # Log initial
    initial_rows, initial_cols = df.shape
    print(f"Dataset initial: {initial_rows} lignes, {initial_cols} colonnes")

    # 1. Suppression des doublons (en conservant row_id)
    duplicates = df.duplicated(subset=df.columns.difference(['row_id']), keep='first')
    df = df[~duplicates]
    print(f"Doublons supprimés: {duplicates.sum()}")

    # 2. Nettoyage colonne par colonne
    # hotel (categorical) - harmonisation des variantes
    valid_hotel = ["City Hotel", "Resort Hotel"]
    df["hotel"] = df["hotel"].apply(lambda v: harmonize_category(v, valid_hotel))

    # is_canceled (numeric) - déjà propre (0/1)
    # lead_time (numeric) - extraction des valeurs numériques
    df["lead_time"] = df["lead_time"].apply(extract_numeric)
    # Imputation des valeurs manquantes (0% dans le profil mais au cas où)
    if df["lead_time"].isna().any():
        df["lead_time"] = df["lead_time"].fillna(df["lead_time"].median())

    # arrival_date_year (numeric) - déjà propre
    # arrival_date_month (categorical) - harmonisation
    valid_months = [
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December"
    ]
    df["arrival_date_month"] = df["arrival_date_month"].apply(
        lambda v: harmonize_category(v, valid_months) if pd.notna(v) else v
    )

    # arrival_date_week_number (numeric) - déjà propre
    # arrival_date_day_of_month (numeric) - déjà propre
    # stays_in_weekend_nights (numeric) - déjà propre
    # stays_in_week_nights (numeric) - valeurs aberrantes (max 999)
    q99 = df["stays_in_week_nights"].quantile(0.995)
    outliers = df["stays_in_week_nights"] > q99
    if outliers.any():
        median_val = df.loc[~outliers, "stays_in_week_nights"].median()
        df.loc[outliers, "stays_in_week_nights"] = median_val
        print(f"Valeurs aberrantes corrigées dans stays_in_week_nights: {outliers.sum()}")

    # adults (numeric) - extraction des valeurs numériques
    df["adults"] = df["adults"].apply(extract_numeric)
    # Imputation des valeurs manquantes (0% dans le profil mais au cas où)
    if df["adults"].isna().any():
        df["adults"] = df["adults"].fillna(df["adults"].median())

    # children (numeric) - extraction des valeurs numériques et imputation
    df["children"] = df["children"].apply(extract_numeric)
    # Remplacement des valeurs aberrantes (max 10 dans le profil)
    df.loc[df["children"] > 10, "children"] = df["children"].median()
    # Imputation des valeurs manquantes (8.32%)
    mode_children = df["children"].mode()[0]
    df["children"] = df["children"].fillna(mode_children)

    # babies (numeric) - valeurs aberrantes (max 49)
    q99_babies = df["babies"].quantile(0.995)
    outliers_babies = df["babies"] > q99_babies
    if outliers_babies.any():
        median_babies = df.loc[~outliers_babies, "babies"].median()
        df.loc[outliers_babies, "babies"] = median_babies
        print(f"Valeurs aberrantes corrigées dans babies: {outliers_babies.sum()}")

    # meal (categorical) - harmonisation et imputation
    valid_meal = ["BB", "HB", "FB", "SC", "Undefined"]
    df["meal"] = df["meal"].apply(lambda v: harmonize_category(v, valid_meal))
    # Imputation des valeurs manquantes (8.35%)
    mode_meal = df["meal"].mode()[0]
    df["meal"] = df["meal"].fillna(mode_meal)

    # country (categorical) - harmonisation (codes pays ISO 3 lettres)
    # Pas d'harmonisation car haute cardinalité, seulement nettoyage des espaces/casse
    df["country"] = df["country"].str.strip().str.upper()
    # Imputation des valeurs manquantes (9.16%)
    mode_country = df["country"].mode()[0]
    df["country"] = df["country"].fillna(mode_country)

    # market_segment (categorical) - harmonisation
    valid_market = [
        "Online TA", "Offline TA/TO", "Groups", "Direct",
        "Corporate", "Complementary", "Aviation", "Undefined"
    ]
    df["market_segment"] = df["market_segment"].apply(
        lambda v: harmonize_category(v, valid_market)
    )
    # Imputation des valeurs manquantes (8.32%)
    mode_market = df["market_segment"].mode()[0]
    df["market_segment"] = df["market_segment"].fillna(mode_market)

    # distribution_channel (categorical) - harmonisation
    valid_dist = ["TA/TO", "Direct", "Corporate", "GDS", "Undefined"]
    df["distribution_channel"] = df["distribution_channel"].apply(
        lambda v: harmonize_category(v, valid_dist)
    )

    # is_repeated_guest (numeric) - déjà propre
    # previous_cancellations (numeric) - valeurs aberrantes (max 26)
    q99_prev_canc = df["previous_cancellations"].quantile(0.995)
    outliers_prev_canc = df["previous_cancellations"] > q99_prev_canc
    if outliers_prev_canc.any():
        median_prev_canc = df.loc[~outliers_prev_canc, "previous_cancellations"].median()
        df.loc[outliers_prev_canc, "previous_cancellations"] = median_prev_canc

    # previous_bookings_not_canceled (numeric) - valeurs aberrantes (max 72)
    q99_prev_book = df["previous_bookings_not_canceled"].quantile(0.995)
    outliers_prev_book = df["previous_bookings_not_canceled"] > q99_prev_book
    if outliers_prev_book.any():
        median_prev_book = df.loc[~outliers_prev_book, "previous_bookings_not_canceled"].median()
        df.loc[outliers_prev_book, "previous_bookings_not_canceled"] = median_prev_book

    # reserved_room_type (categorical) - harmonisation
    valid_room = ["A", "B", "C", "D", "E", "F", "G", "H", "L", "P"]
    df["reserved_room_type"] = df["reserved_room_type"].apply(
        lambda v: harmonize_category(v, valid_room)
    )

    # assigned_room_type (categorical) - harmonisation
    valid_assigned = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "K", "L", "P"]
    df["assigned_room_type"] = df["assigned_room_type"].apply(
        lambda v: harmonize_category(v, valid_assigned)
    )

    # booking_changes (numeric) - valeurs aberrantes (max 21)
    q99_changes = df["booking_changes"].quantile(0.995)
    outliers_changes = df["booking_changes"] > q99_changes
    if outliers_changes.any():
        median_changes = df.loc[~outliers_changes, "booking_changes"].median()
        df.loc[outliers_changes, "booking_changes"] = median_changes

    # deposit_type (categorical) - harmonisation
    valid_deposit = ["No Deposit", "Non Refund", "Refundable"]
    df["deposit_type"] = df["deposit_type"].apply(
        lambda v: harmonize_category(v, valid_deposit)
    )

    # agent (numeric) - extraction des valeurs numériques et imputation
    df["agent"] = df["agent"].apply(extract_numeric)
    # Imputation des valeurs manquantes (34.29%)
    mode_agent = df["agent"].mode()[0]
    df["agent"] = df["agent"].fillna(mode_agent)

    # company (numeric) - taux de manquants très élevé (94.31%) -> imputation par mode
    mode_company = df["company"].mode()[0]
    df["company"] = df["company"].fillna(mode_company)

    # days_in_waiting_list (numeric) - valeurs aberrantes (max 8993)
    q99_waiting = df["days_in_waiting_list"].quantile(0.995)
    outliers_waiting = df["days_in_waiting_list"] > q99_waiting
    if outliers_waiting.any():
        median_waiting = df.loc[~outliers_waiting, "days_in_waiting_list"].median()
        df.loc[outliers_waiting, "days_in_waiting_list"] = median_waiting

    # customer_type (categorical) - harmonisation
    valid_customer = ["Transient", "Transient-Party", "Contract", "Group"]
    df["customer_type"] = df["customer_type"].apply(
        lambda v: harmonize_category(v, valid_customer)
    )

    # adr (numeric) - valeurs aberrantes (min -494, max 9997)
    # Correction des valeurs négatives
    df.loc[df["adr"] < 0, "adr"] = df["adr"].median()
    # Correction des valeurs trop élevées
    q99_adr = df["adr"].quantile(0.995)
    outliers_adr = df["adr"] > q99_adr
    if outliers_adr.any():
        median_adr = df.loc[~outliers_adr, "adr"].median()
        df.loc[outliers_adr, "adr"] = median_adr

    # required_car_parking_spaces (numeric) - valeurs aberrantes (max 8)
    # Pas de correction car valeurs plausibles (0-8)

    # total_of_special_requests (numeric) - valeurs aberrantes (max 5)
    # Pas de correction car valeurs plausibles (0-5)

    # reservation_status (categorical) - harmonisation
    valid_status = ["Check-Out", "Canceled", "No-Show"]
    df["reservation_status"] = df["reservation_status"].apply(
        lambda v: harmonize_category(v, valid_status)
    )

    # reservation_status_date (date) - parsing et reformatage
    df["reservation_status_date"] = df["reservation_status_date"].apply(parse_date)

    # 3. Vérification finale des types
    # Conversion des colonnes numériques en float/int si nécessaire
    numeric_cols = [
        "lead_time", "arrival_date_year", "arrival_date_week_number",
        "arrival_date_day_of_month", "stays_in_weekend_nights",
        "stays_in_week_nights", "adults", "children", "babies",
        "previous_cancellations", "previous_bookings_not_canceled",
        "booking_changes", "agent", "company", "days_in_waiting_list",
        "adr", "required_car_parking_spaces", "total_of_special_requests"
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            if df[col].isna().any():
                df[col] = df[col].fillna(df[col].median())

    # Log final
    final_rows, final_cols = df.shape
    print(f"Dataset final: {final_rows} lignes, {final_cols} colonnes")
    print(f"Lignes supprimées: {initial_rows - final_rows}")
    print("Nettoyage terminé avec succès.")

    # Sauvegarde
    df.to_csv(OUTPUT_PATH, index=False)

if __name__ == "__main__":
    clean_hotel_bookings()