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
    initial_rows = len(df)
    initial_cols = len(df.columns)
    print(f"Dataset initial: {initial_rows} lignes, {initial_cols} colonnes")

    # 1. Suppression des doublons (conservation du premier)
    df.drop_duplicates(subset=df.columns.difference(['row_id']), keep='first', inplace=True)
    duplicates_removed = initial_rows - len(df)
    print(f"Doublons supprimés: {duplicates_removed}")

    # 2. Nettoyage colonne par colonne
    # hotel (categorical) - harmonisation des variantes
    valid_hotel = ["City Hotel", "Resort Hotel"]
    df["hotel"] = df["hotel"].apply(lambda v: harmonize_category(v, valid_hotel))

    # is_canceled (numeric) - déjà propre (0/1)
    # lead_time (text -> numeric) - extraction des nombres
    df["lead_time"] = df["lead_time"].apply(extract_numeric)
    # Imputation des NaN (0% manquants mais extraction peut en créer)
    if df["lead_time"].isna().any():
        df["lead_time"].fillna(df["lead_time"].median(), inplace=True)

    # arrival_date_year (numeric) - déjà propre
    # arrival_date_month (categorical) - harmonisation
    valid_months = [
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December"
    ]
    df["arrival_date_month"] = df["arrival_date_month"].apply(
        lambda v: harmonize_category(v, valid_months)
    )

    # arrival_date_week_number, arrival_date_day_of_month (numeric) - déjà propres
    # stays_in_weekend_nights (numeric) - extraction si nécessaire
    df["stays_in_weekend_nights"] = df["stays_in_weekend_nights"].apply(extract_numeric)
    # stays_in_week_nights (numeric) - extraction + traitement outliers
    df["stays_in_week_nights"] = df["stays_in_week_nights"].apply(extract_numeric)
    # Outliers: max=999 -> 99.5e percentile (14)
    upper_bound = df["stays_in_week_nights"].quantile(0.995)
    outliers = df["stays_in_week_nights"] > upper_bound
    if outliers.any():
        median_val = df.loc[~outliers, "stays_in_week_nights"].median()
        df.loc[outliers, "stays_in_week_nights"] = median_val

    # adults (text -> numeric) - extraction + traitement des valeurs aberrantes
    df["adults"] = df["adults"].apply(extract_numeric)
    # Outliers: max=45 -> bornes plausibles [0, 10]
    df["adults"] = df["adults"].clip(0, 10)
    # Imputation des NaN (0% manquants mais extraction peut en créer)
    if df["adults"].isna().any():
        df["adults"].fillna(df["adults"].median(), inplace=True)

    # children (text -> numeric) - extraction + imputation des manquants
    df["children"] = df["children"].replace("unknown", np.nan)
    df["children"] = df["children"].apply(extract_numeric)
    # Imputation des NaN (8.32% manquants)
    if df["children"].isna().any():
        df["children"].fillna(df["children"].median(), inplace=True)
    # Outliers: max=10 -> bornes plausibles [0, 5]
    df["children"] = df["children"].clip(0, 5)

    # babies (numeric) - traitement outliers (max=49 -> 99.5e percentile=2)
    upper_bound = df["babies"].quantile(0.995)
    outliers = df["babies"] > upper_bound
    if outliers.any():
        median_val = df.loc[~outliers, "babies"].median()
        df.loc[outliers, "babies"] = median_val

    # meal (categorical) - harmonisation + imputation des manquants
    valid_meal = ["BB", "HB", "FB", "SC", "Undefined"]
    df["meal"] = df["meal"].apply(lambda v: harmonize_category(v, valid_meal))
    # Imputation des NaN (8.35% manquants)
    if df["meal"].isna().any():
        mode_val = df["meal"].mode()[0]
        df["meal"].fillna(mode_val, inplace=True)

    # country (categorical) - harmonisation des codes pays (3 lettres majuscules)
    df["country"] = df["country"].str.strip().str.upper()
    # Imputation des NaN (9.16% manquants)
    if df["country"].isna().any():
        mode_val = df["country"].mode()[0]
        df["country"].fillna(mode_val, inplace=True)

    # market_segment (categorical) - harmonisation + imputation des manquants
    valid_segment = [
        "Online TA", "Offline TA/TO", "Groups", "Direct",
        "Corporate", "Complementary", "Aviation", "Undefined"
    ]
    df["market_segment"] = df["market_segment"].apply(
        lambda v: harmonize_category(v, valid_segment)
    )
    # Imputation des NaN (8.32% manquants)
    if df["market_segment"].isna().any():
        mode_val = df["market_segment"].mode()[0]
        df["market_segment"].fillna(mode_val, inplace=True)

    # distribution_channel (categorical) - harmonisation
    valid_channel = ["TA/TO", "Direct", "Corporate", "GDS", "Undefined"]
    df["distribution_channel"] = df["distribution_channel"].apply(
        lambda v: harmonize_category(v, valid_channel)
    )

    # is_repeated_guest (numeric) - déjà propre (0/1)
    # previous_cancellations (numeric) - traitement outliers (max=26 -> 99.5e percentile=3)
    upper_bound = df["previous_cancellations"].quantile(0.995)
    outliers = df["previous_cancellations"] > upper_bound
    if outliers.any():
        median_val = df.loc[~outliers, "previous_cancellations"].median()
        df.loc[outliers, "previous_cancellations"] = median_val

    # previous_bookings_not_canceled (numeric) - traitement outliers (max=72 -> 99.5e percentile=5)
    upper_bound = df["previous_bookings_not_canceled"].quantile(0.995)
    outliers = df["previous_bookings_not_canceled"] > upper_bound
    if outliers.any():
        median_val = df.loc[~outliers, "previous_bookings_not_canceled"].median()
        df.loc[outliers, "previous_bookings_not_canceled"] = median_val

    # reserved_room_type, assigned_room_type (categorical) - harmonisation
    valid_rooms = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "K", "L", "P"]
    for col in ["reserved_room_type", "assigned_room_type"]:
        df[col] = df[col].apply(lambda v: harmonize_category(v, valid_rooms))

    # booking_changes (numeric) - traitement outliers (max=21 -> 99.5e percentile=4)
    upper_bound = df["booking_changes"].quantile(0.995)
    outliers = df["booking_changes"] > upper_bound
    if outliers.any():
        median_val = df.loc[~outliers, "booking_changes"].median()
        df.loc[outliers, "booking_changes"] = median_val

    # deposit_type (categorical) - harmonisation
    valid_deposit = ["No Deposit", "Non Refund", "Refundable"]
    df["deposit_type"] = df["deposit_type"].apply(
        lambda v: harmonize_category(v, valid_deposit)
    )

    # agent (text -> numeric) - extraction + imputation des manquants
    df["agent"] = df["agent"].apply(extract_numeric)
    # Imputation des NaN (34.29% manquants)
    if df["agent"].isna().any():
        df["agent"].fillna(0, inplace=True)  # 0 = pas d'agent

    # company (numeric) - imputation des manquants (94.31% manquants -> suppression de la colonne)
    if "company" in df.columns:
        df.drop("company", axis=1, inplace=True)

    # days_in_waiting_list (numeric) - traitement outliers (max=8993 -> 99.5e percentile=120)
    upper_bound = df["days_in_waiting_list"].quantile(0.995)
    outliers = df["days_in_waiting_list"] > upper_bound
    if outliers.any():
        median_val = df.loc[~outliers, "days_in_waiting_list"].median()
        df.loc[outliers, "days_in_waiting_list"] = median_val

    # customer_type (categorical) - harmonisation
    valid_customer = ["Transient", "Transient-Party", "Contract", "Group"]
    df["customer_type"] = df["customer_type"].apply(
        lambda v: harmonize_category(v, valid_customer)
    )

    # adr (numeric) - extraction + traitement outliers
    df["adr"] = df["adr"].apply(extract_numeric)
    # Outliers: min=-494, max=9997 -> bornes plausibles [0, 5000]
    df["adr"] = df["adr"].clip(0, 5000)
    # Imputation des NaN (0% manquants mais extraction peut en créer)
    if df["adr"].isna().any():
        df["adr"].fillna(df["adr"].median(), inplace=True)

    # required_car_parking_spaces (numeric) - traitement outliers (max=8 -> 99.5e percentile=1)
    upper_bound = df["required_car_parking_spaces"].quantile(0.995)
    outliers = df["required_car_parking_spaces"] > upper_bound
    if outliers.any():
        mode_val = df.loc[~outliers, "required_car_parking_spaces"].mode()[0]
        df.loc[outliers, "required_car_parking_spaces"] = mode_val

    # total_of_special_requests (numeric) - traitement outliers (max=5 -> déjà propre)
    # reservation_status (categorical) - harmonisation
    valid_status = ["Check-Out", "Canceled", "No-Show"]
    df["reservation_status"] = df["reservation_status"].apply(
        lambda v: harmonize_category(v, valid_status)
    )

    # reservation_status_date (text -> date) - parsing
    df["reservation_status_date"] = df["reservation_status_date"].apply(parse_date)

    # 3. Conversion des types
    numeric_cols = [
        "lead_time", "arrival_date_year", "arrival_date_week_number",
        "arrival_date_day_of_month", "stays_in_weekend_nights",
        "stays_in_week_nights", "adults", "children", "babies",
        "previous_cancellations", "previous_bookings_not_canceled",
        "booking_changes", "agent", "days_in_waiting_list", "adr",
        "required_car_parking_spaces", "total_of_special_requests"
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            if df[col].isna().any():
                df[col].fillna(df[col].median(), inplace=True)

    # 4. Sauvegarde
    df.to_csv(OUTPUT_PATH, index=False)

    # Log final
    final_rows = len(df)
    final_cols = len(df.columns)
    print(f"Dataset nettoyé: {final_rows} lignes, {final_cols} colonnes")
    print(f"Lignes supprimées: {initial_rows - final_rows}")
    print(f"Colonnes supprimées: {initial_cols - final_cols}")

if __name__ == "__main__":
    clean_hotel_bookings()