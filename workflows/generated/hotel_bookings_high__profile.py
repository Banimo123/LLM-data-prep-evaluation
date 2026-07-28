import pandas as pd
import numpy as np
import re
from datetime import datetime
import difflib

INPUT_PATH = r"benchmark\datasets\hotel_bookings\noisy_high.csv"
OUTPUT_PATH = r"results\cleaned_datasets\hotel_bookings\noisy_high__profile.csv"

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
        "%d-%b-%Y", "%Y/%m/%d", "%d.%m.%Y", "%m.%d.%Y"
    ]
    for fmt in date_formats:
        try:
            return datetime.strptime(str(value), fmt).strftime("%Y-%m-%d")
        except (ValueError, TypeError):
            continue
    try:
        if str(value).isdigit() and len(str(value)) in (9, 10):
            return datetime.fromtimestamp(int(value)).strftime("%Y-%m-%d")
    except (ValueError, OSError):
        pass
    return str(value)

def clean_hotel_bookings():
    df = pd.read_csv(INPUT_PATH)

    operations_log = []

    # 1. Suppression des doublons (en conservant row_id)
    initial_rows = len(df)
    df = df.drop_duplicates(subset=df.columns.difference(['row_id']))
    duplicates_removed = initial_rows - len(df)
    if duplicates_removed > 0:
        operations_log.append(f"Doublons supprimés: {duplicates_removed}")

    # 2. Nettoyage colonne par colonne
    # hotel (categorical) - harmonisation des variantes
    if 'hotel' in df.columns:
        valid_hotels = ["City Hotel", "Resort Hotel"]
        df["hotel"] = df["hotel"].apply(lambda v: harmonize_category(v, valid_hotels))

    # is_canceled (numeric) - déjà propre (0/1)
    if 'is_canceled' in df.columns:
        pass

    # lead_time (categorical mais devrait être numeric) - extraction numérique
    if 'lead_time' in df.columns:
        df["lead_time"] = df["lead_time"].apply(extract_numeric)
        # Imputation des valeurs manquantes (0% dans le profil mais au cas où)
        if df["lead_time"].isna().any():
            median_lead = df["lead_time"].median()
            df["lead_time"] = df["lead_time"].fillna(median_lead)
            operations_log.append(f"lead_time: {df['lead_time'].isna().sum()} valeurs manquantes imputées par médiane")

    # arrival_date_year (numeric) - déjà propre
    if 'arrival_date_year' in df.columns:
        pass

    # arrival_date_month (categorical) - harmonisation
    if 'arrival_date_month' in df.columns:
        valid_months = [
            "January", "February", "March", "April", "May", "June",
            "July", "August", "September", "October", "November", "December"
        ]
        df["arrival_date_month"] = df["arrival_date_month"].apply(
            lambda v: harmonize_category(v, valid_months))

    # arrival_date_week_number (numeric) - déjà propre
    if 'arrival_date_week_number' in df.columns:
        pass

    # arrival_date_day_of_month (numeric) - déjà propre
    if 'arrival_date_day_of_month' in df.columns:
        pass

    # stays_in_weekend_nights (numeric) - valeurs aberrantes (max 19 dans profil)
    if 'stays_in_weekend_nights' in df.columns:
        # Bornes physiques: 0-7 (weekend = 2 jours max)
        median_weekend = df["stays_in_weekend_nights"].median()
        df.loc[df["stays_in_weekend_nights"] > 7, "stays_in_weekend_nights"] = median_weekend
        operations_log.append(f"stays_in_weekend_nights: {len(df[df['stays_in_weekend_nights'] > 7])} valeurs aberrantes corrigées")

    # stays_in_week_nights (numeric) - valeurs aberrantes (max 999 dans profil)
    if 'stays_in_week_nights' in df.columns:
        # 99.5e percentile pour éviter de supprimer des séjours longs valides
        p995 = np.percentile(df["stays_in_week_nights"].dropna(), 99.5)
        median_week = df["stays_in_week_nights"].median()
        df.loc[df["stays_in_week_nights"] > p995, "stays_in_week_nights"] = median_week
        operations_log.append(f"stays_in_week_nights: {len(df[df['stays_in_week_nights'] > p995])} valeurs aberrantes corrigées")

    # adults (categorical mais devrait être numeric) - extraction numérique
    if 'adults' in df.columns:
        df["adults"] = df["adults"].apply(extract_numeric)
        # Bornes physiques: 1-4 (max observé 26 dans profil mais probablement erreur)
        median_adults = df["adults"].median()
        df.loc[(df["adults"] < 1) | (df["adults"] > 4), "adults"] = median_adults
        operations_log.append(f"adults: {len(df[(df['adults'] < 1) | (df['adults'] > 4)])} valeurs aberrantes corrigées")

    # children (categorical mais devrait être numeric) - extraction numérique + imputation
    if 'children' in df.columns:
        df["children"] = df["children"].apply(extract_numeric)
        # Imputation des valeurs manquantes (33.29%)
        mode_children = df["children"].mode()[0]
        df["children"] = df["children"].fillna(mode_children)
        operations_log.append(f"children: {df['children'].isna().sum()} valeurs manquantes imputées par mode")
        # Bornes physiques: 0-10 (max observé 10 dans profil)
        df.loc[df["children"] > 10, "children"] = mode_children
        operations_log.append(f"children: {len(df[df['children'] > 10])} valeurs aberrantes corrigées")

    # babies (numeric) - valeurs aberrantes (max 49 dans profil)
    if 'babies' in df.columns:
        # 99.5e percentile
        p995 = np.percentile(df["babies"].dropna(), 99.5)
        median_babies = df["babies"].median()
        df.loc[df["babies"] > p995, "babies"] = median_babies
        operations_log.append(f"babies: {len(df[df['babies'] > p995])} valeurs aberrantes corrigées")

    # meal (categorical) - harmonisation + imputation
    if 'meal' in df.columns:
        valid_meals = ["BB", "HB", "FB", "SC", "Undefined"]
        df["meal"] = df["meal"].apply(lambda v: harmonize_category(v, valid_meals))
        # Imputation des valeurs manquantes (33.43%)
        mode_meal = df["meal"].mode()[0]
        df["meal"] = df["meal"].fillna(mode_meal)
        operations_log.append(f"meal: {df['meal'].isna().sum()} valeurs manquantes imputées par mode")

    # country (categorical) - harmonisation (codes ISO 3 lettres)
    if 'country' in df.columns:
        # Pas de liste exhaustive, mais nettoyage des espaces et casse
        df["country"] = df["country"].str.strip().str.upper()
        # Imputation des valeurs manquantes (33.94%)
        mode_country = df["country"].mode()[0]
        df["country"] = df["country"].fillna(mode_country)
        operations_log.append(f"country: {df['country'].isna().sum()} valeurs manquantes imputées par mode")

    # market_segment (categorical) - harmonisation + imputation
    if 'market_segment' in df.columns:
        valid_segments = [
            "Online TA", "Offline TA/TO", "Groups", "Direct",
            "Corporate", "Complementary", "Aviation", "Undefined"
        ]
        df["market_segment"] = df["market_segment"].apply(
            lambda v: harmonize_category(v, valid_segments))
        # Imputation des valeurs manquantes (33.35%)
        mode_segment = df["market_segment"].mode()[0]
        df["market_segment"] = df["market_segment"].fillna(mode_segment)
        operations_log.append(f"market_segment: {df['market_segment'].isna().sum()} valeurs manquantes imputées par mode")

    # distribution_channel (categorical) - harmonisation
    if 'distribution_channel' in df.columns:
        valid_channels = ["TA/TO", "Direct", "Corporate", "GDS", "Undefined"]
        df["distribution_channel"] = df["distribution_channel"].apply(
            lambda v: harmonize_category(v, valid_channels))

    # is_repeated_guest (numeric) - déjà propre
    if 'is_repeated_guest' in df.columns:
        pass

    # previous_cancellations (numeric) - valeurs aberrantes (max 26 dans profil)
    if 'previous_cancellations' in df.columns:
        # 99.5e percentile
        p995 = np.percentile(df["previous_cancellations"].dropna(), 99.5)
        median_prev_canc = df["previous_cancellations"].median()
        df.loc[df["previous_cancellations"] > p995, "previous_cancellations"] = median_prev_canc
        operations_log.append(f"previous_cancellations: {len(df[df['previous_cancellations'] > p995])} valeurs aberrantes corrigées")

    # previous_bookings_not_canceled (numeric) - valeurs aberrantes (max 72 dans profil)
    if 'previous_bookings_not_canceled' in df.columns:
        # 99.5e percentile
        p995 = np.percentile(df["previous_bookings_not_canceled"].dropna(), 99.5)
        median_prev_book = df["previous_bookings_not_canceled"].median()
        df.loc[df["previous_bookings_not_canceled"] > p995, "previous_bookings_not_canceled"] = median_prev_book
        operations_log.append(f"previous_bookings_not_canceled: {len(df[df['previous_bookings_not_canceled'] > p995])} valeurs aberrantes corrigées")

    # reserved_room_type (categorical) - harmonisation
    if 'reserved_room_type' in df.columns:
        valid_rooms = ["A", "B", "C", "D", "E", "F", "G", "H", "L", "P"]
        df["reserved_room_type"] = df["reserved_room_type"].apply(
            lambda v: harmonize_category(v, valid_rooms))

    # assigned_room_type (categorical) - harmonisation
    if 'assigned_room_type' in df.columns:
        valid_rooms = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "K", "L", "P"]
        df["assigned_room_type"] = df["assigned_room_type"].apply(
            lambda v: harmonize_category(v, valid_rooms))

    # booking_changes (numeric) - valeurs aberrantes (max 21 dans profil)
    if 'booking_changes' in df.columns:
        # 99.5e percentile
        p995 = np.percentile(df["booking_changes"].dropna(), 99.5)
        median_changes = df["booking_changes"].median()
        df.loc[df["booking_changes"] > p995, "booking_changes"] = median_changes
        operations_log.append(f"booking_changes: {len(df[df['booking_changes'] > p995])} valeurs aberrantes corrigées")

    # deposit_type (categorical) - harmonisation + imputation
    if 'deposit_type' in df.columns:
        valid_deposits = ["No Deposit", "Non Refund"]
        df["deposit_type"] = df["deposit_type"].apply(
            lambda v: harmonize_category(v, valid_deposits))

    # agent (categorical mais devrait être numeric) - extraction numérique + imputation
    if 'agent' in df.columns:
        df["agent"] = df["agent"].apply(extract_numeric)
        # Imputation des valeurs manquantes (55.24%)
        mode_agent = df["agent"].mode()[0]
        df["agent"] = df["agent"].fillna(mode_agent)
        operations_log.append(f"agent: {df['agent'].isna().sum()} valeurs manquantes imputées par mode")

    # company (numeric) - trop de manquants (94.31%) -> suppression de la colonne
    if 'company' in df.columns:
        df = df.drop(columns=['company'])
        operations_log.append("company: colonne supprimée (94.31% de valeurs manquantes)")

    # days_in_waiting_list (numeric) - valeurs aberrantes (max 8999 dans profil)
    if 'days_in_waiting_list' in df.columns:
        # 99.5e percentile
        p995 = np.percentile(df["days_in_waiting_list"].dropna(), 99.5)
        median_wait = df["days_in_waiting_list"].median()
        df.loc[df["days_in_waiting_list"] > p995, "days_in_waiting_list"] = median_wait
        operations_log.append(f"days_in_waiting_list: {len(df[df['days_in_waiting_list'] > p995])} valeurs aberrantes corrigées")

    # customer_type (categorical) - harmonisation
    if 'customer_type' in df.columns:
        valid_customers = ["Transient", "Transient-Party", "Contract", "Group"]
        df["customer_type"] = df["customer_type"].apply(
            lambda v: harmonize_category(v, valid_customers))

    # adr (numeric) - valeurs aberrantes (min -496.7, max 9997.01 dans profil)
    if 'adr' in df.columns:
        # Bornes physiques: prix positif, 99.5e percentile
        p995 = np.percentile(df[df["adr"] > 0]["adr"].dropna(), 99.5)
        median_adr = df[df["adr"] > 0]["adr"].median()
        df.loc[(df["adr"] <= 0) | (df["adr"] > p995), "adr"] = median_adr
        operations_log.append(f"adr: {len(df[(df['adr'] <= 0) | (df['adr'] > p995)])} valeurs aberrantes corrigées")

    # required_car_parking_spaces (numeric) - valeurs aberrantes (max 8 dans profil)
    if 'required_car_parking_spaces' in df.columns:
        # Bornes physiques: 0-5 (max observé 8 mais probablement erreur)
        mode_parking = df["required_car_parking_spaces"].mode()[0]
        df.loc[df["required_car_parking_spaces"] > 5, "required_car_parking_spaces"] = mode_parking
        operations_log.append(f"required_car_parking_spaces: {len(df[df['required_car_parking_spaces'] > 5])} valeurs aberrantes corrigées")

    # total_of_special_requests (numeric) - valeurs aberrantes (max 5 dans profil)
    if 'total_of_special_requests' in df.columns:
        # Bornes physiques: 0-5
        mode_requests = df["total_of_special_requests"].mode()[0]
        df.loc[df["total_of_special_requests"] > 5, "total_of_special_requests"] = mode_requests
        operations_log.append(f"total_of_special_requests: {len(df[df['total_of_special_requests'] > 5])} valeurs aberrantes corrigées")

    # reservation_status (categorical) - harmonisation
    if 'reservation_status' in df.columns:
        valid_status = ["Check-Out", "Canceled", "No-Show"]
        df["reservation_status"] = df["reservation_status"].apply(
            lambda v: harmonize_category(v, valid_status))

    # reservation_status_date (date) - parsing + reformatage
    if 'reservation_status_date' in df.columns:
        df["reservation_status_date"] = df["reservation_status_date"].apply(parse_date)

    # 3. Conversion des types
    numeric_cols = [
        'is_canceled', 'lead_time', 'arrival_date_year', 'arrival_date_week_number',
        'arrival_date_day_of_month', 'stays_in_weekend_nights', 'stays_in_week_nights',
        'adults', 'children', 'babies', 'is_repeated_guest', 'previous_cancellations',
        'previous_bookings_not_canceled', 'booking_changes', 'days_in_waiting_list',
        'adr', 'required_car_parking_spaces', 'total_of_special_requests'
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            if df[col].isna().any():
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
                operations_log.append(f"{col}: {df[col].isna().sum()} valeurs manquantes imputées par médiane")

    # 4. Sauvegarde
    df.to_csv(OUTPUT_PATH, index=False)

    # 5. Log des opérations
    print(f"Nettoyage terminé. Lignes initiales: {initial_rows}, lignes finales: {len(df)}")
    for op in operations_log:
        print(f"- {op}")

if __name__ == "__main__":
    clean_hotel_bookings()