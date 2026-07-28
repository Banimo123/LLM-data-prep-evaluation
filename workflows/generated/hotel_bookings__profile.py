import pandas as pd
import numpy as np
import re
from datetime import datetime
import difflib

INPUT_PATH = "benchmark/datasets/hotel_bookings/noisy_low.csv"
OUTPUT_PATH = "results/cleaned_datasets/hotel_bookings/noisy_low__profile.csv"

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
    s = str(value).strip()
    for fmt in date_formats:
        try:
            dt = datetime.strptime(s, fmt)
            return dt.strftime("%Y-%m-%d")
        except ValueError:
            continue
    if s.isdigit() and len(s) in (9, 10):
        try:
            dt = datetime.fromtimestamp(int(s))
            return dt.strftime("%Y-%m-%d")
        except (ValueError, OSError):
            pass
    return value

def clean_hotel_bookings():
    df = pd.read_csv(INPUT_PATH)

    operations_log = []

    # Suppression des doublons (conservation du premier)
    initial_rows = len(df)
    df = df.drop_duplicates(subset=[col for col in df.columns if col != 'row_id'], keep='first')
    duplicates_removed = initial_rows - len(df)
    if duplicates_removed > 0:
        operations_log.append(f"Doublons supprimés: {duplicates_removed}")

    # Nettoyage colonne par colonne
    # hotel (categorical) - harmonisation des variantes
    if 'hotel' in df.columns:
        valid_hotels = ["City Hotel", "Resort Hotel"]
        df["hotel"] = df["hotel"].apply(lambda v: harmonize_category(v, valid_hotels))

    # is_canceled (numeric) - déjà propre (0/1)
    if 'is_canceled' in df.columns:
        pass

    # lead_time (text -> numeric) - extraction des nombres
    if 'lead_time' in df.columns:
        df["lead_time"] = df["lead_time"].apply(extract_numeric)
        # Imputation des valeurs manquantes (0.0% dans profil mais au cas où)
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
            lambda v: harmonize_category(v, valid_months)
        )

    # arrival_date_week_number (numeric) - déjà propre
    if 'arrival_date_week_number' in df.columns:
        pass

    # arrival_date_day_of_month (numeric) - déjà propre
    if 'arrival_date_day_of_month' in df.columns:
        pass

    # stays_in_weekend_nights (numeric) - valeurs aberrantes (max 19 dans profil)
    if 'stays_in_weekend_nights' in df.columns:
        # Pas de valeurs aberrantes évidentes (max 19 est plausible)
        pass

    # stays_in_week_nights (numeric) - valeurs aberrantes (max 999 dans profil)
    if 'stays_in_week_nights' in df.columns:
        # 999 est clairement aberrant (moyenne 8.45)
        median_stays = df["stays_in_week_nights"].median()
        df.loc[df["stays_in_week_nights"] > 30, "stays_in_week_nights"] = median_stays
        operations_log.append(f"stays_in_week_nights: {len(df[df['stays_in_week_nights'] > 30])} valeurs aberrantes corrigées")

    # adults (text -> numeric) - extraction des nombres
    if 'adults' in df.columns:
        df["adults"] = df["adults"].apply(extract_numeric)
        # Imputation des valeurs manquantes (0.0% dans profil mais au cas où)
        if df["adults"].isna().any():
            mode_adults = df["adults"].mode()[0]
            df["adults"] = df["adults"].fillna(mode_adults)
            operations_log.append(f"adults: {df['adults'].isna().sum()} valeurs manquantes imputées par mode")
        # Correction des valeurs aberrantes (max 3 dans profil)
        df.loc[df["adults"] > 10, "adults"] = df["adults"].mode()[0]

    # children (text -> numeric) - extraction des nombres + imputation
    if 'children' in df.columns:
        df["children"] = df["children"].apply(extract_numeric)
        # Imputation des valeurs manquantes (8.32%)
        mode_children = df["children"].mode()[0]
        df["children"] = df["children"].fillna(mode_children)
        operations_log.append(f"children: {df['children'].isna().sum()} valeurs manquantes imputées par mode")
        # Correction des valeurs aberrantes (max 10 dans profil)
        df.loc[df["children"] > 5, "children"] = mode_children

    # babies (numeric) - valeurs aberrantes (max 49 dans profil)
    if 'babies' in df.columns:
        # 49 est aberrant (moyenne 0.3)
        median_babies = df["babies"].median()
        df.loc[df["babies"] > 5, "babies"] = median_babies
        operations_log.append(f"babies: {len(df[df['babies'] > 5])} valeurs aberrantes corrigées")

    # meal (categorical) - harmonisation + imputation
    if 'meal' in df.columns:
        valid_meals = ["BB", "HB", "FB", "SC", "Undefined"]
        df["meal"] = df["meal"].apply(lambda v: harmonize_category(v, valid_meals))
        # Imputation des valeurs manquantes (8.35%)
        mode_meal = df["meal"].mode()[0]
        df["meal"] = df["meal"].fillna(mode_meal)
        operations_log.append(f"meal: {df['meal'].isna().sum()} valeurs manquantes imputées par mode")

    # country (categorical) - harmonisation + imputation
    if 'country' in df.columns:
        # Liste des pays fréquents (3 lettres majuscules)
        valid_countries = [
            "PRT", "GBR", "FRA", "ESP", "DEU", "ITA", "IRL", "BEL", "BRA", "NLD"
        ]
        df["country"] = df["country"].apply(
            lambda v: v.strip().upper() if pd.notna(v) and len(str(v).strip()) == 3 else v
        )
        # Imputation des valeurs manquantes (9.16%)
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
            lambda v: harmonize_category(v, valid_segments)
        )
        # Imputation des valeurs manquantes (8.32%)
        mode_segment = df["market_segment"].mode()[0]
        df["market_segment"] = df["market_segment"].fillna(mode_segment)
        operations_log.append(f"market_segment: {df['market_segment'].isna().sum()} valeurs manquantes imputées par mode")

    # distribution_channel (categorical) - harmonisation
    if 'distribution_channel' in df.columns:
        valid_channels = ["TA/TO", "Direct", "Corporate", "GDS", "Undefined"]
        df["distribution_channel"] = df["distribution_channel"].apply(
            lambda v: harmonize_category(v, valid_channels)
        )

    # is_repeated_guest (numeric) - déjà propre
    if 'is_repeated_guest' in df.columns:
        pass

    # previous_cancellations (numeric) - valeurs aberrantes (max 26 dans profil)
    if 'previous_cancellations' in df.columns:
        # 26 est aberrant (moyenne 0.09)
        median_prev_canc = df["previous_cancellations"].median()
        df.loc[df["previous_cancellations"] > 5, "previous_cancellations"] = median_prev_canc
        operations_log.append(f"previous_cancellations: {len(df[df['previous_cancellations'] > 5])} valeurs aberrantes corrigées")

    # previous_bookings_not_canceled (numeric) - valeurs aberrantes (max 72 dans profil)
    if 'previous_bookings_not_canceled' in df.columns:
        # 72 est aberrant (moyenne 0.14)
        median_prev_book = df["previous_bookings_not_canceled"].median()
        df.loc[df["previous_bookings_not_canceled"] > 10, "previous_bookings_not_canceled"] = median_prev_book
        operations_log.append(f"previous_bookings_not_canceled: {len(df[df['previous_bookings_not_canceled'] > 10])} valeurs aberrantes corrigées")

    # reserved_room_type (categorical) - harmonisation
    if 'reserved_room_type' in df.columns:
        valid_rooms = ["A", "D", "E", "F", "G", "B", "C", "H", "P", "L"]
        df["reserved_room_type"] = df["reserved_room_type"].apply(
            lambda v: harmonize_category(v, valid_rooms)
        )

    # assigned_room_type (categorical) - harmonisation
    if 'assigned_room_type' in df.columns:
        valid_rooms = ["A", "D", "E", "F", "G", "C", "B", "H", "I", "K"]
        df["assigned_room_type"] = df["assigned_room_type"].apply(
            lambda v: harmonize_category(v, valid_rooms)
        )

    # booking_changes (numeric) - valeurs aberrantes (max 21 dans profil)
    if 'booking_changes' in df.columns:
        # 21 est aberrant (moyenne 0.22)
        median_changes = df["booking_changes"].median()
        df.loc[df["booking_changes"] > 5, "booking_changes"] = median_changes
        operations_log.append(f"booking_changes: {len(df[df['booking_changes'] > 5])} valeurs aberrantes corrigées")

    # deposit_type (categorical) - harmonisation + imputation
    if 'deposit_type' in df.columns:
        valid_deposits = ["No Deposit", "Non Refund", "Refundable"]
        df["deposit_type"] = df["deposit_type"].apply(
            lambda v: harmonize_category(v, valid_deposits)
        )

    # agent (text -> numeric) - extraction des nombres + imputation
    if 'agent' in df.columns:
        df["agent"] = df["agent"].apply(extract_numeric)
        # Imputation des valeurs manquantes (34.29%)
        mode_agent = df["agent"].mode()[0]
        df["agent"] = df["agent"].fillna(mode_agent)
        operations_log.append(f"agent: {df['agent'].isna().sum()} valeurs manquantes imputées par mode")

    # company (numeric) - trop de manquants (94.31%) -> imputation par mode
    if 'company' in df.columns:
        mode_company = df["company"].mode()[0]
        df["company"] = df["company"].fillna(mode_company)
        operations_log.append(f"company: {df['company'].isna().sum()} valeurs manquantes imputées par mode")

    # days_in_waiting_list (numeric) - valeurs aberrantes (max 8993 dans profil)
    if 'days_in_waiting_list' in df.columns:
        # 8993 est aberrant (moyenne 57.77)
        median_wait = df["days_in_waiting_list"].median()
        df.loc[df["days_in_waiting_list"] > 365, "days_in_waiting_list"] = median_wait
        operations_log.append(f"days_in_waiting_list: {len(df[df['days_in_waiting_list'] > 365])} valeurs aberrantes corrigées")

    # customer_type (categorical) - harmonisation
    if 'customer_type' in df.columns:
        valid_customers = ["Transient", "Transient-Party", "Contract", "Group"]
        df["customer_type"] = df["customer_type"].apply(
            lambda v: harmonize_category(v, valid_customers)
        )

    # adr (numeric) - valeurs aberrantes (min -494.03, max 9997.54)
    if 'adr' in df.columns:
        # Correction des valeurs négatives
        median_adr = df["adr"].median()
        df.loc[df["adr"] < 0, "adr"] = median_adr
        # Correction des valeurs trop élevées (99e percentile)
        adr_99 = df["adr"].quantile(0.99)
        df.loc[df["adr"] > adr_99, "adr"] = median_adr
        operations_log.append(f"adr: {len(df[df['adr'] < 0]) + len(df[df['adr'] > adr_99])} valeurs aberrantes corrigées")

    # required_car_parking_spaces (numeric) - valeurs aberrantes (max 8 dans profil)
    if 'required_car_parking_spaces' in df.columns:
        # 8 est plausible (moyenne 0.06)
        pass

    # total_of_special_requests (numeric) - valeurs aberrantes (max 5 dans profil)
    if 'total_of_special_requests' in df.columns:
        # 5 est plausible (moyenne 0.57)
        pass

    # reservation_status (categorical) - harmonisation
    if 'reservation_status' in df.columns:
        valid_status = ["Check-Out", "Canceled", "No-Show"]
        df["reservation_status"] = df["reservation_status"].apply(
            lambda v: harmonize_category(v, valid_status)
        )

    # reservation_status_date (text -> date) - parsing
    if 'reservation_status_date' in df.columns:
        df["reservation_status_date"] = df["reservation_status_date"].apply(parse_date)

    # Conversion des colonnes numériques en types appropriés
    numeric_cols = [
        'is_canceled', 'lead_time', 'arrival_date_year', 'arrival_date_week_number',
        'arrival_date_day_of_month', 'stays_in_weekend_nights', 'stays_in_week_nights',
        'adults', 'children', 'babies', 'is_repeated_guest', 'previous_cancellations',
        'previous_bookings_not_canceled', 'booking_changes', 'agent', 'company',
        'days_in_waiting_list', 'adr', 'required_car_parking_spaces',
        'total_of_special_requests'
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            if df[col].isna().any():
                if col in ['adults', 'children', 'babies', 'previous_cancellations',
                          'previous_bookings_not_canceled', 'booking_changes']:
                    df[col] = df[col].fillna(df[col].mode()[0])
                else:
                    df[col] = df[col].fillna(df[col].median())

    # Sauvegarde du dataset nettoyé
    df.to_csv(OUTPUT_PATH, index=False)

    # Log des opérations
    print(f"Nettoyage terminé. Lignes initiales: {initial_rows}, lignes finales: {len(df)}")
    for op in operations_log:
        print(op)
    print(f"Dataset nettoyé sauvegardé dans {OUTPUT_PATH}")

if __name__ == "__main__":
    clean_hotel_bookings()