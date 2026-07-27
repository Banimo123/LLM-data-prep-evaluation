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
        "%d-%b-%Y", "%Y/%m/%d", "%d/%m/%y", "%m/%d/%y"
    ]
    for fmt in date_formats:
        try:
            return datetime.strptime(str(value), fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    if str(value).isdigit() and len(str(value)) in (9, 10):
        try:
            return datetime.fromtimestamp(int(value)).strftime("%Y-%m-%d")
        except (ValueError, OSError):
            pass
    return str(value)

def main():
    df = pd.read_csv(INPUT_PATH)
    operations = []

    # Conservation de row_id intact
    if 'row_id' not in df.columns:
        raise ValueError("La colonne row_id est absente du dataset")

    # 1. Suppression des doublons (en conservant le premier)
    initial_rows = len(df)
    df.drop_duplicates(subset=df.columns.difference(['row_id']), keep='first', inplace=True)
    duplicates_removed = initial_rows - len(df)
    if duplicates_removed > 0:
        operations.append(f"Doublons supprimés: {duplicates_removed}")

    # 2. Nettoyage colonne par colonne
    # hotel (catégoriel) - harmonisation des variantes
    if 'hotel' in df.columns:
        valid_hotels = ["City Hotel", "Resort Hotel"]
        df["hotel"] = df["hotel"].apply(lambda v: harmonize_category(v, valid_hotels))

    # is_canceled (numérique) - pas de traitement nécessaire (0/1 complets)

    # lead_time (texte mais devrait être numérique) - extraction numérique
    if 'lead_time' in df.columns:
        df["lead_time"] = df["lead_time"].apply(extract_numeric)
        # Imputation des valeurs manquantes (0% dans le profil mais au cas où)
        if df["lead_time"].isna().any():
            median_lead = df["lead_time"].median()
            df["lead_time"].fillna(median_lead, inplace=True)
            operations.append(f"lead_time: {df['lead_time'].isna().sum()} valeurs manquantes imputées par médiane")

    # arrival_date_year (numérique) - pas de traitement nécessaire

    # arrival_date_month (catégoriel) - harmonisation des mois
    if 'arrival_date_month' in df.columns:
        valid_months = [
            "January", "February", "March", "April", "May", "June",
            "July", "August", "September", "October", "November", "December"
        ]
        df["arrival_date_month"] = df["arrival_date_month"].apply(
            lambda v: harmonize_category(v, valid_months)
        )

    # arrival_date_week_number (numérique) - pas de traitement nécessaire

    # arrival_date_day_of_month (numérique) - pas de traitement nécessaire

    # stays_in_weekend_nights (numérique) - pas de traitement nécessaire

    # stays_in_week_nights (numérique) - valeurs aberrantes (max=999)
    if 'stays_in_week_nights' in df.columns:
        # Calcul des bornes basées sur le 99.5e percentile
        upper_bound = df["stays_in_week_nights"].quantile(0.995)
        outliers = df["stays_in_week_nights"] > upper_bound
        if outliers.any():
            median_val = df.loc[~outliers, "stays_in_week_nights"].median()
            df.loc[outliers, "stays_in_week_nights"] = median_val
            operations.append(f"stays_in_week_nights: {outliers.sum()} valeurs aberrantes corrigées")

    # adults (texte mais devrait être numérique) - extraction numérique
    if 'adults' in df.columns:
        df["adults"] = df["adults"].apply(extract_numeric)
        # Imputation des valeurs manquantes (0% dans le profil)
        if df["adults"].isna().any():
            mode_adults = df["adults"].mode()[0]
            df["adults"].fillna(mode_adults, inplace=True)
            operations.append(f"adults: {df['adults'].isna().sum()} valeurs manquantes imputées par mode")

    # children (texte avec valeurs manquantes) - extraction numérique puis imputation
    if 'children' in df.columns:
        # Remplacement des valeurs "unknown" par NaN avant extraction
        df["children"] = df["children"].replace("unknown", np.nan)
        df["children"] = df["children"].apply(extract_numeric)
        # Imputation par mode (0.0 est la valeur la plus fréquente)
        mode_children = df["children"].mode()[0]
        df["children"].fillna(mode_children, inplace=True)
        operations.append(f"children: {df['children'].isna().sum()} valeurs manquantes imputées par mode")

    # babies (numérique) - valeurs aberrantes (max=49)
    if 'babies' in df.columns:
        # Bornes basées sur le 99.5e percentile
        upper_bound = df["babies"].quantile(0.995)
        outliers = df["babies"] > upper_bound
        if outliers.any():
            mode_babies = df.loc[~outliers, "babies"].mode()[0]
            df.loc[outliers, "babies"] = mode_babies
            operations.append(f"babies: {outliers.sum()} valeurs aberrantes corrigées")

    # meal (catégoriel) - harmonisation et imputation
    if 'meal' in df.columns:
        valid_meals = ["BB", "HB", "FB", "SC", "Undefined"]
        df["meal"] = df["meal"].apply(lambda v: harmonize_category(v, valid_meals))
        # Imputation des valeurs manquantes par mode
        mode_meal = df["meal"].mode()[0]
        df["meal"].fillna(mode_meal, inplace=True)
        operations.append(f"meal: {df['meal'].isna().sum()} valeurs manquantes imputées par mode")

    # country (catégoriel) - harmonisation des codes pays (3 lettres majuscules)
    if 'country' in df.columns:
        # Remplacement des valeurs "unknown" par NaN
        df["country"] = df["country"].replace("unknown", np.nan)
        # Nettoyage des espaces et mise en majuscules
        df["country"] = df["country"].str.strip().str.upper()
        # Imputation par mode (PRT est le plus fréquent)
        mode_country = df["country"].mode()[0]
        df["country"].fillna(mode_country, inplace=True)
        operations.append(f"country: {df['country'].isna().sum()} valeurs manquantes imputées par mode")

    # market_segment (catégoriel) - harmonisation
    if 'market_segment' in df.columns:
        valid_segments = [
            "Online TA", "Offline TA/TO", "Groups", "Direct",
            "Corporate", "Complementary", "Aviation", "Undefined"
        ]
        df["market_segment"] = df["market_segment"].apply(
            lambda v: harmonize_category(v, valid_segments)
        )
        # Imputation par mode
        mode_segment = df["market_segment"].mode()[0]
        df["market_segment"].fillna(mode_segment, inplace=True)
        operations.append(f"market_segment: {df['market_segment'].isna().sum()} valeurs manquantes imputées par mode")

    # distribution_channel (catégoriel) - harmonisation
    if 'distribution_channel' in df.columns:
        valid_channels = ["TA/TO", "Direct", "Corporate", "GDS", "Undefined"]
        df["distribution_channel"] = df["distribution_channel"].apply(
            lambda v: harmonize_category(v, valid_channels)
        )

    # is_repeated_guest (numérique) - pas de traitement nécessaire

    # previous_cancellations (numérique) - valeurs aberrantes (max=26)
    if 'previous_cancellations' in df.columns:
        # Bornes basées sur le 99.5e percentile
        upper_bound = df["previous_cancellations"].quantile(0.995)
        outliers = df["previous_cancellations"] > upper_bound
        if outliers.any():
            mode_prev_canc = df.loc[~outliers, "previous_cancellations"].mode()[0]
            df.loc[outliers, "previous_cancellations"] = mode_prev_canc
            operations.append(f"previous_cancellations: {outliers.sum()} valeurs aberrantes corrigées")

    # previous_bookings_not_canceled (numérique) - valeurs aberrantes (max=72)
    if 'previous_bookings_not_canceled' in df.columns:
        upper_bound = df["previous_bookings_not_canceled"].quantile(0.995)
        outliers = df["previous_bookings_not_canceled"] > upper_bound
        if outliers.any():
            mode_prev_book = df.loc[~outliers, "previous_bookings_not_canceled"].mode()[0]
            df.loc[outliers, "previous_bookings_not_canceled"] = mode_prev_book
            operations.append(f"previous_bookings_not_canceled: {outliers.sum()} valeurs aberrantes corrigées")

    # reserved_room_type (catégoriel) - harmonisation
    if 'reserved_room_type' in df.columns:
        valid_rooms = ["A", "B", "C", "D", "E", "F", "G", "H", "L", "P"]
        df["reserved_room_type"] = df["reserved_room_type"].apply(
            lambda v: harmonize_category(v, valid_rooms)
        )

    # assigned_room_type (catégoriel) - harmonisation
    if 'assigned_room_type' in df.columns:
        valid_rooms = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "K", "L", "P"]
        df["assigned_room_type"] = df["assigned_room_type"].apply(
            lambda v: harmonize_category(v, valid_rooms)
        )

    # booking_changes (numérique) - valeurs aberrantes (max=21)
    if 'booking_changes' in df.columns:
        upper_bound = df["booking_changes"].quantile(0.995)
        outliers = df["booking_changes"] > upper_bound
        if outliers.any():
            mode_book_changes = df.loc[~outliers, "booking_changes"].mode()[0]
            df.loc[outliers, "booking_changes"] = mode_book_changes
            operations.append(f"booking_changes: {outliers.sum()} valeurs aberrantes corrigées")

    # deposit_type (catégoriel) - harmonisation
    if 'deposit_type' in df.columns:
        valid_deposits = ["No Deposit", "Non Refund"]
        df["deposit_type"] = df["deposit_type"].apply(
            lambda v: harmonize_category(v, valid_deposits)
        )

    # agent (texte mais devrait être numérique) - extraction numérique
    if 'agent' in df.columns:
        df["agent"] = df["agent"].replace("unknown", np.nan)
        df["agent"] = df["agent"].apply(extract_numeric)
        # Imputation par mode (9.0 est le plus fréquent)
        mode_agent = df["agent"].mode()[0]
        df["agent"].fillna(mode_agent, inplace=True)
        operations.append(f"agent: {df['agent'].isna().sum()} valeurs manquantes imputées par mode")

    # company (numérique) - trop de valeurs manquantes (94.31%) -> suppression
    if 'company' in df.columns:
        df.drop(columns=['company'], inplace=True)
        operations.append("company: colonne supprimée (94.31% de valeurs manquantes)")

    # days_in_waiting_list (numérique) - valeurs aberrantes (max=8999)
    if 'days_in_waiting_list' in df.columns:
        upper_bound = df["days_in_waiting_list"].quantile(0.995)
        outliers = df["days_in_waiting_list"] > upper_bound
        if outliers.any():
            median_wait = df.loc[~outliers, "days_in_waiting_list"].median()
            df.loc[outliers, "days_in_waiting_list"] = median_wait
            operations.append(f"days_in_waiting_list: {outliers.sum()} valeurs aberrantes corrigées")

    # customer_type (catégoriel) - harmonisation
    if 'customer_type' in df.columns:
        valid_customers = ["Transient", "Transient-Party", "Contract", "Group"]
        df["customer_type"] = df["customer_type"].apply(
            lambda v: harmonize_category(v, valid_customers)
        )

    # adr (numérique) - valeurs aberrantes (min=-496.7, max=9997.01)
    if 'adr' in df.columns:
        # Bornes basées sur le 0.5e et 99.5e percentiles
        lower_bound = df["adr"].quantile(0.005)
        upper_bound = df["adr"].quantile(0.995)
        outliers = (df["adr"] < lower_bound) | (df["adr"] > upper_bound)
        if outliers.any():
            median_adr = df.loc[~outliers, "adr"].median()
            df.loc[outliers, "adr"] = median_adr
            operations.append(f"adr: {outliers.sum()} valeurs aberrantes corrigées")

    # required_car_parking_spaces (numérique) - pas de traitement nécessaire

    # total_of_special_requests (numérique) - pas de traitement nécessaire

    # reservation_status (catégoriel) - harmonisation
    if 'reservation_status' in df.columns:
        valid_status = ["Check-Out", "Canceled", "No-Show"]
        df["reservation_status"] = df["reservation_status"].apply(
            lambda v: harmonize_category(v, valid_status)
        )

    # reservation_status_date (date) - parsing et reformatage
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
                if col in ['adults', 'children', 'babies', 'previous_cancellations',
                          'previous_bookings_not_canceled', 'booking_changes']:
                    df[col].fillna(df[col].mode()[0], inplace=True)
                else:
                    df[col].fillna(df[col].median(), inplace=True)

    # 4. Sauvegarde
    df.to_csv(OUTPUT_PATH, index=False)

    # Log des opérations
    print(f"Nettoyage terminé. Lignes initiales: {initial_rows}, lignes finales: {len(df)}")
    if duplicates_removed > 0:
        print(f"- {duplicates_removed} doublons supprimés")
    for op in operations:
        print(f"- {op}")

if __name__ == "__main__":
    main()