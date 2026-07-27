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

def clean_dataset():
    df = pd.read_csv(INPUT_PATH)

    # Log initial
    initial_rows = len(df)
    initial_cols = len(df.columns)

    # 1. Suppression des doublons (en conservant le premier)
    df.drop_duplicates(subset=[col for col in df.columns if col != 'row_id'], keep='first', inplace=True)
    duplicates_removed = initial_rows - len(df)

    # 2. Nettoyage des colonnes numériques avec texte parasite
    numeric_cols_with_text = ['lead_time', 'adults', 'children', 'stays_in_week_nights']
    for col in numeric_cols_with_text:
        if col in df.columns:
            df[col] = df[col].apply(extract_numeric)

    # 3. Correction des valeurs aberrantes pour les colonnes numériques
    # stays_in_week_nights: max=999.0 (profil) -> valeur aberrante (99.5e percentile ~ 14)
    if 'stays_in_week_nights' in df.columns:
        q995 = df['stays_in_week_nights'].quantile(0.995)
        outliers = df['stays_in_week_nights'] > q995
        if outliers.any():
            median_val = df.loc[~outliers, 'stays_in_week_nights'].median()
            df.loc[outliers, 'stays_in_week_nights'] = median_val

    # adr: min=-494.03, max=9997.54 -> valeurs aberrantes (99.5e percentile ~ 300)
    if 'adr' in df.columns:
        q995 = df['adr'].quantile(0.995)
        outliers = (df['adr'] > q995) | (df['adr'] < 0)
        if outliers.any():
            median_val = df.loc[~outliers, 'adr'].median()
            df.loc[outliers, 'adr'] = median_val

    # babies: max=49.0 -> valeur aberrante (99.5e percentile ~ 2)
    if 'babies' in df.columns:
        q995 = df['babies'].quantile(0.995)
        outliers = df['babies'] > q995
        if outliers.any():
            mode_val = df.loc[~outliers, 'babies'].mode()[0]
            df.loc[outliers, 'babies'] = mode_val

    # 4. Harmonisation des colonnes catégorielles
    # hotel
    if 'hotel' in df.columns:
        valid_values_hotel = ["City Hotel", "Resort Hotel"]
        df['hotel'] = df['hotel'].apply(lambda v: harmonize_category(v, valid_values_hotel))

    # meal
    if 'meal' in df.columns:
        valid_values_meal = ["BB", "HB", "FB", "SC", "Undefined"]
        df['meal'] = df['meal'].apply(lambda v: harmonize_category(v, valid_values_meal))

    # country (ne pas harmoniser - codes ISO à 3 lettres)
    if 'country' in df.columns:
        df['country'] = df['country'].str.strip().str.upper()

    # market_segment
    if 'market_segment' in df.columns:
        valid_values_market = ["Online TA", "Offline TA/TO", "Groups", "Direct", "Corporate", "Complementary", "Aviation"]
        df['market_segment'] = df['market_segment'].apply(lambda v: harmonize_category(v, valid_values_market))

    # distribution_channel
    if 'distribution_channel' in df.columns:
        valid_values_dist = ["TA/TO", "Direct", "Corporate", "GDS", "Undefined"]
        df['distribution_channel'] = df['distribution_channel'].apply(lambda v: harmonize_category(v, valid_values_dist))

    # reserved_room_type et assigned_room_type
    for col in ['reserved_room_type', 'assigned_room_type']:
        if col in df.columns:
            df[col] = df[col].str.strip().str.upper()

    # deposit_type
    if 'deposit_type' in df.columns:
        valid_values_deposit = ["No Deposit", "Non Refund", "Refundable"]
        df['deposit_type'] = df['deposit_type'].apply(lambda v: harmonize_category(v, valid_values_deposit))

    # customer_type
    if 'customer_type' in df.columns:
        valid_values_customer = ["Transient", "Transient-Party", "Contract", "Group"]
        df['customer_type'] = df['customer_type'].apply(lambda v: harmonize_category(v, valid_values_customer))

    # 5. Conversion des dates
    if 'reservation_status_date' in df.columns:
        df['reservation_status_date'] = df['reservation_status_date'].apply(parse_date)

    # 6. Imputation des valeurs manquantes
    # children: 8.32% manquants -> imputation par mode (0.0)
    if 'children' in df.columns:
        mode_children = df['children'].mode()[0]
        df['children'] = df['children'].fillna(mode_children)

    # meal: 8.35% manquants -> imputation par mode (BB)
    if 'meal' in df.columns:
        mode_meal = df['meal'].mode()[0]
        df['meal'] = df['meal'].fillna(mode_meal)

    # country: 9.16% manquants -> imputation par mode (PRT)
    if 'country' in df.columns:
        mode_country = df['country'].mode()[0]
        df['country'] = df['country'].fillna(mode_country)

    # market_segment: 8.32% manquants -> imputation par mode (Online TA)
    if 'market_segment' in df.columns:
        mode_market = df['market_segment'].mode()[0]
        df['market_segment'] = df['market_segment'].fillna(mode_market)

    # agent: 34.29% manquants -> imputation par mode (9.0)
    if 'agent' in df.columns:
        mode_agent = df['agent'].mode()[0]
        df['agent'] = df['agent'].fillna(mode_agent)

    # company: 94.31% manquants -> trop élevé, on laisse NaN
    # Pas d'imputation pour company

    # 7. Conversion des types
    # is_canceled, is_repeated_guest, arrival_date_year, arrival_date_week_number,
    # arrival_date_day_of_month, stays_in_weekend_nights, stays_in_week_nights,
    # previous_cancellations, previous_bookings_not_canceled, booking_changes,
    # days_in_waiting_list, required_car_parking_spaces, total_of_special_requests
    int_cols = [
        'is_canceled', 'is_repeated_guest', 'arrival_date_year',
        'arrival_date_week_number', 'arrival_date_day_of_month',
        'stays_in_weekend_nights', 'stays_in_week_nights',
        'previous_cancellations', 'previous_bookings_not_canceled',
        'booking_changes', 'days_in_waiting_list',
        'required_car_parking_spaces', 'total_of_special_requests'
    ]
    for col in int_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].fillna(df[col].median()).astype(int)

    # adults, children, babies
    for col in ['adults', 'children', 'babies']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].fillna(df[col].median()).astype(int)

    # adr
    if 'adr' in df.columns:
        df['adr'] = pd.to_numeric(df['adr'], errors='coerce')
        df['adr'] = df['adr'].fillna(df['adr'].median())

    # 8. Nettoyage des espaces dans les chaînes
    str_cols = df.select_dtypes(include=['object']).columns
    for col in str_cols:
        if col != 'row_id':
            df[col] = df[col].str.strip()

    # Log final
    final_rows = len(df)
    final_cols = len(df.columns)

    print(f"Nettoyage terminé:")
    print(f"- Lignes initiales: {initial_rows}, finales: {final_rows} (doublons supprimés: {duplicates_removed})")
    print(f"- Colonnes initiales: {initial_cols}, finales: {final_cols}")
    print(f"- Valeurs manquantes imputées:")
    print(f"  - children: {df['children'].isna().sum()}")
    print(f"  - meal: {df['meal'].isna().sum()}")
    print(f"  - country: {df['country'].isna().sum()}")
    print(f"  - market_segment: {df['market_segment'].isna().sum()}")
    print(f"  - agent: {df['agent'].isna().sum()}")
    print(f"- Valeurs aberrantes corrigées dans stays_in_week_nights, adr, babies")

    # Sauvegarde
    df.to_csv(OUTPUT_PATH, index=False)

if __name__ == "__main__":
    clean_dataset()