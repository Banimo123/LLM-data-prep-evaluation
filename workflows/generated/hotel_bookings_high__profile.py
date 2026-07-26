import pandas as pd
import numpy as np
import re
from datetime import datetime

INPUT_PATH = r"benchmark\datasets\hotel_bookings\noisy_high.csv"
OUTPUT_PATH = r"results\cleaned_datasets\hotel_bookings\noisy_high__profile.csv"

def extract_numeric(value):
    if pd.isna(value):
        return np.nan
    s = str(value).replace(",", ".")
    s = s.replace("O", "0").replace("o", "0")
    match = re.search(r"-?\d+\.?\d*", s)
    return float(match.group()) if match else np.nan

def clean_hotel_bookings():
    df = pd.read_csv(INPUT_PATH)

    operations_log = {
        'missing_values_imputed': {},
        'outliers_corrected': {},
        'categories_harmonized': {},
        'formats_corrected': {},
        'duplicates_removed': 0,
        'rows_before': len(df),
        'rows_after': None
    }

    # 1. Suppression des doublons (en conservant le premier)
    df = df.drop_duplicates(subset=df.columns.difference(['row_id']), keep='first')
    operations_log['duplicates_removed'] = operations_log['rows_before'] - len(df)

    # 2. Traitement des valeurs manquantes
    # children: 20% manquants -> imputation par mode (0.0)
    if 'children' in df.columns:
        mode_children = df['children'].mode()[0]
        df['children'] = df['children'].fillna(mode_children)
        operations_log['missing_values_imputed']['children'] = f"mode: {mode_children}"

    # meal: 20% manquants -> imputation par mode (BB)
    if 'meal' in df.columns:
        mode_meal = df['meal'].mode()[0]
        df['meal'] = df['meal'].fillna(mode_meal)
        operations_log['missing_values_imputed']['meal'] = f"mode: {mode_meal}"

    # country: 20.33% manquants -> imputation par mode (PRT)
    if 'country' in df.columns:
        mode_country = df['country'].mode()[0]
        df['country'] = df['country'].fillna(mode_country)
        operations_log['missing_values_imputed']['country'] = f"mode: {mode_country}"

    # market_segment: 20% manquants -> imputation par mode (Online TA)
    if 'market_segment' in df.columns:
        mode_market_segment = df['market_segment'].mode()[0]
        df['market_segment'] = df['market_segment'].fillna(mode_market_segment)
        operations_log['missing_values_imputed']['market_segment'] = f"mode: {mode_market_segment}"

    # agent: 30.99% manquants -> imputation par mode (9.0)
    if 'agent' in df.columns:
        mode_agent = df['agent'].mode()[0]
        df['agent'] = df['agent'].fillna(mode_agent)
        operations_log['missing_values_imputed']['agent'] = f"mode: {mode_agent}"

    # company: 94.31% manquants -> trop élevé, on laisse NaN
    # Pas d'imputation car trop de valeurs manquantes

    # 3. Correction des valeurs aberrantes numériques
    # adults: contient des valeurs comme "2O" -> extraction numérique
    if 'adults' in df.columns:
        df['adults'] = df['adults'].apply(extract_numeric)
        median_adults = df['adults'].median()
        df['adults'] = df['adults'].fillna(median_adults)
        operations_log['outliers_corrected']['adults'] = f"extracted numeric, median: {median_adults}"

    # children: contient "unknown" -> remplacement par mode (0.0)
    if 'children' in df.columns:
        df['children'] = df['children'].replace('unknown', np.nan)
        df['children'] = df['children'].apply(extract_numeric)
        mode_children = df['children'].mode()[0]
        df['children'] = df['children'].fillna(mode_children)
        operations_log['outliers_corrected']['children'] = f"replaced 'unknown' with mode: {mode_children}"

    # stays_in_week_nights: max 999 -> valeur aberrante (moyenne 26.64)
    if 'stays_in_week_nights' in df.columns:
        q99 = df['stays_in_week_nights'].quantile(0.99)
        df.loc[df['stays_in_week_nights'] > q99, 'stays_in_week_nights'] = q99
        operations_log['outliers_corrected']['stays_in_week_nights'] = f"capped at 99th percentile: {q99}"

    # adr: contient des valeurs négatives (-496.7) -> on les met à 0
    if 'adr' in df.columns:
        df['adr'] = df['adr'].apply(extract_numeric)
        df.loc[df['adr'] < 0, 'adr'] = 0
        operations_log['outliers_corrected']['adr'] = "negative values set to 0"

    # days_in_waiting_list: max 8999 -> valeur aberrante (moyenne 222.8)
    if 'days_in_waiting_list' in df.columns:
        q99 = df['days_in_waiting_list'].quantile(0.99)
        df['days_in_waiting_list'] = df['days_in_waiting_list'].apply(lambda x: q99 if x > q99 else x)
        operations_log['outliers_corrected']['days_in_waiting_list'] = f"capped at 99th percentile: {q99}"

    # 4. Harmonisation des catégories
    # hotel: espaces superflus dans "City Hotel  " -> "City Hotel"
    if 'hotel' in df.columns:
        df['hotel'] = df['hotel'].str.strip()
        operations_log['categories_harmonized']['hotel'] = "stripped whitespace"

    # meal: harmonisation des variantes
    if 'meal' in df.columns:
        meal_mapping = {
            'BB': 'BB',
            'HB': 'HB',
            'SC': 'SC',
            'Undefined': 'SC',
            'Full Board': 'FB'
        }
        df['meal'] = df['meal'].replace(meal_mapping)
        operations_log['categories_harmonized']['meal'] = str(meal_mapping)

    # country: harmonisation des variantes (ex: 'USA' vs 'US')
    if 'country' in df.columns:
        country_mapping = {
            'USA': 'US',
            'United States': 'US',
            'UK': 'GB'
        }
        df['country'] = df['country'].replace(country_mapping)
        operations_log['categories_harmonized']['country'] = str(country_mapping)

    # market_segment: harmonisation des variantes
    if 'market_segment' in df.columns:
        market_segment_mapping = {
            'Online': 'Online TA',
            'Offline': 'Offline TA/TO',
            'Groups': 'Groups',
            'Direct': 'Direct',
            'Corporate': 'Corporate',
            'Complementary': 'Complementary',
            'Aviation': 'Aviation'
        }
        df['market_segment'] = df['market_segment'].replace(market_segment_mapping)
        operations_log['categories_harmonized']['market_segment'] = str(market_segment_mapping)

    # distribution_channel: harmonisation des variantes
    if 'distribution_channel' in df.columns:
        distribution_channel_mapping = {
            'TA': 'TA/TO',
            'TO': 'TA/TO',
            'Direct': 'Direct',
            'Corporate': 'Corporate',
            'GDS': 'GDS'
        }
        df['distribution_channel'] = df['distribution_channel'].replace(distribution_channel_mapping)
        operations_log['categories_harmonized']['distribution_channel'] = str(distribution_channel_mapping)

    # deposit_type: harmonisation des variantes
    if 'deposit_type' in df.columns:
        deposit_type_mapping = {
            'No Deposit': 'No Deposit',
            'Non Refund': 'Non Refund',
            'Refundable': 'Refundable'
        }
        df['deposit_type'] = df['deposit_type'].replace(deposit_type_mapping)
        df['deposit_type'] = df['deposit_type'].str.strip()
        operations_log['categories_harmonized']['deposit_type'] = str(deposit_type_mapping) + ", stripped whitespace"

    # customer_type: harmonisation des variantes
    if 'customer_type' in df.columns:
        customer_type_mapping = {
            'Transient': 'Transient',
            'Transient-Party': 'Transient-Party',
            'Contract': 'Contract',
            'Group': 'Group',
            'Trasnient': 'Transient',
            'TRansient': 'Transient'
        }
        df['customer_type'] = df['customer_type'].replace(customer_type_mapping)
        operations_log['categories_harmonized']['customer_type'] = str(customer_type_mapping)

    # 5. Correction des formats
    # reservation_status_date: harmonisation des formats de date
    if 'reservation_status_date' in df.columns:
        def parse_date(date_str):
            if pd.isna(date_str):
                return date_str
            date_str = str(date_str).strip()
            for fmt in ('%Y-%m-%d', '%d/%m/%Y', '%m/%d/%Y', '%B %d, %Y', '%b %d, %Y'):
                try:
                    return datetime.strptime(date_str, fmt).strftime('%Y-%m-%d')
                except ValueError:
                    continue
            return date_str  # retourne la valeur originale si aucun format ne correspond

        df['reservation_status_date'] = df['reservation_status_date'].apply(parse_date)
        operations_log['formats_corrected']['reservation_status_date'] = "standardized to YYYY-MM-DD"

    # 6. Conversion des types
    numeric_cols = [
        'is_canceled', 'arrival_date_year', 'arrival_date_week_number',
        'arrival_date_day_of_month', 'stays_in_weekend_nights',
        'stays_in_week_nights', 'adults', 'children', 'babies',
        'is_repeated_guest', 'previous_cancellations',
        'previous_bookings_not_canceled', 'booking_changes',
        'days_in_waiting_list', 'adr', 'required_car_parking_spaces',
        'total_of_special_requests'
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: extract_numeric(x) if pd.notna(x) else x)
            if df[col].dtype == 'object':
                df[col] = pd.to_numeric(df[col], errors='coerce')
            if df[col].dtype in ['float64', 'int64']:
                if col in ['days_in_waiting_list', 'children', 'babies', 'previous_cancellations',
                          'previous_bookings_not_canceled', 'booking_changes',
                          'required_car_parking_spaces', 'total_of_special_requests']:
                    df[col] = df[col].round().astype('Int64')
                else:
                    df[col] = df[col].astype('float64')

    operations_log['rows_after'] = len(df)

    # Sauvegarde du fichier nettoyé
    df.to_csv(OUTPUT_PATH, index=False)

    # Affichage du log des opérations
    print("=== Nettoyage terminé ===")
    print(f"Lignes avant nettoyage: {operations_log['rows_before']}")
    print(f"Lignes après nettoyage: {operations_log['rows_after']}")
    print(f"Doublons supprimés: {operations_log['duplicates_removed']}")

    print("\nValeurs manquantes imputées:")
    for col, method in operations_log['missing_values_imputed'].items():
        print(f"- {col}: {method}")

    print("\nValeurs aberrantes corrigées:")
    for col, method in operations_log['outliers_corrected'].items():
        print(f"- {col}: {method}")

    print("\nCatégories harmonisées:")
    for col, method in operations_log['categories_harmonized'].items():
        print(f"- {col}: {method}")

    print("\nFormats corrigés:")
    for col, method in operations_log['formats_corrected'].items():
        print(f"- {col}: {method}")

if __name__ == "__main__":
    clean_hotel_bookings()