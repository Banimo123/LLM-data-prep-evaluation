import pandas as pd
import numpy as np
import re
from datetime import datetime

INPUT_PATH = "benchmark/datasets/hotel_bookings/noisy_low.csv"
OUTPUT_PATH = "results/cleaned_datasets/hotel_bookings/noisy_low__validated.csv"

def clean_hotel_bookings(input_path, output_path):
    df = pd.read_csv(input_path)

    operations_log = {
        'missing_values_handled': {},
        'duplicates_removed': 0,
        'format_corrections': {},
        'outliers_corrected': {},
        'category_standardizations': {}
    }

    # Correction des formats de colonnes
    if 'lead_time' in df.columns:
        df['lead_time'] = df['lead_time'].replace({'O': '0'}, regex=True)
        df['lead_time'] = pd.to_numeric(df['lead_time'], errors='coerce').fillna(0).astype(int)
        operations_log['format_corrections']['lead_time'] = "Corrected O to 0 and converted to int"

    if 'children' in df.columns:
        df['children'] = pd.to_numeric(df['children'], errors='coerce').fillna(0).astype(int)
        operations_log['format_corrections']['children'] = "Converted to int"

    if 'babies' in df.columns:
        df['babies'] = pd.to_numeric(df['babies'], errors='coerce').fillna(0).astype(int)
        operations_log['format_corrections']['babies'] = "Converted to int"

    if 'adr' in df.columns:
        df['adr'] = pd.to_numeric(df['adr'], errors='coerce')
        operations_log['format_corrections']['adr'] = "Converted to numeric"

    # Traitement des dates
    if 'reservation_status_date' in df.columns:
        def parse_date(date_str):
            if pd.isna(date_str):
                return pd.NaT
            try:
                return pd.to_datetime(date_str, format='%B %d, %Y', errors='coerce')
            except:
                try:
                    return pd.to_datetime(date_str, format='%Y-%m-%d', errors='coerce')
                except:
                    return pd.NaT

        df['reservation_status_date'] = df['reservation_status_date'].apply(parse_date)
        operations_log['format_corrections']['reservation_status_date'] = "Standardized date format"

    # Standardisation des catégories
    if 'meal' in df.columns:
        meal_mapping = {
            'BB': 'Bed & Breakfast',
            'HB': 'Half Board',
            'FB': 'Full Board',
            'SC': 'Self Catering',
            'Undefined': 'Undefined/SC'
        }
        df['meal'] = df['meal'].replace(meal_mapping)
        operations_log['category_standardizations']['meal'] = f"Standardized to {list(meal_mapping.values())}"

    if 'market_segment' in df.columns:
        df['market_segment'] = df['market_segment'].str.title()
        operations_log['category_standardizations']['market_segment'] = "Capitalized values"

    if 'distribution_channel' in df.columns:
        df['distribution_channel'] = df['distribution_channel'].str.title()
        operations_log['category_standardizations']['distribution_channel'] = "Capitalized values"

    if 'customer_type' in df.columns:
        df['customer_type'] = df['customer_type'].str.title()
        operations_log['category_standardizations']['customer_type'] = "Capitalized values"

    # Correction des valeurs aberrantes
    if 'adults' in df.columns:
        df['adults'] = df['adults'].clip(lower=0, upper=4)
        operations_log['outliers_corrected']['adults'] = "Clipped to 0-4 range"

    if 'children' in df.columns:
        df['children'] = df['children'].clip(lower=0, upper=10)
        operations_log['outliers_corrected']['children'] = "Clipped to 0-10 range"

    if 'babies' in df.columns:
        df['babies'] = df['babies'].clip(lower=0, upper=4)
        operations_log['outliers_corrected']['babies'] = "Clipped to 0-4 range"

    if 'stays_in_weekend_nights' in df.columns:
        df['stays_in_weekend_nights'] = df['stays_in_weekend_nights'].clip(lower=0, upper=14)
        operations_log['outliers_corrected']['stays_in_weekend_nights'] = "Clipped to 0-14 range"

    if 'stays_in_week_nights' in df.columns:
        df['stays_in_week_nights'] = df['stays_in_week_nights'].clip(lower=0, upper=50)
        operations_log['outliers_corrected']['stays_in_week_nights'] = "Clipped to 0-50 range"

    # Traitement des valeurs manquantes
    if 'country' in df.columns:
        df['country'] = df['country'].fillna('Unknown')
        operations_log['missing_values_handled']['country'] = f"Filled {df['country'].isna().sum()} missing values with 'Unknown'"

    if 'agent' in df.columns:
        df['agent'] = df['agent'].fillna(0).astype(int)
        operations_log['missing_values_handled']['agent'] = f"Filled {df['agent'].isna().sum()} missing values with 0"

    if 'company' in df.columns:
        df['company'] = df['company'].fillna(0).astype(int)
        operations_log['missing_values_handled']['company'] = f"Filled {df['company'].isna().sum()} missing values with 0"

    if 'meal' in df.columns:
        df['meal'] = df['meal'].fillna('Undefined/SC')
        operations_log['missing_values_handled']['meal'] = f"Filled {df['meal'].isna().sum()} missing values with 'Undefined/SC'"

    # Suppression des doublons
    initial_rows = len(df)
    df = df.drop_duplicates()
    operations_log['duplicates_removed'] = initial_rows - len(df)

    # Vérification de la cohérence des données
    if all(col in df.columns for col in ['adults', 'children', 'babies']):
        df = df[(df['adults'] + df['children'] + df['babies']) > 0]
        operations_log['outliers_corrected']['guests'] = f"Removed {initial_rows - len(df)} rows with 0 guests"

    # Sauvegarde du fichier nettoyé
    df.to_csv(output_path, index=False)

    # Affichage du log des opérations
    print("\n=== Data Cleaning Summary ===")
    print(f"Initial rows: {initial_rows}")
    print(f"Final rows: {len(df)}")
    print(f"Duplicates removed: {operations_log['duplicates_removed']}")

    print("\nMissing values handled:")
    for col, action in operations_log['missing_values_handled'].items():
        print(f"- {col}: {action}")

    print("\nFormat corrections:")
    for col, action in operations_log['format_corrections'].items():
        print(f"- {col}: {action}")

    print("\nCategory standardizations:")
    for col, action in operations_log['category_standardizations'].items():
        print(f"- {col}: {action}")

    print("\nOutliers corrected:")
    for col, action in operations_log['outliers_corrected'].items():
        print(f"- {col}: {action}")

clean_hotel_bookings(INPUT_PATH, OUTPUT_PATH)