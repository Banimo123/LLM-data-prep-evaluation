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
    df = df.drop_duplicates(subset=[col for col in df.columns if col != 'row_id'], keep='first')
    operations_log['duplicates_removed'] = operations_log['rows_before'] - len(df)

    # 2. Traitement des valeurs manquantes
    # children (20% manquants) - colonne catégorielle avec valeurs fréquentes 0.0, unknown, ' '
    if 'children' in df.columns:
        mode_children = df['children'].mode()[0]
        df['children'] = df['children'].replace(['', ' ', 'unknown'], np.nan)
        df['children'] = df['children'].fillna(mode_children)
        operations_log['missing_values_imputed']['children'] = df['children'].isna().sum()

    # meal (20% manquants) - colonne catégorielle
    if 'meal' in df.columns:
        mode_meal = df['meal'].mode()[0]
        df['meal'] = df['meal'].fillna(mode_meal)
        operations_log['missing_values_imputed']['meal'] = df['meal'].isna().sum()

    # country (20.33% manquants) - colonne catégorielle
    if 'country' in df.columns:
        mode_country = df['country'].mode()[0]
        df['country'] = df['country'].fillna(mode_country)
        operations_log['missing_values_imputed']['country'] = df['country'].isna().sum()

    # market_segment (20% manquants) - colonne catégorielle
    if 'market_segment' in df.columns:
        mode_market_segment = df['market_segment'].mode()[0]
        df['market_segment'] = df['market_segment'].fillna(mode_market_segment)
        operations_log['missing_values_imputed']['market_segment'] = df['market_segment'].isna().sum()

    # agent (30.99% manquants) - colonne catégorielle
    if 'agent' in df.columns:
        mode_agent = df['agent'].mode()[0]
        df['agent'] = df['agent'].fillna(mode_agent)
        operations_log['missing_values_imputed']['agent'] = df['agent'].isna().sum()

    # company (94.31% manquants) - trop de manquants, on laisse NaN
    if 'company' in df.columns:
        operations_log['missing_values_imputed']['company'] = 0  # pas d'imputation

    # 3. Correction des valeurs aberrantes numériques
    # adults - contient des valeurs comme "2O" (faute de frappe)
    if 'adults' in df.columns:
        df['adults'] = df['adults'].apply(extract_numeric)
        median_adults = df['adults'].median()
        df['adults'] = df['adults'].fillna(median_adults)
        operations_log['outliers_corrected']['adults'] = (df['adults'] > 10).sum()  # valeurs >10 considérées aberrantes

    # stays_in_week_nights - max 999.0 semble aberrant
    if 'stays_in_week_nights' in df.columns:
        q99 = df['stays_in_week_nights'].quantile(0.99)
        df.loc[df['stays_in_week_nights'] > q99, 'stays_in_week_nights'] = q99
        operations_log['outliers_corrected']['stays_in_week_nights'] = (df['stays_in_week_nights'] > q99).sum()

    # adr - contient des valeurs négatives et très élevées
    if 'adr' in df.columns:
        q1 = df['adr'].quantile(0.01)
        q99 = df['adr'].quantile(0.99)
        df.loc[df['adr'] < q1, 'adr'] = q1
        df.loc[df['adr'] > q99, 'adr'] = q99
        operations_log['outliers_corrected']['adr'] = ((df['adr'] < q1) | (df['adr'] > q99)).sum()

    # days_in_waiting_list - max 8999 semble aberrant
    if 'days_in_waiting_list' in df.columns:
        q99 = df['days_in_waiting_list'].quantile(0.99)
        df.loc[df['days_in_waiting_list'] > q99, 'days_in_waiting_list'] = q99
        operations_log['outliers_corrected']['days_in_waiting_list'] = (df['days_in_waiting_list'] > q99).sum()

    # 4. Harmonisation des catégories
    # hotel - correction des espaces superflus
    if 'hotel' in df.columns:
        df['hotel'] = df['hotel'].str.strip()
        df['hotel'] = df['hotel'].replace({'City Hotel  ': 'City Hotel', 'Resort Hotel ': 'Resort Hotel'})
        operations_log['categories_harmonized']['hotel'] = (df['hotel'].isin(['City Hotel', 'Resort Hotel'])).sum()

    # adults - correction des fautes de frappe
    if 'adults' in df.columns:
        df['adults'] = df['adults'].astype(float)
        operations_log['categories_harmonized']['adults'] = (df['adults'].isin([1.0, 2.0])).sum()

    # meal - correction des variantes
    if 'meal' in df.columns:
        df['meal'] = df['meal'].replace({'HB ': 'HB', 'SC ': 'SC'})
        operations_log['categories_harmonized']['meal'] = (df['meal'].isin(['BB', 'HB', 'SC'])).sum()

    # deposit_type - correction des variantes
    if 'deposit_type' in df.columns:
        df['deposit_type'] = df['deposit_type'].str.strip()
        df['deposit_type'] = df['deposit_type'].replace({
            'No Deposit  ': 'No Deposit',
            'No Depossit': 'No Deposit',
            'Non Refund ': 'Non Refund'
        })
        operations_log['categories_harmonized']['deposit_type'] = (df['deposit_type'].isin(['No Deposit', 'Non Refund'])).sum()

    # customer_type - correction de "Trasnient" en "Transient"
    if 'customer_type' in df.columns:
        df['customer_type'] = df['customer_type'].replace({'Trasnient': 'Transient'})
        operations_log['categories_harmonized']['customer_type'] = (df['customer_type'] == 'Transient').sum()

    # 5. Correction des formats
    # reservation_status_date - harmonisation des formats de date
    if 'reservation_status_date' in df.columns:
        def parse_date(date_str):
            if pd.isna(date_str):
                return date_str
            date_str = str(date_str).strip()
            for fmt in ('%Y-%m-%d', '%d/%m/%Y', '%m/%d/%Y', '%B %d, %Y', '%Y/%m/%d'):
                try:
                    return datetime.strptime(date_str, fmt).strftime('%Y-%m-%d')
                except ValueError:
                    continue
            return date_str  # retourne la valeur originale si aucun format ne correspond

        df['reservation_status_date'] = df['reservation_status_date'].apply(parse_date)
        operations_log['formats_corrected']['reservation_status_date'] = df['reservation_status_date'].notna().sum()

    # 6. Conversion des types
    numeric_cols = [
        'is_canceled', 'arrival_date_year', 'arrival_date_week_number',
        'arrival_date_day_of_month', 'stays_in_weekend_nights',
        'stays_in_week_nights', 'babies', 'is_repeated_guest',
        'previous_cancellations', 'previous_bookings_not_canceled',
        'booking_changes', 'days_in_waiting_list', 'adr',
        'required_car_parking_spaces', 'total_of_special_requests'
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='ignore')

    operations_log['rows_after'] = len(df)

    # Sauvegarde du fichier nettoyé
    df.to_csv(OUTPUT_PATH, index=False)

    # Affichage du log
    print("=== Nettoyage terminé ===")
    print(f"Lignes avant: {operations_log['rows_before']}")
    print(f"Lignes après: {operations_log['rows_after']}")
    print(f"Doublons supprimés: {operations_log['duplicates_removed']}")
    print("\nValeurs manquantes imputées:")
    for col, count in operations_log['missing_values_imputed'].items():
        print(f"  {col}: {count}")
    print("\nValeurs aberrantes corrigées:")
    for col, count in operations_log['outliers_corrected'].items():
        print(f"  {col}: {count}")
    print("\nCatégories harmonisées:")
    for col, count in operations_log['categories_harmonized'].items():
        print(f"  {col}: {count}")
    print("\nFormats corrigés:")
    for col, count in operations_log['formats_corrected'].items():
        print(f"  {col}: {count}")

if __name__ == "__main__":
    clean_hotel_bookings()