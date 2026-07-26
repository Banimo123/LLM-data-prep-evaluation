import pandas as pd
import numpy as np
import re
from datetime import datetime

INPUT_PATH = r"benchmark\datasets\hotel_bookings\noisy_medium.csv"
OUTPUT_PATH = r"results\cleaned_datasets\hotel_bookings\noisy_medium__profile.csv"

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
        'duplicates_removed': 0,
        'category_harmonized': {},
        'format_corrected': {}
    }

    # 1. Suppression des doublons (conservation de la première occurrence)
    initial_rows = len(df)
    df.drop_duplicates(subset=df.columns.difference(['row_id']), keep='first', inplace=True)
    operations_log['duplicates_removed'] = initial_rows - len(df)

    # 2. Traitement des colonnes numériques avec valeurs parasites
    numeric_cols_with_text = ['lead_time', 'stays_in_week_nights', 'adults', 'children', 'babies', 'adr']
    for col in numeric_cols_with_text:
        if col in df.columns:
            df[col] = df[col].apply(extract_numeric)
            operations_log['format_corrected'][col] = "Extracted numeric values from text"

    # 3. Correction des valeurs aberrantes basées sur les statistiques fournies
    # stays_in_week_nights: max 999 -> valeur aberrante (moyenne 14.43)
    if 'stays_in_week_nights' in df.columns:
        df['stays_in_week_nights'] = df['stays_in_week_nights'].clip(upper=30)  # 30 comme limite haute plausible
        operations_log['outliers_corrected']['stays_in_week_nights'] = "Clipped values > 30"

    # adr: min -493.26 -> valeur aberrante (prix ne peut pas être négatif)
    if 'adr' in df.columns:
        df['adr'] = df['adr'].clip(lower=0)
        operations_log['outliers_corrected']['adr'] = "Clipped negative values"

    # babies: max 49 -> valeur aberrante (moyenne 0.6)
    if 'babies' in df.columns:
        df['babies'] = df['babies'].clip(upper=10)  # 10 comme limite haute plausible
        operations_log['outliers_corrected']['babies'] = "Clipped values > 10"

    # 4. Traitement des valeurs manquantes
    # children: 10% manquants -> imputation par mode (0.0)
    if 'children' in df.columns:
        mode_children = df['children'].mode()[0]
        df['children'].fillna(mode_children, inplace=True)
        operations_log['missing_values_imputed']['children'] = f"Imputed with mode: {mode_children}"

    # meal: 10% manquants -> imputation par mode (BB)
    if 'meal' in df.columns:
        mode_meal = df['meal'].mode()[0]
        df['meal'].fillna(mode_meal, inplace=True)
        operations_log['missing_values_imputed']['meal'] = f"Imputed with mode: {mode_meal}"

    # country: 10.37% manquants -> imputation par mode (PRT)
    if 'country' in df.columns:
        mode_country = df['country'].mode()[0]
        df['country'].fillna(mode_country, inplace=True)
        operations_log['missing_values_imputed']['country'] = f"Imputed with mode: {mode_country}"

    # market_segment: 10% manquants -> imputation par mode (Online TA)
    if 'market_segment' in df.columns:
        mode_market = df['market_segment'].mode()[0]
        df['market_segment'].fillna(mode_market, inplace=True)
        operations_log['missing_values_imputed']['market_segment'] = f"Imputed with mode: {mode_market}"

    # agent: 22.33% manquants -> imputation par mode (9.0)
    if 'agent' in df.columns:
        mode_agent = df['agent'].mode()[0]
        df['agent'].fillna(mode_agent, inplace=True)
        operations_log['missing_values_imputed']['agent'] = f"Imputed with mode: {mode_agent}"

    # company: 94.31% manquants -> trop élevé, on conserve les NaN
    if 'company' in df.columns:
        operations_log['missing_values_imputed']['company'] = "Too many missing values (94.31%), left as NaN"

    # 5. Harmonisation des catégories
    # hotel: correction des variantes (espaces superflus)
    if 'hotel' in df.columns:
        df['hotel'] = df['hotel'].str.strip()
        df['hotel'] = df['hotel'].replace({'Resoort Hotel': 'Resort Hotel', 'City Hotel  ': 'City Hotel'})
        operations_log['category_harmonized']['hotel'] = "Corrected variants (Resoort Hotel -> Resort Hotel, stripped spaces)"

    # deposit_type: correction des variantes (espaces superflus)
    if 'deposit_type' in df.columns:
        df['deposit_type'] = df['deposit_type'].str.strip()
        df['deposit_type'] = df['deposit_type'].replace({'NoD eposit': 'No Deposit', 'No Deposit  ': 'No Deposit'})
        operations_log['category_harmonized']['deposit_type'] = "Corrected variants (NoD eposit -> No Deposit, stripped spaces)"

    # adults: correction de '2O' -> '2'
    if 'adults' in df.columns:
        df['adults'] = df['adults'].replace('2O', '2')
        operations_log['category_harmonized']['adults'] = "Corrected '2O' to '2'"

    # 6. Correction des formats de date
    if 'reservation_status_date' in df.columns:
        def parse_date(date_str):
            if pd.isna(date_str):
                return date_str
            try:
                # Essaye d'abord le format YYYY-MM-DD
                return pd.to_datetime(date_str, format='%Y-%m-%d', errors='raise')
            except:
                try:
                    # Essaye le format YYYY/MM/DD
                    return pd.to_datetime(date_str, format='%Y/%m/%d', errors='raise')
                except:
                    # Conserve la valeur originale si aucun format ne fonctionne
                    return date_str

        df['reservation_status_date'] = df['reservation_status_date'].apply(parse_date)
        operations_log['format_corrected']['reservation_status_date'] = "Standardized date formats"

    # 7. Conversion des colonnes numériques au bon type
    numeric_cols = ['is_canceled', 'arrival_date_year', 'arrival_date_week_number',
                    'arrival_date_day_of_month', 'stays_in_weekend_nights',
                    'stays_in_week_nights', 'adults', 'children', 'babies',
                    'is_repeated_guest', 'previous_cancellations',
                    'previous_bookings_not_canceled', 'booking_changes',
                    'days_in_waiting_list', 'adr', 'required_car_parking_spaces',
                    'total_of_special_requests']

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='ignore')

    # 8. Vérification finale des types
    df['row_id'] = df['row_id'].astype(str)  # Conservation du row_id comme string

    # Sauvegarde du dataset nettoyé
    df.to_csv(OUTPUT_PATH, index=False)

    # Affichage du log des opérations
    print("=== Nettoyage terminé ===")
    print(f"Lignes supprimées (doublons): {operations_log['duplicates_removed']}")
    print("\nValeurs manquantes imputées:")
    for col, method in operations_log['missing_values_imputed'].items():
        print(f"  - {col}: {method}")

    print("\nValeurs aberrantes corrigées:")
    for col, method in operations_log['outliers_corrected'].items():
        print(f"  - {col}: {method}")

    print("\nCatégories harmonisées:")
    for col, method in operations_log['category_harmonized'].items():
        print(f"  - {col}: {method}")

    print("\nFormats corrigés:")
    for col, method in operations_log['format_corrected'].items():
        print(f"  - {col}: {method}")

    print(f"\nDataset nettoyé sauvegardé dans: {OUTPUT_PATH}")

if __name__ == "__main__":
    clean_hotel_bookings()