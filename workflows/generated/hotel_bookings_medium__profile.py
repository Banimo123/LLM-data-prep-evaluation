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

    # 2. Nettoyage des colonnes numériques avec extraction de valeurs
    numeric_cols = ['is_canceled', 'arrival_date_year', 'arrival_date_week_number',
                    'arrival_date_day_of_month', 'stays_in_weekend_nights',
                    'stays_in_week_nights', 'babies', 'is_repeated_guest',
                    'previous_cancellations', 'previous_bookings_not_canceled',
                    'booking_changes', 'days_in_waiting_list', 'adr',
                    'required_car_parking_spaces', 'total_of_special_requests']

    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].apply(extract_numeric)

            # Correction des valeurs aberrantes basées sur les stats fournies
            if col == 'stays_in_week_nights':
                max_val = 999.0
                df.loc[df[col] > max_val, col] = max_val
                operations_log['outliers_corrected'][col] = f"Valeurs > {max_val} ramenées à {max_val}"
            elif col == 'adr':
                min_val = -493.26
                max_val = 9993.79
                df.loc[df[col] < min_val, col] = min_val
                df.loc[df[col] > max_val, col] = max_val
                operations_log['outliers_corrected'][col] = f"Valeurs hors [{min_val}, {max_val}] ramenées aux bornes"
            elif col == 'babies':
                max_val = 49.0
                df.loc[df[col] > max_val, col] = max_val
                operations_log['outliers_corrected'][col] = f"Valeurs > {max_val} ramenées à {max_val}"

    # 3. Nettoyage des colonnes catégorielles
    # hotel - harmonisation des variantes
    if 'hotel' in df.columns:
        df['hotel'] = df['hotel'].str.strip()
        df['hotel'] = df['hotel'].replace({'Resoort Hotel': 'Resort Hotel', 'City Hotel  ': 'City Hotel'})
        operations_log['category_harmonized']['hotel'] = "Variantes harmonisées (Resoort -> Resort, espaces supprimés)"

    # adults - extraction numérique et harmonisation
    if 'adults' in df.columns:
        df['adults'] = df['adults'].apply(extract_numeric)
        df['adults'] = df['adults'].astype(float)
        operations_log['format_corrected']['adults'] = "Conversion en numérique"

    # children - imputation des manquants (10%) et conversion en numérique
    if 'children' in df.columns:
        mode_children = df['children'].mode()[0]
        df['children'] = df['children'].fillna(mode_children)
        df['children'] = df['children'].apply(extract_numeric)
        operations_log['missing_values_imputed']['children'] = f"Mode: {mode_children}"

    # meal - imputation des manquants (10%) et harmonisation
    if 'meal' in df.columns:
        mode_meal = df['meal'].mode()[0]
        df['meal'] = df['meal'].fillna(mode_meal)
        df['meal'] = df['meal'].str.strip()
        df['meal'] = df['meal'].replace({'HB ': 'HB', 'SC ': 'SC'})
        operations_log['missing_values_imputed']['meal'] = f"Mode: {mode_meal}"
        operations_log['category_harmonized']['meal'] = "Espaces supprimés"

    # country - imputation des manquants (10.37%)
    if 'country' in df.columns:
        mode_country = df['country'].mode()[0]
        df['country'] = df['country'].fillna(mode_country)
        operations_log['missing_values_imputed']['country'] = f"Mode: {mode_country}"

    # market_segment - imputation des manquants (10%)
    if 'market_segment' in df.columns:
        mode_market = df['market_segment'].mode()[0]
        df['market_segment'] = df['market_segment'].fillna(mode_market)
        df['market_segment'] = df['market_segment'].str.strip()
        operations_log['missing_values_imputed']['market_segment'] = f"Mode: {mode_market}"

    # agent - imputation des manquants (22.33%) et conversion en numérique
    if 'agent' in df.columns:
        mode_agent = df['agent'].mode()[0]
        df['agent'] = df['agent'].fillna(mode_agent)
        df['agent'] = df['agent'].apply(extract_numeric)
        operations_log['missing_values_imputed']['agent'] = f"Mode: {mode_agent}"

    # company - imputation des manquants (94.31%) -> trop élevé, on laisse NaN
    if 'company' in df.columns:
        operations_log['missing_values_imputed']['company'] = "Taux de manquants trop élevé (94.31%), non imputé"

    # deposit_type - harmonisation des variantes
    if 'deposit_type' in df.columns:
        df['deposit_type'] = df['deposit_type'].str.strip()
        df['deposit_type'] = df['deposit_type'].replace({
            'NoD eposit': 'No Deposit',
            'Non Refund  ': 'Non Refund',
            'No Deposit  ': 'No Deposit'
        })
        operations_log['category_harmonized']['deposit_type'] = "Variantes harmonisées"

    # customer_type - harmonisation
    if 'customer_type' in df.columns:
        df['customer_type'] = df['customer_type'].str.strip()
        operations_log['category_harmonized']['customer_type'] = "Espaces supprimés"

    # 4. Correction des formats de date
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
                    # Conserve la valeur originale si échec
                    return date_str

        df['reservation_status_date'] = df['reservation_status_date'].apply(parse_date)
        operations_log['format_corrected']['reservation_status_date'] = "Conversion en datetime"

    # 5. Imputation des valeurs manquantes restantes pour les colonnes numériques
    numeric_cols_to_impute = ['lead_time', 'stays_in_week_nights', 'adr']
    for col in numeric_cols_to_impute:
        if col in df.columns and df[col].isna().any():
            if col in ['lead_time', 'stays_in_week_nights']:
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
                operations_log['missing_values_imputed'][col] = f"Médiane: {median_val}"
            else:
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
                operations_log['missing_values_imputed'][col] = f"Médiane: {median_val}"

    # 6. Vérification finale des types
    if 'arrival_date_year' in df.columns:
        df['arrival_date_year'] = df['arrival_date_year'].astype(int)
    if 'arrival_date_week_number' in df.columns:
        df['arrival_date_week_number'] = df['arrival_date_week_number'].astype(int)
    if 'arrival_date_day_of_month' in df.columns:
        df['arrival_date_day_of_month'] = df['arrival_date_day_of_month'].astype(int)
    if 'stays_in_weekend_nights' in df.columns:
        df['stays_in_weekend_nights'] = df['stays_in_weekend_nights'].astype(int)
    if 'babies' in df.columns:
        df['babies'] = df['babies'].astype(int)
    if 'is_repeated_guest' in df.columns:
        df['is_repeated_guest'] = df['is_repeated_guest'].astype(int)
    if 'previous_cancellations' in df.columns:
        df['previous_cancellations'] = df['previous_cancellations'].astype(int)
    if 'previous_bookings_not_canceled' in df.columns:
        df['previous_bookings_not_canceled'] = df['previous_bookings_not_canceled'].astype(int)
    if 'booking_changes' in df.columns:
        df['booking_changes'] = df['booking_changes'].astype(int)
    if 'required_car_parking_spaces' in df.columns:
        df['required_car_parking_spaces'] = df['required_car_parking_spaces'].astype(int)
    if 'total_of_special_requests' in df.columns:
        df['total_of_special_requests'] = df['total_of_special_requests'].astype(int)

    # Sauvegarde du fichier nettoyé
    df.to_csv(OUTPUT_PATH, index=False)

    # Log des opérations
    print("=== Résumé des opérations de nettoyage ===")
    print(f"Lignes initiales: {initial_rows}")
    print(f"Lignes après suppression des doublons: {len(df)} (suppression de {operations_log['duplicates_removed']} doublons)")

    print("\nValeurs manquantes imputées:")
    for col, method in operations_log['missing_values_imputed'].items():
        print(f"- {col}: {method}")

    print("\nValeurs aberrantes corrigées:")
    for col, method in operations_log['outliers_corrected'].items():
        print(f"- {col}: {method}")

    print("\nCatégories harmonisées:")
    for col, method in operations_log['category_harmonized'].items():
        print(f"- {col}: {method}")

    print("\nFormats corrigés:")
    for col, method in operations_log['format_corrected'].items():
        print(f"- {col}: {method}")

if __name__ == "__main__":
    clean_hotel_bookings()