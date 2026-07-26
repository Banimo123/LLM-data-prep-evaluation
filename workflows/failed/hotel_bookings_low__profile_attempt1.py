import pandas as pd
import numpy as np
import re
from datetime import datetime

# Configuration des chemins
INPUT_PATH = r"benchmark\datasets\hotel_bookings\noisy_low.csv"
OUTPUT_PATH = r"results\cleaned_datasets\hotel_bookings\noisy_low__profile.csv"

# Fonction pour extraire les valeurs numériques
def extract_numeric(value):
    if pd.isna(value):
        return np.nan
    s = str(value).replace(",", ".")
    s = s.replace("O", "0").replace("o", "0")
    match = re.search(r"-?\d+\.?\d*", s)
    return float(match.group()) if match else np.nan

# Chargement des données
df = pd.read_csv(INPUT_PATH)

# Initialisation du log
log = {
    "missing_values_imputed": {},
    "outliers_corrected": {},
    "duplicates_removed": 0,
    "category_harmonized": {},
    "format_corrected": {}
}

# 1. Suppression des doublons (en conservant le premier)
initial_rows = len(df)
df.drop_duplicates(subset=df.columns.difference(['row_id']), keep='first', inplace=True)
log["duplicates_removed"] = initial_rows - len(df)

# 2. Traitement des colonnes numériques avec valeurs parasites
numeric_cols_with_text = ['lead_time', 'stays_in_week_nights', 'adr', 'days_in_waiting_list']
for col in numeric_cols_with_text:
    if col in df.columns:
        df[col] = df[col].apply(extract_numeric)
        log["format_corrected"][col] = "Extraction des valeurs numériques"

# 3. Correction des valeurs aberrantes dans les colonnes numériques
# stays_in_week_nights: max observé 999 -> valeur aberrante (moyenne 8.45)
if 'stays_in_week_nights' in df.columns:
    median_stays = df['stays_in_week_nights'].median()
    df.loc[df['stays_in_week_nights'] > 30, 'stays_in_week_nights'] = median_stays
    log["outliers_corrected"]['stays_in_week_nights'] = f"Valeurs >30 remplacées par la médiane ({median_stays})"

# adr: min -494.03 -> valeur aberrante (prix ne peut pas être négatif)
if 'adr' in df.columns:
    median_adr = df['adr'].median()
    df.loc[df['adr'] < 0, 'adr'] = median_adr
    log["outliers_corrected"]['adr'] = f"Valeurs négatives remplacées par la médiane ({median_adr})"

# days_in_waiting_list: max 8993 -> valeur aberrante (moyenne 57.77)
if 'days_in_waiting_list' in df.columns:
    median_wait = df['days_in_waiting_list'].median()
    df.loc[df['days_in_waiting_list'] > 365, 'days_in_waiting_list'] = median_wait
    log["outliers_corrected"]['days_in_waiting_list'] = f"Valeurs >365 remplacées par la médiane ({median_wait})"

# 4. Traitement des valeurs manquantes
# children: 5% manquants -> imputation par mode (0.0)
if 'children' in df.columns:
    mode_children = df['children'].mode()[0]
    df['children'].fillna(mode_children, inplace=True)
    log["missing_values_imputed"]['children'] = f"Mode: {mode_children}"

# meal: 5% manquants -> imputation par mode (BB)
if 'meal' in df.columns:
    mode_meal = df['meal'].mode()[0]
    df['meal'].fillna(mode_meal, inplace=True)
    log["missing_values_imputed"]['meal'] = f"Mode: {mode_meal}"

# country: 5.39% manquants -> imputation par mode (PRT)
if 'country' in df.columns:
    mode_country = df['country'].mode()[0]
    df['country'].fillna(mode_country, inplace=True)
    log["missing_values_imputed"]['country'] = f"Mode: {mode_country}"

# market_segment: 5% manquants -> imputation par mode (Online TA)
if 'market_segment' in df.columns:
    mode_market = df['market_segment'].mode()[0]
    df['market_segment'].fillna(mode_market, inplace=True)
    log["missing_values_imputed"]['market_segment'] = f"Mode: {mode_market}"

# agent: 17.99% manquants -> imputation par mode (9.0)
if 'agent' in df.columns:
    mode_agent = df['agent'].mode()[0]
    df['agent'].fillna(mode_agent, inplace=True)
    log["missing_values_imputed"]['agent'] = f"Mode: {mode_agent}"

# company: 94.31% manquants -> trop élevé, on conserve les NaN
if 'company' in df.columns:
    log["missing_values_imputed"]['company'] = "Trop de valeurs manquantes (94.31%), non imputé"

# 5. Harmonisation des catégories
# market_segment: correction de "unknown" vers le mode (Online TA)
if 'market_segment' in df.columns:
    mode_market = df['market_segment'].mode()[0]
    df['market_segment'] = df['market_segment'].replace('unknown', mode_market)
    log["category_harmonized"]['market_segment'] = f"'unknown' remplacé par {mode_market}"

# meal: correction des variantes avec espaces
if 'meal' in df.columns:
    df['meal'] = df['meal'].str.strip()
    log["category_harmonized"]['meal'] = "Espaces superflus supprimés"

# deposit_type: correction des variantes avec espaces
if 'deposit_type' in df.columns:
    df['deposit_type'] = df['deposit_type'].str.strip()
    log["category_harmonized"]['deposit_type'] = "Espaces superflus supprimés"

# 6. Correction du format de reservation_status_date
if 'reservation_status_date' in df.columns:
    def parse_date(date_str):
        if pd.isna(date_str):
            return date_str
        try:
            # Essaye le format YYYY-MM-DD
            return pd.to_datetime(date_str, format='%Y-%m-%d', errors='raise')
        except:
            try:
                # Essaye le format "Month DD, YYYY" (ex: July 01, 2015)
                return pd.to_datetime(date_str, format='%B %d, %Y', errors='raise')
            except:
                return date_str  # Conserve la valeur originale si échec

    df['reservation_status_date'] = df['reservation_status_date'].apply(parse_date)
    log["format_corrected"]['reservation_status_date'] = "Conversion en datetime (formats multiples)"

# 7. Conversion des colonnes catégorielles textuelles en type 'category' pour optimisation
categorical_cols = ['hotel', 'arrival_date_month', 'meal', 'country', 'market_segment',
                    'distribution_channel', 'reserved_room_type', 'assigned_room_type',
                    'deposit_type', 'customer_type', 'reservation_status']
for col in categorical_cols:
    if col in df.columns:
        df[col] = df[col].astype('category')

# 8. Conversion des colonnes numériques en types appropriés
numeric_cols = ['is_canceled', 'arrival_date_year', 'arrival_date_week_number',
                'arrival_date_day_of_month', 'stays_in_weekend_nights',
                'stays_in_week_nights', 'adults', 'children', 'babies', 'is_repeated_guest',
                'previous_cancellations', 'previous_bookings_not_canceled',
                'booking_changes', 'days_in_waiting_list', 'adr',
                'required_car_parking_spaces', 'total_of_special_requests']
for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='ignore')

# 9. Vérification finale des types
df['row_id'] = df['row_id'].astype(int)

# Sauvegarde du dataset nettoyé
df.to_csv(OUTPUT_PATH, index=False)

# Affichage du log
print("=== Nettoyage terminé ===")
print(f"Lignes initiales: {initial_rows}, lignes finales: {len(df)}")
print(f"Doublons supprimés: {log['duplicates_removed']}")
print("\nValeurs manquantes imputées:")
for col, method in log["missing_values_imputed"].items():
    print(f"  - {col}: {method}")
print("\nValeurs aberrantes corrigées:")
for col, method in log["outliers_corrected"].items():
    print(f"  - {col}: {method}")
print("\nCatégories harmonisées:")
for col, method in log["category_harmonized"].items():
    print(f"  - {col}: {method}")
print("\nFormats corrigés:")
for col, method in log["format_corrected"].items():
    print(f"  - {col}: {method}")