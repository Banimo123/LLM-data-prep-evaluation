import pandas as pd
import numpy as np
import re
from datetime import datetime

# Chemins d'entrée/sortie
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

# Chargement du dataset
df = pd.read_csv(INPUT_PATH)

# Initialisation du log
log = {
    "missing_values_imputed": {},
    "outliers_corrected": {},
    "categories_harmonized": {},
    "duplicates_removed": 0,
    "rows_original": len(df),
    "rows_final": None
}

# 1. Suppression des doublons (en conservant le premier)
df = df.drop_duplicates(subset=df.columns.difference(['row_id']), keep='first')
log["duplicates_removed"] = log["rows_original"] - len(df)

# 2. Traitement des valeurs manquantes
# children (5% manquants) - colonne catégorielle avec valeurs numériques
if 'children' in df.columns:
    mode_children = df['children'].mode()[0]
    df['children'] = df['children'].fillna(mode_children)
    log["missing_values_imputed"]['children'] = df['children'].isna().sum()

# meal (5% manquants) - colonne catégorielle
if 'meal' in df.columns:
    mode_meal = df['meal'].mode()[0]
    df['meal'] = df['meal'].fillna(mode_meal)
    log["missing_values_imputed"]['meal'] = df['meal'].isna().sum()

# country (5.39% manquants) - colonne catégorielle
if 'country' in df.columns:
    mode_country = df['country'].mode()[0]
    df['country'] = df['country'].fillna(mode_country)
    log["missing_values_imputed"]['country'] = df['country'].isna().sum()

# market_segment (5% manquants) - colonne catégorielle
if 'market_segment' in df.columns:
    mode_market_segment = df['market_segment'].mode()[0]
    df['market_segment'] = df['market_segment'].fillna(mode_market_segment)
    log["missing_values_imputed"]['market_segment'] = df['market_segment'].isna().sum()

# agent (17.99% manquants) - colonne catégorielle avec valeurs numériques
if 'agent' in df.columns:
    mode_agent = df['agent'].mode()[0]
    df['agent'] = df['agent'].fillna(mode_agent)
    log["missing_values_imputed"]['agent'] = df['agent'].isna().sum()

# company (94.31% manquants) - trop de manquants, on conserve les valeurs existantes
if 'company' in df.columns:
    df['company'] = df['company'].apply(lambda x: np.nan if pd.isna(x) else x)

# 3. Correction des valeurs aberrantes numériques
# lead_time - contient des valeurs avec 'O' au lieu de '0'
if 'lead_time' in df.columns:
    df['lead_time'] = df['lead_time'].apply(extract_numeric)
    # Vérification des valeurs aberrantes (max observé = 2610)
    median_lead = df['lead_time'].median()
    df['lead_time'] = df['lead_time'].apply(lambda x: x if x <= 3000 else median_lead)

# stays_in_week_nights - max observé = 999.0 (aberrant)
if 'stays_in_week_nights' in df.columns:
    median_stays_week = df['stays_in_week_nights'].median()
    df['stays_in_week_nights'] = df['stays_in_week_nights'].apply(
        lambda x: x if x <= 30 else median_stays_week
    )

# adr - contient des valeurs négatives et très élevées
if 'adr' in df.columns:
    median_adr = df['adr'].median()
    df['adr'] = df['adr'].apply(
        lambda x: x if 0 <= x <= 1000 else median_adr
    )

# babies - max observé = 49.0 (aberrant)
if 'babies' in df.columns:
    mode_babies = df['babies'].mode()[0]
    df['babies'] = df['babies'].apply(
        lambda x: x if x <= 5 else mode_babies
    )

# days_in_waiting_list - max observé = 8993.0 (aberrant)
if 'days_in_waiting_list' in df.columns:
    median_waiting = df['days_in_waiting_list'].median()
    df['days_in_waiting_list'] = df['days_in_waiting_list'].apply(
        lambda x: x if x <= 365 else median_waiting
    )

# adults - correction des valeurs avec 'O' au lieu de '0'
if 'adults' in df.columns:
    df['adults'] = df['adults'].apply(extract_numeric)
    mode_adults = df['adults'].mode()[0]
    df['adults'] = df['adults'].apply(lambda x: x if x <= 10 else mode_adults)

# 4. Harmonisation des catégories
# hotel - correction des espaces superflus
if 'hotel' in df.columns:
    df['hotel'] = df['hotel'].str.strip()
    df['hotel'] = df['hotel'].replace('City Hotel  ', 'City Hotel')
    df['hotel'] = df['hotel'].replace('Reort Hotel', 'Resort Hotel')
    log["categories_harmonized"]['hotel'] = True

# meal - harmonisation des variantes
if 'meal' in df.columns:
    meal_mapping = {
        'BB': 'BB',
        'HB': 'HB',
        'SC': 'SC',
        'Undefined': 'SC',
        'Full Board': 'FB'
    }
    df['meal'] = df['meal'].apply(lambda x: meal_mapping.get(x, x))
    log["categories_harmonized"]['meal'] = True

# market_segment - correction de 'unknown'
if 'market_segment' in df.columns:
    mode_market = df['market_segment'].mode()[0]
    df['market_segment'] = df['market_segment'].replace('unknown', mode_market)
    log["categories_harmonized"]['market_segment'] = True

# distribution_channel - correction des espaces
if 'distribution_channel' in df.columns:
    df['distribution_channel'] = df['distribution_channel'].str.strip()
    log["categories_harmonized"]['distribution_channel'] = True

# deposit_type - correction des espaces
if 'deposit_type' in df.columns:
    df['deposit_type'] = df['deposit_type'].str.strip()
    df['deposit_type'] = df['deposit_type'].replace('No Deposit  ', 'No Deposit')
    df['deposit_type'] = df['deposit_type'].replace('NO DEPosit', 'No Deposit')
    log["categories_harmonized"]['deposit_type'] = True

# customer_type - harmonisation
if 'customer_type' in df.columns:
    df['customer_type'] = df['customer_type'].str.strip()
    df['customer_type'] = df['customer_type'].replace('TRANSIent', 'Transient')
    log["categories_harmonized"]['customer_type'] = True

# 5. Correction des formats de date
if 'reservation_status_date' in df.columns:
    def parse_date(date_str):
        if pd.isna(date_str):
            return date_str
        try:
            # Essai de parsing pour les formats courants
            for fmt in ('%Y-%m-%d', '%B %d, %Y', '%b %d, %Y', '%B %d %Y', '%b %d %Y'):
                try:
                    return datetime.strptime(str(date_str), fmt).strftime('%Y-%m-%d')
                except ValueError:
                    continue
            return date_str  # Retourne la valeur originale si aucun format ne correspond
        except:
            return date_str

    df['reservation_status_date'] = df['reservation_status_date'].apply(parse_date)

# 6. Conversion des colonnes numériques
numeric_cols = [
    'is_canceled', 'arrival_date_year', 'arrival_date_week_number',
    'arrival_date_day_of_month', 'stays_in_weekend_nights',
    'stays_in_week_nights', 'is_repeated_guest', 'previous_cancellations',
    'previous_bookings_not_canceled', 'booking_changes', 'days_in_waiting_list',
    'adr', 'required_car_parking_spaces', 'total_of_special_requests'
]

for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        if df[col].dtype == 'float64':
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna(df[col].mode()[0])

# 7. Vérification finale des types
categorical_cols = [
    'hotel', 'arrival_date_month', 'meal', 'country', 'market_segment',
    'distribution_channel', 'reserved_room_type', 'assigned_room_type',
    'deposit_type', 'customer_type', 'reservation_status'
]

for col in categorical_cols:
    if col in df.columns:
        df[col] = df[col].astype(str).str.strip()

# Sauvegarde du dataset nettoyé
df.to_csv(OUTPUT_PATH, index=False)

# Mise à jour du log
log["rows_final"] = len(df)

# Affichage du résumé
print("=== Nettoyage terminé ===")
print(f"Lignes originales: {log['rows_original']}")
print(f"Lignes après suppression des doublons: {log['rows_final']} (supprimées: {log['duplicates_removed']})")
print("\nValeurs manquantes imputées:")
for col, count in log["missing_values_imputed"].items():
    print(f"- {col}: {count} valeurs")
print("\nColonnes avec harmonisation catégorielle:")
for col in log["categories_harmonized"]:
    print(f"- {col}")