import pandas as pd
import numpy as np
import re
from datetime import datetime

# Chemins d'entrée/sortie
INPUT_PATH = r"benchmark\datasets\hotel_bookings\noisy_high.csv"
OUTPUT_PATH = r"results\cleaned_datasets\hotel_bookings\noisy_high__profile.csv"

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
    "formats_corrected": {},
    "duplicates_removed": 0,
    "rows_before": len(df),
    "rows_after": None
}

# 1. Traitement des valeurs manquantes
# Colonnes avec <= 20% de valeurs manquantes -> imputation
for col in df.columns:
    if col == "row_id":
        continue
    missing_pct = df[col].isna().mean() * 100
    if missing_pct > 0:
        if missing_pct <= 20:
            if df[col].dtype in ['int64', 'float64']:
                # Imputation par médiane pour les colonnes numériques
                median_val = df[col].median()
                df[col].fillna(median_val, inplace=True)
                log["missing_values_imputed"][col] = f"median ({median_val})"
            else:
                # Imputation par mode pour les colonnes catégorielles
                mode_val = df[col].mode()[0]
                df[col].fillna(mode_val, inplace=True)
                log["missing_values_imputed"][col] = f"mode ({mode_val})"
        else:
            # Colonnes avec >20% de valeurs manquantes -> on conserve les NaN
            pass

# 2. Correction des valeurs aberrantes pour les colonnes numériques
numeric_cols = [
    "is_canceled", "arrival_date_year", "arrival_date_week_number",
    "arrival_date_day_of_month", "stays_in_weekend_nights",
    "stays_in_week_nights", "babies", "is_repeated_guest",
    "previous_cancellations", "previous_bookings_not_canceled",
    "booking_changes", "days_in_waiting_list", "adr",
    "required_car_parking_spaces", "total_of_special_requests", "company"
]

for col in numeric_cols:
    if col not in df.columns:
        continue

    # Extraction des valeurs numériques si nécessaire
    if df[col].dtype == 'object':
        df[col] = df[col].apply(extract_numeric)
        log["formats_corrected"][col] = "extracted numeric values"

    # Correction des valeurs aberrantes basées sur min/max du profil
    if col == "stays_in_week_nights":
        # Valeur max aberrante (999) -> on la limite à 30 (valeur plausible)
        df[col] = df[col].clip(upper=30)
        log["outliers_corrected"][col] = "clipped at 30"
    elif col == "babies":
        # Valeur max aberrante (49) -> on la limite à 10
        df[col] = df[col].clip(upper=10)
        log["outliers_corrected"][col] = "clipped at 10"
    elif col == "adr":
        # Valeurs négatives et >5000 aberrantes
        df[col] = df[col].clip(lower=0, upper=5000)
        log["outliers_corrected"][col] = "clipped [0, 5000]"
    elif col == "days_in_waiting_list":
        # Valeur max aberrante (8999) -> on la limite à 365
        df[col] = df[col].clip(upper=365)
        log["outliers_corrected"][col] = "clipped at 365"

# 3. Harmonisation des catégories
# adults: correction de "2O" en "2"
if "adults" in df.columns:
    df["adults"] = df["adults"].replace("2O", "2")
    log["categories_harmonized"]["adults"] = "2O -> 2"

# meal: harmonisation des valeurs
meal_mapping = {
    "BB": "BB",
    "HB": "HB",
    "SC": "SC",
    "Undefined": "SC",
    "Full Board": "FB",
    "Half Board": "HB"
}
if "meal" in df.columns:
    df["meal"] = df["meal"].replace(meal_mapping)
    log["categories_harmonized"]["meal"] = str(meal_mapping)

# country: harmonisation des valeurs vides
if "country" in df.columns:
    df["country"] = df["country"].replace(["", " "], np.nan)

# market_segment: harmonisation
market_segment_mapping = {
    "Online TA": "Online TA",
    "Offline TA/TO": "Offline TA/TO",
    "Groups": "Groups",
    "Direct": "Direct",
    "Corporate": "Corporate",
    "Complementary": "Complementary",
    "Aviation": "Aviation"
}
if "market_segment" in df.columns:
    df["market_segment"] = df["market_segment"].replace(market_segment_mapping)
    log["categories_harmonized"]["market_segment"] = str(market_segment_mapping)

# distribution_channel: harmonisation
dist_channel_mapping = {
    "TA/TO": "TA/TO",
    "Direct": "Direct",
    "Corporate": "Corporate",
    "GDS": "GDS",
    "Undefined": "TA/TO"
}
if "distribution_channel" in df.columns:
    df["distribution_channel"] = df["distribution_channel"].replace(dist_channel_mapping)
    log["categories_harmonized"]["distribution_channel"] = str(dist_channel_mapping)

# customer_type: correction de "Trasnient" en "Transient"
if "customer_type" in df.columns:
    df["customer_type"] = df["customer_type"].replace("Trasnient", "Transient")
    log["categories_harmonized"]["customer_type"] = "Trasnient -> Transient"

# deposit_type: harmonisation
deposit_mapping = {
    "No Deposit": "No Deposit",
    "Non Refund": "Non Refund",
    "Refundable": "Refundable",
    "No Depossit": "No Deposit"
}
if "deposit_type" in df.columns:
    df["deposit_type"] = df["deposit_type"].replace(deposit_mapping)
    log["categories_harmonized"]["deposit_type"] = str(deposit_mapping)

# 4. Correction des formats
# reservation_status_date: harmonisation des formats de date
def parse_date(date_str):
    if pd.isna(date_str):
        return date_str
    date_str = str(date_str).strip()
    formats = [
        "%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%B %d, %Y",
        "%b %d, %Y", "%Y/%m/%d", "%d-%m-%Y", "%Y%m%d"
    ]
    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return date_str

if "reservation_status_date" in df.columns:
    df["reservation_status_date"] = df["reservation_status_date"].apply(parse_date)
    log["formats_corrected"]["reservation_status_date"] = "standardized to YYYY-MM-DD"

# 5. Suppression des doublons (en conservant le row_id)
df.drop_duplicates(subset=df.columns.difference(["row_id"]), keep="first", inplace=True)
log["duplicates_removed"] = log["rows_before"] - len(df)
log["rows_after"] = len(df)

# 6. Conversion des types
# adults: conversion en int
if "adults" in df.columns:
    df["adults"] = pd.to_numeric(df["adults"], errors="coerce").fillna(2).astype(int)

# children: conversion en float (car contient "unknown")
if "children" in df.columns:
    df["children"] = df["children"].replace("unknown", np.nan)
    df["children"] = pd.to_numeric(df["children"], errors="coerce")

# agent: conversion en float (car contient des chaînes)
if "agent" in df.columns:
    df["agent"] = df["agent"].apply(extract_numeric)

# Sauvegarde du dataset nettoyé
df.to_csv(OUTPUT_PATH, index=False)

# Affichage du log
print("=== Nettoyage terminé ===")
print(f"Lignes avant: {log['rows_before']}, après: {log['rows_after']}")
print(f"Doublons supprimés: {log['duplicates_removed']}")
print("\nValeurs manquantes imputées:")
for col, method in log["missing_values_imputed"].items():
    print(f"- {col}: {method}")
print("\nValeurs aberrantes corrigées:")
for col, method in log["outliers_corrected"].items():
    print(f"- {col}: {method}")
print("\nCatégories harmonisées:")
for col, mapping in log["categories_harmonized"].items():
    print(f"- {col}: {mapping}")
print("\nFormats corrigés:")
for col, method in log["formats_corrected"].items():
    print(f"- {col}: {method}")