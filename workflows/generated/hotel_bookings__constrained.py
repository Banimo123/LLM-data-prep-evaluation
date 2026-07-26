import pandas as pd
import numpy as np
import re
from datetime import datetime

INPUT_PATH = "benchmark/datasets/hotel_bookings/noisy_low.csv"
OUTPUT_PATH = "results/cleaned_datasets/hotel_bookings/noisy_low__constrained.csv"

# Chargement du dataset
df = pd.read_csv(INPUT_PATH)

# Initialisation du log
log = {
    "missing_values_filled": 0,
    "missing_values_dropped": 0,
    "duplicates_dropped": 0,
    "numeric_conversions": 0,
    "date_conversions": 0,
    "string_cleanings": 0,
    "outliers_removed": 0,
    "rows_initial": len(df),
    "rows_final": None,
    "cols_initial": len(df.columns),
    "cols_final": None
}

# 1. Nettoyage des espaces et casse pour les colonnes textuelles
text_cols = [
    "hotel", "arrival_date_month", "meal", "country", "market_segment",
    "distribution_channel", "deposit_type", "customer_type", "reservation_status"
]
for col in text_cols:
    if col in df.columns:
        df[col] = df[col].astype(str).str.strip()
        df[col] = df[col].replace("", np.nan)
        log["string_cleanings"] += 1

# 2. Conversion des colonnes numériques avec gestion des erreurs
numeric_cols = [
    "lead_time", "adults", "children", "agent", "adr"
]
for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        log["numeric_conversions"] += 1

# 3. Correction des valeurs aberrantes dans les colonnes numériques
if "babies" in df.columns:
    df = df[df["babies"] <= 10]  # Seuil raisonnable pour des bébés
    log["outliers_removed"] += len(df[df["babies"] > 10])

if "lead_time" in df.columns:
    df = df[df["lead_time"] >= 0]
    log["outliers_removed"] += len(df[df["lead_time"] < 0])

if "adr" in df.columns:
    df = df[df["adr"] >= 0]
    log["outliers_removed"] += len(df[df["adr"] < 0])

if "adults" in df.columns:
    df = df[df["adults"] > 0]
    log["outliers_removed"] += len(df[df["adults"] <= 0])

# 4. Harmonisation des catégories connues
category_mappings = {
    "deposit_type": {
        "no deposit": "No Deposit",
        "no depo": "No Deposit",
        "no  deposit": "No Deposit"
    },
    "market_segment": {
        "unknown": "Unknown"
    },
    "customer_type": {
        "transient": "Transient",
        "transient-party": "Transient-Party"
    },
    "reservation_status": {
        "check-out": "Check-Out",
        "canceled": "Canceled",
        "no-show": "No-Show"
    }
}

for col, mapping in category_mappings.items():
    if col in df.columns:
        df[col] = df[col].replace(mapping)
        log["string_cleanings"] += 1

# 5. Conversion des dates
if "reservation_status_date" in df.columns:
    def parse_date(date_str):
        if pd.isna(date_str):
            return date_str
        date_str = str(date_str).strip()
        try:
            # Essaye d'abord le format "Month Day, Year"
            return pd.to_datetime(date_str, format="%B %d, %Y", errors='raise')
        except:
            try:
                # Essaye le format "YYYY-MM-DD"
                return pd.to_datetime(date_str, format="%Y-%m-%d", errors='raise')
            except:
                try:
                    # Essaye le format "Month Day Year" (ex: July 01 2015)
                    return pd.to_datetime(date_str, format="%B %d %Y", errors='raise')
                except:
                    return date_str  # Conserve la valeur originale si aucun format ne marche

    df["reservation_status_date"] = df["reservation_status_date"].apply(parse_date)
    log["date_conversions"] += 1

# 6. Gestion des valeurs manquantes
# Remplacement des NaN dans les colonnes catégorielles par "Unknown"
categorical_cols = [
    "country", "market_segment", "distribution_channel",
    "meal", "reserved_room_type", "assigned_room_type"
]
for col in categorical_cols:
    if col in df.columns:
        df[col] = df[col].fillna("Unknown")
        log["missing_values_filled"] += df[col].isna().sum()

# Suppression des lignes où row_id est manquant (ne devrait pas arriver)
if "row_id" in df.columns:
    initial_count = len(df)
    df = df.dropna(subset=["row_id"])
    log["missing_values_dropped"] += initial_count - len(df)

# 7. Suppression des doublons (en ignorant row_id)
cols_for_duplicates = [col for col in df.columns if col != "row_id"]
initial_count = len(df)
df = df.drop_duplicates(subset=cols_for_duplicates, keep="first")
log["duplicates_dropped"] += initial_count - len(df)

# 8. Conversion des colonnes qui devraient être entières
int_cols = [
    "is_canceled", "arrival_date_year", "arrival_date_week_number",
    "arrival_date_day_of_month", "stays_in_weekend_nights", "stays_in_week_nights",
    "is_repeated_guest", "previous_cancellations", "previous_bookings_not_canceled",
    "booking_changes", "days_in_waiting_list", "required_car_parking_spaces",
    "total_of_special_requests", "babies", "company"
]
for col in int_cols:
    if col in df.columns and df[col].dtype == "float64":
        df[col] = df[col].fillna(-1).astype(int).replace(-1, np.nan)
        log["numeric_conversions"] += 1

# Mise à jour des compteurs finaux
log["rows_final"] = len(df)
log["cols_final"] = len(df.columns)

# Sauvegarde du dataset nettoyé
df.to_csv(OUTPUT_PATH, index=False)

# Affichage du log
print("=== Nettoyage terminé ===")
print(f"Lignes initiales: {log['rows_initial']}")
print(f"Lignes finales: {log['rows_final']}")
print(f"Colonnes initiales: {log['cols_initial']}")
print(f"Colonnes finales: {log['cols_final']}")
print(f"Valeurs manquantes remplies: {log['missing_values_filled']}")
print(f"Valeurs manquantes supprimées: {log['missing_values_dropped']}")
print(f"Doublons supprimés: {log['duplicates_dropped']}")
print(f"Conversions numériques: {log['numeric_conversions']}")
print(f"Conversions de dates: {log['date_conversions']}")
print(f"Nettoyages de chaînes: {log['string_cleanings']}")
print(f"Valeurs aberrantes supprimées: {log['outliers_removed']}")