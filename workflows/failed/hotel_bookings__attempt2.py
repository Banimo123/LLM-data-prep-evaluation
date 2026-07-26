import pandas as pd
import numpy as np
import re
from datetime import datetime

INPUT_PATH = "benchmark/datasets/hotel_bookings/noisy_low.csv"
OUTPUT_PATH = "results/cleaned_datasets/hotel_bookings/noisy_low__profile.csv"

# Chargement du dataset
df = pd.read_csv(INPUT_PATH)

# Initialisation du log
log = {
    "missing_values_imputed": 0,
    "outliers_corrected": 0,
    "categories_harmonized": 0,
    "duplicates_removed": 0,
    "rows_before": len(df),
    "rows_after": None,
    "columns_processed": []
}

# 1. Suppression des doublons (en conservant le premier)
df.drop_duplicates(subset=[col for col in df.columns if col != "row_id"], keep="first", inplace=True)
log["duplicates_removed"] = log["rows_before"] - len(df)
log["rows_before"] = len(df)

# 2. Traitement de la colonne 'lead_time' (correction du 'O' en '0')
if "lead_time" in df.columns:
    def clean_lead_time(val):
        if pd.isna(val):
            return val
        val_str = str(val)
        if "O" in val_str:
            val_str = val_str.replace("O", "0")
        try:
            return int(val_str)
        except:
            return val
    df["lead_time"] = df["lead_time"].apply(clean_lead_time)
    log["outliers_corrected"] += df["lead_time"].apply(lambda x: isinstance(x, str) and "O" in str(x)).sum()

# 3. Conversion des colonnes numériques avant traitement des valeurs aberrantes
numeric_cols_conversion = {
    "stays_in_week_nights": "int",
    "adults": "int",
    "children": "float",
    "babies": "int",
    "adr": "float",
    "days_in_waiting_list": "int"
}

for col, dtype in numeric_cols_conversion.items():
    if col in df.columns:
        df[col] = pd.to_numeric(df[col].apply(extract_numeric), errors='ignore')
        if dtype == "int":
            df[col] = df[col].astype("Int64")
        else:
            df[col] = df[col].astype("float64")

# 4. Traitement des colonnes numériques avec valeurs aberrantes
numeric_cols = {
    "stays_in_week_nights": {"min": 0, "max": 30},
    "adults": {"min": 0, "max": 20},
    "children": {"min": 0, "max": 10},
    "babies": {"min": 0, "max": 5},
    "adr": {"min": 0, "max": 500},
    "days_in_waiting_list": {"min": 0, "max": 365}
}

for col, bounds in numeric_cols.items():
    if col in df.columns:
        # Vérifier que la colonne est bien numérique
        if pd.api.types.is_numeric_dtype(df[col]):
            mask = (df[col] < bounds["min"]) | (df[col] > bounds["max"])
            outliers = df[mask]
            if not outliers.empty:
                log["outliers_corrected"] += len(outliers)
                median_val = df.loc[~mask, col].median()
                df.loc[mask, col] = median_val

# 5. Traitement des colonnes catégorielles avec variantes
categorical_harmonization = {
    "hotel": {
        "  City Hotel  ": "City Hotel",
        "City  Hotel": "City Hotel",
        "Resort  Hotel": "Resort Hotel"
    },
    "meal": {
        "Undefined": "SC",
        "SC ": "SC",
        " BB": "BB"
    },
    "market_segment": {
        "Online": "Online TA",
        "Offline": "Offline TA/TO",
        "Groups ": "Groups",
        "Corporate ": "Corporate"
    },
    "distribution_channel": {
        "TA /TO": "TA/TO",
        "Direct ": "Direct"
    },
    "deposit_type": {
        "No  Deposit": "No Deposit",
        "Non Refund ": "Non Refund"
    },
    "customer_type": {
        "Transient ": "Transient",
        "Transient-Party ": "Transient-Party"
    }
}

for col, mapping in categorical_harmonization.items():
    if col in df.columns:
        df[col] = df[col].astype(str).str.strip()
        for wrong, correct in mapping.items():
            mask = df[col] == wrong
            if mask.any():
                log["categories_harmonized"] += mask.sum()
                df.loc[mask, col] = correct

# 6. Fonction pour extraire les valeurs numériques
def extract_numeric(val):
    if pd.isna(val):
        return val
    val_str = str(val)
    numeric_part = re.sub(r'[^0-9.]', '', val_str)
    if numeric_part:
        try:
            return float(numeric_part)
        except:
            return val
    return val

# 7. Traitement des colonnes avec valeurs manquantes
missing_cols = {
    "children": "mode",
    "meal": "mode",
    "country": "mode",
    "market_segment": "mode",
    "agent": "drop",
    "company": "drop"
}

for col, strategy in missing_cols.items():
    if col in df.columns:
        if strategy == "mode":
            mode_val = df[col].mode()[0]
            missing_count = df[col].isna().sum()
            if missing_count > 0:
                df[col].fillna(mode_val, inplace=True)
                log["missing_values_imputed"] += missing_count
                log["columns_processed"].append(col)
        elif strategy == "drop":
            log["columns_processed"].append(col + " (NaN conservés)")

# 8. Correction des valeurs numériques corrompues par du texte
numeric_text_cols = ["adults", "children", "agent"]
for col in numeric_text_cols:
    if col in df.columns:
        df[col] = df[col].apply(extract_numeric)
        df[col] = pd.to_numeric(df[col], errors="ignore")

# 9. Traitement de la colonne 'reservation_status_date'
if "reservation_status_date" in df.columns:
    def parse_date(date_str):
        if pd.isna(date_str):
            return date_str
        date_str = str(date_str).strip()
        for fmt in ("%Y-%m-%d", "%B %d, %Y", "%d/%m/%Y"):
            try:
                return datetime.strptime(date_str, fmt).strftime("%Y-%m-%d")
            except:
                continue
        return date_str

    df["reservation_status_date"] = df["reservation_status_date"].apply(parse_date)

# 10. Vérification finale des types
type_conversion = {
    "is_canceled": "int",
    "arrival_date_year": "int",
    "arrival_date_week_number": "int",
    "arrival_date_day_of_month": "int",
    "stays_in_weekend_nights": "int",
    "stays_in_week_nights": "int",
    "is_repeated_guest": "int",
    "previous_cancellations": "int",
    "previous_bookings_not_canceled": "int",
    "booking_changes": "int",
    "days_in_waiting_list": "int",
    "required_car_parking_spaces": "int",
    "total_of_special_requests": "int"
}

for col, dtype in type_conversion.items():
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="ignore")
        if dtype == "int":
            df[col] = df[col].astype("Int64")

# Sauvegarde du dataset nettoyé
df.to_csv(OUTPUT_PATH, index=False)

# Mise à jour du log
log["rows_after"] = len(df)
log["columns_processed"] = list(set(log["columns_processed"]))

# Affichage du résumé
print("=== Nettoyage terminé ===")
print(f"Lignes avant nettoyage: {log['rows_before']}")
print(f"Lignes après nettoyage: {log['rows_after']}")
print(f"Doublons supprimés: {log['duplicates_removed']}")
print(f"Valeurs manquantes imputées: {log['missing_values_imputed']}")
print(f"Valeurs aberrantes corrigées: {log['outliers_corrected']}")
print(f"Catégories harmonisées: {log['categories_harmonized']}")
print(f"Colonnes traitées: {', '.join(log['columns_processed'])}")