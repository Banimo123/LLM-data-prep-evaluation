import pandas as pd
import numpy as np
import re
from datetime import datetime

INPUT_PATH = "benchmark/datasets/hotel_bookings/noisy_low.csv"
OUTPUT_PATH = "results/cleaned_datasets/hotel_bookings/noisy_low__profile.csv"

# Chargement des données
df = pd.read_csv(INPUT_PATH)

# Initialisation du log
log = {
    "missing_values_handled": {},
    "outliers_corrected": {},
    "categories_harmonized": {},
    "formats_corrected": {},
    "duplicates_removed": 0,
    "rows_before": len(df),
    "rows_after": None
}

# 1. Traitement des valeurs manquantes
# children (5% manquants) - imputation par mode (0.0)
if "children" in df.columns:
    mode_children = df["children"].mode()[0]
    df["children"] = df["children"].fillna(mode_children)
    log["missing_values_handled"]["children"] = f"imputed with mode: {mode_children}"

# meal (5% manquants) - imputation par mode (BB)
if "meal" in df.columns:
    mode_meal = df["meal"].mode()[0]
    df["meal"] = df["meal"].fillna(mode_meal)
    log["missing_values_handled"]["meal"] = f"imputed with mode: {mode_meal}"

# country (5.39% manquants) - imputation par mode (PRT)
if "country" in df.columns:
    mode_country = df["country"].mode()[0]
    df["country"] = df["country"].fillna(mode_country)
    log["missing_values_handled"]["country"] = f"imputed with mode: {mode_country}"

# market_segment (5% manquants) - imputation par mode (Online TA)
if "market_segment" in df.columns:
    mode_market_segment = df["market_segment"].mode()[0]
    df["market_segment"] = df["market_segment"].fillna(mode_market_segment)
    log["missing_values_handled"]["market_segment"] = f"imputed with mode: {mode_market_segment}"

# agent (17.99% manquants) - imputation par mode (9.0)
if "agent" in df.columns:
    mode_agent = df["agent"].mode()[0]
    df["agent"] = df["agent"].fillna(mode_agent)
    log["missing_values_handled"]["agent"] = f"imputed with mode: {mode_agent}"

# company (94.31% manquants) - suppression de la colonne (trop de manquants)
if "company" in df.columns:
    df.drop("company", axis=1, inplace=True)
    log["missing_values_handled"]["company"] = "column dropped (94.31% missing)"

# 2. Correction des valeurs aberrantes
# lead_time - correction des valeurs textuelles (ex: "342O" -> 342)
if "lead_time" in df.columns:
    df["lead_time"] = df["lead_time"].apply(lambda x: re.sub(r'[^0-9]', '', str(x)) if isinstance(x, str) else x)
    df["lead_time"] = pd.to_numeric(df["lead_time"], errors="coerce")
    log["formats_corrected"]["lead_time"] = "corrected text values to numeric"

# stays_in_week_nights - correction des valeurs aberrantes (max 999 -> 99)
if "stays_in_week_nights" in df.columns:
    q99 = df["stays_in_week_nights"].quantile(0.99)
    q99 = int(round(q99))
    df["stays_in_week_nights"] = pd.to_numeric(df["stays_in_week_nights"], errors="coerce")
    df["stays_in_week_nights"] = df["stays_in_week_nights"].fillna(0)
    df["stays_in_week_nights"] = df["stays_in_week_nights"].astype(int)
    df.loc[df["stays_in_week_nights"] > q99, "stays_in_week_nights"] = q99
    log["outliers_corrected"]["stays_in_week_nights"] = f"capped at 99th percentile: {q99}"

# adr - correction des valeurs négatives (min -494.03 -> 0)
if "adr" in df.columns:
    df.loc[df["adr"] < 0, "adr"] = 0
    log["outliers_corrected"]["adr"] = "negative values set to 0"

# days_in_waiting_list - correction des valeurs aberrantes (max 8993 -> 99th percentile)
if "days_in_waiting_list" in df.columns:
    q99 = df["days_in_waiting_list"].quantile(0.99)
    q99 = int(round(q99))
    df["days_in_waiting_list"] = pd.to_numeric(df["days_in_waiting_list"], errors="coerce")
    df["days_in_waiting_list"] = df["days_in_waiting_list"].fillna(0)
    df["days_in_waiting_list"] = df["days_in_waiting_list"].astype(int)
    df.loc[df["days_in_waiting_list"] > q99, "days_in_waiting_list"] = q99
    log["outliers_corrected"]["days_in_waiting_list"] = f"capped at 99th percentile: {q99}"

# 3. Harmonisation des catégories
# hotel - harmonisation des variantes (ex: "City Hotel  " -> "City Hotel")
if "hotel" in df.columns:
    df["hotel"] = df["hotel"].str.strip()
    df["hotel"] = df["hotel"].replace({"City Hotel  ": "City Hotel", "Reort Hotel": "Resort Hotel"})
    log["categories_harmonized"]["hotel"] = "stripped and harmonized variants"

# deposit_type - harmonisation des variantes (ex: "No Deposit  " -> "No Deposit")
if "deposit_type" in df.columns:
    df["deposit_type"] = df["deposit_type"].str.strip()
    df["deposit_type"] = df["deposit_type"].replace({"No Deposit  ": "No Deposit", "NO DEPosit": "No Deposit"})
    log["categories_harmonized"]["deposit_type"] = "stripped and harmonized variants"

# customer_type - harmonisation des variantes
if "customer_type" in df.columns:
    df["customer_type"] = df["customer_type"].str.strip().str.title()
    df["customer_type"] = df["customer_type"].replace({"Transient": "Transient", "Transient-Party": "Transient-Party"})
    log["categories_harmonized"]["customer_type"] = "stripped and harmonized variants"

# adults - conversion en numérique (certaines valeurs sont textuelles)
if "adults" in df.columns:
    df["adults"] = df["adults"].apply(lambda x: re.sub(r'[^0-9]', '', str(x)) if isinstance(x, str) else x)
    df["adults"] = pd.to_numeric(df["adults"], errors="coerce")
    df["adults"] = df["adults"].fillna(0).astype(int)
    log["formats_corrected"]["adults"] = "converted to numeric and cleaned text values"

# children - conversion en numérique (certaines valeurs sont textuelles)
if "children" in df.columns:
    df["children"] = pd.to_numeric(df["children"], errors="coerce")
    df["children"] = df["children"].fillna(0).astype(int)
    log["formats_corrected"]["children"] = "converted to numeric"

# 4. Correction des formats
# reservation_status_date - conversion en format date standard
if "reservation_status_date" in df.columns:
    df["reservation_status_date"] = df["reservation_status_date"].apply(
        lambda x: datetime.strptime(x, "%B %d, %Y").strftime("%Y-%m-%d")
        if isinstance(x, str) and "," in x
        else x
    )
    df["reservation_status_date"] = pd.to_datetime(df["reservation_status_date"], errors="coerce")
    log["formats_corrected"]["reservation_status_date"] = "converted to YYYY-MM-DD format"

# 5. Suppression des doublons (en conservant row_id)
if "row_id" in df.columns:
    df.drop_duplicates(subset=df.columns.difference(["row_id"]), keep="first", inplace=True)
    log["duplicates_removed"] = log["rows_before"] - len(df)

# Mise à jour du nombre de lignes après nettoyage
log["rows_after"] = len(df)

# Sauvegarde du dataset nettoyé
df.to_csv(OUTPUT_PATH, index=False)

# Affichage du log
print("=== Nettoyage terminé ===")
print(f"Lignes avant nettoyage: {log['rows_before']}")
print(f"Lignes après nettoyage: {log['rows_after']}")
print(f"Doublons supprimés: {log['duplicates_removed']}")
print("\nValeurs manquantes traitées:")
for col, action in log["missing_values_handled"].items():
    print(f"- {col}: {action}")
print("\nValeurs aberrantes corrigées:")
for col, action in log["outliers_corrected"].items():
    print(f"- {col}: {action}")
print("\nCatégories harmonisées:")
for col, action in log["categories_harmonized"].items():
    print(f"- {col}: {action}")
print("\nFormats corrigés:")
for col, action in log["formats_corrected"].items():
    print(f"- {col}: {action}")