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
    "rows_initial": len(df),
    "rows_final": 0
}

# 1. Suppression des doublons (en conservant le premier)
df.drop_duplicates(subset=[col for col in df.columns if col != "row_id"], keep="first", inplace=True)
log["duplicates_removed"] = log["rows_initial"] - len(df)

# 2. Traitement des valeurs manquantes
# children (5% manquants) - colonne catégorielle avec valeurs numériques -> imputation par mode
if "children" in df.columns:
    mode_children = df["children"].mode()[0]
    df["children"] = df["children"].fillna(mode_children)
    log["missing_values_imputed"]["children"] = df["children"].isna().sum()

# meal (5% manquants) - colonne catégorielle -> imputation par mode
if "meal" in df.columns:
    mode_meal = df["meal"].mode()[0]
    df["meal"] = df["meal"].fillna(mode_meal)
    log["missing_values_imputed"]["meal"] = df["meal"].isna().sum()

# country (5.39% manquants) - colonne catégorielle -> imputation par mode
if "country" in df.columns:
    mode_country = df["country"].mode()[0]
    df["country"] = df["country"].fillna(mode_country)
    log["missing_values_imputed"]["country"] = df["country"].isna().sum()

# market_segment (5% manquants) - colonne catégorielle -> imputation par mode
if "market_segment" in df.columns:
    mode_market_segment = df["market_segment"].mode()[0]
    df["market_segment"] = df["market_segment"].fillna(mode_market_segment)
    log["missing_values_imputed"]["market_segment"] = df["market_segment"].isna().sum()

# agent (17.99% manquants) - colonne catégorielle -> imputation par mode
if "agent" in df.columns:
    mode_agent = df["agent"].mode()[0]
    df["agent"] = df["agent"].fillna(mode_agent)
    log["missing_values_imputed"]["agent"] = df["agent"].isna().sum()

# company (94.31% manquants) - trop de manquants -> imputation par mode (valeur la plus fréquente)
if "company" in df.columns:
    mode_company = df["company"].mode()[0]
    df["company"] = df["company"].fillna(mode_company)
    log["missing_values_imputed"]["company"] = df["company"].isna().sum()

# 3. Correction des valeurs aberrantes numériques
# lead_time - contient des valeurs textuelles (ex: "342O") -> extraction numérique
if "lead_time" in df.columns:
    df["lead_time"] = df["lead_time"].apply(extract_numeric)
    # Vérification des valeurs aberrantes (max observé: 2610)
    df["lead_time"] = df["lead_time"].clip(0, 2610)

# stays_in_week_nights - max observé: 999 -> correction des valeurs aberrantes
if "stays_in_week_nights" in df.columns:
    df["stays_in_week_nights"] = df["stays_in_week_nights"].clip(0, 50)  # 50 comme seuil raisonnable

# adr - min: -494.03, max: 9997.54 -> correction des valeurs aberrantes
if "adr" in df.columns:
    df["adr"] = df["adr"].clip(0, 1000)  # ADR ne peut pas être négatif ni > 1000

# days_in_waiting_list - max observé: 8993 -> correction
if "days_in_waiting_list" in df.columns:
    df["days_in_waiting_list"] = df["days_in_waiting_list"].clip(0, 365)  # 1 an max

# 4. Harmonisation des catégories
# market_segment - correction de "unknown" vers le mode
if "market_segment" in df.columns:
    mode_market_segment = df["market_segment"].mode()[0]
    df["market_segment"] = df["market_segment"].replace("unknown", mode_market_segment)
    log["categories_harmonized"]["market_segment"] = (df["market_segment"] == "unknown").sum()

# meal - harmonisation des espaces superflus
if "meal" in df.columns:
    df["meal"] = df["meal"].str.strip()
    # Correction des variantes (ex: "BB  " -> "BB")
    meal_mapping = {
        "BB": "BB",
        "HB": "HB",
        "SC": "SC",
        "Undefined": "Undefined"
    }
    df["meal"] = df["meal"].apply(lambda x: meal_mapping.get(x, x))

# country - harmonisation des espaces superflus
if "country" in df.columns:
    df["country"] = df["country"].str.strip()

# deposit_type - harmonisation des espaces superflus et variantes
if "deposit_type" in df.columns:
    df["deposit_type"] = df["deposit_type"].str.strip()
    deposit_mapping = {
        "No Deposit": "No Deposit",
        "Non Refund": "Non Refund",
        "Refundable": "Refundable"
    }
    df["deposit_type"] = df["deposit_type"].apply(lambda x: deposit_mapping.get(x, x))
    log["categories_harmonized"]["deposit_type"] = (df["deposit_type"].isin(["No Deposit  ", "No Deposit "])).sum()

# hotel - harmonisation des espaces superflus et variantes
if "hotel" in df.columns:
    df["hotel"] = df["hotel"].str.strip()
    hotel_mapping = {
        "Resort Hotel": "Resort Hotel",
        "City Hotel": "City Hotel"
    }
    df["hotel"] = df["hotel"].apply(lambda x: hotel_mapping.get(x, x))

# customer_type - harmonisation des espaces superflus et variantes
if "customer_type" in df.columns:
    df["customer_type"] = df["customer_type"].str.strip()
    customer_type_mapping = {
        "Transient": "Transient",
        "Contract": "Contract",
        "Transient-Party": "Transient-Party",
        "Group": "Group"
    }
    df["customer_type"] = df["customer_type"].apply(lambda x: customer_type_mapping.get(x, x))

# 5. Correction des formats de date
if "reservation_status_date" in df.columns:
    def parse_date(date_str):
        if pd.isna(date_str):
            return date_str
        try:
            # Essai de parsing avec plusieurs formats
            for fmt in ("%Y-%m-%d", "%B %d, %Y", "%d/%m/%Y", "%B %d %Y"):
                try:
                    return datetime.strptime(str(date_str).strip(), fmt).date()
                except ValueError:
                    continue
            return date_str  # Retourne la valeur originale si aucun format ne correspond
        except:
            return date_str

    df["reservation_status_date"] = df["reservation_status_date"].apply(parse_date)
    # Conversion en datetime avec gestion individuelle des erreurs
    def safe_to_datetime(date_val):
        if pd.isna(date_val) or isinstance(date_val, datetime):
            return date_val
        try:
            return pd.to_datetime(date_val, errors='raise')
        except:
            return date_val

    df["reservation_status_date"] = df["reservation_status_date"].apply(safe_to_datetime)
    # Formatage en chaîne si c'est un datetime
    if pd.api.types.is_datetime64_any_dtype(df["reservation_status_date"]):
        df["reservation_status_date"] = df["reservation_status_date"].dt.strftime("%Y-%m-%d")

# 6. Correction des types de données
numeric_cols = [
    "is_canceled", "arrival_date_year", "arrival_date_week_number",
    "arrival_date_day_of_month", "stays_in_weekend_nights", "stays_in_week_nights",
    "babies", "is_repeated_guest", "previous_cancellations",
    "previous_bookings_not_canceled", "booking_changes", "days_in_waiting_list",
    "adr", "required_car_parking_spaces", "total_of_special_requests"
]
for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='ignore')

# adults - conversion en numérique (valeurs comme "2.0" -> 2)
if "adults" in df.columns:
    df["adults"] = df["adults"].apply(extract_numeric).astype("Int64")

# children - conversion en numérique
if "children" in df.columns:
    df["children"] = df["children"].apply(extract_numeric).astype("Int64")

# agent - conversion en numérique
if "agent" in df.columns:
    df["agent"] = df["agent"].apply(extract_numeric).astype("Int64")

# company - conversion en numérique
if "company" in df.columns:
    df["company"] = df["company"].apply(extract_numeric).astype("Int64")

# 7. Vérification finale des valeurs aberrantes
# babies - max observé: 49 -> correction
if "babies" in df.columns:
    df["babies"] = df["babies"].clip(0, 10)  # 10 comme seuil raisonnable

# Enregistrement du dataset nettoyé
df.to_csv(OUTPUT_PATH, index=False)

# Mise à jour du log
log["rows_final"] = len(df)

# Affichage du résumé
print("=== Nettoyage terminé ===")
print(f"Lignes initiales: {log['rows_initial']}")
print(f"Lignes finales: {log['rows_final']}")
print(f"Doublons supprimés: {log['duplicates_removed']}")
print("\nValeurs manquantes imputées:")
for col, count in log["missing_values_imputed"].items():
    print(f"- {col}: {count}")
print("\nCatégories harmonisées:")
for col, count in log["categories_harmonized"].items():
    print(f"- {col}: {count}")