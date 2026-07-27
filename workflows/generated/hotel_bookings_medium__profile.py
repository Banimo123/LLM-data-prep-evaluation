import pandas as pd
import numpy as np
import re
from datetime import datetime
import difflib

INPUT_PATH = r"benchmark\datasets\hotel_bookings\noisy_medium.csv"
OUTPUT_PATH = r"results\cleaned_datasets\hotel_bookings\noisy_medium__profile.csv"

def extract_numeric(value):
    if pd.isna(value):
        return np.nan
    s = str(value).replace(",", ".")
    s = s.replace("O", "0").replace("o", "0")
    match = re.search(r"-?\d+\.?\d*", s)
    return float(match.group()) if match else np.nan

def harmonize_category(value, valid_values):
    if pd.isna(value):
        return value
    s = str(value).strip()
    if s in valid_values:
        return s
    for v in valid_values:
        if v.lower() == s.lower():
            return v
    match = difflib.get_close_matches(s, valid_values, n=1, cutoff=0.6)
    return match[0] if match else value

def parse_date(value):
    if pd.isna(value):
        return value
    date_formats = [
        "%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%B %d, %Y",
        "%d-%b-%Y", "%Y/%m/%d"
    ]
    s = str(value).strip()
    for fmt in date_formats:
        try:
            return datetime.strptime(s, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    if s.isdigit() and len(s) in (9, 10):
        try:
            return datetime.fromtimestamp(int(s)).strftime("%Y-%m-%d")
        except (ValueError, OSError):
            pass
    return value

# Chargement du dataset
df = pd.read_csv(INPUT_PATH)
original_shape = df.shape
operations = []

# Suppression des doublons (conservation du premier)
duplicates_before = df.duplicated().sum()
df = df.drop_duplicates(keep='first')
duplicates_after = df.duplicated().sum()
if duplicates_before > 0:
    operations.append(f"Doublons supprimés: {duplicates_before}")

# Nettoyage colonne par colonne
# ------------------------------------------------------------------
# hotel (categorical) - harmonisation des variantes
valid_hotel = ["City Hotel", "Resort Hotel"]
df["hotel"] = df["hotel"].apply(lambda v: harmonize_category(v, valid_hotel))

# is_canceled (numeric) - pas de valeurs manquantes, pas de correction nécessaire

# lead_time (categorical/text) - extraction numérique et conversion en int
df["lead_time"] = df["lead_time"].apply(extract_numeric)
df["lead_time"] = df["lead_time"].fillna(df["lead_time"].median()).astype(int)

# arrival_date_year (numeric) - pas de valeurs manquantes, pas de correction nécessaire

# arrival_date_month (categorical) - harmonisation des mois
month_map = {
    "January": "January", "February": "February", "March": "March", "April": "April",
    "May": "May", "June": "June", "July": "July", "August": "August",
    "September": "September", "October": "October", "November": "November", "December": "December"
}
df["arrival_date_month"] = df["arrival_date_month"].apply(
    lambda v: harmonize_category(v, list(month_map.keys()))
)

# arrival_date_week_number (numeric) - pas de valeurs manquantes, pas de correction nécessaire

# arrival_date_day_of_month (numeric) - pas de valeurs manquantes, pas de correction nécessaire

# stays_in_weekend_nights (numeric) - valeurs aberrantes (max=19)
q99 = df["stays_in_weekend_nights"].quantile(0.99)
df["stays_in_weekend_nights"] = df["stays_in_weekend_nights"].apply(
    lambda x: df["stays_in_weekend_nights"].median() if x > q99 else x
)

# stays_in_week_nights (numeric) - valeurs aberrantes (max=999)
q99 = df["stays_in_week_nights"].quantile(0.995)
df["stays_in_week_nights"] = df["stays_in_week_nights"].apply(
    lambda x: df["stays_in_week_nights"].median() if x > q99 else x
)

# adults (categorical/text) - extraction numérique et correction des fautes de frappe
df["adults"] = df["adults"].apply(extract_numeric)
df["adults"] = df["adults"].fillna(df["adults"].mode()[0]).astype(int)
# Correction des valeurs aberrantes (min=1, max=4 d'après le profil)
df["adults"] = df["adults"].apply(lambda x: 2 if x < 1 or x > 4 else x)

# children (categorical/text) - extraction numérique et imputation des manquants
df["children"] = df["children"].apply(extract_numeric)
df["children"] = df["children"].fillna(df["children"].mode()[0])
# Correction des valeurs aberrantes (max=10)
df["children"] = df["children"].apply(lambda x: 0 if x > 10 else x)

# babies (numeric) - valeurs aberrantes (max=49)
q99 = df["babies"].quantile(0.995)
df["babies"] = df["babies"].apply(lambda x: df["babies"].median() if x > q99 else x)

# meal (categorical) - harmonisation et imputation des manquants
valid_meal = ["BB", "HB", "FB", "SC", "Undefined"]
df["meal"] = df["meal"].apply(lambda v: harmonize_category(v, valid_meal))
df["meal"] = df["meal"].fillna(df["meal"].mode()[0])

# country (categorical) - harmonisation et imputation des manquants
valid_country = ["PRT", "GBR", "FRA", "ESP", "DEU", "ITA", "IRL", "BEL", "BRA"]
df["country"] = df["country"].apply(lambda v: harmonize_category(v, valid_country))
df["country"] = df["country"].fillna("PRT")

# market_segment (categorical) - harmonisation et imputation des manquants
valid_market = ["Online TA", "Offline TA/TO", "Groups", "Direct", "Corporate", "Complementary", "Aviation"]
df["market_segment"] = df["market_segment"].apply(lambda v: harmonize_category(v, valid_market))
df["market_segment"] = df["market_segment"].fillna(df["market_segment"].mode()[0])

# distribution_channel (categorical) - harmonisation
valid_dist = ["TA/TO", "Direct", "Corporate", "GDS", "Undefined"]
df["distribution_channel"] = df["distribution_channel"].apply(lambda v: harmonize_category(v, valid_dist))

# is_repeated_guest (numeric) - pas de correction nécessaire

# previous_cancellations (numeric) - valeurs aberrantes (max=26)
q99 = df["previous_cancellations"].quantile(0.995)
df["previous_cancellations"] = df["previous_cancellations"].apply(
    lambda x: df["previous_cancellations"].median() if x > q99 else x
)

# previous_bookings_not_canceled (numeric) - valeurs aberrantes (max=72)
q99 = df["previous_bookings_not_canceled"].quantile(0.995)
df["previous_bookings_not_canceled"] = df["previous_bookings_not_canceled"].apply(
    lambda x: df["previous_bookings_not_canceled"].median() if x > q99 else x
)

# reserved_room_type (categorical) - harmonisation
valid_room = ["A", "D", "E", "F", "G", "B", "C", "H", "P", "L"]
df["reserved_room_type"] = df["reserved_room_type"].apply(lambda v: harmonize_category(v, valid_room))

# assigned_room_type (categorical) - harmonisation
valid_room = ["A", "D", "E", "F", "G", "C", "B", "H", "I", "K"]
df["assigned_room_type"] = df["assigned_room_type"].apply(lambda v: harmonize_category(v, valid_room))

# booking_changes (numeric) - valeurs aberrantes (max=21)
q99 = df["booking_changes"].quantile(0.995)
df["booking_changes"] = df["booking_changes"].apply(
    lambda x: df["booking_changes"].median() if x > q99 else x
)

# deposit_type (categorical) - harmonisation
valid_deposit = ["No Deposit", "Non Refund"]
df["deposit_type"] = df["deposit_type"].apply(lambda v: harmonize_category(v, valid_deposit))

# agent (categorical/text) - extraction numérique et imputation des manquants
df["agent"] = df["agent"].apply(extract_numeric)
df["agent"] = df["agent"].fillna(df["agent"].mode()[0])

# company (numeric) - trop de valeurs manquantes (94%) -> suppression de la colonne
if "company" in df.columns:
    df = df.drop(columns=["company"])
    operations.append("Colonne 'company' supprimée (94% de valeurs manquantes)")

# days_in_waiting_list (numeric) - valeurs aberrantes (max=8996)
q99 = df["days_in_waiting_list"].quantile(0.995)
df["days_in_waiting_list"] = df["days_in_waiting_list"].apply(
    lambda x: df["days_in_waiting_list"].median() if x > q99 else x
)

# customer_type (categorical) - harmonisation
valid_customer = ["Transient", "Transient-Party", "Contract", "Group"]
df["customer_type"] = df["customer_type"].apply(lambda v: harmonize_category(v, valid_customer))

# adr (numeric) - valeurs aberrantes (min=-493, max=9993)
df["adr"] = df["adr"].apply(extract_numeric)
q99 = df["adr"].quantile(0.995)
df["adr"] = df["adr"].apply(lambda x: df["adr"].median() if x < 0 or x > q99 else x)

# required_car_parking_spaces (numeric) - valeurs aberrantes (max=8)
df["required_car_parking_spaces"] = df["required_car_parking_spaces"].apply(
    lambda x: 0 if x > 5 else x
)

# total_of_special_requests (numeric) - valeurs aberrantes (max=5)
df["total_of_special_requests"] = df["total_of_special_requests"].apply(
    lambda x: 0 if x > 5 else x
)

# reservation_status (categorical) - harmonisation
valid_status = ["Check-Out", "Canceled", "No-Show"]
df["reservation_status"] = df["reservation_status"].apply(lambda v: harmonize_category(v, valid_status))

# reservation_status_date (categorical/text) - parsing des dates
df["reservation_status_date"] = df["reservation_status_date"].apply(parse_date)

# ------------------------------------------------------------------
# Sauvegarde du dataset nettoyé
df.to_csv(OUTPUT_PATH, index=False)

# Résumé des opérations
print(f"Dataset original: {original_shape[0]} lignes, {original_shape[1]} colonnes")
print(f"Dataset nettoyé: {df.shape[0]} lignes, {df.shape[1]} colonnes")
for op in operations:
    print(f"- {op}")
print("- Valeurs manquantes imputées avec médiane/mode selon le type de colonne")
print("- Valeurs aberrantes corrigées avec médiane/mode")
print("- Catégories harmonisées avec rapprochement des variantes")
print("- Formats de date standardisés")