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

# lead_time (text mais devrait être numérique) - extraction des valeurs numériques
df["lead_time"] = df["lead_time"].apply(extract_numeric)
# Imputation des valeurs manquantes (0% dans le profil mais extraction peut en créer)
if df["lead_time"].isna().any():
    median_lead_time = df["lead_time"].median()
    df["lead_time"] = df["lead_time"].fillna(median_lead_time)
    operations.append(f"lead_time: {df['lead_time'].isna().sum()} valeurs manquantes imputées par médiane")

# arrival_date_year (numeric) - pas de correction nécessaire

# arrival_date_month (categorical) - harmonisation des mois
month_mapping = {
    'January': 'January', 'February': 'February', 'March': 'March',
    'April': 'April', 'May': 'May', 'June': 'June',
    'July': 'July', 'August': 'August', 'September': 'September',
    'October': 'October', 'November': 'November', 'December': 'December'
}
df["arrival_date_month"] = df["arrival_date_month"].apply(
    lambda x: month_mapping.get(x.strip().capitalize(), x)
)

# arrival_date_week_number (numeric) - pas de correction nécessaire
# arrival_date_day_of_month (numeric) - pas de correction nécessaire

# stays_in_weekend_nights (numeric) - correction des valeurs aberrantes
# Max observé: 19, moyenne: 0.93 -> pas d'aberrations évidentes
# stays_in_week_nights (numeric) - correction des valeurs aberrantes
# Max observé: 999 (aberrant), 99.5e percentile à calculer
q995 = df["stays_in_week_nights"].quantile(0.995)
df["stays_in_week_nights"] = df["stays_in_week_nights"].apply(
    lambda x: x if x <= q995 else df["stays_in_week_nights"].median()
)
operations.append(f"stays_in_week_nights: valeurs > {q995} remplacées par médiane")

# adults (text mais devrait être numérique) - extraction des valeurs numériques
df["adults"] = df["adults"].apply(extract_numeric)
# Imputation des valeurs manquantes (0% dans le profil mais extraction peut en créer)
if df["adults"].isna().any():
    mode_adults = df["adults"].mode()[0]
    df["adults"] = df["adults"].fillna(mode_adults)
    operations.append(f"adults: {df['adults'].isna().sum()} valeurs manquantes imputées par mode")
# Correction des valeurs aberrantes (min:0, max:45 dans le profil)
df["adults"] = df["adults"].apply(lambda x: x if 0 <= x <= 10 else df["adults"].mode()[0])

# children (text avec valeurs manquantes) - extraction des valeurs numériques
df["children"] = df["children"].apply(extract_numeric)
# Imputation des valeurs manquantes (16.68%)
if df["children"].isna().any():
    mode_children = df["children"].mode()[0]
    df["children"] = df["children"].fillna(mode_children)
    operations.append(f"children: {df['children'].isna().sum()} valeurs manquantes imputées par mode")

# babies (numeric) - correction des valeurs aberrantes (max:49 aberrant)
df["babies"] = df["babies"].apply(lambda x: x if x <= 10 else df["babies"].median())

# meal (categorical) - harmonisation et imputation des valeurs manquantes (16.59%)
valid_meal = ["BB", "HB", "FB", "SC", "Undefined"]
df["meal"] = df["meal"].apply(lambda v: harmonize_category(v, valid_meal))
if df["meal"].isna().any():
    mode_meal = df["meal"].mode()[0]
    df["meal"] = df["meal"].fillna(mode_meal)
    operations.append(f"meal: {df['meal'].isna().sum()} valeurs manquantes imputées par mode")

# country (categorical) - harmonisation des codes pays (3 lettres majuscules)
df["country"] = df["country"].apply(
    lambda x: x.strip().upper() if pd.notna(x) and len(str(x).strip()) == 3 else x
)
# Imputation des valeurs manquantes (17.39%)
if df["country"].isna().any():
    mode_country = df["country"].mode()[0]
    df["country"] = df["country"].fillna(mode_country)
    operations.append(f"country: {df['country'].isna().sum()} valeurs manquantes imputées par mode")

# market_segment (categorical) - harmonisation et imputation (16.66%)
valid_market_segment = [
    "Online TA", "Offline TA/TO", "Groups", "Direct",
    "Corporate", "Complementary", "Aviation", "Undefined"
]
df["market_segment"] = df["market_segment"].apply(
    lambda v: harmonize_category(v, valid_market_segment)
)
if df["market_segment"].isna().any():
    mode_market_segment = df["market_segment"].mode()[0]
    df["market_segment"] = df["market_segment"].fillna(mode_market_segment)
    operations.append(f"market_segment: {df['market_segment'].isna().sum()} valeurs manquantes imputées par mode")

# distribution_channel (categorical) - harmonisation
valid_distribution_channel = ["TA/TO", "Direct", "Corporate", "GDS", "Undefined"]
df["distribution_channel"] = df["distribution_channel"].apply(
    lambda v: harmonize_category(v, valid_distribution_channel)
)

# is_repeated_guest (numeric) - pas de correction nécessaire
# previous_cancellations (numeric) - pas de correction nécessaire
# previous_bookings_not_canceled (numeric) - pas de correction nécessaire

# reserved_room_type (categorical) - harmonisation
valid_room_type = ["A", "B", "C", "D", "E", "F", "G", "H", "L", "P"]
df["reserved_room_type"] = df["reserved_room_type"].apply(
    lambda v: harmonize_category(v, valid_room_type)
)

# assigned_room_type (categorical) - harmonisation
df["assigned_room_type"] = df["assigned_room_type"].apply(
    lambda v: harmonize_category(v, valid_room_type)
)

# booking_changes (numeric) - pas de correction nécessaire
# deposit_type (categorical) - harmonisation
valid_deposit_type = ["No Deposit", "Non Refund"]
df["deposit_type"] = df["deposit_type"].apply(
    lambda v: harmonize_category(v, valid_deposit_type)
)

# agent (text mais devrait être numérique) - extraction et imputation (41.27%)
df["agent"] = df["agent"].apply(extract_numeric)
if df["agent"].isna().any():
    mode_agent = df["agent"].mode()[0]
    df["agent"] = df["agent"].fillna(mode_agent)
    operations.append(f"agent: {df['agent'].isna().sum()} valeurs manquantes imputées par mode")

# company (numeric) - trop de valeurs manquantes (94.31%) -> suppression de la colonne
if "company" in df.columns:
    df = df.drop(columns=["company"])
    operations.append("company: colonne supprimée (94.31% de valeurs manquantes)")

# days_in_waiting_list (numeric) - correction des valeurs aberrantes (max:8996 aberrant)
q995_waiting = df["days_in_waiting_list"].quantile(0.995)
df["days_in_waiting_list"] = df["days_in_waiting_list"].apply(
    lambda x: x if x <= q995_waiting else df["days_in_waiting_list"].median()
)
operations.append(f"days_in_waiting_list: valeurs > {q995_waiting} remplacées par médiane")

# customer_type (categorical) - harmonisation
valid_customer_type = ["Transient", "Transient-Party", "Contract", "Group"]
df["customer_type"] = df["customer_type"].apply(
    lambda v: harmonize_category(v, valid_customer_type)
)

# adr (numeric) - correction des valeurs aberrantes (min:-493.26, max:9993.79)
# Valeurs négatives impossibles -> remplacement par médiane
df["adr"] = df["adr"].apply(lambda x: x if x >= 0 else df["adr"].median())
# Valeurs > 99.5e percentile -> remplacement par médiane
q995_adr = df["adr"].quantile(0.995)
df["adr"] = df["adr"].apply(lambda x: x if x <= q995_adr else df["adr"].median())
operations.append(f"adr: valeurs <0 ou >{q995_adr} remplacées par médiane")

# required_car_parking_spaces (numeric) - pas de correction nécessaire
# total_of_special_requests (numeric) - pas de correction nécessaire

# reservation_status (categorical) - harmonisation
valid_reservation_status = ["Check-Out", "Canceled", "No-Show"]
df["reservation_status"] = df["reservation_status"].apply(
    lambda v: harmonize_category(v, valid_reservation_status)
)

# reservation_status_date (date) - parsing et reformatage
df["reservation_status_date"] = df["reservation_status_date"].apply(parse_date)

# Vérification finale des types
numeric_cols = [
    "is_canceled", "lead_time", "arrival_date_year", "arrival_date_week_number",
    "arrival_date_day_of_month", "stays_in_weekend_nights", "stays_in_week_nights",
    "adults", "children", "babies", "previous_cancellations",
    "previous_bookings_not_canceled", "booking_changes", "days_in_waiting_list",
    "adr", "required_car_parking_spaces", "total_of_special_requests", "is_repeated_guest"
]
for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        if df[col].isna().any():
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            operations.append(f"{col}: {df[col].isna().sum()} valeurs manquantes imputées par médiane")

# Sauvegarde du dataset nettoyé
df.to_csv(OUTPUT_PATH, index=False)

# Résumé des opérations
print(f"Nettoyage terminé. Dataset original: {original_shape[0]} lignes, {original_shape[1]} colonnes")
print(f"Dataset nettoyé: {df.shape[0]} lignes, {df.shape[1]} colonnes")
print("\nOpérations effectuées:")
for op in operations:
    print(f"- {op}")
if duplicates_before > 0:
    print(f"- Doublons supprimés: {duplicates_before}")