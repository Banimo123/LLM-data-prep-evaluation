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

# Chargement des données
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
operations.append("Colonne 'hotel' harmonisée")

# is_canceled (numeric) - pas de valeurs manquantes, pas de correction nécessaire

# lead_time (text mais devrait être numérique) - extraction des valeurs numériques
df["lead_time"] = df["lead_time"].apply(extract_numeric)
# Imputation des valeurs manquantes (0% de manquants mais extraction peut en créer)
if df["lead_time"].isna().sum() > 0:
    median_lead_time = df["lead_time"].median()
    df["lead_time"] = df["lead_time"].fillna(median_lead_time)
    operations.append(f"Colonne 'lead_time': {df['lead_time'].isna().sum()} valeurs manquantes imputées par médiane")

# arrival_date_year (numeric) - pas de correction nécessaire

# arrival_date_month (categorical) - harmonisation des mois
month_mapping = {
    'January': 'January', 'February': 'February', 'March': 'March',
    'April': 'April', 'May': 'May', 'June': 'June',
    'July': 'July', 'August': 'August', 'September': 'September',
    'October': 'October', 'November': 'November', 'December': 'December'
}
df["arrival_date_month"] = df["arrival_date_month"].apply(
    lambda x: month_mapping.get(x, x) if pd.notna(x) else x
)

# arrival_date_week_number et arrival_date_day_of_month (numeric) - pas de correction nécessaire

# stays_in_weekend_nights (numeric) - valeurs aberrantes (max=19)
# 99.5e percentile = 3, on remplace les valeurs >3 par la médiane (0)
upper_bound = df["stays_in_weekend_nights"].quantile(0.995)
outliers = df["stays_in_weekend_nights"] > upper_bound
if outliers.sum() > 0:
    median_val = df.loc[~outliers, "stays_in_weekend_nights"].median()
    df.loc[outliers, "stays_in_weekend_nights"] = median_val
    operations.append(f"Colonne 'stays_in_weekend_nights': {outliers.sum()} outliers corrigés")

# stays_in_week_nights (numeric) - valeurs aberrantes (max=999)
# 99.5e percentile = 7, on remplace les valeurs >7 par la médiane (1)
upper_bound = df["stays_in_week_nights"].quantile(0.995)
outliers = df["stays_in_week_nights"] > upper_bound
if outliers.sum() > 0:
    median_val = df.loc[~outliers, "stays_in_week_nights"].median()
    df.loc[outliers, "stays_in_week_nights"] = median_val
    operations.append(f"Colonne 'stays_in_week_nights': {outliers.sum()} outliers corrigés")

# adults (text mais devrait être numérique) - extraction des valeurs numériques
df["adults"] = df["adults"].apply(extract_numeric)
# Imputation des valeurs manquantes (0% de manquants mais extraction peut en créer)
if df["adults"].isna().sum() > 0:
    mode_adults = df["adults"].mode()[0]
    df["adults"] = df["adults"].fillna(mode_adults)
    operations.append(f"Colonne 'adults': {df['adults'].isna().sum()} valeurs manquantes imputées par mode")
# Correction des valeurs aberrantes (min=1, max=45 dans le profil)
df["adults"] = df["adults"].clip(1, 4)

# children (text avec valeurs manquantes) - extraction des valeurs numériques
df["children"] = df["children"].apply(extract_numeric)
# Imputation des valeurs manquantes (16.68%)
if df["children"].isna().sum() > 0:
    mode_children = df["children"].mode()[0]
    df["children"] = df["children"].fillna(mode_children)
    operations.append(f"Colonne 'children': {df['children'].isna().sum()} valeurs manquantes imputées par mode")

# babies (numeric) - valeurs aberrantes (max=49)
# 99.5e percentile = 2, on remplace les valeurs >2 par la médiane (0)
upper_bound = df["babies"].quantile(0.995)
outliers = df["babies"] > upper_bound
if outliers.sum() > 0:
    median_val = df.loc[~outliers, "babies"].median()
    df.loc[outliers, "babies"] = median_val
    operations.append(f"Colonne 'babies': {outliers.sum()} outliers corrigés")

# meal (categorical) - harmonisation et imputation des valeurs manquantes (16.59%)
valid_meal = ["BB", "HB", "FB", "SC", "Undefined"]
df["meal"] = df["meal"].apply(lambda v: harmonize_category(v, valid_meal))
if df["meal"].isna().sum() > 0:
    mode_meal = df["meal"].mode()[0]
    df["meal"] = df["meal"].fillna(mode_meal)
    operations.append(f"Colonne 'meal': {df['meal'].isna().sum()} valeurs manquantes imputées par mode")

# country (categorical) - imputation des valeurs manquantes (17.39%)
if df["country"].isna().sum() > 0:
    mode_country = df["country"].mode()[0]
    df["country"] = df["country"].fillna(mode_country)
    operations.append(f"Colonne 'country': {df['country'].isna().sum()} valeurs manquantes imputées par mode")

# market_segment (categorical) - harmonisation et imputation des valeurs manquantes (16.66%)
valid_market = ["Online TA", "Offline TA/TO", "Groups", "Direct", "Corporate", "Complementary", "Aviation", "Undefined"]
df["market_segment"] = df["market_segment"].apply(lambda v: harmonize_category(v, valid_market))
if df["market_segment"].isna().sum() > 0:
    mode_market = df["market_segment"].mode()[0]
    df["market_segment"] = df["market_segment"].fillna(mode_market)
    operations.append(f"Colonne 'market_segment': {df['market_segment'].isna().sum()} valeurs manquantes imputées par mode")

# distribution_channel (categorical) - harmonisation
valid_dist = ["TA/TO", "Direct", "Corporate", "GDS", "Undefined"]
df["distribution_channel"] = df["distribution_channel"].apply(lambda v: harmonize_category(v, valid_dist))

# is_repeated_guest (numeric) - pas de correction nécessaire

# previous_cancellations (numeric) - valeurs aberrantes (max=26)
# 99.5e percentile = 2, on remplace les valeurs >2 par la médiane (0)
upper_bound = df["previous_cancellations"].quantile(0.995)
outliers = df["previous_cancellations"] > upper_bound
if outliers.sum() > 0:
    median_val = df.loc[~outliers, "previous_cancellations"].median()
    df.loc[outliers, "previous_cancellations"] = median_val
    operations.append(f"Colonne 'previous_cancellations': {outliers.sum()} outliers corrigés")

# previous_bookings_not_canceled (numeric) - valeurs aberrantes (max=72)
# 99.5e percentile = 2, on remplace les valeurs >2 par la médiane (0)
upper_bound = df["previous_bookings_not_canceled"].quantile(0.995)
outliers = df["previous_bookings_not_canceled"] > upper_bound
if outliers.sum() > 0:
    median_val = df.loc[~outliers, "previous_bookings_not_canceled"].median()
    df.loc[outliers, "previous_bookings_not_canceled"] = median_val
    operations.append(f"Colonne 'previous_bookings_not_canceled': {outliers.sum()} outliers corrigés")

# reserved_room_type et assigned_room_type (categorical) - harmonisation
valid_room = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "K", "L", "P"]
df["reserved_room_type"] = df["reserved_room_type"].apply(lambda v: harmonize_category(v, valid_room))
df["assigned_room_type"] = df["assigned_room_type"].apply(lambda v: harmonize_category(v, valid_room))

# booking_changes (numeric) - valeurs aberrantes (max=21)
# 99.5e percentile = 3, on remplace les valeurs >3 par la médiane (0)
upper_bound = df["booking_changes"].quantile(0.995)
outliers = df["booking_changes"] > upper_bound
if outliers.sum() > 0:
    median_val = df.loc[~outliers, "booking_changes"].median()
    df.loc[outliers, "booking_changes"] = median_val
    operations.append(f"Colonne 'booking_changes': {outliers.sum()} outliers corrigés")

# deposit_type (categorical) - harmonisation
valid_deposit = ["No Deposit", "Non Refund"]
df["deposit_type"] = df["deposit_type"].apply(lambda v: harmonize_category(v, valid_deposit))

# agent (text mais devrait être numérique) - extraction des valeurs numériques
df["agent"] = df["agent"].apply(extract_numeric)
# Imputation des valeurs manquantes (41.27%)
if df["agent"].isna().sum() > 0:
    mode_agent = df["agent"].mode()[0]
    df["agent"] = df["agent"].fillna(mode_agent)
    operations.append(f"Colonne 'agent': {df['agent'].isna().sum()} valeurs manquantes imputées par mode")

# company (numeric) - trop de valeurs manquantes (94.31%) -> suppression de la colonne
if "company" in df.columns:
    df = df.drop(columns=["company"])
    operations.append("Colonne 'company' supprimée (94.31% de valeurs manquantes)")

# days_in_waiting_list (numeric) - valeurs aberrantes (max=8996)
# 99.5e percentile = 123, on remplace les valeurs >123 par la médiane (0)
upper_bound = df["days_in_waiting_list"].quantile(0.995)
outliers = df["days_in_waiting_list"] > upper_bound
if outliers.sum() > 0:
    median_val = df.loc[~outliers, "days_in_waiting_list"].median()
    df.loc[outliers, "days_in_waiting_list"] = median_val
    operations.append(f"Colonne 'days_in_waiting_list': {outliers.sum()} outliers corrigés")

# customer_type (categorical) - harmonisation
valid_customer = ["Transient", "Transient-Party", "Contract", "Group"]
df["customer_type"] = df["customer_type"].apply(lambda v: harmonize_category(v, valid_customer))

# adr (numeric) - valeurs aberrantes (min=-493.26, max=9993.79)
# 99.5e percentile = 300, on remplace les valeurs <0 ou >300 par la médiane (80)
lower_bound = 0
upper_bound = df["adr"].quantile(0.995)
outliers = (df["adr"] < lower_bound) | (df["adr"] > upper_bound)
if outliers.sum() > 0:
    median_val = df.loc[~outliers, "adr"].median()
    df.loc[outliers, "adr"] = median_val
    operations.append(f"Colonne 'adr': {outliers.sum()} outliers corrigés")

# required_car_parking_spaces (numeric) - valeurs aberrantes (max=8)
# 99.5e percentile = 1, on remplace les valeurs >1 par la médiane (0)
upper_bound = df["required_car_parking_spaces"].quantile(0.995)
outliers = df["required_car_parking_spaces"] > upper_bound
if outliers.sum() > 0:
    median_val = df.loc[~outliers, "required_car_parking_spaces"].median()
    df.loc[outliers, "required_car_parking_spaces"] = median_val
    operations.append(f"Colonne 'required_car_parking_spaces': {outliers.sum()} outliers corrigés")

# total_of_special_requests (numeric) - valeurs aberrantes (max=5)
# 99.5e percentile = 3, on remplace les valeurs >3 par la médiane (0)
upper_bound = df["total_of_special_requests"].quantile(0.995)
outliers = df["total_of_special_requests"] > upper_bound
if outliers.sum() > 0:
    median_val = df.loc[~outliers, "total_of_special_requests"].median()
    df.loc[outliers, "total_of_special_requests"] = median_val
    operations.append(f"Colonne 'total_of_special_requests': {outliers.sum()} outliers corrigés")

# reservation_status (categorical) - harmonisation
valid_status = ["Check-Out", "Canceled", "No-Show"]
df["reservation_status"] = df["reservation_status"].apply(lambda v: harmonize_category(v, valid_status))

# reservation_status_date (date) - parsing et reformatage
df["reservation_status_date"] = df["reservation_status_date"].apply(parse_date)

# Conversion des colonnes numériques en types appropriés
numeric_cols = [
    "is_canceled", "lead_time", "arrival_date_year", "arrival_date_week_number",
    "arrival_date_day_of_month", "stays_in_weekend_nights", "stays_in_week_nights",
    "adults", "children", "babies", "is_repeated_guest", "previous_cancellations",
    "previous_bookings_not_canceled", "booking_changes", "days_in_waiting_list",
    "adr", "required_car_parking_spaces", "total_of_special_requests"
]
for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        if df[col].isna().sum() > 0:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)

# Sauvegarde du dataset nettoyé
df.to_csv(OUTPUT_PATH, index=False)

# Résumé des opérations
print(f"Nettoyage terminé. Dataset original: {original_shape[0]} lignes, {original_shape[1]} colonnes")
print(f"Dataset nettoyé: {df.shape[0]} lignes, {df.shape[1]} colonnes")
print("\nOpérations effectuées:")
for op in operations:
    print(f"- {op}")