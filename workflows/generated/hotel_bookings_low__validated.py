import pandas as pd
import numpy as np
import re
from datetime import datetime
import difflib

INPUT_PATH = "benchmark/datasets/hotel_bookings/noisy_low.csv"
OUTPUT_PATH = "results/cleaned_datasets/hotel_bookings/noisy_low__validated.csv"

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

# Suppression des doublons (en conservant le premier)
duplicates_before = df.duplicated().sum()
df = df.drop_duplicates(keep='first')
duplicates_after = df.duplicated().sum()
if duplicates_before > 0:
    operations.append(f"Doublons supprimés: {duplicates_before}")

# Nettoyage colonne par colonne
# hotel (categorical) - harmonisation des variantes
valid_hotel = ["City Hotel", "Resort Hotel"]
df["hotel"] = df["hotel"].apply(lambda v: harmonize_category(v, valid_hotel))
operations.append("Colonne 'hotel' harmonisée")

# is_canceled (numeric) - déjà propre (0/1)
# lead_time (text -> numeric) - extraction des nombres
df["lead_time"] = df["lead_time"].apply(extract_numeric)
# Imputation des valeurs manquantes (peu probable après extraction)
if df["lead_time"].isna().any():
    median_lead = df["lead_time"].median()
    df["lead_time"] = df["lead_time"].fillna(median_lead)
    operations.append(f"Colonne 'lead_time': {df['lead_time'].isna().sum()} valeurs manquantes imputées par médiane")

# arrival_date_year (numeric) - déjà propre
# arrival_date_month (categorical) - harmonisation
valid_months = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December"
]
df["arrival_date_month"] = df["arrival_date_month"].apply(lambda v: harmonize_category(v, valid_months))
operations.append("Colonne 'arrival_date_month' harmonisée")

# arrival_date_week_number (numeric) - déjà propre
# arrival_date_day_of_month (numeric) - déjà propre
# stays_in_weekend_nights (numeric) - valeurs aberrantes (max 19)
q99_weekend = df["stays_in_weekend_nights"].quantile(0.995)
df["stays_in_weekend_nights"] = df["stays_in_weekend_nights"].apply(
    lambda x: df["stays_in_weekend_nights"].median() if x > q99_weekend else x
)
operations.append("Colonne 'stays_in_weekend_nights': valeurs aberrantes corrigées")

# stays_in_week_nights (numeric) - valeurs aberrantes (max 999)
q99_week = df["stays_in_week_nights"].quantile(0.995)
df["stays_in_week_nights"] = df["stays_in_week_nights"].apply(
    lambda x: df["stays_in_week_nights"].median() if x > q99_week else x
)
operations.append("Colonne 'stays_in_week_nights': valeurs aberrantes corrigées")

# adults (text -> numeric) - extraction des nombres
df["adults"] = df["adults"].apply(extract_numeric)
# Imputation des valeurs manquantes
if df["adults"].isna().any():
    mode_adults = df["adults"].mode()[0]
    df["adults"] = df["adults"].fillna(mode_adults)
    operations.append(f"Colonne 'adults': {df['adults'].isna().sum()} valeurs manquantes imputées par mode")

# children (text -> numeric) - extraction des nombres
df["children"] = df["children"].apply(extract_numeric)
# Imputation des valeurs manquantes (8.32%)
if df["children"].isna().any():
    mode_children = df["children"].mode()[0]
    df["children"] = df["children"].fillna(mode_children)
    operations.append(f"Colonne 'children': {df['children'].isna().sum()} valeurs manquantes imputées par mode")

# babies (numeric) - valeurs aberrantes (max 49)
q99_babies = df["babies"].quantile(0.995)
df["babies"] = df["babies"].apply(
    lambda x: df["babies"].median() if x > q99_babies else x
)
operations.append("Colonne 'babies': valeurs aberrantes corrigées")

# meal (categorical) - harmonisation
valid_meal = ["BB", "HB", "FB", "SC", "Undefined"]
df["meal"] = df["meal"].apply(lambda v: harmonize_category(v, valid_meal))
# Imputation des valeurs manquantes (8.35%)
if df["meal"].isna().any():
    mode_meal = df["meal"].mode()[0]
    df["meal"] = df["meal"].fillna(mode_meal)
    operations.append(f"Colonne 'meal': {df['meal'].isna().sum()} valeurs manquantes imputées par mode")

# country (categorical) - harmonisation (codes pays ISO)
df["country"] = df["country"].str.strip().str.upper()
# Imputation des valeurs manquantes (9.16%)
if df["country"].isna().any():
    mode_country = df["country"].mode()[0]
    df["country"] = df["country"].fillna(mode_country)
    operations.append(f"Colonne 'country': {df['country'].isna().sum()} valeurs manquantes imputées par mode")

# market_segment (categorical) - harmonisation
valid_market = ["Online TA", "Offline TA/TO", "Groups", "Direct", "Corporate", "Complementary", "Aviation"]
df["market_segment"] = df["market_segment"].apply(lambda v: harmonize_category(v, valid_market))
# Imputation des valeurs manquantes (8.32%)
if df["market_segment"].isna().any():
    mode_market = df["market_segment"].mode()[0]
    df["market_segment"] = df["market_segment"].fillna(mode_market)
    operations.append(f"Colonne 'market_segment': {df['market_segment'].isna().sum()} valeurs manquantes imputées par mode")

# distribution_channel (categorical) - harmonisation
valid_dist = ["TA/TO", "Direct", "Corporate", "GDS", "Undefined"]
df["distribution_channel"] = df["distribution_channel"].apply(lambda v: harmonize_category(v, valid_dist))
operations.append("Colonne 'distribution_channel' harmonisée")

# is_repeated_guest (numeric) - déjà propre
# previous_cancellations (numeric) - valeurs aberrantes (max 26)
q99_prev_cancel = df["previous_cancellations"].quantile(0.995)
df["previous_cancellations"] = df["previous_cancellations"].apply(
    lambda x: df["previous_cancellations"].median() if x > q99_prev_cancel else x
)
operations.append("Colonne 'previous_cancellations': valeurs aberrantes corrigées")

# previous_bookings_not_canceled (numeric) - valeurs aberrantes (max 72)
q99_prev_book = df["previous_bookings_not_canceled"].quantile(0.995)
df["previous_bookings_not_canceled"] = df["previous_bookings_not_canceled"].apply(
    lambda x: df["previous_bookings_not_canceled"].median() if x > q99_prev_book else x
)
operations.append("Colonne 'previous_bookings_not_canceled': valeurs aberrantes corrigées")

# reserved_room_type (categorical) - harmonisation
valid_room = ["A", "B", "C", "D", "E", "F", "G", "H", "L", "P"]
df["reserved_room_type"] = df["reserved_room_type"].apply(lambda v: harmonize_category(v, valid_room))
operations.append("Colonne 'reserved_room_type' harmonisée")

# assigned_room_type (categorical) - harmonisation
df["assigned_room_type"] = df["assigned_room_type"].apply(lambda v: harmonize_category(v, valid_room))
operations.append("Colonne 'assigned_room_type' harmonisée")

# booking_changes (numeric) - valeurs aberrantes (max 21)
q99_changes = df["booking_changes"].quantile(0.995)
df["booking_changes"] = df["booking_changes"].apply(
    lambda x: df["booking_changes"].median() if x > q99_changes else x
)
operations.append("Colonne 'booking_changes': valeurs aberrantes corrigées")

# deposit_type (categorical) - harmonisation
valid_deposit = ["No Deposit", "Non Refund", "Refundable"]
df["deposit_type"] = df["deposit_type"].apply(lambda v: harmonize_category(v, valid_deposit))
operations.append("Colonne 'deposit_type' harmonisée")

# agent (text -> numeric) - extraction des nombres
df["agent"] = df["agent"].apply(extract_numeric)
# Imputation des valeurs manquantes (34.29%)
if df["agent"].isna().any():
    mode_agent = df["agent"].mode()[0]
    df["agent"] = df["agent"].fillna(mode_agent)
    operations.append(f"Colonne 'agent': {df['agent'].isna().sum()} valeurs manquantes imputées par mode")

# company (numeric) - trop de valeurs manquantes (94.31%) -> suppression
df = df.drop(columns=["company"])
operations.append("Colonne 'company' supprimée (94.31% de valeurs manquantes)")

# days_in_waiting_list (numeric) - valeurs aberrantes (max 8993)
q99_waiting = df["days_in_waiting_list"].quantile(0.995)
df["days_in_waiting_list"] = df["days_in_waiting_list"].apply(
    lambda x: df["days_in_waiting_list"].median() if x > q99_waiting else x
)
operations.append("Colonne 'days_in_waiting_list': valeurs aberrantes corrigées")

# customer_type (categorical) - harmonisation
valid_customer = ["Transient", "Transient-Party", "Contract", "Group"]
df["customer_type"] = df["customer_type"].apply(lambda v: harmonize_category(v, valid_customer))
operations.append("Colonne 'customer_type' harmonisée")

# adr (numeric) - valeurs aberrantes (min -494, max 9997)
q99_adr = df["adr"].quantile(0.995)
df["adr"] = df["adr"].apply(
    lambda x: df["adr"].median() if x > q99_adr or x < 0 else x
)
operations.append("Colonne 'adr': valeurs aberrantes corrigées")

# required_car_parking_spaces (numeric) - valeurs aberrantes (max 8)
q99_parking = df["required_car_parking_spaces"].quantile(0.995)
df["required_car_parking_spaces"] = df["required_car_parking_spaces"].apply(
    lambda x: df["required_car_parking_spaces"].median() if x > q99_parking else x
)
operations.append("Colonne 'required_car_parking_spaces': valeurs aberrantes corrigées")

# total_of_special_requests (numeric) - valeurs aberrantes (max 5)
q99_requests = df["total_of_special_requests"].quantile(0.995)
df["total_of_special_requests"] = df["total_of_special_requests"].apply(
    lambda x: df["total_of_special_requests"].median() if x > q99_requests else x
)
operations.append("Colonne 'total_of_special_requests': valeurs aberrantes corrigées")

# reservation_status (categorical) - harmonisation
valid_status = ["Check-Out", "Canceled", "No-Show"]
df["reservation_status"] = df["reservation_status"].apply(lambda v: harmonize_category(v, valid_status))
operations.append("Colonne 'reservation_status' harmonisée")

# reservation_status_date (date) - parsing
df["reservation_status_date"] = df["reservation_status_date"].apply(parse_date)
operations.append("Colonne 'reservation_status_date' reformatée en YYYY-MM-DD")

# Conversion des colonnes numériques en types appropriés
numeric_cols = [
    "is_canceled", "lead_time", "arrival_date_year", "arrival_date_week_number",
    "arrival_date_day_of_month", "stays_in_weekend_nights", "stays_in_week_nights",
    "adults", "children", "babies", "is_repeated_guest", "previous_cancellations",
    "previous_bookings_not_canceled", "booking_changes", "agent", "days_in_waiting_list",
    "adr", "required_car_parking_spaces", "total_of_special_requests"
]
for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        if df[col].isna().any():
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            operations.append(f"Colonne '{col}': valeurs manquantes imputées par médiane")

# Sauvegarde du dataset nettoyé
df.to_csv(OUTPUT_PATH, index=False)

# Résumé des opérations
print("=== Résumé du nettoyage ===")
print(f"Lignes initiales: {original_shape[0]}, Colonnes initiales: {original_shape[1]}")
print(f"Lignes finales: {df.shape[0]}, Colonnes finales: {df.shape[1]}")
print("\nOpérations effectuées:")
for op in operations:
    print(f"- {op}")