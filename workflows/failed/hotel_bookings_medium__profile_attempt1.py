import pandas as pd
import numpy as np
import re
from datetime import datetime
import difflib

# Chemins d'entrée/sortie
INPUT_PATH = "datasets/hotel_bookings/noisy_medium.csv"
OUTPUT_PATH = "results/cleaned_datasets/hotel_bookings/noisy_medium__profile.csv"

# Fonctions utilitaires
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
    formats = ["%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%B %d, %Y", "%d-%b-%Y", "%Y/%m/%d"]
    for fmt in formats:
        try:
            return datetime.strptime(str(value), fmt).strftime("%Y-%m-%d")
        except (ValueError, TypeError):
            continue
    try:
        return datetime.fromtimestamp(int(value)).strftime("%Y-%m-%d")
    except (ValueError, TypeError):
        return value

def find_best_grouping_column(df, target_col, is_numeric):
    target = df[target_col]
    observed = df[target.notna()]
    if len(observed) < 30:
        return None
    candidates = [c for c in df.columns if c not in (target_col, "row_id")
                  and not df[c].isna().any() and 1 < df[c].nunique() <= 50]
    best_col, best_gain = None, 0.0
    if is_numeric:
        t = pd.to_numeric(observed[target_col], errors="coerce").dropna()
        if len(t) < 30:
            return None
        global_mad = (t - t.median()).abs().median()
        if global_mad == 0:
            return None
        for cand in candidates:
            sub = observed.loc[t.index, [cand]].copy(); sub["_t"] = t
            w_mad, total = 0.0, len(sub)
            for _, grp in sub.groupby(cand)["_t"]:
                w_mad += (grp - grp.median()).abs().median() * (len(grp) / total) if len(grp) >= 5 else global_mad * (len(grp) / total)
            reduction = 1 - (w_mad / global_mad)
            if reduction > best_gain and reduction >= 0.20:
                best_gain, best_col = reduction, cand
    else:
        t = observed[target_col].astype(str)
        global_share = t.value_counts(normalize=True).iloc[0]
        for cand in candidates:
            sub = pd.DataFrame({"_g": observed[cand], "_t": t})
            w_share, total = 0.0, len(sub)
            for _, grp in sub.groupby("_g")["_t"]:
                w_share += grp.value_counts(normalize=True).iloc[0] * (len(grp) / total)
            if (w_share - global_share) > best_gain and (w_share - global_share) >= 0.05:
                best_gain, best_col = w_share - global_share, cand
    return best_col

# Chargement des données
df = pd.read_csv(INPUT_PATH)
original_shape = df.shape
operations = []

# Suppression des doublons (conservation de la première occurrence)
duplicates_before = df.duplicated().sum()
df = df.drop_duplicates(keep='first')
duplicates_after = df.duplicated().sum()
if duplicates_before > 0:
    operations.append(f"Doublons supprimés: {duplicates_before}")

# Nettoyage colonne par colonne
# hotel (categorical) - harmonisation des variantes
if 'hotel' in df.columns:
    valid_values_hotel = ["City Hotel", "Resort Hotel"]
    df["hotel"] = df["hotel"].apply(lambda v: harmonize_category(v, valid_values_hotel))

# lead_time (text -> numeric) - extraction des nombres
if 'lead_time' in df.columns:
    df["lead_time"] = df["lead_time"].apply(extract_numeric)
    df["lead_time"] = pd.to_numeric(df["lead_time"], errors='coerce')
    median_lead = df["lead_time"].median()
    df["lead_time"] = df["lead_time"].fillna(median_lead)
    operations.append(f"lead_time: {df['lead_time'].isna().sum()} valeurs manquantes imputées par médiane")

# adults (text -> numeric) - extraction des nombres et correction des fautes
if 'adults' in df.columns:
    df["adults"] = df["adults"].apply(extract_numeric)
    df["adults"] = pd.to_numeric(df["adults"], errors='coerce')
    # Correction des valeurs aberrantes (min=1, max=4 d'après le profil)
    df["adults"] = df["adults"].clip(1, 4)
    mode_adults = df["adults"].mode()[0]
    df["adults"] = df["adults"].fillna(mode_adults)
    operations.append(f"adults: {df['adults'].isna().sum()} valeurs manquantes imputées par mode")

# children (text -> numeric) - extraction des nombres et imputation
if 'children' in df.columns:
    df["children"] = df["children"].replace("unknown", np.nan)
    df["children"] = df["children"].apply(extract_numeric)
    df["children"] = pd.to_numeric(df["children"], errors='coerce')
    # Imputation conditionnelle
    group_col = find_best_grouping_column(df, "children", is_numeric=True)
    if group_col:
        df["children"] = df.groupby(group_col)["children"].transform(
            lambda s: s.fillna(s.median() if not s.median() != s.median() else 0)
        )
    df["children"] = df["children"].fillna(df["children"].median())
    operations.append(f"children: {df['children'].isna().sum()} valeurs manquantes imputées")

# babies (numeric) - correction des valeurs aberrantes
if 'babies' in df.columns:
    # Valeurs aberrantes (max=10 d'après le profil)
    df["babies"] = df["babies"].clip(0, 10)
    operations.append("babies: valeurs aberrantes plafonnées à 10")

# meal (categorical) - harmonisation et imputation
if 'meal' in df.columns:
    valid_values_meal = ["BB", "HB", "FB", "SC", "Undefined"]
    df["meal"] = df["meal"].apply(lambda v: harmonize_category(v, valid_values_meal))
    # Imputation conditionnelle
    group_col = find_best_grouping_column(df, "meal", is_numeric=False)
    if group_col:
        df["meal"] = df.groupby(group_col)["meal"].transform(
            lambda s: s.fillna(s.mode()[0] if not s.mode().empty else "BB")
        )
    df["meal"] = df["meal"].fillna(df["meal"].mode()[0])
    operations.append(f"meal: {df['meal'].isna().sum()} valeurs manquantes imputées par mode")

# country (categorical) - harmonisation et imputation
if 'country' in df.columns:
    # Liste des pays fréquents (3 lettres majuscules)
    valid_countries = ["PRT", "GBR", "FRA", "ESP", "DEU", "ITA", "IRL", "BEL", "BRA"]
    df["country"] = df["country"].apply(lambda v: v.strip().upper() if pd.notna(v) else v)
    df["country"] = df["country"].apply(lambda v: v if len(v) == 3 and v.isalpha() else np.nan)
    # Imputation conditionnelle
    group_col = find_best_grouping_column(df, "country", is_numeric=False)
    if group_col:
        df["country"] = df.groupby(group_col)["country"].transform(
            lambda s: s.fillna(s.mode()[0] if not s.mode().empty else "PRT")
        )
    df["country"] = df["country"].fillna(df["country"].mode()[0])
    operations.append(f"country: {df['country'].isna().sum()} valeurs manquantes imputées par mode")

# market_segment (categorical) - harmonisation et imputation
if 'market_segment' in df.columns:
    valid_values_segment = ["Online TA", "Offline TA/TO", "Groups", "Direct", "Corporate", "Complementary", "Aviation"]
    df["market_segment"] = df["market_segment"].apply(lambda v: harmonize_category(v, valid_values_segment))
    # Imputation conditionnelle
    group_col = find_best_grouping_column(df, "market_segment", is_numeric=False)
    if group_col:
        df["market_segment"] = df.groupby(group_col)["market_segment"].transform(
            lambda s: s.fillna(s.mode()[0] if not s.mode().empty else "Online TA")
        )
    df["market_segment"] = df["market_segment"].fillna(df["market_segment"].mode()[0])
    operations.append(f"market_segment: {df['market_segment'].isna().sum()} valeurs manquantes imputées par mode")

# distribution_channel (categorical) - harmonisation
if 'distribution_channel' in df.columns:
    valid_values_channel = ["TA/TO", "Direct", "Corporate", "GDS", "Undefined"]
    df["distribution_channel"] = df["distribution_channel"].apply(lambda v: harmonize_category(v, valid_values_channel))

# reserved_room_type (categorical) - harmonisation
if 'reserved_room_type' in df.columns:
    valid_values_room = ["A", "B", "C", "D", "E", "F", "G", "H", "L", "P"]
    df["reserved_room_type"] = df["reserved_room_type"].apply(lambda v: harmonize_category(v, valid_values_room))

# assigned_room_type (categorical) - harmonisation
if 'assigned_room_type' in df.columns:
    valid_values_room = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "K"]
    df["assigned_room_type"] = df["assigned_room_type"].apply(lambda v: harmonize_category(v, valid_values_room))

# deposit_type (categorical) - harmonisation
if 'deposit_type' in df.columns:
    valid_values_deposit = ["No Deposit", "Non Refund"]
    df["deposit_type"] = df["deposit_type"].apply(lambda v: harmonize_category(v, valid_values_deposit))

# agent (text -> numeric) - extraction des nombres et imputation
if 'agent' in df.columns:
    df["agent"] = df["agent"].apply(extract_numeric)
    df["agent"] = pd.to_numeric(df["agent"], errors='coerce')
    # Imputation conditionnelle
    group_col = find_best_grouping_column(df, "agent", is_numeric=True)
    if group_col:
        df["agent"] = df.groupby(group_col)["agent"].transform(
            lambda s: s.fillna(s.median() if not s.median() != s.median() else 9.0)
        )
    df["agent"] = df["agent"].fillna(df["agent"].median())
    operations.append(f"agent: {df['agent'].isna().sum()} valeurs manquantes imputées par médiane")

# company (numeric) - trop de valeurs manquantes (94%) -> suppression de la colonne
if 'company' in df.columns:
    df = df.drop(columns=['company'])
    operations.append("company: colonne supprimée (94% de valeurs manquantes)")

# stays_in_week_nights (numeric) - correction des valeurs aberrantes
if 'stays_in_week_nights' in df.columns:
    # Valeurs aberrantes (max=30 d'après le profil)
    df["stays_in_week_nights"] = df["stays_in_week_nights"].clip(0, 30)
    operations.append("stays_in_week_nights: valeurs aberrantes plafonnées à 30")

# adr (numeric) - correction des valeurs aberrantes
if 'adr' in df.columns:
    # Valeurs aberrantes (min=0, max=5000 d'après le profil)
    df["adr"] = df["adr"].clip(0, 5000)
    operations.append("adr: valeurs aberrantes plafonnées à [0, 5000]")

# days_in_waiting_list (numeric) - correction des valeurs aberrantes
if 'days_in_waiting_list' in df.columns:
    # Valeurs aberrantes (max=365 d'après le profil)
    df["days_in_waiting_list"] = df["days_in_waiting_list"].clip(0, 365)
    operations.append("days_in_waiting_list: valeurs aberrantes plafonnées à 365")

# customer_type (categorical) - harmonisation
if 'customer_type' in df.columns:
    valid_values_customer = ["Transient", "Transient-Party", "Contract", "Group"]
    df["customer_type"] = df["customer_type"].apply(lambda v: harmonize_category(v, valid_values_customer))

# reservation_status_date (date) - parsing et reformatage
if 'reservation_status_date' in df.columns:
    df["reservation_status_date"] = df["reservation_status_date"].apply(parse_date)

# Vérification finale des types
numeric_cols = ['is_canceled', 'arrival_date_year', 'arrival_date_week_number',
                'arrival_date_day_of_month', 'stays_in_weekend_nights',
                'stays_in_week_nights', 'adults', 'children', 'babies',
                'is_repeated_guest', 'previous_cancellations',
                'previous_bookings_not_canceled', 'booking_changes',
                'days_in_waiting_list', 'adr', 'required_car_parking_spaces',
                'total_of_special_requests']
for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='ignore')

# Sauvegarde du dataset nettoyé
df.to_csv(OUTPUT_PATH, index=False)

# Log des opérations
print(f"Nettoyage terminé. Dataset original: {original_shape[0]} lignes, {original_shape[1]} colonnes")
print(f"Dataset nettoyé: {df.shape[0]} lignes, {df.shape[1]} colonnes")
print("\nOpérations effectuées:")
for op in operations:
    print(f"- {op}")
if duplicates_before > 0:
    print(f"- Doublons supprimés: {duplicates_before}")