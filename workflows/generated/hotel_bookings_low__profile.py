import pandas as pd
import numpy as np
import re
from datetime import datetime
import difflib

INPUT_PATH = "datasets/hotel_bookings/noisy_low.csv"
OUTPUT_PATH = "results/cleaned_datasets/hotel_bookings/noisy_low__profile.csv"

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
    formats = [
        "%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%B %d, %Y",
        "%d-%b-%Y", "%Y/%m/%d", "%b %d, %Y"
    ]
    for fmt in formats:
        try:
            return datetime.strptime(str(value), fmt).strftime("%Y-%m-%d")
        except (ValueError, TypeError):
            continue
    try:
        if str(value).isdigit() and len(str(value)) in (9, 10):
            return datetime.fromtimestamp(int(value)).strftime("%Y-%m-%d")
    except (ValueError, OSError):
        pass
    return str(value)

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

def clean_dataset():
    df = pd.read_csv(INPUT_PATH)

    # Suppression des doublons (conservation de la première occurrence)
    initial_rows = len(df)
    df = df.drop_duplicates(subset=[col for col in df.columns if col != "row_id"], keep="first")
    duplicates_removed = initial_rows - len(df)
    print(f"Doublons supprimés: {duplicates_removed}")

    # Nettoyage colonne par colonne
    operations = []

    # hotel (categorical) - harmonisation des variantes
    if "hotel" in df.columns:
        valid_values_hotel = ["City Hotel", "Resort Hotel"]
        df["hotel"] = df["hotel"].apply(lambda v: harmonize_category(v, valid_values_hotel))
        operations.append("hotel: harmonisation des variantes")

    # is_canceled (numeric) - pas de correction nécessaire (0/1)
    if "is_canceled" in df.columns:
        df["is_canceled"] = pd.to_numeric(df["is_canceled"], errors="coerce")
        operations.append("is_canceled: conversion en numérique")

    # lead_time (categorical -> numeric) - extraction des valeurs numériques
    if "lead_time" in df.columns:
        df["lead_time"] = df["lead_time"].apply(extract_numeric)
        operations.append("lead_time: extraction des valeurs numériques")

    # arrival_date_year (numeric) - pas de correction nécessaire
    if "arrival_date_year" in df.columns:
        df["arrival_date_year"] = pd.to_numeric(df["arrival_date_year"], errors="coerce")
        operations.append("arrival_date_year: conversion en numérique")

    # arrival_date_month (categorical) - harmonisation des mois
    if "arrival_date_month" in df.columns:
        month_map = {
            "January": "January", "February": "February", "March": "March", "April": "April",
            "May": "May", "June": "June", "July": "July", "August": "August",
            "September": "September", "October": "October", "November": "November", "December": "December"
        }
        df["arrival_date_month"] = df["arrival_date_month"].apply(
            lambda x: month_map.get(str(x).strip().capitalize(), str(x))
        )
        operations.append("arrival_date_month: harmonisation des mois")

    # arrival_date_week_number (numeric) - pas de correction nécessaire
    if "arrival_date_week_number" in df.columns:
        df["arrival_date_week_number"] = pd.to_numeric(df["arrival_date_week_number"], errors="coerce")
        operations.append("arrival_date_week_number: conversion en numérique")

    # arrival_date_day_of_month (numeric) - pas de correction nécessaire
    if "arrival_date_day_of_month" in df.columns:
        df["arrival_date_day_of_month"] = pd.to_numeric(df["arrival_date_day_of_month"], errors="coerce")
        operations.append("arrival_date_day_of_month: conversion en numérique")

    # stays_in_weekend_nights (numeric) - extraction des valeurs numériques
    if "stays_in_weekend_nights" in df.columns:
        df["stays_in_weekend_nights"] = df["stays_in_weekend_nights"].apply(extract_numeric)
        # Correction des valeurs aberrantes (max 19 dans le profil)
        df["stays_in_weekend_nights"] = df["stays_in_weekend_nights"].clip(0, 19)
        operations.append("stays_in_weekend_nights: extraction et correction des valeurs")

    # stays_in_week_nights (numeric) - extraction et correction des valeurs aberrantes
    if "stays_in_week_nights" in df.columns:
        df["stays_in_week_nights"] = df["stays_in_week_nights"].apply(extract_numeric)
        # 99.5e percentile = 28 (pour éviter de supprimer des valeurs valides)
        p995 = df["stays_in_week_nights"].quantile(0.995)
        df.loc[df["stays_in_week_nights"] > p995, "stays_in_week_nights"] = df["stays_in_week_nights"].median()
        operations.append("stays_in_week_nights: extraction et correction des valeurs aberrantes")

    # adults (categorical -> numeric) - extraction des valeurs numériques
    if "adults" in df.columns:
        df["adults"] = df["adults"].apply(extract_numeric)
        # Correction des valeurs aberrantes (max 4 dans le profil)
        df["adults"] = df["adults"].clip(0, 4)
        operations.append("adults: extraction et correction des valeurs")

    # children (categorical -> numeric) - traitement des valeurs manquantes et extraction
    if "children" in df.columns:
        df["children"] = df["children"].apply(extract_numeric)
        # Imputation conditionnelle
        group_col = find_best_grouping_column(df, "children", True)
        if group_col:
            df["children"] = df.groupby(group_col)["children"].transform(
                lambda s: s.fillna(s.median())
            )
        df["children"] = df["children"].fillna(df["children"].median())
        operations.append("children: extraction, imputation des valeurs manquantes")

    # babies (numeric) - correction des valeurs aberrantes
    if "babies" in df.columns:
        df["babies"] = pd.to_numeric(df["babies"], errors="coerce")
        # 99.5e percentile = 2 (pour éviter de supprimer des valeurs valides)
        p995 = df["babies"].quantile(0.995)
        df.loc[df["babies"] > p995, "babies"] = df["babies"].median()
        operations.append("babies: correction des valeurs aberrantes")

    # meal (categorical) - harmonisation et imputation
    if "meal" in df.columns:
        valid_values_meal = ["BB", "HB", "FB", "SC", "Undefined"]
        df["meal"] = df["meal"].apply(lambda v: harmonize_category(v, valid_values_meal))
        # Imputation conditionnelle
        group_col = find_best_grouping_column(df, "meal", False)
        if group_col:
            df["meal"] = df.groupby(group_col)["meal"].transform(
                lambda s: s.fillna(s.mode().iloc[0] if not s.mode().empty else "BB")
            )
        df["meal"] = df["meal"].fillna(df["meal"].mode().iloc[0])
        operations.append("meal: harmonisation et imputation des valeurs manquantes")

    # country (categorical) - harmonisation et imputation
    if "country" in df.columns:
        # Liste des pays fréquents (3 lettres majuscules)
        valid_countries = ["PRT", "GBR", "FRA", "ESP", "DEU", "ITA", "IRL", "BEL", "BRA", "NLD"]
        df["country"] = df["country"].apply(
            lambda x: x.strip().upper() if pd.notna(x) and len(str(x).strip()) == 3 else x
        )
        # Imputation conditionnelle
        group_col = find_best_grouping_column(df, "country", False)
        if group_col:
            df["country"] = df.groupby(group_col)["country"].transform(
                lambda s: s.fillna(s.mode().iloc[0] if not s.mode().empty else "PRT")
            )
        df["country"] = df["country"].fillna(df["country"].mode().iloc[0])
        operations.append("country: harmonisation et imputation des valeurs manquantes")

    # market_segment (categorical) - harmonisation et imputation
    if "market_segment" in df.columns:
        valid_values_segment = ["Online TA", "Offline TA/TO", "Groups", "Direct", "Corporate", "Complementary", "Aviation"]
        df["market_segment"] = df["market_segment"].apply(lambda v: harmonize_category(v, valid_values_segment))
        # Imputation conditionnelle
        group_col = find_best_grouping_column(df, "market_segment", False)
        if group_col:
            df["market_segment"] = df.groupby(group_col)["market_segment"].transform(
                lambda s: s.fillna(s.mode().iloc[0] if not s.mode().empty else "Online TA")
            )
        df["market_segment"] = df["market_segment"].fillna(df["market_segment"].mode().iloc[0])
        operations.append("market_segment: harmonisation et imputation des valeurs manquantes")

    # distribution_channel (categorical) - harmonisation
    if "distribution_channel" in df.columns:
        valid_values_channel = ["TA/TO", "Direct", "Corporate", "GDS", "Undefined"]
        df["distribution_channel"] = df["distribution_channel"].apply(lambda v: harmonize_category(v, valid_values_channel))
        operations.append("distribution_channel: harmonisation")

    # is_repeated_guest (numeric) - pas de correction nécessaire
    if "is_repeated_guest" in df.columns:
        df["is_repeated_guest"] = pd.to_numeric(df["is_repeated_guest"], errors="coerce")
        operations.append("is_repeated_guest: conversion en numérique")

    # previous_cancellations (numeric) - correction des valeurs aberrantes
    if "previous_cancellations" in df.columns:
        df["previous_cancellations"] = pd.to_numeric(df["previous_cancellations"], errors="coerce")
        # 99.5e percentile = 4 (pour éviter de supprimer des valeurs valides)
        p995 = df["previous_cancellations"].quantile(0.995)
        df.loc[df["previous_cancellations"] > p995, "previous_cancellations"] = df["previous_cancellations"].median()
        operations.append("previous_cancellations: correction des valeurs aberrantes")

    # previous_bookings_not_canceled (numeric) - correction des valeurs aberrantes
    if "previous_bookings_not_canceled" in df.columns:
        df["previous_bookings_not_canceled"] = pd.to_numeric(df["previous_bookings_not_canceled"], errors="coerce")
        # 99.5e percentile = 5 (pour éviter de supprimer des valeurs valides)
        p995 = df["previous_bookings_not_canceled"].quantile(0.995)
        df.loc[df["previous_bookings_not_canceled"] > p995, "previous_bookings_not_canceled"] = df["previous_bookings_not_canceled"].median()
        operations.append("previous_bookings_not_canceled: correction des valeurs aberrantes")

    # reserved_room_type (categorical) - harmonisation
    if "reserved_room_type" in df.columns:
        valid_values_room = ["A", "D", "E", "F", "G", "B", "C", "H", "P", "L"]
        df["reserved_room_type"] = df["reserved_room_type"].apply(lambda v: harmonize_category(v, valid_values_room))
        operations.append("reserved_room_type: harmonisation")

    # assigned_room_type (categorical) - harmonisation
    if "assigned_room_type" in df.columns:
        valid_values_room = ["A", "D", "E", "F", "G", "C", "B", "H", "I", "K"]
        df["assigned_room_type"] = df["assigned_room_type"].apply(lambda v: harmonize_category(v, valid_values_room))
        operations.append("assigned_room_type: harmonisation")

    # booking_changes (numeric) - correction des valeurs aberrantes
    if "booking_changes" in df.columns:
        df["booking_changes"] = pd.to_numeric(df["booking_changes"], errors="coerce")
        # 99.5e percentile = 5 (pour éviter de supprimer des valeurs valides)
        p995 = df["booking_changes"].quantile(0.995)
        df.loc[df["booking_changes"] > p995, "booking_changes"] = df["booking_changes"].median()
        operations.append("booking_changes: correction des valeurs aberrantes")

    # deposit_type (categorical) - harmonisation
    if "deposit_type" in df.columns:
        valid_values_deposit = ["No Deposit", "Non Refund", "Refundable"]
        df["deposit_type"] = df["deposit_type"].apply(lambda v: harmonize_category(v, valid_values_deposit))
        operations.append("deposit_type: harmonisation")

    # agent (categorical -> numeric) - extraction et imputation
    if "agent" in df.columns:
        df["agent"] = df["agent"].apply(extract_numeric)
        # Imputation conditionnelle
        group_col = find_best_grouping_column(df, "agent", True)
        if group_col:
            df["agent"] = df.groupby(group_col)["agent"].transform(
                lambda s: s.fillna(s.median())
            )
        df["agent"] = df["agent"].fillna(df["agent"].median())
        operations.append("agent: extraction et imputation des valeurs manquantes")

    # company (numeric) - imputation (taux de manquants très élevé)
    if "company" in df.columns:
        df["company"] = pd.to_numeric(df["company"], errors="coerce")
        # Imputation conditionnelle
        group_col = find_best_grouping_column(df, "company", True)
        if group_col:
            df["company"] = df.groupby(group_col)["company"].transform(
                lambda s: s.fillna(s.median())
            )
        df["company"] = df["company"].fillna(df["company"].median())
        operations.append("company: imputation des valeurs manquantes")

    # days_in_waiting_list (numeric) - correction des valeurs aberrantes
    if "days_in_waiting_list" in df.columns:
        df["days_in_waiting_list"] = pd.to_numeric(df["days_in_waiting_list"], errors="coerce")
        # 99.5e percentile = 180 (pour éviter de supprimer des valeurs valides)
        p995 = df["days_in_waiting_list"].quantile(0.995)
        df.loc[df["days_in_waiting_list"] > p995, "days_in_waiting_list"] = df["days_in_waiting_list"].median()
        operations.append("days_in_waiting_list: correction des valeurs aberrantes")

    # customer_type (categorical) - harmonisation
    if "customer_type" in df.columns:
        valid_values_customer = ["Transient", "Transient-Party", "Contract", "Group"]
        df["customer_type"] = df["customer_type"].apply(lambda v: harmonize_category(v, valid_values_customer))
        operations.append("customer_type: harmonisation")

    # adr (numeric) - correction des valeurs aberrantes
    if "adr" in df.columns:
        df["adr"] = pd.to_numeric(df["adr"], errors="coerce")
        # Correction des valeurs négatives (physiquement impossibles)
        df.loc[df["adr"] < 0, "adr"] = df["adr"].median()
        # 99.5e percentile = 500 (pour éviter de supprimer des valeurs valides)
        p995 = df["adr"].quantile(0.995)
        df.loc[df["adr"] > p995, "adr"] = df["adr"].median()
        operations.append("adr: correction des valeurs aberrantes")

    # required_car_parking_spaces (numeric) - correction des valeurs aberrantes
    if "required_car_parking_spaces" in df.columns:
        df["required_car_parking_spaces"] = pd.to_numeric(df["required_car_parking_spaces"], errors="coerce")
        # 99.5e percentile = 1 (pour éviter de supprimer des valeurs valides)
        p995 = df["required_car_parking_spaces"].quantile(0.995)
        df.loc[df["required_car_parking_spaces"] > p995, "required_car_parking_spaces"] = df["required_car_parking_spaces"].median()
        operations.append("required_car_parking_spaces: correction des valeurs aberrantes")

    # total_of_special_requests (numeric) - correction des valeurs aberrantes
    if "total_of_special_requests" in df.columns:
        df["total_of_special_requests"] = pd.to_numeric(df["total_of_special_requests"], errors="coerce")
        # 99.5e percentile = 3 (pour éviter de supprimer des valeurs valides)
        p995 = df["total_of_special_requests"].quantile(0.995)
        df.loc[df["total_of_special_requests"] > p995, "total_of_special_requests"] = df["total_of_special_requests"].median()
        operations.append("total_of_special_requests: correction des valeurs aberrantes")

    # reservation_status (categorical) - harmonisation
    if "reservation_status" in df.columns:
        valid_values_status = ["Check-Out", "Canceled", "No-Show"]
        df["reservation_status"] = df["reservation_status"].apply(lambda v: harmonize_category(v, valid_values_status))
        operations.append("reservation_status: harmonisation")

    # reservation_status_date (date) - parsing et formatage
    if "reservation_status_date" in df.columns:
        df["reservation_status_date"] = df["reservation_status_date"].apply(parse_date)
        operations.append("reservation_status_date: parsing et formatage")

    # Affichage du résumé des opérations
    print("\nRésumé des opérations de nettoyage:")
    for op in operations:
        print(f"- {op}")
    print(f"\nNombre de lignes initial: {initial_rows}")
    print(f"Nombre de lignes après suppression des doublons: {len(df)}")
    print(f"Nombre de colonnes: {len(df.columns)}")

    # Sauvegarde du dataset nettoyé
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nDataset nettoyé sauvegardé dans: {OUTPUT_PATH}")

if __name__ == "__main__":
    clean_dataset()