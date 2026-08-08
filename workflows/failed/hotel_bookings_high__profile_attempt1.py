import pandas as pd
import numpy as np
import re
from datetime import datetime
import difflib

INPUT_PATH = "datasets/hotel_bookings/noisy_high.csv"
OUTPUT_PATH = "results/cleaned_datasets/hotel_bookings/noisy_high__profile.csv"

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
        "%d-%b-%Y", "%Y/%m/%d", "%d.%m.%Y", "%m.%d.%Y"
    ]
    s = str(value).strip()
    for fmt in date_formats:
        try:
            dt = datetime.strptime(s, fmt)
            return dt.strftime("%Y-%m-%d")
        except ValueError:
            continue
    if s.isdigit() and len(s) in (9, 10):
        try:
            dt = datetime.fromtimestamp(int(s))
            return dt.strftime("%Y-%m-%d")
        except (ValueError, OSError):
            pass
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

def clean_dataset():
    df = pd.read_csv(INPUT_PATH)

    # Log initial
    initial_rows, initial_cols = df.shape
    print(f"Dataset initial: {initial_rows} lignes, {initial_cols} colonnes")

    # Suppression des doublons (conservation de la première occurrence)
    duplicates_before = df.duplicated().sum()
    df = df.drop_duplicates().reset_index(drop=True)
    duplicates_after = df.duplicated().sum()
    print(f"Doublons supprimés: {duplicates_before - duplicates_after}")

    # Nettoyage colonne par colonne
    operations_log = []

    # hotel (categorical) - harmonisation des variantes
    if "hotel" in df.columns:
        valid_values_hotel = ["City Hotel", "Resort Hotel"]
        df["hotel"] = df["hotel"].apply(lambda v: harmonize_category(v, valid_values_hotel))
        operations_log.append("hotel: harmonisation des variantes")

    # is_canceled (numeric) - pas de traitement nécessaire (0/1 complets)
    if "is_canceled" in df.columns:
        pass

    # lead_time (categorical -> numeric) - extraction des valeurs numériques
    if "lead_time" in df.columns:
        df["lead_time"] = df["lead_time"].apply(extract_numeric)
        df["lead_time"] = pd.to_numeric(df["lead_time"], errors="coerce")
        operations_log.append("lead_time: extraction des valeurs numériques")

    # arrival_date_year (numeric) - pas de traitement nécessaire
    if "arrival_date_year" in df.columns:
        pass

    # arrival_date_month (categorical) - harmonisation des mois
    if "arrival_date_month" in df.columns:
        valid_months = [
            "January", "February", "March", "April", "May", "June",
            "July", "August", "September", "October", "November", "December"
        ]
        df["arrival_date_month"] = df["arrival_date_month"].apply(
            lambda v: harmonize_category(v, valid_months)
        )
        operations_log.append("arrival_date_month: harmonisation des mois")

    # arrival_date_week_number (numeric) - pas de traitement nécessaire
    if "arrival_date_week_number" in df.columns:
        pass

    # arrival_date_day_of_month (numeric) - pas de traitement nécessaire
    if "arrival_date_day_of_month" in df.columns:
        pass

    # stays_in_weekend_nights (numeric) - traitement des valeurs aberrantes
    if "stays_in_weekend_nights" in df.columns:
        q99 = df["stays_in_weekend_nights"].quantile(0.99)
        outliers = df["stays_in_weekend_nights"] > q99
        df.loc[outliers, "stays_in_weekend_nights"] = df["stays_in_weekend_nights"].median()
        operations_log.append("stays_in_weekend_nights: correction des outliers")

    # stays_in_week_nights (numeric) - traitement des valeurs aberrantes
    if "stays_in_week_nights" in df.columns:
        q99 = df["stays_in_week_nights"].quantile(0.99)
        outliers = df["stays_in_week_nights"] > q99
        df.loc[outliers, "stays_in_week_nights"] = df["stays_in_week_nights"].median()
        operations_log.append("stays_in_week_nights: correction des outliers")

    # adults (categorical -> numeric) - extraction des valeurs numériques
    if "adults" in df.columns:
        df["adults"] = df["adults"].apply(extract_numeric)
        df["adults"] = pd.to_numeric(df["adults"], errors="coerce")
        # Correction des valeurs aberrantes (physiquement impossible)
        df.loc[df["adults"] > 20, "adults"] = df["adults"].median()
        operations_log.append("adults: extraction et correction des valeurs numériques")

    # children (categorical -> numeric) - traitement des valeurs manquantes et aberrantes
    if "children" in df.columns:
        df["children"] = df["children"].replace("unknown", np.nan)
        df["children"] = df["children"].apply(extract_numeric)
        df["children"] = pd.to_numeric(df["children"], errors="coerce")
        # Imputation conditionnelle
        group_col = find_best_grouping_column(df, "children", True)
        if group_col:
            df["children"] = df.groupby(group_col)["children"].transform(
                lambda s: s.fillna(s.median())
            )
        df["children"] = df["children"].fillna(df["children"].median())
        # Correction des valeurs aberrantes
        df.loc[df["children"] > 10, "children"] = df["children"].median()
        operations_log.append("children: imputation et correction des valeurs")

    # babies (numeric) - traitement des valeurs aberrantes
    if "babies" in df.columns:
        q99 = df["babies"].quantile(0.99)
        outliers = df["babies"] > q99
        df.loc[outliers, "babies"] = df["babies"].median()
        operations_log.append("babies: correction des outliers")

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
        operations_log.append("meal: harmonisation et imputation")

    # country (categorical) - harmonisation et imputation
    if "country" in df.columns:
        # Liste des codes pays ISO valides (3 lettres majuscules)
        valid_countries = [
            "PRT", "GBR", "FRA", "ESP", "DEU", "ITA", "IRL", "BEL", "NLD",
            "USA", "BRA", "CHN", "RUS", "JPN", "AUS", "CAN", "CHE", "SWE"
        ]
        df["country"] = df["country"].str.strip().str.upper()
        df["country"] = df["country"].apply(
            lambda v: v if len(v) == 3 and v.isalpha() else np.nan
        )
        # Imputation conditionnelle
        group_col = find_best_grouping_column(df, "country", False)
        if group_col:
            df["country"] = df.groupby(group_col)["country"].transform(
                lambda s: s.fillna(s.mode().iloc[0] if not s.mode().empty else "PRT")
            )
        df["country"] = df["country"].fillna(df["country"].mode().iloc[0])
        operations_log.append("country: harmonisation et imputation")

    # market_segment (categorical) - harmonisation et imputation
    if "market_segment" in df.columns:
        valid_values_segment = [
            "Online TA", "Offline TA/TO", "Groups", "Direct",
            "Corporate", "Complementary", "Aviation", "Undefined"
        ]
        df["market_segment"] = df["market_segment"].apply(
            lambda v: harmonize_category(v, valid_values_segment)
        )
        # Imputation conditionnelle
        group_col = find_best_grouping_column(df, "market_segment", False)
        if group_col:
            df["market_segment"] = df.groupby(group_col)["market_segment"].transform(
                lambda s: s.fillna(s.mode().iloc[0] if not s.mode().empty else "Online TA")
            )
        df["market_segment"] = df["market_segment"].fillna(df["market_segment"].mode().iloc[0])
        operations_log.append("market_segment: harmonisation et imputation")

    # distribution_channel (categorical) - harmonisation
    if "distribution_channel" in df.columns:
        valid_values_channel = ["TA/TO", "Direct", "Corporate", "GDS", "Undefined"]
        df["distribution_channel"] = df["distribution_channel"].apply(
            lambda v: harmonize_category(v, valid_values_channel)
        )
        operations_log.append("distribution_channel: harmonisation")

    # is_repeated_guest (numeric) - pas de traitement nécessaire
    if "is_repeated_guest" in df.columns:
        pass

    # previous_cancellations (numeric) - traitement des valeurs aberrantes
    if "previous_cancellations" in df.columns:
        q99 = df["previous_cancellations"].quantile(0.99)
        outliers = df["previous_cancellations"] > q99
        df.loc[outliers, "previous_cancellations"] = df["previous_cancellations"].median()
        operations_log.append("previous_cancellations: correction des outliers")

    # previous_bookings_not_canceled (numeric) - traitement des valeurs aberrantes
    if "previous_bookings_not_canceled" in df.columns:
        q99 = df["previous_bookings_not_canceled"].quantile(0.99)
        outliers = df["previous_bookings_not_canceled"] > q99
        df.loc[outliers, "previous_bookings_not_canceled"] = df["previous_bookings_not_canceled"].median()
        operations_log.append("previous_bookings_not_canceled: correction des outliers")

    # reserved_room_type (categorical) - harmonisation
    if "reserved_room_type" in df.columns:
        valid_values_room = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "K", "L", "P"]
        df["reserved_room_type"] = df["reserved_room_type"].apply(
            lambda v: harmonize_category(v, valid_values_room)
        )
        operations_log.append("reserved_room_type: harmonisation")

    # assigned_room_type (categorical) - harmonisation
    if "assigned_room_type" in df.columns:
        valid_values_room = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "K", "L", "P"]
        df["assigned_room_type"] = df["assigned_room_type"].apply(
            lambda v: harmonize_category(v, valid_values_room)
        )
        operations_log.append("assigned_room_type: harmonisation")

    # booking_changes (numeric) - traitement des valeurs aberrantes
    if "booking_changes" in df.columns:
        q99 = df["booking_changes"].quantile(0.99)
        outliers = df["booking_changes"] > q99
        df.loc[outliers, "booking_changes"] = df["booking_changes"].median()
        operations_log.append("booking_changes: correction des outliers")

    # deposit_type (categorical) - harmonisation
    if "deposit_type" in df.columns:
        valid_values_deposit = ["No Deposit", "Non Refund", "Refundable"]
        df["deposit_type"] = df["deposit_type"].apply(
            lambda v: harmonize_category(v, valid_values_deposit)
        )
        operations_log.append("deposit_type: harmonisation")

    # agent (categorical -> numeric) - extraction et imputation
    if "agent" in df.columns:
        df["agent"] = df["agent"].replace("unknown", np.nan)
        df["agent"] = df["agent"].apply(extract_numeric)
        df["agent"] = pd.to_numeric(df["agent"], errors="coerce")
        # Imputation conditionnelle
        group_col = find_best_grouping_column(df, "agent", True)
        if group_col:
            df["agent"] = df.groupby(group_col)["agent"].transform(
                lambda s: s.fillna(s.median())
            )
        df["agent"] = df["agent"].fillna(df["agent"].median())
        operations_log.append("agent: extraction et imputation")

    # company (numeric) - trop de valeurs manquantes (94%) -> suppression de la colonne
    if "company" in df.columns:
        df = df.drop(columns=["company"])
        operations_log.append("company: suppression (94% de valeurs manquantes)")

    # days_in_waiting_list (numeric) - traitement des valeurs aberrantes
    if "days_in_waiting_list" in df.columns:
        q99 = df["days_in_waiting_list"].quantile(0.99)
        outliers = df["days_in_waiting_list"] > q99
        df.loc[outliers, "days_in_waiting_list"] = df["days_in_waiting_list"].median()
        operations_log.append("days_in_waiting_list: correction des outliers")

    # customer_type (categorical) - harmonisation
    if "customer_type" in df.columns:
        valid_values_customer = ["Transient", "Transient-Party", "Contract", "Group"]
        df["customer_type"] = df["customer_type"].apply(
            lambda v: harmonize_category(v, valid_values_customer)
        )
        operations_log.append("customer_type: harmonisation")

    # adr (numeric) - traitement des valeurs aberrantes
    if "adr" in df.columns:
        q99 = df["adr"].quantile(0.99)
        outliers = (df["adr"] > q99) | (df["adr"] < 0)
        df.loc[outliers, "adr"] = df["adr"].median()
        operations_log.append("adr: correction des outliers")

    # required_car_parking_spaces (numeric) - traitement des valeurs aberrantes
    if "required_car_parking_spaces" in df.columns:
        q99 = df["required_car_parking_spaces"].quantile(0.99)
        outliers = df["required_car_parking_spaces"] > q99
        df.loc[outliers, "required_car_parking_spaces"] = df["required_car_parking_spaces"].median()
        operations_log.append("required_car_parking_spaces: correction des outliers")

    # total_of_special_requests (numeric) - traitement des valeurs aberrantes
    if "total_of_special_requests" in df.columns:
        q99 = df["total_of_special_requests"].quantile(0.99)
        outliers = df["total_of_special_requests"] > q99
        df.loc[outliers, "total_of_special_requests"] = df["total_of_special_requests"].median()
        operations_log.append("total_of_special_requests: correction des outliers")

    # reservation_status (categorical) - harmonisation
    if "reservation_status" in df.columns:
        valid_values_status = ["Check-Out", "Canceled", "No-Show"]
        df["reservation_status"] = df["reservation_status"].apply(
            lambda v: harmonize_category(v, valid_values_status)
        )
        operations_log.append("reservation_status: harmonisation")

    # reservation_status_date (date) - parsing et standardisation
    if "reservation_status_date" in df.columns:
        df["reservation_status_date"] = df["reservation_status_date"].apply(parse_date)
        operations_log.append("reservation_status_date: parsing et standardisation")

    # Log final
    final_rows, final_cols = df.shape
    print(f"Dataset final: {final_rows} lignes, {final_cols} colonnes")
    print("\nOpérations effectuées:")
    for op in operations_log:
        print(f"- {op}")
    print(f"\nLignes supprimées: {initial_rows - final_rows}")
    print(f"Colonnes supprimées: {initial_cols - final_cols}")

    # Sauvegarde
    df.to_csv(OUTPUT_PATH, index=False)

if __name__ == "__main__":
    clean_dataset()