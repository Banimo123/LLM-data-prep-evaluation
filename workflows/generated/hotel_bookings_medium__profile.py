import pandas as pd
import numpy as np
import re
from datetime import datetime
import difflib

INPUT_PATH = "datasets/hotel_bookings/noisy_medium.csv"
OUTPUT_PATH = "results/cleaned_datasets/hotel_bookings/noisy_medium__profile.csv"

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
    for fmt in date_formats:
        try:
            return datetime.strptime(str(value), fmt).strftime("%Y-%m-%d")
        except (ValueError, TypeError):
            continue
    try:
        if str(value).isdigit() and len(str(value)) in (9, 10):
            return datetime.fromtimestamp(int(value)).strftime("%Y-%m-%d")
    except (ValueError, TypeError):
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

    operations = {
        "missing_values_imputed": 0,
        "outliers_corrected": 0,
        "categories_harmonized": 0,
        "dates_parsed": 0,
        "duplicates_removed": 0
    }

    # Suppression des doublons (conservation de la première occurrence)
    initial_rows = len(df)
    df = df.drop_duplicates(subset=[col for col in df.columns if col != "row_id"], keep="first")
    operations["duplicates_removed"] = initial_rows - len(df)

    # Nettoyage colonne par colonne
    if "hotel" in df.columns:
        valid_values_hotel = ["City Hotel", "Resort Hotel"]
        df["hotel"] = df["hotel"].apply(lambda v: harmonize_category(v, valid_values_hotel))
        operations["categories_harmonized"] += df["hotel"].nunique() - len(valid_values_hotel)

    if "is_canceled" in df.columns:
        df["is_canceled"] = pd.to_numeric(df["is_canceled"], errors="coerce")
        df["is_canceled"] = df["is_canceled"].fillna(df["is_canceled"].median())

    if "lead_time" in df.columns:
        df["lead_time"] = df["lead_time"].apply(extract_numeric)
        median_lead = df["lead_time"].median()
        df["lead_time"] = df["lead_time"].fillna(median_lead)
        operations["missing_values_imputed"] += df["lead_time"].isna().sum()

    if "arrival_date_year" in df.columns:
        df["arrival_date_year"] = pd.to_numeric(df["arrival_date_year"], errors="coerce")
        df["arrival_date_year"] = df["arrival_date_year"].fillna(df["arrival_date_year"].median())

    if "arrival_date_month" in df.columns:
        valid_values_month = [
            "January", "February", "March", "April", "May", "June",
            "July", "August", "September", "October", "November", "December"
        ]
        df["arrival_date_month"] = df["arrival_date_month"].apply(
            lambda v: harmonize_category(v, valid_values_month)
        )

    if "arrival_date_week_number" in df.columns:
        df["arrival_date_week_number"] = pd.to_numeric(df["arrival_date_week_number"], errors="coerce")
        df["arrival_date_week_number"] = df["arrival_date_week_number"].fillna(
            df["arrival_date_week_number"].median()
        )

    if "arrival_date_day_of_month" in df.columns:
        df["arrival_date_day_of_month"] = pd.to_numeric(df["arrival_date_day_of_month"], errors="coerce")
        df["arrival_date_day_of_month"] = df["arrival_date_day_of_month"].fillna(
            df["arrival_date_day_of_month"].median()
        )

    if "stays_in_weekend_nights" in df.columns:
        df["stays_in_weekend_nights"] = pd.to_numeric(df["stays_in_weekend_nights"], errors="coerce")
        df["stays_in_weekend_nights"] = df["stays_in_weekend_nights"].fillna(
            df["stays_in_weekend_nights"].median()
        )
        # Correction des outliers (valeurs > 19)
        upper_bound = 19
        outliers = df["stays_in_weekend_nights"] > upper_bound
        if outliers.any():
            df.loc[outliers, "stays_in_weekend_nights"] = df["stays_in_weekend_nights"].median()
            operations["outliers_corrected"] += outliers.sum()

    if "stays_in_week_nights" in df.columns:
        df["stays_in_week_nights"] = df["stays_in_week_nights"].apply(extract_numeric)
        median_stays = df["stays_in_week_nights"].median()
        df["stays_in_week_nights"] = df["stays_in_week_nights"].fillna(median_stays)
        operations["missing_values_imputed"] += df["stays_in_week_nights"].isna().sum()
        # Correction des outliers (valeurs > 99e percentile)
        upper_bound = df["stays_in_week_nights"].quantile(0.99)
        outliers = df["stays_in_week_nights"] > upper_bound
        if outliers.any():
            df.loc[outliers, "stays_in_week_nights"] = median_stays
            operations["outliers_corrected"] += outliers.sum()

    if "adults" in df.columns:
        df["adults"] = df["adults"].apply(extract_numeric)
        mode_adults = df["adults"].mode()[0]
        df["adults"] = df["adults"].fillna(mode_adults)
        operations["missing_values_imputed"] += df["adults"].isna().sum()
        # Correction des valeurs aberrantes (adults > 4)
        outliers = df["adults"] > 4
        if outliers.any():
            df.loc[outliers, "adults"] = mode_adults
            operations["outliers_corrected"] += outliers.sum()

    if "children" in df.columns:
        df["children"] = df["children"].apply(extract_numeric)
        group_col = find_best_grouping_column(df, "children", True)
        if group_col:
            df["children"] = df.groupby(group_col)["children"].transform(
                lambda s: s.fillna(s.median())
            )
        df["children"] = df["children"].fillna(df["children"].median())
        operations["missing_values_imputed"] += df["children"].isna().sum()

    if "babies" in df.columns:
        df["babies"] = pd.to_numeric(df["babies"], errors="coerce")
        df["babies"] = df["babies"].fillna(df["babies"].median())
        # Correction des outliers (babies > 10)
        outliers = df["babies"] > 10
        if outliers.any():
            df.loc[outliers, "babies"] = df["babies"].median()
            operations["outliers_corrected"] += outliers.sum()

    if "meal" in df.columns:
        valid_values_meal = ["BB", "HB", "FB", "SC", "Undefined"]
        df["meal"] = df["meal"].apply(lambda v: harmonize_category(v, valid_values_meal))
        group_col = find_best_grouping_column(df, "meal", False)
        if group_col:
            df["meal"] = df.groupby(group_col)["meal"].transform(
                lambda s: s.fillna(s.mode()[0] if not s.mode().empty else "BB")
            )
        df["meal"] = df["meal"].fillna(df["meal"].mode()[0])
        operations["missing_values_imputed"] += df["meal"].isna().sum()

    if "country" in df.columns:
        valid_values_country = [
            "PRT", "GBR", "FRA", "ESP", "DEU", "ITA", "IRL", "BEL", "BRA"
        ]
        df["country"] = df["country"].apply(lambda v: harmonize_category(v, valid_values_country))
        group_col = find_best_grouping_column(df, "country", False)
        if group_col:
            df["country"] = df.groupby(group_col)["country"].transform(
                lambda s: s.fillna(s.mode()[0] if not s.mode().empty else "PRT")
            )
        df["country"] = df["country"].fillna(df["country"].mode()[0])
        operations["missing_values_imputed"] += df["country"].isna().sum()

    if "market_segment" in df.columns:
        valid_values_market = [
            "Online TA", "Offline TA/TO", "Groups", "Direct",
            "Corporate", "Complementary", "Aviation", "Undefined"
        ]
        df["market_segment"] = df["market_segment"].apply(
            lambda v: harmonize_category(v, valid_values_market)
        )
        group_col = find_best_grouping_column(df, "market_segment", False)
        if group_col:
            df["market_segment"] = df.groupby(group_col)["market_segment"].transform(
                lambda s: s.fillna(s.mode()[0] if not s.mode().empty else "Online TA")
            )
        df["market_segment"] = df["market_segment"].fillna(df["market_segment"].mode()[0])
        operations["missing_values_imputed"] += df["market_segment"].isna().sum()

    if "distribution_channel" in df.columns:
        valid_values_dist = ["TA/TO", "Direct", "Corporate", "GDS", "Undefined"]
        df["distribution_channel"] = df["distribution_channel"].apply(
            lambda v: harmonize_category(v, valid_values_dist)
        )

    if "is_repeated_guest" in df.columns:
        df["is_repeated_guest"] = pd.to_numeric(df["is_repeated_guest"], errors="coerce")
        df["is_repeated_guest"] = df["is_repeated_guest"].fillna(
            df["is_repeated_guest"].median()
        )

    if "previous_cancellations" in df.columns:
        df["previous_cancellations"] = pd.to_numeric(df["previous_cancellations"], errors="coerce")
        df["previous_cancellations"] = df["previous_cancellations"].fillna(
            df["previous_cancellations"].median()
        )
        # Correction des outliers (valeurs > 10)
        outliers = df["previous_cancellations"] > 10
        if outliers.any():
            df.loc[outliers, "previous_cancellations"] = df["previous_cancellations"].median()
            operations["outliers_corrected"] += outliers.sum()

    if "previous_bookings_not_canceled" in df.columns:
        df["previous_bookings_not_canceled"] = pd.to_numeric(
            df["previous_bookings_not_canceled"], errors="coerce"
        )
        df["previous_bookings_not_canceled"] = df["previous_bookings_not_canceled"].fillna(
            df["previous_bookings_not_canceled"].median()
        )
        # Correction des outliers (valeurs > 20)
        outliers = df["previous_bookings_not_canceled"] > 20
        if outliers.any():
            df.loc[outliers, "previous_bookings_not_canceled"] = df["previous_bookings_not_canceled"].median()
            operations["outliers_corrected"] += outliers.sum()

    if "reserved_room_type" in df.columns:
        valid_values_room = ["A", "B", "C", "D", "E", "F", "G", "H", "L", "P"]
        df["reserved_room_type"] = df["reserved_room_type"].apply(
            lambda v: harmonize_category(v, valid_values_room)
        )

    if "assigned_room_type" in df.columns:
        valid_values_room = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "K", "L", "P"]
        df["assigned_room_type"] = df["assigned_room_type"].apply(
            lambda v: harmonize_category(v, valid_values_room)
        )

    if "booking_changes" in df.columns:
        df["booking_changes"] = pd.to_numeric(df["booking_changes"], errors="coerce")
        df["booking_changes"] = df["booking_changes"].fillna(df["booking_changes"].median())
        # Correction des outliers (valeurs > 10)
        outliers = df["booking_changes"] > 10
        if outliers.any():
            df.loc[outliers, "booking_changes"] = df["booking_changes"].median()
            operations["outliers_corrected"] += outliers.sum()

    if "deposit_type" in df.columns:
        valid_values_deposit = ["No Deposit", "Non Refund"]
        df["deposit_type"] = df["deposit_type"].apply(
            lambda v: harmonize_category(v, valid_values_deposit)
        )

    if "agent" in df.columns:
        df["agent"] = df["agent"].apply(extract_numeric)
        group_col = find_best_grouping_column(df, "agent", True)
        if group_col:
            df["agent"] = df.groupby(group_col)["agent"].transform(
                lambda s: s.fillna(s.median())
            )
        df["agent"] = df["agent"].fillna(df["agent"].median())
        operations["missing_values_imputed"] += df["agent"].isna().sum()

    if "company" in df.columns:
        # Taux de manquants très élevé (94%) -> suppression de la colonne
        df = df.drop(columns=["company"])

    if "days_in_waiting_list" in df.columns:
        df["days_in_waiting_list"] = pd.to_numeric(df["days_in_waiting_list"], errors="coerce")
        df["days_in_waiting_list"] = df["days_in_waiting_list"].fillna(
            df["days_in_waiting_list"].median()
        )
        # Correction des outliers (valeurs > 99e percentile)
        upper_bound = df["days_in_waiting_list"].quantile(0.99)
        outliers = df["days_in_waiting_list"] > upper_bound
        if outliers.any():
            df.loc[outliers, "days_in_waiting_list"] = df["days_in_waiting_list"].median()
            operations["outliers_corrected"] += outliers.sum()

    if "customer_type" in df.columns:
        valid_values_customer = ["Transient", "Transient-Party", "Contract", "Group"]
        df["customer_type"] = df["customer_type"].apply(
            lambda v: harmonize_category(v, valid_values_customer)
        )

    if "adr" in df.columns:
        df["adr"] = df["adr"].apply(extract_numeric)
        median_adr = df["adr"].median()
        df["adr"] = df["adr"].fillna(median_adr)
        operations["missing_values_imputed"] += df["adr"].isna().sum()
        # Correction des valeurs négatives et outliers (valeurs > 99e percentile)
        df["adr"] = df["adr"].clip(lower=0)
        upper_bound = df["adr"].quantile(0.99)
        outliers = df["adr"] > upper_bound
        if outliers.any():
            df.loc[outliers, "adr"] = median_adr
            operations["outliers_corrected"] += outliers.sum()

    if "required_car_parking_spaces" in df.columns:
        df["required_car_parking_spaces"] = pd.to_numeric(
            df["required_car_parking_spaces"], errors="coerce"
        )
        df["required_car_parking_spaces"] = df["required_car_parking_spaces"].fillna(
            df["required_car_parking_spaces"].median()
        )
        # Correction des outliers (valeurs > 5)
        outliers = df["required_car_parking_spaces"] > 5
        if outliers.any():
            df.loc[outliers, "required_car_parking_spaces"] = df["required_car_parking_spaces"].median()
            operations["outliers_corrected"] += outliers.sum()

    if "total_of_special_requests" in df.columns:
        df["total_of_special_requests"] = pd.to_numeric(
            df["total_of_special_requests"], errors="coerce"
        )
        df["total_of_special_requests"] = df["total_of_special_requests"].fillna(
            df["total_of_special_requests"].median()
        )
        # Correction des outliers (valeurs > 5)
        outliers = df["total_of_special_requests"] > 5
        if outliers.any():
            df.loc[outliers, "total_of_special_requests"] = df["total_of_special_requests"].median()
            operations["outliers_corrected"] += outliers.sum()

    if "reservation_status" in df.columns:
        valid_values_status = ["Check-Out", "Canceled", "No-Show"]
        df["reservation_status"] = df["reservation_status"].apply(
            lambda v: harmonize_category(v, valid_values_status)
        )

    if "reservation_status_date" in df.columns:
        df["reservation_status_date"] = df["reservation_status_date"].apply(parse_date)
        operations["dates_parsed"] += df["reservation_status_date"].apply(
            lambda x: 1 if isinstance(x, str) and len(x) == 10 and x[4] == "-" else 0
        ).sum()

    # Sauvegarde du dataset nettoyé
    df.to_csv(OUTPUT_PATH, index=False)

    # Affichage du résumé des opérations
    print("Nettoyage terminé. Résumé des opérations :")
    print(f"- Valeurs manquantes imputées : {operations['missing_values_imputed']}")
    print(f"- Outliers corrigés : {operations['outliers_corrected']}")
    print(f"- Catégories harmonisées : {operations['categories_harmonized']}")
    print(f"- Dates parsées : {operations['dates_parsed']}")
    print(f"- Doublons supprimés : {operations['duplicates_removed']}")
    print(f"- Lignes finales : {len(df)}")
    print(f"- Colonnes finales : {len(df.columns)}")

if __name__ == "__main__":
    clean_dataset()