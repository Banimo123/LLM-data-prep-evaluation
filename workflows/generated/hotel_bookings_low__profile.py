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
    date_formats = [
        "%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%B %d, %Y",
        "%d-%b-%Y", "%Y/%m/%d", "%b %d, %Y"
    ]
    for fmt in date_formats:
        try:
            dt = datetime.strptime(str(value), fmt)
            return dt.strftime("%Y-%m-%d")
        except (ValueError, TypeError):
            continue
    try:
        if str(value).isdigit() and len(str(value)) in (9, 10):
            dt = datetime.fromtimestamp(int(value))
            return dt.strftime("%Y-%m-%d")
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

    # Suppression des doublons (conservation du premier)
    initial_rows = len(df)
    df = df.drop_duplicates(subset=[col for col in df.columns if col != "row_id"], keep="first")
    duplicates_removed = initial_rows - len(df)
    print(f"Doublons supprimés: {duplicates_removed}")

    # Nettoyage colonne par colonne
    # hotel: harmonisation des variantes
    if "hotel" in df.columns:
        valid_hotels = ["City Hotel", "Resort Hotel"]
        df["hotel"] = df["hotel"].apply(lambda v: harmonize_category(v, valid_hotels))

    # is_canceled: déjà propre (0/1)
    if "is_canceled" in df.columns:
        pass

    # lead_time: extraction numérique
    if "lead_time" in df.columns:
        df["lead_time"] = df["lead_time"].apply(extract_numeric)
        df["lead_time"] = df["lead_time"].fillna(df["lead_time"].median())

    # arrival_date_year: déjà propre
    if "arrival_date_year" in df.columns:
        pass

    # arrival_date_month: harmonisation
    if "arrival_date_month" in df.columns:
        valid_months = [
            "January", "February", "March", "April", "May", "June",
            "July", "August", "September", "October", "November", "December"
        ]
        df["arrival_date_month"] = df["arrival_date_month"].apply(
            lambda v: harmonize_category(v, valid_months)
        )

    # arrival_date_week_number: déjà propre
    if "arrival_date_week_number" in df.columns:
        pass

    # arrival_date_day_of_month: déjà propre
    if "arrival_date_day_of_month" in df.columns:
        pass

    # stays_in_weekend_nights: valeurs aberrantes (max 19 dans profil)
    if "stays_in_weekend_nights" in df.columns:
        df["stays_in_weekend_nights"] = df["stays_in_weekend_nights"].apply(extract_numeric)
        median_val = df["stays_in_weekend_nights"].median()
        df["stays_in_weekend_nights"] = df["stays_in_weekend_nights"].fillna(median_val)
        df.loc[df["stays_in_weekend_nights"] > 19, "stays_in_weekend_nights"] = median_val

    # stays_in_week_nights: valeurs aberrantes (max 999 dans profil)
    if "stays_in_week_nights" in df.columns:
        df["stays_in_week_nights"] = df["stays_in_week_nights"].apply(extract_numeric)
        median_val = df["stays_in_week_nights"].median()
        df["stays_in_week_nights"] = df["stays_in_week_nights"].fillna(median_val)
        df.loc[df["stays_in_week_nights"] > 30, "stays_in_week_nights"] = median_val  # 99e percentile

    # adults: extraction numérique et harmonisation
    if "adults" in df.columns:
        df["adults"] = df["adults"].apply(extract_numeric)
        mode_val = df["adults"].mode()[0]
        df["adults"] = df["adults"].fillna(mode_val)
        df.loc[(df["adults"] < 0) | (df["adults"] > 10), "adults"] = mode_val

    # children: extraction numérique et imputation
    if "children" in df.columns:
        df["children"] = df["children"].apply(extract_numeric)
        group_col = find_best_grouping_column(df, "children", True)
        if group_col:
            df["children"] = df.groupby(group_col)["children"].transform(
                lambda s: s.fillna(s.median())
            )
        df["children"] = df["children"].fillna(df["children"].median())
        df.loc[(df["children"] < 0) | (df["children"] > 10), "children"] = df["children"].median()

    # babies: valeurs aberrantes (max 49 dans profil)
    if "babies" in df.columns:
        df["babies"] = df["babies"].apply(extract_numeric)
        median_val = df["babies"].median()
        df["babies"] = df["babies"].fillna(median_val)
        df.loc[df["babies"] > 10, "babies"] = median_val  # borne physique raisonnable

    # meal: harmonisation et imputation
    if "meal" in df.columns:
        valid_meals = ["BB", "HB", "FB", "SC", "Undefined"]
        df["meal"] = df["meal"].apply(lambda v: harmonize_category(v, valid_meals))
        group_col = find_best_grouping_column(df, "meal", False)
        if group_col:
            df["meal"] = df.groupby(group_col)["meal"].transform(
                lambda s: s.fillna(s.mode()[0] if not s.mode().empty else "BB")
            )
        df["meal"] = df["meal"].fillna(df["meal"].mode()[0])

    # country: harmonisation et imputation
    if "country" in df.columns:
        valid_countries = [
            "PRT", "GBR", "FRA", "ESP", "DEU", "ITA", "IRL", "BEL", "BRA", "NLD",
            "USA", "CHE", "CHN", "RUS", "POL", "AUT", "SWE", "DNK", "NOR", "FIN"
        ]
        df["country"] = df["country"].apply(lambda v: harmonize_category(v, valid_countries))
        group_col = find_best_grouping_column(df, "country", False)
        if group_col:
            df["country"] = df.groupby(group_col)["country"].transform(
                lambda s: s.fillna(s.mode()[0] if not s.mode().empty else "PRT")
            )
        df["country"] = df["country"].fillna(df["country"].mode()[0])

    # market_segment: harmonisation et imputation
    if "market_segment" in df.columns:
        valid_segments = [
            "Online TA", "Offline TA/TO", "Groups", "Direct",
            "Corporate", "Complementary", "Aviation", "Undefined"
        ]
        df["market_segment"] = df["market_segment"].apply(
            lambda v: harmonize_category(v, valid_segments)
        )
        group_col = find_best_grouping_column(df, "market_segment", False)
        if group_col:
            df["market_segment"] = df.groupby(group_col)["market_segment"].transform(
                lambda s: s.fillna(s.mode()[0] if not s.mode().empty else "Online TA")
            )
        df["market_segment"] = df["market_segment"].fillna(df["market_segment"].mode()[0])

    # distribution_channel: harmonisation
    if "distribution_channel" in df.columns:
        valid_channels = ["TA/TO", "Direct", "Corporate", "GDS", "Undefined"]
        df["distribution_channel"] = df["distribution_channel"].apply(
            lambda v: harmonize_category(v, valid_channels)
        )

    # is_repeated_guest: déjà propre
    if "is_repeated_guest" in df.columns:
        pass

    # previous_cancellations: valeurs aberrantes (max 26 dans profil)
    if "previous_cancellations" in df.columns:
        df["previous_cancellations"] = df["previous_cancellations"].apply(extract_numeric)
        median_val = df["previous_cancellations"].median()
        df["previous_cancellations"] = df["previous_cancellations"].fillna(median_val)
        df.loc[df["previous_cancellations"] > 10, "previous_cancellations"] = median_val

    # previous_bookings_not_canceled: valeurs aberrantes (max 72 dans profil)
    if "previous_bookings_not_canceled" in df.columns:
        df["previous_bookings_not_canceled"] = df["previous_bookings_not_canceled"].apply(extract_numeric)
        median_val = df["previous_bookings_not_canceled"].median()
        df["previous_bookings_not_canceled"] = df["previous_bookings_not_canceled"].fillna(median_val)
        df.loc[df["previous_bookings_not_canceled"] > 20, "previous_bookings_not_canceled"] = median_val

    # reserved_room_type: harmonisation
    if "reserved_room_type" in df.columns:
        valid_rooms = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "K", "L", "P"]
        df["reserved_room_type"] = df["reserved_room_type"].apply(
            lambda v: harmonize_category(v, valid_rooms)
        )

    # assigned_room_type: harmonisation
    if "assigned_room_type" in df.columns:
        valid_rooms = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "K", "L", "P"]
        df["assigned_room_type"] = df["assigned_room_type"].apply(
            lambda v: harmonize_category(v, valid_rooms)
        )

    # booking_changes: valeurs aberrantes (max 21 dans profil)
    if "booking_changes" in df.columns:
        df["booking_changes"] = df["booking_changes"].apply(extract_numeric)
        median_val = df["booking_changes"].median()
        df["booking_changes"] = df["booking_changes"].fillna(median_val)
        df.loc[df["booking_changes"] > 10, "booking_changes"] = median_val

    # deposit_type: harmonisation et imputation
    if "deposit_type" in df.columns:
        valid_deposits = ["No Deposit", "Non Refund", "Refundable"]
        df["deposit_type"] = df["deposit_type"].apply(
            lambda v: harmonize_category(v, valid_deposits)
        )
        df["deposit_type"] = df["deposit_type"].fillna(df["deposit_type"].mode()[0])

    # agent: extraction numérique et imputation
    if "agent" in df.columns:
        df["agent"] = df["agent"].apply(extract_numeric)
        group_col = find_best_grouping_column(df, "agent", True)
        if group_col:
            df["agent"] = df.groupby(group_col)["agent"].transform(
                lambda s: s.fillna(s.median())
            )
        df["agent"] = df["agent"].fillna(df["agent"].median())

    # company: imputation (94% manquants -> suppression de la colonne si trop de manquants)
    if "company" in df.columns:
        if df["company"].isna().mean() > 0.9:
            df = df.drop(columns=["company"])
        else:
            df["company"] = df["company"].apply(extract_numeric)
            df["company"] = df["company"].fillna(df["company"].median())

    # days_in_waiting_list: valeurs aberrantes (max 8993 dans profil)
    if "days_in_waiting_list" in df.columns:
        df["days_in_waiting_list"] = df["days_in_waiting_list"].apply(extract_numeric)
        median_val = df["days_in_waiting_list"].median()
        df["days_in_waiting_list"] = df["days_in_waiting_list"].fillna(median_val)
        df.loc[df["days_in_waiting_list"] > 365, "days_in_waiting_list"] = median_val

    # customer_type: harmonisation
    if "customer_type" in df.columns:
        valid_customers = ["Transient", "Transient-Party", "Contract", "Group"]
        df["customer_type"] = df["customer_type"].apply(
            lambda v: harmonize_category(v, valid_customers)
        )

    # adr: valeurs aberrantes (min -494, max 9997 dans profil)
    if "adr" in df.columns:
        df["adr"] = df["adr"].apply(extract_numeric)
        median_val = df["adr"].median()
        df["adr"] = df["adr"].fillna(median_val)
        df.loc[(df["adr"] < -100) | (df["adr"] > 1000), "adr"] = median_val

    # required_car_parking_spaces: valeurs aberrantes (max 8 dans profil)
    if "required_car_parking_spaces" in df.columns:
        df["required_car_parking_spaces"] = df["required_car_parking_spaces"].apply(extract_numeric)
        mode_val = df["required_car_parking_spaces"].mode()[0]
        df["required_car_parking_spaces"] = df["required_car_parking_spaces"].fillna(mode_val)
        df.loc[df["required_car_parking_spaces"] > 5, "required_car_parking_spaces"] = mode_val

    # total_of_special_requests: valeurs aberrantes (max 5 dans profil)
    if "total_of_special_requests" in df.columns:
        df["total_of_special_requests"] = df["total_of_special_requests"].apply(extract_numeric)
        mode_val = df["total_of_special_requests"].mode()[0]
        df["total_of_special_requests"] = df["total_of_special_requests"].fillna(mode_val)
        df.loc[df["total_of_special_requests"] > 5, "total_of_special_requests"] = mode_val

    # reservation_status: harmonisation
    if "reservation_status" in df.columns:
        valid_status = ["Check-Out", "Canceled", "No-Show"]
        df["reservation_status"] = df["reservation_status"].apply(
            lambda v: harmonize_category(v, valid_status)
        )

    # reservation_status_date: parsing des dates
    if "reservation_status_date" in df.columns:
        df["reservation_status_date"] = df["reservation_status_date"].apply(parse_date)

    # Sauvegarde du dataset nettoyé
    df.to_csv(OUTPUT_PATH, index=False)

    # Log des opérations
    final_rows = len(df)
    print(f"Lignes initiales: {initial_rows}")
    print(f"Lignes finales: {final_rows}")
    print(f"Colonnes finales: {len(df.columns)}")
    if "company" in df.columns and df["company"].isna().mean() > 0.9:
        print("Colonne 'company' supprimée (trop de valeurs manquantes)")

if __name__ == "__main__":
    clean_dataset()