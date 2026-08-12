import pandas as pd
import numpy as np
import re
from datetime import datetime
import difflib

INPUT_PATH = r"datasets\hospital\noisy_medium.csv"
OUTPUT_PATH = r"results\cleaned_datasets\hospital\noisy_medium__simple.csv"

def extract_numeric(value):
    if pd.isna(value):
        return np.nan
    s = str(value).replace(",", ".")
    s = s.replace("O", "0").replace("o", "0")
    match = re.search(r"-?\d+\.?\d*", s)
    return float(match.group()) if match else np.nan

def parse_date(value):
    if pd.isna(value):
        return value
    date_formats = [
        "%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y",
        "%B %d, %Y", "%d-%b-%Y", "%Y/%m/%d"
    ]
    s = str(value).strip()
    for fmt in date_formats:
        try:
            dt = datetime.strptime(s, fmt)
            return dt.strftime("%Y-%m-%d")
        except ValueError:
            continue
    if re.match(r"^\d{9,10}$", s):
        try:
            dt = datetime.fromtimestamp(int(s))
            return dt.strftime("%Y-%m-%d")
        except (ValueError, OSError):
            pass
    return value

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
        "duplicates_removed": 0,
        "numeric_extracted": 0,
        "dates_parsed": 0,
        "categories_harmonized": 0,
        "outliers_corrected": 0
    }

    # Remove duplicates based on all columns except row_id
    initial_rows = len(df)
    df = df.drop_duplicates(subset=[col for col in df.columns if col != "row_id"], keep="first")
    operations["duplicates_removed"] = initial_rows - len(df)

    # Clean ZipCode (numeric extraction)
    if "ZipCode" in df.columns:
        df["ZipCode"] = df["ZipCode"].apply(extract_numeric)
        operations["numeric_extracted"] += df["ZipCode"].notna().sum() - df["ZipCode"].apply(lambda x: isinstance(x, (int, float))).sum()

    # Clean PhoneNumber (numeric extraction)
    if "PhoneNumber" in df.columns:
        df["PhoneNumber"] = df["PhoneNumber"].apply(extract_numeric)
        operations["numeric_extracted"] += df["PhoneNumber"].notna().sum() - df["PhoneNumber"].apply(lambda x: isinstance(x, (int, float))).sum()

    # Clean Score (numeric extraction)
    if "Score" in df.columns:
        df["Score"] = df["Score"].apply(extract_numeric)
        operations["numeric_extracted"] += df["Score"].notna().sum() - df["Score"].apply(lambda x: isinstance(x, (int, float))).sum()

    # Handle missing values
    for col in df.columns:
        if col in ["row_id", "index"]:
            continue

        if pd.api.types.is_numeric_dtype(df[col]):
            is_numeric = True
        else:
            is_numeric = False

        if df[col].isna().any():
            group_col = find_best_grouping_column(df, col, is_numeric)
            if group_col:
                if is_numeric:
                    df[col] = df.groupby(group_col)[col].transform(
                        lambda s: s.fillna(s.median())
                    )
                else:
                    df[col] = df.groupby(group_col)[col].transform(
                        lambda s: s.fillna(s.mode().iloc[0] if not s.mode().empty else "Unknown")
                    )
            if df[col].isna().any():
                if is_numeric:
                    df[col] = df[col].fillna(df[col].median())
                else:
                    df[col] = df[col].fillna(df[col].mode().iloc[0])
            operations["missing_values_imputed"] += df[col].isna().sum()

    # Harmonize categorical columns
    valid_values_hospital_type = [
        "acute care hospitals", "critical access hospitals",
        "children's hospitals", "psychiatric hospitals"
    ]
    if "HospitalType" in df.columns:
        df["HospitalType"] = df["HospitalType"].apply(
            lambda v: harmonize_category(v, valid_values_hospital_type)
        )
        operations["categories_harmonized"] += 1

    valid_values_hospital_owner = [
        "voluntary non-profit - private", "proprietary",
        "government - local", "government - state",
        "voluntary non-profit - other", "government - federal"
    ]
    if "HospitalOwner" in df.columns:
        df["HospitalOwner"] = df["HospitalOwner"].apply(
            lambda v: harmonize_category(v, valid_values_hospital_owner)
        )
        operations["categories_harmonized"] += 1

    valid_values_emergency = ["yes", "no"]
    if "EmergencyService" in df.columns:
        df["EmergencyService"] = df["EmergencyService"].apply(
            lambda v: harmonize_category(v, valid_values_emergency)
        )
        operations["categories_harmonized"] += 1

    # Correct outliers in Score if present
    if "Score" in df.columns and pd.api.types.is_numeric_dtype(df["Score"]):
        q1 = df["Score"].quantile(0.01)
        q99 = df["Score"].quantile(0.99)
        median = df["Score"].median()
        mask = (df["Score"] < q1) | (df["Score"] > q99)
        if mask.any():
            df.loc[mask, "Score"] = median
            operations["outliers_corrected"] += mask.sum()

    # Clean State (standardize to 2 uppercase letters)
    if "State" in df.columns:
        df["State"] = df["State"].astype(str).str.upper().str[:2]

    # Clean empty strings in Address2/Address3
    for col in ["Address2", "Address3"]:
        if col in df.columns:
            df[col] = df[col].replace("empty", np.nan)
            df[col] = df[col].replace("", np.nan)

    # Save cleaned dataset
    df.to_csv(OUTPUT_PATH, index=False)

    # Print operations summary
    print("Data Cleaning Operations Summary:")
    print(f"- Rows removed (duplicates): {operations['duplicates_removed']}")
    print(f"- Missing values imputed: {operations['missing_values_imputed']}")
    print(f"- Numeric values extracted: {operations['numeric_extracted']}")
    print(f"- Categories harmonized: {operations['categories_harmonized']}")
    print(f"- Outliers corrected: {operations['outliers_corrected']}")
    print(f"Final dataset shape: {df.shape}")

if __name__ == "__main__":
    clean_dataset()