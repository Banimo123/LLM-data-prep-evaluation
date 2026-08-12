import pandas as pd
import numpy as np
import re
from datetime import datetime
import difflib

INPUT_PATH = r'datasets\hospital\noisy_low.csv'
OUTPUT_PATH = r'results\cleaned_datasets\hospital\noisy_low__simple.csv'

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
                if len(grp) >= 5:
                    w_mad += (grp - grp.median()).abs().median() * (len(grp) / total)
                else:
                    w_mad += global_mad * (len(grp) / total)
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
                if len(grp) >= 5:
                    w_share += grp.value_counts(normalize=True).iloc[0] * (len(grp) / total)
                else:
                    w_share += global_share * (len(grp) / total)
            if (w_share - global_share) > best_gain and (w_share - global_share) >= 0.05:
                best_gain, best_col = w_share - global_share, cand
    return best_col

def clean_dataset():
    df = pd.read_csv(INPUT_PATH)

    log = {
        'missing_values_imputed': 0,
        'duplicates_removed': 0,
        'numeric_values_extracted': 0,
        'dates_parsed': 0,
        'categories_harmonized': 0,
        'outliers_corrected': 0
    }

    # Remove duplicates based on all columns except row_id
    initial_rows = len(df)
    df = df.drop_duplicates(subset=[col for col in df.columns if col != "row_id"], keep="first")
    log['duplicates_removed'] = initial_rows - len(df)

    # Handle missing values
    for col in df.columns:
        if col == "row_id":
            continue
        if df[col].isna().any():
            is_numeric = pd.api.types.is_numeric_dtype(df[col]) or df[col].apply(lambda x: isinstance(x, (int, float))).any()
            if is_numeric:
                # Check if column has text that should be extracted
                if df[col].apply(lambda x: isinstance(x, str) and re.search(r"[^\d\.\-]", str(x))).any():
                    initial_missing = df[col].isna().sum()
                    df[col] = df[col].apply(extract_numeric)
                    log['numeric_values_extracted'] += (initial_missing - df[col].isna().sum())
                    is_numeric = True

                # Find best grouping column for imputation
                group_col = find_best_grouping_column(df, col, is_numeric=True)
                if group_col:
                    df[col] = df.groupby(group_col)[col].transform(
                        lambda s: s.fillna(s.median())
                    )
                df[col] = df[col].fillna(df[col].median())
                log['missing_values_imputed'] += df[col].isna().sum()
            else:
                # Categorical column
                group_col = find_best_grouping_column(df, col, is_numeric=False)
                if group_col:
                    df[col] = df.groupby(group_col)[col].transform(
                        lambda s: s.fillna(s.mode().iloc[0] if not s.mode().empty else "Unknown")
                    )
                mode_val = df[col].mode().iloc[0] if not df[col].mode().empty else "Unknown"
                df[col] = df[col].fillna(mode_val)
                log['missing_values_imputed'] += df[col].isna().sum()

    # Harmonize categorical columns
    valid_values_hospital_type = [
        "acute care hospitals", "critical access hospitals",
        "children's hospitals", "psychiatric hospitals"
    ]
    if "HospitalType" in df.columns:
        df["HospitalType"] = df["HospitalType"].apply(
            lambda v: harmonize_category(v, valid_values_hospital_type)
        )
        log['categories_harmonized'] += 1

    valid_values_owner = [
        "voluntary non-profit - private", "proprietary",
        "government - hospital district or authority",
        "voluntary non-profit - other", "government - federal",
        "government - state", "government - local", "church operated"
    ]
    if "HospitalOwner" in df.columns:
        df["HospitalOwner"] = df["HospitalOwner"].apply(
            lambda v: harmonize_category(v, valid_values_owner)
        )
        log['categories_harmonized'] += 1

    valid_values_emergency = ["yes", "no"]
    if "EmergencyService" in df.columns:
        df["EmergencyService"] = df["EmergencyService"].apply(
            lambda v: harmonize_category(v, valid_values_emergency)
        )
        log['categories_harmonized'] += 1

    # Clean Score column (numeric with possible text)
    if "Score" in df.columns:
        initial_missing = df["Score"].isna().sum()
        df["Score"] = df["Score"].apply(extract_numeric)
        log['numeric_values_extracted'] += (initial_missing - df["Score"].isna().sum())

        # Handle outliers in Score (assuming it's a percentage 0-100)
        if df["Score"].notna().any():
            q1 = df["Score"].quantile(0.01)
            q99 = df["Score"].quantile(0.99)
            median = df["Score"].median()
            outliers = (df["Score"] < q1) | (df["Score"] > q99)
            df.loc[outliers, "Score"] = median
            log['outliers_corrected'] += outliers.sum()

    # Clean PhoneNumber (should be 10 digits)
    if "PhoneNumber" in df.columns:
        def clean_phone(value):
            if pd.isna(value):
                return value
            s = str(value)
            digits = re.sub(r"[^\d]", "", s)
            if len(digits) == 10:
                return digits
            return value
        df["PhoneNumber"] = df["PhoneNumber"].apply(clean_phone)

    # Clean ZipCode (should be 5 digits)
    if "ZipCode" in df.columns:
        def clean_zip(value):
            if pd.isna(value):
                return value
            s = str(value)
            digits = re.sub(r"[^\d]", "", s)
            if len(digits) == 5:
                return digits
            return value
        df["ZipCode"] = df["ZipCode"].apply(clean_zip)

    # Clean State (should be 2 uppercase letters)
    if "State" in df.columns:
        def clean_state(value):
            if pd.isna(value):
                return value
            s = str(value).strip().upper()
            if len(s) == 2 and s.isalpha():
                return s
            return value
        df["State"] = df["State"].apply(clean_state)

    # Save cleaned dataset
    df.to_csv(OUTPUT_PATH, index=False)

    # Print log
    print("Data Cleaning Summary:")
    print(f"- Rows before cleaning: {initial_rows}")
    print(f"- Rows after cleaning: {len(df)}")
    print(f"- Duplicates removed: {log['duplicates_removed']}")
    print(f"- Missing values imputed: {log['missing_values_imputed']}")
    print(f"- Numeric values extracted from text: {log['numeric_values_extracted']}")
    print(f"- Categories harmonized: {log['categories_harmonized']}")
    print(f"- Outliers corrected: {log['outliers_corrected']}")
    print(f"Cleaned dataset saved to: {OUTPUT_PATH}")

if __name__ == "__main__":
    clean_dataset()