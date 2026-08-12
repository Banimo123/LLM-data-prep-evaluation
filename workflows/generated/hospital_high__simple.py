import pandas as pd
import numpy as np
import re
from datetime import datetime
import difflib

INPUT_PATH = r'datasets\hospital\noisy_high.csv'
OUTPUT_PATH = r'results\cleaned_datasets\hospital\noisy_high__simple.csv'

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
                grp_mad = (grp - grp.median()).abs().median() if len(grp) >= 5 else global_mad
                w_mad += grp_mad * (len(grp) / total)
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
                grp_share = grp.value_counts(normalize=True).iloc[0] if len(grp) >= 5 else global_share
                w_share += grp_share * (len(grp) / total)
            if (w_share - global_share) > best_gain and (w_share - global_share) >= 0.05:
                best_gain, best_col = w_share - global_share, cand
    return best_col

def clean_dataset():
    df = pd.read_csv(INPUT_PATH)

    # Remove exact duplicates (keeping first occurrence)
    initial_rows = len(df)
    df = df.drop_duplicates(subset=[c for c in df.columns if c != "row_id"], keep="first")
    duplicates_removed = initial_rows - len(df)

    # Standardize empty strings and common placeholders
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].replace(["empty", "NaN", "N/A", "unknown", "null", ""], np.nan)

    # Clean State column
    if "State" in df.columns:
        valid_states = ["AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA",
                        "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD",
                        "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ",
                        "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC",
                        "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY"]
        df["State"] = df["State"].apply(lambda x: harmonize_category(x, valid_states) if pd.notna(x) else x)

    # Clean ZipCode column
    if "ZipCode" in df.columns:
        df["ZipCode"] = df["ZipCode"].apply(lambda x: str(x).strip().split()[0][:5] if pd.notna(x) else x)
        df["ZipCode"] = df["ZipCode"].apply(lambda x: x if re.match(r"^\d{5}$", str(x)) else np.nan)

    # Clean PhoneNumber column
    if "PhoneNumber" in df.columns:
        df["PhoneNumber"] = df["PhoneNumber"].apply(
            lambda x: re.sub(r"[^\d]", "", str(x)) if pd.notna(x) else x
        )
        df["PhoneNumber"] = df["PhoneNumber"].apply(
            lambda x: x if len(str(x)) == 10 else np.nan
        )

    # Clean HospitalType column
    if "HospitalType" in df.columns:
        valid_types = [
            "acute care hospitals", "critical access hospitals",
            "children's hospitals", "psychiatric hospitals",
            "rehabilitation hospitals", "long term care hospitals"
        ]
        df["HospitalType"] = df["HospitalType"].apply(
            lambda x: harmonize_category(x, valid_types) if pd.notna(x) else x
        )

    # Clean HospitalOwner column
    if "HospitalOwner" in df.columns:
        valid_owners = [
            "voluntary non-profit - private", "proprietary",
            "government - federal", "government - state",
            "government - local", "voluntary non-profit - other",
            "voluntary non-profit - church", "physician"
        ]
        df["HospitalOwner"] = df["HospitalOwner"].apply(
            lambda x: harmonize_category(x, valid_owners) if pd.notna(x) else x
        )

    # Clean EmergencyService column
    if "EmergencyService" in df.columns:
        valid_emergency = ["yes", "no"]
        df["EmergencyService"] = df["EmergencyService"].apply(
            lambda x: harmonize_category(x, valid_emergency) if pd.notna(x) else x
        )

    # Clean City column
    if "City" in df.columns:
        df["City"] = df["City"].str.strip().str.title()

    # Clean numeric columns
    numeric_cols = ["ZipCode", "Score"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].apply(extract_numeric)
            is_numeric = pd.api.types.is_numeric_dtype(df[col])
            if is_numeric:
                # Handle outliers for Score (assuming it's a percentage)
                if col == "Score":
                    q99 = df[col].quantile(0.99)
                    median = df[col].median()
                    df[col] = df[col].where(df[col] <= q99, median)
                # Impute missing values
                group_col = find_best_grouping_column(df, col, True)
                if group_col:
                    df[col] = df.groupby(group_col)[col].transform(
                        lambda s: s.fillna(s.median())
                    )
                df[col] = df[col].fillna(df[col].median())

    # Clean categorical columns with missing values
    categorical_cols = ["HospitalType", "HospitalOwner", "EmergencyService", "State", "CountyName"]
    for col in categorical_cols:
        if col in df.columns:
            group_col = find_best_grouping_column(df, col, False)
            if group_col:
                df[col] = df.groupby(group_col)[col].transform(
                    lambda s: s.fillna(s.mode().iloc[0] if not s.mode().empty else "Unknown")
                )
            df[col] = df[col].fillna(df[col].mode().iloc[0] if not df[col].mode().empty else "Unknown")

    # Clean text columns with consistent patterns
    if "Address1" in df.columns:
        df["Address1"] = df["Address1"].str.strip().str.title()

    # Ensure row_id is preserved and not modified
    if "row_id" not in df.columns:
        df.insert(0, "row_id", range(len(df)))

    # Save cleaned dataset
    df.to_csv(OUTPUT_PATH, index=False)

    # Print summary
    print(f"Data cleaning completed. Summary:")
    print(f"- Initial rows: {initial_rows}")
    print(f"- Duplicates removed: {duplicates_removed}")
    print(f"- Final rows: {len(df)}")
    print(f"- Columns processed: {len(df.columns)}")
    for col in df.columns:
        missing = df[col].isna().sum()
        if missing > 0:
            print(f"  - {col}: {missing} missing values after cleaning")

if __name__ == "__main__":
    clean_dataset()