import pandas as pd
import numpy as np
import re
from datetime import datetime
import difflib

INPUT_PATH = "datasets/hospital/noisy_medium.csv"
OUTPUT_PATH = "results/cleaned_datasets/hospital/noisy_medium__simple.csv"

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
        "%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%B %d, %Y",
        "%d-%b-%Y", "%Y/%m/%d"
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
            sub = observed.loc[t.index, [cand]].copy()
            sub["_t"] = t
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
                mode_share = grp.value_counts(normalize=True).iloc[0] if not grp.empty else 0
                w_share += mode_share * (len(grp) / total)
            if (w_share - global_share) > best_gain and (w_share - global_share) >= 0.05:
                best_gain, best_col = w_share - global_share, cand
    return best_col

def clean_hospital_data():
    df = pd.read_csv(INPUT_PATH)

    operations = {
        "missing_values_imputed": 0,
        "duplicates_removed": 0,
        "numeric_values_extracted": 0,
        "dates_parsed": 0,
        "categories_harmonized": 0,
        "outliers_corrected": 0
    }

    # Remove exact duplicates (keeping first occurrence)
    initial_rows = len(df)
    df = df.drop_duplicates(subset=[c for c in df.columns if c != "row_id"], keep="first")
    operations["duplicates_removed"] = initial_rows - len(df)

    # Clean ZipCode (numeric extraction)
    if "ZipCode" in df.columns:
        df["ZipCode"] = df["ZipCode"].apply(extract_numeric)
        operations["numeric_values_extracted"] += df["ZipCode"].notna().sum() - df["ZipCode"].apply(lambda x: isinstance(x, (int, float))).sum()

    # Clean PhoneNumber (numeric extraction)
    if "PhoneNumber" in df.columns:
        df["PhoneNumber"] = df["PhoneNumber"].apply(extract_numeric)
        operations["numeric_values_extracted"] += df["PhoneNumber"].notna().sum() - df["PhoneNumber"].apply(lambda x: isinstance(x, (int, float))).sum()

    # Clean Score (numeric extraction)
    if "Score" in df.columns:
        # First check if Score contains percentage values
        score_is_percentage = df["Score"].astype(str).str.contains('%', na=False).any()
        if score_is_percentage:
            df["Score"] = df["Score"].astype(str).str.replace('%', '', regex=False)
        df["Score"] = df["Score"].apply(extract_numeric)
        operations["numeric_values_extracted"] += df["Score"].notna().sum() - df["Score"].apply(lambda x: isinstance(x, (int, float))).sum()

    # Handle missing values
    for col in df.columns:
        if col in ["row_id", "index"]:
            continue

        if df[col].dtype == "object":
            # Check if column is mostly numeric with some text
            numeric_count = df[col].apply(lambda x: isinstance(extract_numeric(x), (int, float))).sum()
            if numeric_count / len(df) > 0.8 and col not in ["ProviderNumber", "ZipCode", "PhoneNumber", "HospitalName", "Address1", "Address2", "Address3", "City", "CountyName", "Condition", "MeasureCode", "MeasureName", "Sample", "Stateavg"]:
                df[col] = df[col].apply(extract_numeric)
                operations["numeric_values_extracted"] += numeric_count - df[col].apply(lambda x: isinstance(x, (int, float))).sum()

            # Check for repeated text values (likely placeholders)
            value_counts = df[col].value_counts()
            if len(value_counts) > 0 and value_counts.iloc[0] > 10:
                continue  # Skip imputation for likely categorical placeholders

            # Impute missing values
            missing_before = df[col].isna().sum()
            if missing_before > 0:
                group_col = find_best_grouping_column(df, col, False)
                if group_col:
                    df[col] = df.groupby(group_col)[col].transform(
                        lambda s: s.fillna(s.mode().iloc[0] if not s.mode().empty else "Unknown")
                    )
                df[col] = df[col].fillna(df[col].mode().iloc[0] if not df[col].mode().empty else "Unknown")
                operations["missing_values_imputed"] += missing_before - df[col].isna().sum()
        else:
            missing_before = df[col].isna().sum()
            if missing_before > 0:
                group_col = find_best_grouping_column(df, col, True)
                if group_col:
                    df[col] = df.groupby(group_col)[col].transform(
                        lambda s: s.fillna(s.median())
                    )
                df[col] = df[col].fillna(df[col].median())
                operations["missing_values_imputed"] += missing_before - df[col].isna().sum()

    # Harmonize categorical columns
    valid_values_hospital_type = [
        "acute care hospitals", "critical access hospitals",
        "children's hospitals", "psychiatric hospitals"
    ]
    if "HospitalType" in df.columns:
        df["HospitalType"] = df["HospitalType"].astype(str).apply(
            lambda v: harmonize_category(v, valid_values_hospital_type)
        )
        operations["categories_harmonized"] += 1

    valid_values_hospital_owner = [
        "voluntary non-profit - private", "proprietary",
        "government - federal", "government - state",
        "government - local", "voluntary non-profit - other"
    ]
    if "HospitalOwner" in df.columns:
        df["HospitalOwner"] = df["HospitalOwner"].astype(str).apply(
            lambda v: harmonize_category(v, valid_values_hospital_owner)
        )
        operations["categories_harmonized"] += 1

    valid_values_emergency = ["yes", "no"]
    if "EmergencyService" in df.columns:
        df["EmergencyService"] = df["EmergencyService"].astype(str).apply(
            lambda v: harmonize_category(v, valid_values_emergency)
        )
        operations["categories_harmonized"] += 1

    # Clean State (standardize to 2-letter uppercase)
    if "State" in df.columns:
        df["State"] = df["State"].astype(str).str.upper().str.strip()
        df["State"] = df["State"].apply(lambda x: x if len(x) == 2 else np.nan)
        missing_before = df["State"].isna().sum()
        if missing_before > 0:
            df["State"] = df["State"].fillna(df["State"].mode().iloc[0] if not df["State"].mode().empty else "Unknown")
            operations["missing_values_imputed"] += missing_before

    # Clean MeasureCode (standardize format)
    if "MeasureCode" in df.columns:
        df["MeasureCode"] = df["MeasureCode"].astype(str).str.strip().str.lower()
        df["MeasureCode"] = df["MeasureCode"].apply(
            lambda x: x if re.match(r"^[a-z-]+$", str(x)) else np.nan
        )
        missing_before = df["MeasureCode"].isna().sum()
        if missing_before > 0:
            df["MeasureCode"] = df["MeasureCode"].fillna(df["MeasureCode"].mode().iloc[0] if not df["MeasureCode"].mode().empty else "Unknown")
            operations["missing_values_imputed"] += missing_before

    # Clean City (standardize case)
    if "City" in df.columns:
        df["City"] = df["City"].astype(str).str.strip().str.title()

    # Clean Score outliers
    if "Score" in df.columns and df["Score"].dtype in ["int64", "float64"]:
        q_low = df["Score"].quantile(0.01)
        q_high = df["Score"].quantile(0.99)
        outliers = df[(df["Score"] < q_low) | (df["Score"] > q_high)]
        if not outliers.empty:
            median_score = df[(df["Score"] >= q_low) & (df["Score"] <= q_high)]["Score"].median()
            df.loc[(df["Score"] < q_low) | (df["Score"] > q_high), "Score"] = median_score
            operations["outliers_corrected"] += len(outliers)

    # Convert Score back to percentage format if it was originally percentage
    if "Score" in df.columns and score_is_percentage:
        df["Score"] = df["Score"].astype(str) + "%"

    # Save cleaned data
    df.to_csv(OUTPUT_PATH, index=False)

    # Print operations summary
    print("Data Cleaning Operations Summary:")
    print(f"- Rows before cleaning: {initial_rows}")
    print(f"- Rows after cleaning: {len(df)}")
    print(f"- Duplicates removed: {operations['duplicates_removed']}")
    print(f"- Missing values imputed: {operations['missing_values_imputed']}")
    print(f"- Numeric values extracted: {operations['numeric_values_extracted']}")
    print(f"- Categories harmonized: {operations['categories_harmonized']}")
    print(f"- Outliers corrected: {operations['outliers_corrected']}")
    print(f"Cleaned data saved to: {OUTPUT_PATH}")

if __name__ == "__main__":
    clean_hospital_data()