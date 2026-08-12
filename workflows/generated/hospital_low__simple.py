import pandas as pd
import numpy as np
import re
from datetime import datetime
import difflib

INPUT_PATH = r"datasets\hospital\noisy_low.csv"
OUTPUT_PATH = r"results\cleaned_datasets\hospital\noisy_low__simple.csv"

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
        "missing_values": {},
        "duplicates": 0,
        "numeric_corrections": {},
        "date_corrections": {},
        "category_harmonizations": {},
        "outliers": {}
    }

    # Remove duplicates based on all columns except row_id
    initial_rows = len(df)
    df = df.drop_duplicates(subset=[col for col in df.columns if col != "row_id"], keep="first")
    operations["duplicates"] = initial_rows - len(df)

    # Handle missing values and corrections
    for col in df.columns:
        if col == "row_id":
            continue

        # Check if column is numeric
        is_numeric = pd.api.types.is_numeric_dtype(df[col]) or (
            df[col].dropna().apply(lambda x: isinstance(x, (int, float))).any()
        )

        # Handle numeric columns with text parasites
        if is_numeric:
            initial_missing = df[col].isna().sum()
            df[col] = df[col].apply(extract_numeric)
            numeric_corrections = initial_missing - df[col].isna().sum()
            if numeric_corrections > 0:
                operations["numeric_corrections"][col] = numeric_corrections

            # Impute missing values
            missing_before = df[col].isna().sum()
            if missing_before > 0:
                group_col = find_best_grouping_column(df, col, True)
                if group_col:
                    df[col] = df.groupby(group_col)[col].transform(
                        lambda s: s.fillna(s.median())
                    )
                df[col] = df[col].fillna(df[col].median())
                operations["missing_values"][col] = missing_before - df[col].isna().sum()

            # Handle outliers for numeric columns
            if df[col].nunique() > 10:  # Only for continuous-like columns
                q1 = df[col].quantile(0.01)
                q99 = df[col].quantile(0.99)
                median = df[col].median()
                outliers = df[(df[col] < q1) | (df[col] > q99)]
                if not outliers.empty:
                    df.loc[outliers.index, col] = median
                    operations["outliers"][col] = len(outliers)

        # Handle categorical columns
        elif df[col].dtype == "object":
            # Check for date columns
            if df[col].str.match(r"\d{4}-\d{2}-\d{2}|\d{2}/\d{2}/\d{4}").any():
                initial_missing = df[col].isna().sum()
                df[col] = df[col].apply(parse_date)
                date_corrections = initial_missing - df[col].isna().sum()
                if date_corrections > 0:
                    operations["date_corrections"][col] = date_corrections

            # Handle specific categorical columns with known values
            if col == "HospitalType":
                valid_values = [
                    "acute care hospitals", "critical access hospitals",
                    "children's hospitals", "psychiatric hospitals"
                ]
                df[col] = df[col].apply(lambda v: harmonize_category(v, valid_values))
                operations["category_harmonizations"][col] = "HospitalType"

            elif col == "HospitalOwner":
                valid_values = [
                    "voluntary non-profit - private", "proprietary",
                    "voluntary non-profit - other", "government - hospital district or authority",
                    "government - federal", "government - state", "government - local"
                ]
                df[col] = df[col].apply(lambda v: harmonize_category(v, valid_values))
                operations["category_harmonizations"][col] = "HospitalOwner"

            elif col == "EmergencyService":
                valid_values = ["yes", "no"]
                df[col] = df[col].apply(lambda v: harmonize_category(v, valid_values))
                operations["category_harmonizations"][col] = "EmergencyService"

            # Impute missing values for categorical columns
            missing_before = df[col].isna().sum()
            if missing_before > 0:
                group_col = find_best_grouping_column(df, col, False)
                if group_col:
                    df[col] = df.groupby(group_col)[col].transform(
                        lambda s: s.fillna(s.mode().iloc[0] if not s.mode().empty else "Unknown")
                    )
                df[col] = df[col].fillna(df[col].mode().iloc[0])
                operations["missing_values"][col] = missing_before - df[col].isna().sum()

    # Save cleaned dataset
    df.to_csv(OUTPUT_PATH, index=False)

    # Print summary
    print("=== Data Cleaning Summary ===")
    print(f"Initial rows: {initial_rows}")
    print(f"Rows after duplicate removal: {len(df)} (removed {operations['duplicates']})")

    if operations["missing_values"]:
        print("\nMissing values imputed:")
        for col, count in operations["missing_values"].items():
            print(f"  - {col}: {count} values")

    if operations["numeric_corrections"]:
        print("\nNumeric values extracted from text:")
        for col, count in operations["numeric_corrections"].items():
            print(f"  - {col}: {count} values")

    if operations["date_corrections"]:
        print("\nDate values corrected:")
        for col, count in operations["date_corrections"].items():
            print(f"  - {col}: {count} values")

    if operations["category_harmonizations"]:
        print("\nCategory harmonizations applied:")
        for col, _ in operations["category_harmonizations"].items():
            print(f"  - {col}")

    if operations["outliers"]:
        print("\nOutliers corrected:")
        for col, count in operations["outliers"].items():
            print(f"  - {col}: {count} values")

    print(f"\nCleaned dataset saved to: {OUTPUT_PATH}")

if __name__ == "__main__":
    clean_dataset()