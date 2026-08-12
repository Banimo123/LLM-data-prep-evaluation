import pandas as pd
import numpy as np
import re
from datetime import datetime
import difflib

INPUT_PATH = r"datasets\hospital\noisy_high.csv"
OUTPUT_PATH = r"results\cleaned_datasets\hospital\noisy_high__simple.csv"

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
            return datetime.strptime(s, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    if s.isdigit() and len(s) in (9, 10):
        try:
            return datetime.fromtimestamp(int(s)).strftime("%Y-%m-%d")
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

    operations_log = {
        "missing_values_imputed": {},
        "duplicates_removed": 0,
        "numeric_extracted": {},
        "categories_harmonized": {},
        "dates_parsed": 0,
        "outliers_corrected": {}
    }

    # Handle duplicates
    initial_rows = len(df)
    df = df.drop_duplicates(subset=[col for col in df.columns if col != "row_id"], keep="first")
    operations_log["duplicates_removed"] = initial_rows - len(df)

    # Standardize State column
    if "State" in df.columns:
        valid_states = ["AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA",
                        "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD",
                        "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ",
                        "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC",
                        "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY"]
        df["State"] = df["State"].apply(lambda v: harmonize_category(v, valid_states))
        operations_log["categories_harmonized"]["State"] = len(df[df["State"].isin(valid_states)])

    # Clean ZipCode
    if "ZipCode" in df.columns:
        df["ZipCode"] = df["ZipCode"].apply(extract_numeric)
        df["ZipCode"] = df["ZipCode"].apply(lambda x: f"{int(x):05d}" if pd.notna(x) else x)
        operations_log["numeric_extracted"]["ZipCode"] = len(df[df["ZipCode"].notna()])

    # Clean PhoneNumber
    if "PhoneNumber" in df.columns:
        df["PhoneNumber"] = df["PhoneNumber"].apply(extract_numeric)
        df["PhoneNumber"] = df["PhoneNumber"].apply(lambda x: f"{int(x):010d}" if pd.notna(x) and x >= 1e9 else x)
        operations_log["numeric_extracted"]["PhoneNumber"] = len(df[df["PhoneNumber"].notna()])

    # Harmonize HospitalType
    if "HospitalType" in df.columns:
        valid_types = [
            "acute care hospitals", "critical access hospitals",
            "children's hospitals", "psychiatric hospitals",
            "rehabilitation hospitals", "long term care hospitals"
        ]
        df["HospitalType"] = df["HospitalType"].apply(lambda v: harmonize_category(v, valid_types))
        operations_log["categories_harmonized"]["HospitalType"] = len(df[df["HospitalType"].isin(valid_types)])

    # Harmonize HospitalOwner
    if "HospitalOwner" in df.columns:
        valid_owners = [
            "voluntary non-profit - private", "proprietary",
            "government - federal", "government - state",
            "government - local", "voluntary non-profit - other",
            "church operated", "tribal"
        ]
        df["HospitalOwner"] = df["HospitalOwner"].apply(lambda v: harmonize_category(v, valid_owners))
        operations_log["categories_harmonized"]["HospitalOwner"] = len(df[df["HospitalOwner"].isin(valid_owners)])

    # Clean EmergencyService
    if "EmergencyService" in df.columns:
        valid_emergency = ["yes", "no", "unknown"]
        df["EmergencyService"] = df["EmergencyService"].apply(lambda v: harmonize_category(v, valid_emergency))
        operations_log["categories_harmonized"]["EmergencyService"] = len(df[df["EmergencyService"].isin(valid_emergency)])

    # Clean City
    if "City" in df.columns:
        df["City"] = df["City"].str.strip().str.title()
        df["City"] = df["City"].replace({"Birmiingham": "Birmingham"})

    # Impute missing values
    for col in df.columns:
        if col in ["row_id", "index"]:
            continue

        if pd.api.types.is_numeric_dtype(df[col]):
            is_numeric = True
        else:
            is_numeric = False

        if df[col].isna().any():
            initial_missing = df[col].isna().sum()
            group_col = find_best_grouping_column(df, col, is_numeric)

            if group_col:
                df[col] = df.groupby(group_col)[col].transform(
                    lambda s: s.fillna(s.mode().iloc[0] if not s.mode().empty else s.median() if is_numeric else "Unknown")
                )
            if df[col].isna().any():
                if is_numeric:
                    df[col] = df[col].fillna(df[col].median())
                else:
                    df[col] = df[col].fillna(df[col].mode().iloc[0] if not df[col].mode().empty else "Unknown")

            operations_log["missing_values_imputed"][col] = initial_missing - df[col].isna().sum()

    # Handle outliers for numeric columns
    numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col]) and col not in ["row_id", "index"]]
    for col in numeric_cols:
        if col in ["ProviderNumber", "ZipCode", "PhoneNumber"]:
            continue

        q1 = df[col].quantile(0.01)
        q99 = df[col].quantile(0.99)
        median = df[col].median()
        mode = df[col].mode().iloc[0] if not df[col].mode().empty else median

        if df[col].nunique() <= 5:
            replacement = mode
        else:
            replacement = median

        outliers = (df[col] < q1) | (df[col] > q99)
        if outliers.any():
            initial_outliers = outliers.sum()
            df.loc[outliers, col] = replacement
            operations_log["outliers_corrected"][col] = initial_outliers

    # Clean Address columns
    for addr_col in ["Address1", "Address2", "Address3"]:
        if addr_col in df.columns:
            df[addr_col] = df[addr_col].replace({"empty": np.nan, "": np.nan})
            df[addr_col] = df[addr_col].str.strip().str.title()

    # Clean CountyName
    if "CountyName" in df.columns:
        df["CountyName"] = df["CountyName"].str.strip().str.title()

    # Clean MeasureCode
    if "MeasureCode" in df.columns:
        df["MeasureCode"] = df["MeasureCode"].str.strip().str.upper()

    # Clean Score column
    if "Score" in df.columns:
        df["Score"] = df["Score"].apply(extract_numeric)
        operations_log["numeric_extracted"]["Score"] = len(df[df["Score"].notna()])

    # Save cleaned dataset
    df.to_csv(OUTPUT_PATH, index=False)

    # Print operations log
    print("Data Cleaning Operations Summary:")
    print(f"- Duplicates removed: {operations_log['duplicates_removed']}")
    print("- Missing values imputed:")
    for col, count in operations_log["missing_values_imputed"].items():
        print(f"  - {col}: {count}")
    print("- Numeric values extracted:")
    for col, count in operations_log["numeric_extracted"].items():
        print(f"  - {col}: {count}")
    print("- Categories harmonized:")
    for col, count in operations_log["categories_harmonized"].items():
        print(f"  - {col}: {count}")
    print("- Outliers corrected:")
    for col, count in operations_log["outliers_corrected"].items():
        print(f"  - {col}: {count}")

if __name__ == "__main__":
    clean_dataset()