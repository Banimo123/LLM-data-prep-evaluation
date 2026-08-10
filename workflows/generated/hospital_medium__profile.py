import pandas as pd
import numpy as np
import re
from datetime import datetime
import difflib

INPUT_PATH = r'datasets\hospital\noisy_medium.csv'
OUTPUT_PATH = r'results\cleaned_datasets\hospital\noisy_medium__profile.csv'

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
    date_formats = ["%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%B %d, %Y", "%d-%b-%Y", "%Y/%m/%d"]
    for fmt in date_formats:
        try:
            return datetime.strptime(str(value), fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    try:
        return datetime.fromtimestamp(int(value)).strftime("%Y-%m-%d")
    except (ValueError, TypeError):
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

# Load data
df = pd.read_csv(INPUT_PATH)

# Track operations
operations = []

# Remove duplicates (keep first occurrence)
initial_rows = len(df)
df = df.drop_duplicates(subset=[c for c in df.columns if c != "row_id"], keep="first")
duplicates_removed = initial_rows - len(df)
if duplicates_removed > 0:
    operations.append(f"Removed {duplicates_removed} duplicate rows")

# Clean State column (15.9% missing, many variants)
if "State" in df.columns:
    valid_states = ["al", "ak", "la"]
    df["State"] = df["State"].apply(lambda v: harmonize_category(v, valid_states))
    state_missing = df["State"].isna().sum()
    if state_missing > 0:
        group_col = find_best_grouping_column(df, "State", False)
        if group_col:
            df["State"] = df.groupby(group_col)["State"].transform(
                lambda s: s.fillna(s.mode().iloc[0] if not s.mode().empty else "al")
            )
        df["State"] = df["State"].fillna("al")
        operations.append(f"Imputed {state_missing} missing values in State column")

# Clean City column (16.8% missing)
if "City" in df.columns:
    city_missing = df["City"].isna().sum()
    if city_missing > 0:
        group_col = find_best_grouping_column(df, "City", False)
        if group_col:
            df["City"] = df.groupby(group_col)["City"].transform(
                lambda s: s.fillna(s.mode().iloc[0] if not s.mode().empty else "birmingham")
            )
        df["City"] = df["City"].fillna("birmingham")
        operations.append(f"Imputed {city_missing} missing values in City column")

# Clean HospitalOwner column (17.1% missing)
if "HospitalOwner" in df.columns:
    valid_owners = [
        "voluntary non-profit - private", "proprietary", "government - hospital district or authority",
        "voluntary non-profit - other", "voluntary non-profit - church", "government - federal",
        "government - state", "government - local", "unknown"
    ]
    df["HospitalOwner"] = df["HospitalOwner"].apply(lambda v: harmonize_category(v, valid_owners))
    owner_missing = df["HospitalOwner"].isna().sum()
    if owner_missing > 0:
        group_col = find_best_grouping_column(df, "HospitalOwner", False)
        if group_col:
            df["HospitalOwner"] = df.groupby(group_col)["HospitalOwner"].transform(
                lambda s: s.fillna(s.mode().iloc[0] if not s.mode().empty else "voluntary non-profit - private")
            )
        df["HospitalOwner"] = df["HospitalOwner"].fillna("voluntary non-profit - private")
        operations.append(f"Imputed {owner_missing} missing values in HospitalOwner column")

# Clean EmergencyService column (16.1% missing)
if "EmergencyService" in df.columns:
    valid_emergency = ["yes", "no", "unknown"]
    df["EmergencyService"] = df["EmergencyService"].apply(lambda v: harmonize_category(v, valid_emergency))
    emergency_missing = df["EmergencyService"].isna().sum()
    if emergency_missing > 0:
        group_col = find_best_grouping_column(df, "EmergencyService", False)
        if group_col:
            df["EmergencyService"] = df.groupby(group_col)["EmergencyService"].transform(
                lambda s: s.fillna(s.mode().iloc[0] if not s.mode().empty else "yes")
            )
        df["EmergencyService"] = df["EmergencyService"].fillna("yes")
        operations.append(f"Imputed {emergency_missing} missing values in EmergencyService column")

# Clean PhoneNumber column (numeric with possible text)
if "PhoneNumber" in df.columns:
    df["PhoneNumber"] = df["PhoneNumber"].apply(extract_numeric)
    phone_outliers = df["PhoneNumber"][(df["PhoneNumber"] < 2000000000) | (df["PhoneNumber"] > 9999999999)]
    if not phone_outliers.empty:
        median_phone = df["PhoneNumber"].median()
        df.loc[phone_outliers.index, "PhoneNumber"] = median_phone
        operations.append(f"Corrected {len(phone_outliers)} outliers in PhoneNumber column")

# Clean ZipCode column (text with possible numeric corruption)
if "ZipCode" in df.columns:
    df["ZipCode"] = df["ZipCode"].astype(str)
    df["ZipCode"] = df["ZipCode"].str.extract(r'(\d{5})')[0]
    zip_missing = df["ZipCode"].isna().sum()
    if zip_missing > 0:
        df["ZipCode"] = df["ZipCode"].fillna("35233")
        operations.append(f"Imputed {zip_missing} missing values in ZipCode column")

# Clean Score column (text with percentages and 'empty')
if "Score" in df.columns:
    df["Score"] = df["Score"].astype(str)
    df["Score"] = df["Score"].replace("empty", np.nan)
    score_missing = df["Score"].isna().sum()
    if score_missing > 0:
        df["Score"] = df["Score"].fillna("100%")
        operations.append(f"Imputed {score_missing} missing values in Score column")

# Clean Sample column (text with 'empty' and numeric values)
if "Sample" in df.columns:
    df["Sample"] = df["Sample"].astype(str)
    df["Sample"] = df["Sample"].replace("empty", np.nan)
    sample_missing = df["Sample"].isna().sum()
    if sample_missing > 0:
        df["Sample"] = df["Sample"].fillna("0 patients")
        operations.append(f"Imputed {sample_missing} missing values in Sample column")

# Clean ProviderNumber (numeric with plausible range)
if "ProviderNumber" in df.columns:
    provider_outliers = df["ProviderNumber"][(df["ProviderNumber"] < 10000) | (df["ProviderNumber"] > 21000)]
    if not provider_outliers.empty:
        median_provider = df["ProviderNumber"].median()
        df.loc[provider_outliers.index, "ProviderNumber"] = median_provider
        operations.append(f"Corrected {len(provider_outliers)} outliers in ProviderNumber column")

# Save cleaned data
df.to_csv(OUTPUT_PATH, index=False)

# Print operations summary
print("Data cleaning operations performed:")
for op in operations:
    print(f"- {op}")
print(f"\nFinal dataset shape: {df.shape[0]} rows, {df.shape[1]} columns")