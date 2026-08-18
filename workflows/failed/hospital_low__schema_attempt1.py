import pandas as pd
import numpy as np
import re
from datetime import datetime

INPUT_PATH = r"datasets\hospital\noisy_low.csv"
OUTPUT_PATH = r"results\cleaned_datasets\hospital\noisy_low__schema.csv"

def extract_numeric(value):
    if pd.isna(value):
        return np.nan
    s = str(value).replace(",", ".")
    s = s.replace("O", "0").replace("o", "0")
    match = re.search(r"-?\d+\.?\d*", s)
    return float(match.group()) if match else np.nan

def _levenshtein(a, b):
    if a == b:
        return 0
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            curr[j] = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + (ca != cb))
        prev = curr
    return prev[-1]

def harmonize_category(value, valid_values):
    if pd.isna(value):
        return value
    s = str(value).strip()
    if s in valid_values:
        return s
    for v in valid_values:
        if v.lower() == s.lower():
            return v
    scored = sorted(((_levenshtein(s, v), v) for v in valid_values), key=lambda x: x[0])
    if not scored:
        return value
    best_dist, best_val = scored[0]
    max_allowed = max(1, len(best_val) // 4)
    if best_dist > max_allowed:
        return value
    if len(scored) > 1 and scored[1][0] == best_dist:
        return value
    return best_val

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

def parse_date(value):
    if pd.isna(value):
        return value
    formats = ["%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%B %d, %Y", "%d-%b-%Y", "%Y/%m/%d"]
    for fmt in formats:
        try:
            return datetime.strptime(str(value), fmt).strftime("%Y-%m-%d")
        except (ValueError, TypeError):
            continue
    try:
        return datetime.fromtimestamp(int(value)).strftime("%Y-%m-%d")
    except (ValueError, TypeError):
        return value

def clean_dataset():
    df = pd.read_csv(INPUT_PATH)

    operations = {
        'missing_values': {},
        'duplicates_removed': 0,
        'numeric_corrections': {},
        'category_harmonizations': {},
        'outliers_corrected': {}
    }

    # Remove duplicates keeping first occurrence
    initial_rows = len(df)
    df = df.drop_duplicates(subset=[col for col in df.columns if col != "row_id"], keep='first')
    operations['duplicates_removed'] = initial_rows - len(df)

    # ProviderNumber - clean numeric extraction
    if 'ProviderNumber' in df.columns:
        df['ProviderNumber'] = df['ProviderNumber'].apply(extract_numeric)
        df['ProviderNumber'] = df['ProviderNumber'].fillna(df['ProviderNumber'].median()).astype(int)
        operations['numeric_corrections']['ProviderNumber'] = "Extracted numeric values"

    # HospitalName - clean text
    if 'HospitalName' in df.columns:
        df['HospitalName'] = df['HospitalName'].str.strip()
        df['HospitalName'] = df['HospitalName'].str.lower()
        df['HospitalName'] = df['HospitalName'].str.replace(r'[^a-z0-9\s]', '', regex=True)
        df['HospitalName'] = df['HospitalName'].str.title()

    # Address1, Address2, Address3 - clean text
    for col in ['Address1', 'Address2', 'Address3']:
        if col in df.columns:
            df[col] = df[col].replace('empty', np.nan)
            df[col] = df[col].str.strip()
            df[col] = df[col].fillna('')

    # City - harmonize categories
    if 'City' in df.columns:
        valid_cities = ["birmingham", "huntsville", "mobile", "montgomery", "tuscaloosa"]
        df['City'] = df['City'].str.lower().str.strip()
        df['City'] = df['City'].apply(lambda v: harmonize_category(v, valid_cities))
        operations['category_harmonizations']['City'] = f"Harmonized to {valid_cities}"

    # State - harmonize to valid state codes
    if 'State' in df.columns:
        valid_states = ["al", "ak", "az", "ar", "ca", "co", "ct", "de", "fl", "ga",
                        "hi", "id", "il", "in", "ia", "ks", "ky", "la", "me", "md",
                        "ma", "mi", "mn", "ms", "mo", "mt", "ne", "nv", "nh", "nj",
                        "nm", "ny", "nc", "nd", "oh", "ok", "or", "pa", "ri", "sc",
                        "sd", "tn", "tx", "ut", "vt", "va", "wa", "wv", "wi", "wy"]
        df['State'] = df['State'].str.lower().str.strip()
        df['State'] = df['State'].apply(lambda v: harmonize_category(v, valid_states))
        operations['category_harmonizations']['State'] = "Harmonized to valid state codes"

    # ZipCode - clean numeric extraction
    if 'ZipCode' in df.columns:
        df['ZipCode'] = df['ZipCode'].apply(extract_numeric)
        df['ZipCode'] = df['ZipCode'].fillna(df['ZipCode'].median()).astype(int)
        operations['numeric_corrections']['ZipCode'] = "Extracted numeric values"

    # CountyName - harmonize categories
    if 'CountyName' in df.columns:
        df['CountyName'] = df['CountyName'].str.lower().str.strip()
        df['CountyName'] = df['CountyName'].str.title()

    # PhoneNumber - clean numeric extraction
    if 'PhoneNumber' in df.columns:
        df['PhoneNumber'] = df['PhoneNumber'].apply(extract_numeric)
        df['PhoneNumber'] = df['PhoneNumber'].fillna(df['PhoneNumber'].median()).astype(int)
        operations['numeric_corrections']['PhoneNumber'] = "Extracted numeric values"

    # HospitalType - harmonize categories
    if 'HospitalType' in df.columns:
        valid_types = ["acute care hospitals", "critical access hospitals", "children's hospitals"]
        df['HospitalType'] = df['HospitalType'].str.lower().str.strip()
        df['HospitalType'] = df['HospitalType'].apply(lambda v: harmonize_category(v, valid_types))
        operations['category_harmonizations']['HospitalType'] = f"Harmonized to {valid_types}"

    # HospitalOwner - harmonize categories
    if 'HospitalOwner' in df.columns:
        valid_owners = [
            "voluntary non-profit - private",
            "government - hospital district or authority",
            "proprietary",
            "government - federal",
            "voluntary non-profit - church",
            "government - local",
            "voluntary non-profit - other"
        ]
        df['HospitalOwner'] = df['HospitalOwner'].str.lower().str.strip()
        df['HospitalOwner'] = df['HospitalOwner'].apply(lambda v: harmonize_category(v, valid_owners))
        operations['category_harmonizations']['HospitalOwner'] = f"Harmonized to {valid_owners}"

    # EmergencyService - harmonize to yes/no
    if 'EmergencyService' in df.columns:
        valid_emergency = ["yes", "no"]
        df['EmergencyService'] = df['EmergencyService'].str.lower().str.strip()
        df['EmergencyService'] = df['EmergencyService'].apply(lambda v: harmonize_category(v, valid_emergency))
        operations['category_harmonizations']['EmergencyService'] = "Harmonized to yes/no"

    # Condition - clean text
    if 'Condition' in df.columns:
        df['Condition'] = df['Condition'].str.lower().str.strip()
        df['Condition'] = df['Condition'].str.replace(r'[^a-z\s]', '', regex=True)
        df['Condition'] = df['Condition'].str.title()

    # MeasureCode - clean text
    if 'MeasureCode' in df.columns:
        df['MeasureCode'] = df['MeasureCode'].str.lower().str.strip()
        df['MeasureCode'] = df['MeasureCode'].str.replace(r'[^a-z0-9-]', '', regex=True)

    # MeasureName - clean text
    if 'MeasureName' in df.columns:
        df['MeasureName'] = df['MeasureName'].str.lower().str.strip()
        df['MeasureName'] = df['MeasureName'].str.replace(r'[^a-z0-9\s]', '', regex=True)
        df['MeasureName'] = df['MeasureName'].str.title()

    # Score - extract percentage
    if 'Score' in df.columns:
        def extract_percentage(value):
            if pd.isna(value) or value == 'empty':
                return np.nan
            match = re.search(r'(\d+)%', str(value))
            if match:
                return f"{match.group(1)}%"
            return value

        df['Score'] = df['Score'].apply(extract_percentage)
        operations['numeric_corrections']['Score'] = "Extracted percentage values"

    # Sample - extract numeric with unit
    if 'Sample' in df.columns:
        def extract_sample(value):
            if pd.isna(value) or value == 'empty':
                return np.nan
            match = re.search(r'(\d+)\s*patients?', str(value), re.IGNORECASE)
            if match:
                return f"{match.group(1)} patients"
            return value

        df['Sample'] = df['Sample'].apply(extract_sample)
        operations['numeric_corrections']['Sample'] = "Extracted sample values with unit"

    # Stateavg - clean text
    if 'Stateavg' in df.columns:
        df['Stateavg'] = df['Stateavg'].str.lower().str.strip()
        df['Stateavg'] = df['Stateavg'].str.replace(r'[^a-z0-9_-]', '', regex=True)

    # Impute missing values
    for col in df.columns:
        if col == "row_id":
            continue

        if df[col].isna().any():
            is_numeric = pd.api.types.is_numeric_dtype(df[col]) or df[col].apply(lambda x: isinstance(x, (int, float))).any()

            if col in ['Score', 'Sample']:
                is_numeric = False  # These are formatted strings with units

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
                    df[col] = df[col].fillna(df[col].mode().iloc[0] if not df[col].mode().empty else "Unknown")
                operations['missing_values'][col] = df[col].isna().sum()

    # Final cleaning for specific columns
    for col in ['HospitalName', 'City', 'State', 'CountyName', 'HospitalType', 'HospitalOwner', 'EmergencyService']:
        if col in df.columns:
            df[col] = df[col].str.strip()
            df[col] = df[col].replace('', np.nan)
            if df[col].isna().any():
                df[col] = df[col].fillna(df[col].mode().iloc[0])

    # Save cleaned dataset
    df.to_csv(OUTPUT_PATH, index=False)

    # Print operations summary
    print("Data Cleaning Operations Summary:")
    print(f"- Initial rows: {initial_rows}")
    print(f"- Duplicates removed: {operations['duplicates_removed']}")
    print(f"- Final rows: {len(df)}")
    print("\nMissing values imputed:")
    for col, count in operations['missing_values'].items():
        print(f"  - {col}: {count} values")
    print("\nNumeric corrections applied:")
    for col, desc in operations['numeric_corrections'].items():
        print(f"  - {col}: {desc}")
    print("\nCategory harmonizations applied:")
    for col, desc in operations['category_harmonizations'].items():
        print(f"  - {col}: {desc}")

if __name__ == "__main__":
    clean_dataset()