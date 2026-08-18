import pandas as pd
import numpy as np
import re
from datetime import datetime

INPUT_PATH = "datasets/titanic/noisy_high.csv"
OUTPUT_PATH = "results/cleaned_datasets/titanic/noisy_high__simple.csv"

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
    s = str(value).strip()
    for fmt in formats:
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

def clean_dataset():
    df = pd.read_csv(INPUT_PATH)

    operations = {
        "missing_values": 0,
        "duplicates_removed": 0,
        "numeric_corrected": 0,
        "categorical_harmonized": 0,
        "date_parsed": 0,
        "outliers_corrected": 0
    }

    # Remove Unnamed columns if they exist (except row_id)
    unnamed_cols = [col for col in df.columns if col.startswith("Unnamed:")]
    df = df.drop(columns=unnamed_cols, errors="ignore")

    # Remove duplicates based on all columns except row_id
    initial_rows = len(df)
    df = df.drop_duplicates(subset=[col for col in df.columns if col != "row_id"])
    operations["duplicates_removed"] = initial_rows - len(df)

    # Process numeric columns with text parasites
    numeric_cols = ["Age", "Fare", "Family_size"]
    for col in numeric_cols:
        if col in df.columns:
            initial_missing = df[col].isna().sum()
            df[col] = df[col].apply(extract_numeric)
            operations["numeric_corrected"] += (df[col].notna().sum() - (len(df) - initial_missing))

    # Process categorical columns
    categorical_cols = {
        "Sex": ["0", "1"],
        "Pclass_1": ["0", "1"],
        "Pclass_2": ["0", "1"],
        "Pclass_3": ["0", "1"],
        "Title_1": ["0", "1"],
        "Title_2": ["0", "1"],
        "Title_3": ["0", "1"],
        "Title_4": ["0", "1"],
        "Emb_1": ["0", "1"],
        "Emb_2": ["0", "1"],
        "Emb_3": ["0", "1"]
    }

    for col, valid_values in categorical_cols.items():
        if col in df.columns:
            df[col] = df[col].apply(lambda v: harmonize_category(v, valid_values))
            operations["categorical_harmonized"] += (df[col].isin(valid_values).sum() - df[col].notna().sum())

    # Impute missing values
    for col in df.columns:
        if col in ["row_id"]:
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
                    lambda s: s.fillna(s.mode().iloc[0] if not is_numeric else s.median())
                )
            if df[col].isna().any():
                if is_numeric:
                    df[col] = df[col].fillna(df[col].median())
                else:
                    df[col] = df[col].fillna(df[col].mode().iloc[0])
            operations["missing_values"] += (initial_missing - df[col].isna().sum())

    # Correct outliers in numeric columns
    for col in ["Age", "Fare", "Family_size"]:
        if col in df.columns:
            q_low = df[col].quantile(0.01)
            q_high = df[col].quantile(0.99)
            median = df[col].median()
            mask = (df[col] < q_low) | (df[col] > q_high)
            if mask.any():
                operations["outliers_corrected"] += mask.sum()
                df.loc[mask, col] = median

    # Ensure row_id is preserved and not modified
    if "row_id" in df.columns:
        df = df.set_index("row_id", drop=False)

    # Save cleaned dataset
    df.to_csv(OUTPUT_PATH, index=False)

    # Print summary
    print("Data Cleaning Summary:")
    print(f"- Initial rows: {initial_rows}")
    print(f"- Rows after duplicate removal: {len(df)}")
    print(f"- Missing values imputed: {operations['missing_values']}")
    print(f"- Numeric values corrected: {operations['numeric_corrected']}")
    print(f"- Categorical values harmonized: {operations['categorical_harmonized']}")
    print(f"- Outliers corrected: {operations['outliers_corrected']}")
    print(f"Cleaned dataset saved to: {OUTPUT_PATH}")

if __name__ == "__main__":
    clean_dataset()