import pandas as pd
import numpy as np
import re
from datetime import datetime
import difflib

INPUT_PATH = "datasets/titanic/noisy_low.csv"
OUTPUT_PATH = "results/cleaned_datasets/titanic/noisy_low__simple.csv"

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
    try:
        if len(s) in (9, 10) and s.isdigit():
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
        "numeric_values_extracted": 0,
        "categories_harmonized": 0,
        "dates_parsed": 0,
        "outliers_corrected": 0
    }

    # Remove Unnamed columns if they exist
    unnamed_cols = [col for col in df.columns if col.startswith("Unnamed")]
    df = df.drop(columns=unnamed_cols, errors="ignore")

    # Remove duplicates based on row_id
    initial_rows = len(df)
    df = df.drop_duplicates(subset=["row_id"], keep="first")
    operations["duplicates_removed"] = initial_rows - len(df)

    # Process each column
    for col in df.columns:
        if col == "row_id":
            continue

        # Skip if column is empty
        if df[col].isna().all():
            continue

        # Check if column is numeric
        is_numeric = pd.api.types.is_numeric_dtype(df[col]) or df[col].apply(lambda x: isinstance(x, (int, float))).all()

        # Extract numeric values if column contains text
        if not is_numeric and df[col].dtype == "object":
            numeric_count = df[col].apply(lambda x: isinstance(extract_numeric(x), float)).sum()
            if numeric_count / len(df) > 0.5:
                df[col] = df[col].apply(extract_numeric)
                operations["numeric_values_extracted"] += numeric_count

        # Handle missing values
        missing_count = df[col].isna().sum()
        if missing_count > 0:
            is_col_numeric = pd.api.types.is_numeric_dtype(df[col])
            group_col = find_best_grouping_column(df, col, is_col_numeric)

            if group_col:
                if is_col_numeric:
                    df[col] = df.groupby(group_col)[col].transform(
                        lambda s: s.fillna(s.median())
                    )
                else:
                    df[col] = df.groupby(group_col)[col].transform(
                        lambda s: s.fillna(s.mode()[0] if not s.mode().empty else "Unknown")
                    )
            else:
                if is_col_numeric:
                    df[col] = df[col].fillna(df[col].median())
                else:
                    df[col] = df[col].fillna(df[col].mode()[0])

            operations["missing_values_imputed"] += missing_count

        # Handle categorical columns
        if df[col].dtype == "object" and df[col].nunique() <= 10:
            frequent_values = df[col].value_counts().nlargest(5).index.tolist()
            if len(frequent_values) > 1:
                df[col] = df[col].apply(lambda v: harmonize_category(v, frequent_values))
                operations["categories_harmonized"] += 1

        # Handle numeric outliers
        if pd.api.types.is_numeric_dtype(df[col]):
            if df[col].nunique() > 10:  # Only for continuous numeric columns
                q1 = df[col].quantile(0.01)
                q99 = df[col].quantile(0.99)
                median = df[col].median()
                mode = df[col].mode()[0]

                # Check if most values are identical (discrete column)
                if df[col].value_counts(normalize=True).iloc[0] > 0.9:
                    threshold = q99
                else:
                    threshold = q99

                outliers = df[col][(df[col] < q1) | (df[col] > threshold)]
                if len(outliers) > 0:
                    if df[col].nunique() <= 5:  # Discrete column
                        df.loc[outliers.index, col] = mode
                    else:
                        df.loc[outliers.index, col] = median
                    operations["outliers_corrected"] += len(outliers)

    # Save cleaned dataset
    df.to_csv(OUTPUT_PATH, index=False)

    # Print operations summary
    print("Data Cleaning Operations Summary:")
    print(f"- Rows after removing duplicates: {len(df)}")
    print(f"- Missing values imputed: {operations['missing_values_imputed']}")
    print(f"- Numeric values extracted from text: {operations['numeric_values_extracted']}")
    print(f"- Categorical values harmonized: {operations['categories_harmonized']}")
    print(f"- Outliers corrected: {operations['outliers_corrected']}")
    print(f"Cleaned dataset saved to: {OUTPUT_PATH}")

if __name__ == "__main__":
    clean_dataset()