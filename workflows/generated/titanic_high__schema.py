import pandas as pd
import numpy as np
import re
from datetime import datetime

INPUT_PATH = "datasets/titanic/noisy_high.csv"
OUTPUT_PATH = "results/cleaned_datasets/titanic/noisy_high__schema.csv"

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

def clean_dataset():
    df = pd.read_csv(INPUT_PATH)

    operations = {
        'missing_values': {},
        'duplicates': 0,
        'numeric_corrections': {},
        'categorical_harmonizations': {},
        'outliers': {}
    }

    # Remove duplicates based on all columns except row_id
    initial_rows = len(df)
    df = df.drop_duplicates(subset=[col for col in df.columns if col != "row_id"])
    operations['duplicates'] = initial_rows - len(df)

    # Process each column
    for col in df.columns:
        if col == "row_id":
            continue

        # Handle numeric columns
        if pd.api.types.is_numeric_dtype(df[col]):
            # Check for text corruption in numeric columns
            non_numeric = df[col].apply(lambda x: isinstance(x, str) and not str(x).replace('.', '').replace('-', '').isdigit())
            if non_numeric.any():
                operations['numeric_corrections'][col] = non_numeric.sum()
                df[col] = df[col].apply(extract_numeric)

            # Handle missing values
            if df[col].isna().any():
                is_discrete = df[col].nunique() <= 10
                group_col = find_best_grouping_column(df, col, True)
                if group_col:
                    df[col] = df.groupby(group_col)[col].transform(
                        lambda s: s.fillna(s.mode().iloc[0] if is_discrete else s.median())
                    )
                df[col] = df[col].fillna(df[col].mode().iloc[0] if is_discrete else df[col].median())
                operations['missing_values'][col] = df[col].isna().sum()

            # Handle outliers for numeric columns
            if col not in ['Unnamed: 0', 'PassengerId', 'Survived', 'Sex'] and df[col].nunique() > 2:
                q1 = df[col].quantile(0.01)
                q99 = df[col].quantile(0.99)
                median = df[col].median()
                outliers = (df[col] < q1) | (df[col] > q99)
                if outliers.any():
                    is_discrete = df[col].nunique() <= 10
                    df.loc[outliers, col] = df[col].mode().iloc[0] if is_discrete else median
                    operations['outliers'][col] = outliers.sum()

        # Handle categorical/text columns
        else:
            # Handle Age and Fare which are stored as text but should be numeric
            if col in ['Age', 'Fare']:
                operations['numeric_corrections'][col] = df[col].apply(lambda x: not str(x).replace('.', '').replace('-', '').isdigit()).sum()
                df[col] = df[col].apply(extract_numeric)

                # Handle missing values after conversion
                if df[col].isna().any():
                    is_discrete = df[col].nunique() <= 10
                    group_col = find_best_grouping_column(df, col, True)
                    if group_col:
                        df[col] = df.groupby(group_col)[col].transform(
                            lambda s: s.fillna(s.mode().iloc[0] if is_discrete else s.median())
                        )
                    df[col] = df[col].fillna(df[col].mode().iloc[0] if is_discrete else df[col].median())
                    operations['missing_values'][col] = df[col].isna().sum()

                # Handle outliers
                q1 = df[col].quantile(0.01)
                q99 = df[col].quantile(0.99)
                median = df[col].median()
                outliers = (df[col] < q1) | (df[col] > q99)
                if outliers.any():
                    is_discrete = df[col].nunique() <= 10
                    df.loc[outliers, col] = df[col].mode().iloc[0] if is_discrete else median
                    operations['outliers'][col] = outliers.sum()

            # For other categorical columns (none in this dataset)
            else:
                if df[col].isna().any():
                    group_col = find_best_grouping_column(df, col, False)
                    if group_col:
                        df[col] = df.groupby(group_col)[col].transform(
                            lambda s: s.fillna(s.mode().iloc[0] if not s.mode().empty else "Unknown")
                        )
                    df[col] = df[col].fillna(df[col].mode().iloc[0])
                    operations['missing_values'][col] = df[col].isna().sum()

    # Save cleaned dataset
    df.to_csv(OUTPUT_PATH, index=False)

    # Print operations summary
    print("Data Cleaning Operations Summary:")
    print(f"- Rows removed (duplicates): {operations['duplicates']}")
    print("- Missing values imputed:")
    for col, count in operations['missing_values'].items():
        print(f"  - {col}: {count}")
    print("- Numeric corrections applied:")
    for col, count in operations['numeric_corrections'].items():
        print(f"  - {col}: {count}")
    print("- Outliers corrected:")
    for col, count in operations['outliers'].items():
        print(f"  - {col}: {count}")
    print(f"- Final dataset shape: {df.shape}")

if __name__ == "__main__":
    clean_dataset()