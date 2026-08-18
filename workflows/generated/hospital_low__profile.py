import pandas as pd
import numpy as np
import re
from datetime import datetime

INPUT_PATH = r"datasets\hospital\noisy_low.csv"
OUTPUT_PATH = r"results\cleaned_datasets\hospital\noisy_low__profile.csv"

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
        except ValueError:
            continue
    if str(value).isdigit() and len(str(value)) in (9, 10):
        try:
            return datetime.fromtimestamp(int(value)).strftime("%Y-%m-%d")
        except (ValueError, OSError):
            pass
    return value

def clean_dataset():
    df = pd.read_csv(INPUT_PATH)

    operations = []

    # Suppression des doublons (conservation de la première occurrence)
    initial_rows = len(df)
    df = df.drop_duplicates(subset=[c for c in df.columns if c != "row_id"], keep="first")
    duplicates_removed = initial_rows - len(df)
    if duplicates_removed > 0:
        operations.append(f"Doublons supprimés: {duplicates_removed}")

    # Nettoyage colonne par colonne
    if "State" in df.columns:
        valid_states = ["al", "ak", "xl", "ax"]
        df["State"] = df["State"].apply(lambda v: harmonize_category(v, valid_states))

    if "HospitalType" in df.columns:
        valid_hospital_types = ["acute care hospitals"]
        df["HospitalType"] = df["HospitalType"].apply(lambda v: harmonize_category(v, valid_hospital_types))

    if "HospitalOwner" in df.columns:
        valid_owners = [
            "voluntary non-profit - private", "proprietary", "government - hospital district or authority",
            "voluntary non-profit - other", "voluntary non-profit - church", "government - federal",
            "government - state", "government - local"
        ]
        df["HospitalOwner"] = df["HospitalOwner"].apply(lambda v: harmonize_category(v, valid_owners))

    if "EmergencyService" in df.columns:
        valid_emergency = ["yes", "no"]
        df["EmergencyService"] = df["EmergencyService"].apply(lambda v: harmonize_category(v, valid_emergency))

    if "Condition" in df.columns:
        valid_conditions = [
            "surgical infection prevention", "heart attack", "pneumonia", "heart failure",
            "children s asthma care", "heart attaxk"
        ]
        df["Condition"] = df["Condition"].apply(lambda v: harmonize_category(v, valid_conditions))

    if "City" in df.columns:
        df["City"] = df["City"].str.strip()
        df["City"] = df["City"].str.lower()
        common_cities = ["birmingham", "gadsden", "montgomery", "dothan", "huntsville", "anniston"]
        df["City"] = df["City"].apply(lambda v: v if v in common_cities else v)

    if "Score" in df.columns:
        df["Score"] = df["Score"].apply(lambda x: x if x == "empty" else x)
        score_not_empty = df[df["Score"] != "empty"]["Score"]
        if not score_not_empty.empty:
            df["Score"] = df["Score"].apply(
                lambda x: x if x == "empty" else (
                    x if x.endswith("%") else (
                        f"{int(extract_numeric(x))}%" if not pd.isna(extract_numeric(x)) else x
                    )
                )
            )

    if "Sample" in df.columns:
        df["Sample"] = df["Sample"].apply(lambda x: x if x == "empty" else x)
        sample_not_empty = df[df["Sample"] != "empty"]["Sample"]
        if not sample_not_empty.empty:
            df["Sample"] = df["Sample"].apply(
                lambda x: x if x == "empty" else (
                    x if "patients" in x else (
                        f"{int(extract_numeric(x))} patients" if not pd.isna(extract_numeric(x)) else x
                    )
                )
            )

    # Imputation des valeurs manquantes
    for col in df.columns:
        if col in ["row_id", "index"]:
            continue
        if df[col].isna().any():
            is_numeric = pd.api.types.is_numeric_dtype(df[col])
            if col in ["Score", "Sample"]:
                is_numeric = False
            group_col = find_best_grouping_column(df, col, is_numeric)
            if group_col:
                df[col] = df.groupby(group_col)[col].transform(
                    lambda s: s.fillna(s.mode().iloc[0] if not s.mode().empty else s.median() if is_numeric else "Unknown")
                )
            if df[col].isna().any():
                if is_numeric:
                    df[col] = df[col].fillna(df[col].median())
                else:
                    df[col] = df[col].fillna(df[col].mode().iloc[0])
            operations.append(f"Valeurs manquantes imputées dans {col}: {df[col].isna().sum()}")

    # Correction des valeurs aberrantes pour les colonnes numériques
    if "index" in df.columns:
        q1 = df["index"].quantile(0.01)
        q99 = df["index"].quantile(0.99)
        median_val = df["index"].median()
        df["index"] = df["index"].apply(lambda x: median_val if x < q1 or x > q99 else x)

    # Nettoyage des colonnes textuelles avec espaces
    text_cols = ["ProviderNumber", "HospitalName", "Address1", "City", "ZipCode", "CountyName", "PhoneNumber"]
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].str.strip()

    # Sauvegarde du dataset nettoyé
    df.to_csv(OUTPUT_PATH, index=False)

    # Log des opérations
    print("=== Résumé des opérations de nettoyage ===")
    print(f"Lignes initiales: {initial_rows}")
    print(f"Lignes après suppression des doublons: {len(df)}")
    for op in operations:
        print(op)
    print(f"Dataset nettoyé sauvegardé dans: {OUTPUT_PATH}")

if __name__ == "__main__":
    clean_dataset()