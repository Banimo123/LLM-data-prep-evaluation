import pandas as pd
import numpy as np
import re
from datetime import datetime

INPUT_PATH = "datasets/titanic/noisy_high.csv"
OUTPUT_PATH = "results/cleaned_datasets/titanic/noisy_high__profile.csv"

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
            dt = datetime.strptime(str(value), fmt)
            return dt.strftime("%Y-%m-%d")
        except ValueError:
            continue
    if str(value).isdigit() and len(str(value)) in (9, 10):
        try:
            dt = datetime.fromtimestamp(int(value))
            return dt.strftime("%Y-%m-%d")
        except (ValueError, OSError):
            pass
    return str(value)

df = pd.read_csv(INPUT_PATH)

operations_log = []

# Suppression de la colonne Unnamed: 0 (redondante avec row_id)
if "Unnamed: 0" in df.columns:
    df.drop(columns=["Unnamed: 0"], inplace=True)
    operations_log.append("Suppression colonne Unnamed: 0 (redondante)")

# Suppression des doublons (en conservant le premier)
initial_rows = len(df)
df.drop_duplicates(subset=[col for col in df.columns if col != "row_id"], keep="first", inplace=True)
final_rows = len(df)
if initial_rows != final_rows:
    operations_log.append(f"Suppression de {initial_rows - final_rows} doublons")

# Traitement colonne Age (catégorielle avec valeurs numériques sous forme texte)
if "Age" in df.columns:
    # Extraction des valeurs numériques
    df["Age"] = df["Age"].apply(extract_numeric)
    # Imputation des valeurs manquantes (28.16%)
    group_col = find_best_grouping_column(df, "Age", is_numeric=True)
    if group_col:
        df["Age"] = df.groupby(group_col)["Age"].transform(
            lambda s: s.fillna(s.median())
        )
    df["Age"] = df["Age"].fillna(df["Age"].median())
    operations_log.append("Colonne Age : extraction numérique + imputation par médiane")

# Traitement colonne Fare (texte avec valeurs numériques corrompues)
if "Fare" in df.columns:
    # Extraction des valeurs numériques
    df["Fare"] = df["Fare"].apply(extract_numeric)
    # Détection des outliers (valeurs > 99.5e percentile)
    q995 = df["Fare"].quantile(0.995)
    outliers = df["Fare"] > q995
    if outliers.any():
        median_fare = df.loc[~outliers, "Fare"].median()
        df.loc[outliers, "Fare"] = median_fare
        operations_log.append(f"Colonne Fare : {outliers.sum()} outliers remplacés par médiane")
    operations_log.append("Colonne Fare : extraction numérique + correction outliers")

# Colonnes numériques binaires (Pclass_*, Title_*, Emb_*) - pas de traitement nécessaire
binary_cols = [col for col in df.columns if col.startswith(("Pclass_", "Title_", "Emb_"))]
for col in binary_cols:
    if col in df.columns:
        # Vérification que les valeurs sont bien 0 ou 1
        unique_vals = df[col].dropna().unique()
        if not all(val in (0, 1) for val in unique_vals):
            df[col] = df[col].apply(lambda x: 1 if x == 1 else 0)
            operations_log.append(f"Colonne {col} : correction des valeurs non binaires")

# Traitement colonne Family_size (valeurs entre 0 et 1)
if "Family_size" in df.columns:
    # Vérification des valeurs hors [0,1]
    outliers = (df["Family_size"] < 0) | (df["Family_size"] > 1)
    if outliers.any():
        mode_family = df.loc[~outliers, "Family_size"].mode()[0]
        df.loc[outliers, "Family_size"] = mode_family
        operations_log.append(f"Colonne Family_size : {outliers.sum()} valeurs hors [0,1] remplacées par mode")

# Vérification des colonnes PassengerId et Survived (pas de valeurs manquantes)
for col in ["PassengerId", "Survived", "Sex"]:
    if col in df.columns:
        # Vérification que les valeurs sont dans les bornes attendues
        if col == "Survived":
            unique_vals = df[col].dropna().unique()
            if not all(val in (0, 1) for val in unique_vals):
                df[col] = df[col].apply(lambda x: 1 if x == 1 else 0)
                operations_log.append(f"Colonne {col} : correction des valeurs non binaires")
        elif col == "Sex":
            unique_vals = df[col].dropna().unique()
            if not all(val in (0, 1) for val in unique_vals):
                df[col] = df[col].apply(lambda x: 1 if x == 1 else 0)
                operations_log.append(f"Colonne {col} : correction des valeurs non binaires")

# Sauvegarde du dataset nettoyé
df.to_csv(OUTPUT_PATH, index=False)

# Log des opérations
print("=== Résumé des opérations de nettoyage ===")
for op in operations_log:
    print(f"- {op}")
print(f"\nDataset nettoyé sauvegardé dans : {OUTPUT_PATH}")
print(f"Nombre final de lignes : {len(df)}")
print(f"Nombre final de colonnes : {len(df.columns)}")