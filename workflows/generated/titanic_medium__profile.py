import pandas as pd
import numpy as np
import re
from datetime import datetime
import difflib

INPUT_PATH = r'datasets\titanic\noisy_medium.csv'
OUTPUT_PATH = r'results\cleaned_datasets\titanic\noisy_medium__profile.csv'

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
        if str(value).isdigit() and len(str(value)) in (9, 10):
            return datetime.fromtimestamp(int(value)).strftime("%Y-%m-%d")
    except (ValueError, OSError):
        pass
    return str(value)

df = pd.read_csv(INPUT_PATH)

# Suppression des doublons (conservation de la première occurrence)
initial_rows = len(df)
df.drop_duplicates(subset=[col for col in df.columns if col != "row_id"], keep="first", inplace=True)
duplicates_removed = initial_rows - len(df)

# Nettoyage colonne par colonne
operations = []

# Unnamed: 0 - colonne technique à conserver telle quelle (pas de nettoyage)
if "Unnamed: 0" in df.columns:
    operations.append("Unnamed: 0: colonne technique conservée")

# PassengerId - identifiant unique, pas de nettoyage
if "PassengerId" in df.columns:
    operations.append("PassengerId: identifiant unique conservé")

# Survived - binaire (0/1), pas de valeurs manquantes, pas de nettoyage nécessaire
if "Survived" in df.columns:
    operations.append("Survived: colonne binaire valide")

# Sex - binaire (0/1), pas de valeurs manquantes, pas de nettoyage nécessaire
if "Sex" in df.columns:
    operations.append("Sex: colonne binaire valide")

# Age - 15.78% de valeurs manquantes, valeurs fréquentes sous forme numérique (0.35, 0.3, etc.)
# Conversion en numérique et imputation par médiane
if "Age" in df.columns:
    df["Age"] = df["Age"].apply(extract_numeric)
    median_age = df["Age"].median()
    df["Age"] = df["Age"].fillna(median_age)
    operations.append(f"Age: {df['Age'].isna().sum()} valeurs manquantes imputées par médiane ({median_age})")

# Fare - valeurs sous forme textuelle avec nombres décimaux, conversion en numérique
if "Fare" in df.columns:
    df["Fare"] = df["Fare"].apply(extract_numeric)
    operations.append("Fare: conversion en numérique")

# Pclass_1, Pclass_2, Pclass_3 - colonnes binaires (0/1), pas de nettoyage nécessaire
for col in ["Pclass_1", "Pclass_2", "Pclass_3"]:
    if col in df.columns:
        operations.append(f"{col}: colonne binaire valide")

# Family_size - valeurs entre 0 et 1, pas de nettoyage nécessaire
if "Family_size" in df.columns:
    operations.append("Family_size: colonne numérique valide")

# Title_1, Title_2, Title_3, Title_4 - colonnes binaires (0/1), pas de nettoyage nécessaire
for col in ["Title_1", "Title_2", "Title_3", "Title_4"]:
    if col in df.columns:
        operations.append(f"{col}: colonne binaire valide")

# Emb_1, Emb_2, Emb_3 - colonnes binaires (0/1), pas de nettoyage nécessaire
for col in ["Emb_1", "Emb_2", "Emb_3"]:
    if col in df.columns:
        operations.append(f"{col}: colonne binaire valide")

# Vérification des valeurs aberrantes pour les colonnes numériques
numeric_cols = [col for col in df.columns if df[col].dtype in ['int64', 'float64'] and col not in ["row_id", "Unnamed: 0", "PassengerId"]]
for col in numeric_cols:
    if col in df.columns:
        q1 = df[col].quantile(0.01)
        q99 = df[col].quantile(0.995)
        median = df[col].median()
        outliers = df[(df[col] < q1) | (df[col] > q99)]
        if not outliers.empty:
            df.loc[(df[col] < q1) | (df[col] > q99), col] = median
            operations.append(f"{col}: {len(outliers)} valeurs aberrantes remplacées par médiane")

# Sauvegarde du dataset nettoyé
df.to_csv(OUTPUT_PATH, index=False)

# Affichage du résumé des opérations
print("=== Résumé du nettoyage ===")
print(f"Lignes initiales: {initial_rows}")
print(f"Doublons supprimés: {duplicates_removed}")
print(f"Lignes finales: {len(df)}")
print("\nOpérations effectuées:")
for op in operations:
    print(f"- {op}")