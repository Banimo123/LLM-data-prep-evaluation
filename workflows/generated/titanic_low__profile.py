import pandas as pd
import numpy as np
import re
from datetime import datetime
import difflib

INPUT_PATH = "datasets/titanic/noisy_low.csv"
OUTPUT_PATH = "results/cleaned_datasets/titanic/noisy_low__profile.csv"

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

def clean_dataset():
    df = pd.read_csv(INPUT_PATH)

    # Suppression de la colonne Unnamed: 0 (redondante avec row_id)
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    # Suppression des doublons (conservation de la première occurrence)
    initial_rows = len(df)
    df = df.drop_duplicates(subset=[col for col in df.columns if col != "row_id"], keep="first")
    duplicates_removed = initial_rows - len(df)

    # Nettoyage colonne Age (catégorielle avec valeurs manquantes)
    if "Age" in df.columns:
        # Extraction numérique des valeurs corrompues (ex: "25 ans" -> 25)
        df["Age"] = df["Age"].apply(extract_numeric)
        # Imputation des valeurs manquantes
        group_col = find_best_grouping_column(df, "Age", is_numeric=True)
        if group_col:
            df["Age"] = df.groupby(group_col)["Age"].transform(
                lambda s: s.fillna(s.median())
            )
        df["Age"] = df["Age"].fillna(df["Age"].median())

    # Nettoyage colonne Fare (catégorielle avec valeurs numériques corrompues)
    if "Fare" in df.columns:
        # Extraction numérique des valeurs corrompues (ex: "7.25€" -> 7.25)
        df["Fare"] = df["Fare"].apply(extract_numeric)
        # Détection des outliers (bornes basées sur le profil: min=0.014, max=0.139)
        q99 = df["Fare"].quantile(0.99)
        outliers = df["Fare"] > q99
        if outliers.any():
            df.loc[outliers, "Fare"] = df["Fare"].median()
        # Imputation des valeurs manquantes (aucune dans le profil, mais au cas où)
        df["Fare"] = df["Fare"].fillna(df["Fare"].median())

    # Colonnes numériques binaires (Survived, Sex, Pclass_*, Title_*, Emb_*)
    binary_cols = ["Survived", "Sex", "Pclass_1", "Pclass_2", "Pclass_3",
                   "Title_1", "Title_2", "Title_3", "Title_4", "Emb_1", "Emb_2", "Emb_3"]
    for col in binary_cols:
        if col in df.columns:
            # Conversion en numérique (gestion des erreurs)
            df[col] = pd.to_numeric(df[col], errors="coerce")
            # Imputation des valeurs manquantes (aucune dans le profil, mais au cas où)
            df[col] = df[col].fillna(df[col].mode()[0])

    # Vérification des valeurs aberrantes pour Family_size (min=0, max=1 dans le profil)
    if "Family_size" in df.columns:
        # Aucune valeur aberrante attendue (bornes 0-1), mais vérification
        df["Family_size"] = pd.to_numeric(df["Family_size"], errors="coerce")
        df["Family_size"] = df["Family_size"].clip(0, 1)
        df["Family_size"] = df["Family_size"].fillna(df["Family_size"].mode()[0])

    # Résumé des opérations
    print(f"Nettoyage terminé:")
    print(f"- Lignes initiales: {initial_rows}")
    print(f"- Doublons supprimés: {duplicates_removed}")
    print(f"- Lignes finales: {len(df)}")
    print(f"- Colonnes traitées: {len(df.columns)}")

    # Sauvegarde du dataset nettoyé
    df.to_csv(OUTPUT_PATH, index=False)

if __name__ == "__main__":
    clean_dataset()