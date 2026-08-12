import pandas as pd
import numpy as np
import re
from datetime import datetime
import difflib

INPUT_PATH = "datasets/titanic/noisy_medium.csv"
OUTPUT_PATH = "results/cleaned_datasets/titanic/noisy_medium__profile.csv"

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

    # Suppression des doublons (conservation de la première occurrence)
    initial_rows = len(df)
    df = df.drop_duplicates(subset=[col for col in df.columns if col != "row_id"], keep="first")
    duplicates_removed = initial_rows - len(df)
    print(f"Doublons supprimés: {duplicates_removed}")

    # Traitement de la colonne Unnamed: 0 (identifiant technique redondant avec row_id)
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    # Traitement de la colonne Age (catégorielle avec 15.78% de valeurs manquantes)
    if "Age" in df.columns:
        # Extraction numérique des valeurs comme "0.275" qui sont stockées en texte
        df["Age"] = df["Age"].apply(extract_numeric)
        # Imputation des valeurs manquantes par la médiane (colonne discrète mais valeurs continues)
        group_col = find_best_grouping_column(df, "Age", is_numeric=True)
        if group_col:
            df["Age"] = df.groupby(group_col)["Age"].transform(
                lambda s: s.fillna(s.median())
            )
        df["Age"] = df["Age"].fillna(df["Age"].median())

    # Traitement de la colonne Fare (texte avec valeurs numériques)
    if "Fare" in df.columns:
        # Extraction numérique des valeurs comme "0.0157125535690723"
        df["Fare"] = df["Fare"].apply(extract_numeric)
        # Détection des outliers (bornes basées sur le profil: min=0.014, max=0.139)
        q_low, q_high = df["Fare"].quantile([0.005, 0.995])
        outliers = (df["Fare"] < q_low) | (df["Fare"] > q_high)
        if outliers.any():
            median_fare = df.loc[~outliers, "Fare"].median()
            df.loc[outliers, "Fare"] = median_fare

    # Colonnes binaires (Survived, Sex, Pclass_*, Title_*, Emb_*, Family_size)
    binary_cols = [
        "Survived", "Sex", "Pclass_1", "Pclass_2", "Pclass_3",
        "Title_1", "Title_2", "Title_3", "Title_4",
        "Emb_1", "Emb_2", "Emb_3", "Family_size"
    ]
    for col in binary_cols:
        if col in df.columns:
            # Conversion en numérique avec gestion des erreurs
            df[col] = pd.to_numeric(df[col], errors="coerce")
            # Imputation des valeurs manquantes par le mode (0 ou 1)
            df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 0)
            # Correction des valeurs hors [0,1]
            df[col] = df[col].clip(0, 1)

    # Vérification de la cohérence des colonnes Pclass (doivent sommer à 1)
    if all(col in df.columns for col in ["Pclass_1", "Pclass_2", "Pclass_3"]):
        pclass_sum = df[["Pclass_1", "Pclass_2", "Pclass_3"]].sum(axis=1)
        if (pclass_sum != 1).any():
            # Répartition forcée pour les lignes incohérentes
            df["Pclass_1"] = (df["Pclass_1"] == 1).astype(int)
            df["Pclass_2"] = (df["Pclass_2"] == 1).astype(int)
            df["Pclass_3"] = (df["Pclass_3"] == 1).astype(int)
            # Si toujours incohérent, imputation par le mode
            for col in ["Pclass_1", "Pclass_2", "Pclass_3"]:
                df[col] = df[col].fillna(df[col].mode()[0])

    # Vérification de la cohérence des colonnes Title (doivent sommer à 1)
    if all(col in df.columns for col in ["Title_1", "Title_2", "Title_3", "Title_4"]):
        title_sum = df[["Title_1", "Title_2", "Title_3", "Title_4"]].sum(axis=1)
        if (title_sum != 1).any():
            # Répartition forcée pour les lignes incohérentes
            df["Title_1"] = (df["Title_1"] == 1).astype(int)
            df["Title_2"] = (df["Title_2"] == 1).astype(int)
            df["Title_3"] = (df["Title_3"] == 1).astype(int)
            df["Title_4"] = (df["Title_4"] == 1).astype(int)
            # Si toujours incohérent, imputation par le mode
            for col in ["Title_1", "Title_2", "Title_3", "Title_4"]:
                df[col] = df[col].fillna(df[col].mode()[0])

    # Vérification de la cohérence des colonnes Emb (doivent sommer à 1)
    if all(col in df.columns for col in ["Emb_1", "Emb_2", "Emb_3"]):
        emb_sum = df[["Emb_1", "Emb_2", "Emb_3"]].sum(axis=1)
        if (emb_sum != 1).any():
            # Répartition forcée pour les lignes incohérentes
            df["Emb_1"] = (df["Emb_1"] == 1).astype(int)
            df["Emb_2"] = (df["Emb_2"] == 1).astype(int)
            df["Emb_3"] = (df["Emb_3"] == 1).astype(int)
            # Si toujours incohérent, imputation par le mode
            for col in ["Emb_1", "Emb_2", "Emb_3"]:
                df[col] = df[col].fillna(df[col].mode()[0])

    # Sauvegarde du dataset nettoyé
    df.to_csv(OUTPUT_PATH, index=False)

    # Log des opérations
    final_rows = len(df)
    print(f"Lignes initiales: {initial_rows}, lignes finales: {final_rows}")
    print(f"Valeurs manquantes traitées:")
    for col in df.columns:
        if col != "row_id":
            missing_before = initial_rows - df[col].count()
            missing_after = df[col].isna().sum()
            if missing_before > 0:
                print(f"  {col}: {missing_before} -> {missing_after}")

if __name__ == "__main__":
    clean_dataset()