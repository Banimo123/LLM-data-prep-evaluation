import pandas as pd
import numpy as np
import re
from datetime import datetime
import difflib

INPUT_PATH = r'datasets\titanic\noisy_high.csv'
OUTPUT_PATH = r'results\cleaned_datasets\titanic\noisy_high__profile.csv'

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
        "%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y",
        "%B %d, %Y", "%d-%b-%Y", "%Y/%m/%d"
    ]
    for fmt in date_formats:
        try:
            dt = datetime.strptime(str(value), fmt)
            return dt.strftime("%Y-%m-%d")
        except ValueError:
            continue
    try:
        if str(value).isdigit() and len(str(value)) in (9, 10):
            dt = datetime.fromtimestamp(int(value))
            return dt.strftime("%Y-%m-%d")
    except (ValueError, OSError):
        pass
    return str(value)

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

    # Suppression des doublons (conservation du premier)
    initial_rows = len(df)
    df = df.drop_duplicates(subset=[col for col in df.columns if col != "row_id"], keep="first")
    duplicates_removed = initial_rows - len(df)
    print(f"Doublons supprimés: {duplicates_removed}")

    # Nettoyage colonne par colonne
    operations = []

    # Unnamed: 0 - colonne technique à conserver telle quelle (pas de nettoyage)
    if "Unnamed: 0" in df.columns:
        operations.append("Unnamed: 0 conservée (colonne technique)")

    # PassengerId - identifiant unique, pas de nettoyage
    if "PassengerId" in df.columns:
        operations.append("PassengerId conservée (identifiant unique)")

    # Survived - binaire, pas de valeurs manquantes, pas de nettoyage
    if "Survived" in df.columns:
        operations.append("Survived conservée (binaire, pas de valeurs manquantes)")

    # Sex - binaire, pas de valeurs manquantes, pas de nettoyage
    if "Sex" in df.columns:
        operations.append("Sex conservée (binaire, pas de valeurs manquantes)")

    # Age - 28.16% de valeurs manquantes, valeurs textuelles avec nombres décimaux
    if "Age" in df.columns:
        # Extraction des valeurs numériques
        df["Age"] = df["Age"].apply(extract_numeric)
        # Imputation des valeurs manquantes
        group_col = find_best_grouping_column(df, "Age", is_numeric=True)
        if group_col:
            df["Age"] = df.groupby(group_col)["Age"].transform(
                lambda s: s.fillna(s.median())
            )
        df["Age"] = df["Age"].fillna(df["Age"].median())
        operations.append(f"Age: {df['Age'].isna().sum()} valeurs manquantes imputées par médiane (après extraction numérique)")

    # Fare - valeurs textuelles avec nombres décimaux et caractères parasites
    if "Fare" in df.columns:
        # Correction de la valeur aberrante dans l'extrait (O.1036442974556203)
        df["Fare"] = df["Fare"].apply(extract_numeric)
        # Détection des outliers (valeurs > 99.5e percentile)
        q995 = df["Fare"].quantile(0.995)
        outliers = df["Fare"] > q995
        if outliers.any():
            median_fare = df.loc[~outliers, "Fare"].median()
            df.loc[outliers, "Fare"] = median_fare
            operations.append(f"Fare: {outliers.sum()} outliers remplacés par la médiane")
        operations.append("Fare: valeurs numériques extraites et outliers corrigés")

    # Pclass_1, Pclass_2, Pclass_3 - binaires, pas de valeurs manquantes
    for col in ["Pclass_1", "Pclass_2", "Pclass_3"]:
        if col in df.columns:
            operations.append(f"{col} conservée (binaire, pas de valeurs manquantes)")

    # Family_size - valeurs entre 0 et 1, pas de nettoyage nécessaire
    if "Family_size" in df.columns:
        operations.append("Family_size conservée (valeurs dans [0,1])")

    # Title_1 à Title_4 - binaires, pas de valeurs manquantes
    for col in ["Title_1", "Title_2", "Title_3", "Title_4"]:
        if col in df.columns:
            operations.append(f"{col} conservée (binaire, pas de valeurs manquantes)")

    # Emb_1, Emb_2, Emb_3 - binaires, pas de valeurs manquantes
    for col in ["Emb_1", "Emb_2", "Emb_3"]:
        if col in df.columns:
            operations.append(f"{col} conservée (binaire, pas de valeurs manquantes)")

    # Vérification finale des valeurs manquantes
    missing_before = df.isna().sum().sum()
    df = df.fillna({
        col: df[col].mode()[0] if df[col].dtype == 'object' else df[col].median()
        for col in df.columns if col != "row_id"
    })
    missing_after = df.isna().sum().sum()
    if missing_after < missing_before:
        operations.append(f"Valeurs manquantes résiduelles imputées: {missing_before - missing_after}")

    # Sauvegarde du dataset nettoyé
    df.to_csv(OUTPUT_PATH, index=False)

    # Résumé des opérations
    print("\nRésumé des opérations de nettoyage:")
    for op in operations:
        print(f"- {op}")
    print(f"\nDataset nettoyé sauvegardé dans: {OUTPUT_PATH}")
    print(f"Nombre de lignes initial: {initial_rows}")
    print(f"Nombre de lignes final: {len(df)}")

if __name__ == "__main__":
    clean_dataset()