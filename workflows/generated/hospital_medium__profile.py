import pandas as pd
import numpy as np
import re
from datetime import datetime
import difflib

INPUT_PATH = "datasets/hospital/noisy_medium.csv"
OUTPUT_PATH = "results/cleaned_datasets/hospital/noisy_medium__profile.csv"

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
    date_formats = ["%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%B %d, %Y", "%d-%b-%Y", "%Y/%m/%d"]
    for fmt in date_formats:
        try:
            return datetime.strptime(str(value), fmt).strftime("%Y-%m-%d")
        except (ValueError, TypeError):
            continue
    try:
        if str(value).isdigit() and len(str(value)) in (9, 10):
            return datetime.fromtimestamp(int(value)).strftime("%Y-%m-%d")
    except (ValueError, TypeError):
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

def clean_hospital_data():
    df = pd.read_csv(INPUT_PATH)

    operations = []

    # Suppression des doublons (conservation de la première occurrence)
    initial_rows = len(df)
    df = df.drop_duplicates(subset=[col for col in df.columns if col != "row_id"], keep="first")
    duplicates_removed = initial_rows - len(df)
    if duplicates_removed > 0:
        operations.append(f"Doublons supprimés: {duplicates_removed}")

    # Nettoyage colonne par colonne
    for col in df.columns:
        if col == "row_id":
            continue

        # Traitement des valeurs manquantes
        missing_pct = df[col].isna().mean() * 100
        if missing_pct > 0:
            if missing_pct > 50:
                operations.append(f"Colonne {col}: {missing_pct:.1f}% de valeurs manquantes -> suppression de la colonne")
                df.drop(columns=[col], inplace=True)
                continue

            is_numeric = pd.api.types.is_numeric_dtype(df[col])
            if col in ["City", "State", "HospitalOwner", "EmergencyService"]:
                # Imputation conditionnelle pour ces colonnes catégorielles
                group_col = find_best_grouping_column(df, col, False)
                if group_col:
                    df[col] = df.groupby(group_col)[col].transform(
                        lambda s: s.fillna(s.mode().iloc[0] if not s.mode().empty else "Unknown")
                    )
                    operations.append(f"Colonne {col}: imputation conditionnelle par {group_col}")
                df[col] = df[col].fillna(df[col].mode().iloc[0])
                operations.append(f"Colonne {col}: {missing_pct:.1f}% de valeurs manquantes -> imputation par mode")
            elif is_numeric:
                # Imputation conditionnelle pour les colonnes numériques
                group_col = find_best_grouping_column(df, col, True)
                if group_col:
                    df[col] = df.groupby(group_col)[col].transform(
                        lambda s: s.fillna(s.median())
                    )
                    operations.append(f"Colonne {col}: imputation conditionnelle par {group_col}")
                df[col] = df[col].fillna(df[col].median())
                operations.append(f"Colonne {col}: {missing_pct:.1f}% de valeurs manquantes -> imputation par médiane")
            else:
                df[col] = df[col].fillna(df[col].mode().iloc[0])
                operations.append(f"Colonne {col}: {missing_pct:.1f}% de valeurs manquantes -> imputation par mode")

        # Correction des valeurs aberrantes pour les colonnes numériques
        if pd.api.types.is_numeric_dtype(df[col]):
            if col == "ProviderNumber":
                # Bornes basées sur le profil (min: 10001, max: 20018)
                valid_mask = (df[col] >= 10001) & (df[col] <= 20018)
                outliers = df[~valid_mask]
                if not outliers.empty:
                    median_val = df.loc[valid_mask, col].median()
                    df.loc[~valid_mask, col] = median_val
                    operations.append(f"Colonne {col}: {len(outliers)} valeurs aberrantes corrigées par médiane")
            elif col == "PhoneNumber":
                # Bornes basées sur le profil (min: 2052743000, max: 9075436300)
                valid_mask = (df[col] >= 2052743000) & (df[col] <= 9075436300)
                outliers = df[~valid_mask]
                if not outliers.empty:
                    median_val = df.loc[valid_mask, col].median()
                    df.loc[~valid_mask, col] = median_val
                    operations.append(f"Colonne {col}: {len(outliers)} valeurs aberrantes corrigées par médiane")
            elif col == "index":
                # Bornes basées sur le profil (min: 1, max: 1000)
                valid_mask = (df[col] >= 1) & (df[col] <= 1000)
                outliers = df[~valid_mask]
                if not outliers.empty:
                    median_val = df.loc[valid_mask, col].median()
                    df.loc[~valid_mask, col] = median_val
                    operations.append(f"Colonne {col}: {len(outliers)} valeurs aberrantes corrigées par médiane")

        # Harmonisation des catégories pour les colonnes textuelles
        if col in ["State", "HospitalOwner", "EmergencyService", "Condition", "HospitalType"]:
            if col == "State":
                valid_values = ["al", "ak", "la"]
                df[col] = df[col].apply(lambda v: harmonize_category(v, valid_values))
                operations.append(f"Colonne {col}: harmonisation des catégories")
            elif col == "HospitalOwner":
                valid_values = [
                    "voluntary non-profit - private", "proprietary", "government - hospital district or authority",
                    "voluntary non-profit - other", "voluntary non-profit - church", "government - federal",
                    "government - state", "government - local", "unknown"
                ]
                df[col] = df[col].apply(lambda v: harmonize_category(v, valid_values))
                operations.append(f"Colonne {col}: harmonisation des catégories")
            elif col == "EmergencyService":
                valid_values = ["yes", "no", "unknown"]
                df[col] = df[col].apply(lambda v: harmonize_category(v, valid_values))
                operations.append(f"Colonne {col}: harmonisation des catégories")
            elif col == "Condition":
                valid_values = [
                    "surgical infection prevention", "heart attack", "pneumonia",
                    "heart failure", "children s asthma care"
                ]
                df[col] = df[col].apply(lambda v: harmonize_category(v, valid_values))
                operations.append(f"Colonne {col}: harmonisation des catégories")
            elif col == "HospitalType":
                valid_values = ["acute care hospitals"]
                df[col] = df[col].apply(lambda v: harmonize_category(v, valid_values))
                operations.append(f"Colonne {col}: harmonisation des catégories")

        # Correction des formats spécifiques
        if col == "Score":
            # Extraction du pourcentage si présent
            df[col] = df[col].apply(lambda x: x if pd.isna(x) or x == "empty" else
                                  (f"{int(extract_numeric(x))}%" if not pd.isna(extract_numeric(x)) else x))
            operations.append(f"Colonne {col}: formatage des pourcentages")
        elif col == "Sample":
            # Conservation du format "X patients" ou "empty"
            df[col] = df[col].apply(lambda x: x if pd.isna(x) or x == "empty" else
                                  (f"{int(extract_numeric(x))} patients" if not pd.isna(extract_numeric(x)) else x))
            operations.append(f"Colonne {col}: formatage des comptes de patients")
        elif col == "ZipCode":
            # Nettoyage des codes postaux (conservation des 5 chiffres)
            df[col] = df[col].apply(lambda x: re.sub(r"[^\d]", "", str(x))[:5] if not pd.isna(x) else x)
            operations.append(f"Colonne {col}: nettoyage des codes postaux")
        elif col == "PhoneNumber":
            # Conversion en entier (déjà numérique d'après le profil)
            df[col] = pd.to_numeric(df[col], errors="coerce")
            operations.append(f"Colonne {col}: conversion en numérique")

    # Sauvegarde du dataset nettoyé
    df.to_csv(OUTPUT_PATH, index=False)

    # Affichage du résumé des opérations
    print("=== Résumé des opérations de nettoyage ===")
    print(f"Lignes initiales: {initial_rows}")
    print(f"Lignes finales: {len(df)}")
    for op in operations:
        print(f"- {op}")
    print(f"\nDataset nettoyé sauvegardé dans: {OUTPUT_PATH}")

if __name__ == "__main__":
    clean_hospital_data()