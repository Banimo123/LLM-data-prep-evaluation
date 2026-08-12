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

def clean_hospital_data():
    df = pd.read_csv(INPUT_PATH)

    # Conservation des row_id intacts
    original_row_ids = df["row_id"].copy()

    # Suppression des doublons (en conservant la première occurrence)
    initial_rows = len(df)
    df = df.drop_duplicates(subset=[c for c in df.columns if c != "row_id"], keep="first")
    duplicates_removed = initial_rows - len(df)
    df["row_id"] = original_row_ids.loc[df.index]

    # Nettoyage colonne par colonne
    operations = []

    # ProviderNumber (numeric, 0% manquants, valeurs dans [10001, 20018])
    if "ProviderNumber" in df.columns:
        df["ProviderNumber"] = df["ProviderNumber"].apply(extract_numeric)
        # Vérification des bornes physiques
        valid_mask = (df["ProviderNumber"] >= 10000) & (df["ProviderNumber"] <= 20020)
        outliers = df[~valid_mask]["ProviderNumber"]
        if not outliers.empty:
            median_val = df.loc[valid_mask, "ProviderNumber"].median()
            df.loc[~valid_mask, "ProviderNumber"] = median_val
            operations.append(f"ProviderNumber: {len(outliers)} valeurs hors bornes corrigées par médiane")

    # HospitalName (text, 0% manquants, harmonisation des casse/espaces)
    if "HospitalName" in df.columns:
        df["HospitalName"] = df["HospitalName"].str.strip().str.title()
        frequent_names = [
            "Stringfellow Memorial Hospital", "Riverview Regional Medical Center",
            "Mizell Memorial Hospital", "Shelby Baptist Medical Center",
            "Callahan Eye Foundation Hospital", "G H Lanier Memorial Hospital",
            "East Alabama Medical Center And Snf", "Cherokee Medical Center",
            "Huntsville Hospital", "Medical Center Enterprise"
        ]
        df["HospitalName"] = df["HospitalName"].apply(lambda v: harmonize_category(v, frequent_names))

    # Address1 (text, 0% manquants, nettoyage des espaces)
    if "Address1" in df.columns:
        df["Address1"] = df["Address1"].str.strip()

    # Address2 et Address3 (text, 100% "empty" -> remplacement par NaN)
    for col in ["Address2", "Address3"]:
        if col in df.columns:
            df[col] = df[col].replace("empty", np.nan)

    # City (text, 16.8% manquants, imputation conditionnelle)
    if "City" in df.columns:
        missing_mask = df["City"].isna()
        if missing_mask.any():
            group_col = find_best_grouping_column(df, "City", False)
            if group_col:
                df["City"] = df.groupby(group_col)["City"].transform(
                    lambda s: s.fillna(s.mode().iloc[0] if not s.mode().empty else "Unknown")
                )
            df["City"] = df["City"].fillna(df["City"].mode().iloc[0])
            operations.append(f"City: {missing_mask.sum()} valeurs manquantes imputées")

    # State (text, 15.9% manquants, harmonisation des valeurs)
    if "State" in df.columns:
        valid_states = ["al", "ak", "la"]
        df["State"] = df["State"].apply(lambda v: harmonize_category(v, valid_states))
        # Correction des valeurs aberrantes
        state_map = {
            "  ": "al", "l": "al", "  al  ": "al", "Al": "al",
            "aal": "al", "unknwn": "unknown", "unknown": "unknown"
        }
        df["State"] = df["State"].replace(state_map)
        missing_mask = df["State"].isna()
        if missing_mask.any():
            df["State"] = df["State"].fillna(df["State"].mode().iloc[0])
            operations.append(f"State: {missing_mask.sum()} valeurs manquantes imputées")

    # ZipCode (text, 0% manquants, nettoyage des caractères parasites)
    if "ZipCode" in df.columns:
        df["ZipCode"] = df["ZipCode"].astype(str).str.extract(r'(\d{5})')[0]
        df["ZipCode"] = df["ZipCode"].fillna(df["ZipCode"].mode().iloc[0])

    # CountyName (text, 0% manquants, harmonisation)
    if "CountyName" in df.columns:
        frequent_counties = [
            "jefferson", "etowah", "marshall", "marion", "covington",
            "montgomery", "coffee", "houston", "calhoun", "madison"
        ]
        df["CountyName"] = df["CountyName"].str.lower().str.strip()
        df["CountyName"] = df["CountyName"].apply(lambda v: harmonize_category(v, frequent_counties))

    # PhoneNumber (numeric, 0% manquants, extraction des chiffres)
    if "PhoneNumber" in df.columns:
        df["PhoneNumber"] = df["PhoneNumber"].apply(extract_numeric)
        # Vérification des bornes (numéros US valides)
        valid_mask = (df["PhoneNumber"] >= 2010000000) & (df["PhoneNumber"] <= 9999999999)
        outliers = df[~valid_mask]["PhoneNumber"]
        if not outliers.empty:
            median_val = df.loc[valid_mask, "PhoneNumber"].median()
            df.loc[~valid_mask, "PhoneNumber"] = median_val
            operations.append(f"PhoneNumber: {len(outliers)} valeurs hors bornes corrigées")

    # HospitalOwner (text, 17.1% manquants, harmonisation)
    if "HospitalOwner" in df.columns:
        valid_owners = [
            "voluntary non-profit - private", "proprietary",
            "government - hospital district or authority",
            "voluntary non-profit - other", "voluntary non-profit - church",
            "government - federal", "government - state", "government - local",
            "unknown"
        ]
        df["HospitalOwner"] = df["HospitalOwner"].apply(lambda v: harmonize_category(v, valid_owners))
        missing_mask = df["HospitalOwner"].isna()
        if missing_mask.any():
            group_col = find_best_grouping_column(df, "HospitalOwner", False)
            if group_col:
                df["HospitalOwner"] = df.groupby(group_col)["HospitalOwner"].transform(
                    lambda s: s.fillna(s.mode().iloc[0] if not s.mode().empty else "Unknown")
                )
            df["HospitalOwner"] = df["HospitalOwner"].fillna(df["HospitalOwner"].mode().iloc[0])
            operations.append(f"HospitalOwner: {missing_mask.sum()} valeurs manquantes imputées")

    # EmergencyService (text, 16.1% manquants, harmonisation)
    if "EmergencyService" in df.columns:
        valid_emergency = ["yes", "no", "unknown"]
        df["EmergencyService"] = df["EmergencyService"].apply(lambda v: harmonize_category(v, valid_emergency))
        missing_mask = df["EmergencyService"].isna()
        if missing_mask.any():
            df["EmergencyService"] = df["EmergencyService"].fillna(df["EmergencyService"].mode().iloc[0])
            operations.append(f"EmergencyService: {missing_mask.sum()} valeurs manquantes imputées")

    # Condition (text, 0% manquants, harmonisation)
    if "Condition" in df.columns:
        valid_conditions = [
            "surgical infection prevention", "heart attack", "pneumonia",
            "heart failure", "children s asthma care"
        ]
        df["Condition"] = df["Condition"].apply(lambda v: harmonize_category(v, valid_conditions))

    # MeasureName (text, 0% manquants, nettoyage des espaces)
    if "MeasureName" in df.columns:
        df["MeasureName"] = df["MeasureName"].str.strip()

    # Score (text, 0% manquants, extraction des pourcentages)
    if "Score" in df.columns:
        df["Score"] = df["Score"].replace("empty", np.nan)
        # Extraction des pourcentages
        def extract_score(value):
            if pd.isna(value):
                return value
            s = str(value).strip()
            if s.endswith("%"):
                try:
                    return f"{int(float(s[:-1]))}%"
                except ValueError:
                    return s
            return s
        df["Score"] = df["Score"].apply(extract_score)
        # Imputation des valeurs manquantes
        missing_mask = df["Score"].isna()
        if missing_mask.any():
            df["Score"] = df["Score"].fillna(df["Score"].mode().iloc[0])
            operations.append(f"Score: {missing_mask.sum()} valeurs manquantes imputées")

    # Sample (text, 0% manquants, extraction des nombres)
    if "Sample" in df.columns:
        df["Sample"] = df["Sample"].replace("empty", np.nan)
        def extract_sample(value):
            if pd.isna(value):
                return value
            s = str(value).strip()
            if "patients" in s:
                match = re.search(r'(\d+)', s)
                return f"{match.group(1)} patients" if match else s
            return s
        df["Sample"] = df["Sample"].apply(extract_sample)
        missing_mask = df["Sample"].isna()
        if missing_mask.any():
            df["Sample"] = df["Sample"].fillna(df["Sample"].mode().iloc[0])
            operations.append(f"Sample: {missing_mask.sum()} valeurs manquantes imputées")

    # Stateavg (text, 0% manquants, nettoyage des espaces)
    if "Stateavg" in df.columns:
        df["Stateavg"] = df["Stateavg"].str.strip()

    # Sauvegarde du dataset nettoyé
    df.to_csv(OUTPUT_PATH, index=False)

    # Résumé des opérations
    print(f"Nettoyage terminé. {duplicates_removed} doublons supprimés.")
    for op in operations:
        print(f"- {op}")
    print(f"Dataset final: {len(df)} lignes, {len(df.columns)} colonnes.")

if __name__ == "__main__":
    clean_hospital_data()