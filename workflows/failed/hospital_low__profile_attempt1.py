import pandas as pd
import numpy as np
import re
from datetime import datetime
import difflib

INPUT_PATH = "datasets/hospital/noisy_low.csv"
OUTPUT_PATH = "results/cleaned_datasets/hospital/noisy_low__profile.csv"

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

    # Conservation de row_id comme identifiant technique
    if "row_id" not in df.columns:
        raise ValueError("La colonne row_id est absente du dataset")

    # Suppression des doublons (en conservant la première occurrence)
    initial_rows = len(df)
    df = df.drop_duplicates(subset=[c for c in df.columns if c != "row_id"], keep="first")
    duplicates_removed = initial_rows - len(df)
    print(f"Doublons supprimés: {duplicates_removed}")

    # Nettoyage colonne par colonne
    operations = []

    # ProviderNumber - pas de valeurs manquantes, pas de nettoyage nécessaire
    if "ProviderNumber" in df.columns:
        operations.append("ProviderNumber: aucune correction nécessaire")

    # HospitalName - harmonisation des variantes
    if "HospitalName" in df.columns:
        valid_hospitals = [
            "stringfellow memorial hospital", "huntsville hospital", "marshall medical center south",
            "helen keller memorial hospital", "callahan eye foundation hospital",
            "gadsden regional medical center", "chilton medical center", "st vincents east",
            "eliza coffee memorial hospital", "shelby baptist medical center"
        ]
        df["HospitalName"] = df["HospitalName"].apply(lambda v: harmonize_category(v, valid_hospitals))
        operations.append("HospitalName: harmonisation des variantes")

    # Address1 - pas de valeurs manquantes, pas de nettoyage nécessaire
    if "Address1" in df.columns:
        operations.append("Address1: aucune correction nécessaire")

    # Address2 et Address3 - valeurs "empty" répétées, à conserver
    for col in ["Address2", "Address3"]:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: x if str(x).strip().lower() == "empty" else x)
            operations.append(f"{col}: conservation des valeurs 'empty'")

    # City - harmonisation des variantes
    if "City" in df.columns:
        valid_cities = [
            "birmingham", "gadsden", "montgomery", "dothan", "huntsville", "anniston",
            "centre", "thomasville", "clanton", "opelika"
        ]
        df["City"] = df["City"].apply(lambda v: harmonize_category(v, valid_cities))
        operations.append("City: harmonisation des variantes")

    # State - correction des codes états aberrants
    if "State" in df.columns:
        valid_states = ["al", "ak"]
        df["State"] = df["State"].apply(lambda v: harmonize_category(v, valid_states))
        operations.append("State: correction des codes états aberrants")

    # ZipCode - pas de valeurs manquantes, format texte à conserver
    if "ZipCode" in df.columns:
        operations.append("ZipCode: aucune correction nécessaire")

    # CountyName - harmonisation des variantes
    if "CountyName" in df.columns:
        valid_counties = [
            "jefferson", "etowah", "marion", "covington", "marshall", "coffee",
            "montgomery", "houston", "calhoun", "madison"
        ]
        df["CountyName"] = df["CountyName"].apply(lambda v: harmonize_category(v, valid_counties))
        operations.append("CountyName: harmonisation des variantes")

    # PhoneNumber - pas de valeurs manquantes, format texte à conserver
    if "PhoneNumber" in df.columns:
        operations.append("PhoneNumber: aucune correction nécessaire")

    # HospitalType - correction des fautes de frappe
    if "HospitalType" in df.columns:
        valid_types = ["acute care hospitals"]
        df["HospitalType"] = df["HospitalType"].apply(lambda v: harmonize_category(v, valid_types))
        operations.append("HospitalType: correction des fautes de frappe")

    # HospitalOwner - correction des variantes
    if "HospitalOwner" in df.columns:
        valid_owners = [
            "voluntary non-profit - private", "proprietary", "government - hospital district or authority",
            "voluntary non-profit - other", "voluntary non-profit - church", "government - federal",
            "government - state", "government - local"
        ]
        df["HospitalOwner"] = df["HospitalOwner"].apply(lambda v: harmonize_category(v, valid_owners))
        operations.append("HospitalOwner: correction des variantes")

    # EmergencyService - correction des variantes
    if "EmergencyService" in df.columns:
        valid_emergency = ["yes", "no"]
        df["EmergencyService"] = df["EmergencyService"].apply(lambda v: harmonize_category(v, valid_emergency))
        operations.append("EmergencyService: correction des variantes")

    # Condition - correction des fautes de frappe
    if "Condition" in df.columns:
        valid_conditions = [
            "surgical infection prevention", "heart attack", "pneumonia", "heart failure",
            "children s asthma care"
        ]
        df["Condition"] = df["Condition"].apply(lambda v: harmonize_category(v, valid_conditions))
        operations.append("Condition: correction des fautes de frappe")

    # MeasureCode - pas de valeurs manquantes, format texte à conserver
    if "MeasureCode" in df.columns:
        operations.append("MeasureCode: aucune correction nécessaire")

    # MeasureName - pas de valeurs manquantes, format texte à conserver
    if "MeasureName" in df.columns:
        operations.append("MeasureName: aucune correction nécessaire")

    # Score - extraction du pourcentage et imputation des valeurs manquantes
    if "Score" in df.columns:
        df["Score"] = df["Score"].apply(lambda x: x if str(x).strip().lower() == "empty" else x)
        score_numeric = df["Score"].apply(extract_numeric)
        score_numeric = score_numeric / 100 if score_numeric.max() > 1 else score_numeric
        missing_mask = (df["Score"] == "empty") | df["Score"].isna()
        if missing_mask.any():
            group_col = find_best_grouping_column(df, "Score", True)
            if group_col:
                df["Score"] = df.groupby(group_col)["Score"].transform(
                    lambda s: s.fillna(s.median() if not s.mode().empty else 1.0)
                )
            df.loc[missing_mask, "Score"] = df["Score"].median()
        df["Score"] = df["Score"].apply(lambda x: f"{int(float(x)*100)}%" if isinstance(x, (int, float)) else x)
        operations.append("Score: extraction des pourcentages et imputation des valeurs manquantes")

    # Sample - extraction du nombre de patients et imputation des valeurs manquantes
    if "Sample" in df.columns:
        df["Sample"] = df["Sample"].apply(lambda x: x if str(x).strip().lower() == "empty" else x)
        sample_numeric = df["Sample"].apply(extract_numeric)
        missing_mask = (df["Sample"] == "empty") | df["Sample"].isna()
        if missing_mask.any():
            group_col = find_best_grouping_column(df, "Sample", True)
            if group_col:
                df["Sample"] = df.groupby(group_col)["Sample"].transform(
                    lambda s: s.fillna(s.mode().iloc[0] if not s.mode().empty else "0 patients")
                )
            df.loc[missing_mask, "Sample"] = df["Sample"].mode().iloc[0]
        df["Sample"] = df["Sample"].apply(
            lambda x: f"{int(float(x))} patients" if isinstance(x, (int, float)) and float(x).is_integer() else x
        )
        operations.append("Sample: extraction des nombres de patients et imputation des valeurs manquantes")

    # Stateavg - pas de valeurs manquantes, format texte à conserver
    if "Stateavg" in df.columns:
        operations.append("Stateavg: aucune correction nécessaire")

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
    clean_hospital_data()