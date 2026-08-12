import pandas as pd
import numpy as np
import re
from datetime import datetime
import difflib

INPUT_PATH = "datasets/hospital/noisy_high.csv"
OUTPUT_PATH = "results/cleaned_datasets/hospital/noisy_high__profile.csv"

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
            sub = observed.loc[t.index, [cand]].copy()
            sub["_t"] = t
            w_mad, total = 0.0, len(sub)
            for _, grp in sub.groupby(cand)["_t"]:
                if len(grp) >= 5:
                    w_mad += (grp - grp.median()).abs().median() * (len(grp) / total)
                else:
                    w_mad += global_mad * (len(grp) / total)
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
                if len(grp) > 0:
                    w_share += grp.value_counts(normalize=True).iloc[0] * (len(grp) / total)
            if (w_share - global_share) > best_gain and (w_share - global_share) >= 0.05:
                best_gain, best_col = w_share - global_share, cand
    return best_col

def clean_hospital_data():
    df = pd.read_csv(INPUT_PATH)

    # Suppression des doublons basés sur row_id (conservation du premier)
    initial_rows = len(df)
    df = df.drop_duplicates(subset=["row_id"], keep="first")
    duplicates_removed = initial_rows - len(df)
    print(f"Doublons supprimés: {duplicates_removed}")

    # Nettoyage colonne par colonne
    operations = []

    # ProviderNumber: colonne numérique, 0% manquants, pas d'aberrations détectées
    if "ProviderNumber" in df.columns:
        df["ProviderNumber"] = df["ProviderNumber"].apply(extract_numeric)
        operations.append("ProviderNumber: valeurs numériques extraites")

    # HospitalName: texte, 0% manquants, harmonisation des casse/espaces
    if "HospitalName" in df.columns:
        valid_hospitals = [
            "riverview regional medical center", "georgiana hospital",
            "stringfellow memorial hospital", "cherokee medical center",
            "helen keller memorial hospital", "southwest alabama medical center",
            "g h lanier memorial hospital", "mizell memorial hospital",
            "hartselle medical center", "marshall medical center south"
        ]
        df["HospitalName"] = df["HospitalName"].str.strip().str.lower()
        df["HospitalName"] = df["HospitalName"].apply(
            lambda v: harmonize_category(v, valid_hospitals)
        )
        operations.append("HospitalName: harmonisation des noms")

    # Address1: texte, 0% manquants, nettoyage des espaces
    if "Address1" in df.columns:
        df["Address1"] = df["Address1"].str.strip()
        operations.append("Address1: nettoyage des espaces")

    # Address2/Address3: 100% "empty", pas de traitement
    for col in ["Address2", "Address3"]:
        if col in df.columns:
            df[col] = df[col].replace("empty", np.nan)

    # City: 31% manquants, valeurs fréquentes avec espaces/typos
    if "City" in df.columns:
        valid_cities = [
            "birmingham", "montgomery", "gadsden", "dothan",
            "guntersville", "anniston", "ozark", "boaz"
        ]
        df["City"] = df["City"].str.strip().str.lower()
        df["City"] = df["City"].replace("", np.nan)
        df["City"] = df["City"].apply(lambda v: harmonize_category(v, valid_cities))

        # Imputation conditionnelle
        group_col = find_best_grouping_column(df, "City", False)
        if group_col:
            df["City"] = df.groupby(group_col)["City"].transform(
                lambda s: s.fillna(s.mode().iloc[0] if not s.mode().empty else "unknown")
            )
        df["City"] = df["City"].fillna("unknown")
        operations.append("City: harmonisation et imputation des valeurs manquantes")

    # State: 31.4% manquants, nombreuses variantes de "al"
    if "State" in df.columns:
        valid_states = ["al", "la", "ak"]
        df["State"] = df["State"].str.strip().str.lower()
        df["State"] = df["State"].replace("", np.nan)
        df["State"] = df["State"].apply(lambda v: harmonize_category(v, valid_states))

        # Imputation conditionnelle
        group_col = find_best_grouping_column(df, "State", False)
        if group_col:
            df["State"] = df.groupby(group_col)["State"].transform(
                lambda s: s.fillna(s.mode().iloc[0] if not s.mode().empty else "al")
            )
        df["State"] = df["State"].fillna("al")
        operations.append("State: harmonisation et imputation des valeurs manquantes")

    # ZipCode: texte, 0% manquants, nettoyage des caractères non numériques
    if "ZipCode" in df.columns:
        df["ZipCode"] = df["ZipCode"].astype(str).str.extract(r'(\d{5})')[0]
        df["ZipCode"] = df["ZipCode"].fillna("00000")
        operations.append("ZipCode: nettoyage des codes postaux")

    # CountyName: texte, 0% manquants, harmonisation
    if "CountyName" in df.columns:
        valid_counties = [
            "jefferson", "etowah", "marshall", "marion", "covington",
            "montgomery", "coffee", "houston", "calhoun", "madison"
        ]
        df["CountyName"] = df["CountyName"].str.strip().str.lower()
        df["CountyName"] = df["CountyName"].apply(
            lambda v: harmonize_category(v, valid_counties)
        )
        operations.append("CountyName: harmonisation des noms de comtés")

    # PhoneNumber: numérique, 0% manquants, extraction des chiffres
    if "PhoneNumber" in df.columns:
        df["PhoneNumber"] = df["PhoneNumber"].apply(extract_numeric)
        # Vérification des bornes plausibles (numéros US)
        min_phone = 2000000000
        max_phone = 9999999999
        median_phone = df["PhoneNumber"].median()
        df["PhoneNumber"] = df["PhoneNumber"].where(
            (df["PhoneNumber"] >= min_phone) & (df["PhoneNumber"] <= max_phone),
            median_phone
        )
        operations.append("PhoneNumber: extraction numérique et correction des aberrations")

    # HospitalOwner: 33.9% manquants, variantes textuelles
    if "HospitalOwner" in df.columns:
        valid_owners = [
            "voluntary non-profit - private", "proprietary",
            "government - hospital district or authority",
            "voluntary non-profit - other", "voluntary non-profit - church",
            "government - federal", "government - state", "government - local"
        ]
        df["HospitalOwner"] = df["HospitalOwner"].str.strip().str.lower()
        df["HospitalOwner"] = df["HospitalOwner"].apply(
            lambda v: harmonize_category(v, valid_owners)
        )

        # Imputation conditionnelle
        group_col = find_best_grouping_column(df, "HospitalOwner", False)
        if group_col:
            df["HospitalOwner"] = df.groupby(group_col)["HospitalOwner"].transform(
                lambda s: s.fillna(s.mode().iloc[0] if not s.mode().empty else "unknown")
            )
        df["HospitalOwner"] = df["HospitalOwner"].fillna("unknown")
        operations.append("HospitalOwner: harmonisation et imputation")

    # EmergencyService: 33.1% manquants, variantes de "yes"/"no"
    if "EmergencyService" in df.columns:
        valid_emergency = ["yes", "no"]
        df["EmergencyService"] = df["EmergencyService"].str.strip().str.lower()
        df["EmergencyService"] = df["EmergencyService"].apply(
            lambda v: harmonize_category(v, valid_emergency)
        )

        # Imputation conditionnelle
        group_col = find_best_grouping_column(df, "EmergencyService", False)
        if group_col:
            df["EmergencyService"] = df.groupby(group_col)["EmergencyService"].transform(
                lambda s: s.fillna(s.mode().iloc[0] if not s.mode().empty else "unknown")
            )
        df["EmergencyService"] = df["EmergencyService"].fillna("unknown")
        operations.append("EmergencyService: harmonisation et imputation")

    # Condition: texte, 0% manquants, harmonisation
    if "Condition" in df.columns:
        valid_conditions = [
            "surgical infection prevention", "heart attack",
            "pneumonia", "heart failure", "children s asthma care"
        ]
        df["Condition"] = df["Condition"].str.strip().str.lower()
        df["Condition"] = df["Condition"].apply(
            lambda v: harmonize_category(v, valid_conditions)
        )
        operations.append("Condition: harmonisation des conditions")

    # MeasureCode: texte, 0% manquants, nettoyage
    if "MeasureCode" in df.columns:
        df["MeasureCode"] = df["MeasureCode"].str.strip().str.upper()
        operations.append("MeasureCode: nettoyage des codes")

    # MeasureName: texte, 0% manquants, pas de traitement spécifique

    # Score: texte avec pourcentages, 0% manquants mais valeurs "empty"
    if "Score" in df.columns:
        df["Score"] = df["Score"].str.strip()
        df["Score"] = df["Score"].replace("empty", np.nan)

        # Extraction des pourcentages
        def extract_score(value):
            if pd.isna(value):
                return np.nan
            s = str(value)
            match = re.search(r"(\d+)%", s)
            if match:
                return f"{match.group(1)}%"
            return value

        df["Score"] = df["Score"].apply(extract_score)
        df["Score"] = df["Score"].fillna("unknown")
        operations.append("Score: extraction des pourcentages et imputation")

    # Sample: texte avec "patients", 0% manquants mais valeurs "empty"
    if "Sample" in df.columns:
        df["Sample"] = df["Sample"].str.strip()
        df["Sample"] = df["Sample"].replace("empty", np.nan)

        # Extraction du nombre de patients
        def extract_sample(value):
            if pd.isna(value):
                return np.nan
            s = str(value)
            match = re.search(r"(\d+)\s*patients?", s)
            if match:
                return f"{match.group(1)} patients"
            return value

        df["Sample"] = df["Sample"].apply(extract_sample)
        df["Sample"] = df["Sample"].fillna("0 patients")
        operations.append("Sample: extraction des nombres de patients")

    # Stateavg: texte, 0% manquants, nettoyage
    if "Stateavg" in df.columns:
        df["Stateavg"] = df["Stateavg"].str.strip().str.lower()
        operations.append("Stateavg: nettoyage des valeurs")

    # Sauvegarde du dataset nettoyé
    df.to_csv(OUTPUT_PATH, index=False)

    # Résumé des opérations
    print("\nRésumé des opérations de nettoyage:")
    for op in operations:
        print(f"- {op}")
    print(f"\nDataset nettoyé sauvegardé dans: {OUTPUT_PATH}")
    print(f"Nombre final de lignes: {len(df)}")
    print(f"Nombre final de colonnes: {len(df.columns)}")

if __name__ == "__main__":
    clean_hospital_data()