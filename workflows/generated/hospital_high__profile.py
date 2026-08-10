import pandas as pd
import numpy as np
import re
from datetime import datetime
import difflib

INPUT_PATH = r'datasets\hospital\noisy_high.csv'
OUTPUT_PATH = r'results\cleaned_datasets\hospital\noisy_high__profile.csv'

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

def clean_dataset():
    df = pd.read_csv(INPUT_PATH)

    # Conservation de row_id comme identifiant technique
    if 'row_id' not in df.columns:
        raise ValueError("La colonne row_id est absente du dataset")

    # Suppression des doublons (en conservant la première occurrence)
    initial_rows = len(df)
    df = df.drop_duplicates(subset=[col for col in df.columns if col != 'row_id'], keep='first')
    duplicates_removed = initial_rows - len(df)
    print(f"Doublons supprimés: {duplicates_removed}")

    # Nettoyage colonne par colonne
    operations = []

    # ProviderNumber - numérique, pas de manquants, pas d'aberrations évidentes
    if 'ProviderNumber' in df.columns:
        df['ProviderNumber'] = df['ProviderNumber'].apply(extract_numeric)
        operations.append("ProviderNumber: valeurs numériques extraites")

    # HospitalName - catégoriel, pas de manquants, harmonisation des casse/espaces
    if 'HospitalName' in df.columns:
        valid_hospitals = [
            "riverview regional medical center", "georgiana hospital", "stringfellow memorial hospital",
            "cherokee medical center", "helen keller memorial hospital", "southwest alabama medical center",
            "g h lanier memorial hospital", "mizell memorial hospital", "hartselle medical center",
            "marshall medical center south"
        ]
        df['HospitalName'] = df['HospitalName'].str.strip().str.lower()
        df['HospitalName'] = df['HospitalName'].apply(lambda v: harmonize_category(v, valid_hospitals))
        operations.append("HospitalName: harmonisation des noms d'hôpitaux")

    # Address1 - catégoriel, pas de manquants, nettoyage des espaces
    if 'Address1' in df.columns:
        df['Address1'] = df['Address1'].str.strip()
        operations.append("Address1: nettoyage des espaces")

    # Address2 et Address3 - toutes les valeurs sont "empty", pas de traitement
    for col in ['Address2', 'Address3']:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: np.nan if str(x).strip().lower() == 'empty' else x)

    # City - 31% de manquants, imputation conditionnelle
    if 'City' in df.columns:
        valid_cities = [
            "birmingham", "montgomery", "gadsden", "dothan", "guntersville",
            "anniston", "ozark", "boaz", "mobile", "huntsville"
        ]
        df['City'] = df['City'].str.strip().str.lower()
        df['City'] = df['City'].replace('', np.nan)

        # Imputation conditionnelle par CountyName
        group_col = find_best_grouping_column(df, 'City', False)
        if group_col:
            df['City'] = df.groupby(group_col)['City'].transform(
                lambda s: s.fillna(s.mode().iloc[0] if not s.mode().empty else "unknown")
            )
        df['City'] = df['City'].fillna("unknown")
        df['City'] = df['City'].apply(lambda v: harmonize_category(v, valid_cities))
        operations.append("City: imputation des valeurs manquantes et harmonisation")

    # State - 31.4% de manquants, valeurs aberrantes ("l", "aal", "NNA")
    if 'State' in df.columns:
        valid_states = ["al", "la", "ak"]
        df['State'] = df['State'].str.strip().str.lower()
        df['State'] = df['State'].replace('', np.nan)

        # Correction des valeurs aberrantes
        df['State'] = df['State'].apply(lambda x: x if x in valid_states else np.nan)

        # Imputation conditionnelle par CountyName
        group_col = find_best_grouping_column(df, 'State', False)
        if group_col:
            df['State'] = df.groupby(group_col)['State'].transform(
                lambda s: s.fillna(s.mode().iloc[0] if not s.mode().empty else "al")
            )
        df['State'] = df['State'].fillna("al")
        operations.append("State: correction des valeurs aberrantes et imputation")

    # ZipCode - catégoriel, pas de manquants, nettoyage des formats
    if 'ZipCode' in df.columns:
        df['ZipCode'] = df['ZipCode'].astype(str).str.strip()
        df['ZipCode'] = df['ZipCode'].apply(lambda x: x if re.match(r'^\d{5}$', x) else np.nan)
        df['ZipCode'] = df['ZipCode'].fillna("00000")
        operations.append("ZipCode: nettoyage des formats")

    # CountyName - catégoriel, pas de manquants, harmonisation
    if 'CountyName' in df.columns:
        valid_counties = [
            "jefferson", "etowah", "marshall", "marion", "covington",
            "montgomery", "coffee", "houston", "calhoun", "madison"
        ]
        df['CountyName'] = df['CountyName'].str.strip().str.lower()
        df['CountyName'] = df['CountyName'].apply(lambda v: harmonize_category(v, valid_counties))
        operations.append("CountyName: harmonisation des noms de comtés")

    # PhoneNumber - numérique, pas de manquants, extraction des chiffres
    if 'PhoneNumber' in df.columns:
        df['PhoneNumber'] = df['PhoneNumber'].apply(extract_numeric)
        # Vérification des bornes plausibles (numéros US)
        df['PhoneNumber'] = df['PhoneNumber'].apply(
            lambda x: x if 2000000000 <= x <= 9999999999 else np.nan
        )
        df['PhoneNumber'] = df['PhoneNumber'].fillna(df['PhoneNumber'].median())
        operations.append("PhoneNumber: extraction des numéros et correction des valeurs aberrantes")

    # HospitalType - catégoriel, pas de manquants, harmonisation
    if 'HospitalType' in df.columns:
        valid_types = ["acute care hospitals"]
        df['HospitalType'] = df['HospitalType'].str.strip().str.lower()
        df['HospitalType'] = df['HospitalType'].apply(lambda v: harmonize_category(v, valid_types))
        operations.append("HospitalType: harmonisation")

    # HospitalOwner - 33.9% de manquants, valeurs aberrantes ("  ", "unknown")
    if 'HospitalOwner' in df.columns:
        valid_owners = [
            "voluntary non-profit - private", "proprietary", "government - hospital district or authority",
            "voluntary non-profit - other", "voluntary non-profit - church", "government - federal",
            "government - state", "government - local"
        ]
        df['HospitalOwner'] = df['HospitalOwner'].str.strip().str.lower()
        df['HospitalOwner'] = df['HospitalOwner'].replace('', np.nan)

        # Imputation conditionnelle par CountyName
        group_col = find_best_grouping_column(df, 'HospitalOwner', False)
        if group_col:
            df['HospitalOwner'] = df.groupby(group_col)['HospitalOwner'].transform(
                lambda s: s.fillna(s.mode().iloc[0] if not s.mode().empty else "unknown")
            )
        df['HospitalOwner'] = df['HospitalOwner'].fillna("unknown")
        df['HospitalOwner'] = df['HospitalOwner'].apply(lambda v: harmonize_category(v, valid_owners))
        operations.append("HospitalOwner: imputation et harmonisation")

    # EmergencyService - 33.1% de manquants, valeurs aberrantes ("  ", "unknown")
    if 'EmergencyService' in df.columns:
        valid_emergency = ["yes", "no"]
        df['EmergencyService'] = df['EmergencyService'].str.strip().str.lower()
        df['EmergencyService'] = df['EmergencyService'].replace('', np.nan)

        # Imputation conditionnelle par HospitalOwner
        group_col = find_best_grouping_column(df, 'EmergencyService', False)
        if group_col:
            df['EmergencyService'] = df.groupby(group_col)['EmergencyService'].transform(
                lambda s: s.fillna(s.mode().iloc[0] if not s.mode().empty else "yes")
            )
        df['EmergencyService'] = df['EmergencyService'].fillna("yes")
        df['EmergencyService'] = df['EmergencyService'].apply(lambda v: harmonize_category(v, valid_emergency))
        operations.append("EmergencyService: imputation et harmonisation")

    # Condition - catégoriel, pas de manquants, harmonisation
    if 'Condition' in df.columns:
        valid_conditions = [
            "surgical infection prevention", "heart attack", "pneumonia",
            "heart failure", "children s asthma care"
        ]
        df['Condition'] = df['Condition'].str.strip().str.lower()
        df['Condition'] = df['Condition'].apply(lambda v: harmonize_category(v, valid_conditions))
        operations.append("Condition: harmonisation")

    # MeasureCode - catégoriel, pas de manquants, harmonisation
    if 'MeasureCode' in df.columns:
        valid_codes = [
            "ami-2", "hf-3", "hf-4", "pn-2", "hf-2", "pn-3b", "hf-1", "pn-4",
            "ami-3", "scip-card-2", "scip-inf-1", "scip-inf-2", "scip-inf-3",
            "scip-inf-4", "scip-ven-1", "scip-ven-2", "scip-ven-6", "ami-1",
            "ami-7a", "ami-8a", "hf-5", "pn-6", "pn-7", "scip-card-3"
        ]
        df['MeasureCode'] = df['MeasureCode'].str.strip().str.lower()
        df['MeasureCode'] = df['MeasureCode'].apply(lambda v: harmonize_category(v, valid_codes))
        operations.append("MeasureCode: harmonisation")

    # MeasureName - catégoriel, pas de manquants, nettoyage des espaces
    if 'MeasureName' in df.columns:
        df['MeasureName'] = df['MeasureName'].str.strip()
        operations.append("MeasureName: nettoyage des espaces")

    # Score - catégoriel avec valeurs numériques, extraction des pourcentages
    if 'Score' in df.columns:
        def extract_score(value):
            if pd.isna(value) or str(value).strip().lower() == 'empty':
                return np.nan
            s = str(value).strip()
            match = re.search(r'(\d+)%', s)
            if match:
                return int(match.group(1))
            return np.nan

        df['Score'] = df['Score'].apply(extract_score)
        # Imputation des valeurs manquantes
        df['Score'] = df['Score'].fillna(df['Score'].median())
        operations.append("Score: extraction des pourcentages et imputation")

    # Sample - catégoriel avec valeurs numériques, extraction des nombres
    if 'Sample' in df.columns:
        def extract_sample(value):
            if pd.isna(value) or str(value).strip().lower() == 'empty':
                return np.nan
            s = str(value).strip()
            match = re.search(r'(\d+)\s*patients?', s)
            if match:
                return int(match.group(1))
            return np.nan

        df['Sample'] = df['Sample'].apply(extract_sample)
        # Imputation des valeurs manquantes
        df['Sample'] = df['Sample'].fillna(df['Sample'].median())
        operations.append("Sample: extraction des nombres de patients et imputation")

    # Stateavg - catégoriel, pas de manquants, nettoyage des formats
    if 'Stateavg' in df.columns:
        df['Stateavg'] = df['Stateavg'].str.strip().str.lower()
        operations.append("Stateavg: nettoyage des formats")

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