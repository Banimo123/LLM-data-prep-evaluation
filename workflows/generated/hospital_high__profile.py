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

    # Suppression des doublons (conservation de la première occurrence)
    initial_rows = len(df)
    df = df.drop_duplicates(subset=[col for col in df.columns if col != "row_id"], keep='first')
    duplicates_removed = initial_rows - len(df)
    print(f"Doublons supprimés: {duplicates_removed}")

    # Nettoyage colonne par colonne
    operations_log = []

    # ProviderNumber - numérique, pas de manquants, pas d'aberrations apparentes
    if 'ProviderNumber' in df.columns:
        df['ProviderNumber'] = pd.to_numeric(df['ProviderNumber'], errors='coerce')
        # Vérification des bornes (min/max du profil)
        min_val, max_val = 10001.0, 20018.0
        outliers = df[(df['ProviderNumber'] < min_val) | (df['ProviderNumber'] > max_val)]
        if not outliers.empty:
            median_val = df['ProviderNumber'].median()
            df.loc[(df['ProviderNumber'] < min_val) | (df['ProviderNumber'] > max_val), 'ProviderNumber'] = median_val
            operations_log.append(f"ProviderNumber: {len(outliers)} valeurs aberrantes corrigées par médiane")

    # HospitalName - catégoriel, pas de manquants, harmonisation des casse/espaces
    if 'HospitalName' in df.columns:
        valid_hospitals = [
            "riverview regional medical center", "georgiana hospital", "stringfellow memorial hospital",
            "cherokee medical center", "helen keller memorial hospital", "southwest alabama medical center",
            "g h lanier memorial hospital", "mizell memorial hospital", "hartselle medical center",
            "marshall medical center south"
        ]
        df['HospitalName'] = df['HospitalName'].apply(
            lambda v: harmonize_category(v, valid_hospitals) if pd.notna(v) else v
        )
        operations_log.append("HospitalName: harmonisation des variantes de nom")

    # Address1 - catégoriel, pas de manquants, nettoyage des espaces
    if 'Address1' in df.columns:
        df['Address1'] = df['Address1'].str.strip()
        operations_log.append("Address1: nettoyage des espaces superflus")

    # Address2 et Address3 - toutes les valeurs sont "empty", conservation telle quelle
    for col in ['Address2', 'Address3']:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: "empty" if str(x).strip().lower() == "empty" else x)

    # City - 31% de manquants, imputation conditionnelle si possible
    if 'City' in df.columns:
        missing_before = df['City'].isna().sum()
        group_col = find_best_grouping_column(df, 'City', False)
        if group_col:
            df['City'] = df.groupby(group_col)['City'].transform(
                lambda s: s.fillna(s.mode().iloc[0] if not s.mode().empty else "unknown")
            )
        df['City'] = df['City'].fillna("unknown")
        missing_after = df['City'].isna().sum()
        operations_log.append(f"City: {missing_before - missing_after} valeurs manquantes imputées")

        # Harmonisation des valeurs fréquentes
        valid_cities = ["birmingham", "montgomery", "gadsden", "dothan", "guntersville",
                       "anniston", "ozark", "boaz", "unknown"]
        df['City'] = df['City'].apply(lambda v: harmonize_category(v, valid_cities) if pd.notna(v) else v)

    # State - 31.4% de manquants, imputation conditionnelle
    if 'State' in df.columns:
        missing_before = df['State'].isna().sum()
        group_col = find_best_grouping_column(df, 'State', False)
        if group_col:
            df['State'] = df.groupby(group_col)['State'].transform(
                lambda s: s.fillna(s.mode().iloc[0] if not s.mode().empty else "al")
            )
        df['State'] = df['State'].fillna("al")

        # Harmonisation des variantes de "al"
        valid_states = ["al", "la", "ak"]
        df['State'] = df['State'].apply(
            lambda v: harmonize_category(v, valid_states) if pd.notna(v) else v
        )
        operations_log.append(f"State: {missing_before - df['State'].isna().sum()} valeurs manquantes imputées")

    # ZipCode - catégoriel, pas de manquants, nettoyage des formats
    if 'ZipCode' in df.columns:
        df['ZipCode'] = df['ZipCode'].astype(str).str.strip()
        df['ZipCode'] = df['ZipCode'].apply(
            lambda x: x if re.match(r'^\d{5}(-\d{4})?$', x) else np.nan
        )
        df['ZipCode'] = df['ZipCode'].fillna("unknown")
        operations_log.append("ZipCode: nettoyage des formats invalides")

    # CountyName - catégoriel, pas de manquants, harmonisation
    if 'CountyName' in df.columns:
        valid_counties = [
            "jefferson", "etowah", "marshall", "marion", "covington",
            "montgomery", "coffee", "houston", "calhoun", "madison"
        ]
        df['CountyName'] = df['CountyName'].apply(
            lambda v: harmonize_category(v, valid_counties) if pd.notna(v) else v
        )

    # PhoneNumber - numérique, pas de manquants, extraction des chiffres
    if 'PhoneNumber' in df.columns:
        df['PhoneNumber'] = df['PhoneNumber'].apply(extract_numeric)
        # Vérification des bornes (min/max du profil)
        min_val, max_val = 2052743000.0, 9075436300.0
        outliers = df[(df['PhoneNumber'] < min_val) | (df['PhoneNumber'] > max_val)]
        if not outliers.empty:
            median_val = df['PhoneNumber'].median()
            df.loc[(df['PhoneNumber'] < min_val) | (df['PhoneNumber'] > max_val), 'PhoneNumber'] = median_val
            operations_log.append(f"PhoneNumber: {len(outliers)} valeurs aberrantes corrigées par médiane")

    # HospitalOwner - 33.9% de manquants, imputation conditionnelle
    if 'HospitalOwner' in df.columns:
        missing_before = df['HospitalOwner'].isna().sum()
        group_col = find_best_grouping_column(df, 'HospitalOwner', False)
        if group_col:
            df['HospitalOwner'] = df.groupby(group_col)['HospitalOwner'].transform(
                lambda s: s.fillna(s.mode().iloc[0] if not s.mode().empty else "unknown")
            )
        df['HospitalOwner'] = df['HospitalOwner'].fillna("unknown")

        valid_owners = [
            "voluntary non-profit - private", "proprietary", "government - hospital district or authority",
            "voluntary non-profit - other", "voluntary non-profit - church", "government - federal",
            "government - state", "government - local", "unknown"
        ]
        df['HospitalOwner'] = df['HospitalOwner'].apply(
            lambda v: harmonize_category(v, valid_owners) if pd.notna(v) else v
        )
        operations_log.append(f"HospitalOwner: {missing_before - df['HospitalOwner'].isna().sum()} valeurs manquantes imputées")

    # EmergencyService - 33.1% de manquants, imputation conditionnelle
    if 'EmergencyService' in df.columns:
        missing_before = df['EmergencyService'].isna().sum()
        group_col = find_best_grouping_column(df, 'EmergencyService', False)
        if group_col:
            df['EmergencyService'] = df.groupby(group_col)['EmergencyService'].transform(
                lambda s: s.fillna(s.mode().iloc[0] if not s.mode().empty else "yes")
            )
        df['EmergencyService'] = df['EmergencyService'].fillna("yes")

        valid_emergency = ["yes", "no", "unknown"]
        df['EmergencyService'] = df['EmergencyService'].apply(
            lambda v: harmonize_category(v, valid_emergency) if pd.notna(v) else v
        )
        operations_log.append(f"EmergencyService: {missing_before - df['EmergencyService'].isna().sum()} valeurs manquantes imputées")

    # Condition - catégoriel, pas de manquants, harmonisation
    if 'Condition' in df.columns:
        valid_conditions = [
            "surgical infection prevention", "heart attack", "pneumonia",
            "heart failure", "children s asthma care"
        ]
        df['Condition'] = df['Condition'].apply(
            lambda v: harmonize_category(v, valid_conditions) if pd.notna(v) else v
        )

    # MeasureName - catégoriel, pas de manquants, nettoyage des espaces
    if 'MeasureName' in df.columns:
        df['MeasureName'] = df['MeasureName'].str.strip()

    # Score - catégoriel avec format "%", conservation du format
    if 'Score' in df.columns:
        # Extraction des valeurs numériques pour les valeurs avec "%"
        def clean_score(value):
            if pd.isna(value) or str(value).strip().lower() == "empty":
                return value
            s = str(value).strip()
            if "%" in s:
                num = extract_numeric(s)
                return f"{int(num)}%" if not pd.isna(num) else s
            return value

        df['Score'] = df['Score'].apply(clean_score)
        # Imputation des valeurs manquantes (conservation de "empty" comme valeur valide)
        missing_scores = df[df['Score'].isin(["empty", np.nan])]
        if not missing_scores.empty:
            mode_score = df[~df['Score'].isin(["empty", np.nan])]['Score'].mode()[0]
            df.loc[df['Score'].isin(["empty", np.nan]), 'Score'] = mode_score
            operations_log.append(f"Score: {len(missing_scores)} valeurs manquantes imputées par mode")

    # Sample - catégoriel avec format "X patients", conservation du format
    if 'Sample' in df.columns:
        def clean_sample(value):
            if pd.isna(value) or str(value).strip().lower() == "empty":
                return value
            s = str(value).strip()
            if "patients" in s:
                num = extract_numeric(s)
                return f"{int(num)} patients" if not pd.isna(num) else s
            return value

        df['Sample'] = df['Sample'].apply(clean_sample)
        # Imputation des valeurs manquantes (conservation de "empty" comme valeur valide)
        missing_samples = df[df['Sample'].isin(["empty", np.nan])]
        if not missing_samples.empty:
            mode_sample = df[~df['Sample'].isin(["empty", np.nan])]['Sample'].mode()[0]
            df.loc[df['Sample'].isin(["empty", np.nan]), 'Sample'] = mode_sample
            operations_log.append(f"Sample: {len(missing_samples)} valeurs manquantes imputées par mode")

    # Stateavg - catégoriel, pas de manquants, nettoyage des espaces
    if 'Stateavg' in df.columns:
        df['Stateavg'] = df['Stateavg'].str.strip()

    # Sauvegarde du dataset nettoyé
    df.to_csv(OUTPUT_PATH, index=False)

    # Log des opérations
    print("\nRésumé des opérations de nettoyage:")
    for op in operations_log:
        print(f"- {op}")
    print(f"\nDataset nettoyé sauvegardé dans: {OUTPUT_PATH}")
    print(f"Nombre final de lignes: {len(df)}")
    print(f"Nombre final de colonnes: {len(df.columns)}")

if __name__ == "__main__":
    clean_hospital_data()