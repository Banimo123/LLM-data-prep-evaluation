import pandas as pd
import numpy as np
import re
from datetime import datetime
import difflib

INPUT_PATH = r'datasets\hospital\noisy_medium.csv'
OUTPUT_PATH = r'results\cleaned_datasets\hospital\noisy_medium__profile.csv'

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
    if str(value).isdigit() and len(str(value)) in (9, 10):
        try:
            return datetime.fromtimestamp(int(value)).strftime("%Y-%m-%d")
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

    # Conservation de row_id intact
    original_row_ids = df['row_id'].copy()

    # Suppression des doublons (conservation du premier)
    initial_rows = len(df)
    df.drop_duplicates(inplace=True)
    duplicates_removed = initial_rows - len(df)

    # Nettoyage colonne par colonne
    # ProviderNumber: numérique, pas de manquants -> extraction numérique
    if 'ProviderNumber' in df.columns:
        df['ProviderNumber'] = df['ProviderNumber'].apply(extract_numeric)
        df['ProviderNumber'] = pd.to_numeric(df['ProviderNumber'], errors='coerce')
        df['ProviderNumber'] = df['ProviderNumber'].fillna(df['ProviderNumber'].median())

    # HospitalName: catégoriel, pas de manquants -> harmonisation
    if 'HospitalName' in df.columns:
        valid_hospitals = [
            "stringfellow memorial hospital", "riverview regional medical center",
            "mizell memorial hospital", "shelby baptist medical center",
            "callahan eye foundation hospital", "g h lanier memorial hospital",
            "east alabama medical center and snf", "cherokee medical center",
            "huntsville hospital", "medical center enterprise"
        ]
        df['HospitalName'] = df['HospitalName'].apply(lambda v: harmonize_category(v, valid_hospitals))

    # Address1: catégoriel, pas de manquants -> harmonisation
    if 'Address1' in df.columns:
        valid_addresses = [
            "301 east 18th st", "101 sivley rd", "600 south third street",
            "1300 south montgomery avenue", "1720 university blvd", "702 n main st",
            "101 hospital circle", "201 pine street northwest", "8000 alabama highway 69",
            "50 medical park east drive"
        ]
        df['Address1'] = df['Address1'].apply(lambda v: harmonize_category(v, valid_addresses))

    # Address2 et Address3: toutes "empty" -> pas de traitement
    if 'Address2' in df.columns:
        df['Address2'] = df['Address2'].apply(lambda x: 'empty' if str(x).strip().lower() == 'empty' else x)
    if 'Address3' in df.columns:
        df['Address3'] = df['Address3'].apply(lambda x: 'empty' if str(x).strip().lower() == 'empty' else x)

    # City: 16.8% manquants -> imputation conditionnelle
    if 'City' in df.columns:
        group_col = find_best_grouping_column(df, 'City', False)
        if group_col:
            df['City'] = df.groupby(group_col)['City'].transform(
                lambda s: s.fillna(s.mode().iloc[0] if not s.mode().empty else "Unknown")
            )
        df['City'] = df['City'].fillna(df['City'].mode().iloc[0])
        valid_cities = [
            "birmingham", "montgomery", "gadsden", "dothan", "huntsville",
            "thomasville", "boaz", "elba", "valley", "andalusia"
        ]
        df['City'] = df['City'].apply(lambda v: harmonize_category(v, valid_cities))

    # State: 15.9% manquants -> imputation conditionnelle + harmonisation
    if 'State' in df.columns:
        group_col = find_best_grouping_column(df, 'State', False)
        if group_col:
            df['State'] = df.groupby(group_col)['State'].transform(
                lambda s: s.fillna(s.mode().iloc[0] if not s.mode().empty else "Unknown")
            )
        df['State'] = df['State'].fillna(df['State'].mode().iloc[0])
        valid_states = ["al", "ak", "la"]
        df['State'] = df['State'].apply(lambda v: harmonize_category(v, valid_states))

    # ZipCode: catégoriel, pas de manquants -> harmonisation
    if 'ZipCode' in df.columns:
        df['ZipCode'] = df['ZipCode'].astype(str)
        df['ZipCode'] = df['ZipCode'].apply(lambda x: x.strip() if len(x.strip()) == 5 else x)

    # CountyName: catégoriel, pas de manquants -> harmonisation
    if 'CountyName' in df.columns:
        valid_counties = [
            "jefferson", "etowah", "marshall", "marion", "covington",
            "montgomery", "coffee", "houston", "calhoun", "madison"
        ]
        df['CountyName'] = df['CountyName'].apply(lambda v: harmonize_category(v, valid_counties))

    # PhoneNumber: numérique, pas de manquants -> extraction numérique
    if 'PhoneNumber' in df.columns:
        df['PhoneNumber'] = df['PhoneNumber'].apply(extract_numeric)
        df['PhoneNumber'] = pd.to_numeric(df['PhoneNumber'], errors='coerce')
        # Bornes physiques plausibles pour un numéro de téléphone US
        min_phone = 2000000000
        max_phone = 9999999999
        median_phone = df['PhoneNumber'].median()
        df['PhoneNumber'] = df['PhoneNumber'].apply(
            lambda x: median_phone if x < min_phone or x > max_phone else x
        )
        df['PhoneNumber'] = df['PhoneNumber'].fillna(median_phone)

    # HospitalType: toutes "acute care hospitals" -> pas de traitement
    if 'HospitalType' in df.columns:
        df['HospitalType'] = df['HospitalType'].apply(
            lambda x: "acute care hospitals" if str(x).strip().lower() == "acute care hospitals" else x
        )

    # HospitalOwner: 17.1% manquants -> imputation conditionnelle
    if 'HospitalOwner' in df.columns:
        group_col = find_best_grouping_column(df, 'HospitalOwner', False)
        if group_col:
            df['HospitalOwner'] = df.groupby(group_col)['HospitalOwner'].transform(
                lambda s: s.fillna(s.mode().iloc[0] if not s.mode().empty else "Unknown")
            )
        df['HospitalOwner'] = df['HospitalOwner'].fillna(df['HospitalOwner'].mode().iloc[0])
        valid_owners = [
            "voluntary non-profit - private", "proprietary", "government - hospital district or authority",
            "voluntary non-profit - other", "voluntary non-profit - church", "government - federal",
            "government - state", "government - local", "unknown"
        ]
        df['HospitalOwner'] = df['HospitalOwner'].apply(lambda v: harmonize_category(v, valid_owners))

    # EmergencyService: 16.1% manquants -> imputation conditionnelle
    if 'EmergencyService' in df.columns:
        group_col = find_best_grouping_column(df, 'EmergencyService', False)
        if group_col:
            df['EmergencyService'] = df.groupby(group_col)['EmergencyService'].transform(
                lambda s: s.fillna(s.mode().iloc[0] if not s.mode().empty else "Unknown")
            )
        df['EmergencyService'] = df['EmergencyService'].fillna(df['EmergencyService'].mode().iloc[0])
        valid_emergency = ["yes", "no", "unknown"]
        df['EmergencyService'] = df['EmergencyService'].apply(lambda v: harmonize_category(v, valid_emergency))

    # Condition: catégoriel, pas de manquants -> harmonisation
    if 'Condition' in df.columns:
        valid_conditions = [
            "surgical infection prevention", "heart attack", "pneumonia",
            "heart failure", "children s asthma care"
        ]
        df['Condition'] = df['Condition'].apply(lambda v: harmonize_category(v, valid_conditions))

    # MeasureCode: catégoriel, pas de manquants -> harmonisation
    if 'MeasureCode' in df.columns:
        valid_codes = [
            "ami-2", "hf-3", "hf-4", "pn-2", "hf-2", "pn-3b", "hf-1", "pn-4",
            "ami-3", "scip-card-2", "scip-inf-1", "scip-inf-2", "scip-inf-3", "scip-inf-4"
        ]
        df['MeasureCode'] = df['MeasureCode'].apply(lambda v: harmonize_category(v, valid_codes))

    # MeasureName: catégoriel, pas de manquants -> pas de traitement (trop long pour harmoniser)

    # Score: format "%" ou "empty" -> conservation du format
    if 'Score' in df.columns:
        df['Score'] = df['Score'].apply(
            lambda x: x if str(x).endswith('%') or str(x).strip().lower() == 'empty' else x
        )
        # Extraction numérique pour détection d'aberrations
        scores_numeric = df['Score'].apply(
            lambda x: extract_numeric(x) if str(x).endswith('%') else np.nan
        )
        median_score = scores_numeric.median()
        # Remplacement des valeurs aberrantes (hors 0-100)
        df['Score'] = df['Score'].apply(
            lambda x: f"{median_score}%" if str(x).endswith('%') and (extract_numeric(x) < 0 or extract_numeric(x) > 100) else x
        )

    # Sample: format "X patients" ou "empty" -> conservation du format
    if 'Sample' in df.columns:
        df['Sample'] = df['Sample'].apply(
            lambda x: x if str(x).endswith('patients') or str(x).strip().lower() == 'empty' else x
        )
        # Extraction numérique pour détection d'aberrations
        samples_numeric = df['Sample'].apply(
            lambda x: extract_numeric(x) if str(x).endswith('patients') else np.nan
        )
        median_sample = samples_numeric.median()
        # Remplacement des valeurs aberrantes (négatives)
        df['Sample'] = df['Sample'].apply(
            lambda x: f"{int(median_sample)} patients" if str(x).endswith('patients') and extract_numeric(x) < 0 else x
        )

    # Stateavg: format "XX_YY-Z" -> pas de traitement

    # Restauration de row_id original
    df['row_id'] = original_row_ids.loc[df.index]

    # Sauvegarde
    df.to_csv(OUTPUT_PATH, index=False)

    # Log des opérations
    print(f"Nettoyage terminé. Résumé:")
    print(f"- Lignes initiales: {initial_rows}")
    print(f"- Doublons supprimés: {duplicates_removed}")
    print(f"- Lignes finales: {len(df)}")
    print(f"- Fichier sauvegardé: {OUTPUT_PATH}")

if __name__ == "__main__":
    clean_hospital_data()