import pandas as pd
import numpy as np
import re
from datetime import datetime
import difflib

INPUT_PATH = r'datasets\hospital\noisy_low.csv'
OUTPUT_PATH = r'results\cleaned_datasets\hospital\noisy_low__profile.csv'

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
    row_ids = df['row_id'].copy()

    # Suppression des doublons (en conservant la première occurrence)
    initial_rows = len(df)
    df.drop_duplicates(inplace=True)
    final_rows = len(df)
    duplicates_removed = initial_rows - final_rows

    # Nettoyage colonne par colonne
    # ProviderNumber: pas de valeurs manquantes, pas de nettoyage nécessaire
    if 'ProviderNumber' in df.columns:
        pass

    # HospitalName: pas de valeurs manquantes, harmonisation des fautes de frappe
    if 'HospitalName' in df.columns:
        valid_hospital_names = [
            "stringfellow memorial hospital", "huntsville hospital", "marshall medical center south",
            "helen keller memorial hospital", "callahan eye foundation hospital",
            "gadsden regional medical center", "chilton medical center", "st vincents east",
            "eliza coffee memorial hospital", "shelby baptist medical center"
        ]
        df['HospitalName'] = df['HospitalName'].apply(lambda v: harmonize_category(v, valid_hospital_names))

    # Address1: pas de valeurs manquantes, pas de nettoyage spécifique
    if 'Address1' in df.columns:
        pass

    # Address2 et Address3: toutes les valeurs sont "empty", pas de nettoyage
    if 'Address2' in df.columns:
        pass
    if 'Address3' in df.columns:
        pass

    # City: harmonisation des fautes de frappe
    if 'City' in df.columns:
        valid_cities = [
            "birmingham", "gadsden", "montgomery", "dothan", "huntsville", "anniston",
            "centre", "thomasville", "clanton", "opelika"
        ]
        df['City'] = df['City'].apply(lambda v: harmonize_category(v, valid_cities))

    # State: harmonisation des codes états (al, ak, xl, ax -> probablement des erreurs)
    if 'State' in df.columns:
        valid_states = ["al", "ak"]
        df['State'] = df['State'].apply(lambda v: harmonize_category(v, valid_states))

    # ZipCode: pas de valeurs manquantes, pas de nettoyage spécifique
    if 'ZipCode' in df.columns:
        pass

    # CountyName: harmonisation des fautes de frappe
    if 'CountyName' in df.columns:
        valid_counties = [
            "jefferson", "etowah", "marion", "covington", "marshall", "coffee",
            "montgomery", "houston", "calhoun", "madison"
        ]
        df['CountyName'] = df['CountyName'].apply(lambda v: harmonize_category(v, valid_counties))

    # PhoneNumber: pas de valeurs manquantes, pas de nettoyage spécifique
    if 'PhoneNumber' in df.columns:
        pass

    # HospitalType: harmonisation des fautes de frappe
    if 'HospitalType' in df.columns:
        valid_hospital_types = ["acute care hospitals"]
        df['HospitalType'] = df['HospitalType'].apply(lambda v: harmonize_category(v, valid_hospital_types))

    # HospitalOwner: harmonisation des fautes de frappe
    if 'HospitalOwner' in df.columns:
        valid_owners = [
            "voluntary non-profit - private", "proprietary", "government - hospital district or authority",
            "voluntary non-profit - other", "voluntary non-profit - church", "government - federal",
            "government - state", "government - local"
        ]
        df['HospitalOwner'] = df['HospitalOwner'].apply(lambda v: harmonize_category(v, valid_owners))

    # EmergencyService: harmonisation des fautes de frappe
    if 'EmergencyService' in df.columns:
        valid_emergency = ["yes", "no"]
        df['EmergencyService'] = df['EmergencyService'].apply(lambda v: harmonize_category(v, valid_emergency))

    # Condition: harmonisation des fautes de frappe
    if 'Condition' in df.columns:
        valid_conditions = [
            "surgical infection prevention", "heart attack", "pneumonia", "heart failure",
            "children s asthma care"
        ]
        df['Condition'] = df['Condition'].apply(lambda v: harmonize_category(v, valid_conditions))

    # MeasureCode: pas de valeurs manquantes, pas de nettoyage spécifique
    if 'MeasureCode' in df.columns:
        pass

    # MeasureName: pas de valeurs manquantes, pas de nettoyage spécifique
    if 'MeasureName' in df.columns:
        pass

    # Score: extraction des pourcentages et imputation des "empty"
    if 'Score' in df.columns:
        df['Score'] = df['Score'].apply(lambda x: x if x == "empty" else x)
        df['_score_numeric'] = df['Score'].apply(lambda x: extract_numeric(x) if x != "empty" else np.nan)
        group_col = find_best_grouping_column(df, '_score_numeric', True)
        if group_col:
            df['_score_numeric'] = df.groupby(group_col)['_score_numeric'].transform(
                lambda s: s.fillna(s.median())
            )
        df['_score_numeric'] = df['_score_numeric'].fillna(df['_score_numeric'].median())
        df['Score'] = df.apply(lambda row: f"{int(row['_score_numeric'])}%" if pd.notna(row['_score_numeric']) and row['Score'] == "empty" else row['Score'], axis=1)
        df.drop('_score_numeric', axis=1, inplace=True)

    # Sample: extraction des nombres de patients et imputation des "empty"
    if 'Sample' in df.columns:
        df['_sample_numeric'] = df['Sample'].apply(lambda x: extract_numeric(x) if x != "empty" else np.nan)
        group_col = find_best_grouping_column(df, '_sample_numeric', True)
        if group_col:
            df['_sample_numeric'] = df.groupby(group_col)['_sample_numeric'].transform(
                lambda s: s.fillna(s.mode().iloc[0] if not s.mode().empty else 0)
            )
        df['_sample_numeric'] = df['_sample_numeric'].fillna(df['_sample_numeric'].mode().iloc[0])
        df['Sample'] = df.apply(lambda row: f"{int(row['_sample_numeric'])} patients" if pd.notna(row['_sample_numeric']) and row['Sample'] == "empty" else row['Sample'], axis=1)
        df.drop('_sample_numeric', axis=1, inplace=True)

    # Stateavg: pas de valeurs manquantes, pas de nettoyage spécifique
    if 'Stateavg' in df.columns:
        pass

    # Restauration de row_id
    df['row_id'] = row_ids.loc[df.index]

    # Vérification finale qu'aucune colonne n'a été supprimée
    assert 'row_id' in df.columns, "row_id must be preserved"

    # Sauvegarde du dataset nettoyé
    df.to_csv(OUTPUT_PATH, index=False)

    # Log des opérations
    print(f"Nettoyage terminé. Lignes initiales: {initial_rows}, lignes finales: {final_rows}, doublons supprimés: {duplicates_removed}")
    print(f"Valeurs manquantes traitées:")
    for col in df.columns:
        if col != 'row_id':
            missing_before = initial_rows - df[col].count()
            missing_after = len(df) - df[col].count()
            if missing_before > 0:
                print(f"  {col}: {missing_before} -> {missing_after}")

clean_hospital_data()