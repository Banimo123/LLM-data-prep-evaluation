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

def clean_hospital_data():
    df = pd.read_csv(INPUT_PATH)

    # Conservation de row_id intact
    original_row_ids = df['row_id'].copy()

    # Suppression des doublons (conservation du premier)
    initial_rows = len(df)
    df = df.drop_duplicates(subset=[col for col in df.columns if col != 'row_id'], keep='first')
    duplicates_removed = initial_rows - len(df)

    # Nettoyage colonne par colonne
    # ProviderNumber: numérique, pas de manquants, pas d'aberrations apparentes
    if 'ProviderNumber' in df.columns:
        df['ProviderNumber'] = pd.to_numeric(df['ProviderNumber'], errors='coerce')
        # Vérification des bornes (min/max du profil)
        min_val, max_val = 10001.0, 20018.0
        df['ProviderNumber'] = df['ProviderNumber'].apply(
            lambda x: x if pd.isna(x) or (min_val <= x <= max_val) else df['ProviderNumber'].median()
        )

    # HospitalName: catégoriel, pas de manquants, harmonisation des noms
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

    # Address1: catégoriel, pas de manquants, harmonisation des adresses
    if 'Address1' in df.columns:
        valid_addresses = [
            "301 east 18th st", "101 sivley rd", "600 south third street", "1300 south montgomery avenue",
            "1720 university blvd", "702 n main st", "101 hospital circle", "201 pine street northwest",
            "8000 alabama highway 69", "50 medical park east drive"
        ]
        df['Address1'] = df['Address1'].apply(
            lambda v: harmonize_category(v, valid_addresses) if pd.notna(v) else v
        )

    # Address2 et Address3: toujours "empty", pas de traitement nécessaire
    for col in ['Address2', 'Address3']:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: 'empty' if str(x).strip().lower() == 'empty' else x)

    # City: 31% de manquants, imputation conditionnelle
    if 'City' in df.columns:
        valid_cities = [
            "birmingham", "montgomery", "gadsden", "dothan", "unknown", "guntersville",
            "anniston", "ozark", "boaz"
        ]
        # Imputation conditionnelle
        group_col = find_best_grouping_column(df, 'City', False)
        if group_col:
            df['City'] = df.groupby(group_col)['City'].transform(
                lambda s: s.fillna(s.mode().iloc[0] if not s.mode().empty else "unknown")
            )
        df['City'] = df['City'].fillna("unknown")
        df['City'] = df['City'].apply(
            lambda v: harmonize_category(v, valid_cities) if pd.notna(v) else v
        )

    # State: 31.4% de manquants, valeurs aberrantes ("l", "aal", "NNA")
    if 'State' in df.columns:
        valid_states = ["al", "la", "ak"]
        df['State'] = df['State'].apply(
            lambda v: harmonize_category(v, valid_states) if pd.notna(v) else v
        )
        # Imputation conditionnelle
        group_col = find_best_grouping_column(df, 'State', False)
        if group_col:
            df['State'] = df.groupby(group_col)['State'].transform(
                lambda s: s.fillna(s.mode().iloc[0] if not s.mode().empty else "al")
            )
        df['State'] = df['State'].fillna("al")

    # ZipCode: pas de manquants, format texte (5 chiffres)
    if 'ZipCode' in df.columns:
        df['ZipCode'] = df['ZipCode'].astype(str)
        df['ZipCode'] = df['ZipCode'].apply(
            lambda x: x[:5] if len(x) >= 5 and x[:5].isdigit() else "00000"
        )

    # CountyName: pas de manquants, harmonisation
    if 'CountyName' in df.columns:
        valid_counties = [
            "jefferson", "etowah", "marshall", "marion", "covington", "montgomery",
            "coffee", "houston", "calhoun", "madison"
        ]
        df['CountyName'] = df['CountyName'].apply(
            lambda v: harmonize_category(v, valid_counties) if pd.notna(v) else v
        )

    # PhoneNumber: numérique, pas de manquants, extraction des chiffres
    if 'PhoneNumber' in df.columns:
        df['PhoneNumber'] = df['PhoneNumber'].apply(extract_numeric)
        # Vérification des bornes (min/max du profil)
        min_val, max_val = 2052743000.0, 9075436300.0
        df['PhoneNumber'] = df['PhoneNumber'].apply(
            lambda x: x if pd.isna(x) or (min_val <= x <= max_val) else df['PhoneNumber'].median()
        )

    # HospitalType: toujours "acute care hospitals", pas de traitement
    if 'HospitalType' in df.columns:
        df['HospitalType'] = df['HospitalType'].apply(
            lambda x: "acute care hospitals" if pd.notna(x) and "acute" in str(x).lower() else x
        )

    # HospitalOwner: 33.9% de manquants, harmonisation
    if 'HospitalOwner' in df.columns:
        valid_owners = [
            "voluntary non-profit - private", "proprietary", "government - hospital district or authority",
            "voluntary non-profit - other", "voluntary non-profit - church", "government - federal",
            "government - state", "government - local", "unknown"
        ]
        df['HospitalOwner'] = df['HospitalOwner'].apply(
            lambda v: harmonize_category(v, valid_owners) if pd.notna(v) else v
        )
        # Imputation conditionnelle
        group_col = find_best_grouping_column(df, 'HospitalOwner', False)
        if group_col:
            df['HospitalOwner'] = df.groupby(group_col)['HospitalOwner'].transform(
                lambda s: s.fillna(s.mode().iloc[0] if not s.mode().empty else "unknown")
            )
        df['HospitalOwner'] = df['HospitalOwner'].fillna("unknown")

    # EmergencyService: 33.1% de manquants, harmonisation
    if 'EmergencyService' in df.columns:
        valid_emergency = ["yes", "no", "unknown"]
        df['EmergencyService'] = df['EmergencyService'].apply(
            lambda v: harmonize_category(v, valid_emergency) if pd.notna(v) else v
        )
        # Imputation conditionnelle
        group_col = find_best_grouping_column(df, 'EmergencyService', False)
        if group_col:
            df['EmergencyService'] = df.groupby(group_col)['EmergencyService'].transform(
                lambda s: s.fillna(s.mode().iloc[0] if not s.mode().empty else "unknown")
            )
        df['EmergencyService'] = df['EmergencyService'].fillna("unknown")

    # Condition: pas de manquants, harmonisation
    if 'Condition' in df.columns:
        valid_conditions = [
            "surgical infection prevention", "heart attack", "pneumonia", "heart failure",
            "children s asthma care"
        ]
        df['Condition'] = df['Condition'].apply(
            lambda v: harmonize_category(v, valid_conditions) if pd.notna(v) else v
        )

    # MeasureCode: pas de manquants, format standardisé
    if 'MeasureCode' in df.columns:
        df['MeasureCode'] = df['MeasureCode'].str.lower().str.strip()

    # MeasureName: pas de manquants, pas de traitement spécifique

    # Score: format pourcentage ou "empty", pas de conversion numérique
    if 'Score' in df.columns:
        df['Score'] = df['Score'].apply(
            lambda x: str(x).strip() if pd.notna(x) else x
        )
        # Harmonisation des pourcentages
        df['Score'] = df['Score'].apply(
            lambda x: x if x in ["100%", "97%", "98%", "99%", "96%", "95%", "94%", "90%", "93%", "empty"] else x
        )

    # Sample: format "X patients" ou "empty", pas de conversion numérique
    if 'Sample' in df.columns:
        df['Sample'] = df['Sample'].apply(
            lambda x: str(x).strip() if pd.notna(x) else x
        )
        # Harmonisation des valeurs fréquentes
        valid_samples = [
            "0 patients", "empty", "1 patients", "4 patients", "2 patients", "3 patients",
            "10 patients", "5 patients", "15 patients", "6 patients"
        ]
        df['Sample'] = df['Sample'].apply(
            lambda v: harmonize_category(v, valid_samples) if pd.notna(v) else v
        )

    # Stateavg: format standardisé (état_mesure)
    if 'Stateavg' in df.columns:
        df['Stateavg'] = df['Stateavg'].str.lower().str.strip()

    # Restauration de row_id original
    df['row_id'] = original_row_ids.loc[df.index]

    # Sauvegarde du dataset nettoyé
    df.to_csv(OUTPUT_PATH, index=False)

    # Log des opérations
    print(f"Nettoyage terminé. {duplicates_removed} doublons supprimés.")
    print(f"Dataset final: {len(df)} lignes, {len(df.columns)} colonnes.")

    # Calcul des valeurs manquantes restantes
    missing_report = df.isna().sum()
    missing_cols = missing_report[missing_report > 0]
    if not missing_cols.empty:
        print("\nValeurs manquantes restantes par colonne:")
        for col, count in missing_cols.items():
            print(f"- {col}: {count} ({count/len(df):.1%})")
    else:
        print("\nAucune valeur manquante restante.")

if __name__ == "__main__":
    clean_hospital_data()