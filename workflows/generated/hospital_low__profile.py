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
    date_formats = [
        "%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%B %d, %Y",
        "%d-%b-%Y", "%Y/%m/%d"
    ]
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

df = pd.read_csv(INPUT_PATH)

# Conservation de row_id comme identifiant technique
if 'row_id' not in df.columns:
    df['row_id'] = df.index + 1

# Suppression des doublons en conservant le premier (et row_id intact)
initial_rows = len(df)
df = df.drop_duplicates(subset=[c for c in df.columns if c != 'row_id'], keep='first')
duplicates_removed = initial_rows - len(df)

# Nettoyage colonne par colonne
operations_log = []

# ProviderNumber: colonne numérique sans valeurs manquantes -> vérification des bornes
if 'ProviderNumber' in df.columns:
    min_val, max_val = 10001.0, 20018.0
    df['ProviderNumber'] = df['ProviderNumber'].apply(
        lambda x: x if pd.isna(x) or (min_val <= x <= max_val) else df['ProviderNumber'].median()
    )
    operations_log.append(f"ProviderNumber: {len(df[df['ProviderNumber'].isna()])} valeurs aberrantes corrigées")

# HospitalName: harmonisation des noms d'hôpitaux (fautes de frappe, casse)
if 'HospitalName' in df.columns:
    valid_hospitals = [
        "huntsville hospital", "riverview regional medical center", "stringfellow memorial hospital",
        "helen keller memorial hospital", "callahan eye foundation hospital",
        "southwest alabama medical center", "elba general hospital", "cullman regional medical center",
        "wedowee hospital", "g h lanier memorial hospital"
    ]
    df['HospitalName'] = df['HospitalName'].apply(
        lambda v: harmonize_category(v, valid_hospitals)
    )
    operations_log.append("HospitalName: harmonisation des variantes de noms")

# Address1: pas de traitement spécifique (uniques mais valides)
# Address2/Address3: colonne "empty" à conserver telle quelle (règle 20)
if 'Address2' in df.columns:
    df['Address2'] = df['Address2'].apply(lambda x: x if x == "empty" else x)
if 'Address3' in df.columns:
    df['Address3'] = df['Address3'].apply(lambda x: x if x == "empty" else x)

# City: 8.3% de valeurs manquantes -> imputation conditionnelle
if 'City' in df.columns:
    group_col = find_best_grouping_column(df, 'City', False)
    if group_col:
        df['City'] = df.groupby(group_col)['City'].transform(
            lambda s: s.fillna(s.mode().iloc[0] if not s.mode().empty else "Unknown")
        )
    df['City'] = df['City'].fillna(df['City'].mode().iloc[0])
    operations_log.append(f"City: {df['City'].isna().sum()} valeurs manquantes imputées")

# State: 8.5% de valeurs manquantes et variantes -> harmonisation + imputation
if 'State' in df.columns:
    valid_states = ["al", "ak", "la"]
    df['State'] = df['State'].apply(lambda v: harmonize_category(v, valid_states))
    group_col = find_best_grouping_column(df, 'State', False)
    if group_col:
        df['State'] = df.groupby(group_col)['State'].transform(
            lambda s: s.fillna(s.mode().iloc[0] if not s.mode().empty else "al")
        )
    df['State'] = df['State'].fillna("al")
    operations_log.append(f"State: {df['State'].isna().sum()} valeurs manquantes imputées et harmonisées")

# ZipCode: pas de valeurs manquantes -> nettoyage des formats
if 'ZipCode' in df.columns:
    df['ZipCode'] = df['ZipCode'].astype(str).str.extract(r'(\d{5})')[0]
    df['ZipCode'] = df['ZipCode'].fillna(df['ZipCode'].mode().iloc[0])

# CountyName: pas de valeurs manquantes -> harmonisation
if 'CountyName' in df.columns:
    valid_counties = [
        "jefferson", "etowah", "marshall", "marion", "covington", "montgomery",
        "coffee", "houston", "calhoun", "madison"
    ]
    df['CountyName'] = df['CountyName'].apply(
        lambda v: harmonize_category(v, valid_counties)
    )

# PhoneNumber: colonne numérique sans valeurs manquantes -> extraction numérique
if 'PhoneNumber' in df.columns:
    df['PhoneNumber'] = df['PhoneNumber'].apply(extract_numeric)
    min_val, max_val = 2052743000.0, 9075436300.0
    df['PhoneNumber'] = df['PhoneNumber'].apply(
        lambda x: x if pd.isna(x) or (min_val <= x <= max_val) else df['PhoneNumber'].median()
    )

# HospitalOwner: 7.9% de valeurs manquantes -> imputation conditionnelle
if 'HospitalOwner' in df.columns:
    valid_owners = [
        "voluntary non-profit - private", "proprietary", "government - hospital district or authority",
        "voluntary non-profit - other", "voluntary non-profit - church", "government - federal",
        "government - state", "government - local", "unknown"
    ]
    df['HospitalOwner'] = df['HospitalOwner'].apply(
        lambda v: harmonize_category(v, valid_owners)
    )
    group_col = find_best_grouping_column(df, 'HospitalOwner', False)
    if group_col:
        df['HospitalOwner'] = df.groupby(group_col)['HospitalOwner'].transform(
            lambda s: s.fillna(s.mode().iloc[0] if not s.mode().empty else "voluntary non-profit - private")
        )
    df['HospitalOwner'] = df['HospitalOwner'].fillna("voluntary non-profit - private")
    operations_log.append(f"HospitalOwner: {df['HospitalOwner'].isna().sum()} valeurs manquantes imputées")

# EmergencyService: 8.4% de valeurs manquantes -> imputation conditionnelle
if 'EmergencyService' in df.columns:
    valid_emergency = ["yes", "no", "unknown"]
    df['EmergencyService'] = df['EmergencyService'].apply(
        lambda v: harmonize_category(v, valid_emergency)
    )
    group_col = find_best_grouping_column(df, 'EmergencyService', False)
    if group_col:
        df['EmergencyService'] = df.groupby(group_col)['EmergencyService'].transform(
            lambda s: s.fillna(s.mode().iloc[0] if not s.mode().empty else "yes")
        )
    df['EmergencyService'] = df['EmergencyService'].fillna("yes")
    operations_log.append(f"EmergencyService: {df['EmergencyService'].isna().sum()} valeurs manquantes imputées")

# Condition: pas de valeurs manquantes -> harmonisation
if 'Condition' in df.columns:
    valid_conditions = [
        "surgical infection prevention", "heart attack", "pneumonia",
        "heart failure", "children s asthma care"
    ]
    df['Condition'] = df['Condition'].apply(
        lambda v: harmonize_category(v, valid_conditions)
    )

# MeasureCode: pas de valeurs manquantes -> harmonisation
if 'MeasureCode' in df.columns:
    valid_codes = [
        "ami-2", "hf-3", "hf-4", "pn-2", "hf-2", "pn-3b", "hf-1", "pn-4",
        "ami-3", "scip-card-2", "scip-inf-1", "scip-inf-2", "scip-inf-3", "scip-inf-4"
    ]
    df['MeasureCode'] = df['MeasureCode'].apply(
        lambda v: harmonize_category(v, valid_codes)
    )

# Score: colonne avec "empty" et pourcentages -> conservation du format
if 'Score' in df.columns:
    df['Score'] = df['Score'].apply(
        lambda x: x if x in ["empty", "100%", "97%", "98%", "99%", "96%", "95%", "94%", "90%", "93%"] else
        (x if str(x).endswith('%') else f"{x}%" if str(x).replace('%', '').isdigit() else "empty")
    )

# Sample: colonne avec "empty" et "X patients" -> conservation du format
if 'Sample' in df.columns:
    df['Sample'] = df['Sample'].apply(
        lambda x: x if x in ["empty", "0 patients", "1 patients", "2 patients", "3 patients",
                            "4 patients", "5 patients", "6 patients", "10 patients", "15 patients"] else
        (x if "patients" in str(x) else "empty")
    )

# Stateavg: pas de valeurs manquantes -> harmonisation
if 'Stateavg' in df.columns:
    valid_stateavgs = [
        "al_ami-5", "al_ami-4", "al_ami-3", "al_ami-2", "al_pn-2", "al_hf-4",
        "al_hf-3", "al_hf-2", "al_pn-3b", "al_hf-1", "al_scip-card-2", "al_scip-inf-1"
    ]
    df['Stateavg'] = df['Stateavg'].apply(
        lambda v: harmonize_category(v, valid_stateavgs)
    )

# Sauvegarde du dataset nettoyé
df.to_csv(OUTPUT_PATH, index=False)

# Log des opérations
print(f"Nettoyage terminé. {duplicates_removed} doublons supprimés.")
for op in operations_log:
    print(op)
print(f"Dataset final: {len(df)} lignes, {len(df.columns)} colonnes.")