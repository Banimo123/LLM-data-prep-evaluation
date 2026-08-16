import pandas as pd
import numpy as np
import re
from datetime import datetime
import difflib

INPUT_PATH = "datasets\\hospital\\noisy_low.csv"
OUTPUT_PATH = "results\\cleaned_datasets\\hospital\\noisy_low__profile.csv"

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
    df = df.drop_duplicates()
    final_rows = len(df)
    duplicates_removed = initial_rows - final_rows

    # Nettoyage colonne par colonne
    # ProviderNumber: pas de valeurs manquantes, pas de nettoyage nécessaire
    # HospitalName: pas de valeurs manquantes, harmonisation des fautes de frappe
    valid_hospital_names = [
        "stringfellow memorial hospital", "huntsville hospital", "marshall medical center south",
        "helen keller memorial hospital", "callahan eye foundation hospital",
        "gadsden regional medical center", "chilton medical center", "st vincents east",
        "eliza coffee memorial hospital", "shelby baptist medical center"
    ]
    df["HospitalName"] = df["HospitalName"].apply(lambda v: harmonize_category(v, valid_hospital_names))

    # Address1: pas de valeurs manquantes, pas de nettoyage spécifique
    # Address2 et Address3: toutes les valeurs sont "empty", pas de nettoyage nécessaire

    # City: harmonisation des fautes de frappe
    valid_cities = [
        "birmingham", "gadsden", "montgomery", "dothan", "huntsville", "anniston",
        "centre", "thomasville", "clanton", "opelika"
    ]
    df["City"] = df["City"].apply(lambda v: harmonize_category(v, valid_cities))

    # State: harmonisation des codes états
    valid_states = ["al", "ak", "xl", "ax"]
    df["State"] = df["State"].apply(lambda v: harmonize_category(v, valid_states))

    # ZipCode: pas de nettoyage spécifique (codes postaux valides)

    # CountyName: harmonisation des fautes de frappe
    valid_counties = [
        "jefferson", "etowah", "marion", "covington", "marshall", "coffee",
        "montgomery", "houston", "calhoun", "madison"
    ]
    df["CountyName"] = df["CountyName"].apply(lambda v: harmonize_category(v, valid_counties))

    # PhoneNumber: pas de nettoyage spécifique (numéros valides)

    # HospitalType: harmonisation des fautes de frappe
    valid_hospital_types = ["acute care hospitals"]
    df["HospitalType"] = df["HospitalType"].apply(lambda v: harmonize_category(v, valid_hospital_types))

    # HospitalOwner: harmonisation des fautes de frappe
    valid_owners = [
        "voluntary non-profit - private", "proprietary", "government - hospital district or authority",
        "voluntary non-profit - other", "voluntary non-profit - church", "government - federal",
        "government - state", "government - local"
    ]
    df["HospitalOwner"] = df["HospitalOwner"].apply(lambda v: harmonize_category(v, valid_owners))

    # EmergencyService: harmonisation des réponses
    valid_emergency = ["yes", "no"]
    df["EmergencyService"] = df["EmergencyService"].apply(lambda v: harmonize_category(v, valid_emergency))

    # Condition: harmonisation des conditions médicales
    valid_conditions = [
        "surgical infection prevention", "heart attack", "pneumonia", "heart failure",
        "children s asthma care"
    ]
    df["Condition"] = df["Condition"].apply(lambda v: harmonize_category(v, valid_conditions))

    # MeasureCode: pas de nettoyage spécifique (codes valides)

    # MeasureName: pas de nettoyage spécifique (noms valides)

    # Score: extraction des pourcentages et imputation des valeurs manquantes
    df["Score"] = df["Score"].apply(lambda x: x if x == "empty" else extract_numeric(x))
    # Imputation des valeurs manquantes (après extraction)
    group_col = find_best_grouping_column(df, "Score", True)
    if group_col:
        df["Score"] = df.groupby(group_col)["Score"].transform(
            lambda s: s.fillna(s.median())
        )
    df["Score"] = df["Score"].fillna(df["Score"].median())
    # Reformatage en pourcentage
    df["Score"] = df["Score"].apply(lambda x: f"{int(x)}%" if not pd.isna(x) else "empty")

    # Sample: extraction des nombres de patients et imputation des valeurs manquantes
    def extract_patients(value):
        if pd.isna(value) or value == "empty":
            return np.nan
        match = re.search(r"(\d+)\s*patients?", str(value))
        return int(match.group(1)) if match else np.nan

    df["Sample"] = df["Sample"].apply(extract_patients)
    # Imputation des valeurs manquantes
    group_col = find_best_grouping_column(df, "Sample", True)
    if group_col:
        df["Sample"] = df.groupby(group_col)["Sample"].transform(
            lambda s: s.fillna(s.mode().iloc[0] if not s.mode().empty else 0)
        )
    df["Sample"] = df["Sample"].fillna(df["Sample"].mode().iloc[0])
    # Reformatage avec "patients"
    df["Sample"] = df["Sample"].apply(lambda x: f"{int(x)} patients" if not pd.isna(x) else "empty")

    # Stateavg: pas de nettoyage spécifique (codes valides)

    # Restauration de row_id
    df['row_id'] = row_ids.loc[df.index]

    # Sauvegarde du dataset nettoyé
    df.to_csv(OUTPUT_PATH, index=False)

    # Log des opérations
    print(f"Nettoyage terminé. {duplicates_removed} doublons supprimés.")
    print(f"Dataset final: {len(df)} lignes, {len(df.columns)} colonnes.")

if __name__ == "__main__":
    clean_hospital_data()