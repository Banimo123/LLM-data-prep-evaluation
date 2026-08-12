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

    # Conservation de row_id intact
    row_id = df['row_id'].copy()

    # Suppression des doublons (en conservant la première occurrence)
    initial_rows = len(df)
    df = df.drop_duplicates()
    duplicates_removed = initial_rows - len(df)
    print(f"Doublons supprimés: {duplicates_removed}")

    # Nettoyage colonne par colonne
    operations = []

    # ProviderNumber: colonne numérique, pas de valeurs manquantes, pas d'aberrations évidentes
    if 'ProviderNumber' in df.columns:
        df['ProviderNumber'] = pd.to_numeric(df['ProviderNumber'], errors='coerce')
        # Vérification des bornes (min/max du profil)
        min_val, max_val = 10001.0, 20018.0
        outliers = df[(df['ProviderNumber'] < min_val) | (df['ProviderNumber'] > max_val)]
        if not outliers.empty:
            median_val = df['ProviderNumber'].median()
            df.loc[(df['ProviderNumber'] < min_val) | (df['ProviderNumber'] > max_val), 'ProviderNumber'] = median_val
            operations.append(f"ProviderNumber: {len(outliers)} valeurs aberrantes corrigées par la médiane")

    # HospitalName: harmonisation des catégories (fautes de frappe, casse)
    if 'HospitalName' in df.columns:
        valid_hospitals = [
            "huntsville hospital", "riverview regional medical center", "stringfellow memorial hospital",
            "helen keller memorial hospital", "callahan eye foundation hospital",
            "southwest alabama medical center", "elba general hospital", "cullman regional medical center",
            "wedowee hospital", "g h lanier memorial hospital"
        ]
        df['HospitalName'] = df['HospitalName'].apply(lambda v: harmonize_category(v, valid_hospitals))

    # Address1: pas de valeurs manquantes, pas de traitement spécifique
    # Address2 et Address3: toutes les valeurs sont "empty", pas de traitement

    # City: 8.3% de valeurs manquantes -> imputation par mode conditionnel
    if 'City' in df.columns:
        missing_before = df['City'].isna().sum()
        group_col = find_best_grouping_column(df, 'City', is_numeric=False)
        if group_col:
            df['City'] = df.groupby(group_col)['City'].transform(
                lambda s: s.fillna(s.mode().iloc[0] if not s.mode().empty else "Unknown")
            )
        df['City'] = df['City'].fillna(df['City'].mode().iloc[0])
        missing_after = df['City'].isna().sum()
        operations.append(f"City: {missing_before - missing_after} valeurs manquantes imputées")

    # State: 8.5% de valeurs manquantes, nombreuses variantes de "al" -> harmonisation
    if 'State' in df.columns:
        valid_states = ["al", "ak", "la"]
        df['State'] = df['State'].apply(lambda v: harmonize_category(v, valid_states))
        missing_before = df['State'].isna().sum()
        group_col = find_best_grouping_column(df, 'State', is_numeric=False)
        if group_col:
            df['State'] = df.groupby(group_col)['State'].transform(
                lambda s: s.fillna(s.mode().iloc[0] if not s.mode().empty else "Unknown")
            )
        df['State'] = df['State'].fillna(df['State'].mode().iloc[0])
        missing_after = df['State'].isna().sum()
        operations.append(f"State: {missing_before - missing_after} valeurs manquantes imputées")

    # ZipCode: pas de valeurs manquantes, format texte cohérent
    if 'ZipCode' in df.columns:
        df['ZipCode'] = df['ZipCode'].astype(str).str.strip()
        df['ZipCode'] = df['ZipCode'].apply(lambda x: x if len(x) == 5 and x.isdigit() else np.nan)
        missing_before = df['ZipCode'].isna().sum()
        if missing_before > 0:
            df['ZipCode'] = df['ZipCode'].fillna(df['ZipCode'].mode().iloc[0])
            operations.append(f"ZipCode: {missing_before} valeurs corrigées par le mode")

    # CountyName: pas de valeurs manquantes, pas de traitement spécifique

    # PhoneNumber: colonne numérique, pas de valeurs manquantes, bornes cohérentes
    if 'PhoneNumber' in df.columns:
        df['PhoneNumber'] = df['PhoneNumber'].apply(extract_numeric)
        min_val, max_val = 2052743000.0, 9075436300.0
        outliers = df[(df['PhoneNumber'] < min_val) | (df['PhoneNumber'] > max_val)]
        if not outliers.empty:
            median_val = df['PhoneNumber'].median()
            df.loc[(df['PhoneNumber'] < min_val) | (df['PhoneNumber'] > max_val), 'PhoneNumber'] = median_val
            operations.append(f"PhoneNumber: {len(outliers)} valeurs aberrantes corrigées par la médiane")

    # HospitalType: toutes les valeurs sont "acute care hospitals", pas de traitement
    if 'HospitalType' in df.columns:
        df['HospitalType'] = "acute care hospitals"

    # HospitalOwner: 7.9% de valeurs manquantes, harmonisation des catégories
    if 'HospitalOwner' in df.columns:
        valid_owners = [
            "voluntary non-profit - private", "proprietary", "government - hospital district or authority",
            "voluntary non-profit - other", "voluntary non-profit - church", "government - federal",
            "government - state", "government - local", "unknown"
        ]
        df['HospitalOwner'] = df['HospitalOwner'].apply(lambda v: harmonize_category(v, valid_owners))
        missing_before = df['HospitalOwner'].isna().sum()
        group_col = find_best_grouping_column(df, 'HospitalOwner', is_numeric=False)
        if group_col:
            df['HospitalOwner'] = df.groupby(group_col)['HospitalOwner'].transform(
                lambda s: s.fillna(s.mode().iloc[0] if not s.mode().empty else "Unknown")
            )
        df['HospitalOwner'] = df['HospitalOwner'].fillna(df['HospitalOwner'].mode().iloc[0])
        missing_after = df['HospitalOwner'].isna().sum()
        operations.append(f"HospitalOwner: {missing_before - missing_after} valeurs manquantes imputées")

    # EmergencyService: 8.4% de valeurs manquantes, harmonisation
    if 'EmergencyService' in df.columns:
        valid_emergency = ["yes", "no", "unknown"]
        df['EmergencyService'] = df['EmergencyService'].apply(lambda v: harmonize_category(v, valid_emergency))
        missing_before = df['EmergencyService'].isna().sum()
        group_col = find_best_grouping_column(df, 'EmergencyService', is_numeric=False)
        if group_col:
            df['EmergencyService'] = df.groupby(group_col)['EmergencyService'].transform(
                lambda s: s.fillna(s.mode().iloc[0] if not s.mode().empty else "Unknown")
            )
        df['EmergencyService'] = df['EmergencyService'].fillna(df['EmergencyService'].mode().iloc[0])
        missing_after = df['EmergencyService'].isna().sum()
        operations.append(f"EmergencyService: {missing_before - missing_after} valeurs manquantes imputées")

    # Condition: pas de valeurs manquantes, pas de traitement spécifique
    # MeasureCode: pas de valeurs manquantes, pas de traitement spécifique
    # MeasureName: pas de valeurs manquantes, pas de traitement spécifique

    # Score: valeurs comme "100%", "empty" -> conservation du format texte
    if 'Score' in df.columns:
        df['Score'] = df['Score'].astype(str).str.strip()
        # Remplacement des valeurs "empty" par le mode (100%)
        df['Score'] = df['Score'].replace("empty", np.nan)
        missing_before = df['Score'].isna().sum()
        if missing_before > 0:
            mode_score = df['Score'].mode().iloc[0]
            df['Score'] = df['Score'].fillna(mode_score)
            operations.append(f"Score: {missing_before} valeurs 'empty' remplacées par le mode ({mode_score})")

    # Sample: valeurs comme "0 patients", "empty" -> conservation du format texte
    if 'Sample' in df.columns:
        df['Sample'] = df['Sample'].astype(str).str.strip()
        # Remplacement des valeurs "empty" par le mode ("0 patients")
        df['Sample'] = df['Sample'].replace("empty", np.nan)
        missing_before = df['Sample'].isna().sum()
        if missing_before > 0:
            mode_sample = df['Sample'].mode().iloc[0]
            df['Sample'] = df['Sample'].fillna(mode_sample)
            operations.append(f"Sample: {missing_before} valeurs 'empty' remplacées par le mode ({mode_sample})")

    # Stateavg: pas de valeurs manquantes, format cohérent

    # Restauration de row_id
    df['row_id'] = row_id.loc[df.index]

    # Sauvegarde du dataset nettoyé
    df.to_csv(OUTPUT_PATH, index=False)

    # Affichage du résumé des opérations
    print("\nRésumé des opérations de nettoyage:")
    for op in operations:
        print(f"- {op}")
    print(f"\nDataset nettoyé sauvegardé dans {OUTPUT_PATH}")
    print(f"Nombre final de lignes: {len(df)}")
    print(f"Nombre final de colonnes: {len(df.columns)}")

if __name__ == "__main__":
    clean_dataset()