import pandas as pd
import numpy as np
import re
from datetime import datetime
import difflib

INPUT_PATH = r'datasets\hospital\noisy_low.csv'
OUTPUT_PATH = r'results\cleaned_datasets\hospital\noisy_low__profile.csv'

# Fonction pour extraire les valeurs numériques
def extract_numeric(value):
    if pd.isna(value):
        return np.nan
    s = str(value).replace(",", ".")
    s = s.replace("O", "0").replace("o", "0")
    match = re.search(r"-?\d+\.?\d*", s)
    return float(match.group()) if match else np.nan

# Fonction pour harmoniser les catégories
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

# Fonction pour trouver la meilleure colonne de regroupement pour l'imputation
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

# Chargement du dataset
df = pd.read_csv(INPUT_PATH)

# Initialisation du log
log = []

# Suppression des doublons (en conservant le premier)
initial_rows = len(df)
df = df.drop_duplicates(subset=[col for col in df.columns if col != "row_id"], keep='first')
final_rows = len(df)
if initial_rows != final_rows:
    log.append(f"Doublons supprimés: {initial_rows - final_rows}")

# Nettoyage de la colonne State (8.5% manquants, valeurs incohérentes)
if 'State' in df.columns:
    valid_states = ["al", "ak", "la"]  # D'après les valeurs fréquentes
    df['State'] = df['State'].apply(lambda v: harmonize_category(v, valid_states))
    # Imputation des valeurs manquantes
    group_col = find_best_grouping_column(df, 'State', False)
    if group_col:
        df['State'] = df.groupby(group_col)['State'].transform(
            lambda s: s.fillna(s.mode().iloc[0] if not s.mode().empty else "al")
        )
    df['State'] = df['State'].fillna(df['State'].mode().iloc[0])
    log.append("Colonne State: harmonisation des catégories et imputation des manquants")

# Nettoyage de la colonne City (8.3% manquants)
if 'City' in df.columns:
    # Imputation des valeurs manquantes
    group_col = find_best_grouping_column(df, 'City', False)
    if group_col:
        df['City'] = df.groupby(group_col)['City'].transform(
            lambda s: s.fillna(s.mode().iloc[0] if not s.mode().empty else "birmingham")
        )
    df['City'] = df['City'].fillna(df['City'].mode().iloc[0])
    log.append("Colonne City: imputation des manquants")

# Nettoyage de la colonne HospitalOwner (7.9% manquants)
if 'HospitalOwner' in df.columns:
    valid_owners = [
        "voluntary non-profit - private", "proprietary", "government - hospital district or authority",
        "voluntary non-profit - other", "voluntary non-profit - church", "government - federal",
        "government - state", "government - local", "unknown"
    ]
    df['HospitalOwner'] = df['HospitalOwner'].apply(lambda v: harmonize_category(v, valid_owners))
    # Imputation des valeurs manquantes
    group_col = find_best_grouping_column(df, 'HospitalOwner', False)
    if group_col:
        df['HospitalOwner'] = df.groupby(group_col)['HospitalOwner'].transform(
            lambda s: s.fillna(s.mode().iloc[0] if not s.mode().empty else "voluntary non-profit - private")
        )
    df['HospitalOwner'] = df['HospitalOwner'].fillna(df['HospitalOwner'].mode().iloc[0])
    log.append("Colonne HospitalOwner: harmonisation des catégories et imputation des manquants")

# Nettoyage de la colonne EmergencyService (8.4% manquants)
if 'EmergencyService' in df.columns:
    valid_emergency = ["yes", "no", "unknown"]
    df['EmergencyService'] = df['EmergencyService'].apply(lambda v: harmonize_category(v, valid_emergency))
    # Imputation des valeurs manquantes
    group_col = find_best_grouping_column(df, 'EmergencyService', False)
    if group_col:
        df['EmergencyService'] = df.groupby(group_col)['EmergencyService'].transform(
            lambda s: s.fillna(s.mode().iloc[0] if not s.mode().empty else "yes")
        )
    df['EmergencyService'] = df['EmergencyService'].fillna(df['EmergencyService'].mode().iloc[0])
    log.append("Colonne EmergencyService: harmonisation des catégories et imputation des manquants")

# Nettoyage de la colonne PhoneNumber (0% manquants, valeurs numériques)
if 'PhoneNumber' in df.columns:
    # Extraction des valeurs numériques
    df['PhoneNumber'] = df['PhoneNumber'].apply(extract_numeric)
    # Vérification des bornes (min: 2052743000, max: 9075436300)
    phone_min, phone_max = 2052743000, 9075436300
    outliers = df[(df['PhoneNumber'] < phone_min) | (df['PhoneNumber'] > phone_max)]
    if not outliers.empty:
        median_phone = df[(df['PhoneNumber'] >= phone_min) & (df['PhoneNumber'] <= phone_max)]['PhoneNumber'].median()
        df.loc[(df['PhoneNumber'] < phone_min) | (df['PhoneNumber'] > phone_max), 'PhoneNumber'] = median_phone
        log.append(f"Colonne PhoneNumber: {len(outliers)} valeurs aberrantes corrigées")

# Nettoyage de la colonne Score (0% manquants, valeurs en pourcentage)
if 'Score' in df.columns:
    # Extraction des valeurs numériques pour les scores en pourcentage
    def extract_score(value):
        if pd.isna(value) or value == "empty":
            return value
        s = str(value).strip()
        if '%' in s:
            num = extract_numeric(s)
            return f"{int(num)}%" if not pd.isna(num) else value
        return value

    df['Score'] = df['Score'].apply(extract_score)
    # Harmonisation des valeurs "empty" vers "0%"
    df['Score'] = df['Score'].replace("empty", "0%")
    log.append("Colonne Score: harmonisation des valeurs en pourcentage")

# Nettoyage de la colonne Sample (0% manquants, valeurs avec "patients")
if 'Sample' in df.columns:
    # Extraction des valeurs numériques pour les échantillons
    def extract_sample(value):
        if pd.isna(value) or value == "empty":
            return value
        s = str(value).strip()
        if "patients" in s:
            num = extract_numeric(s)
            return f"{int(num)} patients" if not pd.isna(num) else value
        return value

    df['Sample'] = df['Sample'].apply(extract_sample)
    # Harmonisation des valeurs "empty" vers "0 patients"
    df['Sample'] = df['Sample'].replace("empty", "0 patients")
    log.append("Colonne Sample: harmonisation des valeurs avec 'patients'")

# Nettoyage de la colonne ZipCode (0% manquants, format texte)
if 'ZipCode' in df.columns:
    # Suppression des espaces et harmonisation
    df['ZipCode'] = df['ZipCode'].astype(str).str.strip()
    log.append("Colonne ZipCode: nettoyage des espaces")

# Nettoyage des colonnes Address2 et Address3 (toutes "empty")
if 'Address2' in df.columns:
    df['Address2'] = df['Address2'].replace("empty", "")
if 'Address3' in df.columns:
    df['Address3'] = df['Address3'].replace("empty", "")
log.append("Colonnes Address2 et Address3: remplacement de 'empty' par chaîne vide")

# Sauvegarde du dataset nettoyé
df.to_csv(OUTPUT_PATH, index=False)

# Affichage du log
print("Résumé des opérations de nettoyage:")
for entry in log:
    print(f"- {entry}")
print(f"- Lignes initiales: {initial_rows}, lignes finales: {final_rows}")