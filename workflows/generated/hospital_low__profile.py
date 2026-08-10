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
        except ValueError:
            continue
    try:
        return datetime.fromtimestamp(int(value)).strftime("%Y-%m-%d")
    except (ValueError, TypeError):
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

def clean_data():
    df = pd.read_csv(INPUT_PATH)

    # Conservation de row_id comme identifiant technique
    if 'row_id' not in df.columns:
        raise ValueError("La colonne row_id est absente du dataset")

    # Suppression des doublons (conservation de la première occurrence)
    initial_rows = len(df)
    df = df.drop_duplicates(subset=[col for col in df.columns if col != 'row_id'], keep='first')
    duplicates_removed = initial_rows - len(df)

    # Nettoyage colonne par colonne
    operations = []

    # ProviderNumber - numérique, pas de valeurs manquantes, pas d'aberrations détectées
    if 'ProviderNumber' in df.columns:
        df['ProviderNumber'] = df['ProviderNumber'].apply(extract_numeric)
        operations.append("ProviderNumber: valeurs numériques extraites")

    # HospitalName - catégoriel, pas de valeurs manquantes, harmonisation des noms
    if 'HospitalName' in df.columns:
        valid_hospitals = [
            "huntsville hospital", "riverview regional medical center", "stringfellow memorial hospital",
            "helen keller memorial hospital", "callahan eye foundation hospital",
            "southwest alabama medical center", "elba general hospital", "cullman regional medical center",
            "wedowee hospital", "g h lanier memorial hospital"
        ]
        df['HospitalName'] = df['HospitalName'].apply(lambda v: harmonize_category(v, valid_hospitals))
        operations.append("HospitalName: harmonisation des noms d'hôpitaux")

    # Address1 - catégoriel, pas de valeurs manquantes, pas de traitement spécifique
    # Address2 et Address3 - toutes les valeurs sont "empty", pas de traitement

    # City - 8.3% de valeurs manquantes, imputation par mode conditionnel
    if 'City' in df.columns:
        if df['City'].isna().mean() > 0.05:
            group_col = find_best_grouping_column(df, 'City', False)
            if group_col:
                df['City'] = df.groupby(group_col)['City'].transform(
                    lambda s: s.fillna(s.mode().iloc[0] if not s.mode().empty else "Unknown")
                )
            df['City'] = df['City'].fillna(df['City'].mode().iloc[0])
            operations.append(f"City: {df['City'].isna().sum()} valeurs manquantes imputées par mode")
        valid_cities = [
            "birmingham", "montgomery", "gadsden", "dothan", "anniston", "huntsville", "valley",
            "wedowee", "alabaster", "thomasville"
        ]
        df['City'] = df['City'].apply(lambda v: harmonize_category(v, valid_cities))

    # State - 8.5% de valeurs manquantes, nombreuses variantes de "AL"
    if 'State' in df.columns:
        if df['State'].isna().mean() > 0.05:
            group_col = find_best_grouping_column(df, 'State', False)
            if group_col:
                df['State'] = df.groupby(group_col)['State'].transform(
                    lambda s: s.fillna(s.mode().iloc[0] if not s.mode().empty else "Unknown")
                )
            df['State'] = df['State'].fillna(df['State'].mode().iloc[0])
            operations.append(f"State: {df['State'].isna().sum()} valeurs manquantes imputées par mode")

        valid_states = ["al", "ak", "la"]
        df['State'] = df['State'].apply(lambda v: harmonize_category(v, valid_states))
        operations.append("State: harmonisation des codes d'état")

    # ZipCode - catégoriel, pas de valeurs manquantes, pas de traitement spécifique

    # CountyName - catégoriel, pas de valeurs manquantes, harmonisation
    if 'CountyName' in df.columns:
        valid_counties = [
            "jefferson", "etowah", "marshall", "marion", "covington", "montgomery", "coffee",
            "houston", "calhoun", "madison"
        ]
        df['CountyName'] = df['CountyName'].apply(lambda v: harmonize_category(v, valid_counties))
        operations.append("CountyName: harmonisation des noms de comtés")

    # PhoneNumber - numérique, pas de valeurs manquantes, extraction des numéros
    if 'PhoneNumber' in df.columns:
        df['PhoneNumber'] = df['PhoneNumber'].apply(extract_numeric)
        operations.append("PhoneNumber: extraction des numéros de téléphone")

    # HospitalType - catégoriel, pas de valeurs manquantes, pas de traitement (une seule valeur)
    # HospitalOwner - 7.9% de valeurs manquantes, imputation par mode conditionnel
    if 'HospitalOwner' in df.columns:
        if df['HospitalOwner'].isna().mean() > 0.05:
            group_col = find_best_grouping_column(df, 'HospitalOwner', False)
            if group_col:
                df['HospitalOwner'] = df.groupby(group_col)['HospitalOwner'].transform(
                    lambda s: s.fillna(s.mode().iloc[0] if not s.mode().empty else "Unknown")
                )
            df['HospitalOwner'] = df['HospitalOwner'].fillna(df['HospitalOwner'].mode().iloc[0])
            operations.append(f"HospitalOwner: {df['HospitalOwner'].isna().sum()} valeurs manquantes imputées par mode")

        valid_owners = [
            "voluntary non-profit - private", "proprietary", "government - hospital district or authority",
            "voluntary non-profit - other", "voluntary non-profit - church", "government - federal",
            "government - state", "government - local", "unknown"
        ]
        df['HospitalOwner'] = df['HospitalOwner'].apply(lambda v: harmonize_category(v, valid_owners))

    # EmergencyService - 8.4% de valeurs manquantes, imputation par mode conditionnel
    if 'EmergencyService' in df.columns:
        if df['EmergencyService'].isna().mean() > 0.05:
            group_col = find_best_grouping_column(df, 'EmergencyService', False)
            if group_col:
                df['EmergencyService'] = df.groupby(group_col)['EmergencyService'].transform(
                    lambda s: s.fillna(s.mode().iloc[0] if not s.mode().empty else "Unknown")
                )
            df['EmergencyService'] = df['EmergencyService'].fillna(df['EmergencyService'].mode().iloc[0])
            operations.append(f"EmergencyService: {df['EmergencyService'].isna().sum()} valeurs manquantes imputées par mode")

        valid_emergency = ["yes", "no", "unknown"]
        df['EmergencyService'] = df['EmergencyService'].apply(lambda v: harmonize_category(v, valid_emergency))

    # Condition - catégoriel, pas de valeurs manquantes, harmonisation
    if 'Condition' in df.columns:
        valid_conditions = [
            "surgical infection prevention", "heart attack", "pneumonia", "heart failure",
            "children s asthma care"
        ]
        df['Condition'] = df['Condition'].apply(lambda v: harmonize_category(v, valid_conditions))

    # MeasureCode - catégoriel, pas de valeurs manquantes, pas de traitement spécifique
    # MeasureName - catégoriel, pas de valeurs manquantes, pas de traitement spécifique

    # Score - catégoriel avec valeurs numériques et "empty", conversion en numérique si possible
    if 'Score' in df.columns:
        df['Score'] = df['Score'].replace('empty', np.nan)
        df['Score'] = df['Score'].str.rstrip('%').apply(extract_numeric)
        df['Score'] = df['Score'].fillna(df['Score'].median())
        operations.append("Score: conversion en numérique et imputation des valeurs manquantes")

    # Sample - catégoriel avec valeurs comme "0 patients", extraction du nombre
    if 'Sample' in df.columns:
        df['Sample'] = df['Sample'].replace('empty', np.nan)
        df['Sample'] = df['Sample'].str.extract(r'(\d+)')[0].apply(extract_numeric)
        df['Sample'] = df['Sample'].fillna(df['Sample'].median())
        operations.append("Sample: extraction des nombres et imputation des valeurs manquantes")

    # Stateavg - catégoriel, pas de valeurs manquantes, pas de traitement spécifique

    # Sauvegarde du dataset nettoyé
    df.to_csv(OUTPUT_PATH, index=False)

    # Affichage du résumé des opérations
    print("=== Résumé du nettoyage ===")
    print(f"Lignes initiales: {initial_rows}")
    print(f"Doublons supprimés: {duplicates_removed}")
    print(f"Lignes finales: {len(df)}")
    print("\nOpérations effectuées:")
    for op in operations:
        print(f"- {op}")

if __name__ == "__main__":
    clean_data()