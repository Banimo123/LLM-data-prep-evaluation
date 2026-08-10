import pandas as pd
import numpy as np
import re
from datetime import datetime
import difflib

INPUT_PATH = r'datasets\titanic\noisy_low.csv'
OUTPUT_PATH = r'results\cleaned_datasets\titanic\noisy_low__profile.csv'

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

def parse_date(value):
    if pd.isna(value):
        return value
    formats = ["%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%B %d, %Y", "%d-%b-%Y", "%Y/%m/%d"]
    for fmt in formats:
        try:
            return datetime.strptime(str(value), fmt).strftime("%Y-%m-%d")
        except (ValueError, TypeError):
            continue
    if str(value).isdigit() and len(str(value)) in (9, 10):
        try:
            return datetime.fromtimestamp(int(value)).strftime("%Y-%m-%d")
        except (ValueError, OSError):
            pass
    return str(value)

df = pd.read_csv(INPUT_PATH)

# Suppression des doublons en conservant la première occurrence
initial_rows = len(df)
df = df.drop_duplicates(subset=[col for col in df.columns if col != "row_id"], keep='first')
duplicates_removed = initial_rows - len(df)

# Nettoyage colonne par colonne
operations = []

# Unnamed: 0 - colonne technique, pas de nettoyage nécessaire
if 'Unnamed: 0' in df.columns:
    pass

# PassengerId - identifiant, pas de nettoyage nécessaire
if 'PassengerId' in df.columns:
    pass

# Survived - binaire, pas de valeurs manquantes, pas d'aberrations détectées
if 'Survived' in df.columns:
    pass

# Sex - binaire, pas de valeurs manquantes, pas d'aberrations détectées
if 'Sex' in df.columns:
    pass

# Age - colonne catégorielle avec 7.45% de valeurs manquantes, valeurs fréquentes sous forme numérique
if 'Age' in df.columns:
    # Extraction des valeurs numériques (certaines peuvent être sous forme texte)
    df['Age'] = df['Age'].apply(extract_numeric)

    # Détection des valeurs aberrantes (physiquement impossibles)
    age_median = df['Age'].median()
    age_q99 = df['Age'].quantile(0.99)
    age_min = 0
    age_max = 100  # borne physique raisonnable

    def correct_age(value):
        if pd.isna(value):
            return value
        if value < age_min or value > age_max or value > age_q99:
            return age_median
        return value

    df['Age'] = df['Age'].apply(correct_age)

    # Imputation des valeurs manquantes
    group_col = find_best_grouping_column(df, 'Age', True)
    if group_col:
        df['Age'] = df.groupby(group_col)['Age'].transform(
            lambda s: s.fillna(s.median())
        )
    df['Age'] = df['Age'].fillna(age_median)
    operations.append(f"Age: {df['Age'].isna().sum()} valeurs manquantes imputées par médiane")

# Fare - colonne catégorielle avec valeurs numériques sous forme texte
if 'Fare' in df.columns:
    # Extraction des valeurs numériques
    df['Fare'] = df['Fare'].apply(extract_numeric)

    # Détection des valeurs aberrantes
    fare_median = df['Fare'].median()
    fare_q99 = df['Fare'].quantile(0.99)
    fare_min = 0

    def correct_fare(value):
        if pd.isna(value):
            return value
        if value < fare_min or value > fare_q99:
            return fare_median
        return value

    df['Fare'] = df['Fare'].apply(correct_fare)

    # Pas de valeurs manquantes à imputer (0% dans le profil)
    operations.append(f"Fare: valeurs aberrantes corrigées")

# Colonnes Pclass_* - binaires, pas de nettoyage nécessaire
for col in ['Pclass_1', 'Pclass_2', 'Pclass_3']:
    if col in df.columns:
        pass

# Family_size - binaire, pas de nettoyage nécessaire
if 'Family_size' in df.columns:
    pass

# Colonnes Title_* - binaires, pas de nettoyage nécessaire
for col in ['Title_1', 'Title_2', 'Title_3', 'Title_4']:
    if col in df.columns:
        pass

# Colonnes Emb_* - binaires, pas de nettoyage nécessaire
for col in ['Emb_1', 'Emb_2', 'Emb_3']:
    if col in df.columns:
        pass

# Vérification finale des types
for col in df.columns:
    if col == 'row_id':
        continue
    if df[col].dtype == 'object':
        # Vérification si la colonne peut être convertie en numérique
        try:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            if df[col].notna().all():
                df[col] = df[col].fillna(df[col].median())
        except:
            pass

# Sauvegarde du dataset nettoyé
df.to_csv(OUTPUT_PATH, index=False)

# Log des opérations
print(f"Nettoyage terminé. {duplicates_removed} doublons supprimés.")
for op in operations:
    print(op)
print(f"Dataset final: {len(df)} lignes, {len(df.columns)} colonnes.")