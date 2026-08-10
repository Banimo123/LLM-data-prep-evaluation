import pandas as pd
import numpy as np
import re
from datetime import datetime
import difflib

INPUT_PATH = r"datasets\flights\noisy_low.csv"
OUTPUT_PATH = r"results\cleaned_datasets\flights\noisy_low__profile.csv"

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

def parse_time(value):
    if pd.isna(value):
        return value
    s = str(value).strip().lower()
    s = s.replace("a.m.", "AM").replace("p.m.", "PM").replace("a. m.", "AM").replace("p. m.", "PM")
    s = s.replace("am", "AM").replace("pm", "PM")
    try:
        dt = datetime.strptime(s, "%I:%M %p")
        return dt.strftime("%H:%M")
    except ValueError:
        try:
            dt = datetime.strptime(s, "%I:%M%p")
            return dt.strftime("%H:%M")
        except ValueError:
            try:
                dt = datetime.strptime(s, "%H:%M")
                return dt.strftime("%H:%M")
            except ValueError:
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

    # Conservation des opérations effectuées pour le log final
    operations = {
        "missing_values_imputed": {},
        "outliers_corrected": {},
        "categories_harmonized": {},
        "duplicates_removed": 0,
        "time_formats_standardized": set()
    }

    # Nettoyage de la colonne tuple_id (numeric)
    if "tuple_id" in df.columns:
        # Vérification des valeurs aberrantes (aucune selon le profil)
        df["tuple_id"] = pd.to_numeric(df["tuple_id"], errors="coerce")
        # Imputation des valeurs manquantes (0% selon le profil, mais au cas où)
        if df["tuple_id"].isna().any():
            operations["missing_values_imputed"]["tuple_id"] = df["tuple_id"].isna().sum()
            df["tuple_id"] = df["tuple_id"].fillna(df["tuple_id"].median())

    # Nettoyage de la colonne src (categorical)
    if "src" in df.columns:
        # Harmonisation des catégories
        valid_src = ["orbitz", "panynj", "helloflight", "airtravelcenter", "gofox",
                     "myrateplan", "flytecomm", "flightstats", "flights", "businesstravellogue"]
        df["src"] = df["src"].apply(lambda v: harmonize_category(v, valid_src))
        operations["categories_harmonized"]["src"] = True

        # Imputation des valeurs manquantes (8.46%)
        if df["src"].isna().any():
            group_col = find_best_grouping_column(df, "src", False)
            if group_col:
                df["src"] = df.groupby(group_col)["src"].transform(
                    lambda s: s.fillna(s.mode().iloc[0] if not s.mode().empty else "orbitz")
                )
            df["src"] = df["src"].fillna(df["src"].mode().iloc[0])
            operations["missing_values_imputed"]["src"] = df["src"].isna().sum()

    # Nettoyage de la colonne flight (categorical)
    if "flight" in df.columns:
        # Harmonisation des valeurs fréquentes
        valid_flight_prefixes = ["AA", "UA", "DL", "WN", "B6", "NK", "F9", "AS", "HA", "G4"]
        def clean_flight(value):
            if pd.isna(value):
                return value
            s = str(value).strip()
            parts = s.split("-")
            if len(parts) >= 2 and parts[0] in valid_flight_prefixes:
                return f"{parts[0]}-{'-'.join(parts[1:])}"
            return s
        df["flight"] = df["flight"].apply(clean_flight)

        # Imputation des valeurs manquantes (7.87%)
        if df["flight"].isna().any():
            group_col = find_best_grouping_column(df, "flight", False)
            if group_col:
                df["flight"] = df.groupby(group_col)["flight"].transform(
                    lambda s: s.fillna(s.mode().iloc[0] if not s.mode().empty else "AA-1640-MIA-MCO")
                )
            df["flight"] = df["flight"].fillna(df["flight"].mode().iloc[0])
            operations["missing_values_imputed"]["flight"] = df["flight"].isna().sum()

    # Nettoyage des colonnes de temps
    time_cols = ["sched_dep_time", "act_dep_time", "sched_arr_time", "act_arr_time"]
    for col in time_cols:
        if col in df.columns:
            df[col] = df[col].apply(parse_time)
            operations["time_formats_standardized"].add(col)

    # Suppression des doublons (en conservant le premier)
    initial_rows = len(df)
    df = df.drop_duplicates(subset=[c for c in df.columns if c != "row_id"], keep="first")
    operations["duplicates_removed"] = initial_rows - len(df)

    # Sauvegarde du dataset nettoyé
    df.to_csv(OUTPUT_PATH, index=False)

    # Affichage du résumé des opérations
    print("=== Nettoyage terminé ===")
    print(f"Lignes supprimées (doublons) : {operations['duplicates_removed']}")
    print("\nValeurs manquantes imputées par colonne :")
    for col, count in operations["missing_values_imputed"].items():
        print(f"  {col} : {count} valeurs")
    print("\nColonnes catégorielles harmonisées :")
    for col in operations["categories_harmonized"]:
        print(f"  {col}")
    print("\nFormats de temps standardisés :")
    for col in operations["time_formats_standardized"]:
        print(f"  {col}")

if __name__ == "__main__":
    clean_dataset()