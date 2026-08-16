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
    formats = [
        ("%I:%M %p", r"^\d{1,2}:\d{2} [ap]\.?m\.?$"),
        ("%I:%M%p", r"^\d{1,2}:\d{2}[ap]\.?m\.?$"),
        ("%H:%M", r"^\d{1,2}:\d{2}$")
    ]
    for fmt, pattern in formats:
        if re.match(pattern, s):
            try:
                dt = datetime.strptime(s, fmt)
                return dt.strftime("%I:%M %p").replace("AM", "a.m.").replace("PM", "p.m.")
            except ValueError:
                continue
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
                if len(grp) >= 5:
                    w_mad += (grp - grp.median()).abs().median() * (len(grp) / total)
                else:
                    w_mad += global_mad * (len(grp) / total)
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
    original_row_ids = df["row_id"].copy()

    # Nettoyage de tuple_id (numeric)
    if "tuple_id" in df.columns:
        # Vérification des valeurs aberrantes (min=1, max=2376)
        df["tuple_id"] = pd.to_numeric(df["tuple_id"], errors="coerce")
        mask = (df["tuple_id"] < 1) | (df["tuple_id"] > 2376)
        if mask.any():
            median_val = df.loc[~mask, "tuple_id"].median()
            df.loc[mask, "tuple_id"] = median_val
        # Imputation des valeurs manquantes (0% dans le profil, mais au cas où)
        df["tuple_id"] = df["tuple_id"].fillna(df["tuple_id"].median())

    # Nettoyage de src (categorical)
    if "src" in df.columns:
        valid_src = ["helloflight", "boston", "flightview", "airtravelcenter", "flights",
                     "panynj", "gofox", "allegiantair", "myrateplan", "aa"]
        df["src"] = df["src"].apply(lambda v: harmonize_category(v, valid_src))
        # Imputation des valeurs manquantes (0% dans le profil)
        df["src"] = df["src"].fillna(df["src"].mode()[0])

    # Nettoyage de flight (categorical)
    if "flight" in df.columns:
        # Extraction des valeurs fréquentes comme canoniques
        valid_flight = ["UA-664-ORD-PHL", "AA-59-JFK-SFO", "AA-1640-MIA-MCO", "AA-2050-ORD-MIA",
                        "AA-789-ORD-DEN", "AA-3786-IAH-ORD", "AA-1221-MCO-ORD", "AA-4307-ORD-DTW",
                        "AA-431-MIA-SFO", "AA-1733-ORD-PHX"]
        df["flight"] = df["flight"].apply(lambda v: harmonize_category(v, valid_flight))
        # Imputation des valeurs manquantes (0% dans le profil)
        df["flight"] = df["flight"].fillna(df["flight"].mode()[0])

    # Nettoyage des colonnes de temps
    time_cols = ["sched_dep_time", "act_dep_time", "sched_arr_time", "act_arr_time"]
    for col in time_cols:
        if col in df.columns:
            # Standardisation du format des heures
            df[col] = df[col].apply(parse_time)
            # Imputation conditionnelle des valeurs manquantes
            group_col = find_best_grouping_column(df, col, is_numeric=False)
            if group_col:
                df[col] = df.groupby(group_col)[col].transform(
                    lambda s: s.fillna(s.mode().iloc[0] if not s.mode().empty else "Unknown")
                )
            df[col] = df[col].fillna(df[col].mode()[0])

    # Suppression des doublons (en conservant row_id d'origine)
    initial_count = len(df)
    df = df.drop_duplicates(subset=[c for c in df.columns if c != "row_id"])
    final_count = len(df)
    duplicates_removed = initial_count - final_count

    # Restauration de row_id
    df["row_id"] = original_row_ids.loc[df.index]

    # Log des opérations
    print(f"Nettoyage terminé:")
    print(f"- Valeurs manquantes imputées: {df.isna().sum().sum()}")
    print(f"- Doublons supprimés: {duplicates_removed}")
    print(f"- Lignes restantes: {len(df)}")
    print(f"- Colonnes: {len(df.columns)}")

    # Sauvegarde
    df.to_csv(OUTPUT_PATH, index=False)

if __name__ == "__main__":
    clean_dataset()