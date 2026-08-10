import pandas as pd
import numpy as np
import re
from datetime import datetime
import difflib

INPUT_PATH = r"datasets\flights\noisy_medium.csv"
OUTPUT_PATH = r"results\cleaned_datasets\flights\noisy_medium__profile.csv"

def extract_numeric(value):
    if pd.isna(value):
        return np.nan
    s = str(value).replace(",", ".")
    s = s.replace("O", "0").replace("o", "0")
    match = re.search(r"-?\d+\.?\d*", s)
    return float(match.group()) if match else np.nan

def parse_time(value):
    if pd.isna(value) or value == "":
        return value
    value = str(value).strip().lower()
    if value == "unknown":
        return value
    try:
        for fmt in ["%I:%M %p", "%H:%M", "%I:%M%p", "%I:%M%P"]:
            try:
                dt = datetime.strptime(value, fmt)
                return dt.strftime("%I:%M %p").lower()
            except ValueError:
                continue
        return value
    except:
        return value

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

def clean_dataset():
    df = pd.read_csv(INPUT_PATH)

    # Log initial
    initial_rows, initial_cols = df.shape
    print(f"Dataset initial: {initial_rows} lignes, {initial_cols} colonnes")

    # Suppression des doublons (conservation du premier)
    duplicates_before = df.duplicated().sum()
    df = df.drop_duplicates().reset_index(drop=True)
    duplicates_after = df.duplicated().sum()
    print(f"Doublons supprimés: {duplicates_before - duplicates_after}")

    # Nettoyage colonne tuple_id (numeric)
    if "tuple_id" in df.columns:
        df["tuple_id"] = df["tuple_id"].apply(extract_numeric)
        df["tuple_id"] = pd.to_numeric(df["tuple_id"], errors="coerce")
        # Imputation des valeurs manquantes (0% dans le profil)
        df["tuple_id"] = df["tuple_id"].fillna(df["tuple_id"].median())
        # Correction des valeurs aberrantes (min=1, max=2376)
        df["tuple_id"] = df["tuple_id"].clip(1, 2376)
        df["tuple_id"] = df["tuple_id"].round().astype(int)

    # Nettoyage colonne src (categorical)
    if "src" in df.columns:
        valid_values_src = ["gofox", "orbitz", "businesstravellogue", "flightview", "flights",
                           "panynj", "allegiantair", "helloflight", "flightstats", "myrateplan"]
        df["src"] = df["src"].apply(lambda v: harmonize_category(v, valid_values_src))
        # Imputation des valeurs manquantes (15.99%)
        group_col = find_best_grouping_column(df, "src", is_numeric=False)
        if group_col:
            df["src"] = df.groupby(group_col)["src"].transform(
                lambda s: s.fillna(s.mode().iloc[0] if not s.mode().empty else "gofox")
            )
        df["src"] = df["src"].fillna(df["src"].mode().iloc[0])

    # Nettoyage colonne flight (categorical)
    if "flight" in df.columns:
        df["flight"] = df["flight"].astype(str).str.strip()
        # Correction des valeurs aberrantes (ex: "6:300 p.m." dans l'extrait)
        df["flight"] = df["flight"].apply(lambda x: x if re.match(r"^[A-Za-z]{2}-\d{1,4}-[A-Z]{3}-[A-Z]{3}$", x) else np.nan)
        # Imputation des valeurs manquantes (15.78%)
        group_col = find_best_grouping_column(df, "flight", is_numeric=False)
        if group_col:
            df["flight"] = df.groupby(group_col)["flight"].transform(
                lambda s: s.fillna(s.mode().iloc[0] if not s.mode().empty else "unknown")
            )
        df["flight"] = df["flight"].fillna("unknown")

    # Nettoyage colonnes de temps (sched_dep_time, act_dep_time, sched_arr_time, act_arr_time)
    time_cols = ["sched_dep_time", "act_dep_time", "sched_arr_time", "act_arr_time"]
    for col in time_cols:
        if col in df.columns:
            df[col] = df[col].apply(parse_time)
            # Imputation des valeurs manquantes (0% dans le profil)
            df[col] = df[col].fillna(df[col].mode().iloc[0])

    # Correction spécifique pour l'erreur visible dans l'extrait (6:300 p.m.)
    if "act_dep_time" in df.columns:
        df["act_dep_time"] = df["act_dep_time"].replace("6:300 p.m.", "6:30 p.m.")

    # Vérification finale des types
    for col in df.columns:
        if col == "row_id":
            continue
        if df[col].dtype == "object":
            df[col] = df[col].astype(str).replace("nan", np.nan)
        elif np.issubdtype(df[col].dtype, np.number):
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Log final
    final_rows, final_cols = df.shape
    print(f"Dataset nettoyé: {final_rows} lignes, {final_cols} colonnes")
    print(f"Lignes supprimées: {initial_rows - final_rows}")
    print(f"Valeurs manquantes après nettoyage:\n{df.isna().sum()}")

    # Sauvegarde
    df.to_csv(OUTPUT_PATH, index=False)

if __name__ == "__main__":
    clean_dataset()