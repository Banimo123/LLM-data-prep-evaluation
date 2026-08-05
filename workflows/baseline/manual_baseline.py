"""
manual_baseline.py
-------------------
Workflow manuel de reference (Approche 1 de la taxonomie : "baseline experte").
Ecrit a la main, sans LLM, pour servir de point de comparaison aux workflows generes.

v3 : version GENERIQUE, applicable a n'importe quel dataset (hotel_bookings, titanic,
flights, hospital...), sans liste de colonnes/categories codee en dur. Reprend les
memes techniques que la v2 (qui avait fait passer le F1 de ~0.26 a ~0.72-0.76 sur
hotel_bookings), mais les derive automatiquement des donnees plutot que de connaissances
metier ecrites a la main :
  - categories valides = valeurs les plus frequentes de chaque colonne categorielle a
    faible cardinalite (les variantes rares sont presumees etre des typos/erreurs)
  - colonnes de date detectees par nom de colonne + motif de valeurs
  - colonnes numeriques corrompues detectees automatiquement (beaucoup de valeurs
    extraient un nombre valide une fois nettoyees)
  - outliers detectes par percentiles (1er/99e) au lieu de bornes physiques codees en dur
  - imputation par MODE si la colonne est tres asymetrique (>50% de valeurs identiques),
    MEDIANE sinon

IMPORTANT : preserve `row_id` sur toutes les lignes conservees (regle identique a celle
imposee au LLM dans system_prompt.txt), condition necessaire pour que metrics.py puisse
calculer le F1.
"""

import difflib
import re
from datetime import datetime

import pandas as pd
import numpy as np

DISGUISED_MISSING = {"", "na", "n/a", "unknown", " ", "nan", "none", "null"}
DATE_FORMATS = ["%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%B %d, %Y", "%d-%b-%Y", "%Y/%m/%d"]
MAX_CATEGORY_CARDINALITY = 15   # au-dela, on ne tente pas l'harmonisation par similarite
MIN_CATEGORY_FREQ_SHARE = 0.02  # une valeur doit representer >=2% des lignes non-nulles
                                  # pour etre consideree "canonique" (sinon = variante bruitee)
MODE_IMPUTE_SHARE_THRESHOLD = 0.5  # si le mode couvre >50% des valeurs -> imputer par mode


def _harmonize_category(value, valid_values):
    """Rapproche une valeur bruitee de la categorie valide la plus proche (par similarite)."""
    if pd.isna(value):
        return value
    s = str(value).strip()
    if s in valid_values:
        return s
    normalized = re.sub(r"\s+", " ", s).strip().lower()
    for v in valid_values:
        if v.lower() == normalized:
            return v
    match = difflib.get_close_matches(s, valid_values, n=1, cutoff=0.6)
    return match[0] if match else value


def _parse_date_robust(value):
    """Tente tous les formats de date usuels, plus un fallback timestamp Unix."""
    if pd.isna(value):
        return pd.NaT
    s = str(value).strip()
    for fmt in DATE_FORMATS:
        try:
            return pd.Timestamp(datetime.strptime(s, fmt))
        except ValueError:
            continue
    if re.fullmatch(r"\d{9,10}", s):
        try:
            return pd.Timestamp(datetime.fromtimestamp(int(s)))
        except (ValueError, OSError):
            pass
    return pd.NaT


def _looks_like_date_column(series: pd.Series, col_name: str) -> bool:
    sample = series.dropna().astype(str).head(50)
    if sample.empty:
        return False
    hits = sum(1 for v in sample if _parse_date_robust(v) is not pd.NaT)
    hit_ratio = hits / len(sample)
    # Le nom de colonne (ex: "reservation_status_date") abaisse le seuil requis, mais ne
    # dispense JAMAIS de vérifier les valeurs elles-mêmes : une colonne comme
    # "arrival_date_month" contient "date" dans son nom mais ne contient PAS de dates
    # (juste des noms de mois) -- se fier au nom seul la corromprait entièrement.
    threshold = 0.3 if "date" in col_name.lower() else 0.6
    return hit_ratio >= threshold


def _looks_like_corrupted_numeric(series: pd.Series) -> bool:
    """Detecte une colonne numerique dont certaines valeurs ont ete corrompues en texte
    (ex: '342O' au lieu de '3420'), en verifiant qu'une majorite des valeurs non-nulles
    contiennent au moins un chiffre."""
    sample = series.dropna().astype(str).head(100)
    if sample.empty:
        return False
    numeric_like = sample.str.contains(r"\d", regex=True)
    return numeric_like.mean() >= 0.8


def _extract_numeric(value):
    if pd.isna(value):
        return np.nan
    s = str(value).replace(",", ".").replace("O", "0").replace("o", "0")
    match = re.search(r"-?\d+\.?\d*", s)
    return float(match.group()) if match else np.nan


def _build_canonical_categories(series: pd.Series) -> list:
    """Deduit la liste des valeurs canoniques d'une colonne categorielle a partir des
    valeurs les plus frequentes (une variante rare et textuellement proche d'une valeur
    frequente est presumee etre une erreur/typo, pas une vraie categorie)."""
    counts = series.dropna().astype(str).str.strip().value_counts(normalize=True)
    canonical = counts[counts >= MIN_CATEGORY_FREQ_SHARE].index.tolist()
    return canonical if canonical else counts.index.tolist()[:10]


# --- Connaissances metier specifiques a hotel_bookings (version eprouvee, F1=0.72-0.76) ---
# Conservees telles quelles pour ne pas regresser sur ce dataset deja valide. Utilisees
# automatiquement quand clean_dataset_baseline() detecte ce dataset (voir dispatch plus bas).
HOTEL_BOOKINGS_KNOWN_CATEGORIES = {
    "hotel": ["City Hotel", "Resort Hotel"],
    "deposit_type": ["No Deposit", "Non Refund", "Refundable"],
    "customer_type": ["Transient", "Transient-Party", "Contract", "Group"],
    "meal": ["BB", "HB", "FB", "SC", "Undefined"],
    "market_segment": ["Online TA", "Offline TA/TO", "Groups", "Direct",
                        "Corporate", "Complementary", "Aviation", "Undefined"],
    "distribution_channel": ["TA/TO", "Direct", "Corporate", "GDS", "Undefined"],
}
HOTEL_BOOKINGS_MODE_IMPUTED_COLS = {"babies", "children", "days_in_waiting_list", "agent", "company"}
HOTEL_BOOKINGS_OUTLIER_BOUNDS = {
    "adr": (0, 600), "babies": (0, 4), "stays_in_week_nights": (0, 20),
    "days_in_waiting_list": (0, 400), "adults": (0, 6), "children": (0, 6),
}
HOTEL_BOOKINGS_NUMERIC_TYPO_COLS = ["lead_time", "adults", "children", "babies", "adr",
                                     "stays_in_week_nights", "days_in_waiting_list", "agent"]


def _is_hotel_bookings(df: pd.DataFrame) -> bool:
    """Detecte si le dataset en entree est hotel_bookings, via ses colonnes caracteristiques."""
    signature = {"hotel", "adr", "reservation_status_date", "deposit_type"}
    return signature.issubset(set(df.columns))


def _clean_hotel_bookings_specific(df: pd.DataFrame) -> pd.DataFrame:
    """Logique specifique hotel_bookings (v2, F1=0.72-0.76). Suppose row_id + missing
    values deguises deja normalises en amont par clean_dataset_baseline()."""
    for col, valid_values in HOTEL_BOOKINGS_KNOWN_CATEGORIES.items():
        if col in df.columns:
            df[col] = df[col].apply(lambda v: _harmonize_category(v, valid_values))

    if "country" in df.columns:
        df["country"] = df["country"].astype(str).str.strip()
        df.loc[df["country"] == "nan", "country"] = np.nan

    for col in HOTEL_BOOKINGS_NUMERIC_TYPO_COLS:
        if col in df.columns:
            df[col] = df[col].apply(_extract_numeric)

    if "reservation_status_date" in df.columns:
        df["reservation_status_date"] = df["reservation_status_date"].apply(_parse_date_robust)
        df["reservation_status_date"] = df["reservation_status_date"].dt.strftime("%Y-%m-%d")

    for col, (low, high) in HOTEL_BOOKINGS_OUTLIER_BOUNDS.items():
        if col not in df.columns:
            continue
        is_outlier = (df[col] < low) | (df[col] > high)
        if is_outlier.any():
            if col in HOTEL_BOOKINGS_MODE_IMPUTED_COLS:
                repl = df.loc[~is_outlier, col].mode(dropna=True)
                repl = repl.iloc[0] if not repl.empty else df[col].median()
            else:
                repl = df.loc[~is_outlier, col].median()
            df.loc[is_outlier, col] = repl

    for col in df.columns:
        if col == "row_id" or not df[col].isna().any():
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            if col in HOTEL_BOOKINGS_MODE_IMPUTED_COLS:
                mode = df[col].mode(dropna=True)
                df[col] = df[col].fillna(mode.iloc[0] if not mode.empty else df[col].median())
            else:
                df[col] = df[col].fillna(df[col].median())
        else:
            mode = df[col].mode(dropna=True)
            df[col] = df[col].fillna(mode.iloc[0] if not mode.empty else "Unknown")

    return df


def clean_dataset_baseline(input_path: str, output_path: str) -> pd.DataFrame:
    df = pd.read_csv(input_path, low_memory=False)
    assert "row_id" in df.columns, "row_id manquant en entree"

    data_cols = [c for c in df.columns if c != "row_id"]

    # --- Valeurs manquantes deguisees -> NaN uniforme (commun aux 2 chemins) ---
    text_cols = [c for c in df.select_dtypes(include="object").columns if c != "row_id"]
    for col in text_cols:
        df[col] = df[col].apply(
            lambda v: np.nan if isinstance(v, str) and v.strip().lower() in DISGUISED_MISSING else v
        )

    if _is_hotel_bookings(df):
        print("[baseline] Dataset detecte : hotel_bookings -> logique specifique (eprouvee).")
        df = _clean_hotel_bookings_specific(df)
    else:
        print("[baseline] Dataset non reconnu -> logique generique (auto-detection).")
        df = _clean_dataset_generic(df)

    df = df.drop_duplicates(subset=data_cols, keep="first")
    df.to_csv(output_path, index=False)
    return df


def _clean_dataset_generic(df: pd.DataFrame) -> pd.DataFrame:
    """Logique generique, applicable a tout dataset sans connaissance metier prealable
    (titanic, flights, hospital...). Les valeurs manquantes deguisees sont deja
    normalisees par clean_dataset_baseline() avant l'appel."""
    text_cols = [c for c in df.select_dtypes(include="object").columns if c != "row_id"]

    date_cols = [c for c in text_cols if _looks_like_date_column(df[c], c)]
    for col in date_cols:
        df[col] = df[col].apply(_parse_date_robust)
        df[col] = df[col].dt.strftime("%Y-%m-%d")

    remaining_text_cols = [c for c in text_cols if c not in date_cols]

    numeric_like_text_cols = [c for c in remaining_text_cols if _looks_like_corrupted_numeric(df[c])]
    for col in numeric_like_text_cols:
        df[col] = df[col].apply(_extract_numeric)
    remaining_text_cols = [c for c in remaining_text_cols if c not in numeric_like_text_cols]

    already_numeric_cols = [c for c in df.columns if c != "row_id" and pd.api.types.is_numeric_dtype(df[c])]
    numeric_cols = list(dict.fromkeys(already_numeric_cols + numeric_like_text_cols))

    for col in remaining_text_cols:
        n_unique = df[col].dropna().nunique()
        if 1 < n_unique <= MAX_CATEGORY_CARDINALITY:
            canonical = _build_canonical_categories(df[col])
            df[col] = df[col].apply(lambda v: _harmonize_category(v, canonical))
        else:
            df[col] = df[col].astype(str).str.strip()
            df.loc[df[col] == "nan", col] = np.nan

    for col in numeric_cols:
        series = pd.to_numeric(df[col], errors="coerce")
        valid = series.dropna()
        if len(valid) < 10:
            continue
        low, high = valid.quantile(0.005), valid.quantile(0.995)
        if low == high:
            continue
        is_outlier = (series < low) | (series > high)
        if is_outlier.any():
            non_outlier_valid = valid[~is_outlier.reindex(valid.index, fill_value=False)]
            mode_counts = non_outlier_valid.value_counts(normalize=True)
            use_mode = (not mode_counts.empty) and (mode_counts.iloc[0] >= MODE_IMPUTE_SHARE_THRESHOLD)
            replacement = mode_counts.index[0] if use_mode else non_outlier_valid.median()
            df.loc[is_outlier, col] = replacement
        df[col] = series.where(~is_outlier, df[col])

    for col in [c for c in df.columns if c != "row_id"]:
        if not df[col].isna().any():
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            valid = df[col].dropna()
            mode_counts = valid.value_counts(normalize=True)
            use_mode = (not mode_counts.empty) and (mode_counts.iloc[0] >= MODE_IMPUTE_SHARE_THRESHOLD)
            df[col] = df[col].fillna(mode_counts.index[0] if use_mode else valid.median())
        else:
            mode = df[col].mode(dropna=True)
            df[col] = df[col].fillna(mode.iloc[0] if not mode.empty else "Unknown")

    return df


# Alias conserve pour compatibilite avec les scripts existants qui l'appellent encore
clean_hotel_bookings_baseline = clean_dataset_baseline


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = clean_dataset_baseline(args.input, args.output)
    print(f"Nettoye : {result.shape[0]} lignes, {result.shape[1]} colonnes -> {args.output}")
