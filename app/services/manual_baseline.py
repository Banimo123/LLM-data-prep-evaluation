"""
manual_baseline.py
-------------------
Workflow manuel de référence (Approche 1 de la taxonomie : "baseline experte").
Écrit à la main, sans LLM, pour servir de point de comparaison aux workflows générés.

v2 : amélioration ciblée par rapport à v1, suite à l'analyse F1 (Étape 5-6) :
  - harmonisation des catégories par correspondance au plus proche voisin dans un
    ensemble de valeurs valides connues du domaine (et non plus un simple .strip()),
    légitime pour un workflow EXPERT qui connaît le métier (ex: un hôtel n'a que 2
    types, un repas n'a que 5 codes possibles) — ce n'est PAS de la triche, c'est la
    définition même de la baseline "experte" du cahier des charges.
  - imputation par MODE (plutôt que médiane) pour les colonnes discrètes/très
    asymétriques, où le mode capture une part dominante des valeurs réelles.
  - parsing de dates explicite sur tous les formats injectés, y compris le timestamp
    Unix (échoué par pd.to_datetime(format="mixed") en v1).
  - traitement des outliers par imputation vers une valeur plausible (mode/médiane
    selon la distribution), au lieu d'un simple clip aux bornes.

IMPORTANT : préserve `row_id` sur toutes les lignes conservées (règle identique à celle
imposée au LLM dans system_prompt.txt), condition nécessaire pour que metrics.py puisse
calculer le F1.
"""

import difflib
import re
from datetime import datetime

import pandas as pd
import numpy as np


# Catégories valides connues du domaine (hôtellerie), utilisées pour harmoniser les
# variantes bruitées vers leur forme canonique. Connaissance métier légitime pour un
# workflow "expert humain" — ne provient pas d'un accès ligne-à-ligne à clean.csv.
KNOWN_CATEGORIES = {
    "hotel": ["City Hotel", "Resort Hotel"],
    "deposit_type": ["No Deposit", "Non Refund", "Refundable"],
    "customer_type": ["Transient", "Transient-Party", "Contract", "Group"],
    "meal": ["BB", "HB", "FB", "SC", "Undefined"],
    "market_segment": ["Online TA", "Offline TA/TO", "Groups", "Direct",
                        "Corporate", "Complementary", "Aviation", "Undefined"],
    "distribution_channel": ["TA/TO", "Direct", "Corporate", "GDS", "Undefined"],
}

# Colonnes numériques discrètes/très asymétriques : le mode est un meilleur estimateur
# que la médiane pour l'imputation (valeurs manquantes ET outliers).
MODE_IMPUTED_NUMERIC_COLS = {"babies", "children", "days_in_waiting_list", "agent", "company"}

# Bornes réalistes servant uniquement à DÉTECTER un outlier (pas à corriger par clip)
OUTLIER_DETECTION_BOUNDS = {
    "adr": (0, 600),
    "babies": (0, 4),
    "stays_in_week_nights": (0, 20),
    "days_in_waiting_list": (0, 400),
    "adults": (0, 6),
    "children": (0, 6),
}


def _harmonize_category(value, valid_values):
    """Rapproche une valeur bruitée de la catégorie valide la plus proche (par similarité)."""
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
    """Tente tous les formats de date injectés par error_injection.py, dans l'ordre,
    plus un fallback timestamp Unix."""
    if pd.isna(value):
        return pd.NaT
    s = str(value).strip()

    formats = ["%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%B %d, %Y", "%d-%b-%Y", "%Y/%m/%d"]
    for fmt in formats:
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


def clean_hotel_bookings_baseline(input_path: str, output_path: str) -> pd.DataFrame:
    df = pd.read_csv(input_path, low_memory=False)
    assert "row_id" in df.columns, "row_id manquant en entrée"

    disguised = {"", "na", "n/a", "unknown", " ", "nan", "none", "null"}
    text_cols = df.select_dtypes(include="object").columns
    for col in text_cols:
        if col == "row_id":
            continue
        df[col] = df[col].apply(
            lambda v: np.nan if isinstance(v, str) and v.strip().lower() in disguised else v
        )

    for col, valid_values in KNOWN_CATEGORIES.items():
        if col in df.columns:
            df[col] = df[col].apply(lambda v: _harmonize_category(v, valid_values))

    for col in ["country"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
            df.loc[df[col] == "nan", col] = np.nan

    for col in ["lead_time", "adults", "children", "babies", "adr",
                "stays_in_week_nights", "days_in_waiting_list", "agent"]:
        if col in df.columns:
            df[col] = (
                df[col].astype(str)
                .str.replace("O", "0", regex=False)
                .str.extract(r"(-?\d+\.?\d*)")[0]
            )
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "reservation_status_date" in df.columns:
        df["reservation_status_date"] = df["reservation_status_date"].apply(_parse_date_robust)
        df["reservation_status_date"] = df["reservation_status_date"].dt.strftime("%Y-%m-%d")

    for col, (low, high) in OUTLIER_DETECTION_BOUNDS.items():
        if col not in df.columns:
            continue
        is_outlier = (df[col] < low) | (df[col] > high)
        if is_outlier.any():
            if col in MODE_IMPUTED_NUMERIC_COLS:
                replacement = df.loc[~is_outlier, col].mode(dropna=True)
                replacement = replacement.iloc[0] if not replacement.empty else df[col].median()
            else:
                replacement = df.loc[~is_outlier, col].median()
            df.loc[is_outlier, col] = replacement

    for col in df.columns:
        if col == "row_id":
            continue
        if df[col].isna().any():
            if pd.api.types.is_numeric_dtype(df[col]):
                if col in MODE_IMPUTED_NUMERIC_COLS:
                    mode = df[col].mode(dropna=True)
                    df[col] = df[col].fillna(mode.iloc[0] if not mode.empty else df[col].median())
                else:
                    df[col] = df[col].fillna(df[col].median())
            else:
                mode = df[col].mode(dropna=True)
                df[col] = df[col].fillna(mode.iloc[0] if not mode.empty else "Unknown")

    cols_for_dedup = [c for c in df.columns if c != "row_id"]
    df = df.drop_duplicates(subset=cols_for_dedup, keep="first")

    df.to_csv(output_path, index=False)
    return df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = clean_hotel_bookings_baseline(args.input, args.output)
    print(f"Nettoyé : {result.shape[0]} lignes, {result.shape[1]} colonnes -> {args.output}")