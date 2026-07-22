"""
error_injection.py
-------------------
Module d'injection d'erreurs contrôlées dans le dataset hotel_bookings.

Origine : code fusionné de l'équipe (Yassir : format + outliers ;
Fatima-Ezzahra : missing values + typos), déplacé tel quel depuis
le notebook `03-fusion-inject-all.ipynb`, aucune logique de fonction modifiée.

Ajout dans cette version : génération de 3 niveaux de bruit (low / medium / high)
en faisant varier les `error_rate` de chaque famille d'erreur, comme demandé par
le cahier des charges (Phase 2).

Répartition des colonnes (validée en équipe, sans conflit) :
    reservation_status_date                                -> format (Yassir)
    adr, babies, stays_in_week_nights, days_in_waiting_list -> outliers (Yassir)
    country, agent, children, market_segment, meal          -> missing values (Fatima-Ezzahra)
    hotel, deposit_type, customer_type, lead_time, adults    -> typos (Fatima-Ezzahra)

Utilisation :
    python -m app.services.error_injection \
        --input benchmark/datasets/hotel_bookings/clean.csv \
        --output_dir benchmark/datasets/hotel_bookings
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 1. inject_format_errors — Yassir
# ---------------------------------------------------------------------------
def inject_format_errors(df, columns=None, error_rate=0.15, random_state=42):
    """
    Injecte des incohérences de FORMAT sur des colonnes de type date.
    """
    rng = np.random.default_rng(random_state)
    df_noisy = df.copy()
    columns = columns or ["reservation_status_date"]
    log_records = []

    def reformat(date_str):
        try:
            d = pd.to_datetime(date_str)
        except Exception:
            return date_str, "unparsable_original"

        variants = [
            d.strftime("%d/%m/%Y"),
            d.strftime("%m/%d/%Y"),
            d.strftime("%B %d, %Y"),
            d.strftime("%d-%b-%Y"),
            d.strftime("%Y/%m/%d"),
            str(int(d.timestamp())),
        ]
        choice = rng.integers(0, len(variants))
        return variants[choice], "date_format_inconsistency"

    for col in columns:
        if col not in df_noisy.columns:
            continue
        n_rows = len(df_noisy)
        n_to_corrupt = int(n_rows * error_rate)
        idx_to_corrupt = rng.choice(n_rows, size=n_to_corrupt, replace=False)

        for idx in idx_to_corrupt:
            original_value = df_noisy.at[idx, col]
            new_value, err_type = reformat(original_value)
            df_noisy.at[idx, col] = new_value
            log_records.append({
                "row_index": idx,
                "column": col,
                "original_value": original_value,
                "injected_value": new_value,
                "error_type": err_type,
            })

    error_log = pd.DataFrame(log_records)
    return df_noisy, error_log


# ---------------------------------------------------------------------------
# 2. inject_outliers — Yassir
# ---------------------------------------------------------------------------
def inject_outliers(df, column_bounds=None, error_rate=0.02, random_state=42):
    """
    Injecte des valeurs ABERRANTES sur des colonnes numériques.
    """
    rng = np.random.default_rng(random_state)
    df_noisy = df.copy()
    log_records = []

    default_bounds = {
        "adr": (-500, 10000),
        "babies": (10, 50),
        "stays_in_week_nights": (200, 1000),
        "days_in_waiting_list": (2000, 9000),
    }
    column_bounds = column_bounds or default_bounds

    for col, (low, high) in column_bounds.items():
        if col not in df_noisy.columns:
            continue
        n_rows = len(df_noisy)
        n_to_corrupt = max(1, int(n_rows * error_rate))
        idx_to_corrupt = rng.choice(n_rows, size=n_to_corrupt, replace=False)

        for idx in idx_to_corrupt:
            original_value = df_noisy.at[idx, col]
            outlier_value = rng.uniform(low, high)
            if pd.api.types.is_integer_dtype(df[col]):
                outlier_value = int(outlier_value)
            else:
                outlier_value = round(float(outlier_value), 2)

            df_noisy.at[idx, col] = outlier_value
            log_records.append({
                "row_index": idx,
                "column": col,
                "original_value": original_value,
                "injected_value": outlier_value,
                "error_type": "outlier_value",
            })

    error_log = pd.DataFrame(log_records)
    return df_noisy, error_log


# ---------------------------------------------------------------------------
# 3. inject_missing_values — Fatima-Ezzahra
# ---------------------------------------------------------------------------
def inject_missing_values(df, columns=None, error_rate=0.10, random_state=42):
    """
    Injecte des valeurs MANQUANTES (explicites ou déguisées) sur les colonnes ciblées.
    """
    rng = np.random.default_rng(random_state)
    df_noisy = df.copy()
    columns = columns or ["country", "agent", "children", "market_segment", "meal"]
    log_records = []

    variants = [
        (np.nan, "missing_nan"),
        ("", "missing_empty_string"),
        ("NA", "missing_NA_string"),
        ("N/A", "missing_N/A_string"),
        ("unknown", "missing_unknown_string"),
        (" ", "missing_whitespace"),
    ]

    for col in columns:
        if col not in df_noisy.columns:
            continue
        df_noisy[col] = df_noisy[col].astype(object)

        n_rows = len(df_noisy)
        n_to_corrupt = int(n_rows * error_rate)
        idx_to_corrupt = rng.choice(n_rows, size=n_to_corrupt, replace=False)

        for idx in idx_to_corrupt:
            original_value = df.at[idx, col]
            choice_idx = rng.integers(0, len(variants))
            new_value, err_type = variants[choice_idx]
            df_noisy.at[idx, col] = new_value
            log_records.append({
                "row_index": idx,
                "column": col,
                "original_value": original_value,
                "injected_value": new_value,
                "error_type": err_type,
            })

    error_log = pd.DataFrame(log_records)
    return df_noisy, error_log


# ---------------------------------------------------------------------------
# 4. inject_typos — Fatima-Ezzahra
# ---------------------------------------------------------------------------
def _typo_text(value, rng):
    """Applique une faute de frappe aléatoire à une chaîne de caractères."""
    s = str(value)
    if len(s) < 2:
        return s, "typo_too_short_unchanged"

    op = rng.integers(0, 5)
    i = rng.integers(0, len(s) - 1)

    if op == 0:
        new_s = s[:i] + s[i + 1:]
        err_type = "typo_missing_letter"
    elif op == 1:
        new_s = s[:i] + s[i] + s[i:]
        err_type = "typo_duplicated_letter"
    elif op == 2:
        new_s = s[:i] + s[i + 1] + s[i] + s[i + 2:]
        err_type = "typo_swapped_letters"
    elif op == 3:
        cut = rng.integers(1, len(s))
        new_s = s[:cut].upper() + s[cut:].lower()
        err_type = "typo_random_case"
    else:
        new_s = "  " + s + " "
        err_type = "typo_extra_whitespace"

    return new_s, err_type


def _typo_numeric(value, rng):
    """Injecte un ou plusieurs caractères non numériques dans une valeur numérique."""
    s = str(value)
    letter_pool = ["O", "l", "S", "B", "€", "kg", "j"]
    letter = letter_pool[rng.integers(0, len(letter_pool))]

    op = rng.integers(0, 3)
    if op == 0:
        if "0" in s:
            idx0 = s.index("0")
            new_s = s[:idx0] + "O" + s[idx0 + 1:]
        else:
            new_s = s + "O"
        err_type = "typo_zero_as_letter_O"
    elif op == 1:
        new_s = s + letter
        err_type = "typo_unit_suffix_injected"
    else:
        i = rng.integers(1, max(2, len(s)))
        new_s = s[:i] + letter + s[i:]
        err_type = "typo_letter_inside_number"

    return new_s, err_type


def inject_typos(df, text_columns=None, numeric_columns=None, error_rate=0.10, random_state=42):
    """
    Injecte des erreurs TYPOGRAPHIQUES : fautes de frappe sur colonnes textuelles,
    et caractères non numériques insérés dans des colonnes numériques.
    """
    rng = np.random.default_rng(random_state)
    df_noisy = df.copy()
    text_columns = text_columns or ["hotel", "deposit_type", "customer_type"]
    numeric_columns = numeric_columns or ["lead_time", "adults"]
    log_records = []

    def apply_corruption(columns, corrupt_fn):
        for col in columns:
            if col not in df_noisy.columns:
                continue
            df_noisy[col] = df_noisy[col].astype(object)

            n_rows = len(df_noisy)
            n_to_corrupt = int(n_rows * error_rate)
            idx_to_corrupt = rng.choice(n_rows, size=n_to_corrupt, replace=False)

            for idx in idx_to_corrupt:
                original_value = df.at[idx, col]
                if pd.isna(original_value):
                    continue
                new_value, err_type = corrupt_fn(original_value, rng)
                df_noisy.at[idx, col] = new_value
                log_records.append({
                    "row_index": idx,
                    "column": col,
                    "original_value": original_value,
                    "injected_value": new_value,
                    "error_type": err_type,
                })

    apply_corruption(text_columns, _typo_text)
    apply_corruption(numeric_columns, _typo_numeric)

    error_log = pd.DataFrame(log_records)
    return df_noisy, error_log


# ---------------------------------------------------------------------------
# 5. profile_dataset — Fatima-Ezzahra (utile pour l'approche "prompt profilage")
# ---------------------------------------------------------------------------
DISGUISED_MISSING = {"", "na", "n/a", "unknown", " ", "nan", "none", "null"}


def profile_dataset(df, top_n=5):
    """
    Construit un profil statistique du dataset, utilisable dans un prompt LLM (approche
    "prompt avec profilage").
    """
    records = []
    n_rows = len(df)

    for col in df.columns:
        series = df[col]
        is_numeric = pd.api.types.is_numeric_dtype(series)

        n_na_real = series.isna().sum()

        if not is_numeric:
            n_na_disguised = series.astype(str).str.strip().str.lower().isin(DISGUISED_MISSING).sum()
        else:
            n_na_disguised = 0

        pct_missing = round(100 * (n_na_real + n_na_disguised) / n_rows, 2)

        record = {
            "column": col,
            "dtype_detected": "numeric" if is_numeric else "categorical/text",
            "pct_missing": pct_missing,
            "n_unique": series.nunique(dropna=True),
        }

        if is_numeric:
            record["min"] = series.min()
            record["max"] = series.max()
            record["mean"] = round(series.mean(), 2)
            record["std"] = round(series.std(), 2)
        else:
            top_values = series.value_counts().head(top_n).to_dict()
            record["top_values"] = top_values

        records.append(record)

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# 6. inject_all — application séquentielle sur le même dataset
# ---------------------------------------------------------------------------
def inject_all(df_clean, random_state=42, rates=None):
    """
    Applique les 4 types d'erreurs sur une seule copie du dataset clean.
    """
    rates = rates or {"missing": 0.10, "format": 0.15, "outliers": 0.02, "typos": 0.08}

    df_current = df_clean.copy()
    all_logs = []

    df_current, log_missing = inject_missing_values(
        df_current,
        columns=["country", "agent", "children", "market_segment", "meal"],
        error_rate=rates["missing"], random_state=random_state)
    log_missing["error_family"] = "missing_values"
    all_logs.append(log_missing)

    df_current, log_format = inject_format_errors(
        df_current,
        columns=["reservation_status_date"],
        error_rate=rates["format"], random_state=random_state)
    log_format["error_family"] = "format_errors"
    all_logs.append(log_format)

    df_current, log_outliers = inject_outliers(
        df_current,
        column_bounds={
            "adr": (-500, 10000),
            "babies": (10, 50),
            "stays_in_week_nights": (200, 1000),
            "days_in_waiting_list": (2000, 9000),
        },
        error_rate=rates["outliers"], random_state=random_state)
    log_outliers["error_family"] = "outliers"
    all_logs.append(log_outliers)

    df_current, log_typos = inject_typos(
        df_current,
        text_columns=["hotel", "deposit_type", "customer_type"],
        numeric_columns=["lead_time", "adults"],
        error_rate=rates["typos"], random_state=random_state)
    log_typos["error_family"] = "typos"
    all_logs.append(log_typos)

    full_log = pd.concat(all_logs, ignore_index=True)
    return df_current, full_log


# ---------------------------------------------------------------------------
# 7. Niveaux de bruit — low / medium / high
# ---------------------------------------------------------------------------
# "medium" reprend les taux d'origine du notebook de l'équipe.
# "low" et "high" sont dérivés en multipliant chaque taux (avec un plafond à 0.5
# pour éviter de corrompre plus de la moitié d'une colonne).
BASE_RATES = {"missing": 0.10, "format": 0.15, "outliers": 0.02, "typos": 0.08}

NOISE_LEVEL_MULTIPLIERS = {
    "low": 0.5,
    "medium": 1.0,
    "high": 2.0,
}


def get_rates_for_level(level):
    multiplier = NOISE_LEVEL_MULTIPLIERS[level]
    return {k: min(0.5, round(v * multiplier, 4)) for k, v in BASE_RATES.items()}


# ---------------------------------------------------------------------------
# 8. Script principal
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Injection d'erreurs contrôlées (hotel_bookings).")
    parser.add_argument("--input", required=True, help="Chemin vers clean.csv")
    parser.add_argument("--output_dir", required=True, help="Dossier de sortie (ex: benchmark/datasets/hotel_bookings)")
    parser.add_argument("--random_state", type=int, default=42)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Lecture de {args.input} ...")
    df_clean = pd.read_csv(args.input)
    print(f"Dataset chargé : {df_clean.shape[0]} lignes, {df_clean.shape[1]} colonnes.\n")

    for level in ["low", "medium", "high"]:
        rates = get_rates_for_level(level)
        print(f"--- Niveau de bruit : {level} (rates={rates}) ---")

        df_final, full_log = inject_all(df_clean, random_state=args.random_state, rates=rates)

        out_csv = output_dir / f"noisy_{level}.csv"
        df_final.to_csv(out_csv, index=False)

        errors_csv = output_dir / f"injected_errors_{level}.csv"
        full_log.to_csv(errors_csv, index=False)

        print(f"  -> {out_csv} ({df_final.shape[0]} lignes)")
        print(f"  -> {errors_csv} ({len(full_log)} erreurs injectées)")
        print(f"  -> réparties par famille :")
        print(full_log.groupby("error_family").size().to_string())
        print()

    profile = profile_dataset(df_clean)
    profile_path = output_dir / "profile_clean.csv"
    profile.to_csv(profile_path, index=False)
    print(f"Profil du dataset clean sauvegardé dans {profile_path}")


if __name__ == "__main__":
    main()