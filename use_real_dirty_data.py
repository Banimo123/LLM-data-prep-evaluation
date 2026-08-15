"""
use_real_dirty_data.py
------------------------
Remplace l'injection d'erreurs SYNTHETIQUE par de VRAIES donnees sales (clean.csv / dirty.csv
deja fournis, alignes ligne a ligne). Reconstruit automatiquement, par diff cellule par cellule,
le fichier injected_errors_low.csv necessaire au calcul du F1 -- sans lui, impossible de savoir
quelles cellules etaient "a corriger".

Un seul niveau de bruit est genere (low), puisqu'il n'existe qu'une seule version "dirty" reelle
(pas de notion de "moyen"/"fort" pour des donnees reelles, contrairement au bruit synthetique).
Les eventuels anciens noisy_medium.csv/noisy_high.csv et injected_errors_medium/high.csv de ce
dataset sont supprimes, pour ne pas laisser trainer des fichiers perimes.

Usage :
    python use_real_dirty_data.py --dataset flights --clean flights/clean.csv --dirty flights/dirty.csv --output_dir datasets/flights --column_mapping "tuple_id:tuple_id,src:src,..."
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


def _classify_error(clean_val, noisy_val) -> str:
    """Heuristique simple pour classer le type d'erreur reelle observee, afin de rester
    compatible avec les 4 familles deja utilisees par le reste du pipeline (missing_values,
    format_errors, outliers, typos). Par defaut : 'typos' (categorie generique la plus sure)."""
    c_is_na = pd.isna(clean_val)
    n_is_na = pd.isna(noisy_val) or (isinstance(noisy_val, str) and noisy_val.strip() == "")
    if n_is_na and not c_is_na:
        return "missing_values"
    if c_is_na:
        return "typos"

    c_str, n_str = str(clean_val).strip(), str(noisy_val).strip()

    # Ecart purement numerique important -> probable outlier
    try:
        c_num, n_num = float(c_str), float(n_str)
        if c_num != 0 and abs(n_num - c_num) / abs(c_num) > 0.5:
            return "outliers"
        if c_num != n_num:
            return "typos"
    except (ValueError, TypeError):
        pass

    # Meme valeur mais casse/espaces differents, ou motif date -> format_errors
    if c_str.lower().replace(" ", "") == n_str.lower().replace(" ", ""):
        return "typos"
    if re.search(r"\d{1,4}[/-]\d{1,2}[/-]\d{1,4}", c_str) or re.search(r"\d{1,4}[/-]\d{1,2}[/-]\d{1,4}", n_str):
        return "format_errors"

    return "typos"


def diff_to_injected_errors(df_clean: pd.DataFrame, df_noisy: pd.DataFrame) -> pd.DataFrame:
    """Compare clean et noisy (deja alignes par row_id et memes noms de colonnes) et produit
    un DataFrame au meme format que ceux generes par error_injection.py."""
    records = []
    data_cols = [c for c in df_clean.columns if c != "row_id"]
    clean_idx = df_clean.set_index("row_id")
    noisy_idx = df_noisy.set_index("row_id")

    for col in data_cols:
        c_series = clean_idx[col]
        n_series = noisy_idx[col]
        for rid in clean_idx.index:
            c_val, n_val = c_series.loc[rid], n_series.loc[rid]
            same = (pd.isna(c_val) and pd.isna(n_val)) or (
                not pd.isna(c_val) and not pd.isna(n_val) and str(c_val).strip() == str(n_val).strip()
            )
            if same:
                continue
            family = _classify_error(c_val, n_val)
            records.append({
                "row_id": rid,
                "column": col,
                "original_value": c_val,
                "injected_value": n_val,
                "error_type": "real_world",
                "error_family": family,
            })
    return pd.DataFrame(records)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean", required=True)
    parser.add_argument("--dirty", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--id_col", default=None,
                         help="Nom de la colonne d'identifiant commune aux 2 fichiers pour l'alignement (ex: 'index'). Si absent, suppose que les lignes sont deja dans le meme ordre.")
    parser.add_argument("--rename_dirty_columns", default=None,
                         help="Mapping 'ancien_nom:nouveau_nom,...' si dirty.csv utilise des noms de colonnes differents de clean.csv")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df_clean = pd.read_csv(args.clean, low_memory=False)
    df_dirty = pd.read_csv(args.dirty, low_memory=False)

    if args.rename_dirty_columns:
        mapping = dict(p.split(":") for p in args.rename_dirty_columns.split(","))
        df_dirty = df_dirty.rename(columns=mapping)

    assert list(df_clean.columns) == list(df_dirty.columns), (
        f"Colonnes non alignees apres renommage : {list(df_clean.columns)} vs {list(df_dirty.columns)}"
    )
    assert len(df_clean) == len(df_dirty), "Nombre de lignes different entre clean et dirty"

    # row_id : soit derive d'une colonne d'identifiant existante, soit position (0..N-1)
    if args.id_col:
        row_id = df_clean[args.id_col].values
        assert (df_clean[args.id_col].values == df_dirty[args.id_col].values).all(), \
            "La colonne d'identifiant ne correspond pas ligne a ligne entre clean et dirty"
    else:
        row_id = range(len(df_clean))

    df_clean = df_clean.copy()
    df_dirty = df_dirty.copy()
    df_clean.insert(0, "row_id", row_id)
    df_dirty.insert(0, "row_id", row_id)

    # Sauvegarde clean_with_id.csv et noisy_low.csv
    df_clean.to_csv(output_dir / "clean_with_id.csv", index=False)
    df_dirty.to_csv(output_dir / "noisy_low.csv", index=False)

    # Diff -> injected_errors_low.csv
    errors_df = diff_to_injected_errors(df_clean, df_dirty)
    errors_df.to_csv(output_dir / "injected_errors_low.csv", index=False)

    # Nettoyage des anciens fichiers synthetiques medium/high (plus applicables : un seul
    # niveau reel disponible)
    for level in ["medium", "high"]:
        for pattern in [f"noisy_{level}.csv", f"injected_errors_{level}.csv"]:
            f = output_dir / pattern
            if f.exists():
                f.unlink()
                print(f"Supprime (synthetique, plus applicable) : {f}")

    print(f"\nOK : {len(errors_df)} differences reelles detectees entre clean et dirty ({len(df_clean)} lignes, {len(df_clean.columns)-1} colonnes).")
    print(f"Repartition par famille :")
    print(errors_df["error_family"].value_counts().to_string())
    print(f"\nFichiers ecrits dans {output_dir} : clean_with_id.csv, noisy_low.csv, injected_errors_low.csv")


if __name__ == "__main__":
    main()
