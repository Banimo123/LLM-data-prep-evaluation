"""
metrics.py
----------
Calcule les métriques de QUALITÉ du nettoyage (Phase 5 du cahier des charges) :
    - F1 cellule-par-cellule (précision/rappel de correction)
    - F1 de déduplication

Prérequis impératif : les 3 fichiers comparés (clean, noisy, cleaned) doivent tous
contenir une colonne `row_id` stable, permettant de réaligner les lignes même si le
workflow de nettoyage en a supprimé (doublons, filtres, etc.).

Définitions (alignées avec le cahier des charges, section "Évaluation de la qualité") :

    TP (vrai positif)  : cellule marquée erronée dans injected_errors.csv, ET la valeur
                          nettoyée == la valeur d'origine dans clean.csv (correction réussie).
    FN (faux négatif)  : cellule marquée erronée, mais valeur nettoyée != valeur d'origine
                          (non corrigée, ou mal corrigée).
    FP (faux positif)  : cellule NON marquée erronée (donc correcte dans noisy.csv), mais dont
                          la valeur a été modifiée par le workflow ET diffère de la valeur
                          d'origine (= nouvelle erreur introduite / sur-nettoyage).
    TN (vrai négatif)  : cellule NON marquée erronée et non modifiée par le workflow. Non utilisé
                          dans le F1 (comme pour toute tâche déséquilibrée), mais retourné pour info.

    Precision = TP / (TP + FP)   -> "quand le workflow change une cellule, a-t-il raison ?"
    Recall    = TP / (TP + FN)   -> "sur toutes les erreurs réelles, combien sont corrigées ?"
    F1        = 2 * P * R / (P + R)

Une ligne supprimée par le workflow (absente de cleaned.csv) :
    - si son row_id correspond à une ligne marquée comme doublon dans injected_errors
      (error_family == "duplicate", si vous l'ajoutez plus tard) -> comptée dans le F1 de dédup,
      PAS dans le F1 cellule-par-cellule.
    - sinon -> toutes ses cellules erronées comptent comme FN (ligne perdue = non corrigée),
      et c'est signalé séparément ("lignes perdues à tort") pour l'analyse d'erreurs (Phase 6).

Utilisation basique :
    from app.services.evaluation.metrics import evaluate_workflow

    report = evaluate_workflow(
        clean_csv="benchmark/datasets/hotel_bookings/clean.csv",
        noisy_csv="benchmark/datasets/hotel_bookings/noisy_low.csv",
        injected_errors_csv="benchmark/datasets/hotel_bookings/injected_errors_low.csv",
        cleaned_csv="results/cleaned_datasets/hotel_bookings/noisy_low__simple.csv",
    )
    print(report["global"])
"""

import pandas as pd
import numpy as np


def _to_comparable_str(value) -> str:
    """
    Normalise une valeur pour comparaison stricte (évite les faux FN/FP dus uniquement
    à des différences de type, ex: 3 vs 3.0 vs "3", ou NaN vs None vs "").
    """
    if pd.isna(value):
        return "<NA>"
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip()


def compute_deduplication_f1(clean_row_ids: set, cleaned_row_ids: set, duplicate_row_ids: set) -> dict:
    """
    Évalue la capacité du workflow à supprimer les BONNES lignes en double, et seulement
    celles-là.

    Parameters
    ----------
    clean_row_ids : set
        Tous les row_id du dataset bruité en entrée (avant nettoyage).
    cleaned_row_ids : set
        Les row_id encore présents dans le fichier nettoyé (après le workflow).
    duplicate_row_ids : set
        Les row_id qui étaient de VRAIS doublons à supprimer (vérité terrain). Si vous
        n'avez pas encore de doublons injectés explicitement, passez un set vide : le
        F1 de dédup sera alors non défini (None) plutôt que trompeur.
    """
    if not duplicate_row_ids:
        return {"precision": None, "recall": None, "f1": None, "note": "aucun doublon connu injecté"}

    removed_row_ids = clean_row_ids - cleaned_row_ids

    tp = len(removed_row_ids & duplicate_row_ids)
    fp = len(removed_row_ids - duplicate_row_ids)   # supprimées à tort (pas de vrais doublons)
    fn = len(duplicate_row_ids - removed_row_ids)   # doublons non supprimés

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "n_removed_total": len(removed_row_ids),
        "n_true_duplicates": len(duplicate_row_ids),
        "tp": tp, "fp": fp, "fn": fn,
    }


def _is_numeric_series(series: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(series)


def _values_match(clean_raw, cleaned_raw, is_numeric: bool, rel_tol: float, abs_tol: float) -> bool:
    """
    Compare une valeur nettoyée à la valeur de vérité terrain.

    - Colonnes catégorielles/texte : égalité stricte (après normalisation via
      _to_comparable_str), car il n'y a pas de notion de "proche" pour du texte.
    - Colonnes numériques : tolérance relative + absolue, car exiger une égalité
      stricte sur une variable continue (ex: adr = 45.23€) est irréaliste — aucun
      workflow, même excellent, ne peut deviner la décimale exacte d'une valeur
      corrompue ou manquante. On considère la correction réussie si la valeur
      retombe suffisamment proche de la vraie valeur (tolérance par défaut : 1%
      relatif, ou 0.5 en absolu pour les valeurs proches de 0).
    """
    if pd.isna(clean_raw):
        return pd.isna(cleaned_raw)

    if is_numeric:
        try:
            clean_f = float(clean_raw)
            cleaned_f = float(cleaned_raw)
        except (TypeError, ValueError):
            return False
        if pd.isna(cleaned_f):
            return False
        diff = abs(clean_f - cleaned_f)
        tolerance = max(abs_tol, rel_tol * abs(clean_f))
        return diff <= tolerance

    return _to_comparable_str(clean_raw) == _to_comparable_str(cleaned_raw)


def compute_cell_level_f1(
    df_clean: pd.DataFrame,
    df_noisy: pd.DataFrame,
    df_errors: pd.DataFrame,
    df_cleaned: pd.DataFrame,
    id_col: str = "row_id",
    numeric_rel_tol: float = 0.01,
    numeric_abs_tol: float = 0.5,
) -> dict:
    """
    Calcule le F1 cellule-par-cellule, colonne par colonne puis globalement.

    df_errors doit contenir au minimum : row_id, column (nom de la colonne bruitée),
    error_family (ex: "missing_values", "typos", "format_errors", "outliers").

    numeric_rel_tol / numeric_abs_tol : tolérance appliquée aux colonnes numériques
    continues pour juger qu'une correction "a réussi" (voir _values_match). Mettre les
    deux à 0 pour revenir à une égalité stricte.
    """
    numeric_cols = {c for c in df_clean.columns if c != id_col and _is_numeric_series(df_clean[c])}

    df_clean = df_clean.set_index(id_col)
    df_noisy = df_noisy.set_index(id_col)
    df_cleaned_indexed = df_cleaned.set_index(id_col)

    cleaned_ids = set(df_cleaned_indexed.index)
    lost_ids = set(df_noisy.index) - cleaned_ids  # lignes supprimées par le workflow

    # Ensemble des cellules erronées connues : {(row_id, column): error_family}
    error_cells = {
        (row.row_id, row.column): row.error_family
        for row in df_errors.itertuples(index=False)
    }

    per_column_counts = {}   # col -> {"tp":.., "fp":.., "fn":..}
    per_family_counts = {}   # error_family -> {"tp":.., "fn":..}  (FP n'a pas de famille)
    n_cells_lost_to_dropped_rows = 0
    columns_dropped_by_workflow = []

    data_columns = [c for c in df_clean.columns if c != id_col]

    for col in data_columns:
        per_column_counts.setdefault(col, {"tp": 0, "fp": 0, "fn": 0})
        is_numeric = col in numeric_cols

        # Colonne entière supprimée par le workflow (viole la règle du system_prompt) :
        # toutes ses cellules d'erreur connue comptent en FN, sans planter.
        column_dropped = col not in df_cleaned_indexed.columns
        if column_dropped:
            columns_dropped_by_workflow.append(col)

        for rid in df_noisy.index:
            is_error_cell = (rid, col) in error_cells

            if rid in lost_ids or column_dropped:
                # Ligne supprimée par le workflow, OU colonne entière supprimée : si elle
                # contenait une erreur connue, c'est une non-correction (FN). Sinon, on ignore.
                if is_error_cell:
                    per_column_counts[col]["fn"] += 1
                    fam = error_cells[(rid, col)]
                    per_family_counts.setdefault(fam, {"tp": 0, "fn": 0})
                    per_family_counts[fam]["fn"] += 1
                    if rid in lost_ids:
                        n_cells_lost_to_dropped_rows += 1
                continue

            clean_raw = df_clean.at[rid, col] if rid in df_clean.index else None
            cleaned_raw = df_cleaned_indexed.at[rid, col]
            noisy_val = _to_comparable_str(df_noisy.at[rid, col])
            clean_val_str = _to_comparable_str(clean_raw) if clean_raw is not None else None

            matches_ground_truth = (clean_raw is not None) and _values_match(
                clean_raw, cleaned_raw, is_numeric, numeric_rel_tol, numeric_abs_tol
            )

            if is_error_cell:
                fam = error_cells[(rid, col)]
                per_family_counts.setdefault(fam, {"tp": 0, "fn": 0})
                if matches_ground_truth:
                    per_column_counts[col]["tp"] += 1
                    per_family_counts[fam]["tp"] += 1
                else:
                    per_column_counts[col]["fn"] += 1
                    per_family_counts[fam]["fn"] += 1
            else:
                # Cellule correcte à l'origine : si le workflow l'a modifiée ET que le
                # résultat est faux -> nouvelle erreur introduite (FP).
                #
                # Exception : si clean_val == "<NA>", la vraie valeur est intrinsèquement
                # inconnue (déjà manquante dans clean.csv AVANT toute injection d'erreur,
                # ex: beaucoup de réservations sans agent connu). Un workflow ne peut pas
                # deviner qu'il doit laisser cette cellule vide sans avoir accès à clean.csv
                # -> on ne le pénalise pas pour une imputation "raisonnable" de ce cas précis.
                if clean_val_str == "<NA>":
                    continue
                cleaned_val_str = _to_comparable_str(cleaned_raw)
                was_changed = cleaned_val_str != noisy_val
                if was_changed and not matches_ground_truth:
                    per_column_counts[col]["fp"] += 1

    def _prf(tp, fp, fn):
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        return round(precision, 4), round(recall, 4), round(f1, 4)

    per_column_report = {}
    tp_total = fp_total = fn_total = 0
    for col, counts in per_column_counts.items():
        p, r, f1 = _prf(counts["tp"], counts["fp"], counts["fn"])
        per_column_report[col] = {**counts, "precision": p, "recall": r, "f1": f1}
        tp_total += counts["tp"]
        fp_total += counts["fp"]
        fn_total += counts["fn"]

    per_family_report = {}
    for fam, counts in per_family_counts.items():
        p, r, f1 = _prf(counts["tp"], 0, counts["fn"])  # FP non attribuable à une famille
        per_family_report[fam] = {**counts, "recall": r, "f1_sans_fp": f1}

    global_precision, global_recall, global_f1 = _prf(tp_total, fp_total, fn_total)

    return {
        "global": {
            "precision": global_precision,
            "recall": global_recall,
            "f1": global_f1,
            "tp": tp_total, "fp": fp_total, "fn": fn_total,
            "n_rows_lost_by_workflow": len(lost_ids),
            "n_error_cells_lost_with_dropped_rows": n_cells_lost_to_dropped_rows,
            "columns_dropped_by_workflow": columns_dropped_by_workflow,
        },
        "per_column": per_column_report,
        "per_error_family": per_family_report,
    }


def evaluate_workflow(
    clean_csv: str,
    noisy_csv: str,
    injected_errors_csv: str,
    cleaned_csv: str,
    duplicate_row_ids: set = None,
    id_col: str = "row_id",
    numeric_rel_tol: float = 0.01,
    numeric_abs_tol: float = 0.5,
) -> dict:
    """
    Point d'entrée haut niveau : charge les 4 fichiers et retourne le rapport complet
    (F1 cellule-par-cellule + F1 de déduplication si duplicate_row_ids est fourni).

    numeric_rel_tol / numeric_abs_tol : tolérance de comparaison pour les colonnes
    numériques continues (voir _values_match). Par défaut 1% relatif ou ±0.5 en absolu.
    """
    df_clean = pd.read_csv(clean_csv, low_memory=False)
    df_noisy = pd.read_csv(noisy_csv, low_memory=False)
    df_errors = pd.read_csv(injected_errors_csv, low_memory=False)
    df_cleaned = pd.read_csv(cleaned_csv, low_memory=False)

    for name, df in [("clean_csv", df_clean), ("noisy_csv", df_noisy), ("cleaned_csv", df_cleaned)]:
        if id_col not in df.columns:
            raise ValueError(
                f"'{id_col}' est absent de {name} ({clean_csv if name=='clean_csv' else ''}"
                f"{noisy_csv if name=='noisy_csv' else ''}{cleaned_csv if name=='cleaned_csv' else ''}). "
                f"Impossible de calculer le F1 sans identifiant de ligne stable."
            )

    cell_report = compute_cell_level_f1(
        df_clean, df_noisy, df_errors, df_cleaned, id_col=id_col,
        numeric_rel_tol=numeric_rel_tol, numeric_abs_tol=numeric_abs_tol,
    )

    dedup_report = compute_deduplication_f1(
        clean_row_ids=set(df_noisy[id_col]),
        cleaned_row_ids=set(df_cleaned[id_col]),
        duplicate_row_ids=duplicate_row_ids or set(),
    )

    return {
        "global": cell_report["global"],
        "per_column": cell_report["per_column"],
        "per_error_family": cell_report["per_error_family"],
        "deduplication": dedup_report,
    }


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Calcule le F1 qualité d'un workflow de nettoyage.")
    parser.add_argument("--clean_csv", required=True)
    parser.add_argument("--noisy_csv", required=True)
    parser.add_argument("--injected_errors_csv", required=True)
    parser.add_argument("--cleaned_csv", required=True)
    parser.add_argument("--output_json", default=None)
    args = parser.parse_args()

    report = evaluate_workflow(
        clean_csv=args.clean_csv,
        noisy_csv=args.noisy_csv,
        injected_errors_csv=args.injected_errors_csv,
        cleaned_csv=args.cleaned_csv,
    )

    print(json.dumps(report["global"], indent=2, ensure_ascii=False))
    print("\n--- Par famille d'erreur ---")
    print(json.dumps(report["per_error_family"], indent=2, ensure_ascii=False))

    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"\nRapport complet sauvegardé dans {args.output_json}")