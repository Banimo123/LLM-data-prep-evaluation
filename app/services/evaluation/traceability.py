"""
traceability.py
------------------
Produit un journal structuré des transformations effectuées par un workflow de
nettoyage : quelles cellules ont été modifiées, dans quelles colonnes, et quelle
était la valeur avant/après (Phase 5 du cahier des charges, dimension "Traçabilité").

Contrairement à metrics.py (qui compare le résultat à une vérité terrain clean.csv,
donc évalue la QUALITÉ), ce module compare le résultat au fichier BRUITÉ D'ENTRÉE
(noisy.csv), donc répond à la question "qu'est-ce que le workflow a réellement fait ?"
— utile même sur des données réelles où l'on n'a pas de vérité terrain.

Répond directement aux questions du cahier des charges (section évaluation traçabilité) :
    - Peut-on savoir quelles cellules ont été modifiées ?
    - Peut-on savoir quelle opération (colonne) a modifié quelle cellule ?
    - Le workflow produit-il un journal des transformations ?
    - Peut-on revenir à la valeur initiale ? (oui : value_before est conservée)

Utilisation basique :
    from app.services.evaluation.traceability import compute_traceability

    report = compute_traceability(
        noisy_csv="benchmark/datasets/hotel_bookings/noisy_low.csv",
        cleaned_csv="results/cleaned_datasets/hotel_bookings/noisy_low__manual_baseline.csv",
    )
    print(report["summary_global"])
"""

import argparse
import json
from pathlib import Path

import pandas as pd


def _to_comparable_str(value) -> str:
    """
    Normalise une valeur pour comparaison stricte (évite les faux positifs dus à des
    différences de type/représentation, ex: 3 vs 3.0 vs "3" vs "3.0").

    Essaie d'abord une interprétation numérique (fonctionne que la valeur soit déjà
    un float/int, ou une chaîne comme "304.0"), pour gérer les colonnes de type mixte
    qui apparaissent après injection de bruit (une partie des cellules devient du texte).
    Si la valeur n'est pas numériquement interprétable, retombe sur une comparaison
    texte classique.
    """
    if pd.isna(value):
        return "<NA>"

    try:
        f = float(value)
        if f.is_integer():
            return str(int(f))
        return str(f)
    except (ValueError, TypeError):
        return str(value).strip()


def compute_traceability(noisy_csv: str, cleaned_csv: str, id_col: str = "row_id") -> dict:
    """
    Compare le dataset bruité (entrée) au dataset nettoyé (sortie) et produit :
        - detailed_log : DataFrame avec une ligne par cellule modifiée
                          (row_id, column, value_before, value_after)
        - summary_per_column : dict {colonne: {n_modified, pct_modified}}
        - summary_global : dict avec les statistiques globales

    Si `cleaned_csv` ne contient pas `row_id`, retourne une erreur explicite plutôt
    que de planter (violation de la règle system_prompt sur la préservation de row_id).
    """
    df_noisy = pd.read_csv(noisy_csv, low_memory=False)
    df_cleaned = pd.read_csv(cleaned_csv, low_memory=False)

    if id_col not in df_cleaned.columns:
        return {
            "error": f"'{id_col}' absent du fichier nettoyé — traçabilité impossible sans identifiant stable.",
            "detailed_log": pd.DataFrame(),
            "summary_per_column": {},
            "summary_global": {},
        }

    df_noisy = df_noisy.set_index(id_col)
    df_cleaned = df_cleaned.set_index(id_col)

    noisy_ids = set(df_noisy.index)
    cleaned_ids = set(df_cleaned.index)
    deleted_row_ids = noisy_ids - cleaned_ids  # lignes supprimées par le workflow

    noisy_cols = set(df_noisy.columns)
    cleaned_cols = set(df_cleaned.columns)
    deleted_columns = sorted(noisy_cols - cleaned_cols)
    invented_columns = sorted(cleaned_cols - noisy_cols)  # ne devrait jamais arriver (règle system_prompt)

    common_columns = sorted(noisy_cols & cleaned_cols)
    active_ids = sorted(cleaned_ids & noisy_ids)  # lignes présentes des deux côtés, comparables

    records = []
    per_column_counts = {col: 0 for col in common_columns}

    for col in common_columns:
        before = df_noisy.loc[active_ids, col]
        after = df_cleaned.loc[active_ids, col]

        before_str = before.map(_to_comparable_str)
        after_str = after.map(_to_comparable_str)

        modified_mask = before_str != after_str
        n_modified = int(modified_mask.sum())
        per_column_counts[col] = n_modified

        if n_modified > 0:
            modified_ids = [rid for rid, m in zip(active_ids, modified_mask) if m]
            for rid in modified_ids:
                records.append({
                    "row_id": rid,
                    "column": col,
                    "value_before": before.loc[rid],
                    "value_after": after.loc[rid],
                })

    # Lignes supprimées : journalisées séparément (toutes les colonnes "perdues" d'un coup)
    for rid in sorted(deleted_row_ids):
        records.append({
            "row_id": rid,
            "column": "<ALL>",
            "value_before": "<row present>",
            "value_after": "<row deleted>",
        })

    detailed_log = pd.DataFrame(records)

    summary_per_column = {
        col: {
            "n_modified": count,
            "pct_modified": round(100 * count / len(active_ids), 2) if active_ids else 0.0,
        }
        for col, count in sorted(per_column_counts.items(), key=lambda x: -x[1])
    }

    total_cells_modified = sum(per_column_counts.values())

    summary_global = {
        "n_rows_before": len(df_noisy),
        "n_rows_after": len(df_cleaned),
        "n_rows_deleted": len(deleted_row_ids),
        "n_columns_before": len(df_noisy.columns),
        "n_columns_after": len(df_cleaned.columns),
        "columns_deleted": deleted_columns,
        "columns_invented": invented_columns,
        "total_cells_modified": total_cells_modified,
        "pct_cells_modified": round(
            100 * total_cells_modified / (len(active_ids) * len(common_columns)), 2
        ) if active_ids and common_columns else 0.0,
    }

    return {
        "detailed_log": detailed_log,
        "summary_per_column": summary_per_column,
        "summary_global": summary_global,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Génère le journal de traçabilité d'un workflow de nettoyage.")
    parser.add_argument("--noisy_csv", required=True, help="Fichier bruité d'entrée (avant nettoyage)")
    parser.add_argument("--cleaned_csv", required=True, help="Fichier nettoyé produit par le workflow (après)")
    parser.add_argument("--output_csv", default=None, help="Chemin où sauvegarder le journal détaillé (CSV)")
    args = parser.parse_args()

    report = compute_traceability(args.noisy_csv, args.cleaned_csv)

    if "error" in report and report["error"]:
        print(f"Erreur : {report['error']}")
        exit(1)

    print("--- Résumé global ---")
    print(json.dumps(report["summary_global"], indent=2, ensure_ascii=False))

    print("\n--- Cellules modifiées par colonne (triées par volume décroissant) ---")
    print(json.dumps(report["summary_per_column"], indent=2, ensure_ascii=False))

    if args.output_csv:
        output_path = Path(args.output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        report["detailed_log"].to_csv(output_path, index=False)
        print(f"\nJournal détaillé ({len(report['detailed_log'])} lignes) sauvegardé dans {output_path}")