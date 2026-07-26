"""
quality.py
----------
Calcule les métriques de qualité du nettoyage (Phase 5, dimension 1) :
précision, rappel, F1-score, taux d'erreurs corrigées/restantes, taux de
nouvelles erreurs introduites (sur-nettoyage).

Principe de la comparaison (version avec row_id) :
    - clean.csv              : vérité terrain (jamais modifié, pas de colonne row_id)
    - noisy_{level}.csv      : version donnée au LLM, avec colonne row_id ajoutée par error_injection.py
    - injected_errors_{level}.csv : log exact de row_index/column/original_value (row_index = row_id)
    - cleaned_output.csv     : résultat produit par le workflow généré, DOIT contenir row_id
                               (le system_prompt impose au LLM de la préserver)

L'alignement des lignes se fait par la colonne "row_id", ce qui reste fiable même si
le workflow généré a supprimé des lignes (déduplication, filtrage) ou les a réordonnées.

Pour chaque cellule qui était bruitée (listée dans injected_errors), sur une ligne encore
présente dans le résultat (row_id retrouvé) :
    - Vrai Positif (TP)  : le workflow a restauré la valeur originale (clean.csv)
    - Faux Négatif (FN)  : le workflow n'a pas corrigé (valeur toujours bruitée ou différente)

Pour chaque cellule qui N'ÉTAIT PAS bruitée, sur une ligne encore présente :
    - Faux Positif (FP)  : le workflow a modifié une valeur qui était pourtant correcte (sur-nettoyage)
    - Vrai Négatif (TN)  : le workflow n'a pas touché une valeur qui était déjà correcte

Les lignes dont le row_id a disparu du résultat nettoyé (supprimées par le workflow) sont
comptabilisées séparément dans 'rows_dropped' et ne participent pas au calcul cellule par cellule.

Utilisation basique :
    from app.services.evaluation.quality import evaluate_quality

    metrics = evaluate_quality(
        clean_path="benchmark/datasets/hotel_bookings/clean.csv",
        injected_errors_path="benchmark/datasets/hotel_bookings/injected_errors_low.csv",
        cleaned_output_path="results/cleaned_datasets/hotel_bookings/noisy_low__validated.csv",
    )
"""

import pandas as pd


def evaluate_quality(clean_path: str, injected_errors_path: str, cleaned_output_path: str) -> dict:
    """
    Calcule les métriques de qualité du nettoyage, en réalignant les lignes par row_id.

    Returns
    -------
    dict avec les clés :
        precision, recall, f1_score : float (0 à 1)
        true_positives, false_negatives, false_positives, true_negatives : int
        n_errors_injected : int
        n_errors_corrected, n_errors_remaining, n_new_errors_introduced : int
        rows_dropped : int (lignes dont le row_id a disparu du résultat)
        rows_compared : int (lignes effectivement comparées, row_id retrouvé des deux côtés)
        row_id_missing_in_output : bool (True si cleaned_output n'a pas de colonne row_id -> résultat invalide)
    """
    df_clean = pd.read_csv(clean_path)
    df_clean = df_clean.reset_index().rename(columns={"index": "row_id"})

    errors_log = pd.read_csv(injected_errors_path)
    df_cleaned = pd.read_csv(cleaned_output_path)

    if "row_id" not in df_cleaned.columns:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0,
            "true_positives": 0,
            "false_negatives": 0,
            "false_positives": 0,
            "true_negatives": 0,
            "n_errors_injected": len(errors_log),
            "n_errors_corrected": 0,
            "n_errors_remaining": len(errors_log),
            "n_new_errors_introduced": 0,
            "rows_dropped": len(df_clean),
            "rows_compared": 0,
            "row_id_missing_in_output": True,
            "warning": "La colonne 'row_id' est absente du résultat nettoyé : le script généré ne l'a pas "
                       "préservée comme demandé dans le system_prompt. Impossible de réaligner les lignes "
                       "de façon fiable — métriques non calculables.",
        }

    n_total_rows = len(df_clean)
    rows_dropped = n_total_rows - df_cleaned["row_id"].isin(df_clean["row_id"]).sum()

    # Fusion sur row_id : ne garde que les lignes présentes dans les deux
    merged = df_clean.merge(df_cleaned, on="row_id", suffixes=("_clean", "_cleaned"))
    n_rows_compared = len(merged)

    injected_cells = set(zip(errors_log["row_index"], errors_log["column"]))

    common_columns = [
        c for c in df_clean.columns
        if c != "row_id" and c in df_cleaned.columns
    ]

    true_positives = 0
    false_negatives = 0
    false_positives = 0
    true_negatives = 0
    natural_missing_filled = 0  # informatif : cellules naturellement NaN dans clean.csv, remplies par le workflow

    for _, row in merged.iterrows():
        row_id = row["row_id"]

        for col in common_columns:
            clean_value = row[f"{col}_clean"] if f"{col}_clean" in row else row[col]
            cleaned_value = row[f"{col}_cleaned"] if f"{col}_cleaned" in row else row[col]

            was_injected = (row_id, col) in injected_cells
            is_now_correct = _values_match(clean_value, cleaned_value)

            if was_injected:
                if is_now_correct:
                    true_positives += 1
                else:
                    false_negatives += 1
            else:
                if is_now_correct:
                    true_negatives += 1
                elif pd.isna(clean_value):
                    # La cellule était déjà manquante dans clean.csv (pas une erreur injectée) :
                    # il n'existe pas de "bonne valeur" de référence à retrouver ici.
                    # Le workflow a choisi de la remplir (comportement raisonnable), on ne
                    # pénalise pas ce choix comme un faux positif.
                    natural_missing_filled += 1
                else:
                    false_positives += 1

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1_score, 4),
        "true_positives": true_positives,
        "false_negatives": false_negatives,
        "false_positives": false_positives,
        "true_negatives": true_negatives,
        "natural_missing_filled": natural_missing_filled,
        "n_errors_injected": len(injected_cells),
        "n_errors_corrected": true_positives,
        "n_errors_remaining": false_negatives,
        "n_new_errors_introduced": false_positives,
        "rows_dropped": int(rows_dropped),
        "rows_compared": n_rows_compared,
        "row_id_missing_in_output": False,
    }


def _values_match(a, b) -> bool:
    """Compare deux valeurs en tolérant les différences de type (ex: '0' vs 0.0) et les NaN."""
    if pd.isna(a) and pd.isna(b):
        return True
    if pd.isna(a) or pd.isna(b):
        return False

    if str(a).strip() == str(b).strip():
        return True

    try:
        return float(a) == float(b)
    except (ValueError, TypeError):
        return False


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Calcule les métriques de qualité du nettoyage.")
    parser.add_argument("--clean", required=True, help="Chemin vers clean.csv")
    parser.add_argument("--injected_errors", required=True, help="Chemin vers injected_errors_{level}.csv")
    parser.add_argument("--cleaned_output", required=True, help="Chemin vers le dataset nettoyé par le workflow")
    args = parser.parse_args()

    metrics = evaluate_quality(args.clean, args.injected_errors, args.cleaned_output)

    print(json.dumps(metrics, indent=2, ensure_ascii=False))
