"""
evaluate_all_generated.py
--------------------------
À lancer depuis la racine du dépôt (là où se trouve benchmark/, results/, app/).

Parcourt tous les fichiers results/cleaned_datasets/hotel_bookings/noisy_{level}__{approche}.csv,
calcule le F1 (via fast_metrics.py, corrigé) pour chacun, affiche un tableau et sauvegarde
un JSON complet à renvoyer pour interprétation.

Usage :
    python evaluate_all_generated.py
"""

import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fast_metrics import evaluate_fast

CLEAN_WITH_ID = "benchmark/datasets/hotel_bookings/clean_with_id.csv"
NOISY_TEMPLATE = "benchmark/datasets/hotel_bookings/noisy_{level}.csv"
ERRORS_TEMPLATE = "benchmark/datasets/hotel_bookings/injected_errors_{level}.csv"
CLEANED_DIR = "results/cleaned_datasets/hotel_bookings"
OUTPUT_JSON = "results/metrics_tables/f1_comparaison_complet.json"


def main():
    if not os.path.exists(CLEAN_WITH_ID):
        print(f"ERREUR : {CLEAN_WITH_ID} introuvable. Vérifiez que vous lancez ce script "
              f"depuis la racine du dépôt, et que clean_with_id.csv existe bien.")
        sys.exit(1)

    results = []
    for fname in sorted(os.listdir(CLEANED_DIR)):
        m = re.match(r"noisy_(low|medium|high)__(.+)\.csv$", fname)
        if not m:
            continue
        level, approach = m.groups()

        noisy_csv = NOISY_TEMPLATE.format(level=level)
        errors_csv = ERRORS_TEMPLATE.format(level=level)
        cleaned_csv = os.path.join(CLEANED_DIR, fname)

        try:
            report = evaluate_fast(CLEAN_WITH_ID, noisy_csv, errors_csv, cleaned_csv)
            g = report["global"]
            row = {
                "approche": approach,
                "niveau_de_bruit": level,
                "fichier": fname,
                "precision": g["precision"],
                "recall": g["recall"],
                "f1": g["f1"],
                "tp": g["tp"], "fp": g["fp"], "fn": g["fn"],
                "n_rows_lost_by_workflow": g["n_rows_lost_by_workflow"],
                "columns_dropped_by_workflow": g.get("columns_dropped_by_workflow", []),
                "per_error_family": report["per_error_family"],
                "erreur": None,
            }
        except Exception as e:
            row = {
                "approche": approach, "niveau_de_bruit": level, "fichier": fname,
                "erreur": str(e),
            }

        results.append(row)

    # --- Affichage tableau ---
    print(f"{'Approche':20s} {'Niveau':8s} {'Precision':>10s} {'Recall':>8s} {'F1':>8s} {'LignesPerdues':>14s}")
    print("-" * 72)
    for r in sorted(results, key=lambda r: (r["approche"], r["niveau_de_bruit"])):
        if r.get("erreur"):
            print(f"{r['approche']:20s} {r['niveau_de_bruit']:8s} ERREUR: {r['erreur']}")
        else:
            print(f"{r['approche']:20s} {r['niveau_de_bruit']:8s} "
                  f"{r['precision']:>10.4f} {r['recall']:>8.4f} {r['f1']:>8.4f} "
                  f"{r['n_rows_lost_by_workflow']:>14}")

    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nRésultats complets sauvegardés dans : {OUTPUT_JSON}")
    print("-> Envoyez ce fichier JSON pour interprétation.")


if __name__ == "__main__":
    main()