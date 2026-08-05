"""
evaluate_all_generated.py
--------------------------
A lancer depuis la racine du depot (la ou se trouve datasets/, results/, app/).

Parcourt AUTOMATIQUEMENT tous les datasets presents dans results/cleaned_datasets/<dataset>/,
calcule le F1 (via fast_metrics.py) pour chaque fichier noisy_{level}__{approche}.csv trouve,
affiche un tableau et sauvegarde un JSON complet a renvoyer pour interpretation.

Usage :
    python evaluate_all_generated.py
    python evaluate_all_generated.py hotel_bookings          # un seul dataset
    python evaluate_all_generated.py hotel_bookings titanic  # plusieurs datasets precis
"""

import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fast_metrics import evaluate_fast

DATASETS_DIR = Path("datasets")
CLEANED_ROOT = Path("results/cleaned_datasets")
OUTPUT_JSON = "results/metrics/f1_comparaison_complet.json"


def discover_datasets():
    """Liste les datasets ayant a la fois des donnees sources ET des resultats nettoyes."""
    if not CLEANED_ROOT.exists():
        return []
    found = []
    for d in sorted(CLEANED_ROOT.iterdir()):
        if d.is_dir() and (DATASETS_DIR / d.name / "clean_with_id.csv").exists():
            found.append(d.name)
    return found


def evaluate_dataset(dataset_name):
    clean_with_id = str(DATASETS_DIR / dataset_name / "clean_with_id.csv")
    noisy_template = str(DATASETS_DIR / dataset_name / "noisy_{level}.csv")
    errors_template = str(DATASETS_DIR / dataset_name / "injected_errors_{level}.csv")
    cleaned_dir = CLEANED_ROOT / dataset_name

    results = []
    for fname in sorted(os.listdir(cleaned_dir)):
        m = re.match(r"noisy_(low|medium|high)__(.+)\.csv$", fname)
        if not m:
            continue
        level, approach = m.groups()

        noisy_csv = noisy_template.format(level=level)
        errors_csv = errors_template.format(level=level)
        cleaned_csv = str(cleaned_dir / fname)

        try:
            report = evaluate_fast(clean_with_id, noisy_csv, errors_csv, cleaned_csv)
            g = report["global"]
            row = {
                "dataset": dataset_name,
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
                "dataset": dataset_name, "approche": approach,
                "niveau_de_bruit": level, "fichier": fname,
                "erreur": str(e),
            }
        results.append(row)
    return results


def main():
    requested = sys.argv[1:]
    dataset_names = requested if requested else discover_datasets()

    if not dataset_names:
        print("ERREUR : aucun dataset trouve. Verifiez que vous lancez ce script depuis la "
              "racine du depot, et que results/cleaned_datasets/<dataset>/ contient des fichiers.")
        sys.exit(1)

    print(f"Datasets evalues : {', '.join(dataset_names)}\n")

    all_results = []
    for dataset_name in dataset_names:
        clean_with_id = DATASETS_DIR / dataset_name / "clean_with_id.csv"
        if not clean_with_id.exists():
            print(f"[SKIP] {dataset_name} : {clean_with_id} introuvable")
            continue
        all_results.extend(evaluate_dataset(dataset_name))

    # --- Affichage tableau ---
    print(f"{'Dataset':16s} {'Approche':18s} {'Niveau':8s} {'Precision':>10s} {'Recall':>8s} {'F1':>8s} {'LignesPerdues':>14s}")
    print("-" * 90)
    for r in sorted(all_results, key=lambda r: (r["dataset"], r["approche"], r["niveau_de_bruit"])):
        if r.get("erreur"):
            print(f"{r['dataset']:16s} {r['approche']:18s} {r['niveau_de_bruit']:8s} ERREUR: {r['erreur']}")
        else:
            print(f"{r['dataset']:16s} {r['approche']:18s} {r['niveau_de_bruit']:8s} "
                  f"{r['precision']:>10.4f} {r['recall']:>8.4f} {r['f1']:>8.4f} "
                  f"{r['n_rows_lost_by_workflow']:>14}")

    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nResultats complets sauvegardes dans : {OUTPUT_JSON}")
    print("-> Envoyez ce fichier JSON pour interpretation.")


if __name__ == "__main__":
    main()
