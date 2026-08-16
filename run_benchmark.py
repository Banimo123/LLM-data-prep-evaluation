"""
run_benchmark.py
------------------
Lance le pipeline complet (génération + exécution + boucle de correction +
évaluation qualité via metrics.py) sur les 3 niveaux de bruit (low/medium/high).

Répond aux deux tâches assignées par l'encadrant :
    - Benchmarking (F1 score)
    - Ajout du taux de bruit dans les résultats

Prérequis : avoir relancé error_injection.py (version corrigée) pour régénérer
les datasets avec la colonne "row_id" cohérente partout, y compris clean_with_id.csv.

À lancer depuis la racine du projet :
    python run_benchmark.py
"""

import json
from pathlib import Path

import pandas as pd

from app.services.prompt_builder import build_prompt, load_template, load_system_prompt, build_schema_description
from app.services.llm import generate_workflow
from app.services.safe_executor import execute_workflow
from app.services.evaluation.metrics import evaluate_workflow
from app.services.error_injection import get_rates_for_level


DATASET_NAME = "flights"
DATASET_DIR = Path(f"datasets/{DATASET_NAME}")
CLEANED_DIR = Path(f"results/cleaned_datasets/{DATASET_NAME}")
GENERATED_DIR = Path("workflows/generated")
FAILED_DIR = Path("workflows/failed")
METRICS_DIR = Path("results/metrics")

MAX_ATTEMPTS = 3
APPROACH = "profile"

# Detection automatique des niveaux de bruit disponibles pour ce dataset : certains
# datasets (donnees reelles, un seul fichier "dirty") n'ont qu'un seul niveau ("low"),
# d'autres (bruit synthetique) en ont 3. Une liste fixe ["low","medium","high"] plante
# des qu'un niveau n'existe pas -> on ne garde que les niveaux dont noisy_{level}.csv
# existe reellement sur le disque.
_ALL_POSSIBLE_LEVELS = ["low", "medium", "high"]
NOISE_LEVELS = [lvl for lvl in _ALL_POSSIBLE_LEVELS if (DATASET_DIR / f"noisy_{lvl}.csv").exists()]
if not NOISE_LEVELS:
    raise FileNotFoundError(
        f"Aucun fichier noisy_{{level}}.csv trouve dans {DATASET_DIR} -- "
        f"verifiez que le dataset '{DATASET_NAME}' a bien ete prepare (injection ou donnees reelles)."
    )
print(f"Niveaux de bruit detectes pour '{DATASET_NAME}' : {NOISE_LEVELS}\n")

for d in [CLEANED_DIR, GENERATED_DIR, FAILED_DIR, METRICS_DIR]:
    d.mkdir(parents=True, exist_ok=True)


def build_validation_prompt(dataset_csv_path, input_path, output_path, previous_script, execution_feedback):
    df = pd.read_csv(dataset_csv_path)
    template = load_template("validation_loop")
    replacements = {
        "{schema_description}": build_schema_description(df),
        "{input_path}": input_path,
        "{output_path}": output_path,
        "{previous_script}": previous_script,
        "{execution_feedback}": execution_feedback,
    }
    final_prompt = template
    for placeholder, value in replacements.items():
        final_prompt = final_prompt.replace(placeholder, str(value))
    return load_system_prompt() + "\n\n" + final_prompt


def run_pipeline_for_level(level: str) -> dict:
    dataset_csv = str(DATASET_DIR / f"noisy_{level}.csv")
    injected_errors_csv = str(DATASET_DIR / f"injected_errors_{level}.csv")
    clean_with_id_csv = str(DATASET_DIR / "clean_with_id.csv")
    output_path = str(CLEANED_DIR / f"noisy_{level}__{APPROACH}.csv")
    generated_script_path = GENERATED_DIR / f"{DATASET_NAME}_{level}__{APPROACH}.py"

    rates = get_rates_for_level(level)

    print(f"\n{'='*70}\nNIVEAU DE BRUIT : {level.upper()}  (rates={rates})\n{'='*70}")

    prompt = build_prompt(
        approach=APPROACH,
        dataset_csv_path=dataset_csv,
        input_path=dataset_csv,
        output_path=output_path,
    )
    script = generate_workflow(prompt)
    generated_script_path.write_text(script, encoding="utf-8")

    execution_success = False
    final_result = None

    for attempt in range(1, MAX_ATTEMPTS + 1):
        print(f"--- Exécution, tentative {attempt}/{MAX_ATTEMPTS} ---")
        result = execute_workflow(str(generated_script_path), timeout=120)
        final_result = result

        if result["success"]:
            print(f"✅ Succès (durée : {result['duration_seconds']}s)")
            execution_success = True
            break

        print(f"❌ Échec : {result['error_message']}")
        failed_path = FAILED_DIR / f"{DATASET_NAME}_{level}__{APPROACH}_attempt{attempt}.py"
        failed_path.write_text(script, encoding="utf-8")

        if attempt == MAX_ATTEMPTS:
            print(f"⚠️ Échec après {MAX_ATTEMPTS} tentatives pour le niveau {level}.")
            break

        correction_prompt = build_validation_prompt(
            dataset_csv, dataset_csv, output_path,
            previous_script=script,
            execution_feedback=result["error_message"] + "\n\n" + result["stderr"],
        )
        script = generate_workflow(correction_prompt)
        generated_script_path.write_text(script, encoding="utf-8")

    if not execution_success:
        return {
            "noise_level": level,
            "noise_rates": rates,
            "execution_success": False,
            "f1": None, "precision": None, "recall": None,
            "execution_error": final_result["error_message"] if final_result else "Inconnu",
        }

    report = evaluate_workflow(
        clean_csv=clean_with_id_csv,
        noisy_csv=dataset_csv,
        injected_errors_csv=injected_errors_csv,
        cleaned_csv=output_path,
    )
    result_row = {
        "noise_level": level,
        "noise_rates": rates,
        "execution_success": True,
        "execution_error": None,
        **report["global"],
    }
    result_row["per_error_family"] = report["per_error_family"]
    return result_row


def main():
    all_results = []

    for level in NOISE_LEVELS:
        result = run_pipeline_for_level(level)
        all_results.append(result)
        if result["execution_success"]:
            print(f"F1 pour {level} : {result['f1']}")

    results_df = pd.DataFrame(all_results)
    csv_path = METRICS_DIR / f"benchmark_{APPROACH}_all_levels.csv"
    results_df.drop(columns=["per_error_family"], errors="ignore").to_csv(csv_path, index=False)

    json_path = METRICS_DIR / f"benchmark_{APPROACH}_all_levels.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)

    print(f"\n{'='*70}\nRÉSUMÉ DU BENCHMARK (approche: {APPROACH})\n{'='*70}")
    display_cols = ["noise_level", "execution_success", "precision", "recall", "f1"]
    print(results_df[[c for c in display_cols if c in results_df.columns]].to_string(index=False))
    print(f"\nRésultats complets sauvegardés dans :\n  - {csv_path}\n  - {json_path}")


if __name__ == "__main__":
    main()
