"""
run_with_validation_loop.py
-----------------------------
Approche 6 de la taxonomie : génère un script, l'exécute, et si ça échoue,
renvoie l'erreur au LLM pour qu'il corrige, jusqu'à réussite ou nombre
maximal de tentatives atteint.

À lancer depuis la racine du projet :
    python run_with_validation_loop.py
"""

from pathlib import Path

from app.services.prompt_builder import build_prompt, load_template, load_system_prompt
from app.services.llm import generate_workflow
from app.services.safe_executor import execute_workflow


MAX_ATTEMPTS = 3

DATASET_CSV = "benchmark/datasets/hotel_bookings/noisy_low.csv"
INPUT_PATH = "benchmark/datasets/hotel_bookings/noisy_low.csv"
OUTPUT_PATH = "results/cleaned_datasets/hotel_bookings/noisy_low__validated.csv"
GENERATED_SCRIPT_PATH = Path("workflows/generated/hotel_bookings__validated.py")
FAILED_SCRIPT_DIR = Path("workflows/failed")

# S'assurer que les dossiers existent
GENERATED_SCRIPT_PATH.parent.mkdir(parents=True, exist_ok=True)
FAILED_SCRIPT_DIR.mkdir(parents=True, exist_ok=True)
Path("results/cleaned_datasets/hotel_bookings").mkdir(parents=True, exist_ok=True)


def build_validation_prompt(previous_script: str, execution_feedback: str) -> str:
    """Construit le prompt de correction (approche validation_loop) sans repasser par un CSV."""
    import pandas as pd

    df = pd.read_csv(DATASET_CSV)
    from app.services.prompt_builder import build_schema_description

    template = load_template("validation_loop")
    replacements = {
        "{schema_description}": build_schema_description(df),
        "{input_path}": INPUT_PATH,
        "{output_path}": OUTPUT_PATH,
        "{previous_script}": previous_script,
        "{execution_feedback}": execution_feedback,
    }
    final_prompt = template
    for placeholder, value in replacements.items():
        final_prompt = final_prompt.replace(placeholder, str(value))

    return load_system_prompt() + "\n\n" + final_prompt


def main():
    # --- Tentative 1 : approche "simple" ---
    print("=== Tentative 1 : génération initiale (approche simple) ===")
    prompt = build_prompt(
        approach="simple",
        dataset_csv_path=DATASET_CSV,
        input_path=INPUT_PATH,
        output_path=OUTPUT_PATH,
    )
    script = generate_workflow(prompt)
    GENERATED_SCRIPT_PATH.write_text(script, encoding="utf-8")

    for attempt in range(1, MAX_ATTEMPTS + 1):
        print(f"\n--- Exécution, tentative {attempt}/{MAX_ATTEMPTS} ---")
        result = execute_workflow(str(GENERATED_SCRIPT_PATH), timeout=120)

        if result["success"]:
            print(f"✅ Succès à la tentative {attempt} (durée : {result['duration_seconds']}s)")
            print(result["stdout"])
            print(f"\nScript final sauvegardé dans {GENERATED_SCRIPT_PATH}")
            print(f"Dataset nettoyé disponible dans {OUTPUT_PATH}")
            return

        print(f"❌ Échec : {result['error_message']}")

        # Sauvegarde du script en échec pour analyse (Phase 6)
        failed_path = FAILED_SCRIPT_DIR / f"hotel_bookings__attempt{attempt}.py"
        failed_path.write_text(script, encoding="utf-8")

        if attempt == MAX_ATTEMPTS:
            print(f"\n⚠️ Échec après {MAX_ATTEMPTS} tentatives. Abandon.")
            print(f"Scripts en échec sauvegardés dans {FAILED_SCRIPT_DIR}/")
            return

        # --- Construction du prompt de correction avec l'erreur réelle ---
        print("Envoi de l'erreur au LLM pour correction...")
        correction_prompt = build_validation_prompt(
            previous_script=script,
            execution_feedback=result["error_message"] + "\n\n" + result["stderr"],
        )
        script = generate_workflow(correction_prompt)
        GENERATED_SCRIPT_PATH.write_text(script, encoding="utf-8")


if __name__ == "__main__":
    main()
