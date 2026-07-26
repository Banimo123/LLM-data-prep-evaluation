"""
test_end_to_end.py
-------------------
Script de test temporaire : combine prompt_builder.py et llm.py pour
générer le tout premier script de nettoyage via Mistral (Ollama).

À lancer depuis la racine du projet :
    python test_end_to_end.py
"""

from pathlib import Path

from app.services.prompt_builder import build_prompt
from app.services.llm import generate_workflow

# Assure-toi que le dossier de sortie existe
Path("workflows/generated").mkdir(parents=True, exist_ok=True)
Path("results/cleaned_datasets/hotel_bookings").mkdir(parents=True, exist_ok=True)

prompt = build_prompt(
    approach="simple",
    dataset_csv_path="benchmark/datasets/hotel_bookings/noisy_low.csv",
    input_path="benchmark/datasets/hotel_bookings/noisy_low.csv",
    output_path="results/cleaned_datasets/hotel_bookings/noisy_low__simple.csv",
)

print("Génération en cours (peut prendre 30s à 2 min avec Ollama)...")
script = generate_workflow(prompt)

print("=" * 60)
print("SCRIPT GÉNÉRÉ :")
print("=" * 60)
print(script)

output_path = Path("workflows/generated/hotel_bookings__simple.py")
output_path.write_text(script, encoding="utf-8")
print(f"\nScript sauvegardé dans {output_path}")
