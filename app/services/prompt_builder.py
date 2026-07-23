"""
prompt_builder.py
------------------
Construit dynamiquement le prompt final à envoyer au LLM, en combinant :
    - un template générique (prompts/*.txt)
    - les vraies données du dataset traité (extrait, schéma, profil)

Ce module est générique : il fonctionne pour n'importe quel dataset
(hotel_bookings aujourd'hui, titanic/flights/hospitals plus tard),
à condition de lui fournir le bon chemin de fichier CSV.

Utilisation basique :
    from app.services.prompt_builder import build_prompt

    prompt = build_prompt(
        approach="simple",
        dataset_csv_path="benchmark/datasets/hotel_bookings/noisy_low.csv",
        input_path="benchmark/datasets/hotel_bookings/noisy_low.csv",
        output_path="results/cleaned_datasets/hotel_bookings/noisy_low__prompt_simple.csv",
    )
"""

import pandas as pd
from pathlib import Path

from app.services.error_injection import profile_dataset


PROMPTS_DIR = Path("prompts")

# Mapping entre le nom court de l'approche et le fichier template correspondant
APPROACH_TEMPLATES = {
    "simple": "prompt_simple.txt",
    "schema": "prompt_schema.txt",
    "profile": "prompt_profile.txt",
    "constrained": "prompt_constrained.txt",
    "fewshot": "prompt_fewshot.txt",
    "validation_loop": "prompt_validation_loop.txt",
}


def load_template(approach: str) -> str:
    """Charge le contenu brut d'un template de prompt selon l'approche demandée."""
    if approach not in APPROACH_TEMPLATES:
        raise ValueError(
            f"Approche inconnue : '{approach}'. Choix possibles : {list(APPROACH_TEMPLATES.keys())}"
        )
    template_path = PROMPTS_DIR / APPROACH_TEMPLATES[approach]
    if not template_path.exists():
        raise FileNotFoundError(f"Template introuvable : {template_path}")
    return template_path.read_text(encoding="utf-8")


def load_system_prompt() -> str:
    """Charge le prompt système commun, à préfixer à tous les appels LLM."""
    system_path = PROMPTS_DIR / "system_prompt.txt"
    return system_path.read_text(encoding="utf-8")


def build_dataset_sample(df: pd.DataFrame, n_rows: int = 5) -> str:
    """Construit un extrait lisible des premières lignes du dataset, au format texte."""
    return df.head(n_rows).to_string(index=False)


def build_schema_description(df: pd.DataFrame) -> str:
    """
    Construit une description texte du schéma : nom de colonne, type détecté,
    et 3 exemples de valeurs non nulles.
    """
    lines = []
    for col in df.columns:
        dtype = "numeric" if pd.api.types.is_numeric_dtype(df[col]) else "categorical/text"
        examples = df[col].dropna().unique()[:3]
        examples_str = ", ".join(str(v) for v in examples)
        lines.append(f"- {col} ({dtype}) — exemples : {examples_str}")
    return "\n".join(lines)


def build_profile_description(df: pd.DataFrame) -> str:
    """
    Construit une description texte du profil statistique complet du dataset,
    réutilisant la fonction profile_dataset() de error_injection.py.
    """
    profile_df = profile_dataset(df)
    lines = []
    for _, row in profile_df.iterrows():
        if row["dtype_detected"] == "numeric":
            lines.append(
                f"- {row['column']} (numeric) — "
                f"manquants: {row['pct_missing']}%, "
                f"uniques: {row['n_unique']}, "
                f"min: {row['min']}, max: {row['max']}, "
                f"moyenne: {row['mean']}, écart-type: {row['std']}"
            )
        else:
            top_values = row.get("top_values", {})
            top_str = ", ".join(f"{k} ({v})" for k, v in list(top_values.items())[:3]) if isinstance(top_values, dict) else ""
            lines.append(
                f"- {row['column']} (categorical/text) — "
                f"manquants: {row['pct_missing']}%, "
                f"uniques: {row['n_unique']}, "
                f"valeurs fréquentes: {top_str}"
            )
    return "\n".join(lines)


def build_prompt(
    approach: str,
    dataset_csv_path: str,
    input_path: str,
    output_path: str,
    n_sample_rows: int = 5,
    previous_script: str = "",
    execution_feedback: str = "",
    include_system_prompt: bool = True,
) -> str:
    """
    Construit le prompt final complet pour une approche donnée.

    Parameters
    ----------
    approach : str
        Une des clés de APPROACH_TEMPLATES : 'simple', 'schema', 'profile',
        'constrained', 'fewshot', 'validation_loop'.
    dataset_csv_path : str
        Chemin du CSV à analyser pour construire l'extrait/schéma/profil
        (généralement le fichier noisy_{level}.csv à nettoyer).
    input_path : str
        Chemin qui sera écrit dans le prompt comme INPUT_PATH (utilisé par le
        script généré par le LLM pour lire le fichier).
    output_path : str
        Chemin qui sera écrit dans le prompt comme OUTPUT_PATH.
    n_sample_rows : int
        Nombre de lignes à inclure dans l'extrait du dataset.
    previous_script : str
        Utilisé uniquement pour l'approche 'validation_loop' : le script précédent.
    execution_feedback : str
        Utilisé uniquement pour l'approche 'validation_loop' : l'erreur/le problème rencontré.
    include_system_prompt : bool
        Si True, préfixe le prompt système commun devant le prompt spécifique à l'approche.

    Returns
    -------
    str : le prompt final, prêt à être envoyé au LLM.
    """
    df = pd.read_csv(dataset_csv_path)
    template = load_template(approach)

    replacements = {
        "{dataset_sample}": build_dataset_sample(df, n_sample_rows),
        "{input_path}": input_path,
        "{output_path}": output_path,
    }

    if approach in ("schema", "constrained", "fewshot", "validation_loop"):
        replacements["{schema_description}"] = build_schema_description(df)

    if approach == "profile":
        replacements["{profile_description}"] = build_profile_description(df)

    if approach == "validation_loop":
        replacements["{previous_script}"] = previous_script
        replacements["{execution_feedback}"] = execution_feedback

    final_prompt = template
    for placeholder, value in replacements.items():
        final_prompt = final_prompt.replace(placeholder, str(value))

    if include_system_prompt:
        final_prompt = load_system_prompt() + "\n\n" + final_prompt

    return final_prompt


if __name__ == "__main__":
    # Test rapide en ligne de commande
    import argparse

    parser = argparse.ArgumentParser(description="Construit et affiche un prompt pour test.")
    parser.add_argument("--approach", required=True, choices=list(APPROACH_TEMPLATES.keys()))
    parser.add_argument("--dataset_csv", required=True)
    parser.add_argument("--input_path", required=True)
    parser.add_argument("--output_path", required=True)
    args = parser.parse_args()

    prompt = build_prompt(
        approach=args.approach,
        dataset_csv_path=args.dataset_csv,
        input_path=args.input_path,
        output_path=args.output_path,
    )
    print(prompt)
