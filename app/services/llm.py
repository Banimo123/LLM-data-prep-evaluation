"""
llm.py
------
Client LLM unifié : bascule entre Ollama (local) et Mistral API (cloud)
selon la variable d'environnement LLM_PROVIDER (.env).

Utilisation basique :
    from app.services.llm import generate_workflow

    script = generate_workflow(prompt)
"""

import os
import re
import time
import requests
from dotenv import load_dotenv

load_dotenv()

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", "")
MISTRAL_MODEL = os.getenv("MISTRAL_MODEL", "mistral-large-latest")


def _call_ollama(prompt: str, model: str = None, temperature: float = 0.2) -> str:
    """Appelle Ollama en local via son API HTTP."""
    model = model or OLLAMA_MODEL
    response = requests.post(
        f"{OLLAMA_HOST}/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature},
        },
        timeout=300,
    )
    response.raise_for_status()
    return response.json()["response"]


def _call_mistral_api(prompt: str, model: str = None, temperature: float = 0.2, max_retries: int = 3) -> str:
    """
    Appelle l'API Mistral (tier gratuit "Experiment").
    Gère le rate limit (erreur 429) avec un retry + backoff exponentiel.
    """
    if not MISTRAL_API_KEY:
        raise ValueError("MISTRAL_API_KEY n'est pas défini dans le fichier .env")

    model = model or MISTRAL_MODEL
    url = "https://api.mistral.ai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {MISTRAL_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
    }

    for attempt in range(max_retries):
        response = requests.post(url, headers=headers, json=payload, timeout=120)

        if response.status_code == 429:
            wait_time = 2 ** attempt  # backoff exponentiel : 1s, 2s, 4s...
            print(f"Rate limit atteint (429). Nouvelle tentative dans {wait_time}s...")
            time.sleep(wait_time)
            continue

        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

    raise RuntimeError("Échec après plusieurs tentatives : rate limit Mistral API dépassé.")


def call_llm(prompt: str, provider: str = None, model: str = None, temperature: float = 0.2) -> str:
    """
    Point d'entrée unique : appelle le LLM configuré (Ollama par défaut, ou Mistral API).

    Parameters
    ----------
    prompt : str
        Le prompt complet à envoyer (déjà construit par prompt_builder.py).
    provider : str or None
        'ollama' ou 'mistral_api'. Si None, utilise LLM_PROVIDER du .env.
    model : str or None
        Nom du modèle à utiliser. Si None, utilise la valeur par défaut du provider.
    temperature : float
        Température de génération (0 = déterministe, 1 = créatif).
        Une température basse est recommandée ici pour la reproductibilité (Phase 5).

    Returns
    -------
    str : la réponse brute du LLM (texte, incluant probablement un bloc ```python ... ```).
    """
    provider = provider or LLM_PROVIDER

    if provider == "ollama":
        return _call_ollama(prompt, model=model, temperature=temperature)
    elif provider == "mistral_api":
        return _call_mistral_api(prompt, model=model, temperature=temperature)
    else:
        raise ValueError(f"Provider inconnu : '{provider}'. Utilise 'ollama' ou 'mistral_api'.")


def extract_python_code(llm_response: str) -> str:
    """
    Extrait le code Python d'une réponse LLM, en supprimant les balises ```python ... ```
    si présentes. Si aucune balise n'est trouvée, retourne la réponse telle quelle
    (le LLM a peut-être répondu sans balises, malgré la consigne du system_prompt).
    """
    match = re.search(r"```python\s*(.*?)```", llm_response, re.DOTALL)
    if match:
        return match.group(1).strip()

    match = re.search(r"```\s*(.*?)```", llm_response, re.DOTALL)
    if match:
        return match.group(1).strip()

    return llm_response.strip()


def generate_workflow(prompt: str, provider: str = None, model: str = None, temperature: float = 0.2) -> str:
    """
    Fonction de haut niveau : envoie le prompt au LLM et retourne directement
    le code Python extrait (prêt à être sauvegardé dans workflows/generated/).
    """
    raw_response = call_llm(prompt, provider=provider, model=model, temperature=temperature)
    return extract_python_code(raw_response)


if __name__ == "__main__":
    # Test rapide en ligne de commande
    import argparse

    parser = argparse.ArgumentParser(description="Teste un appel LLM avec un prompt simple.")
    parser.add_argument("--prompt", default="Réponds en une phrase : bonjour, qui es-tu ?")
    parser.add_argument("--provider", default=None, choices=["ollama", "mistral_api", None])
    args = parser.parse_args()

    print(f"Provider utilisé : {args.provider or LLM_PROVIDER}")
    result = call_llm(args.prompt, provider=args.provider)
    print("Réponse brute :")
    print(result)
