"""
app/api/routes.py
-------------------
Endpoints exposant les fonctionnalités du projet via l'API FastAPI :
    - GET  /health                    : vérifie que l'API tourne
    - GET  /results/benchmark         : résultats du dernier benchmark LLM (3 niveaux)
    - GET  /results/comparison        : comparaison complète baseline vs approches LLM
    - GET  /noise-rates/{level}       : taux de bruit exact utilisé pour un niveau donné
    - POST /inject-errors             : relance l'injection d'erreurs sur hotel_bookings
"""

import json
from pathlib import Path

import pandas as pd
from fastapi import APIRouter, HTTPException

from app.services.error_injection import get_rates_for_level, inject_all

router = APIRouter()

RESULTS_DIR = Path("results")
DATASET_DIR = Path("benchmark/datasets/hotel_bookings")


@router.get("/health")
def health_check():
    return {"status": "ok"}


@router.get("/results/benchmark")
def get_benchmark_results():
    """Retourne les résultats du dernier benchmark LLM (approche profile, 3 niveaux de bruit)."""
    path = RESULTS_DIR / "metrics" / "benchmark_profile_all_levels.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Aucun résultat de benchmark trouvé. Lancez d'abord run_benchmark.py.")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return {"approach": "profile", "results": data}


@router.get("/results/comparison")
def get_comparison_results():
    """Retourne la comparaison complète entre la baseline manuelle et les approches LLM testées."""
    path = RESULTS_DIR / "metrics_tables" / "f1_comparaison_complet.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Fichier de comparaison introuvable.")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return {"comparison": data}


@router.get("/noise-rates/{level}")
def get_noise_rates(level: str):
    """Retourne le taux de bruit exact (par famille d'erreur) utilisé pour un niveau donné."""
    if level not in ("low", "medium", "high"):
        raise HTTPException(status_code=400, detail="Niveau invalide. Utilisez 'low', 'medium' ou 'high'.")
    rates = get_rates_for_level(level)
    return {"noise_level": level, "rates": rates}


@router.post("/inject-errors/{level}")
def trigger_error_injection(level: str):
    """
    Relance l'injection d'erreurs sur le dataset hotel_bookings, pour UN niveau donné
    (évite de tout relancer d'un coup, plus rapide et sûr pour une démo live).
    Régénère noisy_{level}.csv et injected_errors_{level}.csv.
    """
    if level not in ("low", "medium", "high"):
        raise HTTPException(status_code=400, detail="Niveau invalide. Utilisez 'low', 'medium' ou 'high'.")

    clean_path = DATASET_DIR / "clean.csv"
    if not clean_path.exists():
        raise HTTPException(status_code=404, detail=f"Fichier introuvable : {clean_path}")

    try:
        df_clean = pd.read_csv(clean_path)
        if "row_id" not in df_clean.columns:
            df_clean.insert(0, "row_id", range(len(df_clean)))

        rates = get_rates_for_level(level)
        df_final, full_log = inject_all(df_clean, random_state=42, rates=rates)

        out_csv = DATASET_DIR / f"noisy_{level}.csv"
        errors_csv = DATASET_DIR / f"injected_errors_{level}.csv"
        df_final.to_csv(out_csv, index=False)
        full_log.to_csv(errors_csv, index=False)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Échec de l'injection d'erreurs : {e}")

    return {
        "status": "success",
        "noise_level": level,
        "noise_rates": rates,
        "n_errors_injected": len(full_log),
        "output_files": [str(out_csv), str(errors_csv)],
    }