"""
safe_executor.py
------------------
Exécute un script Python généré par le LLM dans un environnement contrôlé,
sans Docker : isolation via sous-processus, timeout strict, capture des
erreurs (Phase 4 du cahier des charges).

Utilisation basique :
    from app.services.safe_executor import execute_workflow

    result = execute_workflow("workflows/generated/hotel_bookings__simple.py")
    print(result["success"], result["stdout"], result["stderr"])
"""

import subprocess
import sys
import time
import json
from pathlib import Path


def execute_workflow(script_path: str, timeout: int = 120, working_dir: str = None) -> dict:
    """
    Exécute un script Python généré, de façon isolée et chronométrée.

    Parameters
    ----------
    script_path : str
        Chemin vers le script .py à exécuter (ex: workflows/generated/hotel_bookings__simple.py).
    timeout : int
        Temps maximal d'exécution en secondes avant interruption forcée.
    working_dir : str or None
        Dossier de travail pour l'exécution. Par défaut : racine du projet (répertoire courant),
        nécessaire pour que les chemins relatifs INPUT_PATH/OUTPUT_PATH du script fonctionnent.

    Returns
    -------
    dict avec les clés :
        success : bool
        stdout : str
        stderr : str
        returncode : int or None
        duration_seconds : float
        error_message : str or None (résumé lisible en cas d'échec)
    """
    script_path = Path(script_path)
    if not script_path.exists():
        return {
            "success": False,
            "stdout": "",
            "stderr": "",
            "returncode": None,
            "duration_seconds": 0.0,
            "error_message": f"Fichier introuvable : {script_path}",
        }

    working_dir = working_dir or str(Path.cwd())
    start_time = time.time()

    try:
        result = subprocess.run(
            [sys.executable, str(script_path.resolve())],
            cwd=working_dir,
            timeout=timeout,
            capture_output=True,
            text=True,
        )
        duration = round(time.time() - start_time, 2)

        return {
            "success": result.returncode == 0,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode,
            "duration_seconds": duration,
            "error_message": None if result.returncode == 0 else result.stderr.strip().splitlines()[-1] if result.stderr else "Erreur inconnue",
        }

    except subprocess.TimeoutExpired:
        duration = round(time.time() - start_time, 2)
        return {
            "success": False,
            "stdout": "",
            "stderr": "",
            "returncode": None,
            "duration_seconds": duration,
            "error_message": f"Timeout dépassé ({timeout}s)",
        }

    except Exception as e:
        duration = round(time.time() - start_time, 2)
        return {
            "success": False,
            "stdout": "",
            "stderr": str(e),
            "returncode": None,
            "duration_seconds": duration,
            "error_message": str(e),
        }


def execute_and_log(script_path: str, log_output_path: str, timeout: int = 120, working_dir: str = None) -> dict:
    """
    Exécute un script et sauvegarde le résultat complet (succès/échec, logs, durée)
    dans un fichier JSON, pour la traçabilité (results/execution_logs/).
    """
    result = execute_workflow(script_path, timeout=timeout, working_dir=working_dir)
    result["script_path"] = str(script_path)

    log_path = Path(log_output_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Exécute un script généré de façon contrôlée.")
    parser.add_argument("--script", required=True, help="Chemin vers le script .py à exécuter")
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--log_output", default=None, help="Chemin du fichier JSON de log (optionnel)")
    args = parser.parse_args()

    if args.log_output:
        result = execute_and_log(args.script, args.log_output, timeout=args.timeout)
    else:
        result = execute_workflow(args.script, timeout=args.timeout)

    print(f"Succès : {result['success']}")
    print(f"Durée : {result['duration_seconds']}s")
    if result["stdout"]:
        print("--- STDOUT ---")
        print(result["stdout"])
    if not result["success"]:
        print("--- ERREUR ---")
        print(result["error_message"])
        if result["stderr"]:
            print("--- STDERR complet ---")
            print(result["stderr"])
