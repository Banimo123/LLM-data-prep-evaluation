"""
diagnose_fp.py
---------------
Identifie precisement quelles colonnes generent le plus de "faux positifs" (cellules
correctes cassees par le workflow), pour n'importe quel dataset/approche/niveau.

Usage (depuis la racine du depot) :
    python diagnose_fp.py <dataset> <approche> <niveau>
Exemple :
    python diagnose_fp.py flights manual_baseline low
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fast_metrics import _norm_str_series


def main():
    if len(sys.argv) != 4:
        print("Usage : python diagnose_fp.py <dataset> <approche> <niveau>")
        print("Exemple : python diagnose_fp.py flights manual_baseline low")
        sys.exit(1)

    dataset, approach, level = sys.argv[1], sys.argv[2], sys.argv[3]

    clean_csv = f"datasets/{dataset}/clean_with_id.csv"
    noisy_csv = f"datasets/{dataset}/noisy_{level}.csv"
    errors_csv = f"datasets/{dataset}/injected_errors_{level}.csv"
    cleaned_csv = f"results/cleaned_datasets/{dataset}/noisy_{level}__{approach}.csv"

    clean = pd.read_csv(clean_csv, low_memory=False).set_index("row_id")
    noisy = pd.read_csv(noisy_csv, low_memory=False).set_index("row_id")
    cleaned = pd.read_csv(cleaned_csv, low_memory=False).set_index("row_id")
    errors = pd.read_csv(errors_csv, low_memory=False)
    err_by_col = {c: set(g["row_id"]) for c, g in errors.groupby("column")}

    lost = set(noisy.index) - set(cleaned.index)
    active = noisy.index[~noisy.index.isin(lost)]

    fp_counts = {}
    fp_examples = {}

    for col in clean.columns:
        if col not in cleaned.columns:
            continue
        cv = clean.loc[active, col]
        clv = cleaned.reindex(active)[col]
        nv = noisy.loc[active, col]
        is_num = pd.api.types.is_numeric_dtype(clean[col])

        if is_num:
            cf = pd.to_numeric(cv, errors="coerce")
            clf = pd.to_numeric(clv, errors="coerce")
            cna = cf.isna()
            diff = (cf - clf).abs()
            tol = np.maximum(0.5, 0.01 * cf.abs())
            matches = np.where(cna, clf.isna(), diff <= tol)
            matches = pd.Series(matches, index=active).fillna(False)
        else:
            cs = _norm_str_series(cv)
            cls = _norm_str_series(clv)
            matches = cs == cls
            cna = cs == "<NA>"

        ns = _norm_str_series(nv)
        cs2 = _norm_str_series(clv)
        changed = cs2 != ns
        is_err = pd.Series(active.isin(err_by_col.get(col, set())), index=active)
        fp_mask = (~is_err) & changed & (~matches) & (~cna)

        n = int(fp_mask.sum())
        if n > 0:
            fp_counts[col] = n
            fp_rows = active[fp_mask][:3]
            examples = []
            for rid in fp_rows:
                examples.append({
                    "row_id": int(rid),
                    "clean": clean.loc[rid, col],
                    "noisy": noisy.loc[rid, col],
                    "cleaned": cleaned.loc[rid, col] if rid in cleaned.index else None,
                })
            fp_examples[col] = examples

    print(f"=== Faux positifs par colonne : {dataset} / {approach} / {level} ===\n")
    for col, n in sorted(fp_counts.items(), key=lambda x: -x[1]):
        print(f"{col:30s} {n:>8d} faux positifs")
        for ex in fp_examples[col]:
            print(f"    row_id={ex['row_id']:>7d} | clean='{ex['clean']}' -> "
                  f"noisy='{ex['noisy']}' -> cleaned='{ex['cleaned']}'")
        print()


if __name__ == "__main__":
    main()
