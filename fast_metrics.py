import re

import pandas as pd
import numpy as np

_MIDNIGHT_SUFFIX_RE = re.compile(r"^(\d{4}-\d{2}-\d{2}) 00:00:00(\.0+)?$")


def _norm_str_series(s: pd.Series) -> pd.Series:
    out = s.astype(object)
    is_na = pd.isna(out)
    # entiers stockés en float (3.0 -> "3")
    def conv(v):
        if isinstance(v, float) and v.is_integer():
            return str(int(v))
        sv = str(v).strip()
        m = _MIDNIGHT_SUFFIX_RE.match(sv)
        return m.group(1) if m else sv
    out = out.where(is_na, out.map(conv))
    out[is_na] = "<NA>"
    return out


def evaluate_fast(clean_csv, noisy_csv, injected_errors_csv, cleaned_csv,
                   id_col="row_id", numeric_rel_tol=0.01, numeric_abs_tol=0.5):
    df_clean = pd.read_csv(clean_csv, low_memory=False).set_index(id_col)
    df_noisy = pd.read_csv(noisy_csv, low_memory=False).set_index(id_col)
    df_errors = pd.read_csv(injected_errors_csv, low_memory=False)
    df_cleaned = pd.read_csv(cleaned_csv, low_memory=False).set_index(id_col)

    numeric_cols = {c for c in df_clean.columns if pd.api.types.is_numeric_dtype(df_clean[c])}
    lost_ids = set(df_noisy.index) - set(df_cleaned.index)

    err_by_col = {c: set(g["row_id"]) for c, g in df_errors.groupby("column")}
    err_fam_by_col = {c: dict(zip(g["row_id"], g["error_family"])) for c, g in df_errors.groupby("column")}

    data_columns = [c for c in df_clean.columns]
    per_column = {}
    per_family = {}
    columns_dropped = []

    active_ids = df_noisy.index[~df_noisy.index.isin(lost_ids)]

    for col in data_columns:
        error_ids_col = err_by_col.get(col, set())
        col_dropped = col not in df_cleaned.columns
        if col_dropped:
            columns_dropped.append(col)

        tp = fp = fn = 0

        if col_dropped:
            # toutes les cellules d'erreur de cette colonne -> FN (lost ou pas)
            n_err = len(error_ids_col)
            fn += n_err
            for rid in error_ids_col:
                fam = err_fam_by_col[col][rid]
                d = per_family.setdefault(fam, {"tp": 0, "fn": 0})
                d["fn"] += 1
            per_column[col] = {"tp": 0, "fp": 0, "fn": n_err}
            continue

        # lignes perdues : cellules d'erreur -> FN
        lost_err_ids = error_ids_col & lost_ids
        fn += len(lost_err_ids)
        for rid in lost_err_ids:
            fam = err_fam_by_col[col][rid]
            d = per_family.setdefault(fam, {"tp": 0, "fn": 0})
            d["fn"] += 1

        # lignes actives (ni perdues, colonne présente)
        idx = active_ids
        clean_v = df_clean.loc[idx, col]
        cleaned_v = df_cleaned.reindex(idx)[col]
        noisy_v = df_noisy.loc[idx, col]

        is_numeric = col in numeric_cols

        if is_numeric:
            clean_f = pd.to_numeric(clean_v, errors="coerce")
            cleaned_f = pd.to_numeric(cleaned_v, errors="coerce")
            clean_is_na = clean_f.isna()
            cleaned_is_na = cleaned_f.isna()
            diff = (clean_f - cleaned_f).abs()
            tol = np.maximum(numeric_abs_tol, numeric_rel_tol * clean_f.abs())
            matches = np.where(clean_is_na, cleaned_is_na, diff <= tol)
            matches = pd.Series(matches, index=idx).fillna(False)
            clean_na_mask = clean_is_na
        else:
            clean_s = _norm_str_series(clean_v)
            cleaned_s = _norm_str_series(cleaned_v)
            matches = (clean_s == cleaned_s)
            clean_na_mask = (clean_s == "<NA>")

        noisy_s = _norm_str_series(noisy_v)
        cleaned_s_cmp = _norm_str_series(cleaned_v)
        changed = (cleaned_s_cmp != noisy_s)

        is_error = idx.isin(error_ids_col)
        is_error = pd.Series(is_error, index=idx)

        # TP / FN sur cellules d'erreur
        col_tp = int((matches & is_error).sum())
        col_fn_active = int((~matches & is_error).sum())

        # FP sur cellules non-erreur, hors NA naturel dans clean
        non_error_mask = ~is_error
        fp_mask = non_error_mask & changed & (~matches) & (~clean_na_mask)
        col_fp = int(fp_mask.sum())

        tp += col_tp
        fn += col_fn_active
        fp += col_fp

        per_column[col] = {"tp": tp, "fp": fp, "fn": fn}

        # décompte par famille pour les cellules actives (TP/FN)
        if error_ids_col:
            err_rows = idx[is_error.values]
            for rid in err_rows:
                fam = err_fam_by_col[col][rid]
                d = per_family.setdefault(fam, {"tp": 0, "fn": 0})
                if matches.loc[rid]:
                    d["tp"] += 1
                else:
                    d["fn"] += 1

    def _prf(tp, fp, fn):
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        return round(precision, 4), round(recall, 4), round(f1, 4)

    tp_total = sum(v["tp"] for v in per_column.values())
    fp_total = sum(v["fp"] for v in per_column.values())
    fn_total = sum(v["fn"] for v in per_column.values())
    p, r, f1 = _prf(tp_total, fp_total, fn_total)

    per_family_report = {}
    for fam, counts in per_family.items():
        _, rr, ff = _prf(counts["tp"], 0, counts["fn"])
        per_family_report[fam] = {**counts, "recall": rr, "f1_sans_fp": ff}

    return {
        "global": {
            "precision": p, "recall": r, "f1": f1,
            "tp": tp_total, "fp": fp_total, "fn": fn_total,
            "n_rows_lost_by_workflow": len(lost_ids),
            "columns_dropped_by_workflow": columns_dropped,
        },
        "per_error_family": per_family_report,
    }


if __name__ == "__main__":
    import sys, json
    result = evaluate_fast(*sys.argv[1:5])
    print(json.dumps(result, indent=2, ensure_ascii=False))