"""
diagnose_quality.py (v2 - généralisé)
----------------------------------------
Diagnostic détaillé : décompose les FP/FN par colonne, pour identifier
précisément quelles colonnes posent problème dans un script généré donné.

Utilisation :
    python diagnose_quality.py --level medium --approach profile
"""

import argparse
import pandas as pd

DATASET_DIR = "benchmark/datasets/hotel_bookings"
CLEANED_DIR = "results/cleaned_datasets/hotel_bookings"


def values_match(a, b) -> bool:
    if pd.isna(a) and pd.isna(b):
        return True
    if pd.isna(a) or pd.isna(b):
        return False
    if str(a).strip() == str(b).strip():
        return True
    try:
        return float(a) == float(b)
    except (ValueError, TypeError):
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", required=True, choices=["low", "medium", "high"])
    parser.add_argument("--approach", default="profile")
    args = parser.parse_args()

    clean_path = f"{DATASET_DIR}/clean_with_id.csv"
    injected_errors_path = f"{DATASET_DIR}/injected_errors_{args.level}.csv"
    cleaned_output_path = f"{CLEANED_DIR}/noisy_{args.level}__{args.approach}.csv"

    df_clean = pd.read_csv(clean_path)
    errors_log = pd.read_csv(injected_errors_path)
    df_cleaned = pd.read_csv(cleaned_output_path)

    if "row_id" not in df_cleaned.columns:
        print("⚠️ row_id absent du résultat nettoyé — impossible de diagnostiquer.")
        return

    merged = df_clean.merge(df_cleaned, on="row_id", suffixes=("_clean", "_cleaned"))
    injected_cells = set(zip(errors_log["row_id"], errors_log["column"]))

    common_columns = [c for c in df_clean.columns if c != "row_id" and c in df_cleaned.columns]
    missing_columns = [c for c in df_clean.columns if c != "row_id" and c not in df_cleaned.columns]

    if missing_columns:
        print(f"⚠️ Colonnes supprimées par le workflow (violation du system_prompt) : {missing_columns}\n")

    stats_per_column = {col: {"TP": 0, "FN": 0, "FP": 0, "TN": 0} for col in common_columns}
    fp_examples = {col: [] for col in common_columns}

    for _, row in merged.iterrows():
        row_id = row["row_id"]
        for col in common_columns:
            clean_value = row[f"{col}_clean"] if f"{col}_clean" in row else row[col]
            cleaned_value = row[f"{col}_cleaned"] if f"{col}_cleaned" in row else row[col]

            was_injected = (row_id, col) in injected_cells
            is_correct = values_match(clean_value, cleaned_value)

            if was_injected:
                stats_per_column[col]["TP" if is_correct else "FN"] += 1
            else:
                stats_per_column[col]["TN" if is_correct else "FP"] += 1
                if not is_correct and len(fp_examples[col]) < 3:
                    fp_examples[col].append((row_id, clean_value, cleaned_value))

    print(f"{'Colonne':<30} {'TP':>7} {'FN':>7} {'FP':>9} {'TN':>9}")
    print("-" * 65)

    sorted_columns = sorted(common_columns, key=lambda c: stats_per_column[c]["FP"], reverse=True)

    for col in sorted_columns:
        s = stats_per_column[col]
        print(f"{col:<30} {s['TP']:>7} {s['FN']:>7} {s['FP']:>9} {s['TN']:>9}")

    print("\n" + "=" * 65)
    print("EXEMPLES DE FAUX POSITIFS (top 5 colonnes les plus touchées) :")
    print("=" * 65)
    for col in sorted_columns[:5]:
        if fp_examples[col]:
            print(f"\n--- {col} ---")
            for row_id, clean_val, cleaned_val in fp_examples[col]:
                print(f"  row_id={row_id} : clean='{clean_val}'  ->  cleaned='{cleaned_val}'")


if __name__ == "__main__":
    main()