import pandas as pd
import numpy as np
import re
from datetime import datetime
import difflib

INPUT_PATH = r"datasets\flights\noisy_high.csv"
OUTPUT_PATH = r"results\cleaned_datasets\flights\noisy_high__profile.csv"

def extract_numeric(value):
    if pd.isna(value):
        return np.nan
    s = str(value).replace(",", ".")
    s = s.replace("O", "0").replace("o", "0")
    match = re.search(r"-?\d+\.?\d*", s)
    return float(match.group()) if match else np.nan

def parse_time(value):
    if pd.isna(value):
        return value
    s = str(value).strip().lower()
    s = s.replace("a.m.", "AM").replace("p.m.", "PM").replace("a.m", "AM").replace("p.m", "PM")
    s = re.sub(r'\s+', ' ', s)
    formats = [
        "%I:%M %p", "%I:%M%p", "%I %p", "%H:%M", "%I:%M:%S %p",
        "%I:%M:%S%p", "%H:%M:%S", "%I.%M %p", "%I.%M%p"
    ]
    for fmt in formats:
        try:
            dt = datetime.strptime(s, fmt)
            return dt.strftime("%H:%M")
        except ValueError:
            continue
    return value

def harmonize_category(value, valid_values):
    if pd.isna(value):
        return value
    s = str(value).strip()
    if s in valid_values:
        return s
    for v in valid_values:
        if v.lower() == s.lower():
            return v
    match = difflib.get_close_matches(s, valid_values, n=1, cutoff=0.6)
    return match[0] if match else value

def find_best_grouping_column(df, target_col, is_numeric):
    target = df[target_col]
    observed = df[target.notna()]
    if len(observed) < 30:
        return None
    candidates = [c for c in df.columns if c not in (target_col, "row_id")
                  and not df[c].isna().any() and 1 < df[c].nunique() <= 50]
    best_col, best_gain = None, 0.0
    if is_numeric:
        t = pd.to_numeric(observed[target_col], errors="coerce").dropna()
        if len(t) < 30:
            return None
        global_mad = (t - t.median()).abs().median()
        if global_mad == 0:
            return None
        for cand in candidates:
            sub = observed.loc[t.index, [cand]].copy(); sub["_t"] = t
            w_mad, total = 0.0, len(sub)
            for _, grp in sub.groupby(cand)["_t"]:
                w_mad += (grp - grp.median()).abs().median() * (len(grp) / total) if len(grp) >= 5 else global_mad * (len(grp) / total)
            reduction = 1 - (w_mad / global_mad)
            if reduction > best_gain and reduction >= 0.20:
                best_gain, best_col = reduction, cand
    else:
        t = observed[target_col].astype(str)
        global_share = t.value_counts(normalize=True).iloc[0]
        for cand in candidates:
            sub = pd.DataFrame({"_g": observed[cand], "_t": t})
            w_share, total = 0.0, len(sub)
            for _, grp in sub.groupby("_g")["_t"]:
                w_share += grp.value_counts(normalize=True).iloc[0] * (len(grp) / total)
            if (w_share - global_share) > best_gain and (w_share - global_share) >= 0.05:
                best_gain, best_col = w_share - global_share, cand
    return best_col

def clean_dataset():
    df = pd.read_csv(INPUT_PATH)

    log = []

    # tuple_id: numeric, no missing values, no outliers detected (min/max reasonable)
    if 'tuple_id' in df.columns:
        df['tuple_id'] = pd.to_numeric(df['tuple_id'], errors='coerce')
        df['tuple_id'] = df['tuple_id'].fillna(df['tuple_id'].median())
        log.append("tuple_id: converted to numeric and imputed median for any NaN")

    # src: categorical, 30.98% missing, many unique values but some frequent ones
    if 'src' in df.columns:
        valid_src = ["aa", "flylouisville", "flightview", "allegiantair", "helloflight",
                     "businesstravellogue", "quicktrip", "foxbusiness", "flights", "gofox"]
        df['src'] = df['src'].apply(lambda v: harmonize_category(v, valid_src))
        group_col = find_best_grouping_column(df, 'src', False)
        if group_col:
            df['src'] = df.groupby(group_col)['src'].transform(
                lambda s: s.fillna(s.mode().iloc[0] if not s.mode().empty else "aa")
            )
        df['src'] = df['src'].fillna("aa")
        log.append("src: harmonized categories and imputed missing values with mode")

    # flight: categorical, 31.44% missing, many unique values but some frequent patterns
    if 'flight' in df.columns:
        df['flight'] = df['flight'].astype(str).str.strip()
        df['flight'] = df['flight'].replace("", "unknown")
        df['flight'] = df['flight'].replace("nan", "unknown")
        group_col = find_best_grouping_column(df, 'flight', False)
        if group_col:
            df['flight'] = df.groupby(group_col)['flight'].transform(
                lambda s: s.fillna(s.mode().iloc[0] if not s.mode().empty else "unknown")
            )
        df['flight'] = df['flight'].fillna("unknown")
        log.append("flight: standardized empty values and imputed missing with mode")

    # sched_dep_time: no missing, but inconsistent formats
    if 'sched_dep_time' in df.columns:
        df['sched_dep_time'] = df['sched_dep_time'].apply(parse_time)
        log.append("sched_dep_time: standardized time formats")

    # act_dep_time: no missing, but inconsistent formats
    if 'act_dep_time' in df.columns:
        df['act_dep_time'] = df['act_dep_time'].apply(parse_time)
        log.append("act_dep_time: standardized time formats")

    # sched_arr_time: no missing, but inconsistent formats
    if 'sched_arr_time' in df.columns:
        df['sched_arr_time'] = df['sched_arr_time'].apply(parse_time)
        log.append("sched_arr_time: standardized time formats")

    # act_arr_time: no missing, but inconsistent formats
    if 'act_arr_time' in df.columns:
        df['act_arr_time'] = df['act_arr_time'].apply(parse_time)
        log.append("act_arr_time: standardized time formats")

    # Remove duplicates (keeping first occurrence)
    initial_rows = len(df)
    df = df.drop_duplicates(subset=[c for c in df.columns if c != "row_id"], keep='first')
    final_rows = len(df)
    duplicates_removed = initial_rows - final_rows
    if duplicates_removed > 0:
        log.append(f"Removed {duplicates_removed} duplicate rows")

    df.to_csv(OUTPUT_PATH, index=False)

    print("=== Cleaning Summary ===")
    for entry in log:
        print(f"- {entry}")
    print(f"- Initial rows: {initial_rows}, Final rows: {final_rows}")
    print(f"Cleaned dataset saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    clean_dataset()