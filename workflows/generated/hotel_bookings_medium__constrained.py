import pandas as pd
import numpy as np
import re
from datetime import datetime
import difflib

INPUT_PATH = r"benchmark\datasets\hotel_bookings\noisy_medium.csv"
OUTPUT_PATH = r"results\cleaned_datasets\hotel_bookings\noisy_medium__constrained.csv"

def extract_numeric(value):
    if pd.isna(value):
        return np.nan
    s = str(value).replace(",", ".")
    s = s.replace("O", "0").replace("o", "0")
    match = re.search(r"-?\d+\.?\d*", s)
    return float(match.group()) if match else np.nan

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

def parse_date(value):
    if pd.isna(value):
        return value
    formats = ["%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%B %d, %Y", "%d-%b-%Y", "%Y/%m/%d"]
    for fmt in formats:
        try:
            return datetime.strptime(str(value), fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    try:
        if str(value).isdigit() and len(str(value)) in (9, 10):
            return datetime.fromtimestamp(int(value)).strftime("%Y-%m-%d")
    except (ValueError, OSError):
        pass
    return str(value)

def clean_dataset():
    df = pd.read_csv(INPUT_PATH)

    operations_log = {
        'missing_values_imputed': 0,
        'duplicates_dropped': 0,
        'numeric_extracted': 0,
        'categories_harmonized': 0,
        'dates_standardized': 0,
        'outliers_corrected': 0
    }

    # Handle missing values
    for col in df.columns:
        if col == 'row_id':
            continue
        if df[col].isna().any():
            if df[col].dtype == 'object':
                mode_val = df[col].mode()[0]
                df[col].fillna(mode_val, inplace=True)
                operations_log['missing_values_imputed'] += df[col].isna().sum()
            else:
                if df[col].nunique() < 10 or df[col].value_counts(normalize=True).iloc[0] > 0.9:
                    mode_val = df[col].mode()[0]
                    df[col].fillna(mode_val, inplace=True)
                else:
                    median_val = df[col].median()
                    df[col].fillna(median_val, inplace=True)
                operations_log['missing_values_imputed'] += df[col].isna().sum()

    # Drop duplicates (keeping first occurrence)
    initial_rows = len(df)
    df.drop_duplicates(subset=[c for c in df.columns if c != 'row_id'], keep='first', inplace=True)
    operations_log['duplicates_dropped'] = initial_rows - len(df)

    # Extract numeric values from text-corrupted numeric columns
    numeric_cols = ['lead_time', 'adults', 'children', 'agent', 'adr']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].apply(extract_numeric)
            operations_log['numeric_extracted'] += df[col].isna().sum()

    # Convert numeric columns to appropriate types
    numeric_cols = [
        'is_canceled', 'arrival_date_year', 'arrival_date_week_number',
        'arrival_date_day_of_month', 'stays_in_weekend_nights',
        'stays_in_week_nights', 'adults', 'children', 'babies',
        'previous_cancellations', 'previous_bookings_not_canceled',
        'booking_changes', 'company', 'days_in_waiting_list', 'adr',
        'required_car_parking_spaces', 'total_of_special_requests'
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            if df[col].isna().any():
                if df[col].nunique() < 10 or df[col].value_counts(normalize=True).iloc[0] > 0.9:
                    mode_val = df[col].mode()[0]
                    df[col].fillna(mode_val, inplace=True)
                else:
                    median_val = df[col].median()
                    df[col].fillna(median_val, inplace=True)
                operations_log['missing_values_imputed'] += df[col].isna().sum()

    # Harmonize categorical columns
    categorical_harmonization = {
        'hotel': ['Resort Hotel', 'City Hotel'],
        'meal': ['BB', 'HB', 'FB', 'SC', 'Undefined'],
        'market_segment': ['Direct', 'Corporate', 'Online TA', 'Offline TA/TO', 'Groups', 'Complementary', 'Aviation'],
        'distribution_channel': ['Direct', 'Corporate', 'TA/TO', 'GDS'],
        'deposit_type': ['No Deposit', 'Non Refund', 'Refundable'],
        'customer_type': ['Transient', 'Contract', 'Transient-Party', 'Group'],
        'reservation_status': ['Check-Out', 'Canceled', 'No-Show']
    }
    for col, valid_values in categorical_harmonization.items():
        if col in df.columns:
            df[col] = df[col].apply(lambda v: harmonize_category(v, valid_values))
            operations_log['categories_harmonized'] += df[col].apply(lambda x: x not in valid_values and not pd.isna(x)).sum()

    # Standardize date column
    if 'reservation_status_date' in df.columns:
        df['reservation_status_date'] = df['reservation_status_date'].apply(parse_date)
        operations_log['dates_standardized'] = len(df)

    # Correct outliers in numeric columns
    outlier_cols = {
        'lead_time': (0, 730),
        'stays_in_weekend_nights': (0, 14),
        'stays_in_week_nights': (0, 50),
        'adults': (1, 4),
        'children': (0, 10),
        'babies': (0, 10),
        'previous_cancellations': (0, 10),
        'previous_bookings_not_canceled': (0, 10),
        'booking_changes': (0, 20),
        'days_in_waiting_list': (0, 365),
        'adr': (0, 5000),
        'required_car_parking_spaces': (0, 5),
        'total_of_special_requests': (0, 5)
    }
    for col, (min_val, max_val) in outlier_cols.items():
        if col in df.columns:
            mask = (df[col] < min_val) | (df[col] > max_val)
            if mask.any():
                if df[col].nunique() < 10 or df[col].value_counts(normalize=True).iloc[0] > 0.9:
                    mode_val = df[col].mode()[0]
                    df.loc[mask, col] = mode_val
                else:
                    median_val = df[col].median()
                    df.loc[mask, col] = median_val
                operations_log['outliers_corrected'] += mask.sum()

    # Strip whitespace from all string columns
    for col in df.select_dtypes(include=['object']).columns:
        if col != 'row_id':
            df[col] = df[col].str.strip()

    # Save cleaned dataset
    df.to_csv(OUTPUT_PATH, index=False)

    # Print operations summary
    print("Data Cleaning Operations Summary:")
    print(f"- Rows after cleaning: {len(df)}")
    print(f"- Columns: {len(df.columns)}")
    print(f"- Missing values imputed: {operations_log['missing_values_imputed']}")
    print(f"- Duplicates dropped: {operations_log['duplicates_dropped']}")
    print(f"- Numeric values extracted from text: {operations_log['numeric_extracted']}")
    print(f"- Categorical values harmonized: {operations_log['categories_harmonized']}")
    print(f"- Dates standardized: {operations_log['dates_standardized']}")
    print(f"- Outliers corrected: {operations_log['outliers_corrected']}")

if __name__ == "__main__":
    clean_dataset()