import pandas as pd
import numpy as np
import re
from datetime import datetime
import difflib

INPUT_PATH = r"benchmark\datasets\hotel_bookings\noisy_low.csv"
OUTPUT_PATH = r"results\cleaned_datasets\hotel_bookings\noisy_low__constrained.csv"

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
    date_formats = [
        "%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%B %d, %Y",
        "%d-%b-%Y", "%Y/%m/%d", "%B %d %Y", "%d %B %Y"
    ]
    s = str(value).strip()
    for fmt in date_formats:
        try:
            dt = datetime.strptime(s, fmt)
            return dt.strftime("%Y-%m-%d")
        except ValueError:
            continue
    if s.isdigit() and len(s) in (9, 10):
        try:
            dt = datetime.fromtimestamp(int(s))
            return dt.strftime("%Y-%m-%d")
        except (ValueError, OSError):
            pass
    return value

def clean_dataset():
    df = pd.read_csv(INPUT_PATH)

    operations_log = {
        'missing_values_imputed': 0,
        'duplicates_dropped': 0,
        'numeric_values_extracted': 0,
        'dates_standardized': 0,
        'categories_harmonized': 0,
        'outliers_corrected': 0
    }

    # Handle duplicates
    initial_rows = len(df)
    df = df.drop_duplicates(subset=[col for col in df.columns if col != 'row_id'])
    operations_log['duplicates_dropped'] = initial_rows - len(df)

    # Numeric extraction for specific columns
    numeric_cols = ['lead_time', 'adults', 'children', 'babies', 'adr']
    for col in numeric_cols:
        if col in df.columns:
            initial_missing = df[col].isna().sum()
            df[col] = df[col].apply(extract_numeric)
            operations_log['numeric_values_extracted'] += (initial_missing - df[col].isna().sum())

    # Convert numeric columns to appropriate types
    numeric_cols_convert = {
        'is_canceled': 'int',
        'arrival_date_year': 'int',
        'arrival_date_week_number': 'int',
        'arrival_date_day_of_month': 'int',
        'stays_in_weekend_nights': 'int',
        'stays_in_week_nights': 'int',
        'babies': 'int',
        'is_repeated_guest': 'int',
        'previous_cancellations': 'int',
        'previous_bookings_not_canceled': 'int',
        'booking_changes': 'int',
        'days_in_waiting_list': 'int',
        'required_car_parking_spaces': 'int',
        'total_of_special_requests': 'int'
    }

    for col, dtype in numeric_cols_convert.items():
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            if dtype == 'int':
                df[col] = df[col].fillna(0).astype(int)

    # Handle missing values
    categorical_cols = ['hotel', 'meal', 'country', 'market_segment',
                        'distribution_channel', 'deposit_type', 'customer_type',
                        'reservation_status', 'reserved_room_type', 'assigned_room_type']
    numeric_impute_cols = ['lead_time', 'adults', 'children', 'adr']

    for col in categorical_cols:
        if col in df.columns:
            mode_val = df[col].mode()[0]
            df[col] = df[col].fillna(mode_val)
            operations_log['missing_values_imputed'] += df[col].isna().sum()

    for col in numeric_impute_cols:
        if col in df.columns:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            operations_log['missing_values_imputed'] += df[col].isna().sum()

    # Harmonize categorical values
    valid_values = {
        'hotel': ["Resort Hotel", "City Hotel"],
        'meal': ["BB", "HB", "FB", "SC", "Undefined"],
        'market_segment': ["Direct", "Corporate", "Online TA", "Offline TA/TO", "Complementary", "Groups", "Aviation"],
        'distribution_channel': ["Direct", "Corporate", "TA/TO", "GDS"],
        'deposit_type': ["No Deposit", "Non Refund", "Refundable"],
        'customer_type': ["Transient", "Contract", "Transient-Party", "Group"],
        'reservation_status': ["Check-Out", "Canceled", "No-Show"]
    }

    for col, values in valid_values.items():
        if col in df.columns:
            df[col] = df[col].apply(lambda v: harmonize_category(v, values))
            operations_log['categories_harmonized'] += 1

    # Standardize date columns
    if 'reservation_status_date' in df.columns:
        initial_missing = df['reservation_status_date'].isna().sum()
        df['reservation_status_date'] = df['reservation_status_date'].apply(parse_date)
        operations_log['dates_standardized'] += (initial_missing - df['reservation_status_date'].isna().sum())

    # Correct outliers in numeric columns
    outlier_cols = {
        'lead_time': (0, 730),
        'adults': (1, 4),
        'children': (0, 10),
        'babies': (0, 10),
        'adr': (0, 5000),
        'days_in_waiting_list': (0, 365),
        'previous_cancellations': (0, 10),
        'previous_bookings_not_canceled': (0, 10),
        'booking_changes': (0, 10),
        'required_car_parking_spaces': (0, 5),
        'total_of_special_requests': (0, 5)
    }

    for col, (min_val, max_val) in outlier_cols.items():
        if col in df.columns:
            mask = (df[col] < min_val) | (df[col] > max_val)
            if mask.any():
                if col in ['children', 'babies', 'previous_cancellations',
                          'previous_bookings_not_canceled', 'booking_changes',
                          'required_car_parking_spaces', 'total_of_special_requests']:
                    impute_val = df[col].mode()[0]
                else:
                    impute_val = df[col].median()
                df.loc[mask, col] = impute_val
                operations_log['outliers_corrected'] += mask.sum()

    # Strip whitespace from string columns
    str_cols = [col for col in df.columns if df[col].dtype == 'object']
    for col in str_cols:
        df[col] = df[col].str.strip()

    # Save cleaned dataset
    df.to_csv(OUTPUT_PATH, index=False)

    # Print operations summary
    print("Data Cleaning Operations Summary:")
    print(f"- Rows after dropping duplicates: {len(df)} (dropped {operations_log['duplicates_dropped']})")
    print(f"- Missing values imputed: {operations_log['missing_values_imputed']}")
    print(f"- Numeric values extracted: {operations_log['numeric_values_extracted']}")
    print(f"- Dates standardized: {operations_log['dates_standardized']}")
    print(f"- Categories harmonized: {operations_log['categories_harmonized']} columns")
    print(f"- Outliers corrected: {operations_log['outliers_corrected']}")

if __name__ == "__main__":
    clean_dataset()