import pandas as pd
import numpy as np
import re
from datetime import datetime, date

INPUT_PATH = "benchmark/datasets/hotel_bookings/noisy_low.csv"
OUTPUT_PATH = "results/cleaned_datasets/hotel_bookings/noisy_low__validated.csv"

# Load the dataset
df = pd.read_csv(INPUT_PATH)

# Initialize logging variables
operations_log = {
    'missing_values_filled': 0,
    'duplicates_removed': 0,
    'date_formats_corrected': 0,
    'numeric_corrections': 0,
    'categorical_standardized': 0,
    'rows_removed': 0
}

# Ensure row_id is present and set as index for deduplication
if 'row_id' in df.columns:
    df.set_index('row_id', inplace=True, drop=False)

# 1. Handle duplicates (keeping first occurrence)
initial_rows = len(df)
df.drop_duplicates(subset=[col for col in df.columns if col != 'row_id'], keep='first', inplace=True)
operations_log['duplicates_removed'] = initial_rows - len(df)

# 2. Correct lead_time (detected '342O' which should be 342)
if 'lead_time' in df.columns:
    def correct_lead_time(val):
        if isinstance(val, str):
            val = re.sub(r'[^0-9]', '', val)
            if val:
                return int(val)
        return val
    df['lead_time'] = df['lead_time'].apply(correct_lead_time)
    operations_log['numeric_corrections'] += df['lead_time'].apply(lambda x: isinstance(x, str)).sum()

# 3. Correct children column (detected 0.0 which should be 0)
if 'children' in df.columns:
    def safe_convert_to_int(val):
        if pd.isna(val):
            return 0
        try:
            if isinstance(val, str) and '.' in val:
                return int(float(val))
            return int(val)
        except (ValueError, TypeError):
            return val
    df['children'] = df['children'].apply(safe_convert_to_int)
    operations_log['missing_values_filled'] += df['children'].isna().sum()
    operations_log['numeric_corrections'] += df['children'].apply(lambda x: isinstance(x, (float, str))).sum()

# 4. Correct meal column (standardize 'BB' to 'BB' - no change needed as per rules)
if 'meal' in df.columns:
    valid_meals = ['BB', 'HB', 'FB', 'SC', 'Undefined']
    df['meal'] = df['meal'].apply(lambda x: x if x in valid_meals else x)

# 5. Correct country column (standardize case)
if 'country' in df.columns:
    df['country'] = df['country'].str.upper()
    operations_log['categorical_standardized'] += df['country'].notna().sum()

# 6. Correct date formats in reservation_status_date
if 'reservation_status_date' in df.columns:
    def parse_date(date_str):
        if pd.isna(date_str):
            return date_str
        date_str = str(date_str).strip()
        # Try multiple date formats
        for fmt in ('%B %d, %Y', '%Y-%m-%d', '%d/%m/%Y', '%m/%d/%Y'):
            try:
                return datetime.strptime(date_str, fmt).date()
            except ValueError:
                continue
        return date_str  # Return original if no format matches

    df['reservation_status_date'] = df['reservation_status_date'].apply(parse_date)
    operations_log['date_formats_corrected'] += df['reservation_status_date'].apply(
        lambda x: isinstance(x, (datetime, date))).sum()

# 7. Correct arrival_date_month (standardize case)
if 'arrival_date_month' in df.columns:
    month_map = {
        'january': 'January', 'february': 'February', 'march': 'March',
        'april': 'April', 'may': 'May', 'june': 'June',
        'july': 'July', 'august': 'August', 'september': 'September',
        'october': 'October', 'november': 'November', 'december': 'December'
    }
    df['arrival_date_month'] = df['arrival_date_month'].str.capitalize().map(month_map).fillna(df['arrival_date_month'])
    operations_log['categorical_standardized'] += df['arrival_date_month'].notna().sum()

# 8. Correct adults column (minimum 1 adult per booking)
if 'adults' in df.columns:
    def safe_convert_adults(val):
        if pd.isna(val):
            return 1
        try:
            if isinstance(val, str):
                val = re.sub(r'[^0-9]', '', val)
                if val:
                    return max(1, int(val))
            return max(1, int(val))
        except (ValueError, TypeError):
            return 1
    df['adults'] = df['adults'].apply(safe_convert_adults)
    operations_log['numeric_corrections'] += (df['adults'] < 1).sum()

# 9. Correct babies column (ensure non-negative)
if 'babies' in df.columns:
    def safe_convert_babies(val):
        if pd.isna(val):
            return 0
        try:
            return max(0, int(float(val))) if isinstance(val, str) and '.' in val else max(0, int(val))
        except (ValueError, TypeError):
            return 0
    df['babies'] = df['babies'].apply(safe_convert_babies)
    operations_log['numeric_corrections'] += (df['babies'] < 0).sum()

# 10. Correct adr (average daily rate) - ensure non-negative
if 'adr' in df.columns:
    def safe_convert_adr(val):
        if pd.isna(val):
            return 0.0
        try:
            return max(0.0, float(val))
        except (ValueError, TypeError):
            return val
    df['adr'] = df['adr'].apply(safe_convert_adr)
    operations_log['numeric_corrections'] += (df['adr'] < 0).sum()

# 11. Correct stays_in_weekend_nights and stays_in_week_nights (ensure non-negative)
for col in ['stays_in_weekend_nights', 'stays_in_week_nights']:
    if col in df.columns:
        def safe_convert_nights(val):
            if pd.isna(val):
                return 0
            try:
                return max(0, int(float(val))) if isinstance(val, str) and '.' in val else max(0, int(val))
            except (ValueError, TypeError):
                return 0
        df[col] = df[col].apply(safe_convert_nights)
        operations_log['numeric_corrections'] += (df[col] < 0).sum()

# 12. Correct previous_cancellations and previous_bookings_not_canceled (ensure non-negative)
for col in ['previous_cancellations', 'previous_bookings_not_canceled']:
    if col in df.columns:
        def safe_convert_count(val):
            if pd.isna(val):
                return 0
            try:
                return max(0, int(float(val))) if isinstance(val, str) and '.' in val else max(0, int(val))
            except (ValueError, TypeError):
                return 0
        df[col] = df[col].apply(safe_convert_count)
        operations_log['numeric_corrections'] += (df[col] < 0).sum()

# 13. Correct days_in_waiting_list (ensure non-negative)
if 'days_in_waiting_list' in df.columns:
    def safe_convert_waiting(val):
        if pd.isna(val):
            return 0
        try:
            return max(0, int(float(val))) if isinstance(val, str) and '.' in val else max(0, int(val))
        except (ValueError, TypeError):
            return 0
    df['days_in_waiting_list'] = df['days_in_waiting_list'].apply(safe_convert_waiting)
    operations_log['numeric_corrections'] += (df['days_in_waiting_list'] < 0).sum()

# 14. Correct total_of_special_requests (ensure non-negative)
if 'total_of_special_requests' in df.columns:
    def safe_convert_requests(val):
        if pd.isna(val):
            return 0
        try:
            return max(0, int(float(val))) if isinstance(val, str) and '.' in val else max(0, int(val))
        except (ValueError, TypeError):
            return 0
    df['total_of_special_requests'] = df['total_of_special_requests'].apply(safe_convert_requests)
    operations_log['numeric_corrections'] += (df['total_of_special_requests'] < 0).sum()

# 15. Correct required_car_parking_spaces (ensure non-negative)
if 'required_car_parking_spaces' in df.columns:
    def safe_convert_parking(val):
        if pd.isna(val):
            return 0
        try:
            return max(0, int(float(val))) if isinstance(val, str) and '.' in val else max(0, int(val))
        except (ValueError, TypeError):
            return 0
    df['required_car_parking_spaces'] = df['required_car_parking_spaces'].apply(safe_convert_parking)
    operations_log['numeric_corrections'] += (df['required_car_parking_spaces'] < 0).sum()

# 16. Correct agent and company (convert to string if numeric, keep NaN as is)
for col in ['agent', 'company']:
    if col in df.columns:
        df[col] = df[col].apply(lambda x: str(int(float(x))) if pd.notna(x) and str(x).replace('.', '').isdigit() else x)

# 17. Correct market_segment and distribution_channel (standardize case)
for col in ['market_segment', 'distribution_channel']:
    if col in df.columns:
        df[col] = df[col].str.upper()
        operations_log['categorical_standardized'] += df[col].notna().sum()

# 18. Correct customer_type (standardize case)
if 'customer_type' in df.columns:
    df['customer_type'] = df['customer_type'].str.capitalize()
    operations_log['categorical_standardized'] += df['customer_type'].notna().sum()

# 19. Correct deposit_type (standardize case)
if 'deposit_type' in df.columns:
    df['deposit_type'] = df['deposit_type'].str.title()
    operations_log['categorical_standardized'] += df['deposit_type'].notna().sum()

# 20. Correct reservation_status (standardize case)
if 'reservation_status' in df.columns:
    df['reservation_status'] = df['reservation_status'].str.title()
    operations_log['categorical_standardized'] += df['reservation_status'].notna().sum()

# 21. Remove rows with invalid dates (after correction attempts)
if 'reservation_status_date' in df.columns:
    invalid_dates = df['reservation_status_date'].apply(lambda x: not isinstance(x, (datetime, date)) and pd.notna(x))
    operations_log['rows_removed'] += invalid_dates.sum()
    df = df[~invalid_dates]

# 22. Ensure row_id is preserved in output
if 'row_id' in df.columns:
    df.reset_index(drop=True, inplace=True)

# Save the cleaned dataset
df.to_csv(OUTPUT_PATH, index=False)

# Print operation summary
print("Data Cleaning Summary:")
print(f"- Rows before cleaning: {initial_rows}")
print(f"- Rows after cleaning: {len(df)}")
print(f"- Duplicates removed: {operations_log['duplicates_removed']}")
print(f"- Missing values filled: {operations_log['missing_values_filled']}")
print(f"- Date formats corrected: {operations_log['date_formats_corrected']}")
print(f"- Numeric corrections applied: {operations_log['numeric_corrections']}")
print(f"- Categorical values standardized: {operations_log['categorical_standardized']}")
print(f"- Rows removed due to invalid data: {operations_log['rows_removed']}")
print(f"Cleaned dataset saved to: {OUTPUT_PATH}")