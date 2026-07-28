import pandas as pd
import numpy as np
import re
from datetime import datetime
import difflib

INPUT_PATH = r"benchmark\datasets\hotel_bookings\noisy_high.csv"
OUTPUT_PATH = r"results\cleaned_datasets\hotel_bookings\noisy_high__constrained.csv"

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
        "%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y",
        "%B %d, %Y", "%d-%b-%Y", "%Y/%m/%d"
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

df = pd.read_csv(INPUT_PATH)

log = {
    "missing_values_imputed": {},
    "duplicates_dropped": 0,
    "numeric_values_extracted": {},
    "categorical_values_harmonized": {},
    "dates_standardized": 0,
    "outliers_corrected": {}
}

# Handle duplicates
initial_rows = len(df)
df = df.drop_duplicates(subset=[col for col in df.columns if col != "row_id"], keep="first")
log["duplicates_dropped"] = initial_rows - len(df)

# Standardize hotel column
valid_hotel = ["Resort Hotel", "City Hotel"]
df["hotel"] = df["hotel"].apply(lambda v: harmonize_category(v, valid_hotel))
log["categorical_values_harmonized"]["hotel"] = "Harmonized to Resort Hotel/City Hotel"

# Convert is_canceled to numeric
df["is_canceled"] = pd.to_numeric(df["is_canceled"], errors="coerce")
missing_before = df["is_canceled"].isna().sum()
df["is_canceled"] = df["is_canceled"].fillna(df["is_canceled"].mode()[0])
log["missing_values_imputed"]["is_canceled"] = missing_before

# Extract numeric from lead_time
initial_missing = df["lead_time"].isna().sum()
df["lead_time"] = df["lead_time"].apply(extract_numeric)
missing_after = df["lead_time"].isna().sum()
log["numeric_values_extracted"]["lead_time"] = initial_missing - missing_after
df["lead_time"] = df["lead_time"].fillna(df["lead_time"].median())

# arrival_date_year - already numeric, just handle missing
missing_before = df["arrival_date_year"].isna().sum()
df["arrival_date_year"] = pd.to_numeric(df["arrival_date_year"], errors="coerce")
df["arrival_date_year"] = df["arrival_date_year"].fillna(df["arrival_date_year"].median())
log["missing_values_imputed"]["arrival_date_year"] = missing_before

# Standardize arrival_date_month
valid_months = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December"
]
df["arrival_date_month"] = df["arrival_date_month"].apply(lambda v: harmonize_category(v, valid_months))
log["categorical_values_harmonized"]["arrival_date_month"] = "Harmonized to full month names"

# arrival_date_week_number - numeric
missing_before = df["arrival_date_week_number"].isna().sum()
df["arrival_date_week_number"] = pd.to_numeric(df["arrival_date_week_number"], errors="coerce")
df["arrival_date_week_number"] = df["arrival_date_week_number"].fillna(df["arrival_date_week_number"].median())
log["missing_values_imputed"]["arrival_date_week_number"] = missing_before

# arrival_date_day_of_month - numeric
missing_before = df["arrival_date_day_of_month"].isna().sum()
df["arrival_date_day_of_month"] = pd.to_numeric(df["arrival_date_day_of_month"], errors="coerce")
df["arrival_date_day_of_month"] = df["arrival_date_day_of_month"].fillna(df["arrival_date_day_of_month"].median())
log["missing_values_imputed"]["arrival_date_day_of_month"] = missing_before

# stays_in_weekend_nights - numeric
initial_missing = df["stays_in_weekend_nights"].isna().sum()
df["stays_in_weekend_nights"] = df["stays_in_weekend_nights"].apply(extract_numeric)
missing_after = df["stays_in_weekend_nights"].isna().sum()
log["numeric_values_extracted"]["stays_in_weekend_nights"] = initial_missing - missing_after
df["stays_in_weekend_nights"] = df["stays_in_weekend_nights"].fillna(df["stays_in_weekend_nights"].median())

# stays_in_week_nights - numeric
initial_missing = df["stays_in_week_nights"].isna().sum()
df["stays_in_week_nights"] = df["stays_in_week_nights"].apply(extract_numeric)
missing_after = df["stays_in_week_nights"].isna().sum()
log["numeric_values_extracted"]["stays_in_week_nights"] = initial_missing - missing_after
df["stays_in_week_nights"] = df["stays_in_week_nights"].fillna(df["stays_in_week_nights"].median())

# adults - extract numeric
initial_missing = df["adults"].isna().sum()
df["adults"] = df["adults"].apply(extract_numeric)
missing_after = df["adults"].isna().sum()
log["numeric_values_extracted"]["adults"] = initial_missing - missing_after
df["adults"] = df["adults"].fillna(df["adults"].median())

# children - extract numeric
initial_missing = df["children"].isna().sum()
df["children"] = df["children"].apply(extract_numeric)
missing_after = df["children"].isna().sum()
log["numeric_values_extracted"]["children"] = initial_missing - missing_after
df["children"] = df["children"].fillna(df["children"].median())

# babies - numeric
missing_before = df["babies"].isna().sum()
df["babies"] = pd.to_numeric(df["babies"], errors="coerce")
df["babies"] = df["babies"].fillna(df["babies"].median())
log["missing_values_imputed"]["babies"] = missing_before

# meal - harmonize
valid_meal = ["BB", "HB", "FB", "SC", "Undefined"]
df["meal"] = df["meal"].apply(lambda v: harmonize_category(v, valid_meal))
log["categorical_values_harmonized"]["meal"] = "Harmonized to BB/HB/FB/SC/Undefined"

# country - clean empty strings
df["country"] = df["country"].str.strip()
df["country"] = df["country"].replace("", np.nan)
missing_before = df["country"].isna().sum()
df["country"] = df["country"].fillna(df["country"].mode()[0])
log["missing_values_imputed"]["country"] = missing_before

# market_segment - harmonize
valid_market_segment = ["Direct", "Corporate", "Online TA", "Offline TA/TO", "Complementary", "Groups", "Undefined"]
df["market_segment"] = df["market_segment"].apply(lambda v: harmonize_category(v, valid_market_segment))
log["categorical_values_harmonized"]["market_segment"] = "Harmonized to standard market segments"

# distribution_channel - harmonize
valid_distribution = ["Direct", "Corporate", "TA/TO", "Undefined"]
df["distribution_channel"] = df["distribution_channel"].apply(lambda v: harmonize_category(v, valid_distribution))
log["categorical_values_harmonized"]["distribution_channel"] = "Harmonized to Direct/Corporate/TA/TO"

# is_repeated_guest - numeric
df["is_repeated_guest"] = pd.to_numeric(df["is_repeated_guest"], errors="coerce")
missing_before = df["is_repeated_guest"].isna().sum()
df["is_repeated_guest"] = df["is_repeated_guest"].fillna(df["is_repeated_guest"].mode()[0])
log["missing_values_imputed"]["is_repeated_guest"] = missing_before

# previous_cancellations - numeric
initial_missing = df["previous_cancellations"].isna().sum()
df["previous_cancellations"] = df["previous_cancellations"].apply(extract_numeric)
missing_after = df["previous_cancellations"].isna().sum()
log["numeric_values_extracted"]["previous_cancellations"] = initial_missing - missing_after
df["previous_cancellations"] = df["previous_cancellations"].fillna(df["previous_cancellations"].median())

# previous_bookings_not_canceled - numeric
initial_missing = df["previous_bookings_not_canceled"].isna().sum()
df["previous_bookings_not_canceled"] = df["previous_bookings_not_canceled"].apply(extract_numeric)
missing_after = df["previous_bookings_not_canceled"].isna().sum()
log["numeric_values_extracted"]["previous_bookings_not_canceled"] = initial_missing - missing_after
df["previous_bookings_not_canceled"] = df["previous_bookings_not_canceled"].fillna(df["previous_bookings_not_canceled"].median())

# reserved_room_type - clean
df["reserved_room_type"] = df["reserved_room_type"].str.strip().str.upper()
df["reserved_room_type"] = df["reserved_room_type"].replace("", np.nan)
missing_before = df["reserved_room_type"].isna().sum()
df["reserved_room_type"] = df["reserved_room_type"].fillna(df["reserved_room_type"].mode()[0])
log["missing_values_imputed"]["reserved_room_type"] = missing_before

# assigned_room_type - clean
df["assigned_room_type"] = df["assigned_room_type"].str.strip().str.upper()
df["assigned_room_type"] = df["assigned_room_type"].replace("", np.nan)
missing_before = df["assigned_room_type"].isna().sum()
df["assigned_room_type"] = df["assigned_room_type"].fillna(df["assigned_room_type"].mode()[0])
log["missing_values_imputed"]["assigned_room_type"] = missing_before

# booking_changes - numeric
initial_missing = df["booking_changes"].isna().sum()
df["booking_changes"] = df["booking_changes"].apply(extract_numeric)
missing_after = df["booking_changes"].isna().sum()
log["numeric_values_extracted"]["booking_changes"] = initial_missing - missing_after
df["booking_changes"] = df["booking_changes"].fillna(df["booking_changes"].median())

# deposit_type - harmonize
valid_deposit = ["No Deposit", "Non Refund", "Refundable"]
df["deposit_type"] = df["deposit_type"].apply(lambda v: harmonize_category(v, valid_deposit))
log["categorical_values_harmonized"]["deposit_type"] = "Harmonized to No Deposit/Non Refund/Refundable"

# agent - extract numeric
initial_missing = df["agent"].isna().sum()
df["agent"] = df["agent"].apply(extract_numeric)
missing_after = df["agent"].isna().sum()
log["numeric_values_extracted"]["agent"] = initial_missing - missing_after
df["agent"] = df["agent"].fillna(df["agent"].median())

# company - numeric
initial_missing = df["company"].isna().sum()
df["company"] = df["company"].apply(extract_numeric)
missing_after = df["company"].isna().sum()
log["numeric_values_extracted"]["company"] = initial_missing - missing_after
df["company"] = df["company"].fillna(df["company"].median())

# days_in_waiting_list - numeric
initial_missing = df["days_in_waiting_list"].isna().sum()
df["days_in_waiting_list"] = df["days_in_waiting_list"].apply(extract_numeric)
missing_after = df["days_in_waiting_list"].isna().sum()
log["numeric_values_extracted"]["days_in_waiting_list"] = initial_missing - missing_after
df["days_in_waiting_list"] = df["days_in_waiting_list"].fillna(df["days_in_waiting_list"].median())

# customer_type - harmonize
valid_customer_type = ["Transient", "Contract", "Transient-Party", "Group"]
df["customer_type"] = df["customer_type"].apply(lambda v: harmonize_category(v, valid_customer_type))
log["categorical_values_harmonized"]["customer_type"] = "Harmonized to standard customer types"

# adr - extract numeric
initial_missing = df["adr"].isna().sum()
df["adr"] = df["adr"].apply(extract_numeric)
missing_after = df["adr"].isna().sum()
log["numeric_values_extracted"]["adr"] = initial_missing - missing_after
df["adr"] = df["adr"].fillna(df["adr"].median())

# required_car_parking_spaces - numeric
initial_missing = df["required_car_parking_spaces"].isna().sum()
df["required_car_parking_spaces"] = df["required_car_parking_spaces"].apply(extract_numeric)
missing_after = df["required_car_parking_spaces"].isna().sum()
log["numeric_values_extracted"]["required_car_parking_spaces"] = initial_missing - missing_after
df["required_car_parking_spaces"] = df["required_car_parking_spaces"].fillna(df["required_car_parking_spaces"].median())

# total_of_special_requests - numeric
initial_missing = df["total_of_special_requests"].isna().sum()
df["total_of_special_requests"] = df["total_of_special_requests"].apply(extract_numeric)
missing_after = df["total_of_special_requests"].isna().sum()
log["numeric_values_extracted"]["total_of_special_requests"] = initial_missing - missing_after
df["total_of_special_requests"] = df["total_of_special_requests"].fillna(df["total_of_special_requests"].median())

# reservation_status - harmonize
valid_status = ["Check-Out", "Canceled", "No-Show"]
df["reservation_status"] = df["reservation_status"].apply(lambda v: harmonize_category(v, valid_status))
log["categorical_values_harmonized"]["reservation_status"] = "Harmonized to Check-Out/Canceled/No-Show"

# reservation_status_date - standardize
initial_missing = df["reservation_status_date"].isna().sum()
df["reservation_status_date"] = df["reservation_status_date"].apply(parse_date)
missing_after = df["reservation_status_date"].isna().sum()
log["dates_standardized"] = initial_missing - missing_after
df["reservation_status_date"] = df["reservation_status_date"].fillna(df["reservation_status_date"].mode()[0])

# Correct outliers for numeric columns with plausible bounds
numeric_cols = [
    "lead_time", "arrival_date_year", "arrival_date_week_number",
    "arrival_date_day_of_month", "stays_in_weekend_nights", "stays_in_week_nights",
    "adults", "children", "babies", "previous_cancellations",
    "previous_bookings_not_canceled", "booking_changes", "agent", "company",
    "days_in_waiting_list", "adr", "required_car_parking_spaces",
    "total_of_special_requests"
]

for col in numeric_cols:
    if col in df.columns:
        q995 = df[col].quantile(0.995)
        median = df[col].median()
        mode = df[col].mode()[0]

        if col in ["babies", "children", "previous_cancellations", "previous_bookings_not_canceled",
                   "booking_changes", "required_car_parking_spaces", "total_of_special_requests"]:
            # Use mode for discrete/asymmetric columns
            df.loc[df[col] > q995, col] = mode
            log["outliers_corrected"][col] = f"Replaced {len(df[df[col] > q995])} outliers with mode {mode}"
        else:
            # Use median for continuous columns
            df.loc[df[col] > q995, col] = median
            log["outliers_corrected"][col] = f"Replaced {len(df[df[col] > q995])} outliers with median {median}"

# Ensure all numeric columns are of correct type
for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# Save cleaned dataset
df.to_csv(OUTPUT_PATH, index=False)

# Print log summary
print("=== Data Cleaning Summary ===")
print(f"Initial rows: {initial_rows}")
print(f"Rows after dropping duplicates: {len(df)} (dropped {log['duplicates_dropped']})")
print("\nMissing values imputed:")
for col, count in log["missing_values_imputed"].items():
    print(f"- {col}: {count} values")
print("\nNumeric values extracted from text:")
for col, count in log["numeric_values_extracted"].items():
    print(f"- {col}: {count} values")
print("\nCategorical values harmonized:")
for col, msg in log["categorical_values_harmonized"].items():
    print(f"- {col}: {msg}")
print(f"\nDates standardized: {log['dates_standardized']} values")
print("\nOutliers corrected:")
for col, msg in log["outliers_corrected"].items():
    print(f"- {col}: {msg}")
print(f"\nCleaned dataset saved to: {OUTPUT_PATH}")