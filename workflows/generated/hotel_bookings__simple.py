import pandas as pd
import numpy as np
import re
from datetime import datetime

INPUT_PATH = "benchmark/datasets/hotel_bookings/noisy_low.csv"
OUTPUT_PATH = "results/cleaned_datasets/hotel_bookings/noisy_low__simple.csv"

df = pd.read_csv(INPUT_PATH, error_bad_lines=False)

# Corriger les dates
def correct_date(s):
    if s.notnull():
        try:
            return datetime.strptime(s, "%Y%m%d")
        except ValueError:
            pass
    return np.nan

df['arrival_date'] = df['arrival_date'].apply(correct_date)
df['reservation_status_date'] = df['reservation_status_date'].apply(correct_date)

# Corriger les valeurs manquantes
def fill_missing_values(s):
    if s.isnull().all():
        return s.apply(lambda x: re.sub('[^a-zA-Z ]+', '', x))
    return s

df['hotel'] = fill_missing_values(df['hotel'])
df['country'] = fill_missing_values(df['country'])
df['market_segment'] = fill_missing_values(df['market_segment'])
df['distribution_channel'] = fill_missing_values(df['distribution_channel'])
df['customer_type'] = fill_missing_values(df['customer_type'])

# Supprimer les doublons
df.drop_duplicates(inplace=True)

# Standardiser les catégories
df['company'].replace(['C', 'unknown'], ['Corporate'], inplace=True)
df['agent'].replace(['C', 'PRT'], ['Direct'], inplace=True)
df['meal'].replace(['BB'], ['Breakfast'], inplace=True)
df['customer_type'].replace(['Transient'], ['Individual'], inplace=True)

# Corriger les valeurs aberrantes ou incohérentes
df['stays_in_weekend_nights'] = df['stays_in_weekend_nights'].apply(lambda x: 0 if np.isnan(x) else int(x))
df['stays_in_week_nights'] = df['stays_in_week_nights'].apply(lambda x: 0 if np.isnan(x) or x > 7 else int(x))
df['adults'] = df['adults'].apply(lambda x: 0 if np.isnan(x) else int(x))
df['children'] = df['children'].apply(lambda x: 0 if np.isnan(x) or x > 10 else int(x))
df['babies'] = df['babies'].apply(lambda x: 0 if np.isnan(x) or x > 5 else int(x))
df['days_in_waiting_list'] = df['days_in_waiting_list'].apply(lambda x: 0 if np.isnull(x) else float(x))
df['customer_type_price'] = df['customer_type'].map({'Individual': 75, 'Corporate': 98})
df['required_car_parking_spaces'] = df['required_car_parking_spaces'].apply(lambda x: 0 if np.isnull(x) else int(x))
df['total_of_special_requests'] = df['total_of_special_requests'].apply(lambda x: 0 if np.isnull(x) else float(x))

# Sauvegarder le dataset nettoyé
df.to_csv(OUTPUT_PATH, index=False)

# Log minimal
print(f"Nombre de valeurs manquantes traitées : {sum(pd.isna(df).sum())}")
print(f"Nombre de doublons supprimés : {len(df) - len(df.drop_duplicates())}")
print(f"Nombre de lignes/colonnes modifiées : {sum([df.isnull().sum()].sum()) + df.shape[1] - len(df.columns)}")