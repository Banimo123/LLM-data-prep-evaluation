import pandas as pd
import numpy as np
import re
from datetime import datetime

INPUT_PATH = r"benchmark\datasets\hotel_bookings\noisy_medium.csv"
OUTPUT_PATH = r"results\cleaned_datasets\hotel_bookings\noisy_medium__profile.csv"

def extract_numeric(value):
    if pd.isna(value):
        return np.nan
    s = str(value).replace(",", ".")
    s = s.replace("O", "0").replace("o", "0")
    match = re.search(r"-?\d+\.?\d*", s)
    return float(match.group()) if match else np.nan

def clean_dataset():
    df = pd.read_csv(INPUT_PATH)

    # Log initial
    initial_rows = len(df)
    initial_cols = len(df.columns)
    operations = []

    # 1. Suppression des doublons (en conservant le premier)
    duplicates_before = df.duplicated().sum()
    df = df.drop_duplicates(keep='first')
    if duplicates_before > 0:
        operations.append(f"Suppression de {duplicates_before} doublons")

    # 2. Nettoyage colonne par colonne
    # hotel: pas de manquants, variantes avec espaces à harmoniser
    if 'hotel' in df.columns:
        df['hotel'] = df['hotel'].str.strip()
        df['hotel'] = df['hotel'].replace({'City Hotel  ': 'City Hotel', 'Resoort Hotel': 'Resort Hotel'})

    # is_canceled: déjà propre (0/1)
    # lead_time: contient du texte parasite -> extraction numérique
    if 'lead_time' in df.columns:
        df['lead_time'] = df['lead_time'].apply(extract_numeric)
        # Imputation des valeurs manquantes post-extraction
        if df['lead_time'].isna().sum() > 0:
            median_lead_time = df['lead_time'].median()
            df['lead_time'] = df['lead_time'].fillna(median_lead_time)
            operations.append(f"lead_time: {df['lead_time'].isna().sum()} valeurs manquantes imputées par médiane ({median_lead_time})")

    # arrival_date_year: déjà propre
    # arrival_date_month: harmonisation des noms de mois
    if 'arrival_date_month' in df.columns:
        month_mapping = {
            'January': 'January', 'February': 'February', 'March': 'March',
            'April': 'April', 'May': 'May', 'June': 'June',
            'July': 'July', 'August': 'August', 'September': 'September',
            'October': 'October', 'November': 'November', 'December': 'December'
        }
        df['arrival_date_month'] = df['arrival_date_month'].str.capitalize()
        df['arrival_date_month'] = df['arrival_date_month'].replace(month_mapping)

    # arrival_date_week_number: déjà propre
    # arrival_date_day_of_month: déjà propre
    # stays_in_weekend_nights: déjà propre
    # stays_in_week_nights: valeurs aberrantes (max 999) -> clipping à 30 (99e percentile)
    if 'stays_in_week_nights' in df.columns:
        q99 = df['stays_in_week_nights'].quantile(0.99)
        df['stays_in_week_nights'] = df['stays_in_week_nights'].clip(upper=q99)

    # adults: texte parasite -> extraction numérique
    if 'adults' in df.columns:
        df['adults'] = df['adults'].apply(extract_numeric)
        # Imputation des valeurs manquantes post-extraction
        if df['adults'].isna().sum() > 0:
            mode_adults = df['adults'].mode()[0]
            df['adults'] = df['adults'].fillna(mode_adults)
            operations.append(f"adults: {df['adults'].isna().sum()} valeurs manquantes imputées par mode ({mode_adults})")

    # children: 10% manquants + texte parasite -> extraction numérique puis imputation
    if 'children' in df.columns:
        df['children'] = df['children'].apply(extract_numeric)
        # Imputation des valeurs manquantes
        mode_children = df['children'].mode()[0]
        df['children'] = df['children'].fillna(mode_children)
        operations.append(f"children: {df['children'].isna().sum()} valeurs manquantes imputées par mode ({mode_children})")

    # babies: valeurs aberrantes (max 49) -> clipping à 5
    if 'babies' in df.columns:
        df['babies'] = df['babies'].clip(upper=5)

    # meal: 10% manquants -> imputation par mode
    if 'meal' in df.columns:
        df['meal'] = df['meal'].str.strip()
        df['meal'] = df['meal'].replace({'SC ': 'SC', 'HB ': 'HB'})
        mode_meal = df['meal'].mode()[0]
        df['meal'] = df['meal'].fillna(mode_meal)
        operations.append(f"meal: {df['meal'].isna().sum()} valeurs manquantes imputées par mode ({mode_meal})")

    # country: 10.37% manquants -> imputation par mode
    if 'country' in df.columns:
        df['country'] = df['country'].str.strip().str.upper()
        mode_country = df['country'].mode()[0]
        df['country'] = df['country'].fillna(mode_country)
        operations.append(f"country: {df['country'].isna().sum()} valeurs manquantes imputées par mode ({mode_country})")

    # market_segment: 10% manquants -> imputation par mode
    if 'market_segment' in df.columns:
        df['market_segment'] = df['market_segment'].str.strip()
        mode_market = df['market_segment'].mode()[0]
        df['market_segment'] = df['market_segment'].fillna(mode_market)
        operations.append(f"market_segment: {df['market_segment'].isna().sum()} valeurs manquantes imputées par mode ({mode_market})")

    # distribution_channel: déjà propre
    # is_repeated_guest: déjà propre
    # previous_cancellations: déjà propre
    # previous_bookings_not_canceled: déjà propre
    # reserved_room_type: harmonisation des espaces
    if 'reserved_room_type' in df.columns:
        df['reserved_room_type'] = df['reserved_room_type'].str.strip()

    # assigned_room_type: harmonisation des espaces
    if 'assigned_room_type' in df.columns:
        df['assigned_room_type'] = df['assigned_room_type'].str.strip()

    # booking_changes: déjà propre
    # deposit_type: variantes avec espaces à harmoniser
    if 'deposit_type' in df.columns:
        df['deposit_type'] = df['deposit_type'].str.strip()
        df['deposit_type'] = df['deposit_type'].replace({'NoD eposit': 'No Deposit', 'Non Refund  ': 'Non Refund'})

    # agent: 22.33% manquants -> imputation par mode (colonne discrète)
    if 'agent' in df.columns:
        df['agent'] = df['agent'].apply(extract_numeric)
        mode_agent = df['agent'].mode()[0]
        df['agent'] = df['agent'].fillna(mode_agent)
        operations.append(f"agent: {df['agent'].isna().sum()} valeurs manquantes imputées par mode ({mode_agent})")

    # company: 94.31% manquants -> trop élevé, on conserve les NaN
    # days_in_waiting_list: valeurs aberrantes (max 8996) -> clipping à 365 (1 an)
    if 'days_in_waiting_list' in df.columns:
        df['days_in_waiting_list'] = df['days_in_waiting_list'].clip(upper=365)

    # customer_type: variantes avec espaces à harmoniser
    if 'customer_type' in df.columns:
        df['customer_type'] = df['customer_type'].str.strip()
        df['customer_type'] = df['customer_type'].replace({'Transient-Party ': 'Transient-Party'})

    # adr: valeurs aberrantes (min -493.26, max 9993.79) -> clipping entre 0 et 1000
    if 'adr' in df.columns:
        df['adr'] = df['adr'].clip(lower=0, upper=1000)

    # required_car_parking_spaces: déjà propre
    # total_of_special_requests: déjà propre
    # reservation_status: harmonisation des espaces
    if 'reservation_status' in df.columns:
        df['reservation_status'] = df['reservation_status'].str.strip()

    # reservation_status_date: harmonisation du format de date
    if 'reservation_status_date' in df.columns:
        def parse_date(date_str):
            if pd.isna(date_str):
                return date_str
            try:
                # Essaye plusieurs formats
                for fmt in ('%Y-%m-%d', '%Y/%m/%d', '%d/%m/%Y', '%d-%m-%Y'):
                    try:
                        return datetime.strptime(str(date_str), fmt).strftime('%Y-%m-%d')
                    except ValueError:
                        continue
                return date_str  # Retourne la valeur originale si aucun format ne marche
            except:
                return date_str

        df['reservation_status_date'] = df['reservation_status_date'].apply(parse_date)

    # 3. Conversion des types
    numeric_cols = [
        'is_canceled', 'arrival_date_year', 'arrival_date_week_number',
        'arrival_date_day_of_month', 'stays_in_weekend_nights',
        'stays_in_week_nights', 'adults', 'children', 'babies',
        'is_repeated_guest', 'previous_cancellations',
        'previous_bookings_not_canceled', 'booking_changes',
        'days_in_waiting_list', 'adr', 'required_car_parking_spaces',
        'total_of_special_requests'
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='ignore')

    # 4. Vérification finale
    final_rows = len(df)
    final_cols = len(df.columns)

    # Log des opérations
    print(f"Nettoyage terminé:")
    print(f"- Lignes initiales: {initial_rows}, finales: {final_rows} (suppression de {initial_rows - final_rows} lignes)")
    print(f"- Colonnes initiales: {initial_cols}, finales: {final_cols}")
    print("\nOpérations effectuées:")
    for op in operations:
        print(f"- {op}")
    if duplicates_before > 0:
        print(f"- Suppression de {duplicates_before} doublons")

    # Sauvegarde
    df.to_csv(OUTPUT_PATH, index=False)

if __name__ == "__main__":
    clean_dataset()