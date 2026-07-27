import pandas as pd
import numpy as np
import re
from datetime import datetime

INPUT_PATH = r"benchmark\datasets\hotel_bookings\noisy_high.csv"
OUTPUT_PATH = r"results\cleaned_datasets\hotel_bookings\noisy_high__profile.csv"

def extract_numeric(value):
    if pd.isna(value):
        return np.nan
    s = str(value).replace(",", ".")
    s = s.replace("O", "0").replace("o", "0")
    match = re.search(r"-?\d+\.?\d*", s)
    return float(match.group()) if match else np.nan

def parse_date(value):
    if pd.isna(value):
        return value
    date_formats = [
        "%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%B %d, %Y",
        "%d-%b-%Y", "%Y/%m/%d", "%d-%m-%Y", "%m-%d-%Y"
    ]
    for fmt in date_formats:
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

def clean_hotel_bookings():
    df = pd.read_csv(INPUT_PATH)

    operations_log = []

    # Suppression des doublons (conservation du premier)
    initial_rows = len(df)
    df.drop_duplicates(subset=[col for col in df.columns if col != 'row_id'], keep='first', inplace=True)
    duplicates_removed = initial_rows - len(df)
    if duplicates_removed > 0:
        operations_log.append(f"Doublons supprimés: {duplicates_removed}")

    # Nettoyage colonne par colonne
    # hotel - harmonisation des variantes
    if 'hotel' in df.columns:
        hotel_mapping = {
            'City Hotel': 'City Hotel',
            'Resort Hotel': 'Resort Hotel',
            'CITY hotel': 'City Hotel',
            'CITY HOTEl': 'City Hotel',
            'City Htoel': 'City Hotel',
            'CIty hotel': 'City Hotel',
            'CITY Hotel': 'City Hotel',
            'Ciy Hotel': 'City Hotel',
            'Resort Hotel  ': 'Resort Hotel',
            'CITY hotel  ': 'City Hotel'
        }
        df['hotel'] = df['hotel'].str.strip()
        df['hotel'] = df['hotel'].replace(hotel_mapping)
        operations_log.append("Colonne 'hotel' harmonisée")

    # is_canceled - déjà propre (0/1)
    if 'is_canceled' in df.columns:
        pass

    # lead_time - conversion en numérique
    if 'lead_time' in df.columns:
        df['lead_time'] = df['lead_time'].apply(extract_numeric)
        operations_log.append("Colonne 'lead_time' convertie en numérique")

    # arrival_date_year - déjà propre
    if 'arrival_date_year' in df.columns:
        pass

    # arrival_date_month - harmonisation
    if 'arrival_date_month' in df.columns:
        month_mapping = {
            'January': 'January', 'February': 'February', 'March': 'March',
            'April': 'April', 'May': 'May', 'June': 'June',
            'July': 'July', 'August': 'August', 'September': 'September',
            'October': 'October', 'November': 'November', 'December': 'December'
        }
        df['arrival_date_month'] = df['arrival_date_month'].str.capitalize()
        df['arrival_date_month'] = df['arrival_date_month'].replace(month_mapping)
        operations_log.append("Colonne 'arrival_date_month' harmonisée")

    # arrival_date_week_number - déjà propre
    if 'arrival_date_week_number' in df.columns:
        pass

    # arrival_date_day_of_month - déjà propre
    if 'arrival_date_day_of_month' in df.columns:
        pass

    # stays_in_weekend_nights - valeurs aberrantes (max 19)
    if 'stays_in_weekend_nights' in df.columns:
        median_val = df['stays_in_weekend_nights'].median()
        df['stays_in_weekend_nights'] = df['stays_in_weekend_nights'].apply(
            lambda x: median_val if x > 19 else x
        )
        operations_log.append("Valeurs aberrantes corrigées dans 'stays_in_weekend_nights'")

    # stays_in_week_nights - valeurs aberrantes (max 999)
    if 'stays_in_week_nights' in df.columns:
        median_val = df['stays_in_week_nights'].median()
        df['stays_in_week_nights'] = df['stays_in_week_nights'].apply(
            lambda x: median_val if x > 365 else x  # 1 an max
        )
        operations_log.append("Valeurs aberrantes corrigées dans 'stays_in_week_nights'")

    # adults - nettoyage des valeurs corrompues
    if 'adults' in df.columns:
        df['adults'] = df['adults'].apply(extract_numeric)
        mode_val = df['adults'].mode()[0]
        df['adults'] = df['adults'].apply(lambda x: mode_val if pd.isna(x) or x > 10 else x)
        operations_log.append("Colonne 'adults' nettoyée et valeurs aberrantes corrigées")

    # children - imputation des manquants (33.29%)
    if 'children' in df.columns:
        df['children'] = df['children'].apply(extract_numeric)
        mode_val = df['children'].mode()[0]
        df['children'] = df['children'].fillna(mode_val)
        df['children'] = df['children'].replace('unknown', mode_val)
        operations_log.append("Valeurs manquantes imputées dans 'children'")

    # babies - valeurs aberrantes (max 49)
    if 'babies' in df.columns:
        median_val = df['babies'].median()
        df['babies'] = df['babies'].apply(lambda x: median_val if x > 10 else x)
        operations_log.append("Valeurs aberrantes corrigées dans 'babies'")

    # meal - harmonisation et imputation (33.43% manquants)
    if 'meal' in df.columns:
        meal_mapping = {
            'BB': 'BB', 'HB': 'HB', 'SC': 'SC', 'FB': 'FB',
            'Undefined': 'Undefined', 'No Meal': 'No Meal',
            'undefined': 'Undefined', 'No meal': 'No Meal'
        }
        df['meal'] = df['meal'].str.strip()
        df['meal'] = df['meal'].replace(meal_mapping)
        mode_val = df['meal'].mode()[0]
        df['meal'] = df['meal'].fillna(mode_val)
        operations_log.append("Colonne 'meal' harmonisée et valeurs manquantes imputées")

    # country - imputation (33.94% manquants)
    if 'country' in df.columns:
        mode_val = df['country'].mode()[0]
        df['country'] = df['country'].fillna(mode_val)
        df['country'] = df['country'].replace('unknown', mode_val)
        operations_log.append("Valeurs manquantes imputées dans 'country'")

    # market_segment - harmonisation et imputation (33.35% manquants)
    if 'market_segment' in df.columns:
        segment_mapping = {
            'Online TA': 'Online TA', 'Offline TA/TO': 'Offline TA/TO',
            'Groups': 'Groups', 'Direct': 'Direct', 'Corporate': 'Corporate',
            'Complementary': 'Complementary', 'Aviation': 'Aviation',
            'Undefined': 'Undefined'
        }
        df['market_segment'] = df['market_segment'].str.strip()
        df['market_segment'] = df['market_segment'].replace(segment_mapping)
        mode_val = df['market_segment'].mode()[0]
        df['market_segment'] = df['market_segment'].fillna(mode_val)
        operations_log.append("Colonne 'market_segment' harmonisée et valeurs manquantes imputées")

    # distribution_channel - déjà propre
    if 'distribution_channel' in df.columns:
        pass

    # is_repeated_guest - déjà propre
    if 'is_repeated_guest' in df.columns:
        pass

    # previous_cancellations - valeurs aberrantes (max 26)
    if 'previous_cancellations' in df.columns:
        median_val = df['previous_cancellations'].median()
        df['previous_cancellations'] = df['previous_cancellations'].apply(
            lambda x: median_val if x > 20 else x
        )
        operations_log.append("Valeurs aberrantes corrigées dans 'previous_cancellations'")

    # previous_bookings_not_canceled - valeurs aberrantes (max 72)
    if 'previous_bookings_not_canceled' in df.columns:
        median_val = df['previous_bookings_not_canceled'].median()
        df['previous_bookings_not_canceled'] = df['previous_bookings_not_canceled'].apply(
            lambda x: median_val if x > 50 else x
        )
        operations_log.append("Valeurs aberrantes corrigées dans 'previous_bookings_not_canceled'")

    # reserved_room_type - harmonisation
    if 'reserved_room_type' in df.columns:
        room_types = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'P']
        df['reserved_room_type'] = df['reserved_room_type'].str.strip().str.upper()
        df['reserved_room_type'] = df['reserved_room_type'].apply(
            lambda x: x if x in room_types else 'A'
        )
        operations_log.append("Colonne 'reserved_room_type' harmonisée")

    # assigned_room_type - harmonisation
    if 'assigned_room_type' in df.columns:
        room_types = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'P']
        df['assigned_room_type'] = df['assigned_room_type'].str.strip().str.upper()
        df['assigned_room_type'] = df['assigned_room_type'].apply(
            lambda x: x if x in room_types else 'A'
        )
        operations_log.append("Colonne 'assigned_room_type' harmonisée")

    # booking_changes - valeurs aberrantes (max 21)
    if 'booking_changes' in df.columns:
        median_val = df['booking_changes'].median()
        df['booking_changes'] = df['booking_changes'].apply(
            lambda x: median_val if x > 10 else x
        )
        operations_log.append("Valeurs aberrantes corrigées dans 'booking_changes'")

    # deposit_type - harmonisation
    if 'deposit_type' in df.columns:
        deposit_mapping = {
            'No Deposit': 'No Deposit', 'Non Refund': 'Non Refund',
            'Refundable': 'Refundable', 'No Depoosit': 'No Deposit',
            'No DDeposit': 'No Deposit', 'N Deposit': 'No Deposit',
            'No Deeposit': 'No Deposit'
        }
        df['deposit_type'] = df['deposit_type'].str.strip()
        df['deposit_type'] = df['deposit_type'].replace(deposit_mapping)
        operations_log.append("Colonne 'deposit_type' harmonisée")

    # agent - imputation (55.24% manquants)
    if 'agent' in df.columns:
        mode_val = df['agent'].mode()[0]
        df['agent'] = df['agent'].fillna(mode_val)
        df['agent'] = df['agent'].replace('unknown', mode_val)
        operations_log.append("Valeurs manquantes imputées dans 'agent'")

    # company - trop de manquants (94.31%) -> suppression
    if 'company' in df.columns:
        df.drop('company', axis=1, inplace=True)
        operations_log.append("Colonne 'company' supprimée (trop de valeurs manquantes)")

    # days_in_waiting_list - valeurs aberrantes (max 8999)
    if 'days_in_waiting_list' in df.columns:
        median_val = df['days_in_waiting_list'].median()
        df['days_in_waiting_list'] = df['days_in_waiting_list'].apply(
            lambda x: median_val if x > 365 else x  # 1 an max
        )
        operations_log.append("Valeurs aberrantes corrigées dans 'days_in_waiting_list'")

    # customer_type - harmonisation
    if 'customer_type' in df.columns:
        customer_mapping = {
            'Transient': 'Transient', 'Transient-Party': 'Transient-Party',
            'Contract': 'Contract', 'Group': 'Group',
            'Transiient': 'Transient', 'Trnasient': 'Transient',
            'TRANSient': 'Transient', 'TTransient': 'Transient'
        }
        df['customer_type'] = df['customer_type'].str.strip()
        df['customer_type'] = df['customer_type'].replace(customer_mapping)
        operations_log.append("Colonne 'customer_type' harmonisée")

    # adr - valeurs aberrantes (min -496.7, max 9997.01)
    if 'adr' in df.columns:
        median_val = df['adr'].median()
        df['adr'] = df['adr'].apply(
            lambda x: median_val if x < 0 or x > 1000 else x
        )
        operations_log.append("Valeurs aberrantes corrigées dans 'adr'")

    # required_car_parking_spaces - valeurs aberrantes (max 8)
    if 'required_car_parking_spaces' in df.columns:
        median_val = df['required_car_parking_spaces'].median()
        df['required_car_parking_spaces'] = df['required_car_parking_spaces'].apply(
            lambda x: median_val if x > 5 else x
        )
        operations_log.append("Valeurs aberrantes corrigées dans 'required_car_parking_spaces'")

    # total_of_special_requests - valeurs aberrantes (max 5)
    if 'total_of_special_requests' in df.columns:
        median_val = df['total_of_special_requests'].median()
        df['total_of_special_requests'] = df['total_of_special_requests'].apply(
            lambda x: median_val if x > 5 else x
        )
        operations_log.append("Valeurs aberrantes corrigées dans 'total_of_special_requests'")

    # reservation_status - harmonisation
    if 'reservation_status' in df.columns:
        status_mapping = {
            'Check-Out': 'Check-Out', 'Canceled': 'Canceled',
            'No-Show': 'No-Show', 'Check Out': 'Check-Out',
            'Cancel': 'Canceled'
        }
        df['reservation_status'] = df['reservation_status'].str.strip()
        df['reservation_status'] = df['reservation_status'].replace(status_mapping)
        operations_log.append("Colonne 'reservation_status' harmonisée")

    # reservation_status_date - conversion en format standard
    if 'reservation_status_date' in df.columns:
        df['reservation_status_date'] = df['reservation_status_date'].apply(parse_date)
        operations_log.append("Colonne 'reservation_status_date' convertie en format standard")

    # Affichage du log des opérations
    print("Résumé des opérations de nettoyage:")
    for op in operations_log:
        print(f"- {op}")
    print(f"\nDataset final: {len(df)} lignes, {len(df.columns)} colonnes")

    # Sauvegarde
    df.to_csv(OUTPUT_PATH, index=False)

if __name__ == "__main__":
    clean_hotel_bookings()