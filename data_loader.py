import pandas as pd


def load_data():
    try:
        districts_df = pd.read_csv('data/tehran_disricts_clean.csv', encoding='utf-8')
        print(f"✓ Loaded districts: data/tehran_disricts_clean.csv")
        print(f"  Records: {len(districts_df)}\n")
    except Exception as e:
        print(f" Error loading districts CSV: {e}")
        return None

    try:
        hospitals_df = pd.read_csv('data/cardiac_hospitals_tehran.csv', encoding='utf-8')
        print(f"✓ Loaded hospitals: data/cardiac_hospitals_tehran.csv")
        print(f"  Records: {len(hospitals_df)}\n")
    except Exception as e:
        print(f" Error loading hospitals CSV: {e}")
        return None

    districts_coords = districts_df[['Latitude', 'Longitude']].values
    hospitals_coords = hospitals_df[['Latitude', 'Longitude']].values

    return districts_df, hospitals_df, districts_coords, hospitals_coords
