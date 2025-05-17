# src/preprocessing.py

import pandas as pd

def load_and_preprocess_data(csv_path):
    """
    Load the raw earthquake CSV data, clean it, and engineer features.
    Returns a pandas DataFrame ready for analysis/modeling.
    """
    # Load the dataset
    df = pd.read_csv(csv_path)
    print(f"Original data shape: {df.shape}")
    
    # Convert time to datetime and extract components
    df['time'] = pd.to_datetime(df['time'], errors='coerce')
    df['year'] = df['time'].dt.year
    df['month'] = df['time'].dt.month
    df['day'] = df['time'].dt.day
    df['hour'] = df['time'].dt.hour
    df['weekday'] = df['time'].dt.weekday
    
    # Extract region from 'place' by taking text after comma (if present)
    def extract_region(place):
        if isinstance(place, str) and ',' in place:
            return place.split(',')[-1].strip()
        return place
    df['region'] = df['place'].apply(extract_region)
    
    # Keep only relevant columns for modeling
    relevant_cols = ['latitude', 'longitude', 'depth', 'mag', 'magType',
                     'time', 'year', 'month', 'day', 'hour', 'weekday', 'region']
    df = df[relevant_cols]
    print(f"Selected relevant columns: {relevant_cols}")
    
    # Drop rows with missing values in key columns
    df = df.dropna(subset=['latitude', 'longitude', 'depth', 'mag'])
    print(f"Data shape after dropping missing values: {df.shape}")
    
    # Feature encoding for 'region' (one-hot encoding)
    region_dummies = pd.get_dummies(df['region'], prefix='region')
    df = pd.concat([df, region_dummies], axis=1)
    
    # Example: drop 'place' if it were still in dataframe (we kept only relevant columns already)
    # df = df.drop(columns=['place'])
    
    return df
