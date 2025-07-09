#!/usr/bin/env python3
"""
CLEAR-E Dataset Preprocessing Pipeline
Comprehensive preprocessing for ECL, GEFCom2014, and Southern China datasets
"""

import os
import pandas as pd
import numpy as np
import sqlite3
import zipfile
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class DatasetPreprocessor:
    """Unified preprocessing pipeline for CLEAR-E datasets"""
    
    def __init__(self, data_dir=".", output_dir="processed"):
        self.data_dir = data_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def preprocess_all_datasets(self):
        """Preprocess all datasets for CLEAR-E experiments"""
        print("=" * 60)
        print("CLEAR-E Dataset Preprocessing Pipeline")
        print("=" * 60)
        
        # Process each dataset
        datasets = {}
        
        print("\n1. Processing Southern China Dataset...")
        datasets['southern_china'] = self.preprocess_southern_china()
        
        print("\n2. Processing ECL Dataset...")
        datasets['ecl'] = self.preprocess_ecl()
        
        print("\n3. Processing GEFCom2014 Dataset...")
        datasets['gefcom2014'] = self.preprocess_gefcom2014()
        
        # Generate summary statistics
        print("\n4. Generating Dataset Summary...")
        self.generate_summary(datasets)
        
        print("\n" + "=" * 60)
        print("Preprocessing Complete!")
        print(f"Processed datasets saved in: {self.output_dir}/")
        print("=" * 60)
        
        return datasets
    
    def preprocess_southern_china(self):
        """Preprocess Southern China electricity dataset"""
        print("  Loading Southern China data from SQLite database...")
        
        # Connect to database
        db_path = os.path.join(self.data_dir, "SouthernChina_data", "Transformer_DB.db")
        conn = sqlite3.connect(db_path)
        
        # Load transformer data
        transformer_query = """
        SELECT TRANSFORMER_ID, LOAD, DATETIME 
        FROM transformer_raw 
        ORDER BY DATETIME, TRANSFORMER_ID
        """
        transformer_df = pd.read_sql_query(transformer_query, conn)
        transformer_df['DATETIME'] = pd.to_datetime(transformer_df['DATETIME'])
        
        # Load weather data
        weather_query = """
        SELECT DATETIME, TEMP, DEWP, RH, WDSP, PRCP, MAX, MIN, STATION_ID
        FROM weather 
        ORDER BY DATETIME, STATION_ID
        """
        weather_df = pd.read_sql_query(weather_query, conn)
        weather_df['DATETIME'] = pd.to_datetime(weather_df['DATETIME'])
        
        # Load holiday data
        holiday_query = "SELECT * FROM holiday"
        holiday_df = pd.read_sql_query(holiday_query, conn)
        
        # Load extreme weather events
        extreme_query = "SELECT * FROM extreme_weather_calculated"
        extreme_df = pd.read_sql_query(extreme_query, conn)
        
        conn.close()
        
        print(f"    Loaded {len(transformer_df)} transformer records")
        print(f"    Loaded {len(weather_df)} weather records")
        print(f"    Date range: {transformer_df['DATETIME'].min()} to {transformer_df['DATETIME'].max()}")
        
        # Aggregate transformer data by hour
        print("  Aggregating transformer data...")
        transformer_df['hour'] = transformer_df['DATETIME'].dt.floor('H')
        hourly_load = transformer_df.groupby(['hour', 'TRANSFORMER_ID'])['LOAD'].mean().reset_index()
        
        # Pivot to get transformers as columns
        load_pivot = hourly_load.pivot(index='hour', columns='TRANSFORMER_ID', values='LOAD')
        load_pivot = load_pivot.ffill().bfill()
        
        # Aggregate weather data by hour and station
        print("  Processing weather data...")
        weather_df['hour'] = weather_df['DATETIME'].dt.floor('H')
        hourly_weather = weather_df.groupby(['hour', 'STATION_ID']).agg({
            'TEMP': 'mean',
            'DEWP': 'mean', 
            'RH': 'mean',
            'WDSP': 'mean',
            'PRCP': 'sum',
            'MAX': 'max',
            'MIN': 'min'
        }).reset_index()
        
        # Average across all weather stations
        weather_avg = hourly_weather.groupby('hour').agg({
            'TEMP': 'mean',
            'DEWP': 'mean',
            'RH': 'mean', 
            'WDSP': 'mean',
            'PRCP': 'mean',
            'MAX': 'mean',
            'MIN': 'mean'
        }).reset_index()
        
        # Merge load and weather data
        print("  Merging load and weather data...")
        merged_df = load_pivot.reset_index().merge(weather_avg, left_on='hour', right_on='hour', how='inner')
        merged_df = merged_df.set_index('hour').sort_index()
        
        # Add calendar features
        print("  Adding calendar features...")
        merged_df = self.add_calendar_features(merged_df)
        
        # Add holiday information
        if not holiday_df.empty and 'date' in holiday_df.columns:
            holiday_df['date'] = pd.to_datetime(holiday_df['date'])
            holiday_dates = set(holiday_df['date'].dt.date)
            merged_df['is_holiday'] = merged_df.index.date.isin(holiday_dates).astype(int)
        else:
            merged_df['is_holiday'] = 0
        
        # Clean and validate data
        print("  Cleaning and validating data...")
        merged_df = self.clean_data(merged_df)
        
        # Save processed data
        output_path = os.path.join(self.output_dir, "southern_china_processed.csv")
        merged_df.to_csv(output_path)
        print(f"  Saved to: {output_path}")
        print(f"  Final shape: {merged_df.shape}")
        
        return merged_df
    
    def preprocess_ecl(self):
        """Preprocess ECL dataset"""
        print("  Extracting ECL data...")
        
        # Extract ZIP file
        zip_path = os.path.join(self.data_dir, "ECL_data.zip")
        extract_path = os.path.join(self.data_dir, "ECL_extracted")
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
        
        # Find the main data file
        data_files = []
        for root, dirs, files in os.walk(extract_path):
            for file in files:
                if file.endswith('.csv') or file.endswith('.txt'):
                    data_files.append(os.path.join(root, file))
        
        print(f"    Found {len(data_files)} data files")
        
        # Load the main dataset (usually the largest file)
        main_file = max(data_files, key=os.path.getsize)
        print(f"    Loading main file: {os.path.basename(main_file)}")
        
        # Try different separators and encodings
        df = None
        for sep in [';', ',', '\t']:
            for encoding in ['utf-8', 'latin-1', 'cp1252']:
                try:
                    df = pd.read_csv(main_file, sep=sep, encoding=encoding, low_memory=False)
                    if df.shape[1] > 1:  # Valid if more than 1 column
                        break
                except:
                    continue
            if df is not None and df.shape[1] > 1:
                break
        
        print(f"    Loaded ECL data: {df.shape}")
        
        # Process ECL data (adapt based on actual structure)
        df = self.process_ecl_structure(df)
        
        # Add calendar features
        df = self.add_calendar_features(df)
        
        # Clean data
        df = self.clean_data(df)
        
        # Save processed data
        output_path = os.path.join(self.output_dir, "ecl_processed.csv")
        df.to_csv(output_path)
        print(f"  Saved to: {output_path}")
        print(f"  Final shape: {df.shape}")
        
        return df
    
    def preprocess_gefcom2014(self):
        """Preprocess GEFCom2014 dataset"""
        print("  Extracting GEFCom2014 data...")
        
        # Extract ZIP file
        zip_path = os.path.join(self.data_dir, "GEFCom2014_data.zip")
        extract_path = os.path.join(self.data_dir, "GEFCom2014_extracted")
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
        
        # Find data files
        data_files = []
        for root, dirs, files in os.walk(extract_path):
            for file in files:
                if file.endswith('.csv') or file.endswith('.txt'):
                    data_files.append(os.path.join(root, file))
        
        print(f"    Found {len(data_files)} data files")
        
        # Load and combine files
        dfs = []
        for file_path in data_files:
            try:
                df_temp = pd.read_csv(file_path, low_memory=False)
                dfs.append(df_temp)
                print(f"    Loaded: {os.path.basename(file_path)} - {df_temp.shape}")
            except Exception as e:
                print(f"    Error loading {file_path}: {e}")
        
        # Combine all dataframes
        if dfs:
            df = pd.concat(dfs, ignore_index=True)
        else:
            raise ValueError("No valid GEFCom2014 data files found")
        
        print(f"    Combined GEFCom2014 data: {df.shape}")
        
        # Process GEFCom2014 structure
        df = self.process_gefcom_structure(df)
        
        # Add calendar features
        df = self.add_calendar_features(df)
        
        # Clean data
        df = self.clean_data(df)
        
        # Save processed data
        output_path = os.path.join(self.output_dir, "gefcom2014_processed.csv")
        df.to_csv(output_path)
        print(f"  Saved to: {output_path}")
        print(f"  Final shape: {df.shape}")
        
        return df

    def process_ecl_structure(self, df):
        """Process ECL dataset structure"""
        print("    Processing ECL structure...")

        # ECL dataset has datetime in first column and load values in subsequent columns
        # First column should be datetime
        datetime_col = df.columns[0]

        # Convert first column to datetime
        try:
            df['datetime'] = pd.to_datetime(df[datetime_col])
        except:
            # If direct conversion fails, try to parse manually
            df['datetime'] = pd.to_datetime(df[datetime_col], errors='coerce')

        # Drop rows with invalid datetime
        df = df.dropna(subset=['datetime'])

        # Set datetime as index
        df = df.set_index('datetime').sort_index()

        # Remove the original datetime column if it still exists
        if datetime_col in df.columns:
            df = df.drop(columns=[datetime_col])

        # Keep only numeric columns (load data)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df = df[numeric_cols]

        # Handle missing values
        df = df.ffill().bfill()

        return df

    def process_gefcom_structure(self, df):
        """Process GEFCom2014 dataset structure"""
        print("    Processing GEFCom2014 structure...")

        # GEFCom2014 typically has datetime, zone, load, and weather columns
        # Standardize column names
        col_mapping = {
            'Date': 'datetime',
            'ZONEID': 'zone',
            'LOAD': 'load',
            'TEMP': 'temperature',
            'DEWP': 'dewpoint'
        }

        for old_col, new_col in col_mapping.items():
            if old_col in df.columns:
                df = df.rename(columns={old_col: new_col})

        # Create datetime if not exists
        if 'datetime' not in df.columns:
            if 'Year' in df.columns and 'Month' in df.columns and 'Day' in df.columns:
                df['datetime'] = pd.to_datetime(df[['Year', 'Month', 'Day', 'Hour']])
            else:
                df['datetime'] = pd.to_datetime(df.iloc[:, 0])
        else:
            df['datetime'] = pd.to_datetime(df['datetime'])

        # Set datetime as index
        df = df.set_index('datetime').sort_index()

        # Aggregate by zone if multiple zones exist
        if 'zone' in df.columns:
            # Pivot zones to columns
            load_cols = [col for col in df.columns if 'load' in col.lower()]
            if load_cols:
                df_pivot = df.pivot_table(index=df.index, columns='zone', values=load_cols[0], aggfunc='mean')
                df_pivot.columns = [f'zone_{col}' for col in df_pivot.columns]

                # Add weather data (average across zones)
                weather_cols = [col for col in df.columns if col not in ['zone'] + load_cols]
                weather_avg = df.groupby(df.index)[weather_cols].mean()

                df = pd.concat([df_pivot, weather_avg], axis=1)

        # Handle missing values
        df = df.ffill().bfill()

        return df

    def add_calendar_features(self, df):
        """Add calendar-based features"""
        print("    Adding calendar features...")

        # Ensure index is datetime
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

        # Basic time features
        df['hour'] = df.index.hour
        df['day_of_week'] = df.index.dayofweek
        df['day_of_year'] = df.index.dayofyear
        df['month'] = df.index.month
        df['quarter'] = df.index.quarter
        df['year'] = df.index.year

        # Cyclical encoding for time features
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

        # Weekend indicator
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

        # Season indicator
        df['season'] = df['month'].map({12: 0, 1: 0, 2: 0,  # Winter
                                       3: 1, 4: 1, 5: 1,   # Spring
                                       6: 2, 7: 2, 8: 2,   # Summer
                                       9: 3, 10: 3, 11: 3}) # Fall

        return df

    def clean_data(self, df):
        """Clean and validate dataset"""
        print("    Cleaning data...")

        # Remove infinite values
        df = df.replace([np.inf, -np.inf], np.nan)

        # Handle missing values
        df = df.ffill().bfill()

        # Remove outliers (values beyond 3 standard deviations)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col not in ['hour', 'day_of_week', 'month', 'year', 'is_weekend', 'is_holiday', 'season']:
                mean_val = df[col].mean()
                std_val = df[col].std()
                df[col] = df[col].clip(lower=mean_val - 3*std_val, upper=mean_val + 3*std_val)

        # Ensure no negative load values
        load_cols = [col for col in df.columns if any(keyword in col.lower() for keyword in ['load', 'consumption', 'demand'])]
        for col in load_cols:
            df[col] = df[col].clip(lower=0)

        return df

    def generate_summary(self, datasets):
        """Generate summary statistics for all datasets"""
        print("  Generating summary statistics...")

        summary_data = []
        for name, df in datasets.items():
            if df is not None:
                load_cols = [col for col in df.columns if any(keyword in col.lower()
                           for keyword in ['load', 'consumption', 'demand']) and
                           not any(exclude in col.lower() for exclude in ['hour', 'day', 'month'])]

                weather_cols = [col for col in df.columns if any(keyword in col.lower()
                              for keyword in ['temp', 'humidity', 'wind', 'prcp', 'rh'])]

                summary_data.append({
                    'Dataset': name.replace('_', ' ').title(),
                    'Shape': f"{df.shape[0]} × {df.shape[1]}",
                    'Date Range': f"{df.index.min().strftime('%Y-%m-%d')} to {df.index.max().strftime('%Y-%m-%d')}",
                    'Load Columns': len(load_cols),
                    'Weather Columns': len(weather_cols),
                    'Missing Values': df.isnull().sum().sum(),
                    'Avg Load': f"{df[load_cols].mean().mean():.2f}" if load_cols else "N/A"
                })

        summary_df = pd.DataFrame(summary_data)
        summary_path = os.path.join(self.output_dir, "dataset_summary.csv")
        summary_df.to_csv(summary_path, index=False)

        print("\n  Dataset Summary:")
        print(summary_df.to_string(index=False))
        print(f"\n  Summary saved to: {summary_path}")

def main():
    """Main preprocessing function"""
    preprocessor = DatasetPreprocessor()
    datasets = preprocessor.preprocess_all_datasets()
    return datasets

if __name__ == "__main__":
    datasets = main()
