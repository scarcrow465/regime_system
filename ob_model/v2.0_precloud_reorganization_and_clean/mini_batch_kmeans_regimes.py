#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from tqdm import tqdm

# Centralized parameters - Adjust these as needed
TEST_SLICE = None  # Number of rows to use from the end of the dataset (set to None for full dataset)
K_CLUSTERS = 5  # Number of clusters for MiniBatchKMeans
BATCH_SIZE = 100  # Batch size for MiniBatchKMeans
LOOKBACK = 200  # Lookback period for rolling calculations
DATA_FILE = 'combined_NQ_15m_data.csv'  # Path to your CSV file
OUTPUT_FILE = 'regime_labeled_data_3_full.csv'  # Output CSV file name
TIMEFRAME = '15min'  # Timeframe for data loading

# Simple data loading function (assuming CSV with Date, open, high, low, close columns)
def load_csv_data(file_path, timeframe):
    print(f"Loading data from {file_path}")
    df = pd.read_csv(file_path, parse_dates=['Date'])
    df.set_index('Date', inplace=True)
    df = df.sort_index()
    print(f"Loaded {len(df)} rows")
    return df

# Main function to run the analysis
def main():
    # Load data with tqdm progress
    with tqdm(total=1, desc="Loading Data", ncols=80) as pbar:
        data = load_csv_data(DATA_FILE, TIMEFRAME)
        pbar.update(1)

    # Slice data if test_slice is set
    if TEST_SLICE is not None:
        data = data.tail(TEST_SLICE)
        print(f"Using last {TEST_SLICE} rows for testing")

    # Calculate ATR if not present (assuming TR is True Range, but we'll compute it)
    data['TR'] = np.maximum.reduce([data['high'] - data['low'], abs(data['high'] - data['close'].shift(1)), abs(data['low'] - data['close'].shift(1))])
    data['ATR'] = data['TR'].rolling(window=5).mean()

    # Calculate features with tqdm progress
    with tqdm(total=1, desc="Calculating Features", ncols=80) as pbar:
        data['Price_Change'] = (data['close'] - data['close'].shift(3)) / data['ATR']
        data['Volatility_Ratio'] = data['ATR'] / data['ATR'].rolling(window=50).mean()
        data['Price_Range'] = (data['high'].rolling(window=10).max() - data['low'].rolling(window=10).min()) / data['ATR']
        data['Momentum'] = (data['close'] - data['close'].shift(5)) / data['close'].shift(5)
        data['Vol_Change'] = data['ATR'].diff(5)
        features = data[['Price_Change', 'Volatility_Ratio', 'Price_Range', 'Momentum', 'Vol_Change']].dropna()
        features = (features - features.mean()) / features.std()  # Normalize features
        pbar.update(1)

    # Run MiniBatchKMeans with tqdm progress
    with tqdm(total=1, desc="Running Clustering", ncols=80) as pbar:
        model = MiniBatchKMeans(n_clusters=K_CLUSTERS, batch_size=BATCH_SIZE, random_state=42)
        data['Regime'] = np.nan
        data.loc[features.index, 'Regime'] = model.fit_predict(features)
        pbar.update(1)

    # Map clusters to regime labels (inspect centroids to assign)
    centroids = pd.DataFrame(model.cluster_centers_, columns=features.columns)
    # Example mapping: Sort by Price_Change for labeling
    centroids_sorted = centroids.sort_values('Price_Change')
    regime_map = {
        centroids_sorted.index[0]: "Bear Breakout",
        centroids_sorted.index[1]: "Consolidation",
        centroids_sorted.index[2]: "Neutral",
        centroids_sorted.index[3]: "Post-Bull",
        centroids_sorted.index[4]: "Bull Breakout"
    }
    data['Regime_Label'] = data['Regime'].map(regime_map)

    # Add numbered index for continuous plotting (ignores time gaps)
    data['Index'] = range(len(data))

    # Reset index to include Date as column
    data = data.reset_index()

    # Split Date (datetime) into separate Date and Time columns
    data['Date_Separate'] = data['Date'].dt.date
    data['Time'] = data['Date'].dt.time

    # Export to CSV with separate Date and Time, and Index
    output_columns = ['Index', 'Date_Separate', 'Time', 'open', 'high', 'low', 'close', 'Regime_Label']  # Add Volume if present
    data[output_columns].to_csv(OUTPUT_FILE, index=False)
    print(f"Exported labeled data to {OUTPUT_FILE}")

    # Print regime distribution
    print("\nRegime Distribution:")
    regime_counts = data['Regime_Label'].value_counts()
    total = len(data)
    for regime, count in regime_counts.items():
        percentage = (count / total) * 100
        print(f"{regime}: {count} ({percentage:.2f}%)")

    # Print centroids
    print("\nCluster Centroids:")
    print(centroids)

if __name__ == "__main__":
    main()

