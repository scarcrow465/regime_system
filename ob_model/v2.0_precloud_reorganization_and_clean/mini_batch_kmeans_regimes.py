#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from tqdm import tqdm

# Centralized parameters - Adjust these as needed
TEST_SLICE = 10000  # Number of rows to use from the end of the dataset (set to None for full dataset)
K_CLUSTERS = 4  # Reduced to 4 base clusters (Bull Breakout, Bear Breakout, Consolidation, Neutral)
BATCH_SIZE = 100  # Batch size for MiniBatchKMeans
LOOKBACK = 200  # Base lookback period for rolling calculations
DATA_FILE = 'combined_NQ_15m_data.csv'  # Path to your CSV file
OUTPUT_FILE = 'regime_labeled_data.csv'  # Output CSV file name
TIMEFRAME = '15min'  # Timeframe for data loading
MIN_REGIME_PERSISTENCE = 2  # Minimum bars for regime confirmation

# Simple data loading function (assuming CSV with Date, open, high, low, close columns)
def load_csv_data(file_path, timeframe):
    print(f"Loading data from {file_path}")
    df = pd.read_csv(file_path, parse_dates=['Date'])
    df.set_index('Date', inplace=True)
    df = df.sort_index()
    print(f"Loaded {len(df)} rows")
    return df

# Calculate adaptive lookback periods based on volatility
def calculate_adaptive_features(data):
    """Calculate features with adaptive lookback periods based on market volatility"""
    
    # Base volatility measure for adaptive scaling
    data['Volatility_Regime'] = data['ATR'] / data['ATR'].rolling(window=100).mean()
    
    # Adaptive lookback: shorter in high vol, longer in low vol
    data['Adaptive_Short'] = np.where(data['Volatility_Regime'] > 1.5, 3, 
                                     np.where(data['Volatility_Regime'] < 0.7, 8, 5))
    data['Adaptive_Medium'] = np.where(data['Volatility_Regime'] > 1.5, 8, 
                                      np.where(data['Volatility_Regime'] < 0.7, 20, 12))
    data['Adaptive_Long'] = np.where(data['Volatility_Regime'] > 1.5, 20, 
                                    np.where(data['Volatility_Regime'] < 0.7, 50, 30))
    
    # Multi-timeframe price changes (using adaptive lookbacks)
    data['Price_Change_Short'] = (data['close'] - data['close'].shift(data['Adaptive_Short'])) / data['ATR']
    data['Price_Change_Medium'] = (data['close'] - data['close'].shift(data['Adaptive_Medium'])) / data['ATR']
    data['Price_Change_Long'] = (data['close'] - data['close'].shift(data['Adaptive_Long'])) / data['ATR']
    
    # Multi-timeframe volatility ratios
    data['Vol_Ratio_Short'] = data['ATR'].rolling(window=5).mean() / data['ATR'].rolling(window=20).mean()
    data['Vol_Ratio_Medium'] = data['ATR'].rolling(window=20).mean() / data['ATR'].rolling(window=50).mean()
    
    # Adaptive range calculations
    adaptive_range_window = data['Adaptive_Medium'].fillna(method='ffill').astype(int)
    data['Price_Range_Adaptive'] = np.nan
    for i in range(len(data)):
        if i >= adaptive_range_window.iloc[i]:
            window = int(adaptive_range_window.iloc[i])
            high_max = data['high'].iloc[i-window:i+1].max()
            low_min = data['low'].iloc[i-window:i+1].min()
            data.loc[data.index[i], 'Price_Range_Adaptive'] = (high_max - low_min) / data['ATR'].iloc[i]
    
    # Momentum persistence (trend consistency)
    data['Momentum_Short'] = (data['close'] - data['close'].shift(data['Adaptive_Short'])) / data['close'].shift(data['Adaptive_Short'])
    data['Momentum_Medium'] = (data['close'] - data['close'].shift(data['Adaptive_Medium'])) / data['close'].shift(data['Adaptive_Medium'])
    
    # Breakout strength indicators
    data['Breakout_Strength'] = np.abs(data['Price_Change_Medium']) * data['Vol_Ratio_Short']
    
    # Consolidation tightness (lower values = tighter consolidation)
    data['Consolidation_Tightness'] = data['Price_Range_Adaptive'] / data['Volatility_Regime']
    
    return data

# Apply expanding window normalization to avoid lookahead bias
def expanding_normalize(series, min_periods=50):
    """Normalize using expanding window to avoid lookahead bias"""
    expanding_mean = series.expanding(min_periods=min_periods).mean()
    expanding_std = series.expanding(min_periods=min_periods).std()
    return (series - expanding_mean) / expanding_std

# Apply temporal logic for post-breakout regimes
def apply_temporal_logic(data):
    """Apply temporal sequencing for Post-Bull and Post-Bear regimes"""
    
    # Initialize regime tracking
    data['Base_Regime'] = data['Regime_Label'].copy()
    data['Final_Regime'] = data['Regime_Label'].copy()
    
    # Track last breakout type
    last_breakout = None
    post_breakout_counter = 0
    
    for i in range(len(data)):
        current_regime = data['Base_Regime'].iloc[i]
        
        # Check for new breakouts
        if current_regime == 'Bull Breakout':
            last_breakout = 'Bull'
            post_breakout_counter = 0
        elif current_regime == 'Bear Breakout':
            last_breakout = 'Bear'
            post_breakout_counter = 0
        
        # Apply post-breakout logic
        elif current_regime in ['Consolidation', 'Neutral'] and last_breakout is not None:
            post_breakout_counter += 1
            
            # Convert to post-breakout regime if within reasonable timeframe
            if post_breakout_counter <= 50:  # Max 50 bars after breakout
                if last_breakout == 'Bull':
                    data.loc[data.index[i], 'Final_Regime'] = 'Post-Bull'
                elif last_breakout == 'Bear':
                    data.loc[data.index[i], 'Final_Regime'] = 'Post-Bear'
        
        # Reset if we get a strong opposite signal
        elif (current_regime == 'Bear Breakout' and last_breakout == 'Bull') or \
             (current_regime == 'Bull Breakout' and last_breakout == 'Bear'):
            last_breakout = 'Bull' if current_regime == 'Bull Breakout' else 'Bear'
            post_breakout_counter = 0
    
    return data

# Add regime persistence confirmation
def add_regime_persistence(data, min_persistence=MIN_REGIME_PERSISTENCE):
    """Add regime persistence to avoid rapid switching"""
    
    data['Confirmed_Regime'] = data['Final_Regime'].copy()
    
    # Apply persistence filter
    for i in range(min_persistence, len(data)):
        recent_regimes = data['Final_Regime'].iloc[i-min_persistence:i+1]
        
        # If regime has been consistent for min_persistence periods, confirm it
        if len(recent_regimes.unique()) == 1:
            data.loc[data.index[i], 'Confirmed_Regime'] = recent_regimes.iloc[-1]
        else:
            # Keep previous confirmed regime if no consistency
            if i > 0:
                data.loc[data.index[i], 'Confirmed_Regime'] = data['Confirmed_Regime'].iloc[i-1]
    
    return data

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

    # Calculate ATR if not present
    data['TR'] = np.maximum.reduce([data['high'] - data['low'], 
                                   abs(data['high'] - data['close'].shift(1)), 
                                   abs(data['low'] - data['close'].shift(1))])
    data['ATR'] = data['TR'].rolling(window=5).mean()

    # Calculate enhanced adaptive features
    with tqdm(total=1, desc="Calculating Adaptive Features", ncols=80) as pbar:
        data = calculate_adaptive_features(data)
        pbar.update(1)

    # Prepare features for clustering with expanding normalization
    with tqdm(total=1, desc="Normalizing Features", ncols=80) as pbar:
        feature_columns = ['Price_Change_Short', 'Price_Change_Medium', 'Price_Change_Long',
                          'Vol_Ratio_Short', 'Vol_Ratio_Medium', 'Price_Range_Adaptive',
                          'Momentum_Short', 'Momentum_Medium', 'Breakout_Strength', 
                          'Consolidation_Tightness']
        
        features_raw = data[feature_columns].copy()
        
        # Apply expanding window normalization to avoid lookahead bias
        features_normalized = pd.DataFrame(index=features_raw.index)
        for col in feature_columns:
            features_normalized[col] = expanding_normalize(features_raw[col], min_periods=100)
        
        features = features_normalized.dropna()
        pbar.update(1)

    # Run MiniBatchKMeans with tqdm progress
    with tqdm(total=1, desc="Running Clustering", ncols=80) as pbar:
        model = MiniBatchKMeans(n_clusters=K_CLUSTERS, batch_size=BATCH_SIZE, random_state=42)
        data['Regime'] = np.nan
        data.loc[features.index, 'Regime'] = model.fit_predict(features)
        pbar.update(1)

    # Enhanced regime mapping based on multiple characteristics
    with tqdm(total=1, desc="Mapping Regimes", ncols=80) as pbar:
        centroids = pd.DataFrame(model.cluster_centers_, columns=features.columns)
        
        # Create composite scores for regime identification
        centroids['Breakout_Score'] = centroids['Breakout_Strength'] + np.abs(centroids['Price_Change_Medium'])
        centroids['Consolidation_Score'] = -centroids['Consolidation_Tightness'] - np.abs(centroids['Price_Change_Medium'])
        centroids['Bull_Score'] = centroids['Price_Change_Medium'] + centroids['Momentum_Medium']
        centroids['Bear_Score'] = -centroids['Price_Change_Medium'] - centroids['Momentum_Medium']
        
        # Map regimes based on composite scores
        regime_assignments = []
        for i in range(K_CLUSTERS):
            if centroids['Breakout_Score'].iloc[i] > centroids['Breakout_Score'].quantile(0.5):
                if centroids['Bull_Score'].iloc[i] > centroids['Bear_Score'].iloc[i]:
                    regime_assignments.append('Bull Breakout')
                else:
                    regime_assignments.append('Bear Breakout')
            else:
                if centroids['Consolidation_Score'].iloc[i] > centroids['Consolidation_Score'].quantile(0.5):
                    regime_assignments.append('Consolidation')
                else:
                    regime_assignments.append('Neutral')
        
        regime_map = {i: regime_assignments[i] for i in range(K_CLUSTERS)}
        data['Regime_Label'] = data['Regime'].map(regime_map)
        pbar.update(1)

    # Apply temporal logic for post-breakout regimes
    with tqdm(total=1, desc="Applying Temporal Logic", ncols=80) as pbar:
        data = apply_temporal_logic(data)
        pbar.update(1)

    # Add regime persistence confirmation
    with tqdm(total=1, desc="Adding Regime Persistence", ncols=80) as pbar:
        data = add_regime_persistence(data)
        pbar.update(1)

    # Add dummy columns for each regime (1000000 if true, NaN otherwise)
    regimes = ["Bear Breakout", "Bull Breakout", "Consolidation", "Neutral", "Post-Bear", "Post-Bull"]
    for regime in regimes:
        data[regime.replace(" ", "_") + "_Dummy"] = np.where(data['Confirmed_Regime'] == regime, 1000000, np.nan)

    # Add numbered index for continuous plotting (ignores time gaps)
    data['Index'] = range(len(data))

    # Reset index to include Date as column
    data = data.reset_index()

    # Split Date (datetime) into separate Date and Time columns
    data['Date_Separate'] = data['Date'].dt.date
    data['Time'] = data['Date'].dt.time

    # Export to CSV with enhanced regime information
    dummy_columns = [regime.replace(" ", "_") + "_Dummy" for regime in regimes]
    output_columns = ['Index', 'Date_Separate', 'Time', 'open', 'high', 'low', 'close', 
                     'Confirmed_Regime', 'ATR', 'Volatility_Regime'] + dummy_columns
    data[output_columns].to_csv(OUTPUT_FILE, index=False)
    print(f"Exported labeled data to {OUTPUT_FILE}")

    # Print regime distribution
    print("\nFinal Regime Distribution:")
    regime_counts = data['Confirmed_Regime'].value_counts()
    total = len(data.dropna(subset=['Confirmed_Regime']))
    for regime, count in regime_counts.items():
        percentage = (count / total) * 100
        print(f"{regime}: {count} ({percentage:.2f}%)")

    # Print enhanced centroids analysis
    print("\nCluster Centroids Analysis:")
    centroids_display = centroids[['Price_Change_Medium', 'Breakout_Strength', 
                                  'Consolidation_Tightness', 'Vol_Ratio_Short']].round(3)
    for i, regime in regime_map.items():
        print(f"Cluster {i} ({regime}):")
        print(f"  Price Change: {centroids_display.iloc[i]['Price_Change_Medium']}")
        print(f"  Breakout Strength: {centroids_display.iloc[i]['Breakout_Strength']}")
        print(f"  Consolidation Tightness: {centroids_display.iloc[i]['Consolidation_Tightness']}")
        print(f"  Volatility Ratio: {centroids_display.iloc[i]['Vol_Ratio_Short']}")

if __name__ == "__main__":
    main()

