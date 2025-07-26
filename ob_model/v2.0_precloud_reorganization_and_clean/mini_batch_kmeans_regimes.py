#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from tqdm import tqdm

# Centralized parameters - Adjust these as needed
TEST_SLICE = None  # Number of rows to use from the end of the dataset (set to None for full dataset)
DATA_FILE = 'combined_NQ_15m_data.csv'  # Path to your CSV file
OUTPUT_FILE = 'regime_labeled_data_11.csv'  # Output CSV file name
TIMEFRAME = '15min'  # Timeframe for data loading

# Adaptive regime parameters (base multipliers that scale with market conditions)
ADAPTIVE_PARAMS = {
    'breakout_base_threshold': 0.75,  # Base ATR multiple for breakout (reduced from 1.5)
    'consolidation_base_threshold': 3.5,  # Base ATR multiple for consolidation range (increased from 0.5)
    'momentum_lookback': 20,  # Bars to look back for adaptive momentum calculation
    'volatility_lookback': 50,  # Bars for volatility regime calculation
    'adaptive_window_base': [8, 15, 35],  # Base windows that scale with volatility
    'min_consolidation_bars': 3,  # Minimum bars to consider consolidation
    'post_breakout_max': 8  # Maximum bars for post-breakout regime
}

# Simple data loading function
def load_csv_data(file_path, timeframe):
    print(f"Loading data from {file_path}")
    df = pd.read_csv(file_path, parse_dates=['Date'])
    df.set_index('Date', inplace=True)
    df = df.sort_index()
    print(f"Loaded {len(df)} rows")
    return df

# Calculate adaptive market environment measures
def calculate_adaptive_environment(data):
    """Calculate adaptive measures that adjust to current market environment"""
    
    # Base ATR calculations
    data['ATR_5'] = data['TR'].rolling(window=5).mean()
    data['ATR_14'] = data['TR'].rolling(window=14).mean()
    data['ATR_50'] = data['TR'].rolling(window=50).mean()
    
    # Volatility regime classification (adaptive baseline)
    data['Volatility_Ratio'] = data['ATR_14'] / data['ATR_50']
    data['Volatility_Regime'] = np.where(
        data['Volatility_Ratio'] > 1.3, 'High',
        np.where(data['Volatility_Ratio'] < 0.7, 'Low', 'Normal')
    )
    
    # Adaptive momentum baseline (key for NQ's changing price levels)
    momentum_lookback = ADAPTIVE_PARAMS['momentum_lookback']
    data['Recent_Price_Changes'] = data['close'].pct_change().rolling(window=momentum_lookback).std()
    data['Adaptive_Momentum_Threshold'] = data['Recent_Price_Changes'] * 2.0  # 1.5x recent volatility
    
    # Adaptive breakout threshold (scales with recent volatility)
    base_threshold = ADAPTIVE_PARAMS['breakout_base_threshold']
    data['Adaptive_Breakout_Threshold'] = np.where(
        data['Volatility_Regime'] == 'High', base_threshold * 0.7,  # Lower threshold in high vol
        np.where(data['Volatility_Regime'] == 'Low', base_threshold * 1.3, base_threshold)  # Higher in low vol
    )
    
    # Adaptive consolidation threshold
    base_consol = ADAPTIVE_PARAMS['consolidation_base_threshold']
    data['Adaptive_Consolidation_Threshold'] = np.where(
        data['Volatility_Regime'] == 'High', base_consol * 1.4,  # Wider ranges allowed in high vol
        np.where(data['Volatility_Regime'] == 'Low', base_consol * 0.8, base_consol)  # Tighter in low vol
    )
    
    # Adaptive timeframes (shorter in high volatility, longer in low volatility)
    base_windows = ADAPTIVE_PARAMS['adaptive_window_base']
    for i, base_window in enumerate(base_windows):
        col_name = f'Adaptive_Window_{i+1}'
        data[col_name] = np.where(
            data['Volatility_Regime'] == 'High', int(base_window * 0.7),
            np.where(data['Volatility_Regime'] == 'Low', int(base_window * 1.4), base_window)
        ).astype(int)
    
    return data

# Calculate dynamic consolidation features with adaptive windows
def calculate_dynamic_consolidation_features(data):
    """Calculate consolidation features with adaptive windows and thresholds"""
    
    # Initialize consolidation score components
    data['Consolidation_Score_Components'] = 0
    data['Range_Tightness_Score'] = 0
    data['Volatility_Score'] = 0
    data['Position_Score'] = 0
    
    # Calculate features for each adaptive window
    for i in range(3):  # 3 adaptive windows
        window_col = f'Adaptive_Window_{i+1}'
        
        # Ensure we have the adaptive window column
        if window_col not in data.columns:
            continue
            
        # Calculate rolling ranges for each bar (handling variable windows)
        range_values = []
        position_values = []
        
        for j in range(len(data)):
            if j < 50:  # Need minimum data
                range_values.append(np.nan)
                position_values.append(np.nan)
                continue
                
            # Get adaptive window for this bar
            window = int(data[window_col].iloc[j]) if not pd.isna(data[window_col].iloc[j]) else 15
            window = max(5, min(window, j))  # Ensure reasonable window size
            
            # Calculate range for this window
            high_max = data['high'].iloc[j-window:j+1].max()
            low_min = data['low'].iloc[j-window:j+1].min()
            range_size = (high_max - low_min) / data['ATR_14'].iloc[j] if data['ATR_14'].iloc[j] > 0 else np.nan
            
            # Calculate position within range
            if high_max != low_min:
                position = (data['close'].iloc[j] - low_min) / (high_max - low_min)
            else:
                position = 0.5
                
            range_values.append(range_size)
            position_values.append(position)
        
        data[f'Range_Size_{i+1}'] = range_values
        data[f'Range_Position_{i+1}'] = position_values
        
        # Central position score (higher when price is in middle of range)
        data[f'Central_Score_{i+1}'] = 1 - 2 * np.abs(pd.Series(position_values) - 0.5)
    
    # Adaptive range compression detection
    data['Range_Compression_Signal'] = 0
    for i in range(3):
        range_col = f'Range_Size_{i+1}'
        if range_col in data.columns:
            # Compare range to adaptive threshold
            compressed = data[range_col] < data['Adaptive_Consolidation_Threshold']
            data['Range_Compression_Signal'] += compressed.astype(int)
    
    # Trend consistency (adaptive momentum threshold)
    data['Price_Change_5'] = data['close'].pct_change(5)
    data['Price_Change_15'] = data['close'].pct_change(15)
    
    # Use adaptive momentum thresholds
    data['Low_Momentum_Signal'] = (
        (np.abs(data['Price_Change_5']) < data['Adaptive_Momentum_Threshold']) &
        (np.abs(data['Price_Change_15']) < data['Adaptive_Momentum_Threshold'] * 2)
    )
    
    # Low volatility signal (relative to recent environment)
    data['Low_Volatility_Signal'] = data['Volatility_Ratio'] < 0.9
    
    # Central positioning signal
    data['Central_Position_Signal'] = 0
    for i in range(3):
        central_col = f'Central_Score_{i+1}'
        if central_col in data.columns:
            central = data[central_col] > 0.15  # At least somewhat central
            data['Central_Position_Signal'] += central.astype(int)
    
    return data

# Calculate dynamic breakout features
def calculate_dynamic_breakout_features(data):
    """Calculate breakout features with adaptive thresholds"""
    
    # Use adaptive window for breakout calculation
    data['Breakout_Window'] = data['Adaptive_Window_2'].fillna(15).astype(int)  # Use middle window
    
    # Initialize breakout columns
    data['Recent_High'] = np.nan
    data['Recent_Low'] = np.nan
    data['Bull_Breakout_Signal'] = 0
    data['Bear_Breakout_Signal'] = 0
    data['Bull_Breakout_Strength'] = 0
    data['Bear_Breakout_Strength'] = 0
    
    # Calculate breakouts with adaptive windows
    for i in range(len(data)):
        if i < 20:  # Need minimum data
            continue
            
        window = int(data['Breakout_Window'].iloc[i])
        window = max(10, min(window, i-1))  # Ensure reasonable window, exclude current bar
        
        # Calculate recent high/low (excluding current bar)
        recent_high = data['high'].iloc[i-window:i].max()
        recent_low = data['low'].iloc[i-window:i].min()
        
        data.loc[data.index[i], 'Recent_High'] = recent_high
        data.loc[data.index[i], 'Recent_Low'] = recent_low
        
        # Get adaptive breakout threshold for this bar
        breakout_threshold = data['Adaptive_Breakout_Threshold'].iloc[i]
        atr_value = data['ATR_14'].iloc[i]
        current_close = data['close'].iloc[i]
        
        if pd.isna(atr_value) or atr_value <= 0:
            continue
            
        # Bull breakout detection
        if current_close > recent_high:
            bull_strength = (current_close - recent_high) / atr_value
            if bull_strength > breakout_threshold:
                data.loc[data.index[i], 'Bull_Breakout_Signal'] = 1
                data.loc[data.index[i], 'Bull_Breakout_Strength'] = bull_strength
        
        # Bear breakout detection
        if current_close < recent_low:
            bear_strength = (recent_low - current_close) / atr_value
            if bear_strength > breakout_threshold:
                data.loc[data.index[i], 'Bear_Breakout_Signal'] = 1
                data.loc[data.index[i], 'Bear_Breakout_Strength'] = bear_strength
    
    # Adaptive momentum confirmation
    data['Momentum_5'] = data['close'].pct_change(5)
    data['Strong_Momentum'] = np.abs(data['Momentum_5']) > data['Adaptive_Momentum_Threshold']
    
    return data

# Rule-based regime classification with adaptive thresholds
def classify_regimes_adaptive(data):
    """Classify regimes using adaptive rule-based logic"""
    
    data['Regime'] = 'Neutral'  # Default
    
    # 1. Breakout detection (highest priority)
    bull_breakout_condition = (
        (data['Bull_Breakout_Signal'] == 1) & 
        (data['Strong_Momentum'] == True) &
        (data['Bull_Breakout_Strength'] > 0.5)  # Lower minimum threshold
    )
    
    bear_breakout_condition = (
        (data['Bear_Breakout_Signal'] == 1) & 
        (data['Strong_Momentum'] == True) &
        (data['Bear_Breakout_Strength'] > 0.5)  # Lower minimum threshold
    )
    
    data.loc[bull_breakout_condition, 'Regime'] = 'Bull Breakout'
    data.loc[bear_breakout_condition, 'Regime'] = 'Bear Breakout'
    
    # 2. Simple consolidation detection - core logic only
    # Consolidation = not breaking out AND no sustained momentum

    # Check for sustained momentum (stronger than adaptive threshold over longer period)
    sustained_bull_momentum = (
        (data['Price_Change_5'] > data['Adaptive_Momentum_Threshold']) &
        (data['Price_Change_15'] > data['Adaptive_Momentum_Threshold'] * 1.5)
    )

    sustained_bear_momentum = (
        (data['Price_Change_5'] < -data['Adaptive_Momentum_Threshold']) &
        (data['Price_Change_15'] < -data['Adaptive_Momentum_Threshold'] * 1.5)
    )

    # Consolidation = everything that's not breakout and not sustained trending
    consolidation_condition = (
        ~bull_breakout_condition &          # Not bull breakout
        ~bear_breakout_condition &          # Not bear breakout  
        ~sustained_bull_momentum &          # Not sustained uptrend
        ~sustained_bear_momentum            # Not sustained downtrend
    )

    data.loc[consolidation_condition, 'Regime'] = 'Consolidation'
    
    return data

# Apply strict temporal logic for post-breakout regimes
def apply_adaptive_temporal_logic(data):
    """Apply temporal sequencing with adaptive post-breakout duration"""
    
    data['Final_Regime'] = data['Regime'].copy()
    
    # Use adaptive post-breakout duration based on volatility
    max_duration = ADAPTIVE_PARAMS['post_breakout_max']
    
    for i in range(1, len(data)):
        current_regime = data['Regime'].iloc[i]
        previous_regime = data['Regime'].iloc[i-1]
        
        # Adaptive duration based on volatility regime
        vol_regime = data['Volatility_Regime'].iloc[i]
        if vol_regime == 'High':
            duration = max(3, int(max_duration * 0.6))  # Shorter in high vol
        elif vol_regime == 'Low':
            duration = int(max_duration * 1.3)  # Longer in low vol
        else:
            duration = max_duration
        
        # Post-Bull logic
        if current_regime in ['Consolidation', 'Neutral'] and previous_regime == 'Bull Breakout':
            data.loc[data.index[i], 'Final_Regime'] = 'Post-Bull'
        elif current_regime in ['Consolidation', 'Neutral']:
            # Check if we're within duration of a bull breakout
            lookback = min(duration, i)
            recent_regimes = data['Regime'].iloc[i-lookback:i]
            if 'Bull Breakout' in recent_regimes.values:
                # Check if no other breakout occurred since
                last_bull_idx = recent_regimes[::-1].eq('Bull Breakout').idxmax()
                regimes_since_bull = recent_regimes.loc[last_bull_idx:]
                if not any(x in ['Bear Breakout'] for x in regimes_since_bull):
                    data.loc[data.index[i], 'Final_Regime'] = 'Post-Bull'
        
        # Post-Bear logic (similar to Post-Bull)
        if current_regime in ['Consolidation', 'Neutral'] and previous_regime == 'Bear Breakout':
            data.loc[data.index[i], 'Final_Regime'] = 'Post-Bear'
        elif current_regime in ['Consolidation', 'Neutral']:
            lookback = min(duration, i)
            recent_regimes = data['Regime'].iloc[i-lookback:i]
            if 'Bear Breakout' in recent_regimes.values:
                last_bear_idx = recent_regimes[::-1].eq('Bear Breakout').idxmax()
                regimes_since_bear = recent_regimes.loc[last_bear_idx:]
                if not any(x in ['Bull Breakout'] for x in regimes_since_bear):
                    data.loc[data.index[i], 'Final_Regime'] = 'Post-Bear'
    
    return data

# Minimal persistence for non-breakout regimes
def apply_minimal_persistence(data):
    """Apply minimal persistence to reduce noise"""
    
    data['Confirmed_Regime'] = data['Final_Regime'].copy()
    min_persistence = 2  # Reduced from 3
    
    for i in range(min_persistence, len(data)):
        current_regime = data['Final_Regime'].iloc[i]
        
        # Skip persistence for breakouts
        if current_regime in ['Bull Breakout', 'Bear Breakout']:
            continue
            
        # Apply minimal persistence for other regimes
        recent_regimes = data['Final_Regime'].iloc[i-min_persistence:i+1]
        
        if current_regime in ['Consolidation', 'Neutral', 'Post-Bull', 'Post-Bear']:
            # Check for consistency over shorter window
            if len(recent_regimes.unique()) == 1:
                data.loc[data.index[i], 'Confirmed_Regime'] = recent_regimes.iloc[-1]
            else:
                # Keep previous confirmed regime
                if i > 0:
                    data.loc[data.index[i], 'Confirmed_Regime'] = data['Confirmed_Regime'].iloc[i-1]
    
    return data

# Main function
def main():
    # Load data
    with tqdm(total=1, desc="Loading Data", ncols=80) as pbar:
        data = load_csv_data(DATA_FILE, TIMEFRAME)
        pbar.update(1)

    # Slice data if test_slice is set
    if TEST_SLICE is not None:
        data = data.tail(TEST_SLICE)
        print(f"Using last {TEST_SLICE} rows for testing")

    # Calculate ATR and True Range
    data['TR'] = np.maximum.reduce([
        data['high'] - data['low'], 
        abs(data['high'] - data['close'].shift(1)), 
        abs(data['low'] - data['close'].shift(1))
    ])

    # Calculate adaptive environment measures
    with tqdm(total=1, desc="Calculating Adaptive Environment", ncols=80) as pbar:
        data = calculate_adaptive_environment(data)
        pbar.update(1)

    # Calculate dynamic consolidation features
    with tqdm(total=1, desc="Calculating Dynamic Consolidation Features", ncols=80) as pbar:
        data = calculate_dynamic_consolidation_features(data)
        pbar.update(1)

    # Calculate dynamic breakout features
    with tqdm(total=1, desc="Calculating Dynamic Breakout Features", ncols=80) as pbar:
        data = calculate_dynamic_breakout_features(data)
        pbar.update(1)

    # Adaptive regime classification
    with tqdm(total=1, desc="Classifying Regimes Adaptively", ncols=80) as pbar:
        data = classify_regimes_adaptive(data)
        pbar.update(1)

    # Apply adaptive temporal logic
    with tqdm(total=1, desc="Applying Adaptive Temporal Logic", ncols=80) as pbar:
        data = apply_adaptive_temporal_logic(data)
        pbar.update(1)

    # Apply minimal persistence
    with tqdm(total=1, desc="Applying Minimal Persistence", ncols=80) as pbar:
        data = apply_minimal_persistence(data)
        pbar.update(1)

    # Add dummy columns for visualization
    regimes = ["Bear Breakout", "Bull Breakout", "Consolidation", "Neutral", "Post-Bear", "Post-Bull"]
    for regime in regimes:
        data[regime.replace(" ", "_") + "_Dummy"] = np.where(data['Confirmed_Regime'] == regime, 1000000, np.nan)

    # Add numbered index for continuous plotting
    data['Index'] = range(len(data))

    # Reset index to include Date as column
    data = data.reset_index()

    # Split Date into separate Date and Time columns
    data['Date_Separate'] = data['Date'].dt.date
    data['Time'] = data['Date'].dt.time

    # Export to CSV with adaptive regime information
    dummy_columns = [regime.replace(" ", "_") + "_Dummy" for regime in regimes]
    output_columns = ['Index', 'Date_Separate', 'Time', 'open', 'high', 'low', 'close', 
                     'Confirmed_Regime', 'ATR_14', 'Volatility_Regime', 'Adaptive_Momentum_Threshold'] + dummy_columns
    
    data[output_columns].to_csv(OUTPUT_FILE, index=False)
    print(f"Exported labeled data to {OUTPUT_FILE}")

    # Print regime distribution
    print("\nFinal Regime Distribution:")
    regime_counts = data['Confirmed_Regime'].value_counts()
    total = len(data.dropna(subset=['Confirmed_Regime']))
    for regime, count in regime_counts.items():
        percentage = (count / total) * 100
        print(f"{regime}: {count} ({percentage:.2f}%)")

    # Print adaptive thresholds analysis
    print("\nAdaptive Thresholds Analysis:")
    print(f"Breakout Thresholds - Min: {data['Adaptive_Breakout_Threshold'].min():.3f}, Max: {data['Adaptive_Breakout_Threshold'].max():.3f}, Avg: {data['Adaptive_Breakout_Threshold'].mean():.3f}")
    print(f"Consolidation Thresholds - Min: {data['Adaptive_Consolidation_Threshold'].min():.3f}, Max: {data['Adaptive_Consolidation_Threshold'].max():.3f}, Avg: {data['Adaptive_Consolidation_Threshold'].mean():.3f}")
    print(f"Momentum Thresholds - Min: {data['Adaptive_Momentum_Threshold'].min():.5f}, Max: {data['Adaptive_Momentum_Threshold'].max():.5f}, Avg: {data['Adaptive_Momentum_Threshold'].mean():.5f}")

    # Print volatility regime distribution
    print("\nVolatility Regime Distribution:")
    vol_regime_counts = data['Volatility_Regime'].value_counts()
    for regime, count in vol_regime_counts.items():
        percentage = (count / len(data)) * 100
        print(f"{regime} Volatility: {count} ({percentage:.2f}%)")

    # Print consolidation analysis
    print("\nConsolidation Detection Analysis:")
    consolidation_data = data[data['Confirmed_Regime'] == 'Consolidation']
    if len(consolidation_data) > 0:
        avg_volatility = consolidation_data['Volatility_Ratio'].mean()
        avg_compression = consolidation_data['Range_Compression_Signal'].mean()
        
        print(f"Average Volatility Ratio during Consolidations: {avg_volatility:.3f}")
        print(f"Average Range Compression Score: {avg_compression:.3f}")

    # Print breakout analysis
    print("\nBreakout Analysis:")
    bull_breakouts = data[data['Confirmed_Regime'] == 'Bull Breakout']
    bear_breakouts = data[data['Confirmed_Regime'] == 'Bear Breakout']
    
    if len(bull_breakouts) > 0:
        avg_bull_strength = bull_breakouts['Bull_Breakout_Strength'].mean()
        print(f"Average Bull Breakout Strength: {avg_bull_strength:.3f} ATR")
    
    if len(bear_breakouts) > 0:
        avg_bear_strength = bear_breakouts['Bear_Breakout_Strength'].mean()
        print(f"Average Bear Breakout Strength: {avg_bear_strength:.3f} ATR")

if __name__ == "__main__":
    main()

