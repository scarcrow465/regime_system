#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from tqdm import tqdm

# Centralized parameters - Adjust these as needed
TEST_SLICE = 10000  # Number of rows to use from the end of the dataset (set to None for full dataset)
DATA_FILE = 'combined_NQ_15m_data.csv'  # Path to your CSV file
OUTPUT_FILE = 'regime_labeled_data_7.csv'  # Output CSV file name
TIMEFRAME = '15min'  # Timeframe for data loading

# Regime-specific parameters
CONSOLIDATION_PARAMS = {
    'range_compression_threshold': 0.5,  # ATR multiple for tight range
    'volatility_threshold': 0.8,  # Relative to rolling average
    'min_consolidation_bars': 5,  # Minimum bars to consider consolidation
    'lookback_windows': [10, 20, 50]  # Multiple timeframes for consolidation detection
}

BREAKOUT_PARAMS = {
    'breakout_threshold': 1.5,  # ATR multiple for breakout
    'momentum_threshold': 0.02,  # Minimum momentum for breakout
    'volume_confirmation': False,  # Whether to use volume (set to False if no volume data)
    'lookback_window': 20  # Window for range calculation
}

POST_BREAKOUT_PARAMS = {
    'max_duration': 10,  # Maximum bars after breakout to consider post-breakout
    'adjacent_only': True  # Only immediate next periods can be post-breakout
}

# Simple data loading function
def load_csv_data(file_path, timeframe):
    print(f"Loading data from {file_path}")
    df = pd.read_csv(file_path, parse_dates=['Date'])
    df.set_index('Date', inplace=True)
    df = df.sort_index()
    print(f"Loaded {len(df)} rows")
    return df

# Calculate multi-timeframe consolidation features
def calculate_consolidation_features(data):
    """Calculate features specifically designed to detect consolidations of variable sizes"""
    
    # Base volatility measures
    data['ATR_5'] = data['TR'].rolling(window=5).mean()
    data['ATR_14'] = data['TR'].rolling(window=14).mean()
    data['ATR_50'] = data['TR'].rolling(window=50).mean()
    
    # Relative volatility (current vs historical)
    data['Volatility_Ratio'] = data['ATR_5'] / data['ATR_50']
    
    # Multi-timeframe range compression detection
    for window in CONSOLIDATION_PARAMS['lookback_windows']:
        col_name = f'Range_Compression_{window}'
        
        # Calculate rolling range as percentage of ATR
        rolling_high = data['high'].rolling(window=window).max()
        rolling_low = data['low'].rolling(window=window).min()
        range_size = (rolling_high - rolling_low) / data['ATR_14']
        
        data[col_name] = range_size
        
        # Range compression score (lower = more compressed)
        data[f'Compression_Score_{window}'] = np.where(
            range_size < CONSOLIDATION_PARAMS['range_compression_threshold'] * window / 10,
            1, 0
        )
    
    # Price position within recent ranges
    for window in [10, 20]:
        rolling_high = data['high'].rolling(window=window).max()
        rolling_low = data['low'].rolling(window=window).min()
        range_position = (data['close'] - rolling_low) / (rolling_high - rolling_low)
        data[f'Range_Position_{window}'] = range_position
        
        # Central position score (higher when price is in middle of range)
        data[f'Central_Position_Score_{window}'] = 1 - 2 * np.abs(range_position - 0.5)
    
    # Trend consistency (lower = more sideways)
    data['Price_Change_5'] = data['close'].pct_change(5)
    data['Price_Change_20'] = data['close'].pct_change(20)
    data['Trend_Consistency'] = np.abs(data['Price_Change_20'])
    
    return data

# Calculate breakout features
def calculate_breakout_features(data):
    """Calculate features specifically designed to detect breakouts"""
    
    # Recent range for breakout calculation
    window = BREAKOUT_PARAMS['lookback_window']
    data['Recent_High'] = data['high'].rolling(window=window).max().shift(1)  # Exclude current bar
    data['Recent_Low'] = data['low'].rolling(window=window).min().shift(1)
    data['Recent_Range'] = data['Recent_High'] - data['Recent_Low']
    
    # Breakout detection
    data['Bull_Breakout_Signal'] = np.where(
        (data['close'] > data['Recent_High']) & 
        (data['close'] - data['Recent_High'] > BREAKOUT_PARAMS['breakout_threshold'] * data['ATR_14']),
        1, 0
    )
    
    data['Bear_Breakout_Signal'] = np.where(
        (data['close'] < data['Recent_Low']) & 
        (data['Recent_Low'] - data['close'] > BREAKOUT_PARAMS['breakout_threshold'] * data['ATR_14']),
        1, 0
    )
    
    # Breakout strength
    data['Bull_Breakout_Strength'] = np.maximum(0, data['close'] - data['Recent_High']) / data['ATR_14']
    data['Bear_Breakout_Strength'] = np.maximum(0, data['Recent_Low'] - data['close']) / data['ATR_14']
    
    # Momentum confirmation
    data['Momentum_5'] = data['close'].pct_change(5)
    data['Momentum_Confirmation'] = np.abs(data['Momentum_5']) > BREAKOUT_PARAMS['momentum_threshold']
    
    return data

# Rule-based regime classification
def classify_regimes_rule_based(data):
    """Classify regimes using rule-based logic instead of clustering"""
    
    data['Regime'] = 'Neutral'  # Default
    
    # 1. Breakout detection (highest priority, no persistence needed)
    bull_breakout_condition = (
        (data['Bull_Breakout_Signal'] == 1) & 
        (data['Momentum_Confirmation'] == True) &
        (data['Bull_Breakout_Strength'] > 1.0)
    )
    
    bear_breakout_condition = (
        (data['Bear_Breakout_Signal'] == 1) & 
        (data['Momentum_Confirmation'] == True) &
        (data['Bear_Breakout_Strength'] > 1.0)
    )
    
    data.loc[bull_breakout_condition, 'Regime'] = 'Bull Breakout'
    data.loc[bear_breakout_condition, 'Regime'] = 'Bear Breakout'
    
    # 2. Consolidation detection (multi-timeframe approach)
    # Require consolidation signals across multiple timeframes
    consolidation_signals = []
    for window in CONSOLIDATION_PARAMS['lookback_windows']:
        signal = data[f'Compression_Score_{window}'] == 1
        consolidation_signals.append(signal)
    
    # Additional consolidation conditions
    low_volatility = data['Volatility_Ratio'] < CONSOLIDATION_PARAMS['volatility_threshold']
    low_trend = data['Trend_Consistency'] < 0.01  # Less than 1% change over 20 bars
    central_position = (data['Central_Position_Score_10'] > 0.3) | (data['Central_Position_Score_20'] > 0.3)
    
    # Consolidation requires multiple confirmations
    consolidation_condition = (
        (consolidation_signals[0] | consolidation_signals[1]) &  # At least one short-term compression
        low_volatility & 
        low_trend & 
        central_position &
        ~bull_breakout_condition &  # Not during breakouts
        ~bear_breakout_condition
    )
    
    data.loc[consolidation_condition, 'Regime'] = 'Consolidation'
    
    return data

# Apply temporal logic for post-breakout regimes (strict adjacency)
def apply_strict_temporal_logic(data):
    """Apply strict temporal sequencing - post-breakout only immediately after breakouts"""
    
    data['Final_Regime'] = data['Regime'].copy()
    
    # Track regime changes
    regime_changed = data['Regime'] != data['Regime'].shift(1)
    
    for i in range(1, len(data)):
        current_regime = data['Regime'].iloc[i]
        previous_regime = data['Regime'].iloc[i-1]
        
        # Only apply post-breakout logic for immediate transitions
        if POST_BREAKOUT_PARAMS['adjacent_only']:
            # Post-Bull: only immediately after Bull Breakout
            if (current_regime in ['Consolidation', 'Neutral']) and (previous_regime == 'Bull Breakout'):
                # Check if we're still within reasonable range of the breakout
                recent_bars = min(POST_BREAKOUT_PARAMS['max_duration'], i)
                recent_regimes = data['Regime'].iloc[i-recent_bars:i]
                
                if 'Bull Breakout' in recent_regimes.values:
                    data.loc[data.index[i], 'Final_Regime'] = 'Post-Bull'
            
            # Post-Bear: only immediately after Bear Breakout  
            elif (current_regime in ['Consolidation', 'Neutral']) and (previous_regime == 'Bear Breakout'):
                recent_bars = min(POST_BREAKOUT_PARAMS['max_duration'], i)
                recent_regimes = data['Regime'].iloc[i-recent_bars:i]
                
                if 'Bear Breakout' in recent_regimes.values:
                    data.loc[data.index[i], 'Final_Regime'] = 'Post-Bear'
    
    return data

# No persistence for breakouts, limited persistence for others
def apply_selective_persistence(data):
    """Apply persistence only to consolidation/neutral regimes, not breakouts"""
    
    data['Confirmed_Regime'] = data['Final_Regime'].copy()
    min_persistence = 3
    
    for i in range(min_persistence, len(data)):
        current_regime = data['Final_Regime'].iloc[i]
        
        # Skip persistence for breakouts - they can be single bars
        if current_regime in ['Bull Breakout', 'Bear Breakout']:
            continue
            
        # Apply persistence for other regimes
        recent_regimes = data['Final_Regime'].iloc[i-min_persistence:i+1]
        
        if current_regime in ['Consolidation', 'Neutral', 'Post-Bull', 'Post-Bear']:
            # Check for consistency
            if len(recent_regimes.unique()) == 1:
                data.loc[data.index[i], 'Confirmed_Regime'] = recent_regimes.iloc[-1]
            else:
                # Keep previous confirmed regime if no consistency
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
    data['ATR'] = data['TR'].rolling(window=14).mean()

    # Calculate regime-specific features
    with tqdm(total=1, desc="Calculating Consolidation Features", ncols=80) as pbar:
        data = calculate_consolidation_features(data)
        pbar.update(1)

    with tqdm(total=1, desc="Calculating Breakout Features", ncols=80) as pbar:
        data = calculate_breakout_features(data)
        pbar.update(1)

    # Rule-based regime classification
    with tqdm(total=1, desc="Classifying Regimes", ncols=80) as pbar:
        data = classify_regimes_rule_based(data)
        pbar.update(1)

    # Apply temporal logic
    with tqdm(total=1, desc="Applying Temporal Logic", ncols=80) as pbar:
        data = apply_strict_temporal_logic(data)
        pbar.update(1)

    # Apply selective persistence
    with tqdm(total=1, desc="Applying Selective Persistence", ncols=80) as pbar:
        data = apply_selective_persistence(data)
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

    # Export to CSV
    dummy_columns = [regime.replace(" ", "_") + "_Dummy" for regime in regimes]
    output_columns = ['Index', 'Date_Separate', 'Time', 'open', 'high', 'low', 'close', 
                     'Confirmed_Regime', 'ATR', 'Volatility_Ratio'] + dummy_columns
    
    data[output_columns].to_csv(OUTPUT_FILE, index=False)
    print(f"Exported labeled data to {OUTPUT_FILE}")

    # Print regime distribution
    print("\nFinal Regime Distribution:")
    regime_counts = data['Confirmed_Regime'].value_counts()
    total = len(data.dropna(subset=['Confirmed_Regime']))
    for regime, count in regime_counts.items():
        percentage = (count / total) * 100
        print(f"{regime}: {count} ({percentage:.2f}%)")

    # Print consolidation analysis
    print("\nConsolidation Detection Analysis:")
    consolidation_data = data[data['Confirmed_Regime'] == 'Consolidation']
    if len(consolidation_data) > 0:
        avg_volatility = consolidation_data['Volatility_Ratio'].mean()
        avg_range_10 = consolidation_data['Range_Compression_10'].mean()
        avg_range_20 = consolidation_data['Range_Compression_20'].mean()
        
        print(f"Average Volatility Ratio during Consolidations: {avg_volatility:.3f}")
        print(f"Average 10-bar Range Compression: {avg_range_10:.3f}")
        print(f"Average 20-bar Range Compression: {avg_range_20:.3f}")

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

