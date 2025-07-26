#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
from tqdm import tqdm

# Centralized parameters
TEST_SLICE = None  # Number of rows to use from the end of the dataset (set to None for full dataset)
DATA_FILE = 'combined_NQ_15m_data.csv'  # Path to your CSV file
OUTPUT_FILE = 'ma_regime_labeled_data.csv'  # Output CSV file name
TIMEFRAME = '15min'  # Timeframe for data loading

# Enhancement toggle - set to True to enable adaptive features, False for pure Excel logic
ENHANCED_FEATURES = True

# Core Excel parameters (exact replication)
CORE_PARAMS = {
    'short_ma_period': 5,     # O56 - 5-period SMA
    'long_ma_period': 13,     # M56 - 13-period SMA
    'short_atr_period': 5,    # R56 - 5-period ATR
    'long_atr_period': 50,    # S56 - 50-period ATR for volatility ratio
    'slope_strong_threshold': 0.0005,    # Strong trend slope
    'slope_weak_threshold': 0.0002,      # Weak trend slope  
    'volatility_low_threshold': 0.8,     # Low volatility threshold
    'volatility_high_threshold': 1.2,    # High volatility threshold
    'transitioning_factor': 0.5,         # Factor for transitioning threshold
    'base_persistence': 2                # Base persistence requirement
}

# Enhanced parameters (only used when ENHANCED_FEATURES = True)
ENHANCED_PARAMS = {
    'slope_adaptation_factor': 0.3,      # How much to adapt slope thresholds
    'volatility_adaptation_factor': 0.2, # How much to adapt volatility thresholds
    'strong_persistence': 1,             # Persistence for STRONG regimes
    'weak_persistence': 2,               # Persistence for WEAK regimes  
    'between_persistence': 3,            # Persistence for BETWEEN regimes
    'session_adaptation': False           # Enable session-based adjustments
}

def load_csv_data(file_path, timeframe):
    """Load CSV data with date parsing"""
    print(f"Loading data from {file_path}")
    df = pd.read_csv(file_path, parse_dates=['Date'])
    df.set_index('Date', inplace=True)
    df = df.sort_index()
    print(f"Loaded {len(df)} rows")
    return df

def calculate_helper_columns(data):
    """Calculate all helper columns exactly as in Excel"""
    
    # Calculate True Range and ATR
    data['TR'] = np.maximum.reduce([
        data['high'] - data['low'], 
        abs(data['high'] - data['close'].shift(1)), 
        abs(data['low'] - data['close'].shift(1))
    ])
    
    # O56: 5-period SMA (short MA)
    data['SMA_5'] = data['close'].rolling(window=CORE_PARAMS['short_ma_period']).mean()
    
    # M56: 13-period SMA (long MA)  
    data['SMA_13'] = data['close'].rolling(window=CORE_PARAMS['long_ma_period']).mean()
    
    # V56: Slope of 13-SMA = (M56 - M55) / M55
    data['SMA_13_Slope'] = data['SMA_13'].pct_change()
    
    # R56: 5-period ATR
    data['ATR_5'] = data['TR'].rolling(window=CORE_PARAMS['short_atr_period']).mean()
    
    # S56: 50-period ATR (for volatility ratio)
    data['ATR_50'] = data['TR'].rolling(window=CORE_PARAMS['long_atr_period']).mean()
    
    # Volatility ratio R56/S56
    data['Volatility_Ratio'] = data['ATR_5'] / data['ATR_50']
    
    # W56: Dynamic multiplier = 0.15 + 0.2 / (1 + 10000 * ABS(V56))
    data['Dynamic_Multiplier'] = 0.15 + 0.2 / (1 + 10000 * np.abs(data['SMA_13_Slope']))
    
    # Crossover thresholds
    data['Upper_Threshold'] = data['SMA_13'] + data['Dynamic_Multiplier'] * data['ATR_5']
    data['Lower_Threshold'] = data['SMA_13'] - data['Dynamic_Multiplier'] * data['ATR_5']
    
    return data

def get_adaptive_thresholds(data, row_idx):
    """Get adaptive thresholds if enhanced features enabled"""
    
    if not ENHANCED_FEATURES:
        return CORE_PARAMS.copy()
    
    # Get current volatility regime for adaptation
    current_vol_ratio = data['Volatility_Ratio'].iloc[row_idx] if row_idx < len(data) else 1.0
    
    # Adaptive slope thresholds (tighter in high vol, looser in low vol)
    slope_adaptation = ENHANCED_PARAMS['slope_adaptation_factor']
    if current_vol_ratio > 1.3:  # High volatility
        slope_multiplier = 1 - slope_adaptation  # Tighter thresholds
    elif current_vol_ratio < 0.7:  # Low volatility  
        slope_multiplier = 1 + slope_adaptation  # Looser thresholds
    else:
        slope_multiplier = 1.0
    
    # Adaptive volatility thresholds
    vol_adaptation = ENHANCED_PARAMS['volatility_adaptation_factor']
    vol_regime_lookback = min(100, row_idx)
    if vol_regime_lookback > 20:
        recent_vol_std = data['Volatility_Ratio'].iloc[row_idx-vol_regime_lookback:row_idx].std()
        if recent_vol_std > 0.3:  # Volatile period
            vol_multiplier = 1 + vol_adaptation  # Wider bands
        else:
            vol_multiplier = 1 - vol_adaptation  # Tighter bands
    else:
        vol_multiplier = 1.0
    
    # Session-based adjustments (if enabled)
    session_multiplier = 1.0
    if ENHANCED_PARAMS['session_adaptation'] and row_idx < len(data):
        current_time = data.index[row_idx].time()
        hour = current_time.hour
        
        # NY session (8-11 AM ET) - more sensitive
        if 13 <= hour <= 16:  # Convert to UTC if needed
            session_multiplier = 0.9
        # Asia session (6 PM - 2 AM ET) - less sensitive  
        elif hour >= 23 or hour <= 7:
            session_multiplier = 1.1
    
    # Build adaptive parameters
    adaptive_params = CORE_PARAMS.copy()
    adaptive_params['slope_strong_threshold'] *= slope_multiplier * session_multiplier
    adaptive_params['slope_weak_threshold'] *= slope_multiplier * session_multiplier
    adaptive_params['volatility_low_threshold'] *= vol_multiplier
    adaptive_params['volatility_high_threshold'] *= vol_multiplier
    
    return adaptive_params

def classify_regime_excel_logic(data, row_idx, params):
    """Exact Excel nested IF logic for regime classification"""
    
    # Get current row values
    slope = data['SMA_13_Slope'].iloc[row_idx]
    short_ma = data['SMA_5'].iloc[row_idx]
    long_ma = data['SMA_13'].iloc[row_idx]
    upper_threshold = data['Upper_Threshold'].iloc[row_idx]
    lower_threshold = data['Lower_Threshold'].iloc[row_idx]
    volatility_ratio = data['Volatility_Ratio'].iloc[row_idx]
    dynamic_multiplier = data['Dynamic_Multiplier'].iloc[row_idx]
    atr_5 = data['ATR_5'].iloc[row_idx]
    
    # Handle NaN values
    if pd.isna(slope) or pd.isna(short_ma) or pd.isna(long_ma) or pd.isna(volatility_ratio):
        return 'BETWEEN'
    
    # Exact Excel nested IF structure
    # STRONG ABOVE: slope > 0.0005 AND short_ma > upper_threshold AND volatility > 1.2
    if (slope > params['slope_strong_threshold'] and 
        short_ma > upper_threshold and 
        volatility_ratio > params['volatility_high_threshold']):
        return 'STRONG ABOVE'
    
    # WEAK ABOVE: slope > 0.0002 AND short_ma > upper_threshold
    elif (slope > params['slope_weak_threshold'] and 
          short_ma > upper_threshold):
        return 'WEAK ABOVE'
    
    # STRONG BELOW: slope < -0.0005 AND short_ma < lower_threshold AND volatility > 1.2
    elif (slope < -params['slope_strong_threshold'] and 
          short_ma < lower_threshold and 
          volatility_ratio > params['volatility_high_threshold']):
        return 'STRONG BELOW'
    
    # WEAK BELOW: slope < -0.0002 AND short_ma < lower_threshold
    elif (slope < -params['slope_weak_threshold'] and 
          short_ma < lower_threshold):
        return 'WEAK BELOW'
    
    # CONTRACTING BETWEEN: abs(slope) <= 0.0002 AND volatility < 0.8
    elif (abs(slope) <= params['slope_weak_threshold'] and 
          volatility_ratio < params['volatility_low_threshold']):
        return 'CONTRACTING BETWEEN'
    
    # EXPANDING BETWEEN: abs(slope) <= 0.0002 AND volatility > 1.2
    elif (abs(slope) <= params['slope_weak_threshold'] and 
          volatility_ratio > params['volatility_high_threshold']):
        return 'EXPANDING BETWEEN'
    
    # TRANSITIONING: abs(slope) <= 0.0002 AND abs(short_ma - long_ma) <= 0.5 * dynamic_multiplier * atr_5
    elif (abs(slope) <= params['slope_weak_threshold'] and 
          abs(short_ma - long_ma) <= params['transitioning_factor'] * dynamic_multiplier * atr_5):
        return 'TRANSITIONING'
    
    # Default case
    else:
        return 'BETWEEN'

def apply_regime_classification(data):
    """Apply regime classification to entire dataset"""
    
    # Initialize regime column
    data['Raw_Regime'] = 'BETWEEN'
    
    # Classify each row
    for i in tqdm(range(len(data)), desc="Classifying Regimes", ncols=80):
        if i >= CORE_PARAMS['long_ma_period']:  # Need enough data for calculations
            params = get_adaptive_thresholds(data, i)
            regime = classify_regime_excel_logic(data, i, params)
            data.loc[data.index[i], 'Raw_Regime'] = regime
    
    return data

def apply_persistence(data):
    """Apply persistence requirement (2-period default, or adaptive if enhanced)"""
    
    data['Confirmed_Regime'] = data['Raw_Regime'].copy()
    
    for i in range(len(data)):
        if i < 2:  # Not enough history
            continue
            
        current_regime = data['Raw_Regime'].iloc[i]
        
        # Determine persistence requirement
        if ENHANCED_FEATURES:
            if 'STRONG' in current_regime:
                persistence_needed = ENHANCED_PARAMS['strong_persistence']
            elif 'WEAK' in current_regime:
                persistence_needed = ENHANCED_PARAMS['weak_persistence']
            else:  # BETWEEN regimes
                persistence_needed = ENHANCED_PARAMS['between_persistence']
        else:
            persistence_needed = CORE_PARAMS['base_persistence']
        
        # Check if we have enough history
        if i < persistence_needed:
            continue
            
        # Check persistence
        recent_regimes = data['Raw_Regime'].iloc[i-persistence_needed+1:i+1]
        
        if len(recent_regimes.unique()) == 1:
            # All recent regimes are the same
            data.loc[data.index[i], 'Confirmed_Regime'] = current_regime
        else:
            # Keep previous confirmed regime
            if i > 0:
                data.loc[data.index[i], 'Confirmed_Regime'] = data['Confirmed_Regime'].iloc[i-1]
    
    return data

def main():
    """Main execution function"""
    
    print(f"Enhanced Features: {'ENABLED' if ENHANCED_FEATURES else 'DISABLED'}")
    
    # Load data
    with tqdm(total=1, desc="Loading Data", ncols=80) as pbar:
        data = load_csv_data(DATA_FILE, TIMEFRAME)
        pbar.update(1)

    # Slice data if test_slice is set
    if TEST_SLICE is not None:
        data = data.tail(TEST_SLICE)
        print(f"Using last {TEST_SLICE} rows for testing")

    # Calculate helper columns (exact Excel replication)
    with tqdm(total=1, desc="Calculating Helper Columns", ncols=80) as pbar:
        data = calculate_helper_columns(data)
        pbar.update(1)

    # Apply regime classification
    data = apply_regime_classification(data)

    # Apply persistence
    with tqdm(total=1, desc="Applying Persistence", ncols=80) as pbar:
        data = apply_persistence(data)
        pbar.update(1)

    # Create dummy columns for visualization
    regimes = ["STRONG ABOVE", "WEAK ABOVE", "STRONG BELOW", "WEAK BELOW", 
               "CONTRACTING BETWEEN", "EXPANDING BETWEEN", "TRANSITIONING", "BETWEEN"]
    
    for regime in regimes:
        col_name = regime.replace(" ", "_") + "_Dummy"
        data[col_name] = np.where(data['Confirmed_Regime'] == regime, 1000000, np.nan)

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
                     'Confirmed_Regime', 'SMA_5', 'SMA_13', 'SMA_13_Slope', 'ATR_5', 
                     'Volatility_Ratio', 'Dynamic_Multiplier'] + dummy_columns
    
    data[output_columns].to_csv(OUTPUT_FILE, index=False)
    print(f"Exported labeled data to {OUTPUT_FILE}")

    # Print regime distribution
    print("\nRegime Distribution:")
    regime_counts = data['Confirmed_Regime'].value_counts()
    total = len(data.dropna(subset=['Confirmed_Regime']))
    for regime, count in regime_counts.items():
        percentage = (count / total) * 100
        print(f"{regime}: {count} ({percentage:.2f}%)")

    # Print system analysis
    print("\nSystem Analysis:")
    print(f"Enhanced Features: {'ENABLED' if ENHANCED_FEATURES else 'DISABLED'}")
    
    # Slope analysis
    valid_slopes = data['SMA_13_Slope'].dropna()
    print(f"Slope Range: {valid_slopes.min():.6f} to {valid_slopes.max():.6f}")
    print(f"Slope Std: {valid_slopes.std():.6f}")
    
    # Volatility analysis  
    valid_vol = data['Volatility_Ratio'].dropna()
    print(f"Volatility Ratio Range: {valid_vol.min():.3f} to {valid_vol.max():.3f}")
    print(f"Volatility Ratio Mean: {valid_vol.mean():.3f}")
    
    # Dynamic multiplier analysis
    valid_mult = data['Dynamic_Multiplier'].dropna()
    print(f"Dynamic Multiplier Range: {valid_mult.min():.3f} to {valid_mult.max():.3f}")
    print(f"Dynamic Multiplier Mean: {valid_mult.mean():.3f}")

    # Enhanced features analysis (if enabled)
    if ENHANCED_FEATURES:
        print("\nEnhanced Features Analysis:")
        strong_regimes = data[data['Confirmed_Regime'].str.contains('STRONG', na=False)]
        weak_regimes = data[data['Confirmed_Regime'].str.contains('WEAK', na=False)]
        between_regimes = data[data['Confirmed_Regime'].str.contains('BETWEEN', na=False)]
        
        print(f"STRONG regimes: {len(strong_regimes)} ({len(strong_regimes)/total*100:.2f}%)")
        print(f"WEAK regimes: {len(weak_regimes)} ({len(weak_regimes)/total*100:.2f}%)")
        print(f"BETWEEN regimes: {len(between_regimes)} ({len(between_regimes)/total*100:.2f}%)")

if __name__ == "__main__":
    main()

