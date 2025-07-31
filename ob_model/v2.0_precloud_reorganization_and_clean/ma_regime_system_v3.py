#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Centralized parameters
TEST_SLICE = 5000 # Number of rows to use from the end of the dataset (set to None for full dataset)
DATA_FILE = 'combined_NQ_15m_data.csv'  # Path to your CSV file
OUTPUT_FILE = 'ma_regime_labeled_data_with_perfect_v23.csv'  # Output CSV file name
SCORING_FILE = 'regime_scoring_metrics_v23.csv'  # Scoring metrics output
TIMEFRAME = '15min'  # Timeframe for data loading

# Enhancement toggle - set to True to enable adaptive features, False for pure Excel logic
ENHANCED_FEATURES = True

# Core Excel parameters (exact replication)
CORE_PARAMS = {
    'short_ma_period': 5,     # O56 - 5-period SMA
    'long_ma_period': 13,     # M56 - 13-period SMA
    'short_atr_period': 5,    # R56 - 5-period ATR
    'long_atr_period': 50,    # S56 - 50-period ATR for volatility ratio
    'base_slope_lookback': 300,          # Base lookback for dynamic slope thresholds
    'bull_weak_percentile': 0.65,        # 65th percentile for bull weak
    'bull_strong_percentile': 0.85,      # 85th percentile for bull strong
    'bear_weak_percentile': 0.35,        # 35th percentile for bear weak (inverted)
    'bear_strong_percentile': 0.15,      # 15th percentile for bear strong (inverted)
    'volatility_lookback': 100,          # Lookback for dynamic volatility thresholds
    'volatility_filter_min': 0.7,        # Min volatility for "normal" periods
    'volatility_filter_max': 1.3,        # Max volatility for "normal" periods
    'volatility_low_percentile': 0.25,        # 25th percentile for low volatility (general)
    'volatility_high_percentile': 0.85,       # 75th percentile for high volatility (general)
    'volatility_high_bull_percentile': 0.85,  # Lower threshold for bull STRONG moves
    'volatility_high_bear_percentile': 0.85,  # Higher threshold for bear STRONG moves
    'volatility_low_bull_percentile': 0.25,   # Bull-specific low volatility threshold
    'volatility_low_bear_percentile': 0.25,   # Bear-specific low volatility threshold
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
    'session_adaptation': False,          # Enable session-based adjustments
    'between_scale_sensitivity': 1.0,
    'volatility_scale_factor': 1.0
}

# Perfect system parameters
PERFECT_PARAMS = {
    'min_forward_bars': 100,              # Minimum bars to look forward
    'max_forward_bars': 200,             # Maximum bars to look forward
    'adaptive_forward': True,            # Use adaptive forward looking based on volatility
    'move_lookback': 500,                # Lookback for calculating dynamic move thresholds
    'strong_move_percentile': 0.85,      # Percentile for strong move threshold
    'weak_move_percentile': 0.65,        # Percentile for weak move threshold
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
    data['TR'] = np.maximum.reduce([
        data['high'] - data['low'], 
        abs(data['high'] - data['close'].shift(1)), 
        abs(data['low'] - data['close'].shift(1))
    ])
    data['SMA_5'] = data['close'].rolling(window=CORE_PARAMS['short_ma_period']).mean()
    data['SMA_13'] = data['close'].rolling(window=CORE_PARAMS['long_ma_period']).mean()
    data['SMA_13_Slope'] = data['SMA_13'].pct_change()
    data['ATR_5'] = data['TR'].rolling(window=CORE_PARAMS['short_atr_period']).mean()
    data['ATR_50'] = data['TR'].rolling(window=CORE_PARAMS['long_atr_period']).mean()
    data['Volatility_Ratio'] = data['ATR_5'] / data['ATR_50']
    data['Up_Candle'] = data['close'] > data['close'].shift(1)
    data['Down_Candle'] = data['close'] < data['close'].shift(1)
    data['Up_TR'] = np.where(data['Up_Candle'], data['TR'], 0)
    data['Down_TR'] = np.where(data['Down_Candle'], data['TR'], 0)
    data['Up_ATR_5'] = data['Up_TR'].rolling(window=CORE_PARAMS['short_atr_period']).sum() / data['Up_Candle'].rolling(window=CORE_PARAMS['short_atr_period']).sum()
    data['Down_ATR_5'] = data['Down_TR'].rolling(window=CORE_PARAMS['short_atr_period']).sum() / data['Down_Candle'].rolling(window=CORE_PARAMS['short_atr_period']).sum()
    data['Up_ATR_5'] = data['Up_ATR_5'].fillna(data['ATR_5'])
    data['Down_ATR_5'] = data['Down_ATR_5'].fillna(data['ATR_5'])
    data['Up_Volatility_Ratio'] = data['Up_ATR_5'] / data['ATR_50']
    data['Down_Volatility_Ratio'] = data['Down_ATR_5'] / data['ATR_50']
    data['Dynamic_Multiplier'] = 0.15 + 0.2 / (1 + 10000 * np.abs(data['SMA_13_Slope']))
    data['Upper_Threshold'] = data['SMA_13'] + data['Dynamic_Multiplier'] * data['ATR_5']
    data['Lower_Threshold'] = data['SMA_13'] - data['Dynamic_Multiplier'] * data['ATR_5']
    
    vol_regime_factor = data['Volatility_Ratio'].rolling(window=50).mean()
    base_lookback = CORE_PARAMS['base_slope_lookback']
    data['Adaptive_Slope_Lookback'] = np.where(
        vol_regime_factor > 1.2, int(base_lookback * 0.7),
        np.where(vol_regime_factor < 0.8, int(base_lookback * 1.3), base_lookback)
    ).astype(int)
    
    vol_filter_min = CORE_PARAMS['volatility_filter_min']
    vol_filter_max = CORE_PARAMS['volatility_filter_max']
    
    data['Dynamic_Strong_Move_Threshold'] = np.nan
    data['Dynamic_Weak_Move_Threshold'] = np.nan
    
    move_lookback = PERFECT_PARAMS['move_lookback']
    print("Calculating Dynamic Move Thresholds (vectorized)...")
    future_window = 20
    data['Future_High_20'] = data['high'].rolling(window=future_window, min_periods=1).max().shift(-future_window)
    data['Future_Low_20'] = data['low'].rolling(window=future_window, min_periods=1).min().shift(-future_window)
    data['Future_Up_Move'] = (data['Future_High_20'] - data['close']) / data['close']
    data['Future_Down_Move'] = (data['close'] - data['Future_Low_20']) / data['close']
    data['Max_Future_Move'] = data[['Future_Up_Move', 'Future_Down_Move']].max(axis=1)
    data['Dynamic_Strong_Move_Threshold'] = data['Max_Future_Move'].rolling(
        window=move_lookback, min_periods=100
    ).quantile(PERFECT_PARAMS['strong_move_percentile'])
    data['Dynamic_Weak_Move_Threshold'] = data['Max_Future_Move'].rolling(
        window=move_lookback, min_periods=100
    ).quantile(PERFECT_PARAMS['weak_move_percentile'])
    data.drop(['Future_High_20', 'Future_Low_20', 'Future_Up_Move', 'Future_Down_Move', 'Max_Future_Move'], axis=1, inplace=True)
    
    data['Dynamic_Bull_Weak'] = np.nan
    data['Dynamic_Bull_Strong'] = np.nan
    data['Dynamic_Bear_Weak'] = np.nan
    data['Dynamic_Bear_Strong'] = np.nan
    data['Dynamic_Vol_Low'] = np.nan
    data['Dynamic_Vol_High'] = np.nan
    data['Dynamic_Vol_High_Bull'] = np.nan
    data['Dynamic_Vol_High_Bear'] = np.nan
    data['Dynamic_Vol_Low_Bull'] = np.nan
    data['Dynamic_Vol_Low_Bear'] = np.nan
    data['Perfect_Bull_Weak'] = np.nan
    data['Perfect_Bull_Strong'] = np.nan
    data['Perfect_Bear_Weak'] = np.nan
    data['Perfect_Bear_Strong'] = np.nan
    data['Perfect_Vol_Low'] = np.nan
    data['Perfect_Vol_High'] = np.nan
    data['Perfect_Vol_High_Bull'] = np.nan
    data['Perfect_Vol_High_Bear'] = np.nan
    data['Perfect_Vol_Low_Bull'] = np.nan
    data['Perfect_Vol_Low_Bear'] = np.nan
    
    # Calculate directional slope thresholds with volatility filtering (LIVE SYSTEM)
    def rolling_masked_quantile(series, lookback, percentile, mask_series, min_val, max_val):
        def custom_quantile(arr):
            mask = (arr >= min_val) & (arr <= max_val)
            if mask.sum() < 50:  # Increased from 20
                return np.quantile(arr, percentile)
            return np.quantile(arr[mask], percentile)
        return series.rolling(lookback).apply(custom_quantile, raw=True)

    data['Dynamic_Bull_Weak'] = rolling_masked_quantile(
        data['SMA_13_Slope'],
        CORE_PARAMS['base_slope_lookback'],
        CORE_PARAMS['bull_weak_percentile'],
        data['Volatility_Ratio'],
        CORE_PARAMS['volatility_filter_min'],
        CORE_PARAMS['volatility_filter_max']
    )

    data['Dynamic_Bull_Strong'] = rolling_masked_quantile(
        data['SMA_13_Slope'],
        CORE_PARAMS['base_slope_lookback'],
        CORE_PARAMS['bull_strong_percentile'],
        data['Volatility_Ratio'],
        CORE_PARAMS['volatility_filter_min'],
        CORE_PARAMS['volatility_filter_max']
    )

    data['Dynamic_Bear_Weak'] = rolling_masked_quantile(
        data['SMA_13_Slope'].abs(),
        CORE_PARAMS['base_slope_lookback'],
        CORE_PARAMS['bear_weak_percentile'],
        data['Volatility_Ratio'],
        CORE_PARAMS['volatility_filter_min'],
        CORE_PARAMS['volatility_filter_max']
    )

    data['Dynamic_Bear_Strong'] = rolling_masked_quantile(
        data['SMA_13_Slope'].abs(),
        CORE_PARAMS['base_slope_lookback'],
        CORE_PARAMS['bear_strong_percentile'],
        data['Volatility_Ratio'],
        CORE_PARAMS['volatility_filter_min'],
        CORE_PARAMS['volatility_filter_max']
    )

    data['Dynamic_Vol_Low'] = data['Volatility_Ratio'].rolling(CORE_PARAMS['volatility_lookback']).quantile(CORE_PARAMS['volatility_low_percentile'])
    data['Dynamic_Vol_High'] = data['Volatility_Ratio'].rolling(CORE_PARAMS['volatility_lookback']).quantile(CORE_PARAMS['volatility_high_percentile'])

    data['Dynamic_Vol_High_Bull'] = data['Up_Volatility_Ratio'].rolling(CORE_PARAMS['volatility_lookback']).quantile(CORE_PARAMS['volatility_high_bull_percentile'])
    data['Dynamic_Vol_High_Bear'] = data['Down_Volatility_Ratio'].rolling(CORE_PARAMS['volatility_lookback']).quantile(CORE_PARAMS['volatility_high_bear_percentile'])
    data['Dynamic_Vol_Low_Bull'] = data['Up_Volatility_Ratio'].rolling(CORE_PARAMS['volatility_lookback']).quantile(CORE_PARAMS['volatility_low_bull_percentile'])
    data['Dynamic_Vol_Low_Bear'] = data['Down_Volatility_Ratio'].rolling(CORE_PARAMS['volatility_lookback']).quantile(CORE_PARAMS['volatility_low_bear_percentile'])

    data['Dynamic_Bull_Weak'] = data['Dynamic_Bull_Weak'].fillna(0.0002)
    data['Dynamic_Bull_Strong'] = data['Dynamic_Bull_Strong'].fillna(0.0005)
    data['Dynamic_Bear_Weak'] = data['Dynamic_Bear_Weak'].fillna(0.0002)
    data['Dynamic_Bear_Strong'] = data['Dynamic_Bear_Strong'].fillna(0.0005)
    data['Dynamic_Vol_Low'] = data['Dynamic_Vol_Low'].fillna(0.8)
    data['Dynamic_Vol_High'] = data['Dynamic_Vol_High'].fillna(1.2)
    data['Dynamic_Vol_High_Bull'] = data['Dynamic_Vol_High_Bull'].fillna(1.0)
    data['Dynamic_Vol_High_Bear'] = data['Dynamic_Vol_High_Bear'].fillna(1.2)
    data['Dynamic_Vol_Low_Bull'] = data['Dynamic_Vol_Low_Bull'].fillna(0.8)
    data['Dynamic_Vol_Low_Bear'] = data['Dynamic_Vol_Low_Bear'].fillna(0.8)
    
    # Perfect thresholds: Backward fallback for NaNs
    for i in range(len(data)):
        if pd.isna(data['Perfect_Bull_Weak'].iloc[i]):
            lookback = min(100, i)
            past_slopes = data['SMA_13_Slope'].iloc[i-lookback:i]
            data['Perfect_Bull_Weak'].iloc[i] = past_slopes.quantile(CORE_PARAMS['bull_weak_percentile'])
        # Repeat for other perfect thresholds (Perfect_Bull_Strong, Perfect_Bear_Weak, etc.)
    
    return data

def get_adaptive_thresholds(data, row_idx, use_perfect=False):
    """Get adaptive thresholds using directional and volatility-filtered percentiles"""
    
    dynamic_params = CORE_PARAMS.copy()
    
    # Choose column prefix based on system type
    prefix = 'Perfect_' if use_perfect else 'Dynamic_'
    
    if row_idx < len(data):
        # Get directional slope thresholds
        dynamic_params['bull_weak_threshold'] = data[f'{prefix}Bull_Weak'].iloc[row_idx]
        dynamic_params['bull_strong_threshold'] = data[f'{prefix}Bull_Strong'].iloc[row_idx]
        dynamic_params['bear_weak_threshold'] = data[f'{prefix}Bear_Weak'].iloc[row_idx]
        dynamic_params['bear_strong_threshold'] = data[f'{prefix}Bear_Strong'].iloc[row_idx]
        
        # Get volatility thresholds
        dynamic_params['volatility_low_threshold'] = data[f'{prefix}Vol_Low'].iloc[row_idx]
        dynamic_params['volatility_high_threshold'] = data[f'{prefix}Vol_High'].iloc[row_idx]
        dynamic_params['volatility_high_bull_threshold'] = data[f'{prefix}Vol_High_Bull'].iloc[row_idx]
        dynamic_params['volatility_high_bear_threshold'] = data[f'{prefix}Vol_High_Bear'].iloc[row_idx]
        dynamic_params['volatility_low_bull_threshold'] = data[f'{prefix}Vol_Low_Bull'].iloc[row_idx]
        dynamic_params['volatility_low_bear_threshold'] = data[f'{prefix}Vol_Low_Bear'].iloc[row_idx]
        
        # Handle NaN values with reasonable fallbacks
        if pd.isna(dynamic_params['bull_weak_threshold']):
            dynamic_params['bull_weak_threshold'] = 0.0002
        if pd.isna(dynamic_params['bull_strong_threshold']):
            dynamic_params['bull_strong_threshold'] = 0.0005
        if pd.isna(dynamic_params['bear_weak_threshold']):
            dynamic_params['bear_weak_threshold'] = 0.0002
        if pd.isna(dynamic_params['bear_strong_threshold']):
            dynamic_params['bear_strong_threshold'] = 0.0005
        if pd.isna(dynamic_params['volatility_low_threshold']):
            dynamic_params['volatility_low_threshold'] = 0.8
        if pd.isna(dynamic_params['volatility_high_threshold']):
            dynamic_params['volatility_high_threshold'] = 1.2
        if pd.isna(dynamic_params['volatility_high_bull_threshold']):
            dynamic_params['volatility_high_bull_threshold'] = 1.0
        if pd.isna(dynamic_params['volatility_high_bear_threshold']):
            dynamic_params['volatility_high_bear_threshold'] = 1.2
        if pd.isna(dynamic_params['volatility_low_bull_threshold']):
            dynamic_params['volatility_low_bull_threshold'] = 0.8
        if pd.isna(dynamic_params['volatility_low_bear_threshold']):
            dynamic_params['volatility_low_bear_threshold'] = 0.8
    else:
        # Fallback to original fixed values
        dynamic_params['bull_weak_threshold'] = 0.0002
        dynamic_params['bull_strong_threshold'] = 0.0005
        dynamic_params['bear_weak_threshold'] = 0.0002
        dynamic_params['bear_strong_threshold'] = 0.0005
        dynamic_params['volatility_low_threshold'] = 0.8
        dynamic_params['volatility_high_threshold'] = 1.2
        dynamic_params['volatility_high_bull_threshold'] = 1.0
        dynamic_params['volatility_high_bear_threshold'] = 1.2
        dynamic_params['volatility_low_bull_threshold'] = 0.8
        dynamic_params['volatility_low_bear_threshold'] = 0.8
    
    return dynamic_params

def classify_regime_excel_logic(data, row_idx, params):
    slope = data['SMA_13_Slope'].iloc[row_idx]
    short_ma = data['SMA_5'].iloc[row_idx]
    long_ma = data['SMA_13'].iloc[row_idx]
    upper_threshold = data['Upper_Threshold'].iloc[row_idx]
    lower_threshold = data['Lower_Threshold'].iloc[row_idx]
    volatility_ratio = data['Volatility_Ratio'].iloc[row_idx]
    dynamic_multiplier = data['Dynamic_Multiplier'].iloc[row_idx]
    atr_5 = data['ATR_5'].iloc[row_idx]
    
    if pd.isna(slope) or pd.isna(short_ma) or pd.isna(long_ma) or pd.isna(volatility_ratio):
        return 'BETWEEN'
    
    up_volatility_ratio = data['Up_Volatility_Ratio'].iloc[row_idx]
    down_volatility_ratio = data['Down_Volatility_Ratio'].iloc[row_idx]
    
    abs_slope = abs(slope)
    max_strong = max(params['bull_strong_threshold'], abs(params['bear_strong_threshold']))
    between_scale_factor = max(0, min(1, 1 - (abs_slope / (max_strong * ENHANCED_PARAMS['between_scale_sensitivity'])))) if max_strong > 0 else 1
    vol_factor = min(1.5, max(0.5, volatility_ratio / params['volatility_high_threshold']))
    effective_bull_weak = params['bull_weak_threshold'] * between_scale_factor * vol_factor
    effective_bear_weak = params['bear_weak_threshold'] * between_scale_factor * vol_factor
    
    if (slope > params['bull_strong_threshold'] and short_ma > upper_threshold and up_volatility_ratio > params['volatility_high_bull_threshold']):
        return 'STRONG ABOVE'
    elif (slope > params['bull_weak_threshold'] and short_ma > upper_threshold):
        return 'WEAK ABOVE'
    elif (slope < -params['bear_strong_threshold'] and short_ma < lower_threshold and down_volatility_ratio > params['volatility_high_bear_threshold']):
        return 'STRONG BELOW'
    elif (slope < -params['bear_weak_threshold'] and short_ma < lower_threshold):
        return 'WEAK BELOW'
    elif (abs(slope) <= effective_bull_weak):
        if abs(short_ma - long_ma) <= params['transitioning_factor'] * dynamic_multiplier * atr_5:
            return 'TRANSITIONING'
        elif volatility_ratio > params['volatility_high_threshold']:
            return 'EXPANDING BETWEEN'
        elif volatility_ratio < params['volatility_low_threshold']:
            return 'CONTRACTING BETWEEN'
    return 'BETWEEN'

def validate_perfect_regime_with_future(data, row_idx, classified_regime):
    """Validate perfect regime classification using dynamic future price movements with conservative upgrade logic"""
    
    # DEBUG: Track validation decisions every 500 bars
    debug_this_bar = (row_idx % 500 == 0)
    
    # Check if we have enough future data
    forward_check = min(20, len(data) - row_idx - 1)  # Aligned to 20 bars
    if forward_check < 10:
        return classified_regime  # Not enough data to validate
    
    # Get dynamic thresholds for this bar
    strong_threshold = data['Dynamic_Strong_Move_Threshold'].iloc[row_idx]
    weak_threshold = data['Dynamic_Weak_Move_Threshold'].iloc[row_idx]
    
    # Use fallback if dynamic thresholds not available
    if pd.isna(strong_threshold):
        strong_threshold = 0.008
    if pd.isna(weak_threshold):
        weak_threshold = 0.004
    
    # Get current market conditions for upgrade validation
    slope = data['SMA_13_Slope'].iloc[row_idx]
    short_ma = data['SMA_5'].iloc[row_idx]
    long_ma = data['SMA_13'].iloc[row_idx]
    upper_threshold = data['Upper_Threshold'].iloc[row_idx]
    lower_threshold = data['Lower_Threshold'].iloc[row_idx]
    
    current_price = data['close'].iloc[row_idx]
    future_prices = data['close'].iloc[row_idx+1:row_idx+forward_check+1]
    
    # Calculate actual future movement
    max_up_move = (future_prices.max() - current_price) / current_price
    max_down_move = (current_price - future_prices.min()) / current_price
    
    if debug_this_bar:
        print(f"Bar {row_idx}: Classified as {classified_regime}")
        print(f"  Thresholds - Strong: {strong_threshold:.4f}, Weak: {weak_threshold:.4f}")
        print(f"  Actual moves - Up: {max_up_move:.4f}, Down: {max_down_move:.4f}")
        print(f"  Market conditions - Slope: {slope:.6f}, Short MA vs Upper: {short_ma:.2f} vs {upper_threshold:.2f}")
    
    # CONSERVATIVE UPGRADE logic: Only upgrade if both future move AND current conditions support it
    if 'WEAK ABOVE' in classified_regime and max_up_move >= strong_threshold:
        if debug_this_bar:
            print(f"  UPGRADED: WEAK ABOVE -> STRONG ABOVE (move {max_up_move:.4f} >= {strong_threshold:.4f})")
        return 'STRONG ABOVE'
    
    elif 'WEAK BELOW' in classified_regime and max_down_move >= strong_threshold:
        if debug_this_bar:
            print(f"  UPGRADED: WEAK BELOW -> STRONG BELOW (move {max_down_move:.4f} >= {strong_threshold:.4f})")
        return 'STRONG BELOW'
    
    elif 'BETWEEN' in classified_regime or 'CONTRACTING BETWEEN' in classified_regime or 'EXPANDING BETWEEN' in classified_regime or 'TRANSITIONING' in classified_regime:
        # CONSERVATIVE: Only upgrade BETWEEN if BOTH future move AND current market structure support it
        if (max_up_move >= strong_threshold and 
            slope > 0 and 
            short_ma > upper_threshold):
            if debug_this_bar:
                print(f"  CONSERVATIVE UPGRADE: {classified_regime} -> STRONG ABOVE (move {max_up_move:.4f} >= {strong_threshold:.4f} + bullish structure)")
            return 'STRONG ABOVE'
            
        elif (max_down_move >= strong_threshold and 
              slope < 0 and 
              short_ma < lower_threshold):
            if debug_this_bar:
                print(f"  CONSERVATIVE UPGRADE: {classified_regime} -> STRONG BELOW (move {max_down_move:.4f} >= {strong_threshold:.4f} + bearish structure)")
            return 'STRONG BELOW'
            
        elif (max_up_move >= weak_threshold and 
              slope > 0 and 
              short_ma > long_ma):
            if debug_this_bar:
                print(f"  CONSERVATIVE UPGRADE: {classified_regime} -> WEAK ABOVE (move {max_up_move:.4f} >= {weak_threshold:.4f} + mild bullish structure)")
            return 'WEAK ABOVE'
            
        elif (max_down_move >= weak_threshold and 
              slope < 0 and 
              short_ma < long_ma):
            if debug_this_bar:
                print(f"  CONSERVATIVE UPGRADE: {classified_regime} -> WEAK BELOW (move {max_down_move:.4f} >= {weak_threshold:.4f} + mild bearish structure)")
            return 'WEAK BELOW'
    
    # DOWNGRADE logic: Downgrade STRONG regimes if they don't follow through
    if 'STRONG ABOVE' in classified_regime and max_up_move < weak_threshold:
        if debug_this_bar:
            print(f"  DOWNGRADED: STRONG ABOVE -> WEAK ABOVE (move {max_up_move:.4f} < {weak_threshold:.4f})")
        return 'WEAK ABOVE'
    
    elif 'STRONG BELOW' in classified_regime and max_down_move < weak_threshold:
        if debug_this_bar:
            print(f"  DOWNGRADED: STRONG BELOW -> WEAK BELOW (move {max_down_move:.4f} < {weak_threshold:.4f})")
        return 'WEAK BELOW'
    
    # If no changes needed, return original classification
    if debug_this_bar:
        print(f"  NO CHANGE: Regime stays {classified_regime}")
    
    return classified_regime

def apply_regime_classification(data, use_perfect=False):
    """Apply regime classification to entire dataset"""
    
    # Initialize regime column
    regime_col = 'Perfect_Raw_Regime' if use_perfect else 'Live_Raw_Regime'
    data[regime_col] = 'BETWEEN'
    
    # Classify each row
    desc = "Classifying Perfect Regimes" if use_perfect else "Classifying Live Regimes"
    for i in tqdm(range(len(data)), desc=desc, ncols=80):
        if i >= CORE_PARAMS['long_ma_period']:  # Need enough data for calculations
            params = get_adaptive_thresholds(data, i, use_perfect=use_perfect)
            regime = classify_regime_excel_logic(data, i, params)
            
            # Additional validation for perfect system
            if use_perfect:
                regime = validate_perfect_regime_with_future(data, i, regime)
            
            data.loc[data.index[i], regime_col] = regime
    
    return data

def apply_transition_aware_persistence(data, regime_col, confirmed_col):
    """Apply transition-aware persistence with variable requirements"""
    
    data[confirmed_col] = np.nan  # Start empty for sequential build

    for i in range(len(data)):
        current_raw = data[regime_col].iloc[i]
        if pd.isna(current_raw):
            if i > 0:
                data.iloc[i, data.columns.get_loc(confirmed_col)] = data[confirmed_col].iloc[i-1]
            continue
        
        # Persistence for current raw regime
        if ENHANCED_FEATURES:
            if 'STRONG' in current_raw:
                needed = ENHANCED_PARAMS['strong_persistence']
            elif 'WEAK' in current_raw:
                needed = ENHANCED_PARAMS['weak_persistence']
            else:
                needed = ENHANCED_PARAMS['between_persistence']
        else:
            needed = CORE_PARAMS['base_persistence']
        
        # Check consecutive raw regimes
        count = 1  # Current bar
        for j in range(1, needed):
            if i - j < 0:
                break
            if data[regime_col].iloc[i - j] == current_raw:
                count += 1
            else:
                break
        
        if count >= needed:
            data.iloc[i, data.columns.get_loc(confirmed_col)] = current_raw
        else:
            if i > 0:
                data.iloc[i, data.columns.get_loc(confirmed_col)] = data[confirmed_col].iloc[i-1]
    
    data[confirmed_col] = data[confirmed_col].ffill().bfill()
    
    return data

def calculate_scoring_metrics(data):
    """Calculate comprehensive scoring metrics comparing live vs perfect systems"""
    
    metrics = {}
    
    # Get valid rows (where both systems have regimes)
    valid_mask = (data['Live_Confirmed_Regime'].notna() & 
                  data['Perfect_Confirmed_Regime'].notna())
    
    valid_data = data[valid_mask].copy()
    
    if len(valid_data) == 0:
        return pd.DataFrame([metrics])
    
    # Overall accuracy
    exact_matches = valid_data['Live_Confirmed_Regime'] == valid_data['Perfect_Confirmed_Regime']
    metrics['Overall_Accuracy'] = exact_matches.sum() / len(valid_data) * 100
    
    # Calculate metrics for each regime type
    regime_types = ['STRONG ABOVE', 'WEAK ABOVE', 'STRONG BELOW', 'WEAK BELOW', 
                    'CONTRACTING BETWEEN', 'EXPANDING BETWEEN', 'TRANSITIONING', 'BETWEEN']
    
    for regime in regime_types:
        # Precision: When live says regime X, how often is perfect also regime X?
        live_regime_mask = valid_data['Live_Confirmed_Regime'] == regime
        if live_regime_mask.sum() > 0:
            precision = (valid_data[live_regime_mask]['Perfect_Confirmed_Regime'] == regime).sum() / live_regime_mask.sum()
            metrics[f'{regime}_Precision'] = precision * 100
        else:
            metrics[f'{regime}_Precision'] = 0
        
        # Recall: When perfect says regime X, how often does live also say regime X?
        perfect_regime_mask = valid_data['Perfect_Confirmed_Regime'] == regime
        if perfect_regime_mask.sum() > 0:
            recall = (valid_data[perfect_regime_mask]['Live_Confirmed_Regime'] == regime).sum() / perfect_regime_mask.sum()
            metrics[f'{regime}_Recall'] = recall * 100
        else:
            metrics[f'{regime}_Recall'] = 0
    
    # Directional accuracy (bullish vs bearish vs neutral)
    def get_direction(regime):
        if pd.isna(regime) or regime is None:
            return 'NEUTRAL'
        if 'ABOVE' in regime:
            return 'BULLISH'
        elif 'BELOW' in regime:
            return 'BEARISH'
        else:
            return 'NEUTRAL'
    
    valid_data['Live_Direction'] = valid_data['Live_Confirmed_Regime'].apply(get_direction)
    valid_data['Perfect_Direction'] = valid_data['Perfect_Confirmed_Regime'].apply(get_direction)
    
    direction_matches = valid_data['Live_Direction'] == valid_data['Perfect_Direction']
    metrics['Directional_Accuracy'] = direction_matches.sum() / len(valid_data) * 100
    
    # Timing lag analysis
    perfect_changes = valid_data['Perfect_Confirmed_Regime'].ne(valid_data['Perfect_Confirmed_Regime'].shift())
    change_indices = valid_data.index[perfect_changes]
    
    lag_values = []
    for change_idx in change_indices[1:]:
        perfect_regime = valid_data.loc[change_idx, 'Perfect_Confirmed_Regime']
        idx_pos = valid_data.index.get_loc(change_idx)
        for lag in range(0, min(20, len(valid_data) - idx_pos)):
            if idx_pos + lag < len(valid_data):
                check_idx = valid_data.index[idx_pos + lag]
                if valid_data.loc[check_idx, 'Live_Confirmed_Regime'] == perfect_regime:
                    lag_values.append(lag)
                    break
    
    if lag_values:
        metrics['Average_Lag_Bars'] = np.mean(lag_values)
        metrics['Median_Lag_Bars'] = np.median(lag_values)
        metrics['Max_Lag_Bars'] = np.max(lag_values)
    else:
        metrics['Average_Lag_Bars'] = 0
        metrics['Median_Lag_Bars'] = 0
        metrics['Max_Lag_Bars'] = 0
    
    # Weighted accuracy (STRONG regimes weighted more)
    weights = {
        'STRONG ABOVE': 2.0,
        'STRONG BELOW': 2.0,
        'WEAK ABOVE': 1.5,
        'WEAK BELOW': 1.5,
        'CONTRACTING BETWEEN': 1.0,
        'EXPANDING BETWEEN': 1.0,
        'TRANSITIONING': 1.0,
        'BETWEEN': 0.5
    }
    
    valid_data['Weight'] = valid_data['Perfect_Confirmed_Regime'].map(weights)
    valid_data['Weighted_Match'] = exact_matches * valid_data['Weight']
    
    metrics['Weighted_Accuracy'] = valid_data['Weighted_Match'].sum() / valid_data['Weight'].sum() * 100
    
    # Add timestamp
    metrics['Calculation_Timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    return pd.DataFrame([metrics])

def main():
    """Main execution function"""
    
    print(f"Enhanced Features: {'ENABLED' if ENHANCED_FEATURES else 'DISABLED'}")
    print(f"Perfect Benchmark System: ENABLED")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load data
    with tqdm(total=1, desc="Loading Data", ncols=80) as pbar:
        data = load_csv_data(DATA_FILE, TIMEFRAME)
        pbar.update(1)

    # Slice data if test_slice is set
    if TEST_SLICE is not None:
        data = data.tail(TEST_SLICE)
        print(f"Using last {TEST_SLICE} rows for testing")

    # Calculate helper columns (includes both live and perfect thresholds)
    with tqdm(total=1, desc="Calculating Helper Columns", ncols=80) as pbar:
        data = calculate_helper_columns(data)
        pbar.update(1)

    # Apply LIVE regime classification
    data = apply_regime_classification(data, use_perfect=False)
    
    # Apply PERFECT regime classification
    data = apply_regime_classification(data, use_perfect=True)

    # CRITICAL: Shift labels by 1 bar to eliminate lookahead bias (for live system only)
    data['Live_Raw_Regime'] = data['Live_Raw_Regime'].shift(1)
    print("Applied 1-bar shift to live system to eliminate lookahead bias")

    # Apply transition-aware persistence to both systems
    with tqdm(total=1, desc="Applying Persistence to Live System", ncols=80) as pbar:
        data = apply_transition_aware_persistence(data, 'Live_Raw_Regime', 'Live_Confirmed_Regime')
        pbar.update(1)
        
    with tqdm(total=1, desc="Applying Persistence to Perfect System", ncols=80) as pbar:
        data = apply_transition_aware_persistence(data, 'Perfect_Raw_Regime', 'Perfect_Confirmed_Regime')
        pbar.update(1)

    # Create dummy columns for visualization (both systems)
    regimes = ["STRONG ABOVE", "WEAK ABOVE", "STRONG BELOW", "WEAK BELOW", 
               "CONTRACTING BETWEEN", "EXPANDING BETWEEN", "TRANSITIONING", "BETWEEN"]
    
    # Live system dummies
    for regime in regimes:
        col_name = "Live_" + regime.replace(" ", "_") + "_Dummy"
        data[col_name] = np.where(data['Live_Confirmed_Regime'] == regime, 1000000, np.nan)
    
    # Perfect system dummies
    for regime in regimes:
        col_name = "Perfect_" + regime.replace(" ", "_") + "_Dummy"
        data[col_name] = np.where(data['Perfect_Confirmed_Regime'] == regime, 2000000, np.nan)

    # Add numbered index for continuous plotting
    data['Index'] = range(len(data))

    # Calculate scoring metrics
    print("\nCalculating Scoring Metrics...")
    scoring_df = calculate_scoring_metrics(data)
    
    # Save scoring metrics
    scoring_df.to_csv(SCORING_FILE, index=False)
    print(f"Saved scoring metrics to {SCORING_FILE}")
    
    # Print key metrics
    print(f"\nKey Performance Metrics:")
    print(f"Overall Accuracy: {scoring_df['Overall_Accuracy'].iloc[0]:.2f}%")
    print(f"Directional Accuracy: {scoring_df['Directional_Accuracy'].iloc[0]:.2f}%")
    print(f"Weighted Accuracy: {scoring_df['Weighted_Accuracy'].iloc[0]:.2f}%")
    print(f"Average Lag: {scoring_df['Average_Lag_Bars'].iloc[0]:.1f} bars")

    # Reset index to include Date as column
    data = data.reset_index()

    # Split Date into separate Date and Time columns with timestamp
    data['Date_Only'] = data['Date'].dt.date
    data['Time'] = data['Date'].dt.time
    data['Full_Timestamp'] = data['Date'].dt.strftime('%Y-%m-%d %H:%M:%S')

    # Export to CSV with both systems side by side
    live_dummy_columns = ["Live_" + regime.replace(" ", "_") + "_Dummy" for regime in regimes]
    perfect_dummy_columns = ["Perfect_" + regime.replace(" ", "_") + "_Dummy" for regime in regimes]
    
    output_columns = [
        'Index', 'Full_Timestamp', 'Date_Only', 'Time', 
        'open', 'high', 'low', 'close',
        # Live system columns
        'Live_Confirmed_Regime', 'Live_Raw_Regime',
        # Perfect system columns  
        'Perfect_Confirmed_Regime', 'Perfect_Raw_Regime',
        # Technical indicators
        'SMA_5', 'SMA_13', 'SMA_13_Slope', 'ATR_5', 
        'Volatility_Ratio', 'Dynamic_Multiplier',
        # Live thresholds
        'Dynamic_Bull_Weak', 'Dynamic_Bull_Strong', 
        'Dynamic_Bear_Weak', 'Dynamic_Bear_Strong',
        'Dynamic_Vol_Low', 'Dynamic_Vol_High',
        # Perfect thresholds
        'Perfect_Bull_Weak', 'Perfect_Bull_Strong',
        'Perfect_Bear_Weak', 'Perfect_Bear_Strong', 
        'Perfect_Vol_Low', 'Perfect_Vol_High'
    ] + live_dummy_columns + perfect_dummy_columns
    
    data[output_columns].to_csv(OUTPUT_FILE, index=False)
    print(f"\nExported labeled data to {OUTPUT_FILE}")

    # DEBUG: Analyze threshold differences
    print("\n" + "="*60)
    print("DEBUG ANALYSIS")
    print("="*60)

    # Volatility comparison
    valid_data = data.dropna(subset=['Volatility_Ratio', 'Up_Volatility_Ratio', 'Down_Volatility_Ratio'])
    if len(valid_data) > 0:
        print(f"\nVolatility Ratio Comparison:")
        print(f"  Regular Volatility - Mean: {valid_data['Volatility_Ratio'].mean():.3f}, Std: {valid_data['Volatility_Ratio'].std():.3f}")
        print(f"  Up Volatility - Mean: {valid_data['Up_Volatility_Ratio'].mean():.3f}, Std: {valid_data['Up_Volatility_Ratio'].std():.3f}")
        print(f"  Down Volatility - Mean: {valid_data['Down_Volatility_Ratio'].mean():.3f}, Std: {valid_data['Down_Volatility_Ratio'].std():.3f}")

    # Move threshold analysis
    move_data = data.dropna(subset=['Dynamic_Strong_Move_Threshold', 'Dynamic_Weak_Move_Threshold'])
    if len(move_data) > 0:
        print(f"\nMove Threshold Analysis:")
        print(f"  Strong Move Threshold - Mean: {move_data['Dynamic_Strong_Move_Threshold'].mean():.4f} ({move_data['Dynamic_Strong_Move_Threshold'].mean()*100:.2f}%)")
        print(f"  Weak Move Threshold - Mean: {move_data['Dynamic_Weak_Move_Threshold'].mean():.4f} ({move_data['Dynamic_Weak_Move_Threshold'].mean()*100:.2f}%)")
        print(f"  Strong Move Threshold - Range: {move_data['Dynamic_Strong_Move_Threshold'].min():.4f} to {move_data['Dynamic_Strong_Move_Threshold'].max():.4f}")

    # Compare thresholds used in classification
    bull_thresh_data = data.dropna(subset=['Dynamic_Vol_High_Bull', 'Dynamic_Vol_High_Bear'])
    if len(bull_thresh_data) > 0:
        print(f"\nDirectional Volatility Thresholds:")
        print(f"  Bull High Volatility Threshold - Mean: {bull_thresh_data['Dynamic_Vol_High_Bull'].mean():.3f}")
        print(f"  Bear High Volatility Threshold - Mean: {bull_thresh_data['Dynamic_Vol_High_Bear'].mean():.3f}")
        print(f"  Regular High Volatility Threshold - Mean: {bull_thresh_data['Dynamic_Vol_High'].mean():.3f}")

    # Check how often directional volatility conditions are met
    bull_vol_met = (valid_data['Up_Volatility_Ratio'] > 0.5).sum()
    bear_vol_met = (valid_data['Down_Volatility_Ratio'] > 0.5).sum()
    total_bars = len(valid_data)

    print(f"\nDirectional Volatility Condition Analysis:")
    print(f"  Bars with Up_Volatility > 0.5: {bull_vol_met} ({bull_vol_met/total_bars*100:.2f}%)")
    print(f"  Bars with Down_Volatility > 0.5: {bear_vol_met} ({bear_vol_met/total_bars*100:.2f}%)")
    print(f"  Bars with Regular_Volatility > 0.5: {(valid_data['Volatility_Ratio'] > 0.5).sum()} ({(valid_data['Volatility_Ratio'] > 0.5).sum()/total_bars*100:.2f}%)")

    # Print regime distribution comparison
    print("\n" + "="*60)
    print("REGIME DISTRIBUTION COMPARISON")
    print("="*60)
    
    print("\nLive System Distribution:")
    live_counts = data['Live_Confirmed_Regime'].value_counts()
    total_live = len(data.dropna(subset=['Live_Confirmed_Regime']))
    for regime, count in live_counts.items():
        percentage = (count / total_live) * 100
        print(f"{regime:25} {count:6} ({percentage:5.2f}%)")
    
    print("\nPerfect System Distribution:")
    perfect_counts = data['Perfect_Confirmed_Regime'].value_counts()
    total_perfect = len(data.dropna(subset=['Perfect_Confirmed_Regime']))
    for regime, count in perfect_counts.items():
        percentage = (count / total_perfect) * 100
        print(f"{regime:25} {count:6} ({percentage:5.2f}%)")

    # Print system analysis
    print("\n" + "="*60)
    print("SYSTEM ANALYSIS")
    print("="*60)
    
    # Slope analysis
    valid_slopes = data['SMA_13_Slope'].dropna()
    print(f"\nSlope Statistics:")
    print(f"  Range: {valid_slopes.min():.6f} to {valid_slopes.max():.6f}")
    print(f"  Mean: {valid_slopes.mean():.6f}")
    print(f"  Std: {valid_slopes.std():.6f}")
    
    # Volatility analysis  
    valid_vol = data['Volatility_Ratio'].dropna()
    print(f"\nVolatility Ratio Statistics:")
    print(f"  Range: {valid_vol.min():.3f} to {valid_vol.max():.3f}")
    print(f"  Mean: {valid_vol.mean():.3f}")
    print(f"  Std: {valid_vol.std():.3f}")
    
    # Perfect system specific analysis
    print(f"\nPerfect System Analysis:")
    print(f"  Adaptive forward lookback: {PERFECT_PARAMS['adaptive_forward']}")
    print(f"  Forward bars range: {PERFECT_PARAMS['min_forward_bars']}-{PERFECT_PARAMS['max_forward_bars']}")
    print(f"  Dynamic move thresholds: {PERFECT_PARAMS['strong_move_percentile']*100:.0f}th percentile (strong), {PERFECT_PARAMS['weak_move_percentile']*100:.0f}th percentile (weak)")
    print(f"  Move calculation lookback: {PERFECT_PARAMS['move_lookback']} bars")
    
    print(f"\nPersistence Settings:")
    if ENHANCED_FEATURES:
        print(f"  STRONG regimes: {ENHANCED_PARAMS['strong_persistence']} bars")
        print(f"  WEAK regimes: {ENHANCED_PARAMS['weak_persistence']} bars") 
        print(f"  BETWEEN regimes: {ENHANCED_PARAMS['between_persistence']} bars")
    else:
        print(f"  All regimes: {CORE_PARAMS['base_persistence']} bars")
    
    print(f"\nTransition-Aware Logic: ENABLED")
    print(f"  Regime changes use new regime's persistence requirement")
    print(f"  Continuing regimes use minimal (1 bar) persistence")

    print("\n" + "="*60)
    print("Processing completed successfully!")
    print(f"Main output: {OUTPUT_FILE}")
    print(f"Scoring metrics: {SCORING_FILE}")
    print("="*60)

if __name__ == "__main__":
    main()

