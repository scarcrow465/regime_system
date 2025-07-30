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
TEST_SLICE = 20000  # Number of rows to use from the end of the dataset (set to None for full dataset)
DATA_FILE = 'combined_NQ_15m_data.csv'  # Path to your CSV file
OUTPUT_FILE = 'ma_regime_labeled_data_with_perfect_v14_test.csv'  # Output CSV file name
SCORING_FILE = 'regime_scoring_metrics_v14_test.csv'  # Scoring metrics output
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
    'session_adaptation': False          # Enable session-based adjustments
}

# Perfect system parameters
PERFECT_PARAMS = {
    'min_forward_bars': 10,               # Minimum bars to look forward (reduced from 100)
    'max_forward_bars': 30,               # Maximum bars to look forward (reduced from 200)
    'adaptive_forward': True,             # Use adaptive forward looking based on volatility
    'move_lookback': 500,                 # Lookback for calculating dynamic move thresholds
    'strong_move_percentile': 0.85,       # Percentile for strong move threshold
    'weak_move_percentile': 0.65,         # Percentile for weak move threshold
    'validation_forward_bars': 20,        # Bars to validate if regime actually plays out
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

    # Calculate directional True Range for up/down CANDLES (not just moves)
    data['Up_Candle'] = data['close'] > data['close'].shift(1)
    data['Down_Candle'] = data['close'] < data['close'].shift(1)

    # Calculate True Range for up and down candles separately
    data['Up_TR'] = np.where(data['Up_Candle'], data['TR'], 0)
    data['Down_TR'] = np.where(data['Down_Candle'], data['TR'], 0)

    # Directional ATR (5-period) - average TR during up/down candles
    data['Up_ATR_5'] = data['Up_TR'].rolling(window=CORE_PARAMS['short_atr_period']).sum() / data['Up_Candle'].rolling(window=CORE_PARAMS['short_atr_period']).sum()
    data['Down_ATR_5'] = data['Down_TR'].rolling(window=CORE_PARAMS['short_atr_period']).sum() / data['Down_Candle'].rolling(window=CORE_PARAMS['short_atr_period']).sum()

    # Handle division by zero (when no up or down candles in window)
    data['Up_ATR_5'] = data['Up_ATR_5'].fillna(data['ATR_5'])
    data['Down_ATR_5'] = data['Down_ATR_5'].fillna(data['ATR_5'])

    # Directional volatility ratios
    data['Up_Volatility_Ratio'] = data['Up_ATR_5'] / data['ATR_50']
    data['Down_Volatility_Ratio'] = data['Down_ATR_5'] / data['ATR_50']
    
    # W56: Dynamic multiplier = 0.15 + 0.2 / (1 + 10000 * ABS(V56))
    data['Dynamic_Multiplier'] = 0.15 + 0.2 / (1 + 10000 * np.abs(data['SMA_13_Slope']))
    
    # Crossover thresholds
    data['Upper_Threshold'] = data['SMA_13'] + data['Dynamic_Multiplier'] * data['ATR_5']
    data['Lower_Threshold'] = data['SMA_13'] - data['Dynamic_Multiplier'] * data['ATR_5']
    
    # Calculate adaptive lookback based on volatility regime
    vol_regime_factor = data['Volatility_Ratio'].rolling(window=50).mean()
    base_lookback = CORE_PARAMS['base_slope_lookback']
    
    # Shorter lookbacks in volatile periods, longer in stable periods
    data['Adaptive_Slope_Lookback'] = np.where(
        vol_regime_factor > 1.2, int(base_lookback * 0.7),  # Volatile: shorter lookback
        np.where(vol_regime_factor < 0.8, int(base_lookback * 1.3), base_lookback)  # Stable: longer
    ).astype(int)
    
    # Create volatility filter mask for "normal" periods only
    vol_filter_min = CORE_PARAMS['volatility_filter_min']
    vol_filter_max = CORE_PARAMS['volatility_filter_max']

    # Calculate dynamic perfect move thresholds
    data['Dynamic_Strong_Move_Threshold'] = np.nan
    data['Dynamic_Weak_Move_Threshold'] = np.nan

    # Calculate historical move percentiles for perfect system validation
    move_lookback = PERFECT_PARAMS['move_lookback']
    for i in tqdm(range(len(data)), desc="Calculating Dynamic Move Thresholds", ncols=80):
        if i >= move_lookback:
            # Calculate actual moves that occurred over the lookback period
            historical_moves = []
            for j in range(i-move_lookback, i):
                if j + 20 < len(data):  # Need 20 bars of future data
                    future_high = data['high'].iloc[j:j+20].max()
                    future_low = data['low'].iloc[j:j+20].min()
                    current_close = data['close'].iloc[j]
                    
                    up_move = (future_high - current_close) / current_close
                    down_move = (current_close - future_low) / current_close
                    max_move = max(up_move, down_move)
                    historical_moves.append(max_move)
            
            if len(historical_moves) >= 100:
                moves_series = pd.Series(historical_moves)
                strong_thresh = moves_series.quantile(PERFECT_PARAMS['strong_move_percentile'])
                weak_thresh = moves_series.quantile(PERFECT_PARAMS['weak_move_percentile'])
                
                data.loc[data.index[i], 'Dynamic_Strong_Move_Threshold'] = strong_thresh
                data.loc[data.index[i], 'Dynamic_Weak_Move_Threshold'] = weak_thresh
                
                # Debug: Print sample thresholds every 1000 bars
                if i % 1000 == 0:
                    print(f"Bar {i}: Strong threshold: {strong_thresh:.4f} ({strong_thresh*100:.2f}%), Weak threshold: {weak_thresh:.4f} ({weak_thresh*100:.2f}%)")
                    print(f"  Sample moves: min={min(historical_moves):.4f}, max={max(historical_moves):.4f}, mean={np.mean(historical_moves):.4f}")
    
    # Initialize dynamic threshold columns (for live system)
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
    
    # Initialize perfect system threshold columns
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
    for i in tqdm(range(len(data)), desc="Calculating Live Dynamic Thresholds", ncols=80):
        if i < base_lookback:
            continue
            
        # Get adaptive lookback for this period
        lookback = int(data['Adaptive_Slope_Lookback'].iloc[i])
        lookback = min(lookback, i)  # Don't exceed available data
        
        # Get recent data window (BACKWARD LOOKING)
        recent_slopes = data['SMA_13_Slope'].iloc[i-lookback:i]
        recent_vol = data['Volatility_Ratio'].iloc[i-lookback:i]
        
        # Filter for normal volatility periods only
        normal_vol_mask = (recent_vol >= vol_filter_min) & (recent_vol <= vol_filter_max)
        
        if normal_vol_mask.sum() < 20:  # Need minimum data points
            # Fallback to all data if too little normal volatility data
            filtered_slopes = recent_slopes
        else:
            filtered_slopes = recent_slopes[normal_vol_mask]
        
        # Separate bull and bear slopes from filtered data
        bull_slopes = filtered_slopes[filtered_slopes > 0]
        bear_slopes = filtered_slopes[filtered_slopes < 0]
        
        # Calculate bull thresholds (positive slopes)
        if len(bull_slopes) >= 10:  # Need minimum bull slope data
            data.loc[data.index[i], 'Dynamic_Bull_Weak'] = bull_slopes.quantile(CORE_PARAMS['bull_weak_percentile'])
            data.loc[data.index[i], 'Dynamic_Bull_Strong'] = bull_slopes.quantile(CORE_PARAMS['bull_strong_percentile'])
        
        # Calculate bear thresholds (negative slopes - use absolute values)
        if len(bear_slopes) >= 10:  # Need minimum bear slope data
            bear_slopes_abs = abs(bear_slopes)
            data.loc[data.index[i], 'Dynamic_Bear_Weak'] = bear_slopes_abs.quantile(CORE_PARAMS['bear_weak_percentile'])
            data.loc[data.index[i], 'Dynamic_Bear_Strong'] = bear_slopes_abs.quantile(CORE_PARAMS['bear_strong_percentile'])
        
        # Calculate volatility thresholds (directional logic)
        vol_lookback = CORE_PARAMS['volatility_lookback']
        if i >= vol_lookback:
            vol_window = data['Volatility_Ratio'].iloc[i-vol_lookback:i]
            data.loc[data.index[i], 'Dynamic_Vol_Low'] = vol_window.quantile(CORE_PARAMS['volatility_low_percentile'])
            data.loc[data.index[i], 'Dynamic_Vol_High'] = vol_window.quantile(CORE_PARAMS['volatility_high_percentile'])
            
            # Calculate directional volatility thresholds using directional ATR
            up_vol_window = data['Up_Volatility_Ratio'].iloc[i-vol_lookback:i]
            down_vol_window = data['Down_Volatility_Ratio'].iloc[i-vol_lookback:i]

            data.loc[data.index[i], 'Dynamic_Vol_High_Bull'] = up_vol_window.quantile(CORE_PARAMS['volatility_high_bull_percentile'])
            data.loc[data.index[i], 'Dynamic_Vol_High_Bear'] = down_vol_window.quantile(CORE_PARAMS['volatility_high_bear_percentile'])
            data.loc[data.index[i], 'Dynamic_Vol_Low_Bull'] = up_vol_window.quantile(CORE_PARAMS['volatility_low_bull_percentile'])
            data.loc[data.index[i], 'Dynamic_Vol_Low_Bear'] = down_vol_window.quantile(CORE_PARAMS['volatility_low_bear_percentile'])
    
    # Calculate perfect system thresholds (FORWARD LOOKING)
    for i in tqdm(range(len(data)), desc="Calculating Perfect Forward Thresholds", ncols=80):
        # Determine adaptive forward lookback based on current volatility
        current_vol = data['Volatility_Ratio'].iloc[i] if i < len(data) else 1.0

        # Handle NaN values by using a default
        if pd.isna(current_vol):
            current_vol = 1.0  # Default to neutral volatility

        if PERFECT_PARAMS['adaptive_forward']:
            # Fewer forward bars in volatile periods, more in stable
            if current_vol > 1.3:
                forward_bars = PERFECT_PARAMS['min_forward_bars']  # 10 bars in volatile
            elif current_vol < 0.7:
                forward_bars = PERFECT_PARAMS['max_forward_bars']   # 30 bars in stable
            else:
                # Linear interpolation between 10-30 bars
                forward_bars = int(PERFECT_PARAMS['min_forward_bars'] + 
                                (PERFECT_PARAMS['max_forward_bars'] - PERFECT_PARAMS['min_forward_bars']) * 
                                (1.3 - current_vol) / 0.6)
        else:
            forward_bars = PERFECT_PARAMS['validation_forward_bars']  # Default 20 bars
        
        # Ensure we don't exceed data bounds
        if i + forward_bars >= len(data):
            forward_bars = len(data) - i - 1
            
        if forward_bars < 20:  # Not enough future data
            continue
        
        # Get future data window (FORWARD LOOKING)
        future_slopes = data['SMA_13_Slope'].iloc[i+1:i+forward_bars+1]
        future_vol = data['Volatility_Ratio'].iloc[i+1:i+forward_bars+1]
        
        # Filter for normal volatility periods
        normal_vol_mask = (future_vol >= vol_filter_min) & (future_vol <= vol_filter_max)
        
        if normal_vol_mask.sum() < 10:
            filtered_slopes = future_slopes
        else:
            filtered_slopes = future_slopes[normal_vol_mask]
        
        # Separate bull and bear slopes
        bull_slopes = filtered_slopes[filtered_slopes > 0]
        bear_slopes = filtered_slopes[filtered_slopes < 0]
        
        # Calculate perfect thresholds
        if len(bull_slopes) >= 5:
            data.loc[data.index[i], 'Perfect_Bull_Weak'] = bull_slopes.quantile(CORE_PARAMS['bull_weak_percentile'])
            data.loc[data.index[i], 'Perfect_Bull_Strong'] = bull_slopes.quantile(CORE_PARAMS['bull_strong_percentile'])
        
        if len(bear_slopes) >= 5:
            bear_slopes_abs = abs(bear_slopes)
            data.loc[data.index[i], 'Perfect_Bear_Weak'] = bear_slopes_abs.quantile(CORE_PARAMS['bear_weak_percentile'])
            data.loc[data.index[i], 'Perfect_Bear_Strong'] = bear_slopes_abs.quantile(CORE_PARAMS['bear_strong_percentile'])
        
        # Future volatility thresholds
        if forward_bars >= 50:
            future_vol_window = data['Volatility_Ratio'].iloc[i+1:i+min(forward_bars, 100)+1]
            data.loc[data.index[i], 'Perfect_Vol_Low'] = future_vol_window.quantile(CORE_PARAMS['volatility_low_percentile'])
            data.loc[data.index[i], 'Perfect_Vol_High'] = future_vol_window.quantile(CORE_PARAMS['volatility_high_percentile'])
            
            # Calculate future directional volatility thresholds
            future_up_vol = data['Up_Volatility_Ratio'].iloc[i+1:i+min(forward_bars, 100)+1]
            future_down_vol = data['Down_Volatility_Ratio'].iloc[i+1:i+min(forward_bars, 100)+1]

            data.loc[data.index[i], 'Perfect_Vol_High_Bull'] = future_up_vol.quantile(CORE_PARAMS['volatility_high_bull_percentile'])
            data.loc[data.index[i], 'Perfect_Vol_High_Bear'] = future_down_vol.quantile(CORE_PARAMS['volatility_high_bear_percentile'])
            data.loc[data.index[i], 'Perfect_Vol_Low_Bull'] = future_up_vol.quantile(CORE_PARAMS['volatility_low_bull_percentile'])
            data.loc[data.index[i], 'Perfect_Vol_Low_Bear'] = future_down_vol.quantile(CORE_PARAMS['volatility_low_bear_percentile'])
    
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
    """Exact Excel nested IF logic with directional slope thresholds"""
    
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
    
    # Get directional volatility
    up_volatility_ratio = data['Up_Volatility_Ratio'].iloc[row_idx]
    down_volatility_ratio = data['Down_Volatility_Ratio'].iloc[row_idx]

    # STRONG ABOVE: slope > bull_strong_threshold AND short_ma > upper_threshold AND UP volatility > bull_high_threshold
    if (slope > params['bull_strong_threshold'] and 
        short_ma > upper_threshold and 
        up_volatility_ratio > params['volatility_high_bull_threshold']):
        return 'STRONG ABOVE'
    
    # WEAK ABOVE: slope > bull_weak_threshold AND short_ma > upper_threshold
    elif (slope > params['bull_weak_threshold'] and 
          short_ma > upper_threshold):
        return 'WEAK ABOVE'

    # STRONG BELOW: slope < -bear_strong_threshold AND short_ma < lower_threshold AND DOWN volatility > bear_high_threshold
    elif (slope < -params['bear_strong_threshold'] and 
        short_ma < lower_threshold and 
        down_volatility_ratio > params['volatility_high_bear_threshold']):
        return 'STRONG BELOW'
    
    # WEAK BELOW: slope < -bear_weak_threshold AND short_ma < lower_threshold
    elif (slope < -params['bear_weak_threshold'] and 
          short_ma < lower_threshold):
        return 'WEAK BELOW'
    
    # CONTRACTING BETWEEN: abs(slope) <= min(bull_weak, bear_weak) AND volatility < low_threshold
    elif (abs(slope) <= min(params['bull_weak_threshold'], params['bear_weak_threshold']) and 
          volatility_ratio < params['volatility_low_threshold']):
        return 'CONTRACTING BETWEEN'
    
    # EXPANDING BETWEEN: abs(slope) <= min(bull_weak, bear_weak) AND volatility > high_threshold
    elif (abs(slope) <= min(params['bull_weak_threshold'], params['bear_weak_threshold']) and 
          volatility_ratio > params['volatility_high_threshold']):
        return 'EXPANDING BETWEEN'
    
    # TRANSITIONING: abs(slope) <= min(bull_weak, bear_weak) AND abs(short_ma - long_ma) <= 0.5 * dynamic_multiplier * atr_5
    elif (abs(slope) <= min(params['bull_weak_threshold'], params['bear_weak_threshold']) and 
          abs(short_ma - long_ma) <= params['transitioning_factor'] * dynamic_multiplier * atr_5):
        return 'TRANSITIONING'
    
    # Default case
    else:
        return 'BETWEEN'

def validate_perfect_regime_with_future(data, row_idx, classified_regime):
    """Perfect regime validation - identify optimal regime based on what actually happens"""
    
    # Check if we have enough future data
    validation_bars = PERFECT_PARAMS['validation_forward_bars']
    if row_idx + validation_bars >= len(data):
        return classified_regime  # Not enough data to validate
    
    # Get current market conditions
    current_price = data['close'].iloc[row_idx]
    slope = data['SMA_13_Slope'].iloc[row_idx]
    short_ma = data['SMA_5'].iloc[row_idx]
    long_ma = data['SMA_13'].iloc[row_idx]
    upper_threshold = data['Upper_Threshold'].iloc[row_idx]
    lower_threshold = data['Lower_Threshold'].iloc[row_idx]
    volatility_ratio = data['Volatility_Ratio'].iloc[row_idx]
    
    # Get dynamic thresholds
    strong_threshold = data['Dynamic_Strong_Move_Threshold'].iloc[row_idx]
    weak_threshold = data['Dynamic_Weak_Move_Threshold'].iloc[row_idx]
    
    # Use fallbacks if thresholds not available
    if pd.isna(strong_threshold):
        strong_threshold = 0.008
    if pd.isna(weak_threshold):
        weak_threshold = 0.004
    
    # Look at what happens in the immediate future (next 20 bars)
    future_prices = data['close'].iloc[row_idx+1:row_idx+validation_bars+1]
    future_highs = data['high'].iloc[row_idx+1:row_idx+validation_bars+1]
    future_lows = data['low'].iloc[row_idx+1:row_idx+validation_bars+1]
    
    # Calculate actual future movement
    max_future_high = future_highs.max()
    min_future_low = future_lows.min()
    
    max_up_move = (max_future_high - current_price) / current_price
    max_down_move = (current_price - min_future_low) / current_price
    
    # Also check if the move is sustained (not just a spike)
    # Look at where price settles after half the validation period
    halfway_point = validation_bars // 2
    if halfway_point > 0:
        sustained_prices = data['close'].iloc[row_idx+halfway_point:row_idx+validation_bars+1]
        avg_sustained_price = sustained_prices.mean()
        sustained_up_move = (avg_sustained_price - current_price) / current_price
        sustained_down_move = (current_price - avg_sustained_price) / current_price
    else:
        sustained_up_move = 0
        sustained_down_move = 0
    
    # PERFECT REGIME LOGIC: What regime best describes what actually happens?
    
    # STRONG ABOVE: Strong upward move that sustains
    if (max_up_move >= strong_threshold and 
        sustained_up_move >= weak_threshold and
        slope > 0 and short_ma > upper_threshold):
        return 'STRONG ABOVE'
    
    # STRONG BELOW: Strong downward move that sustains
    elif (max_down_move >= strong_threshold and 
          sustained_down_move >= weak_threshold and
          slope < 0 and short_ma < lower_threshold):
        return 'STRONG BELOW'
    
    # WEAK ABOVE: Moderate upward move with bullish structure
    elif (max_up_move >= weak_threshold and 
          slope > 0 and short_ma > long_ma):
        return 'WEAK ABOVE'
    
    # WEAK BELOW: Moderate downward move with bearish structure  
    elif (max_down_move >= weak_threshold and
          slope < 0 and short_ma < long_ma):
        return 'WEAK BELOW'
    
    # EXPANDING BETWEEN: High volatility with no clear direction
    elif (max(max_up_move, max_down_move) >= weak_threshold and
          abs(sustained_up_move - sustained_down_move) < weak_threshold/2 and
          volatility_ratio > 1.2):
        return 'EXPANDING BETWEEN'
    
    # CONTRACTING BETWEEN: Low volatility, minimal movement
    elif (max(max_up_move, max_down_move) < weak_threshold/2 and
          volatility_ratio < 0.8):
        return 'CONTRACTING BETWEEN'
    
    # TRANSITIONING: Market is changing direction
    elif (abs(slope) < 0.0001 and 
          abs(short_ma - long_ma) / current_price < 0.001):
        return 'TRANSITIONING'
    
    # Default: BETWEEN - no clear regime
    else:
        return 'BETWEEN'

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
    
    data[confirmed_col] = data[regime_col].copy()
    
    for i in range(len(data)):
        if i < 2:  # Not enough history
            continue
            
        current_regime = data[regime_col].iloc[i]
        previous_confirmed = data[confirmed_col].iloc[i-1] if i > 0 else current_regime
        
        # Check if regime is changing
        if current_regime != previous_confirmed:
            # Regime change detected - use new regime's persistence requirement
            if ENHANCED_FEATURES:
                if 'STRONG' in current_regime:
                    persistence_needed = ENHANCED_PARAMS['strong_persistence']
                elif 'WEAK' in current_regime:
                    persistence_needed = ENHANCED_PARAMS['weak_persistence']
                else:  # BETWEEN regimes
                    persistence_needed = ENHANCED_PARAMS['between_persistence']
            else:
                persistence_needed = CORE_PARAMS['base_persistence']
        else:
            # Same regime - minimal persistence
            persistence_needed = 1
        
        # Check if we have enough history
        if i < persistence_needed:
            continue
            
        # Check persistence for the new regime
        recent_regimes = data[regime_col].iloc[i-persistence_needed+1:i+1]
        
        if len(recent_regimes.unique()) == 1 and recent_regimes.iloc[-1] == current_regime:
            # New regime has persisted long enough
            data.loc[data.index[i], confirmed_col] = current_regime
        else:
            # Keep previous confirmed regime
            data.loc[data.index[i], confirmed_col] = previous_confirmed
    
    return data

def calculate_scoring_metrics(data):
    """Calculate comprehensive scoring metrics comparing live vs perfect systems"""
    
    metrics = {}
    
    # Get valid rows (where both systems have regimes)
    valid_mask = (data['Live_Confirmed_Regime'].notna() & 
                  data['Perfect_Confirmed_Regime'].notna() &
                  (data['Live_Confirmed_Regime'] != 'BETWEEN') |
                  (data['Perfect_Confirmed_Regime'] != 'BETWEEN'))
    
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
    # Find regime change points in perfect system
    perfect_changes = valid_data['Perfect_Confirmed_Regime'].ne(valid_data['Perfect_Confirmed_Regime'].shift())
    change_indices = valid_data.index[perfect_changes]
    
    lag_values = []
    for change_idx in change_indices[1:]:  # Skip first change
        # Get perfect regime at change
        perfect_regime = valid_data.loc[change_idx, 'Perfect_Confirmed_Regime']
        
        # Find when live system catches up (within next 20 bars)
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

