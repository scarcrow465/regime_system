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
TEST_SLICE = 5000
DATA_FILE = 'combined_NQ_15m_data.csv'
OUTPUT_FILE = 'ma_regime_labeled_data_with_perfect_v53_test.csv'
SCORING_FILE = 'regime_scoring_metrics_v53_test.csv'
TIMEFRAME = '15min'

ENHANCED_FEATURES = True

CORE_PARAMS = {
    'short_ma_period': 5,
    'long_ma_period': 13,
    'short_atr_period': 5,
    'long_atr_period': 50,
    'base_slope_lookback': 100,
    'bull_weak_percentile': 0.65,
    'bull_strong_percentile': 0.85,
    'bear_weak_percentile': 0.35,
    'bear_strong_percentile': 0.15,
    'volatility_lookback': 100,
    'volatility_filter_min': 0.7,
    'volatility_filter_max': 1.3,
    'volatility_low_percentile': 0.25,
    'volatility_high_percentile': 0.85,
    'volatility_high_bull_percentile': 0.85,
    'volatility_high_bear_percentile': 0.85,
    'volatility_low_bull_percentile': 0.25,
    'volatility_low_bear_percentile': 0.25,
    'transitioning_factor': 0.5,
    'base_persistence': 2
}

ENHANCED_PARAMS = {
    'slope_adaptation_factor': 0.3,
    'volatility_adaptation_factor': 0.2,
    'strong_persistence': 1,
    'weak_persistence': 2,
    'between_persistence': 3,
    'session_adaptation': False,
    'between_scale_sensitivity': 5.0,
    'volatility_scale_factor': 2.0
}

PERFECT_PARAMS = {
    'min_forward_bars': 100,
    'max_forward_bars': 200,
    'adaptive_forward': True,
    'perfect_lookahead': 100  # Max bars to scan for stable regime
}

def load_csv_data(file_path, timeframe):
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
    
    def rolling_masked_quantile(series, lookback, percentile, mask_series, min_val, max_val):
        def custom_quantile(arr):
            mask = (arr >= min_val) & (arr <= max_val)
            if mask.sum() < 50:
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

    # Perfect thresholds: Use live thresholds for comparability
    for col in ['Bull_Weak', 'Bull_Strong', 'Bear_Weak', 'Bear_Strong', 'Vol_Low', 'Vol_High', 'Vol_High_Bull', 'Vol_High_Bear', 'Vol_Low_Bull', 'Vol_Low_Bear']:
        data[f'Perfect_{col}'] = data[f'Dynamic_{col}']
    
    return data

def get_adaptive_thresholds(data, row_idx, use_perfect=False):
    dynamic_params = CORE_PARAMS.copy()
    prefix = 'Perfect_' if use_perfect else 'Dynamic_'
    if row_idx < len(data):
        dynamic_params['bull_weak_threshold'] = data[f'{prefix}Bull_Weak'].iloc[row_idx]
        dynamic_params['bull_strong_threshold'] = data[f'{prefix}Bull_Strong'].iloc[row_idx]
        dynamic_params['bear_weak_threshold'] = data[f'{prefix}Bear_Weak'].iloc[row_idx]
        dynamic_params['bear_strong_threshold'] = data[f'{prefix}Bear_Strong'].iloc[row_idx]
        dynamic_params['volatility_low_threshold'] = data[f'{prefix}Vol_Low'].iloc[row_idx]
        dynamic_params['volatility_high_threshold'] = data[f'{prefix}Vol_High'].iloc[row_idx]
        dynamic_params['volatility_high_bull_threshold'] = data[f'{prefix}Vol_High_Bull'].iloc[row_idx]
        dynamic_params['volatility_high_bear_threshold'] = data[f'{prefix}Vol_High_Bear'].iloc[row_idx]
        dynamic_params['volatility_low_bull_threshold'] = data[f'{prefix}Vol_Low_Bull'].iloc[row_idx]
        dynamic_params['volatility_low_bear_threshold'] = data[f'{prefix}Vol_Low_Bear'].iloc[row_idx]
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
    up_volatility_ratio = data['Up_Volatility_Ratio'].iloc[row_idx]
    down_volatility_ratio = data['Down_Volatility_Ratio'].iloc[row_idx]
    
    if pd.isna(slope) or pd.isna(short_ma) or pd.isna(long_ma) or pd.isna(volatility_ratio):
        return 'BETWEEN'
    
    abs_slope = abs(slope)
    max_strong = max(params['bull_strong_threshold'], abs(params['bear_strong_threshold']))
    
    # Slope-based scaling: Near-zero in steep slopes, larger in flat slopes
    between_scale_factor = max(0, min(1, 1 - (abs_slope / max_strong) ** ENHANCED_PARAMS['between_scale_sensitivity'])) if max_strong > 0 else 1
    
    # Volatility adjustment: Narrower in low vol, wider in high vol
    vol_factor = min(2.0, max(0.2, volatility_ratio / params['volatility_high_threshold'] * ENHANCED_PARAMS['volatility_scale_factor']))
    
    effective_bull_weak = params['bull_weak_threshold'] * between_scale_factor * vol_factor
    effective_bear_weak = params['bear_weak_threshold'] * between_scale_factor * vol_factor
    
    if slope > params['bull_strong_threshold'] and short_ma > upper_threshold and up_volatility_ratio > params['volatility_high_bull_threshold']:
        return 'STRONG ABOVE'
    elif slope > params['bull_weak_threshold'] and short_ma > upper_threshold:
        return 'WEAK ABOVE'
    elif slope < -params['bear_strong_threshold'] and short_ma < lower_threshold and down_volatility_ratio > params['volatility_high_bear_threshold']:
        return 'STRONG BELOW'
    elif slope < -params['bear_weak_threshold'] and short_ma < lower_threshold:
        return 'WEAK BELOW'
    elif abs(slope) <= effective_bull_weak:
        if abs(short_ma - long_ma) <= params['transitioning_factor'] * dynamic_multiplier * atr_5:
            return 'TRANSITIONING'
        elif volatility_ratio > params['volatility_high_threshold']:
            return 'EXPANDING BETWEEN'
        elif volatility_ratio < params['volatility_low_threshold']:
            return 'CONTRACTING BETWEEN'
    return 'BETWEEN'

def apply_regime_classification(data, use_perfect=False):
    regime_col = 'Perfect_Raw_Regime' if use_perfect else 'Live_Raw_Regime'
    data[regime_col] = 'BETWEEN'
    desc = "Classifying Perfect Regimes" if use_perfect else "Classifying Live Regimes"
    
    if use_perfect:
        perfect_lookahead = PERFECT_PARAMS['perfect_lookahead']
        for i in tqdm(range(len(data)), desc=desc, ncols=80):
            if i >= CORE_PARAMS['long_ma_period'] and i + perfect_lookahead < len(data):
                # Scan forward to find first stable regime
                for j in range(i, min(i + perfect_lookahead, len(data))):
                    params = get_adaptive_thresholds(data, j, use_perfect=False)  # Use live thresholds
                    regime = classify_regime_excel_logic(data, j, params)
                    # Check persistence
                    needed = (ENHANCED_PARAMS['strong_persistence'] if 'STRONG' in regime else
                              ENHANCED_PARAMS['weak_persistence'] if 'WEAK' in regime else
                              ENHANCED_PARAMS['between_persistence'])
                    if j + needed <= len(data):
                        stable = True
                        for k in range(j, j + needed):
                            if classify_regime_excel_logic(data, k, params) != regime:
                                stable = False
                                break
                        if stable:
                            data.loc[data.index[j], regime_col] = regime
                            break  # Assign to first stable bar and move to next i
    else:
        for i in tqdm(range(len(data)), desc=desc, ncols=80):
            if i >= CORE_PARAMS['long_ma_period']:
                params = get_adaptive_thresholds(data, i, use_perfect=False)
                regime = classify_regime_excel_logic(data, i, params)
                data.loc[data.index[i], regime_col] = regime
    
    return data

def apply_transition_aware_persistence(data, regime_col, confirmed_col):
    data[confirmed_col] = np.nan
    is_live_data = 'Live' in regime_col
    
    for i in range(len(data)):
        current_raw = data[regime_col].iloc[i]
        if pd.isna(current_raw):
            if i > 0:
                data.iloc[i, data.columns.get_loc(confirmed_col)] = data[confirmed_col].iloc[i-1]
            continue
        
        if is_live_data:
            needed = (ENHANCED_PARAMS['strong_persistence'] if 'STRONG' in current_raw else
                      ENHANCED_PARAMS['weak_persistence'] if 'WEAK' in current_raw else
                      ENHANCED_PARAMS['between_persistence'])
            count = 1
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
        else:
            needed = (ENHANCED_PARAMS['strong_persistence'] if 'STRONG' in current_raw else
                      ENHANCED_PARAMS['weak_persistence'] if 'WEAK' in current_raw else
                      ENHANCED_PARAMS['between_persistence'])
            count = 1
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
    metrics = {}
    valid_mask = (data['Live_Confirmed_Regime'].notna() & 
                  data['Perfect_Confirmed_Regime'].notna())
    valid_data = data[valid_mask].copy()
    
    if len(valid_data) == 0:
        return pd.DataFrame([metrics])
    
    exact_matches = valid_data['Live_Confirmed_Regime'] == valid_data['Perfect_Confirmed_Regime']
    metrics['Overall_Accuracy'] = exact_matches.sum() / len(valid_data) * 100
    
    regime_types = ['STRONG ABOVE', 'WEAK ABOVE', 'STRONG BELOW', 'WEAK BELOW', 
                    'CONTRACTING BETWEEN', 'EXPANDING BETWEEN', 'TRANSITIONING', 'BETWEEN']
    
    for regime in regime_types:
        live_regime_mask = valid_data['Live_Confirmed_Regime'] == regime
        if live_regime_mask.sum() > 0:
            precision = (valid_data[live_regime_mask]['Perfect_Confirmed_Regime'] == regime).sum() / live_regime_mask.sum()
            metrics[f'{regime}_Precision'] = precision * 100
        else:
            metrics[f'{regime}_Precision'] = 0
        
        perfect_regime_mask = valid_data['Perfect_Confirmed_Regime'] == regime
        if perfect_regime_mask.sum() > 0:
            recall = (valid_data[perfect_regime_mask]['Live_Confirmed_Regime'] == regime).sum() / perfect_regime_mask.sum()
            metrics[f'{regime}_Recall'] = recall * 100
        else:
            metrics[f'{regime}_Recall'] = 0
    
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
    metrics['Calculation_Timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    return pd.DataFrame([metrics])

def main():
    print(f"Enhanced Features: {'ENABLED' if ENHANCED_FEATURES else 'DISABLED'}")
    print(f"Perfect Benchmark System: ENABLED")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    with tqdm(total=1, desc="Loading Data", ncols=80) as pbar:
        data = load_csv_data(DATA_FILE, TIMEFRAME)
        pbar.update(1)

    if TEST_SLICE is not None:
        data = data.tail(TEST_SLICE)
        print(f"Using last {TEST_SLICE} rows for testing")

    with tqdm(total=1, desc="Calculating Helper Columns", ncols=80) as pbar:
        data = calculate_helper_columns(data)
        pbar.update(1)

    data = apply_regime_classification(data, use_perfect=False)
    data['Live_Raw_Regime'] = data['Live_Raw_Regime'].shift(1)
    print("Applied 1-bar shift to live system to eliminate lookahead bias")
    data = apply_regime_classification(data, use_perfect=True)

    with tqdm(total=1, desc="Applying Persistence to Live System", ncols=80) as pbar:
        data = apply_transition_aware_persistence(data, 'Live_Raw_Regime', 'Live_Confirmed_Regime')
        pbar.update(1)
        
    with tqdm(total=1, desc="Applying Persistence to Perfect System", ncols=80) as pbar:
        data = apply_transition_aware_persistence(data, 'Perfect_Raw_Regime', 'Perfect_Confirmed_Regime')
        pbar.update(1)

    regimes = ["STRONG ABOVE", "WEAK ABOVE", "STRONG BELOW", "WEAK BELOW", 
               "CONTRACTING BETWEEN", "EXPANDING BETWEEN", "TRANSITIONING", "BETWEEN"]
    
    for regime in regimes:
        col_name = "Live_" + regime.replace(" ", "_") + "_Dummy"
        data[col_name] = np.where(data['Live_Confirmed_Regime'] == regime, 1000000, np.nan)
        col_name = "Perfect_" + regime.replace(" ", "_") + "_Dummy"
        data[col_name] = np.where(data['Perfect_Confirmed_Regime'] == regime, 2000000, np.nan)

    data['Index'] = range(len(data))
    data = data.reset_index()
    data['Date_Only'] = data['Date'].dt.date
    data['Time'] = data['Date'].dt.time
    data['Full_Timestamp'] = data['Date'].dt.strftime('%Y-%m-%d %H:%M:%S')

    live_dummy_columns = ["Live_" + regime.replace(" ", "_") + "_Dummy" for regime in regimes]
    perfect_dummy_columns = ["Perfect_" + regime.replace(" ", "_") + "_Dummy" for regime in regimes]
    
    output_columns = [
        'Index', 'Full_Timestamp', 'Date_Only', 'Time', 
        'open', 'high', 'low', 'close',
        'Live_Confirmed_Regime', 'Live_Raw_Regime',
        'Perfect_Confirmed_Regime', 'Perfect_Raw_Regime',
        'Dynamic_Bull_Weak', 'Dynamic_Bull_Strong', 
        'Dynamic_Bear_Weak', 'Dynamic_Bear_Strong',
        'Dynamic_Vol_Low', 'Dynamic_Vol_High'
    ] + live_dummy_columns + perfect_dummy_columns
    
    data[output_columns].to_csv(OUTPUT_FILE, index=False)
    print(f"\nExported labeled data to {OUTPUT_FILE}")

    print("\nCalculating Scoring Metrics...")
    scoring_df = calculate_scoring_metrics(data)
    
    scoring_df.to_csv(SCORING_FILE, index=False)
    print(f"Saved scoring metrics to {SCORING_FILE}")
    
    print(f"\nKey Performance Metrics:")
    print(f"Overall Accuracy: {scoring_df['Overall_Accuracy'].iloc[0]:.2f}%")
    print(f"Directional Accuracy: {scoring_df['Directional_Accuracy'].iloc[0]:.2f}%")
    print(f"Weighted Accuracy: {scoring_df['Weighted_Accuracy'].iloc[0]:.2f}%")
    print(f"Average Lag: {scoring_df['Average_Lag_Bars'].iloc[0]:.1f} bars")

    print("\n" + "="*60)
    print("DEBUG ANALYSIS")
    print("="*60)

    valid_data = data.dropna(subset=['Volatility_Ratio', 'Up_Volatility_Ratio', 'Down_Volatility_Ratio'])
    if len(valid_data) > 0:
        print(f"\nVolatility Ratio Comparison:")
        print(f"  Regular Volatility - Mean: {valid_data['Volatility_Ratio'].mean():.3f}, Std: {valid_data['Volatility_Ratio'].std():.3f}")
        print(f"  Up Volatility - Mean: {valid_data['Up_Volatility_Ratio'].mean():.3f}, Std: {valid_data['Up_Volatility_Ratio'].std():.3f}")
        print(f"  Down Volatility - Mean: {valid_data['Down_Volatility_Ratio'].mean():.3f}, Std: {valid_data['Down_Volatility_Ratio'].std():.3f}")

    bull_thresh_data = data.dropna(subset=['Dynamic_Vol_High_Bull', 'Dynamic_Vol_High_Bear'])
    if len(bull_thresh_data) > 0:
        print(f"\nDirectional Volatility Thresholds:")
        print(f"  Bull High Volatility Threshold - Mean: {bull_thresh_data['Dynamic_Vol_High_Bull'].mean():.3f}")
        print(f"  Bear High Volatility Threshold - Mean: {bull_thresh_data['Dynamic_Vol_High_Bear'].mean():.3f}")
        print(f"  Regular High Volatility Threshold - Mean: {bull_thresh_data['Dynamic_Vol_High'].mean():.3f}")

    bull_vol_met = (valid_data['Up_Volatility_Ratio'] > 0.5).sum()
    bear_vol_met = (valid_data['Down_Volatility_Ratio'] > 0.5).sum()
    total_bars = len(valid_data)

    print(f"\nDirectional Volatility Condition Analysis:")
    print(f"  Bars with Up_Volatility > 0.5: {bull_vol_met} ({bull_vol_met/total_bars*100:.2f}%)")
    print(f"  Bars with Down_Volatility > 0.5: {bear_vol_met} ({bear_vol_met/total_bars*100:.2f}%)")
    print(f"  Bars with Regular_Volatility > 0.5: {(valid_data['Volatility_Ratio'] > 0.5).sum()} ({(valid_data['Volatility_Ratio'] > 0.5).sum()/total_bars*100:.2f}%)")

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

    print("\n" + "="*60)
    print("SYSTEM ANALYSIS")
    print("="*60)
    
    valid_slopes = data['SMA_13_Slope'].dropna()
    print(f"\nSlope Statistics:")
    print(f"  Range: {valid_slopes.min():.6f} to {valid_slopes.max():.6f}")
    print(f"  Mean: {valid_slopes.mean():.6f}")
    print(f"  Std: {valid_slopes.std():.6f}")
    
    valid_vol = data['Volatility_Ratio'].dropna()
    print(f"\nVolatility Ratio Statistics:")
    print(f"  Range: {valid_vol.min():.3f} to {valid_vol.max():.3f}")
    print(f"  Mean: {valid_vol.mean():.3f}")
    print(f"  Std: {valid_vol.std():.3f}")
    
    print(f"\nPerfect System Analysis:")
    print(f"  Adaptive forward lookback: {PERFECT_PARAMS['adaptive_forward']}")
    print(f"  Forward bars range: {PERFECT_PARAMS['min_forward_bars']}-{PERFECT_PARAMS['max_forward_bars']}")

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

