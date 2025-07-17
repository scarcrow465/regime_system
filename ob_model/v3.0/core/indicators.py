#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import sys
import os
BASE_DIR = r"C:\Users\rs\GitProjects\regime_system\ob_model\v3.0"
sys.path.append(BASE_DIR)
import pandas as pd
import numpy as np
from utils.logger import log_message, progress_bar
from config.settings import DEBUG_LEVEL, DATA_PATH
from core.data_loader import load_csv_data
from sklearn.preprocessing import OneHotEncoder

def calculate_ema_at_point(close_series, length, end_idx):
    """Calculate EMA using only data up to end_idx"""
    if end_idx < length:
        return np.nan
    
    # Initialize with SMA
    sma = close_series[end_idx-length+1:end_idx+1].mean()
    multiplier = 2 / (length + 1)
    
    # Calculate EMA
    ema = sma
    for i in range(end_idx-length+1, end_idx+1):
        ema = (close_series.iloc[i] - ema) * multiplier + ema
    
    return ema

def calculate_atr_at_point(high, low, close, length, end_idx):
    """Calculate ATR using only data up to end_idx"""
    if end_idx < length:
        return np.nan
    
    tr_values = []
    for i in range(end_idx-length+1, end_idx+1):
        if i == 0:
            tr = high.iloc[i] - low.iloc[i]
        else:
            high_low = high.iloc[i] - low.iloc[i]
            high_close = abs(high.iloc[i] - close[i-1])
            low_close = abs(low.iloc[i] - close[i-1])
            tr = max(high_low, high_close, low_close)
        tr_values.append(tr)
    
    return np.mean(tr_values)

def calculate_rsi_at_point(close_series, length, end_idx):
    """Calculate RSI using only data up to end_idx"""
    if end_idx < length + 1:
        return np.nan
    
    deltas = close_series[end_idx-length:end_idx+1].diff()
    gains = deltas.where(deltas > 0, 0)
    losses = -deltas.where(deltas < 0, 0)
    
    avg_gain = gains[1:].mean()
    avg_loss = losses[1:].mean()
    
    if avg_loss == 0:
        return 100
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi

def calculate_adx_at_point(high, low, close, length, end_idx):
    """Calculate ADX using only data up to end_idx"""
    if end_idx < length * 2:  # Need more data for ADX
        return np.nan, np.nan, np.nan
    
    # Calculate +DM and -DM
    plus_dm = []
    minus_dm = []
    tr_values = []
    
    for i in range(end_idx-length*2+1, end_idx+1):
        if i == 0:
            continue
            
        high_diff = high.iloc[i] - high[i-1]
        low_diff = low[i-1] - low.iloc[i]
        
        plus_dm_val = high_diff if high_diff > low_diff and high_diff > 0 else 0
        minus_dm_val = low_diff if low_diff > high_diff and low_diff > 0 else 0
        
        plus_dm.append(plus_dm_val)
        minus_dm.append(minus_dm_val)
        
        # TR calculation
        high_low = high.iloc[i] - low.iloc[i]
        high_close = abs(high.iloc[i] - close[i-1])
        low_close = abs(low.iloc[i] - close[i-1])
        tr = max(high_low, high_close, low_close)
        tr_values.append(tr)
    
    # Calculate smoothed values
    atr = np.mean(tr_values[-length:])
    plus_di = 100 * np.mean(plus_dm[-length:]) / atr if atr != 0 else 0
    minus_di = 100 * np.mean(minus_dm[-length:]) / atr if atr != 0 else 0
    
    # Calculate ADX
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di) if (plus_di + minus_di) != 0 else 0
    
    return dx, plus_di, minus_di

def select_and_compute_indicators_live(df, lookback_bars=None):
    """
    Compute indicators WITHOUT look-ahead bias.
    This version can be used for live trading.
    
    Args:
        df: DataFrame with OHLCV data
        lookback_bars: If specified, only compute indicators for the last N bars (faster)
    """
    if len(df) < 200:
        log_message("Insufficient data for indicators", 'error')
        return pd.DataFrame(index=df.index), pd.DataFrame(index=df.index)
    
    # Initialize result dataframe
    indicators = pd.DataFrame(index=df.index)
    
    # Determine which bars to calculate
    if lookback_bars and len(df) > lookback_bars:
        calc_start = len(df) - lookback_bars
    else:
        calc_start = 200  # Need at least 200 bars for EMA200
    
    log_message(f"Calculating indicators for bars {calc_start} to {len(df)}", 'info')
    
    # Calculate indicators bar by bar
    for i in progress_bar(range(calc_start, len(df)), desc="Computing indicators"):
        # Direction indicators
        indicators.loc[df.index.iloc[i], 'EMA_20'] = calculate_ema_at_point(df['close'], 20, i)
        indicators.loc[df.index.iloc[i], 'EMA_50'] = calculate_ema_at_point(df['close'], 50, i)
        indicators.loc[df.index.iloc[i], 'EMA_200'] = calculate_ema_at_point(df['close'], 200, i)
        
        adx, plus_di, minus_di = calculate_adx_at_point(df['high'], df['low'], df['close'], 14, i)
        indicators.loc[df.index.iloc[i], 'ADX_14'] = adx
        indicators.loc[df.index.iloc[i], 'DMI_Plus'] = plus_di
        indicators.loc[df.index.iloc[i], 'DMI_Minus'] = minus_di
        
        # MACD
        if i >= 26:
            ema12 = calculate_ema_at_point(df['close'], 12, i)
            ema26 = calculate_ema_at_point(df['close'], 26, i)
            indicators.loc[df.index.iloc[i], 'MACD'] = ema12 - ema26
        
        # Volatility indicators
        indicators.loc[df.index.iloc[i], 'ATR_7'] = calculate_atr_at_point(df['high'], df['low'], df['close'], 7, i)
        indicators.loc[df.index.iloc[i], 'ATR_14'] = calculate_atr_at_point(df['high'], df['low'], df['close'], 14, i)
        indicators.loc[df.index.iloc[i], 'ATR_30'] = calculate_atr_at_point(df['high'], df['low'], df['close'], 30, i)
        
        # Bollinger Bands
        if i >= 20:
            close_slice = df['close'][i-19:i+1]
            sma = close_slice.mean()
            std = close_slice.std()
            indicators.loc[df.index.iloc[i], 'BB_width'] = (4 * std) / sma if sma != 0 else 0
            
        # Historical volatility
        if i >= 20:
            returns = df['close'][i-19:i+1].pct_change().dropna()
            indicators.loc[df.index.iloc[i], 'Hist_Vol'] = returns.std() * np.sqrt(252)
        
        # Keltner width
        if i >= 20:
            atr20 = calculate_atr_at_point(df['high'], df['low'], df['close'], 20, i)
            ema20 = calculate_ema_at_point(df['close'], 20, i)
            indicators.loc[df.index.iloc[i], 'KC_width'] = (4 * atr20) / ema20 if ema20 != 0 else 0
        
        # Trend strength indicators
        indicators.loc[df.index.iloc[i], 'RSI_7'] = calculate_rsi_at_point(df['close'], 7, i)
        indicators.loc[df.index.iloc[i], 'RSI_14'] = calculate_rsi_at_point(df['close'], 14, i)
        
        # Stochastic
        if i >= 14:
            high_14 = df['high'][i-13:i+1].max()
            low_14 = df['low'][i-13:i+1].min()
            if high_14 != low_14:
                indicators.loc[df.index.iloc[i], 'Stoch'] = 100 * (df['close'].iloc[i] - low_14) / (high_14 - low_14)
            else:
                indicators.loc[df.index.iloc[i], 'Stoch'] = 50
        
        # Momentum indicators
        if i >= 12:
            indicators.loc[df.index.iloc[i], 'ROC_12'] = 100 * (df['close'].iloc[i] / df['close'][i-12] - 1)
        
        # PPO
        if i >= 26:
            ema12 = calculate_ema_at_point(df['close'], 12, i)
            ema26 = calculate_ema_at_point(df['close'], 26, i)
            indicators.loc[df.index.iloc[i], 'PPO'] = 100 * (ema12 - ema26) / ema26 if ema26 != 0 else 0
        
        # CCI
        if i >= 20:
            typical_price = (df['high'][i-19:i+1] + df['low'][i-19:i+1] + df['close'][i-19:i+1]) / 3
            sma_tp = typical_price.mean()
            mad = (typical_price - sma_tp).abs().mean()
            indicators.loc[df.index.iloc[i], 'CCI_20'] = (typical_price.iloc[-1] - sma_tp) / (0.015 * mad) if mad != 0 else 0
        
        # Structure indicators
        if i > 0:
            # OBV
            if i == calc_start:
                indicators.loc[df.index.iloc[i], 'OBV'] = df['volume'].iloc[i] if df['close'].iloc[i] > df['close'][i-1] else -df['volume'].iloc[i]
            else:
                prev_obv = indicators.loc[df.index[i-1], 'OBV']
                if df['close'].iloc[i] > df['close'][i-1]:
                    indicators.loc[df.index.iloc[i], 'OBV'] = prev_obv + df['volume'].iloc[i]
                elif df['close'].iloc[i] < df['close'][i-1]:
                    indicators.loc[df.index.iloc[i], 'OBV'] = prev_obv - df['volume'].iloc[i]
                else:
                    indicators.loc[df.index.iloc[i], 'OBV'] = prev_obv
        
        # VWAP
        if i >= 14:
            typical_price = (df['high'][i-13:i+1] + df['low'][i-13:i+1] + df['close'][i-13:i+1]) / 3
            volume_slice = df['volume'][i-13:i+1]
            indicators.loc[df.index.iloc[i], 'VWAP_14'] = (typical_price * volume_slice).sum() / volume_slice.sum()
        
        # CMF
        if i >= 20:
            mf_multiplier = ((df['close'][i-19:i+1] - df['low'][i-19:i+1]) - 
                            (df['high'][i-19:i+1] - df['close'][i-19:i+1])) / \
                           (df['high'][i-19:i+1] - df['low'][i-19:i+1])
            mf_volume = mf_multiplier * df['volume'][i-19:i+1]
            indicators.loc[df.index.iloc[i], 'CMF_20'] = mf_volume.sum() / df['volume'][i-19:i+1].sum()
    
    # Add session labels
    from core.regime_classifier import add_session_labels
    df = add_session_labels(df)
    indicators['full_session'] = df['full_session']
    indicators['refined_session'] = df['refined_session']
    
    # One-hot encode sessions
    encoder = OneHotEncoder(sparse_output=False)
    session_data = df[['full_session', 'refined_session']].iloc[calc_start:]
    if len(session_data) > 0:
        session_enc = encoder.fit_transform(session_data)
        session_df = pd.DataFrame(session_enc, 
                                 index=df.index[calc_start:], 
                                 columns=encoder.get_feature_names_out())
        for col in session_df.columns:
            indicators.loc[session_df.index, col] = session_df[col]
    
    # Fill NaN values
    indicators = indicators.fillna(0)
    
    # Create raw copy before normalization
    raw_indicators = indicators.copy()
    
    # Normalize using only historical data
    for col in indicators.columns:
        if 'session' in col.lower():
            continue
            
        for i in range(calc_start, len(df)):
            if i < 480:  # Not enough history
                continue
                
            # Use only past data for normalization
            hist_data = indicators[col].iloc[max(0, i-480):i]
            if len(hist_data) > 50:
                median = hist_data.median()
                q75 = hist_data.quantile(0.75)
                q25 = hist_data.quantile(0.25)
                iqr = q75 - q25
                
                if iqr != 0:
                    # Normalize current value using historical statistics
                    indicators.loc[df.index.iloc[i], col] = (indicators.loc[df.index.iloc[i], col] - median) / iqr
    
    return indicators, raw_indicators

# For backward compatibility
def select_and_compute_indicators(df, regime_classes=['direction', 'volatility', 'trend_strength', 'momentum', 'session', 'structure']):
    """Wrapper for backward compatibility"""
    return select_and_compute_indicators_live(df)

if __name__ == "__main__":
    df = load_csv_data(DATA_PATH)
    ind_df, raw_ind_df = select_and_compute_indicators(df)
    log_message("All indicators computed without look-ahead bias", 'info')

