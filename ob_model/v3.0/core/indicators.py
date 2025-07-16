#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import pandas_ta as ta  # All indicators from pandas_ta
import numpy as np
from utils.logger import log_message, progress_bar
from config.settings import DEBUG_LEVEL, DATA_PATH
from core.data_loader import load_csv_data
from sklearn.preprocessing import OneHotEncoder  # For Session

def select_and_compute_indicators(df, regime_classes=['direction', 'volatility', 'trend_strength', 'momentum', 'session', 'structure']):
    """Compute 2-3 low-corr indicators per class."""
    if len(df) < 50:
        log_message("Insufficient data—skipping", 'error')
        return pd.DataFrame(index=df.index)
    
    indicators = {}
    
    if DEBUG_LEVEL in ['debug', 'verbose']:
        log_message("Computing indicators for all classes", 'info')
    
    # Direction: EMA_50, ADX_14, MACD
    if 'direction' in regime_classes:
        ema = ta.ema(df['close'], length=50)
        adx = ta.adx(df['high'], df['low'], df['close'], length=14)['ADX_14']
        macd = ta.macd(df['close'])['MACD_12_26_9']
        indicators['EMA_50'] = ema
        indicators['ADX_14'] = adx
        indicators['MACD'] = macd
        if DEBUG_LEVEL == 'verbose':
            log_message("Direction indicators computed", 'info')
    
    # Volatility: ATR_14, BB_width, Hist Vol (stdev close)
    if 'volatility' in regime_classes:
        atr = ta.atr(df['high'], df['low'], df['close'], length=14)
        bb = ta.bbands(df['close'], length=20)
        bb_width = (bb['BBU_20_2.0'] - bb['BBL_20_2.0']) / bb['BBM_20_2.0']
        hist_vol = df['close'].pct_change().rolling(20).std() * np.sqrt(252)  # Annualized
        indicators['ATR_14'] = atr
        indicators['BB_width'] = bb_width
        indicators['Hist_Vol'] = hist_vol
        if DEBUG_LEVEL == 'verbose':
            log_message("Volatility indicators computed", 'info')
    
    # Trend Strength: RSI_14, Stochastic, DMI (+DI from ADX)
    if 'trend_strength' in regime_classes:
        rsi = ta.rsi(df['close'], length=14)
        stoch = ta.stoch(df['high'], df['low'], df['close'], length=14)['STOCHk_14_3_3']
        adx_full = ta.adx(df['high'], df['low'], df['close'], length=14)
        dmi_plus = adx_full['DMP_14']
        indicators['RSI_14'] = rsi
        indicators['Stoch'] = stoch
        indicators['DMI_Plus'] = dmi_plus
        if DEBUG_LEVEL == 'verbose':
            log_message("Trend Strength indicators computed", 'info')
    
    # Momentum: ROC_12, PPO, CCI
    if 'momentum' in regime_classes:
        roc = ta.roc(df['close'], length=12)
        ppo = ta.ppo(df['close'])['PPO_12_26_9']
        cci = ta.cci(df['high'], df['low'], df['close'], length=20)
        indicators['ROC_12'] = roc
        indicators['PPO'] = ppo
        indicators['CCI_20'] = cci
        if DEBUG_LEVEL == 'verbose':
            log_message("Momentum indicators computed", 'info')
    
    # Session: One-hot encode as categorical features
    if 'session' in regime_classes:
        from core.regime_classifier import add_session_labels  # Reuse
        df = add_session_labels(df)
        encoder = OneHotEncoder(sparse_output=False)
        session_enc = encoder.fit_transform(df[['full_session', 'refined_session']])
        session_df = pd.DataFrame(session_enc, index=df.index, columns=encoder.get_feature_names_out())
        for col in session_df.columns:
            indicators[col] = session_df[col]
        if DEBUG_LEVEL == 'verbose':
            log_message("Session features encoded", 'info')
    
    # Structure: OBV, VWAP, CMF
    if 'structure' in regime_classes:
        obv = ta.obv(df['close'], df['volume'])
        vwap = ta.vwap(df['high'], df['low'], df['close'], df['volume'], length=14)
        cmf = ta.cmf(df['high'], df['low'], df['close'], df['volume'], length=20)
        indicators['OBV'] = obv
        indicators['VWAP_14'] = vwap
        indicators['CMF_20'] = cmf
        if DEBUG_LEVEL == 'verbose':
            log_message("Structure indicators computed", 'info')
    
    ind_df = pd.DataFrame(indicators, index=df.index).fillna(0)  # Fill NaNs
    
    # Normalize
    for col in progress_bar(ind_df.columns, desc="Normalizing"):
        mean = ind_df[col].rolling(36).mean()
        std = ind_df[col].rolling(36).std()
        ind_df[col] = (ind_df[col] - mean) / std.where(std != 0, 1e-8)
    
    ind_df = ind_df.fillna(method='ffill').fillna(0)
    
    # Corr check
    corr = ind_df.corr()
    high_corr = (corr.abs() > 0.7) & (corr.abs() < 1.0)
    if high_corr.any().any():
        log_message("High corr detected—dropping pairs", 'info')
        to_drop = set()
        for col in high_corr.columns:
            correlated = high_corr[col][high_corr[col]].index.tolist()
            if correlated:
                to_drop.add(correlated[0])  # Drop one
        ind_df = ind_df.drop(columns=to_drop)
    
    return ind_df

if __name__ == "__main__":
    df = load_csv_data(DATA_PATH)
    ind_df = select_and_compute_indicators(df)
    log_message("All indicators computed", 'info')

