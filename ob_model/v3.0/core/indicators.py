#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import sys
import os
BASE_DIR = r"C:\Users\rs\GitProjects\regime_system\ob_model\v3.0"  # Hardcode if not importing settings yet
sys.path.append(BASE_DIR)
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
        ema_20 = ta.ema(df['close'], length=20)
        ema_50 = ta.ema(df['close'], length=50)
        ema_200 = ta.ema(df['close'], length=200)
        adx = ta.adx(df['high'], df['low'], df['close'], length=14)['ADX_14']
        macd = ta.macd(df['close'])['MACD_12_26_9']
        indicators['EMA_20'] = ema_20
        indicators['EMA_50'] = ema_50
        indicators['EMA_200'] = ema_200
        indicators['ADX_14'] = adx
        indicators['MACD'] = macd
        if DEBUG_LEVEL == 'verbose':
            log_message("Direction indicators computed", 'info')
    
    # Volatility: ATR_14, BB_width, Hist Vol (stdev close)
    if 'volatility' in regime_classes:
        atr_7 = ta.atr(df['high'], df['low'], df['close'], length=7)
        atr_14 = ta.atr(df['high'], df['low'], df['close'], length=14)
        atr_30 = ta.atr(df['high'], df['low'], df['close'], length=30)
        bb = ta.bbands(df['close'], length=20)
        bb_width = (bb['BBU_20_2.0'] - bb['BBL_20_2.0']) / bb['BBM_20_2.0']
        hist_vol = df['close'].pct_change().rolling(20).std() * np.sqrt(252)  # Annualized

        # Keltner Channel - CORRECTED COLUMN NAMES
        kc = ta.kc(df['high'], df['low'], df['close'], length=20)
        if not kc.empty:
            kc_cols = kc.columns.tolist()
            # Find upper, middle, lower columns
            upper_col = [c for c in kc_cols if 'U' in c][0]  
            lower_col = [c for c in kc_cols if 'L' in c][0]
            middle_col = [c for c in kc_cols if 'B' in c or 'M' in c][0]
            kc_width = (kc[upper_col] - kc[lower_col]) / kc[middle_col]
            indicators['KC_width'] = kc_width
        # Check what columns are returned first
        print(kc.columns)  # Add this temporarily to see exact names
        # Likely columns: KCLe_20_2, KCBe_20_2, KCUe_20_2
        kc_width = (kc.iloc[:, 2] - kc.iloc[:, 0]) / kc.iloc[:, 1]  # Upper - Lower / Middle

        indicators['ATR_7'] = atr_7
        indicators['ATR_14'] = atr_14
        indicators['ATR_30'] = atr_30
        indicators['BB_width'] = bb_width
        indicators['Hist_Vol'] = hist_vol
        indicators['KC_width'] = kc_width
        if DEBUG_LEVEL == 'verbose':
            log_message("Volatility indicators computed", 'info')
    
    # Trend Strength: RSI_14, Stochastic, DMI (+DI from ADX)
    if 'trend_strength' in regime_classes:
        rsi_7 = ta.rsi(df['close'], length=7)
        rsi_14 = ta.rsi(df['close'], length=14)
        stoch = ta.stoch(df['high'], df['low'], df['close'], length=14)['STOCHk_14_3_3']
        adx_full = ta.adx(df['high'], df['low'], df['close'], length=14)
        dmi_plus = adx_full['DMP_14']
        indicators['RSI_7'] = rsi_7
        indicators['RSI_14'] = rsi_14
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
    
    # Corr check - EXCLUDE SESSION FEATURES
    corr = ind_df.corr()
    import matplotlib.pyplot as plt; plt.matshow(corr); plt.colorbar(); plt.savefig(os.path.join(BASE_DIR, 'exports/plots/corr_matrix.png'))

    # Filter out session columns from correlation check
    non_session_cols = [col for col in ind_df.columns if 'session' not in col.lower()]
    corr_subset = ind_df[non_session_cols].corr()

    high_corr = (corr_subset.abs() > 0.8) & (corr_subset.abs() < 1.0)  # CHANGED TO 0.8
    if high_corr.any().any():
        log_message("High corr detected—dropping pairs", 'info')
        to_drop = set()
        for i in range(len(high_corr.columns)):
            for j in range(i+1, len(high_corr.columns)):
                if high_corr.iloc[i, j]:
                    col1, col2 = high_corr.columns[i], high_corr.columns[j]
                    # Keep the one with less NaN values
                    if ind_df[col1].isna().sum() > ind_df[col2].isna().sum():
                        to_drop.add(col1)
                    else:
                        to_drop.add(col2)
        
        if to_drop:
            log_message(f"Dropping high-corr indicators: {to_drop}", 'info')
            ind_df = ind_df.drop(columns=to_drop)
    
    return ind_df

if __name__ == "__main__":
    df = load_csv_data(DATA_PATH)
    ind_df = select_and_compute_indicators(df)
    log_message("All indicators computed", 'info')

