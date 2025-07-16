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
        kc = ta.kc(df['high'], df['low'], df['close'], length=20)
        kc_width = (kc['KCUe_20_2'] - kc['KCLe_20_2']) / kc['KCBe_20_2']  # Now we know the column names

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
        dmi_minus = adx_full['DMN_14']
        indicators['RSI_7'] = rsi_7
        indicators['RSI_14'] = rsi_14
        indicators['Stoch'] = stoch
        indicators['DMI_Plus'] = dmi_plus
        indicators['DMI_Minus'] = dmi_minus
        if DEBUG_LEVEL == 'verbose':
            log_message("Trend Strength indicators computed", 'info')
    
    # Momentum: ROC_12, PPO, CCI
    if 'momentum' in regime_classes:
        roc = ta.roc(df['close'], length=12)
        ppo = ta.ppo(df['close'])['PPO_12_26_9']
        ppo_hist = ta.ppo(df['close'])['PPOh_12_26_9']  # NEW: PPO histogram
        ppo_signal = ta.ppo(df['close'])['PPOs_12_26_9']  # NEW: PPO signal line
        cci = ta.cci(df['high'], df['low'], df['close'], length=20)
        indicators['ROC_12'] = roc
        indicators['PPO'] = ppo
        indicators['PPO_Hist'] = ppo_hist  # NEW
        indicators['PPO_Signal'] = ppo_signal  # NEW
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
        # FIX VWAP - handle timezone issue
        df_temp = df.copy()
        if hasattr(df_temp.index, 'tz'):
            df_temp.index = df_temp.index.tz_localize(None)  # ADD THIS
        vwap = ta.vwap(df_temp['high'], df_temp['low'], df_temp['close'], df_temp['volume'], length=14)  # CHANGED
        cmf = ta.cmf(df['high'], df['low'], df['close'], df['volume'], length=20)
        indicators['OBV'] = obv
        indicators['VWAP_14'] = vwap
        indicators['CMF_20'] = cmf
    
    ind_df = pd.DataFrame(indicators, index=df.index).fillna(0)  # Fill NaNs

    # NEW: #2 Constraint - Check rows per class (simple log if <50)
    class_groups = {
        'direction': [col for col in ind_df if col.startswith('EMA') or col in ['ADX_14', 'MACD']],
        'volatility': [col for col in ind_df if col.startswith('ATR') or col in ['BB_width', 'Hist_Vol', 'KC_width']],
        'trend_strength': [col for col in ind_df if col.startswith('RSI') or col in ['Stoch', 'DMI_Plus', 'DMI_Minus']],
        'momentum': [col for col in ind_df if col in ['ROC_12', 'PPO', 'PPO_Hist', 'PPO_Signal', 'CCI_20']],
        'session': [col for col in ind_df if 'session' in col.lower()],
        'structure': [col for col in ind_df if col in ['OBV', 'VWAP_14', 'CMF_20']]
    }
    for cls, cols in class_groups.items():
        if cols:
            class_df = ind_df[cols].dropna(how='all')
            if len(class_df) < 50:
                log_message(f"Warning: {cls} class has only {len(class_df)} rows (<50)—may need more data", 'warning')
    
    # Normalize - use robust scaling to handle outliers
    for col in progress_bar(ind_df.columns, desc="Normalizing"):
        if 'session' in col.lower():
            continue
        
        # Use robust scaling (median and IQR) instead of mean/std
        median = ind_df[col].rolling(480, min_periods=50).median()
        q75 = ind_df[col].rolling(480, min_periods=50).quantile(0.75)
        q25 = ind_df[col].rolling(480, min_periods=50).quantile(0.25)
        iqr = q75 - q25
        
        # Clip extreme values before normalizing
        ind_df[col] = ind_df[col].clip(lower=q25 - 3*iqr, upper=q75 + 3*iqr)
        ind_df[col] = (ind_df[col] - median) / iqr.where(iqr != 0, 1)
    
    ind_df = ind_df.ffill().fillna(0)
    
    # Corr check - LESS AGGRESSIVE
    corr = ind_df.corr()
    import matplotlib.pyplot as plt; plt.matshow(corr); plt.colorbar(); plt.savefig(os.path.join(BASE_DIR, 'exports/plots/corr_matrix.png'))

    # Filter out session columns from correlation check
    non_session_cols = [col for col in ind_df.columns if 'session' not in col.lower()]
    corr_subset = ind_df[non_session_cols].corr()

    high_corr = (corr_subset.abs() > 0.9) & (corr_subset.abs() < 1.0)  # Unchanged
    if high_corr.any().any():
        log_message("High corr detected—evaluating pairs", 'info')
        to_drop = set()
        
        # Group indicators by type...
        groups = {  # Unchanged, add your classes
            'direction': ['EMA_20', 'EMA_50', 'EMA_200', 'ADX_14', 'MACD'],
            'volatility': ['ATR_7', 'ATR_14', 'ATR_30', 'BB_width', 'Hist_Vol', 'KC_width'],
            'trend_strength': ['RSI_7', 'RSI_14', 'Stoch', 'DMI_Plus'],
            'momentum': ['ROC_12', 'PPO', 'CCI_20'],
            'structure': ['OBV', 'VWAP_14', 'CMF_20']
        }
        
        for i in range(len(high_corr.columns)):
            for j in range(i+1, len(high_corr.columns)):
                if high_corr.iloc[i, j]:
                    col1, col2 = high_corr.columns[i], high_corr.columns[j]
                    
                    # Find groups
                    col1_group = next((g for g, inds in groups.items() if col1 in inds), None)
                    col2_group = next((g for g, inds in groups.items() if col2 in inds), None)
                    
                    if col1_group and col2_group and col1_group == col2_group:
                        corr_value = corr_subset.iloc[i, j]
                        if col1_group == 'volatility' and corr_value < 0.95:
                            continue  # Skip drop for vol if <0.95
                        
                        # Priority keep
                        priority_indicators = ['ATR_14', 'RSI_14', 'EMA_50', 'MACD', 'ADX_14']
                        if col1 in priority_indicators and col2 not in priority_indicators:
                            to_drop.add(col2)
                        elif col2 in priority_indicators and col1 not in priority_indicators:
                            to_drop.add(col1)
                        # NaN check
                        elif ind_df[col1].isna().sum() > ind_df[col2].isna().sum():
                            to_drop.add(col1)
                        else:
                            to_drop.add(col2)
                    elif DEBUG_LEVEL == 'verbose':
                        log_message(f"Skipping drop for {col1} and {col2} - different groups", 'info')

        # Add this after the correlation check:
        # Quality check - ensure we have indicators from each category
        required_types = ['EMA', 'ATR', 'RSI', 'ADX']
        available = [col for col in ind_df.columns]
        for req_type in required_types:
            if not any(req_type in col for col in available):
                log_message(f"Warning: No {req_type} indicator retained after correlation check", 'warning')

        log_message(f"Final indicators: {sorted([c for c in ind_df.columns if 'session' not in c.lower()])}", 'info')
        
        if to_drop:
            log_message(f"Dropping high-corr indicators from same groups: {to_drop}", 'info')
            ind_df = ind_df.drop(columns=to_drop)
        
        log_message(f"Kept {len(ind_df.columns)} indicators after correlation check", 'info')
    
    return ind_df

if __name__ == "__main__":
    df = load_csv_data(DATA_PATH)
    ind_df = select_and_compute_indicators(df)
    log_message("All indicators computed", 'info')

