#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
from ta.trend import EMAIndicator, ADXIndicator, MACD  # For Direction/Trend
from ta.volatility import AverageTrueRange, BollingerBands
from ta.momentum import RSIIndicator, StochasticOscillator, ROCIndicator, PPOIndicator, CCIIndicator  # For Strength/Momentum
from ta.volume import OnBalanceVolumeIndicator, VolumeWeightedAveragePrice, ChaikinMoneyFlowIndicator  # For Structure
from utils.logger import log_message, progress_bar
from config.settings import DEBUG_LEVEL, DATA_PATH
from core.data_loader import load_csv_data
from sklearn.preprocessing import OneHotEncoder  # For Session categorical

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
        ema = EMAIndicator(df['close'], window=50).ema_indicator()
        adx = ADXIndicator(df['high'], df['low'], df['close'], window=14).adx()
        macd = MACD(df['close']).macd()
        indicators['EMA_50'] = ema
        indicators['ADX_14'] = adx
        indicators['MACD'] = macd
        if DEBUG_LEVEL == 'verbose':
            log_message("Direction indicators computed", 'info')
    
    # Volatility: ATR_14, BB_width, Hist Vol (std close)
    if 'volatility' in regime_classes:
        atr = AverageTrueRange(df['high'], df['low'], df['close'], window=14).average_true_range()
        bb = BollingerBands(df['close'], window=20)
        bb_width = (bb.bollinger_hband() - bb.bollinger_lband()) / bb.bollinger_mavg()
        hist_vol = df['close'].pct_change().rolling(20).std() * np.sqrt(252)  # Annualized
        indicators['ATR_14'] = atr
        indicators['BB_width'] = bb_width
        indicators['Hist_Vol'] = hist_vol
        if DEBUG_LEVEL == 'verbose':
            log_message("Volatility indicators computed", 'info')
    
    # Trend Strength: RSI_14, Stochastic, +DI (from ADX)
    if 'trend_strength' in regime_classes:
        rsi = RSIIndicator(df['close'], window=14).rsi()
        stoch = StochasticOscillator(df['high'], df['low'], df['close'], window=14).stoch()
        adx_ind = ADXIndicator(df['high'], df['low'], df['close'], window=14)
        indicators['RSI_14'] = rsi
        indicators['Stoch'] = stoch
        indicators['DI_Plus'] = adx_ind.adx_pos()  # +DI for strength
        if DEBUG_LEVEL == 'verbose':
            log_message("Trend Strength indicators computed", 'info')
    
    # Momentum: ROC_12, PPO, CCI
    if 'momentum' in regime_classes:
        roc = ROCIndicator(df['close'], window=12).roc()
        ppo = PPOIndicator(df['close']).ppo()
        cci = CCIIndicator(df['high'], df['low'], df['close'], window=20).cci()
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
        obv = OnBalanceVolumeIndicator(df['close'], df['volume']).on_balance_volume()
        vwap = VolumeWeightedAveragePrice(df['high'], df['low'], df['close'], df['volume'], window=14).volume_weighted_average_price()
        cmf = ChaikinMoneyFlowIndicator(df['high'], df['low'], df['close'], df['volume'], window=20).chaikin_money_flow()
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
        # Drop one from high-corr pairs (simple: keep first)
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

