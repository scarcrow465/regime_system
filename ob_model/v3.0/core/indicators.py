#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
from ta.trend import EMAIndicator, ADXIndicator
from ta.volatility import AverageTrueRange, BollingerBands
from utils.logger import log_message, progress_bar
from config.settings import DEBUG_LEVEL
from config.settings import DATA_PATH

def select_and_compute_indicators(df, regime_classes=['direction', 'volatility']):
    """Select 2-3 low-corr indicators per class and compute."""
    if len(df) < 50:  # Min for EMA_50
        log_message("Insufficient data length for indicators—skipping", 'error')
        return pd.DataFrame(index=df.index)
    
    indicators = {}
    
    if DEBUG_LEVEL in ['debug', 'verbose']:
        log_message("Selecting and computing indicators", 'info')
    
    # Direction: EMA_50, ADX_14
    if 'direction' in regime_classes:
        ema = EMAIndicator(df['close'], window=50).ema_indicator()
        adx = ADXIndicator(df['high'], df['low'], df['close'], window=14).adx()
        indicators['EMA_50'] = ema
        indicators['ADX_14'] = adx
        if DEBUG_LEVEL == 'verbose':
            log_message(f"Computed direction indicators: {list(indicators.keys())}", 'info')
    
    # Volatility: ATR_14, BB_width
    if 'volatility' in regime_classes:
        atr = AverageTrueRange(df['high'], df['low'], df['close'], window=14).average_true_range()
        bb = BollingerBands(df['close'], window=20)
        bb_width = (bb.bollinger_hband() - bb.bollinger_lband()) / bb.bollinger_mavg()
        indicators['ATR_14'] = atr
        indicators['BB_width'] = bb_width
        if DEBUG_LEVEL == 'verbose':
            log_message(f"Computed volatility indicators: {list(indicators.keys())}", 'info')
    
    ind_df = pd.DataFrame(indicators, index=df.index)
    
    # Normalize (z-score example)
    for col in progress_bar(ind_df.columns, desc="Normalizing indicators"):
        mean = ind_df[col].rolling(window=36).mean()
        std = ind_df[col].rolling(window=36).std()
        ind_df[col] = (ind_df[col] - mean) / std.where(std != 0)  # Avoid div0
    
    # Corr check (<0.7)
    corr = ind_df.corr()
    high_corr = (corr.abs() > 0.7) & (corr.abs() < 1.0)
    if high_corr.any().any():
        log_message("High correlation detected—consider dropping", 'info')
    
    return ind_df

if __name__ == "__main__":
    from core.data_loader import load_csv_data
    csv_paths = [DATA_PATH]  # From settings
    df = load_csv_data(csv_paths)
    ind_df = select_and_compute_indicators(df)
    log_message("Indicators computed successfully", 'info')

