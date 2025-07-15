#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
from ta.trend import EMAIndicator, ADXIndicator  # TA-Lib; pip if needed, but from repo context
from ta.volatility import AverageTrueRange, BollingerBands
from utils.logger import log_message, progress_bar
from config.settings import DEBUG_LEVEL
from core.data_loader import load_csv_data  # Reuse

def select_and_compute_indicators(df, regime_classes=['direction', 'volatility']):
    """Select 2-3 low-corr indicators per class and compute."""
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
            log_message(f"Computed direction indicators: {indicators.keys()}", 'info')
    
    # Volatility: ATR_14, BB_width
    if 'volatility' in regime_classes:
        atr = AverageTrueRange(df['high'], df['low'], df['close'], window=14).average_true_range()
        bb = BollingerBands(df['close'], window=20)
        bb_width = (bb.bollinger_hband() - bb.bollinger_lband()) / bb.bollinger_mavg()
        indicators['ATR_14'] = atr
        indicators['BB_width'] = bb_width
        if DEBUG_LEVEL == 'verbose':
            log_message(f"Computed volatility indicators: {indicators.keys()}", 'info')
    
    ind_df = pd.DataFrame(indicators, index=df.index)
    
    # Normalize (z-score example)
    for col in progress_bar(ind_df.columns, desc="Normalizing indicators"):
        ind_df[col] = (ind_df[col] - ind_df[col].rolling(window=36).mean()) / ind_df[col].rolling(window=36).std()  # From prompts
    
    # Corr check (<0.7)
    corr = ind_df.corr()
    high_corr = (corr.abs() > 0.7) & (corr.abs() < 1.0)
    if high_corr.any().any():
        log_message("High correlation detected—consider dropping", 'info')
    
    return ind_df

if __name__ == "__main__":
    csv_paths = ["path/to/your_wide_csv.csv"]  # Update
    df = load_csv_data(csv_paths)
    ind_df = select_and_compute_indicators(df)
    log_message("Indicators computed successfully", 'info')

