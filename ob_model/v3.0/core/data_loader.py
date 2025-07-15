#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# core/data_loader.py
import pandas as pd
import os

# Reuse your parse_symbol logic here (paste from ob_15m_hold_time_test_1.py)

def load_wide_csv(paths, symbols, start_date=None, end_date=None, timeframe="15min"):
    # Your full load_csv_data function here (verbatim from ob_15m_hold_time_test_1.py or repo)
    # ... (insert code)
    return combined_df

def load_ob_csv(path):
    df = pd.read_csv(path, parse_dates=['entry_date_time', 'exit_date_time'])
    df['entry_time'] = pd.to_datetime(df['entry_time'], errors='coerce').dt.tz_localize('America/New_York')
    # Add any cleaning for manual columns
    return df

# Example usage: ohlc = load_wide_csv(["path/to/master_5m.csv"], ["NQ"]); ob = load_ob_csv("path/to/backtest.csv")

