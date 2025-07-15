#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import os
from utils.logger import log_message, progress_bar
from config.settings import DEBUG_LEVEL, SYMBOLS, START_DATE, END_DATE

def load_csv_data(csv_paths, symbols=SYMBOLS, start_date=START_DATE, end_date=END_DATE):
    all_dfs = []
    if DEBUG_LEVEL in ['debug', 'verbose']:
        log_message(f"Loading CSVs: {csv_paths}", 'info')
    
    for csv_path in progress_bar(csv_paths, desc="Loading CSVs"):
        try:
            chunks = pd.read_csv(csv_path, parse_dates=['Date'], index_col='Date', chunksize=100000, dtype={'Symbol': str})
            for chunk in progress_bar(chunks, desc="Processing chunks", total=None if DEBUG_LEVEL=='summary' else 10):  # Sim total
                # Repo logic here (parse_symbol, extract sub_df, clean, tz_convert, filter)
                # ... (copy from ob_15m_hold_time_test_1.py load_csv_data, replace prints with log_message)
                if DEBUG_LEVEL == 'verbose':
                    log_message(f"Processed chunk with {len(chunk)} rows", 'info')
                all_dfs.append(sub_df)  # Sim
        except Exception as e:
            log_message(f"Error loading {csv_path}: {str(e)}", 'error')
    
    combined_df = pd.concat(all_dfs)
    # More repo logic (sort, dedup, filter dates/hours)
    if DEBUG_LEVEL != 'none':
        log_message(f"Loaded {len(combined_df)} rows for {symbols}", 'info')
    return combined_df

