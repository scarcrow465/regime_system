#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import os
import sys
import numpy as np
from utils.logger import log_message, progress_bar
from config.settings import DEBUG_LEVEL, SYMBOLS, START_DATE, END_DATE
import re  # For case-insensitive
from config.settings import DATA_PATH  # Import first

def parse_symbol(symbol_str):
    """Extract base symbol from futures contract notation."""
    if not isinstance(symbol_str, str) or pd.isna(symbol_str):
        return None
    if len(symbol_str) < 3:
        return symbol_str
    year = symbol_str[-2:]
    if year.isdigit():
        month = symbol_str[-3]
        valid_month_codes = {'F', 'G', 'H', 'J', 'K', 'M', 'N', 'Q', 'U', 'V', 'X', 'Z'}
        if month in valid_month_codes:
            return symbol_str[:-3]
    return symbol_str

def load_csv_data(csv_paths, symbols=SYMBOLS, start_date=START_DATE, end_date=END_DATE):
    all_dfs = []
    if DEBUG_LEVEL in ['debug', 'verbose']:
        log_message(f"Loading CSVs: {csv_paths}", 'info')
    
    for csv_path in progress_bar(csv_paths, desc="Loading CSVs"):
        try:
            df = pd.read_csv(csv_path, parse_dates=['Date'], date_format='%m/%d/%Y', index_col='Date', low_memory=False)
            if df.index.tz is None:
                df.index = df.index.tz_localize('America/New_York', ambiguous='infer')
            else:
                df.index = df.index.tz_convert('America/New_York')
            # Find positions of 'Symbol' columns (case-insensitive)
            symbol_positions = [i for i, col in enumerate(df.columns) if re.match(r'^symbol$', col.lower())]
            if not symbol_positions and DEBUG_LEVEL == 'verbose':
                log_message("No 'Symbol' columns found—checking all", 'info')
                symbol_positions = range(0, len(df.columns), 7)  # Fallback to every 7 columns
            for pos in symbol_positions:
                if pos + 7 > len(df.columns):
                    continue
                block = df.iloc[:, pos:pos+7].copy()
                block.columns = ['symbol', 'open', 'high', 'low', 'close', 'volume', 'openinterest']
                # Clean numeric (handle commas, empty)
                for col in ['open', 'high', 'low', 'close', 'volume', 'openinterest']:
                    block[col] = block[col].astype(str).str.replace(',', '').replace('', np.nan)
                    block[col] = pd.to_numeric(block[col], errors='coerce')
                block = block.dropna(subset=['open', 'high', 'low', 'close'], how='all')  # Allow partial NaN but not all
                if block.empty:
                    continue
                block['BaseSymbol'] = block['symbol'].apply(parse_symbol)
                if symbols and block['BaseSymbol'].iloc[0] not in symbols if not block.empty else True:
                    continue
                all_dfs.append(block)
                if DEBUG_LEVEL == 'verbose':
                    log_message(f"Processed block for {block['symbol'].iloc[0]} with {len(block)} rows", 'info')
        except Exception as e:
            log_message(f"Error loading {csv_path}: {str(e)}", 'error')
    
    if not all_dfs:
        log_message("No valid data loaded—check CSV structure", 'error')
        return pd.DataFrame()
    
    combined_df = pd.concat(all_dfs)
    combined_df = combined_df.sort_index()
    # Date filters (tz-aware)
    if start_date:
        start_dt = pd.to_datetime(start_date, utc=True).tz_convert('America/New_York')
        combined_df = combined_df[combined_df.index >= start_dt]
    if end_date:
        end_dt = pd.to_datetime(end_date, utc=True).tz_convert('America/New_York')
        combined_df = combined_df[combined_df.index <= end_dt]
    combined_df = combined_df.reset_index().groupby(['Date', 'BaseSymbol']).first().reset_index()
    combined_df.set_index('Date', inplace=True)
    log_message(f"Loaded {len(combined_df)} rows for symbols: {combined_df['BaseSymbol'].unique()}", 'info')
    return combined_df

if __name__ == "__main__":
    csv_paths = [DATA_PATH]  # From settings
    df = load_csv_data(csv_paths)
    print(df.head())

