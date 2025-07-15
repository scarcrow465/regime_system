#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import os
import numpy as np
from utils.logger import log_message, progress_bar
from config.settings import DEBUG_LEVEL, SYMBOLS, START_DATE, END_DATE, DATA_PATH

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
            df = pd.read_csv(csv_path, index_col='Date', low_memory=False)
            # Force to datetime (handle format)
            df.index = pd.to_datetime(df.index, errors='coerce', format='%m/%d/%Y %H:%M')
            df = df.dropna()  # Drop invalid dates
            # Handle tz
            if df.index.tz is None:
                df.index = df.index.tz_localize('America/New_York', ambiguous='infer', nonexistent='shift_forward')
            else:
                df.index = df.index.tz_convert('America/New_York')
            if DEBUG_LEVEL == 'verbose':
                log_message("Index converted to DatetimeIndex with tz", 'info')
            # Find 'Symbol' columns (case-insensitive)
            symbol_cols = [col for col in df.columns if col.lower() == 'symbol']
            if not symbol_cols:
                log_message("No 'Symbol' columns—assuming repeating blocks", 'info')
                block_size = 7
                symbol_positions = range(0, len(df.columns), block_size)
            else:
                symbol_positions = [df.columns.get_loc(col) for col in symbol_cols]
            for pos in symbol_positions:
                block_end = min(pos + 7, len(df.columns))
                block = df.iloc[:, pos:block_end].copy()
                if len(block.columns) < 5: continue  # Min for OHLC
                # Set columns assuming order
                block.columns = ['symbol', 'open', 'high', 'low', 'close', 'volume', 'openinterest'][:len(block.columns)]
                # Clean numeric
                for col in ['open', 'high', 'low', 'close', 'volume', 'openinterest']:
                    if col in block.columns:
                        block[col] = block[col].astype(str).str.replace(',', '').replace('', np.nan).astype(float, errors='ignore')
                block = block.dropna(subset=['open', 'high', 'low', 'close'], how='all')
                if block.empty:
                    continue
                block['BaseSymbol'] = block['symbol'].apply(parse_symbol)
                if symbols and block['BaseSymbol'].iloc[0] not in symbols:
                    continue
                all_dfs.append(block)
                if DEBUG_LEVEL == 'verbose':
                    log_message(f"Processed block for {block['symbol'].iloc[0]} with {len(block)} rows", 'info')
        except Exception as e:
            log_message(f"Error loading {csv_path}: {str(e)}", 'error')
    
    if not all_dfs:
        log_message("No valid data loaded—check CSV columns/date format", 'error')
        return pd.DataFrame()
    
    combined_df = pd.concat(all_dfs)
    combined_df = combined_df.sort_index()
    # Date filters
    if start_date:
        combined_df = combined_df[combined_df.index >= pd.to_datetime(start_date, utc=True).tz_convert('America/New_York')]
    if end_date:
        combined_df = combined_df[combined_df.index <= pd.to_datetime(end_date, utc=True).tz_convert('America/New_York')]
    combined_df = combined_df.reset_index().groupby(['Date', 'BaseSymbol']).first().reset_index()
    combined_df.set_index('Date', inplace=True)
    log_message(f"Loaded {len(combined_df)} rows for symbols: {combined_df['BaseSymbol'].unique()}", 'info')
    return combined_df

if __name__ == "__main__":
    csv_paths = [DATA_PATH]
    df = load_csv_data(csv_paths)
    print(df.head())

