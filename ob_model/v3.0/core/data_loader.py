#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import os
import numpy as np
from utils.logger import log_message, progress_bar
from config.settings import DEBUG_LEVEL, SYMBOLS, START_DATE, END_DATE, DATA_PATH
import re

def parse_symbol(symbol_str):
    if not isinstance(symbol_str, str) or pd.isna(symbol_str):
        return None
    symbol_str = str(symbol_str).strip("'\"")
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
            # Read with parse_dates
            df = pd.read_csv(csv_path, parse_dates=['Date'], low_memory=False)
            if DEBUG_LEVEL == 'verbose':
                log_message(f"Raw columns: {df.columns.tolist()}", 'info')
                log_message(f"Sample row: {df.iloc[0].to_dict()}", 'info')
            # Force datetime on 'Date'
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce', format='%m/%d/%Y %H:%M')
            df = df.dropna(subset=['Date']).set_index('Date')
            if df.empty:
                log_message("No valid 'Date' after parsing", 'error')
                continue
            # Tz localize
            df.index = df.index.tz_localize('America/New_York', ambiguous='infer', nonexistent='shift_forward')
            if DEBUG_LEVEL == 'verbose':
                log_message("Index set to tz-aware DatetimeIndex", 'info')
            # Find symbol positions (where column contains 'symbol' or is uppercase code)
            symbol_positions = [i for i, col in enumerate(df.columns) if 'symbol' in col.lower() or re.match(r'^[A-Z]+[A-Z0-9]*$', col)]
            if not symbol_positions:
                log_message("No symbol columns found—check CSV headers", 'error')
                continue
            for pos in symbol_positions:
                block_end = min(pos + 7, len(df.columns))
                block = df.iloc[:, pos:block_end].copy()
                if len(block.columns) < 5: continue
                # Normalize names
                block.columns = [col.lower().replace('.', '') for col in block.columns]
                expected = ['symbol', 'open', 'high', 'low', 'close', 'volume', 'openinterest']
                block = block.rename(columns=dict(zip(block.columns[:7], expected)))
                if DEBUG_LEVEL == 'verbose':
                    log_message(f"Block columns after rename: {block.columns.tolist()}", 'info')
                # Clean numeric
                for col in ['open', 'high', 'low', 'close', 'volume', 'openinterest']:
                    if col in block:
                        block[col] = block[col].astype(str).str.replace(',', '').str.replace('"', '').str.replace("'", '')
                        block[col] = pd.to_numeric(block[col], errors='coerce')
                block = block.dropna(subset=['open', 'high', 'low', 'close'], how='all')
                if block.empty:
                    log_message("Block empty after clean—check numeric values", 'info')
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
        log_message("No valid data loaded—check logs for details", 'error')
        return pd.DataFrame()
    
    combined_df = pd.concat(all_dfs)
    combined_df = combined_df.sort_index()
    # Date filters
    if start_date:
        combined_df = combined_df[combined_df.index >= pd.to_datetime(start_date).tz_localize('America/New_York')]
    if end_date:
        combined_df = combined_df[combined_df.index <= pd.to_datetime(end_date).tz_localize('America/New_York')]
    combined_df = combined_df.reset_index().groupby(['Date', 'BaseSymbol']).first().reset_index()
    combined_df.set_index('Date', inplace=True)
    log_message(f"Loaded {len(combined_df)} rows for symbols: {combined_df['BaseSymbol'].unique()}", 'info')
    return combined_df

if __name__ == "__main__":
    csv_paths = [DATA_PATH]
    df = load_csv_data(csv_paths)
    print(df.head())

