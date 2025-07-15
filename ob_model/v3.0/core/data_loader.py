#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import logging
import os
import re
import pytz
from utils.logger import log_message, progress_bar
from config.settings import DEBUG_LEVEL, SYMBOLS, START_DATE, END_DATE, DATA_PATH

logger = logging.getLogger(__name__)

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
    
    tz = pytz.timezone('America/New_York')
    
    for csv_path in progress_bar(csv_paths, desc="Loading CSVs"):
        try:
            df = pd.read_csv(csv_path, low_memory=False)
            if DEBUG_LEVEL == 'verbose':
                log_message(f"Raw columns: {df.columns.tolist()}", 'info')
                log_message(f"Sample row: {df.iloc[0].to_dict()}", 'info')
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce', format='%m/%d/%Y %H:%M')
            df = df.dropna(subset=['Date']).set_index('Date')
            if df.empty:
                log_message("No valid 'Date' after parsing", 'error')
                continue
            if df.index.tz is None:
                df.index = df.index.tz_localize(tz, ambiguous='infer', nonexistent='shift_forward')
            else:
                df.index = df.index.tz_convert(tz)
            if DEBUG_LEVEL == 'verbose':
                log_message("Index localized with pytz", 'info')
            # Find symbol positions
            symbol_positions = [i for i, col in enumerate(df.columns) if 'symbol' in col.lower() or re.match(r'^[A-Z]+[A-Z0-9]*$', col)]
            if not symbol_positions:
                log_message("No symbol columns found—check CSV headers", 'error')
                continue
            for pos in symbol_positions:
                block_end = min(pos + 7, len(df.columns))
                block = df.iloc[:, pos:block_end].copy()
                if len(block.columns) < 5: continue
                block.columns = [col.strip().lower() for col in block.columns]
                expected = ['symbol', 'open', 'high', 'low', 'close', 'volume', 'openinterest']
                block = block.rename(columns=dict(zip(block.columns, expected[:len(block.columns)])))
                for col in ['open', 'high', 'low', 'close', 'volume', 'openinterest']:
                    if col in block:
                        block[col] = block[col].astype(str).str.replace(',', '').str.replace('"', '').str.replace("'", '')
                        block[col] = pd.to_numeric(block[col], errors='coerce')
                block = block.dropna(subset=['open', 'high', 'low', 'close'], how='all')
                if block.empty:
                    log_message("Block empty after numeric clean", 'info')
                    continue
                block['BaseSymbol'] = block['symbol'].apply(parse_symbol)
                if symbols and block['BaseSymbol'].iloc[0] not in symbols:
                    continue
                block = validate_and_clean_data(block)
                all_dfs.append(block)
                if DEBUG_LEVEL == 'verbose':
                    log_message(f"Processed block for {block['symbol'].iloc[0]} with {len(block)} rows", 'info')
        except Exception as e:
            log_message(f"Error loading {csv_path}: {str(e)}", 'error')
    
    if not all_dfs:
        log_message("No valid data loaded—check CSV for 'Date' and OHLC columns", 'error')
        return pd.DataFrame()
    
    combined_df = pd.concat(all_dfs)
    combined_df = combined_df.sort_index()
    # Date filters with tz check
    if start_date:
        start_dt = pd.to_datetime(start_date)
        if start_dt.tz is None:
            start_dt = start_dt.tz_localize(tz)
        else:
            start_dt = start_dt.tz_convert(tz)
        combined_df = combined_df[combined_df.index >= start_dt]
    if end_date:
        end_dt = pd.to_datetime(end_date)
        if end_dt.tz is None:
            end_dt = end_dt.tz_localize(tz)
        else:
            end_dt = end_dt.tz_convert(tz)
        combined_df = combined_df[combined_df.index <= end_dt]
    combined_df = combined_df.reset_index().groupby(['Date', 'BaseSymbol']).first().reset_index()
    combined_df.set_index('Date', inplace=True)
    log_message(f"Loaded {len(combined_df)} rows for symbols: {combined_df['BaseSymbol'].unique()}", 'info')
    return combined_df

def validate_and_clean_data(df: pd.DataFrame) -> pd.DataFrame:
    initial_rows = len(df)
    
    # Remove duplicates
    df = df[~df.index.duplicated(keep='first')]
    
    # Remove rows with any NaN in OHLC
    ohlc_cols = ['open', 'high', 'low', 'close']
    existing_ohlc = [col for col in ohlc_cols if col in df.columns]
    df = df.dropna(subset=existing_ohlc)
    
    # Validate price relationships
    if all(col in df.columns for col in ['open', 'high', 'low', 'close']):
        invalid_hl = df['high'] < df['low']
        if invalid_hl.any():
            log_message(f"Found {invalid_hl.sum()} rows with high < low, fixing...", 'info')
            df.loc[invalid_hl, 'high'] = df.loc[invalid_hl, ['open', 'close']].max(axis=1)
            df.loc[invalid_hl, 'low'] = df.loc[invalid_hl, ['open', 'close']].min(axis=1)
    
    # Remove rows with zero or negative prices
    for col in existing_ohlc:
        df = df[df[col] > 0]
    
    final_rows = len(df)
    if final_rows < initial_rows:
        log_message(f"Data cleaning removed {initial_rows - final_rows} rows", 'info')
    
    return df

if __name__ == "__main__":
    csv_paths = [DATA_PATH]
    df = load_csv_data(csv_paths)
    print(df.head())

