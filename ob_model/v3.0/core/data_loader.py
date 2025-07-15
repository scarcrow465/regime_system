#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import os
from utils.logger import log_message, progress_bar
from config.settings import DEBUG_LEVEL, SYMBOLS, START_DATE, END_DATE
import re  # For case-insensitive matching

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
            chunks = pd.read_csv(csv_path, parse_dates=['Date'], index_col='Date', chunksize=100000, dtype={'Symbol': str})
            num_chunks = None
            for chunk in progress_bar(chunks, desc="Processing chunks", total=num_chunks if DEBUG_LEVEL in ['debug', 'verbose'] else None):
                chunk = chunk.copy()
                
                # Handle timezone
                try:
                    if chunk.index.tz is None:
                        chunk.index = chunk.index.tz_localize('America/New_York', 
                                                             ambiguous='raise', 
                                                             nonexistent='shift_forward')
                    else:
                        chunk.index = chunk.index.tz_convert('America/New_York')
                except Exception:
                    if DEBUG_LEVEL in ['debug', 'verbose']:
                        log_message("Timezone handling skipped for chunk", 'info')
                    continue
                
                # Find all symbol columns (case-insensitive, handles duplicates as Symbol.1 etc.)
                symbol_cols = [col for col in chunk.columns if re.match(r'^(symbol|Symbol|SYMBOL)(\.\d+)?$', col, re.IGNORECASE)]
                
                if not symbol_cols:
                    if DEBUG_LEVEL == 'verbose':
                        log_message("No symbol columns found in chunk", 'info')
                    continue
                
                for sym_col in symbol_cols:
                    suffix = sym_col[sym_col.find('.'):] if '.' in sym_col else ''
                    
                    # Expected columns (case-insensitive)
                    base_cols = ['open', 'high', 'low', 'close', 'volume']
                    optional_cols = ['openinterest']
                    
                    # Find actual columns
                    actual_cols = {}
                    for base in ['symbol'] + base_cols + optional_cols:
                        for c in chunk.columns:
                            if c.lower() == f'{base}{suffix}'.lower():
                                actual_cols[base] = c
                                break
                    
                    # Required: symbol + base_cols
                    if not all(base in actual_cols for base in ['symbol'] + base_cols):
                        if DEBUG_LEVEL == 'verbose':
                            log_message(f"Missing required columns for {sym_col}", 'info')
                        continue
                    
                    # Extract
                    extract_cols = [actual_cols['symbol']] + [actual_cols[base] for base in base_cols]
                    if 'openinterest' in actual_cols:
                        extract_cols.append(actual_cols['openinterest'])
                    
                    sub_df = chunk[extract_cols].copy()
                    new_cols = ['symbol', 'open', 'high', 'low', 'close', 'volume']
                    if 'openinterest' in actual_cols:
                        new_cols.append('openinterest')
                    sub_df.columns = new_cols
                    
                    # Clean numeric
                    for col in sub_df.columns[1:]:
                        sub_df[col] = sub_df[col].astype(str).str.replace(',', '', regex=False)
                        sub_df[col] = pd.to_numeric(sub_df[col], errors='coerce')
                    
                    sub_df = sub_df.dropna(subset=['open', 'high', 'low', 'close'])
                    
                    if sub_df.empty:
                        continue
                    
                    sub_df['BaseSymbol'] = sub_df['symbol'].apply(parse_symbol)
                    
                    if symbols:
                        sub_df = sub_df[sub_df['BaseSymbol'].isin(symbols)]
                    
                    if not sub_df.empty:
                        all_dfs.append(sub_df)
                        if DEBUG_LEVEL in ['debug', 'verbose']:
                            log_message(f"Processed {sym_col} with {len(sub_df)} rows", 'info')

                    if DEBUG_LEVEL == 'verbose':
                        log_message(f"Found columns for {sym_col}: {actual_cols}", 'info')
                
                if DEBUG_LEVEL == 'verbose':
                    log_message(f"Processed chunk with {len(chunk)} rows", 'info')
        except Exception as e:
            log_message(f"Error loading {csv_path}: {str(e)}", 'error')
    
    if not all_dfs:
        log_message("No valid data loaded", 'error')
        return pd.DataFrame()
    
    combined_df = pd.concat(all_dfs)
    combined_df = combined_df.sort_index()
    
    # Apply date filters with tz-aware
    if start_date is not None:
        start_dt = pd.to_datetime(start_date)
        if start_dt.tz is None:
            start_dt = start_dt.tz_localize('America/New_York')
        else:
            start_dt = start_dt.tz_convert('America/New_York')
        combined_df = combined_df[combined_df.index >= start_dt]
    
    if end_date is not None:
        end_dt = pd.to_datetime(end_date)
        if end_dt.tz is None:
            end_dt = end_dt.tz_localize('America/New_York')
        else:
            end_dt = end_dt.tz_convert('America/New_York')
        combined_df = combined_df[combined_df.index <= end_dt]
    
    combined_df = combined_df.reset_index().groupby(['Date', 'BaseSymbol']).first().reset_index()
    combined_df.set_index('Date', inplace=True)
    
    log_message(f"Loaded {len(combined_df)} total rows for symbols: {combined_df['BaseSymbol'].unique()}", 'info')
    
    return combined_df

