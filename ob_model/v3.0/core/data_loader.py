#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import os
from utils.logger import log_message, progress_bar
from config.settings import DEBUG_LEVEL, SYMBOLS, START_DATE, END_DATE

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
            num_chunks = None  # Could estimate via file size if needed
            for chunk in progress_bar(chunks, desc="Processing chunks", total=num_chunks if DEBUG_LEVEL in ['debug', 'verbose'] else None):
                chunk = chunk.copy()
                
                # Handle timezone (unchanged)
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
                
                # Find all symbol columns (unchanged)
                symbol_cols = [col for col in chunk.columns if col.startswith('Symbol')]
                
                if not symbol_cols:
                    continue
                
                # Process each instrument's columns
                for sym_col in symbol_cols:
                    # Extract suffix... (unchanged)
                    suffix = sym_col.replace('Symbol', '')
                    
                    # Expected columns... (unchanged)
                    expected_cols = {
                        'symbol': sym_col,
                        'open': f'Open{suffix}',
                        'high': f'High{suffix}',
                        'low': f'Low{suffix}',
                        'close': f'Close{suffix}',
                        'volume': f'Volume{suffix}',
                        'openinterest': f'OpenInterest{suffix}'
                    }
                    
                    # Check if all expected... (unchanged)
                    if not all(col in chunk.columns for col in expected_cols.values()):
                        continue
                    
                    # Extract... (unchanged)
                    sub_df = chunk[list(expected_cols.values())].copy()
                    sub_df.columns = ['symbol', 'open', 'high', 'low', 'close', 'volume', 'openinterest']
                    
                    # Clean numeric... (unchanged)
                    for col in ['open', 'high', 'low', 'close', 'volume', 'openinterest']:
                        sub_df[col] = sub_df[col].astype(str).str.replace(',', '', regex=False)
                        sub_df[col] = pd.to_numeric(sub_df[col], errors='coerce')
                    
                    # Drop invalid... (unchanged)
                    sub_df = sub_df.dropna(subset=['open', 'high', 'low', 'close'])
                    
                    if sub_df.empty:
                        continue
                    
                    # Get first valid... (unchanged)
                    symbol_series = sub_df['symbol'].dropna()
                    if symbol_series.empty:
                        continue
                    
                    symbol_value = symbol_series.iloc[0]
                    base_symbol = parse_symbol(symbol_value)
                    
                    if base_symbol and base_symbol in symbols:
                        sub_df['symbol'] = symbol_value
                        sub_df['BaseSymbol'] = base_symbol
                        all_dfs.append(sub_df)
                        if DEBUG_LEVEL in ['debug', 'verbose']:
                            log_message(f"Processed {sym_col} (base: {base_symbol}) with {len(sub_df)} rows", 'info')
                
                if DEBUG_LEVEL == 'verbose':
                    log_message(f"Processed chunk with {len(chunk)} rows", 'info')
                # Removed duplicate append here (was outside loop)
        except Exception as e:
            log_message(f"Error loading {csv_path}: {str(e)}", 'error')
    
    if not all_dfs:
        log_message("No valid data loaded", 'error')
        return pd.DataFrame()
    
    combined_df = pd.concat(all_dfs)
    # Sort by index and remove duplicates (unchanged)
    combined_df = combined_df.sort_index()
    
    # Apply date filters (unchanged)
    if start_date is not None:
        combined_df = combined_df[combined_df.index >= pd.to_datetime(start_date)]
    if end_date is not None:
        combined_df = combined_df[combined_df.index <= pd.to_datetime(end_date)]
    
    combined_df = combined_df.reset_index().groupby(['Date', 'BaseSymbol']).first().reset_index()
    combined_df.set_index('Date', inplace=True)    

    if DEBUG_LEVEL != 'none':
        log_message(f"Loaded {len(combined_df)} total rows for symbols: {combined_df['BaseSymbol'].unique()}", 'info')
    return combined_df

