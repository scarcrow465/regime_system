#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# core/data_loader.py
import pandas as pd
import os


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

def load_wide_csv(paths, symbols, start_date=None, end_date=None, timeframe="15min"):
    """Load and process CSV data for specified symbols with explicit dtype handling."""
    if isinstance(csv_paths, str):
        csv_paths = [csv_paths]
    
    all_dfs = []
    
    for csv_path in csv_paths:
        try:
            # Read CSV in chunks for memory efficiency
            chunks = pd.read_csv(
                csv_path,
                parse_dates=['Date'],
                index_col='Date',
                chunksize=100000,
                dtype={'Symbol': str},
                low_memory=False
            )
            
            for chunk in chunks:
                if chunk.empty or chunk.index.hasnans:
                    continue
                
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
                    continue
                
                # Find all symbol columns
                symbol_cols = [col for col in chunk.columns 
                             if col.startswith('Symbol')]
                
                if not symbol_cols:
                    continue
                
                # Process each instrument's columns
                for sym_col in symbol_cols:
                    # Extract suffix (e.g., '.1', '.2', or '')
                    suffix = sym_col.replace('Symbol', '')
                    
                    # Expected columns for this instrument
                    expected_cols = {
                        'symbol': sym_col,
                        'open': f'Open{suffix}',
                        'high': f'High{suffix}',
                        'low': f'Low{suffix}',
                        'close': f'Close{suffix}',
                        'volume': f'Volume{suffix}',
                        'openinterest': f'OpenInterest{suffix}'
                    }
                    
                    # Check if all expected columns exist
                    if not all(col in chunk.columns for col in expected_cols.values()):
                        continue
                    
                    # Extract instrument data
                    sub_df = chunk[list(expected_cols.values())].copy()
                    sub_df.columns = ['symbol', 'open', 'high', 'low', 'close', 'volume', 'openinterest']
                    
                    # Clean numeric columns
                    for col in ['open', 'high', 'low', 'close', 'volume', 'openinterest']:
                        sub_df[col] = sub_df[col].astype(str).str.replace(',', '', regex=False)
                        sub_df[col] = pd.to_numeric(sub_df[col], errors='coerce')
                    
                    # Drop rows with invalid OHLC data
                    sub_df = sub_df.dropna(subset=['open', 'high', 'low', 'close'])
                    
                    if sub_df.empty:
                        continue
                    
                    # Get first valid symbol value
                    symbol_series = sub_df['symbol'].dropna()
                    if symbol_series.empty:
                        continue
                    
                    symbol_value = symbol_series.iloc[0]
                    base_symbol = parse_symbol(symbol_value)
                    
                    if base_symbol and base_symbol in symbols:
                        sub_df['symbol'] = symbol_value
                        sub_df['BaseSymbol'] = base_symbol
                        all_dfs.append(sub_df)
                        logger.info(f"Processed {sym_col} (base: {base_symbol}) with {len(sub_df)} rows")
                    
        except Exception as e:
            logger.error(f"Error reading CSV {csv_path}: {e}")
            continue
    
    if not all_dfs:
        logger.error("No valid data loaded from any CSV files.")
        return pd.DataFrame()
    
    # Combine all data
    combined_df = pd.concat(all_dfs)
    
    # Sort by index and remove duplicates
    combined_df = combined_df.sort_index()
    
    # Apply date filters
    if start_date is not None:
        combined_df = combined_df[combined_df.index >= start_date]
    if end_date is not None:
        combined_df = combined_df[combined_df.index <= end_date]
    
    # Remove duplicate timestamps for each symbol
    combined_df = combined_df.reset_index().groupby(['Date', 'BaseSymbol']).first().reset_index()
    combined_df.set_index('Date', inplace=True)
    
    logger.info(f"Loaded {len(combined_df)} total rows for symbols: {combined_df['BaseSymbol'].unique()}")
    
    return combined_df

def load_ob_csv(path):
    df = pd.read_csv(path, parse_dates=['entry_date_time', 'exit_date_time'])
    df['entry_time'] = pd.to_datetime(df['entry_time'], errors='coerce').dt.tz_localize('America/New_York')
    # Add any cleaning for manual columns
    return df

# Example usage: ohlc = load_wide_csv(["path/to/master_5m.csv"], ["NQ"]); ob = load_ob_csv("path/to/backtest.csv")

