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
            # Read CSV with date parsing
            df = pd.read_csv(csv_path, 
                 parse_dates=['Date'], 
                 date_format='%m/%d/%Y', 
                 index_col='Date',
                 low_memory=False)  # Add this to fix the mixed types warning
            
            # Localize timezone if not already set
            # Check if we have a DatetimeIndex first
            if isinstance(df.index, pd.DatetimeIndex):
                # Localize timezone if not already set
                if df.index.tz is None:
                    df.index = df.index.tz_localize('America/New_York')
                else:
                    df.index = df.index.tz_convert('America/New_York')
            else:
                log_message(f"Warning: Index is not DatetimeIndex, it's {type(df.index)}", 'warning')
                # Try to convert to datetime
                try:
                    df.index = pd.to_datetime(df.index)
                    df.index = df.index.tz_localize('America/New_York')
                except Exception as e:
                    log_message(f"Failed to convert index to datetime: {str(e)}", 'error')
                    return pd.DataFrame()  # Return empty DataFrame on failure
            
            # Identify symbol columns (every 7th column starting at 0, 7, 14, etc.)
            symbol_indices = range(0, len(df.columns), 7)
            for i in symbol_indices:
                symbol = df.columns[i]
                if 'Symbol' in symbol or not symbol:  # Skip if column is misnamed or empty
                    continue
                
                # Define expected column sequence
                expected_cols = ['Symbol', 'Open', 'High', 'Low', 'Close', 'Volume', 'OpenInterest']
                start_idx = i
                end_idx = min(i + 7, len(df.columns))
                symbol_cols = df.columns[start_idx:end_idx].tolist()
                
                # Extract data for this symbol
                sub_df = df.iloc[:, start_idx:end_idx].copy()
                if len(sub_df.columns) == 7:  # Ensure we have all expected columns
                    sub_df.columns = expected_cols
                    sub_df['symbol'] = sub_df['Symbol']
                    
                    # Convert numeric columns, handling commas
                    for col in ['Open', 'High', 'Low', 'Close', 'Volume', 'OpenInterest']:
                        sub_df[col] = sub_df[col].astype(str).str.replace(',', '', regex=False)
                        sub_df[col] = pd.to_numeric(sub_df[col], errors='coerce')
                    
                    # Drop rows missing essential OHLC data
                    sub_df = sub_df.dropna(subset=['Open', 'High', 'Low', 'Close'])
                    
                    if not sub_df.empty:
                        sub_df['BaseSymbol'] = sub_df['symbol'].apply(parse_symbol)
                        if symbols:
                            sub_df = sub_df[sub_df['BaseSymbol'].isin(symbols)]
                        all_dfs.append(sub_df[['symbol', 'Open', 'High', 'Low', 'Close', 'Volume', 'OpenInterest', 'BaseSymbol']])
        
        except Exception as e:
            log_message(f"Error loading {csv_path}: {str(e)}", 'error')
    
    if not all_dfs:
        log_message("No valid data loaded", 'error')
        return pd.DataFrame()
    
    # Concatenate and standardize column names
    combined_df = pd.concat(all_dfs)
    combined_df = combined_df.rename(columns={
        'Open': 'open',
        'High': 'high',
        'Low': 'low',
        'Close': 'close',
        'Volume': 'volume',
        'OpenInterest': 'openinterest'
    })
    combined_df = combined_df.sort_index()
    
    # Apply date filters
    if start_date is not None:
        start_dt = pd.to_datetime(start_date)
        if start_dt.tz is None:
            start_dt = start_dt.tz_localize('America/New_York')
        combined_df = combined_df[combined_df.index >= start_dt]
    
    if end_date is not None:
        end_dt = pd.to_datetime(end_date)
        if end_dt.tz is None:
            end_dt = end_dt.tz_localize('America/New_York')
        combined_df = combined_df[combined_df.index <= end_dt]
    
    # Group by Date and BaseSymbol to avoid duplicates
    combined_df = combined_df.reset_index().groupby(['Date', 'BaseSymbol']).first().reset_index()
    combined_df.set_index('Date', inplace=True)
    
    log_message(f"Loaded {len(combined_df)} total rows for symbols: {combined_df['BaseSymbol'].unique()}", 'info')
    
    return combined_df

if __name__ == "__main__":
    csv_paths = ["path/to/your_wide_csv.csv"]
    df = load_csv_data(csv_paths)
    print(df.head())

