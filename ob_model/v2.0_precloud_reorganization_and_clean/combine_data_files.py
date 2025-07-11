#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
Fixed combine script based on actual CSV format from project knowledge
Handles wide format: Date + (Symbol.X, Open.X, High.X, Low.X, Close.X, Volume.X, OpenInterest.X) * N instruments
"""

import pandas as pd
import numpy as np
import os
import re
from tqdm import tqdm

def parse_symbol(symbol_str):
    """Extract base symbol from futures contract notation (NQH25 -> NQ)"""
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

def combine_csv_files(instrument='NQ'):
    """
    Combine CSV files and extract specific instrument using project's data format
    """
    print("CSV File Combiner for Regime System")
    print("="*50)
    print(f"Selected Instrument: {instrument}")
    
    # 1 hour data files
    file1 = r"C:\Users\rs\OneDrive\Desktop\Excel\Data\New Data\6. Master Hourly Data - Nearest Unadjusted.csv"
    
    # 1 hour data files
    #file1 = r"C:\Users\rs\OneDrive\Desktop\Excel\Data\New Data\6. Master Hourly Data - Nearest Unadjusted.csv"
    
    # 15 minute data files
    # file1 = r"C:\Users\rs\OneDrive\Desktop\Excel\Data\New Data\7.1 Master 15m Data - Updated - Nearest Unadjusted - 2014_01_01 - 2025_04_01 .csv"
    # file2 = r"C:\Users\rs\OneDrive\Desktop\Excel\Data\New Data\7.2 Master 15m Data - Updated - Nearest Unadjusted - 2000_01_01 - 2013_12_31 .csv"
    
    output_file = f"combined_{instrument}_weekly_data.csv"
    
    all_data = []
    
    for file_num, csv_path in enumerate([file1], 1):  # Process older file first - ([file2, file1], 1)]
        print(f"\nProcessing file {file_num}: {os.path.basename(csv_path)}")
        
        try:
            # Read CSV with proper options
            chunk = pd.read_csv(
                csv_path,
                parse_dates=['Date'],
                index_col='Date',
                dtype={'Symbol': str},
                low_memory=False
            )
            
            print(f"  Loaded {len(chunk)} rows")
            
            # Process each instrument column set
            symbol_cols = [col for col in chunk.columns if re.match(r'Symbol(\.\d+)?$', col)]
            print(f"  Found {len(symbol_cols)} instrument sets")
            
            found_instrument = False
            
            for symbol_col in symbol_cols:
                # Extract suffix
                suffix = symbol_col.replace('Symbol', '')
                
                # Get corresponding columns
                col_names = {
                    'symbol': f'Symbol{suffix}',
                    'open': f'Open{suffix}',
                    'high': f'High{suffix}',
                    'low': f'Low{suffix}',
                    'close': f'Close{suffix}',
                    'volume': f'Volume{suffix}',
                    'openinterest': f'OpenInterest{suffix}'
                }
                
                # Check if all columns exist
                if not all(col in chunk.columns for col in col_names.values()):
                    continue
                
                # Extract data for this instrument set
                sub_df = chunk[[col for col in col_names.values()]].copy()
                sub_df.columns = ['symbol', 'open', 'high', 'low', 'close', 'volume', 'openinterest']
                
                # Clean numeric columns
                for col in ['open', 'high', 'low', 'close', 'volume', 'openinterest']:
                    sub_df[col] = sub_df[col].astype(str).str.replace(',', '', regex=False)
                    sub_df[col] = pd.to_numeric(sub_df[col], errors='coerce', downcast='float')
                
                # Extract base symbol
                sub_df['BaseSymbol'] = sub_df['symbol'].apply(lambda x: parse_symbol(x) if pd.notna(x) else None)
                
                # Filter for our instrument
                instrument_data = sub_df[sub_df['BaseSymbol'] == instrument].copy()
                
                if len(instrument_data) > 0:
                    print(f"  Found {len(instrument_data)} rows for {instrument} in column set {symbol_col}")
                    found_instrument = True
                    all_data.append(instrument_data)
            
            if not found_instrument:
                print(f"  WARNING: No data found for {instrument} in this file")
                
        except Exception as e:
            print(f"  ERROR processing file: {e}")
            continue
    
    if not all_data:
        print(f"\nERROR: No data found for {instrument} in any file!")
        return None
    
    # Combine all data
    print(f"\nCombining all {instrument} data...")
    combined = pd.concat(all_data)
    
    # Sort by index (Date)
    combined = combined.sort_index()
    
    # Remove duplicates (keep first occurrence)
    combined = combined[~combined.index.duplicated(keep='first')]
    
    # Reset index to have Date as a column
    combined.reset_index(inplace=True)
    
    # Select final columns for regime system
    final_columns = ['Date', 'symbol', 'open', 'high', 'low', 'close', 'volume']
    combined = combined[final_columns]
    
    # Save
    print(f"\nSaving to {output_file}...")
    combined.to_csv(output_file, index=False)
    
    print(f"\nSuccess! Combined {len(combined)} rows for {instrument}")
    print(f"Date range: {combined['Date'].min()} to {combined['Date'].max()}")
    print(f"Output saved to: {os.path.abspath(output_file)}")
    
    return output_file

if __name__ == "__main__":
    print("Available instruments: NQ, ES, YM, RTY, GC, CL, ZB, ZW, ZS")
    instrument = input("Enter instrument symbol (default NQ): ").strip().upper()
    
    if not instrument:
        instrument = 'NQ'
    
    output = combine_csv_files(instrument)
    if output:
        print(f"\nNow you can run:")
        print(f'python main.py analyze --data "{output}" --timeframe weekly')


# In[ ]:




