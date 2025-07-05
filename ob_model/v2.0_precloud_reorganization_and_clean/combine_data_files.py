#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
Fixed script to combine CSV files and select a specific instrument
"""

import pandas as pd
import os

def combine_csv_files(instrument='NQ'):
    """
    Combine multiple CSV files and extract data for a specific instrument
    
    Args:
        instrument: Which instrument to extract (default 'NQ')
    """
    
    print("CSV File Combiner for Regime System")
    print("="*50)
    print(f"Selected Instrument: {instrument}")
    
    # Your data files
    file1 = r"C:\Users\rs\OneDrive\Desktop\Excel\Data\New Data\7.1 Master 15m Data - Updated - Nearest Unadjusted - 2014_01_01 - 2025_04_01 .csv"
    file2 = r"C:\Users\rs\OneDrive\Desktop\Excel\Data\New Data\7.2 Master 15m Data - Updated - Nearest Unadjusted - 2000_01_01 - 2013_12_31 .csv"
    
    # Output file
    output_file = f"combined_{instrument}_15m_data.csv"
    
    try:
        # Load first file
        print(f"\nLoading file 1...")
        df1 = pd.read_csv(file1)
        print(f"  Loaded {len(df1)} rows")
        print(f"  Columns found: {df1.columns.tolist()[:10]}...")  # Show first 10 columns
        
        # Load second file
        print(f"\nLoading file 2...")
        df2 = pd.read_csv(file2)
        print(f"  Loaded {len(df2)} rows")
        
        # Function to extract data for specific instrument
        def extract_instrument_data(df, instrument):
            # Find columns that contain the instrument symbol
            symbol_cols = [col for col in df.columns if 'symbol' in col.lower()]
            
            # Find which column set contains our instrument
            instrument_data = None
            col_suffix = ""
            
            # Check main columns first (no suffix)
            if 'symbol' in df.columns and instrument in df['symbol'].values:
                # Extract main columns
                cols_to_keep = ['Date', 'date', 'DateTime', 'datetime', 'Time', 'time']
                cols_to_keep += ['open', 'high', 'low', 'close', 'volume']
                existing_cols = [col for col in cols_to_keep if col in df.columns]
                instrument_data = df[df['symbol'] == instrument][existing_cols].copy()
            else:
                # Check numbered columns
                for col in symbol_cols:
                    if col == 'symbol':
                        continue
                    if instrument in df[col].values:
                        # Extract the suffix number
                        suffix = col.replace('symbol', '')
                        
                        # Get corresponding price columns
                        cols_to_keep = ['Date', 'date', 'DateTime', 'datetime', 'Time', 'time']
                        price_cols = [f'open{suffix}', f'high{suffix}', f'low{suffix}', 
                                     f'close{suffix}', f'volume{suffix}']
                        cols_to_keep += price_cols
                        
                        existing_cols = [col for col in cols_to_keep if col in df.columns]
                        instrument_data = df[df[col] == instrument][existing_cols].copy()
                        
                        # Rename columns to standard names
                        for col in price_cols:
                            if col in instrument_data.columns:
                                new_name = col.replace(suffix, '')
                                instrument_data.rename(columns={col: new_name}, inplace=True)
                        
                        break
            
            return instrument_data
        
        # Extract instrument data from both files
        print(f"\nExtracting {instrument} data from file 1...")
        inst_df1 = extract_instrument_data(df1, instrument)
        if inst_df1 is None or len(inst_df1) == 0:
            print(f"WARNING: No {instrument} data found in file 1")
            inst_df1 = pd.DataFrame()
        else:
            print(f"  Found {len(inst_df1)} rows for {instrument}")
        
        print(f"\nExtracting {instrument} data from file 2...")
        inst_df2 = extract_instrument_data(df2, instrument)
        if inst_df2 is None or len(inst_df2) == 0:
            print(f"WARNING: No {instrument} data found in file 2")
            inst_df2 = pd.DataFrame()
        else:
            print(f"  Found {len(inst_df2)} rows for {instrument}")
        
        # Combine the instrument data
        if len(inst_df1) > 0 and len(inst_df2) > 0:
            print(f"\nCombining {instrument} data...")
            combined = pd.concat([inst_df2, inst_df1], ignore_index=True)
        elif len(inst_df1) > 0:
            combined = inst_df1
        elif len(inst_df2) > 0:
            combined = inst_df2
        else:
            print(f"\nERROR: No data found for {instrument} in either file!")
            return None
        
        # Standardize date column
        date_cols = ['Date', 'date', 'DateTime', 'datetime', 'Time', 'time']
        for col in date_cols:
            if col in combined.columns:
                combined['Date'] = pd.to_datetime(combined[col])
                if col != 'Date':
                    combined.drop(columns=[col], inplace=True)
                break
        
        # Sort by date
        if 'Date' in combined.columns:
            combined = combined.sort_values('Date')
            print(f"  Date range: {combined['Date'].min()} to {combined['Date'].max()}")
        
        # Ensure we have the required columns
        required_cols = ['Date', 'open', 'high', 'low', 'close', 'volume']
        final_cols = [col for col in required_cols if col in combined.columns]
        combined = combined[final_cols]
        
        # Save combined file
        print(f"\nSaving combined {instrument} data to {output_file}...")
        combined.to_csv(output_file, index=False)
        
        print(f"\nSuccess! Combined {len(combined)} rows for {instrument}")
        print(f"Output saved to: {os.path.abspath(output_file)}")
        
        return output_file
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    # Ask user which instrument they want
    print("Available instruments: NQ, ES, YM, RTY, GC, CL, ZB, ZW, ZS")
    instrument = input("Enter instrument symbol (default NQ): ").strip().upper()
    
    if not instrument:
        instrument = 'NQ'
    
    output = combine_csv_files(instrument)
    if output:
        print(f"\nNow you can run:")
        print(f'python main.py analyze --data "{output}" --timeframe 15min')

