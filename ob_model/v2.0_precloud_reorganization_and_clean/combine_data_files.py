#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
Simple script to combine multiple CSV files for regime analysis
Place this in your regime_system directory
"""

# GITIGNORE TEST RUN 2

import pandas as pd
import os

def combine_csv_files():
    """Combine multiple CSV files into one"""
    
    print("CSV File Combiner for Regime System")
    print("="*50)
    
    # Your data files
    file1 = r"C:\Users\rs\OneDrive\Desktop\Excel\Data\New Data\7.1 Master 15m Data - Updated - Nearest Unadjusted - 2014_01_01 - 2025_04_01 .csv"
    file2 = r"C:\Users\rs\OneDrive\Desktop\Excel\Data\New Data\7.2 Master 15m Data - Updated - Nearest Unadjusted - 2000_01_01 - 2013_12_31 .csv"
    
    # Output file
    output_file = "combined_15m_data.csv"
    
    try:
        # Load first file
        print(f"Loading file 1...")
        df1 = pd.read_csv(file1)
        print(f"  Loaded {len(df1)} rows from 2014-2025")
        
        # Load second file
        print(f"Loading file 2...")
        df2 = pd.read_csv(file2)
        print(f"  Loaded {len(df2)} rows from 2000-2013")
        
        # Combine them (older data first)
        print("\nCombining files...")
        combined = pd.concat([df2, df1], ignore_index=True)
        
        # Sort by date if there's a date column
        date_columns = ['Date', 'date', 'Datetime', 'datetime', 'Time', 'time']
        for col in date_columns:
            if col in combined.columns:
                print(f"Sorting by {col}...")
                combined[col] = pd.to_datetime(combined[col])
                combined = combined.sort_values(col)
                break
        
        # Save combined file
        print(f"\nSaving combined data to {output_file}...")
        combined.to_csv(output_file, index=False)
        
        print(f"\nSuccess! Combined {len(combined)} rows total")
        print(f"Output saved to: {os.path.abspath(output_file)}")
        
        return output_file
        
    except Exception as e:
        print(f"\nError: {e}")
        return None

if __name__ == "__main__":
    output = combine_csv_files()
    if output:
        print(f"\nNow you can run:")
        print(f'python main.py analyze --data "{output}" --timeframe 15min')

