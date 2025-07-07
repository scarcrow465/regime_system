#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
Simple script to investigate why SMAs are so highly correlated
Prints 100 rows of data to compare values
"""

import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime

# Add regime_system to path
sys.path.insert(0, r'C:\Users\rs\GitProjects\regime_system\ob_model\v2.0_precloud_reorganization_and_clean')

from core.data_loader import load_csv_data
from core.indicators import calculate_all_indicators

print("="*80)
print("SMA CORRELATION INVESTIGATION")
print("="*80)

# Load data
data_file = r'combined_NQ_15m_data.csv'
print(f"Loading data from: {data_file}")
data = load_csv_data(data_file, timeframe='15min')
print(f"Loaded {len(data)} rows")

# Use a subset for faster calculation
subset_size = 1000
if len(data) > subset_size:
    data = data.tail(subset_size)
    print(f"Using last {subset_size} rows for investigation")

# Calculate indicators
print("\nCalculating indicators...")
data_with_indicators = calculate_all_indicators(data, verbose=False)

# Extract the SMAs we want to examine
sma_columns = ['close', 'SMA_5', 'SMA_10', 'SMA_20', 'SMA_50', 'SMA_100', 'SMA_200']
available_smas = [col for col in sma_columns if col in data_with_indicators.columns]

# Get 100 rows of data (skip first 200 to ensure all SMAs are calculated)
start_row = 300
end_row = start_row + 100

print(f"\nShowing rows {start_row} to {end_row}:")
print("="*80)

# Create a subset dataframe for display
display_df = data_with_indicators.iloc[start_row:end_row][available_smas].round(2)

# Calculate the differences between SMAs
if 'SMA_5' in display_df.columns and 'SMA_50' in display_df.columns:
    display_df['SMA5-50_Diff'] = (display_df['SMA_5'] - display_df['SMA_50']).round(2)
    display_df['SMA5-50_Pct'] = ((display_df['SMA_5'] - display_df['SMA_50']) / display_df['SMA_50'] * 100).round(2)

if 'SMA_5' in display_df.columns and 'SMA_200' in display_df.columns:
    display_df['SMA5-200_Diff'] = (display_df['SMA_5'] - display_df['SMA_200']).round(2)
    display_df['SMA5-200_Pct'] = ((display_df['SMA_5'] - display_df['SMA_200']) / display_df['SMA_200'] * 100).round(2)

# Print the data
print(display_df.to_string())

# Calculate actual correlations for this subset
print("\n" + "="*80)
print("CORRELATION ANALYSIS FOR THIS SUBSET:")
print("="*80)

correlation_pairs = [
    ('SMA_5', 'SMA_10'),
    ('SMA_5', 'SMA_20'),
    ('SMA_5', 'SMA_50'),
    ('SMA_5', 'SMA_100'),
    ('SMA_5', 'SMA_200'),
    ('SMA_20', 'SMA_50'),
    ('SMA_50', 'SMA_200')
]

for col1, col2 in correlation_pairs:
    if col1 in display_df.columns and col2 in display_df.columns:
        corr = display_df[col1].corr(display_df[col2])
        diff_mean = abs(display_df[col1] - display_df[col2]).mean()
        diff_pct = (diff_mean / display_df[col2].mean() * 100)
        print(f"{col1} vs {col2}: Correlation = {corr:.4f}, Avg Diff = {diff_mean:.2f} ({diff_pct:.2f}%)")

# Check if values are actually different
print("\n" + "="*80)
print("VALUE RANGE ANALYSIS:")
print("="*80)

for col in available_smas:
    if col in display_df.columns:
        values = display_df[col]
        print(f"{col}:")
        print(f"  Min: {values.min():.2f}")
        print(f"  Max: {values.max():.2f}")
        print(f"  Range: {values.max() - values.min():.2f}")
        print(f"  Std Dev: {values.std():.2f}")

# Save to CSV for manual inspection
output_file = f"sma_investigation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
display_df.to_csv(output_file)
print(f"\n✓ Data saved to: {output_file} for manual inspection")

print("\n" + "="*80)
print("WHAT TO LOOK FOR:")
print("="*80)
print("1. SMA_5 should change much faster than SMA_200")
print("2. The difference between SMA_5 and SMA_200 should be significant")
print("3. In trending markets, they move together but at different speeds")
print("4. If all values are nearly identical, there's a calculation error")

