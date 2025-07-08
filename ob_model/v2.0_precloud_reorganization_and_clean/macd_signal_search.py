#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
Quick script to check what MACD-related columns actually exist
"""

import pandas as pd
import sys
import os

# Add regime_system to path
sys.path.insert(0, r'C:\Users\rs\GitProjects\regime_system\ob_model\v2.0_precloud_reorganization_and_clean')

from core.data_loader import load_csv_data
from core.indicators import calculate_all_indicators

# Load small sample
data_file = r'combined_NQ_15m_data.csv'
data = load_csv_data(data_file, timeframe='15min').tail(1000)

# Calculate indicators
data_with_indicators = calculate_all_indicators(data, verbose=False)

# Find all MACD-related columns
macd_columns = [col for col in data_with_indicators.columns if 'MACD' in col.upper()]

print("MACD-related columns found:")
for col in macd_columns:
    print(f"  '{col}'")

# Also show all columns containing 'signal'
signal_columns = [col for col in data_with_indicators.columns if 'signal' in col.lower()]
print("\nColumns containing 'signal':")
for col in signal_columns:
    print(f"  '{col}'")

# Check if specific columns exist
check_names = ['MACD_Signal', 'MACD_signal', 'MACD_Signal_Line', 'MACD_signal_line']
print("\nChecking specific column names:")
for name in check_names:
    exists = name in data_with_indicators.columns
    print(f"  '{name}': {'EXISTS' if exists else 'NOT FOUND'}")

