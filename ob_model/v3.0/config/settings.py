#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os

BASE_DIR = r"C:\Users\rs\GitProjects\regime_system\ob_model\v3.0"

# Debug levels: 'none', 'summary', 'debug', 'verbose'
DEBUG_LEVEL = 'verbose'  # Change to toggle output

# Other settings (expand as needed)
TIMEFRAME = '15min'
SYMBOLS = ['NQ']
START_DATE = '2008-01-01 00:00:00-05:00'
END_DATE = '2025-07-14 00:00:00-04:00'  # Current date

DATA_PATH = [
    r"C:\Users\rs\OneDrive\Desktop\Excel\Data\New Data\7.1 Master 15m Data - Updated - Nearest Unadjusted - 2014_01_01 - 2025_04_01 .csv"
    r"C:\Users\rs\OneDrive\Desktop\Excel\Data\New Data\7.2 Master 15m Data - Updated - Nearest Unadjusted - 2000_01_01 - 2013_12_31 .csv"
]

# Dirs (already created, but for reference)
EXPORTS_DIR = os.path.join(BASE_DIR, 'exports')
LOGS_DIR = os.path.join(EXPORTS_DIR, 'logs')
# etc.

TEST_SLICE = 0  # Rows to slice for fast tests (0 for full)

OB_PATH = r"C:\Users\rs\OneDrive\Desktop\TV DB\Backtest_Results\Backtest_Results\20250702_2051_NQ_backtest.csv"


# In[ ]:




