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

# Dirs (already created, but for reference)
EXPORTS_DIR = os.path.join(BASE_DIR, 'exports')
LOGS_DIR = os.path.join(EXPORTS_DIR, 'logs')
# etc.

