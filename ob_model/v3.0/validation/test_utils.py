#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# validation/test_utils.py
import pytest
import sys
import os
from config.settings import BASE_DIR  # Import first
sys.path.append(BASE_DIR)  # Now BASE_DIR is defined
from core.data_loader import load_csv_data  # Updated import

SAMPLE_CSV = os.path.join(BASE_DIR, r"C:\Users\rs\OneDrive\Desktop\Excel\Data\New Data\7.1 Master 15m Data - Updated - Nearest Unadjusted - 2014_01_01 - 2025_04_01 - Most removed.csv")  # Create a sample in /tests/ for real testing

def test_data_loader():
    df = load_csv_data([SAMPLE_CSV], ["NQ"])
    assert not df.empty, "Data load failed"
    assert 'close' in df.columns, "Missing OHLC columns"

# def test_ob_load():  # Comment until Phase 1C
#     ob_df = load_ob_csv("sample_ob.csv")
#     assert 'pnl' in ob_df.columns, "Missing OB columns"

if __name__ == '__main__':
    pass  # For direct run if needed, but use pytest

# Run from BASE_DIR: pytest validation/test_utils.py

