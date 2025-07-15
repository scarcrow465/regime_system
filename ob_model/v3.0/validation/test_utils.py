#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# validation/test_utils.py
import pytest
import sys
import os
from config.settings import BASE_DIR
sys.path.append(BASE_DIR)  # For package imports
from core.data_loader import load_csv_data  # Updated import

SAMPLE_CSV = os.path.join(BASE_DIR, "tests/sample_data.csv")  # Create a sample in /tests/ for real testing

def test_data_loader():
    df = load_csv_data([SAMPLE_CSV], ["NQ"])
    assert not df.empty, "Data load failed"
    assert 'close' in df.columns, "Missing OHLC columns"

# def test_ob_load():  # Comment until Phase 1C (define load_ob_csv in core/ then)
#     ob_df = load_ob_csv("sample_ob.csv")
#     assert 'pnl' in ob_df.columns, "Missing OB columns"

# Run from BASE_DIR: pytest validation/test_utils.py

