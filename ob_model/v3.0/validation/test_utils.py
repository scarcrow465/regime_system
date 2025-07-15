#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# validation/test_utils.py
import pytest
import os
from core.data_loader import load_csv_data
from config.settings import BASE_DIR

SAMPLE_CSV = os.path.join(BASE_DIR, "tests/sample_data.csv")  # Add a sample CSV in /tests/ for real testing

def test_data_loader():
    df = load_csv_data([SAMPLE_CSV], ["NQ"])
    assert not df.empty, "Data load failed"
    assert 'close' in df.columns, "Missing OHLC columns"

# def test_ob_load():  # Comment until Phase 1C (define load_ob_csv then)
#     ob_df = load_ob_csv("sample_ob.csv")
#     assert 'pnl' in ob_df.columns, "Missing OB columns"

# Run: pytest validation/test_utils.py

