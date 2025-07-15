#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# validation/test_utils.py
import pytest
from core.data_loader import load_wide_csv, load_ob_csv

def test_data_loader():
    df = load_wide_csv(["sample_path.csv"], ["NQ"])
    assert not df.empty, "Data load failed"
    assert 'close' in df.columns, "Missing OHLC columns"

def test_ob_load():
    ob_df = load_ob_csv("sample_ob.csv")
    assert 'pnl' in ob_df.columns, "Missing OB columns"

# Run: pytest validation/test_utils.py

