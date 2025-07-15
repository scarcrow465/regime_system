#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pytest
import os
from core.indicators import select_and_compute_indicators
from core.regime_classifier import add_session_labels, fit_gmm
from core.data_loader import load_csv_data
from config.settings import BASE_DIR

SAMPLE_CSV = r"C:\Users\rs\OneDrive\Desktop\Excel\Data\New Data\7.1 Master 15m Data - Updated - Nearest Unadjusted - 2014_01_01 - 2025_04_01 - Most removed.csv" #os.path.join(BASE_DIR, "tests/sample_data.csv")

@pytest.fixture
def sample_df():
    return load_csv_data([SAMPLE_CSV])

def test_indicators(sample_df):
    ind_df = select_and_compute_indicators(sample_df)
    assert not ind_df.empty, "Indicators computation failed"
    assert 'EMA_50' in ind_df.columns, "Missing direction indicator"

def test_sessions(sample_df):
    df = add_session_labels(sample_df)
    assert 'session' in df.columns, "Missing session labels"
    assert 'NY_Open' in df['session'].values, "No NY Open detected"

def test_gmm(sample_df):
    ind_df = select_and_compute_indicators(sample_df)
    model, n = fit_gmm(ind_df)
    assert model is not None, "GMM fit failed"
    assert 2 <= n <= 5, "Invalid n_components"

# Run: pytest validation/indicator_tests.py

