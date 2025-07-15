#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pytest
from core.indicators import select_and_compute_indicators
from core.regime_classifier import add_session_labels, fit_gmm
from core.data_loader import load_csv_data
from utils.logger import log_message
from config.settings import DATA_PATH, TEST_SLICE

@pytest.fixture
def real_df():
    df = load_csv_data([DATA_PATH])
    if len(df) == 0:
        log_message("Failed to load real data—check path/CSV/columns", 'error')
        raise ValueError("Empty DF—see logs for details")
    if TEST_SLICE > 0:
        df = df.head(TEST_SLICE)
        log_message(f"Sliced to {TEST_SLICE} rows for testing", 'info')
    return df

def test_indicators(real_df):
    ind_df = select_and_compute_indicators(real_df)
    assert not ind_df.empty, "Indicators computation failed"
    assert 'EMA_50' in ind_df.columns, "Missing direction indicator"
    log_message("Indicators test passed with real data", 'info')

def test_sessions(real_df):
    df = add_session_labels(real_df)
    assert 'session' in df.columns, "Missing session labels"
    assert 'NY_Open' in df['session'].values, "No NY Open detected (check data hours)"
    log_message("Sessions test passed with real data", 'info')

def test_gmm(real_df):
    ind_df = select_and_compute_indicators(real_df)
    features = ind_df.dropna()
    if len(features) < 50:
        raise ValueError("Insufficient data for GMM (need 50+ rows)")
    model, n = fit_gmm(features)
    assert model is not None, "GMM fit failed"
    assert 2 <= n <= 5, "Invalid n_components"
    log_message("GMM test passed with real data", 'info')

# Run: pytest validation/indicator_tests.py

