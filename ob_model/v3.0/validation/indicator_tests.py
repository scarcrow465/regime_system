#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pytest
from core.indicators import select_and_compute_indicators
from core.regime_classifier import add_session_labels, fit_gmm
from core.data_loader import load_csv_data
from utils.logger import log_message
from config.settings import DATA_PATH  # Use real path from settings

@pytest.fixture
def real_df():
    df = load_csv_data([DATA_PATH])
    if len(df) == 0:
        pytest.fail("Failed to load real data from DATA_PATH—check path/CSV")
    log_message(f"Loaded {len(df)} rows from real data for testing", 'info')
    return df  # Use full, or df.head(1000) if too large for quick tests

def test_indicators(real_df):
    ind_df = select_and_compute_indicators(real_df)
    assert not ind_df.empty, "Indicators computation failed"
    assert 'EMA_50' in ind_df.columns, "Missing direction indicator"
    log_message("Indicators test passed with real data", 'info')

def test_sessions(real_df):
    df = add_session_labels(real_df)
    assert 'session' in df.columns, "Missing session labels"
    assert 'NY_Open' in df['session'].values, "No NY Open detected (check if data has hours 8-10)"
    log_message("Sessions test passed with real data", 'info')

def test_gmm(real_df):
    ind_df = select_and_compute_indicators(real_df)
    if len(ind_df.dropna()) < 50:  # Min for stable GMM
        pytest.skip("Insufficient real data for GMM—need 50+ rows after dropna")
    model, n = fit_gmm(ind_df.dropna())
    assert model is not None, "GMM fit failed"
    assert 2 <= n <= 5, "Invalid n_components"
    log_message("GMM test passed with real data", 'info')

# Run: pytest validation/indicator_tests.py

