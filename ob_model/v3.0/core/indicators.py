#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pytest
import pandas as pd
from core.indicators import select_and_compute_indicators
from core.regime_classifier import add_session_labels, fit_gmm
from utils.logger import log_message
from config.settings import DATA_PATH  # Use real for optional full test

@pytest.fixture
def mock_df():
    # Mock 50 rows for TA-Lib (safe for window=50)
    dates = pd.date_range('2025-03-12 01:00:00', periods=50, freq='15min', tz='America/New_York')
    data = {
        'open': [100 + i * 0.5 for i in range(50)],
        'high': [105 + i * 0.5 for i in range(50)],
        'low': [95 + i * 0.5 for i in range(50)],
        'close': [102 + i * 0.5 for i in range(50)],
        'volume': [50 + i for i in range(50)],
        'BaseSymbol': ['NQ'] * 50,
        'symbol': ['NQH25'] * 50
    }
    df = pd.DataFrame(data, index=dates)
    return df

def test_indicators(mock_df):
    ind_df = select_and_compute_indicators(mock_df)
    assert not ind_df.empty, "Indicators computation failed"
    assert 'EMA_50' in ind_df.columns, "Missing direction indicator"
    log_message("Indicators test passed", 'info')

def test_sessions(mock_df):
    # Add NY_Open hour
    df = mock_df.copy()
    df.loc[df.index[0], 'hour'] = 9  # Force NY_Open
    df = add_session_labels(df)
    assert 'session' in df.columns, "Missing session labels"
    assert 'NY_Open' in df['session'].values, "No NY Open detected"
    log_message("Sessions test passed", 'info')

def test_gmm(mock_df):
    ind_df = select_and_compute_indicators(mock_df)
    if len(ind_df) < 2:  # GMM needs 2+ rows
        pytest.skip("Insufficient data for GMM")
    model, n = fit_gmm(ind_df.dropna())
    assert model is not None, "GMM fit failed"
    assert 2 <= n <= 5, "Invalid n_components"
    log_message("GMM test passed", 'info')

# Optional: Test with real data (comment out if large)
# @pytest.fixture
# def sample_df():
#     return load_csv_data([DATA_PATH])
# 
# def test_indicators_real(sample_df):
#     ind_df = select_and_compute_indicators(sample_df)
#     assert not ind_df.empty, "Real data indicators failed"

# Run: pytest validation/indicator_tests.py

