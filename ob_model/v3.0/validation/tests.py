#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# validation/tests.py (central for all tests)
import pytest
import numpy as np
import pandas as pd
from config.settings import BASE_DIR, DATA_PATH, OB_PATH, TEST_SLICE
from core.data_loader import load_csv_data
from core.indicators import select_and_compute_indicators
from core.regime_classifier import add_session_labels, fit_gmm
from validation.ob_prober import load_ob_csv, merge_regimes, probe_filtering
from optimization.optuna_optimizer import optuna_objective
from utils.logger import log_message
import optuna

@pytest.fixture
def sample_df():
    df = load_csv_data(DATA_PATH)
    if TEST_SLICE > 0:
        df = df.head(TEST_SLICE)
    return df

def test_data_loader(sample_df):
    assert not sample_df.empty, "Data load failed"
    assert 'close' in sample_df.columns, "Missing OHLC columns"

def test_indicators(sample_df):
    ind_df = select_and_compute_indicators(sample_df)
    assert not ind_df.empty, "Indicators computation failed"
    assert 'EMA_50' in ind_df.columns, "Missing direction indicator"

def test_sessions(sample_df):
    df = add_session_labels(sample_df)
    assert 'session' in df.columns, "Missing session labels"
    assert 'NY_Open' in df['session'].values, "No NY Open detected (check data hours)"

def test_gmm(sample_df):
    ind_df = select_and_compute_indicators(sample_df)
    features = ind_df.select_dtypes(include=[np.number]).dropna()
    if len(features) < 5:
        pytest.skip("Insufficient data for GMM test")
    model, n = fit_gmm(features)
    assert model is not None, "GMM fit failed"
    assert 2 <= n <= 5, "Invalid n_components"

def test_load_ob_csv():
    ob_df = load_ob_csv(OB_PATH)
    assert not ob_df.empty, "OB load failed"
    assert 'outcome_win' in ob_df.columns, "Missing outcome_win mapping"

def test_merge_regimes(sample_df):
    ind_df = select_and_compute_indicators(sample_df)
    features = ind_df.select_dtypes(include=[np.number]).dropna()
    model, _ = fit_gmm(features)
    ob_df = load_ob_csv(OB_PATH)
    merged = merge_regimes(ob_df, sample_df, model)
    assert not merged.empty, "Merge failed"
    assert 'regime' in merged.columns, "Missing regime column"
    assert 'hour' in features.columns, "Missing hour in features"

def test_probe_filtering():
    # Sim merged DF
    merged = pd.DataFrame({
        'regime': np.random.randint(0, 3, 100),
        'outcome_win': np.random.choice([0,1], 100),
        'pnl': np.random.normal(0, 100, 100),
        'session': np.random.choice(['NY_Open', 'Other'], 100)
    })
    combos, crosstab = probe_filtering(merged)
    assert not combos.empty, "Probe failed"
    assert 'lift' in combos.columns, "Missing lift calc"

def test_optuna():
    trial = optuna.trial.FixedTrial({'n_components': 3, 'cov_type': 'diag'})
    features = pd.DataFrame(np.random.rand(100, 4))  # Sim
    score = optuna_objective(trial, features)
    assert score is not None, "Objective failed"  # Allow any score for test

# Run: pytest validation/tests.py

