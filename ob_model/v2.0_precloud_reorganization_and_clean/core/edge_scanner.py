#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
Edge Scanner for Fingerprint Detection
Scans for asymmetries (edges) across taxonomy—broad first (low threshold to flag potentials), then conditional (subsets like low-vol for latent edges e.g., RSI2 post-1983).
Why: Avoids missing multiples/conditionals—tests all scopes (scalping to position) on historical returns.
How it ties to vision: Extracts "why" behind OB (e.g., trending edge reliable >5 days), scaling to multi-asset.
Use: Run on data, output edge_map dict for classifier.
"""

import pandas as pd
import numpy as np
from scipy import stats  # For t-tests/p-values
from utils.logger import get_logger, log_execution_time, log_errors
from utils.debug_utils import check_data_sanity, log_var_state
from config.edge_taxonomy import PRIMARY_CATEGORIES, SUB_CLASSIFIERS, THRESHOLDS
import logging  # For logging.WARNING
from config.settings import VERBOSE

logger = get_logger('edge_scanner')  # Define logger first

@log_execution_time
@log_errors
def scan_for_edges(df: pd.DataFrame) -> dict:
    """
    Main scan function—broad tests for each primary category, conditional subsets.
    - Input: df with 'returns' column (from data_loader.py).
    - Output: edge_map = {'behavioral': {'score': 0.45, 'details': ...}, ...}—multiples OK.
    - Why visual: Dict structure easy for heatmaps (rows=primary, columns=scopes).
    """
    df = check_data_sanity(df, logger, 'edge_scanner')  # Debug check
    edge_map = {}
    
    for category, desc in PRIMARY_CATEGORIES.items():
        logger.info(f"Scanning {category}: {desc}")
        
        # Broad test: Basic stat on returns (e.g., mean >0 for directional)
        broad_score, broad_p = basic_asymmetry_test(df['returns'], category)
        log_var_state('broad_results', {'score': broad_score, 'p': broad_p}, logger)
        
        # Always create entry, even if broad low—conditionals might unlock
        edge_map[category] = {'broad_score': broad_score, 'broad_p': broad_p}
        
        # Conditional: Subset tests (e.g., low-vol)
        conditional_score = conditional_subset_test(df, category)
        edge_map[category]['conditional_score'] = conditional_score  # Add always—0 if no boost
        
        # Test scopes: Simulate holds per your definitions
        scope_results = test_scopes(df, category)
        edge_map[category]['scopes'] = scope_results
        
        # Check if all low—warning but continue (avoids missing latents)
        if broad_score <= THRESHOLDS['min_edge_score'] and conditional_score <= THRESHOLDS['min_edge_score']:
            logger.warning(f"{category} low globally/conditionally—check if latent in other filters")
    
    logger.info(f"Scan complete: {len(edge_map)} potential edges found")
    return edge_map

def basic_asymmetry_test(returns: pd.Series, category: str) -> tuple:
    """Basic test per category—e.g., positive mean for directional."""
    if category == 'directional':
        mean_ret = returns.mean()
        t_stat, p_val = stats.ttest_1samp(returns, 0)  # Test >0
        score = mean_ret if p_val < 0.05 else 0
    elif category == 'behavioral':
        # Autocorr for trend (positive) vs reversion (negative)
        autocorr = returns.autocorr(lag=1)
        score = abs(autocorr) if abs(autocorr) > 0.1 else 0
        p_val = 0.05  # Placeholder—use proper test
    # Add for other categories (e.g., temporal: groupby day, test diffs)
    else:
        score, p_val = 0, 1  # Placeholder—expand per category
    
    return score, p_val

def conditional_subset_test(df: pd.DataFrame, category: str) -> float:
    """Subset tests for latent edges—e.g., reversion in high-vol."""
    low_vol_df = df[df['vol'] < df['vol'].quantile(0.3)]
    cond_score, _ = basic_asymmetry_test(low_vol_df['returns'], category)
    return cond_score

def test_scopes(df: pd.DataFrame, category: str) -> dict:
    """Test holds per your scopes—simulate returns for each range."""
    scope_results = {}
    for scope, desc in SUB_CLASSIFIERS['scopes'].items():
        if '1 day' in desc:
            hold_ret = df['returns'].shift(-1)
            score, _ = basic_asymmetry_test(hold_ret.dropna(), category)
            scope_results[scope] = score
        # Expand for other scopes (e.g., scalping: Intraday holds—need LTF data)
    return scope_results

# Example Test (run this in console to see)
if __name__ == "__main__":
    # Fake data for test
    fake_df = pd.DataFrame({'returns': np.random.normal(0.001, 0.02, 100), 'vol': np.random.normal(0.01, 0.005, 100)}, index=pd.date_range('2020-01-01', periods=100))
    edges = scan_for_edges(fake_df)
    print(edges)  # See map in terminal

