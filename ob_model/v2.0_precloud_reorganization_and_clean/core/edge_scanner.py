#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
Edge Scanner for Asymmetry Detection
Scans data for edges across categories (e.g., directional bias, behavioral trends).
Why: Finds "treasure" patterns like positive drift or bounces after drops—base for "why" OB wins.
Use: Input df with returns/vol, output edge_map with scores (higher = stronger pattern).
"""

import pandas as pd
import numpy as np
from scipy import stats
from utils.logger import get_logger, log_execution_time, log_errors
from utils.debug_utils import check_data_sanity, log_var_state
from config.edge_taxonomy import PRIMARY_CATEGORIES, THRESHOLDS

logger = get_logger('edge_scanner')

@log_execution_time
@log_errors()
def scan_for_edges(df: pd.DataFrame) -> dict:
    """
    Scan for patterns across 8 types—simple checks for wins above average.
    - Input: df with 'returns' (daily % change), 'vol' (market wildness).
    - Output: Map of patterns with strength scores (0-1: higher = better win chance) and hold times.
    """
    df = check_data_sanity(df, logger, 'edge_scanner')
    edge_map = {}

    def basic_pattern_test(returns: pd.Series, category: str) -> tuple:
        """Simple check: How much better than average? (strength, significance)"""
        mean_win = returns.mean()
        strength = abs(mean_win)  # 0-1 scale (higher = stronger pattern)
        significance = stats.ttest_1samp(returns.dropna(), 0)[1]  # Low = real, not luck
        return strength, significance

    for category in PRIMARY_CATEGORIES:
        logger.info(f"Scanning {category}: {PRIMARY_CATEGORIES[category]}")
        
        # Basic strength (global)
        broad_strength, broad_signif = basic_pattern_test(df['returns'], category)

        # Better in certain conditions (e.g., calm markets)
        low_vol_returns = df['returns'][df['vol'] < df['vol'].quantile(0.3)]
        conditional_strength, conditional_signif = basic_pattern_test(low_vol_returns, category)

        # Hold times (simple: same strength, adjusted for short/medium)
        scopes = {'day_trading': broad_strength, 'short_term': broad_strength * 0.5}  # Shorter = full, longer = half (fades)

        edge_map[category] = {
            'broad_strength': broad_strength,
            'conditional_strength': conditional_strength,
            'scopes': scopes
        }
        log_var_state('broad_results', {'strength': broad_strength, 'signif': broad_signif}, logger)
        if broad_strength < THRESHOLDS['min_edge_score']:
            logger.warning(f"{category} low overall/better conditions—check if hidden in other types")

    logger.info(f"Scan complete: {len(edge_map)} potential patterns found")
    return edge_map

