#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
Fingerprint Evolver for Edge Tracking
Tracks changes over time (e.g., rolling scores, intensity slopes, breaks like post-1983).
Why: Catches evolutions for latent edges (e.g., RSI2 activated in expansion)—proves OB edge is persistent behavior.
How it ties to vision: Monitors "why" fading (e.g., score <0.5 = reduce sizing), scaling to multi-asset.
Use: Input tagged_map + historical df, output evolved_map with trends.
"""

import pandas as pd
import numpy as np
from scipy import stats  # For ttest_ind on means
from utils.logger import get_logger, log_execution_time, log_errors
from utils.debug_utils import check_data_sanity, log_var_state, safe_save  # For safe PNG save
from config.edge_taxonomy import THRESHOLDS
import os  # For dir check
import matplotlib
matplotlib.use('agg')  # Non-interactive—no popup
import matplotlib.pyplot as plt  # For plots

logger = get_logger('fingerprint_evolver')

@log_execution_time(logger)
@log_errors(logger)
def evolve_edges(tagged_map: dict, df: pd.DataFrame, window_size: int = 252) -> dict:
    """
    Evolve tagged edges—compute rolling scores, intensity (slopes), persistence, breaks.
    - Input: tagged_map from classifier, df with dates/returns.
    - Output: evolved_map with added 'evolution' dict (e.g., 'trend_slope': 0.02, 'break_detected': '2020-06-15').
    - Why visual: Slopes paint "growth curves" for plots (line chart saved with time, rising like a hill).
    """
    df = check_data_sanity(df, logger, 'fingerprint_evolver')
    df['date'] = pd.to_datetime(df.index)  # Assume index is date
    evolved_map = tagged_map.copy()
    
    for category, data in evolved_map.items():
        logger.info(f"Evolving {category}")
        
        # Rolling score: Mean over windows (placeholder—real: Recalc basic_test on window)
        rolling_scores = []
        for start in range(0, len(df) - window_size, window_size // 2):
            window_df = df.iloc[start:start+window_size]
            window_score = data['final_score'] * (1 + np.random.normal(0, 0.1))
            rolling_scores.append(window_score)
        data['evolution'] = {'rolling_avg': np.mean(rolling_scores) if rolling_scores else 0}
        log_var_state('rolling_scores', rolling_scores, logger)
        logger.info(f"{category} rolling scores summary: Avg {data['evolution']['rolling_avg']:.2f}")
        
        # Intensity: Slope on scores (strengthening/fading)
        if len(rolling_scores) > 1:
            x = range(len(rolling_scores))
            slope = np.polyfit(x, rolling_scores, 1)[0]
            data['evolution']['intensity_slope'] = slope
            if slope < -THRESHOLDS['min_edge_score']:
                logger.warning(f"{category} fading—check for breaks")
        
        # Persistence: Avg duration (placeholder—use survival for "half-life" days > threshold)
        data['evolution']['persistence_days'] = len(df) / 10
        
        # Break Detection: ttest_ind for mean difference pre/post mid
        mid = len(df) // 2
        if mid > 0:
            pre, post = df['returns'][:mid], df['returns'][mid:]
            t_stat, p_val = stats.ttest_ind(pre, post, equal_var=False)
            if p_val < THRESHOLDS['evolution_significance']:
                break_date = df['date'].iloc[mid]
                data['evolution']['break_detected'] = str(break_date.date())
                logger.info(f"Break in {category} at {break_date}—evolution shift like RSI2 post-1983")
        
        # Visual Plot: Save line chart using safe_save (creates folder, adds time, no popup)
        if rolling_scores:
            plt.plot(rolling_scores)
            plt.title(f"{category} Edge Evolution—rising line = strengthening like a climbing hill")
            safe_save(plt.gcf(), f"docs/plots/{category}_evolution")  # Creates docs/plots if missing, adds _yyyy-mm-dd_hh-mm.png
            plt.close()

    return evolved_map

# Example Test (run this in console to see)
if __name__ == "__main__":
    fake_df = pd.DataFrame({'returns': np.random.normal(0.001, 0.02, 100)}, index=pd.date_range('2020-01-01', periods=100))
    fake_tagged = {'behavioral': {'final_score': 0.45}}
    evolved = evolve_edges(fake_tagged, fake_df)
    print(evolved)

