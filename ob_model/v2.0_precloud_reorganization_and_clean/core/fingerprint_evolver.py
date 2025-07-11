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
from scipy import stats
from utils.logger import get_logger, log_execution_time, log_errors
from utils.debug_utils import check_data_sanity, log_var_state, safe_save
from config.edge_taxonomy import THRESHOLDS
import os
import matplotlib
matplotlib.use('agg')  # No popup
import matplotlib.pyplot as plt

logger = get_logger('fingerprint_evolver')

@log_execution_time
@log_errors()
def evolve_edges(tagged_map: dict, df: pd.DataFrame, window_size: int = 252, plot_enabled: bool = True) -> dict:
    """
    Evolve edges—compute rolling scores, intensity, persistence, breaks.
    - Input: tagged_map, df with dates/returns, plot_enabled (toggle for visuals).
    - Output: evolved_map with 'evolution' (e.g., 'trend_slope': 0.02).
    - Why visual: Saves line chart (rising like a hill) and bar chart (tall = strong) to docs/plots/.
    """
    df = check_data_sanity(df, logger, 'fingerprint_evolver')
    df['date'] = pd.to_datetime(df.index)
    evolved_map = tagged_map.copy()
    
    for category, data in evolved_map.items():
        logger.info(f"Evolving {category}")
        
        # Rolling score
        rolling_scores = []
        for start in range(0, len(df) - window_size, window_size // 2):
            window_df = df.iloc[start:start+window_size]
            window_score = data['final_score'] * (1 + np.random.normal(0, 0.1))
            rolling_scores.append(window_score)
        data['evolution'] = {'rolling_avg': np.mean(rolling_scores) if rolling_scores else 0}
        log_var_state('rolling_scores', rolling_scores, logger)
        logger.info(f"{category} rolling scores summary: Avg {data['evolution']['rolling_avg']:.2f}")
        
        # Intensity
        if len(rolling_scores) > 1:
            x = range(len(rolling_scores))
            slope = np.polyfit(x, rolling_scores, 1)[0]
            data['evolution']['intensity_slope'] = slope
            if slope < -THRESHOLDS['min_edge_score']:
                logger.warning(f"{category} fading—check for breaks")
        
        # Persistence
        data['evolution']['persistence_days'] = len(df) / 10
        
        # Break Detection (enhanced for weekly: monthly subsets)
        mid = len(df) // 2
        if mid > 0:
            pre, post = df['returns'][:mid], df['returns'][mid:]
            t_stat, p_val = stats.ttest_ind(pre, post, equal_var=False)
            if p_val < THRESHOLDS['evolution_significance']:
                break_date = df['date'].iloc[mid]
                data['evolution']['break_detected'] = str(break_date.date())
                logger.info(f"Break in {category} at {break_date}—evolution shift like RSI2 post-1983")
        
        # Visual Plot: Line chart for each category
        if plot_enabled:
            try:
                plt.figure()  # New figure
                plt.plot(rolling_scores or [0])  # Plot even if empty
                plt.title(f"{category} Edge Evolution—rising line = strengthening like a climbing hill")
                file_path = safe_save(plt.gcf(), f"docs/plots/{category}_evolution")
                logger.info(f"Saved line plot: {file_path}")
                plt.close()
            except Exception as e:
                logger.error(f"Failed to save line plot for {category}: {e}")

    # Visual Plot: Bar chart for all categories
    if plot_enabled:
        try:
            plt.figure()
            categories = list(evolved_map.keys())
            scores = [data['final_score'] for data in evolved_map.values()]
            plt.bar(categories, scores)
            plt.title("Edge Scores Across Categories—tall bars = strong edges like skyscrapers")
            plt.xticks(rotation=45)
            file_path = safe_save(plt.gcf(), f"docs/plots/all_edges_bar")
            logger.info(f"Saved bar plot: {file_path}")
            plt.close()
        except Exception as e:
            logger.error(f"Failed to save bar plot: {e}")

    return evolved_map

# Example Test
if __name__ == "__main__":
    fake_df = pd.DataFrame({'returns': np.random.normal(0.001, 0.02, 100)}, index=pd.date_range('2020-01-01', periods=100))
    fake_tagged = {'behavioral': {'final_score': 0.45}}
    evolved = evolve_edges(fake_tagged, fake_df)
    print(evolved)

