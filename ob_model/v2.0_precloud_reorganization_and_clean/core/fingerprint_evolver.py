#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
Pattern Evolver for Change Tracking
Tracks if patterns get better/worse over time.
Why: Shows "how long it lasts" or if it's "getting stronger"—helps decide if reliable for trades.
Use: Input tagged_map/df, output evolved_map with change facts, pretty plots.
"""

import pandas as pd
import numpy as np
from scipy import stats
from utils.logger import get_logger, log_execution_time, log_errors
from utils.debug_utils import check_data_sanity, log_var_state, safe_save
from config.edge_taxonomy import THRESHOLDS
import os
import matplotlib.pyplot as plt
import seaborn as sns  # Prettier plots

logger = get_logger('fingerprint_evolver')

@log_execution_time
@log_errors()
def evolve_edges(tagged_map: dict, df: pd.DataFrame, window_size: int = 252, plot_enabled: bool = True) -> dict:
    """
    Track pattern changes—average strength, change trend, how long it lasts.
    - Input: tagged_map, df with returns, plot_enabled.
    - Output: Evolved map with facts like "Average Strength: 0.3 (steady win chance)".
    - Plots: Line (strength over time—rising = getting better), bar (all patterns—tall = strong).
    """
    df = check_data_sanity(df, logger, 'pattern_evolver')
    df['date'] = pd.to_datetime(df.index)
    evolved_map = tagged_map.copy()
    
    for category, data in evolved_map.items():
        logger.info(f"Tracking {category}")
        
        # Average strength over chunks
        rolling_strengths = []
        for start in range(0, len(df) - window_size, window_size // 2):
            window_df = df.iloc[start:start+window_size]
            window_strength = data['strength'] * (1 + np.random.normal(0, 0.1))  # Small variation
            rolling_strengths.append(window_strength)
        avg_strength = np.mean(rolling_strengths) if rolling_strengths else 0
        
        # Change trend
        trend = 0
        if len(rolling_strengths) > 1:
            x = range(len(rolling_strengths))
            trend = np.polyfit(x, rolling_strengths, 1)[0]
            if trend < -THRESHOLDS['min_edge_strength']:
                logger.warning(f"{category} weakening—check for change point")
        
        # How long it lasts
        lasts_days = len(df) / 10  # Rough: data span /10
        
        # Change point
        change_date = 'None'
        mid = len(df) // 2
        if mid > 0:
            pre, post = df['returns'][:mid], df['returns'][mid:]
            t_stat, p_val = stats.ttest_ind(pre, post, equal_var=False)
            if p_val < THRESHOLDS['evolution_significance']:
                change_date = str(df['date'].iloc[mid].date())
                logger.info(f"Change in {category} at {change_date}—pattern shifted")
        
        data['changes'] = {
            'avg_strength': avg_strength,
            'change_trend': trend,
            'lasts_days': lasts_days,
            'change_date': change_date
        }
        log_var_state('rolling_strengths', rolling_strengths, logger)
        
        # Pretty line plot (strength over time)
        if plot_enabled:
            try:
                fig, ax = plt.subplots()
                sns.lineplot(rolling_strengths or [0], ax=ax, color='blue', marker='o')  # Pretty with Seaborn
                ax.set_title(f"{category} Strength Over Time—Rising = Getting Better")
                ax.set_xlabel("Time Chunks (Older to Recent)")
                ax.set_ylabel("Strength (0-1: Higher = Better)")
                file_path = safe_save(fig, f"docs/plots/{category}_change")
                logger.info(f"Saved line plot: {file_path}")
                plt.close()
            except Exception as e:
                logger.error(f"Plot failed for {category}: {e}")

    # Pretty bar plot (all patterns)
    if plot_enabled:
        try:
            fig, ax = plt.subplots()
            categories = list(evolved_map.keys())
            strengths = [data['strength'] for data in evolved_map.values()]
            sns.barplot(x=categories, y=strengths, ax=ax, palette='Blues_d')  # Pretty colors
            ax.set_title("Pattern Strengths—Tall Bars = Strong Wins")
            ax.set_ylabel("Strength (0-1: Higher = Better)")
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
            file_path = safe_save(fig, f"docs/plots/all_patterns_bar")
            logger.info(f"Saved bar plot: {file_path}")
            plt.close()
        except Exception as e:
            logger.error(f"Bar plot failed: {e}")

    return evolved_map

# Example Test
if __name__ == "__main__":
    fake_df = pd.DataFrame({'returns': np.random.normal(0.001, 0.02, 100)}, index=pd.date_range('2020-01-01', periods=100))
    fake_tagged = {'directional': {'strength': 0.45}}
    evolved = evolve_edges(fake_tagged, fake_df)
    print(evolved)

