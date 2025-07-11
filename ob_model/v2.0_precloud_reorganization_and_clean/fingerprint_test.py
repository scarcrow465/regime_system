#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
Test Script for Fingerprint Core Chain
Runs scan → classify → evolve on fake data—simple check to see the "treasure hunter" work.
Why: Lets you visually see edges detected (printed map like a treasure list), with logs painting progress (bars in terminal, details in files).
Ties to vision: Shows how fingerprint extracts "why" OB wins (e.g., trending edge across scopes from scalping to position).
"""

import pandas as pd
import numpy as np
from core.edge_scanner import scan_for_edges
from core.fingerprint_classifier import classify_edges
from core.fingerprint_evolver import evolve_edges
from utils.logger import get_logger, progress_wrapper
from config.settings import PLOT_ENABLED, VERBOSE  # Toggle for detail

logger = get_logger('fingerprint_test')

# Create fake data
fake_data = {
    'returns': np.random.normal(0.001, 0.02, 100),
    'vol': np.random.normal(0.01, 0.005, 100)
}
fake_df = pd.DataFrame(fake_data, index=pd.date_range('2024-01-01', periods=100))

logger.info("=== Fingerprint Chain Test Starting ===")
logger.info("VERBOSE mode: %s - Clean summaries (toggle in settings.py for full details).", "On" if VERBOSE else "Off")

# Step 1: Run scan
logger.info("=== Step 1: Scan - Searching for asymmetries (edges) across categories ===")
edge_map = scan_for_edges(fake_df)
logger.info("Scan complete. 8 potential edges found. Here's the raw edge map (broad/conditional scores, scopes):")
# Pretty table with explanation
edge_table = pd.DataFrame.from_dict(edge_map, orient='index')[['broad_score', 'conditional_score', 'scopes']]
print(edge_table.to_string())
logger.info("Explanation: Broad score = global asymmetry, Conditional = subset boost (e.g., low-vol like RSI2). Scopes = hold ranges (e.g., day_trading = 1-day). Low in fake; real NQ shows trending.")

# Step 2: Run classify
logger.info("\n=== Step 2: Classify - Tagging edges with taxonomy (e.g., scopes like day trading) ===")
tagged_map = classify_edges(edge_map)
logger.info("Classification complete. Here's the tagged map (desc, score, best scope):")
tagged_table = pd.DataFrame.from_dict(tagged_map, orient='index')[['primary_desc', 'final_score', 'sub']]
print(tagged_table.to_string())
logger.info("Explanation: Primary desc = edge type (e.g., behavioral for reversion). Final score = viability (>0.1 = potential). Sub = best scope (e.g., day_trading for 1-day holds).")

# Step 3: Run evolve
logger.info("\n=== Step 3: Evolve - Tracking changes (e.g., strengthening slope, persistence) ===")
evolved_map = evolve_edges(tagged_map, fake_df, plot_enabled=PLOT_ENABLED)
logger.info("Evolution complete. Here's the final evolved map (score, evolution metrics):")
evolved_table = pd.DataFrame.from_dict(evolved_map, orient='index')[['final_score', 'evolution']]
print(evolved_table.to_string())
logger.info("Explanation: Evolution = trends (rolling_avg = average strength, intensity_slope = change rate, persistence_days = how long it holds). Plots in docs/plots/ show lines (rising = strengthening) and bars (tall = strong categories).")

logger.info("\n=== Test Complete ===\nCheck logs/fingerprint_test/[name]_[yyyy-mm-dd_hh-mm-ss].log for details. Plots in docs/plots/. Toggle VERBOSE=True for more scan info. Real data next for stronger edges!")

