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
from config.settings import PLOT_ENABLED

logger = get_logger('fingerprint_test')

# Create fake data
fake_data = {
    'returns': np.random.normal(0.001, 0.02, 100),
    'vol': np.random.normal(0.01, 0.005, 100)
}
fake_df = pd.DataFrame(fake_data, index=pd.date_range('2024-01-01', periods=100))

logger.info("Starting fingerprint chain test on fake data...")

# Step 1: Run scan
logger.info("=== Step 1: Scan - Searching for asymmetries (edges) across categories ===")
edge_map = scan_for_edges(fake_df)
logger.info("Scan complete. Here's the raw edge map (potential asymmetries found):")
# Pretty table for edge map
edge_table = pd.DataFrame.from_dict(edge_map, orient='index')[['broad_score', 'conditional_score', 'scopes']]
print(edge_table.to_string())

# Step 2: Run classify
logger.info("\n=== Step 2: Classify - Tagging edges with taxonomy (e.g., scopes like day trading) ===")
tagged_map = classify_edges(edge_map)
logger.info("Classification complete. Here's the tagged map:")
# Pretty table for tagged map
tagged_table = pd.DataFrame.from_dict(tagged_map, orient='index')[['primary_desc', 'final_score', 'sub', 'analytical']]
print(tagged_table.to_string())

# Step 3: Run evolve
logger.info("\n=== Step 3: Evolve - Tracking changes (e.g., strengthening slope, persistence) ===")
evolved_map = evolve_edges(tagged_map, fake_df, plot_enabled=PLOT_ENABLED)
logger.info("Evolution complete. Here's the final evolved map:")
# Pretty table for evolved map
evolved_table = pd.DataFrame.from_dict(evolved_map, orient='index')[['final_score', 'evolution']]
print(evolved_table.to_string())

logger.info("\n=== Test Complete ===\nCheck logs/fingerprint_test/fingerprint_test_[yyyy-mm-dd_hh-mm-ss].log for details. Plots in docs/plots/[category]_evolution_[yyyy-mm-dd_hh-mm-ss].png. If scores >0.1, good—real data will show stronger.")

