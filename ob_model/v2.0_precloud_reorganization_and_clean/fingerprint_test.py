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



logger = get_logger('fingerprint_test')

# Create fake data (like a mini market—100 days with slight upward bias)
fake_data = {
    'returns': np.random.normal(0.001, 0.02, 100),
    'vol': np.random.normal(0.01, 0.005, 100)
}
fake_df = pd.DataFrame(fake_data, index=pd.date_range('2024-01-01', periods=100))

logger.info("Starting fingerprint chain test on fake data...")

# Step 1: Run scan (broad search for edges)
edge_map = scan_for_edges(fake_df)
logger.info("Scan complete—here's the raw edge map (potential asymmetries found):")
print(edge_map)

# Step 2: Run classify (tag with taxonomy, scopes)
tagged_map = classify_edges(edge_map)
logger.info("Classification complete—here's the tagged map (with scopes like day trading):")
print(tagged_map)

# Step 3: Run evolve (track changes, intensity)
evolved_map = evolve_edges(tagged_map, fake_df)
logger.info("Evolution complete—here's the final evolved map (with trends like strengthening slope):")
print(evolved_map)

logger.info("Test complete! Check logs/fingerprint_test/fingerprint_test_[yyyy-mm-dd_hh-mm-ss].log for details (e.g., p-values). If edges look good (scores >0.1), we're set.")

