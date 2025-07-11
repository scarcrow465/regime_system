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
from utils.logger import get_logger, progress_wrapper  # Your logger for visuals/logs

logger = get_logger('fingerprint_test')  # Setup logger—change level in config if needed (e.g., 'DEBUG' for more details)

# Step 1: Create fake data (like a mini market—100 days of returns with slight positive bias for trending edge)
fake_data = {
    'returns': np.random.normal(0.001, 0.02, 100),  # Fake daily returns (small upward drift)
    'vol': np.random.normal(0.01, 0.005, 100)  # Fake volatility for conditional tests
}
fake_df = pd.DataFrame(fake_data, index=pd.date_range('2024-01-01', periods=100))  # Add dates for evolution

logger.info("Starting fingerprint chain test on fake data...")

# Step 2: Run scan (broad search for edges)
edge_map = scan_for_edges(fake_df)
logger.info("Scan complete—here's the raw edge map (potential asymmetries found):")
print(edge_map)  # Terminal print: See dict like {'behavioral': {'broad_score': 0.45, ...}}—visual as a list of "clues"

# Step 3: Run classify (tag with taxonomy, scopes)
tagged_map = classify_edges(edge_map)
logger.info("Classification complete—here's the tagged map (with scopes like day trading):")
print(tagged_map)  # Terminal print: Expanded dict with sub-tags—picture it as labeled boxes (e.g., 'scope': 'day_trading' for 1-day holds)

# Step 4: Run evolve (track changes, intensity)
evolved_map = evolve_edges(tagged_map, fake_df)
logger.info("Evolution complete—here's the final evolved map (with trends like strengthening slope):")
print(evolved_map)  # Terminal print: Added 'evolution' dict—visual as a "growth chart" (e.g., slope positive = edge getting stronger like a growing plant)

logger.info("Test complete! Check logs/fingerprint_test/yyyy-mm-dd_detailed.txt for deep details (e.g., p-values). If edges look good (scores >0.1), we're set.")

# Visual Suggestion: To "see" it better, add this plot (uncomment if matplotlib installed)
import matplotlib.pyplot as plt
scores = [data['final_score'] for data in evolved_map.values()]  # Fake scores list
plt.bar(evolved_map.keys(), scores)
plt.title("Edge Scores Bar Chart—tall bars = strong edges like tall trees")
plt.show()  # Opens a window with bars—redownload if needed

