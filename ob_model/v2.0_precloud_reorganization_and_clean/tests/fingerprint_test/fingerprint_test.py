#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
Test Script for Fingerprint Core Chain
Runs scan → classify → evolve on selected timeframe data—simple check to see the "treasure hunter" work.
Why: Lets you visually see edges detected (printed map like a treasure list), with logs painting progress (bars in terminal, details in files).
Ties to vision: Shows how fingerprint extracts "why" OB wins (e.g., trending edge across scopes from scalping to position).
"""

import pandas as pd
import numpy as np
from core.edge_scanner import scan_for_edges
from core.fingerprint_classifier import classify_edges
from core.fingerprint_evolver import evolve_edges
from utils.logger import get_logger
from config.settings import PLOT_ENABLED, VERBOSE  # Toggle for detail
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

console = Console()  # Rich for pretty output
logger = get_logger('fingerprint_test')

# Timeframe variable - change this to 'daily', '1h', 'weekly', etc.
TIMEFRAME = '1h'  # Select here; run script for that timeframe

# Load data based on timeframe (assume CSVs named like combined_NQ_[timeframe]_data.csv)
data_path = r'C:\Users\rs\GitProjects\regime_system\ob_model\v2.0_precloud_reorganization_and_clean'
df = pd.read_csv(f'{data_path}\\combined_NQ_{TIMEFRAME}_data.csv')
df.index = pd.to_datetime(df.index)  # Ensure datetime index
df['returns'] = df['close'].pct_change()  # Calculate returns if missing
df['vol'] = df['returns'].rolling(window=20).std() * np.sqrt(252)  # Annualized vol (adjust window for timeframe)

console.print(Panel(f"Fingerprint Chain Test Starting on {TIMEFRAME.capitalize()} Data", style="bold green", box=box.ROUNDED))
console.print(f"VERBOSE mode: {'On' if VERBOSE else 'Off'} - Clean summaries (toggle in settings.py for full details).", style="italic")

# Run chain
edge_map = scan_for_edges(df)
tagged_map = classify_edges(edge_map, TIMEFRAME)
evolved_map = evolve_edges(tagged_map, df, plot_enabled=PLOT_ENABLED)

# Display raw edge map
console.print(f"{TIMEFRAME.capitalize()} scan complete. Here's the raw edge map:", style="green")
table = Table(show_header=True, header_style="bold magenta", box=box.SIMPLE)
table.add_column("Category")
table.add_column("Broad Score")
table.add_column("Conditional Score")
table.add_column("Scopes")
for category, data in edge_map.items():
    table.add_row(category, str(data['broad_score']), str(data['conditional_score']), str(data['scopes']))
console.print(table)

# Display tagged map
console.print(f"{TIMEFRAME.capitalize()} classification complete. Here's the tagged map:", style="green")
table = Table(show_header=True, header_style="bold magenta", box=box.SIMPLE)
table.add_column("Category")
table.add_column("Desc")
table.add_column("Final Score")
table.add_column("Best Scope")
for category, data in tagged_map.items():
    table.add_row(category, data['primary_desc'], str(data['final_score']), data['sub']['best_scope'])
console.print(table)

# Display evolved map
console.print(f"{TIMEFRAME.capitalize()} evolution complete. Here's the final evolved map:", style="green")
table = Table(show_header=True, header_style="bold magenta", box=box.SIMPLE)
table.add_column("Category")
table.add_column("Final Score")
table.add_column("Rolling Avg")
table.add_column("Intensity Slope")
table.add_column("Persistence Days")
table.add_column("Break Detected")
for category, data in evolved_map.items():
    evolution = data['evolution']
    slope = str(evolution.get('intensity_slope', 'N/A'))
    break_d = str(evolution.get('break_detected', 'N/A'))
    table.add_row(category, str(data['final_score']), str(evolution['rolling_avg']), slope, str(evolution['persistence_days']), break_d)
console.print(table)
console.print("Explanation: Evolution = trends (rolling_avg = average strength, intensity_slope = change rate, persistence_days = how long it holds). Plots in docs/plots/ show lines (rising = strengthening) and bars (tall = strong categories).", style="dim")

console.print(Panel("Test Complete\nCheck logs/fingerprint_test/[name]_[yyyy-mm-dd_hh-mm-ss].log for details. Plots in docs/plots/. Toggle VERBOSE=True for more scan info. Real data next for stronger edges!", style="bold green", box=box.ROUNDED))

