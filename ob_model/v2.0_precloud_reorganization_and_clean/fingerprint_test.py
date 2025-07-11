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
from utils.logger import get_logger
from config.settings import PLOT_ENABLED, VERBOSE  # Toggle for detail
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

console = Console()  # Rich for pretty output
logger = get_logger('fingerprint_test')

# Load your weekly data (replace with actual path/variable)
weekly_df = pd.read_csv(r'C:\Users\rs\GitProjects\regime_system\ob_model\v2.0_precloud_reorganization_and_clean\combined_NQ_weekly_data.csv')  # Or weekly_df = your_weekly_df_variable
weekly_df.index = pd.to_datetime(weekly_df.index)  # Ensure datetime index
weekly_df['returns'] = weekly_df['close'].pct_change()  # Calculate returns (percentage change)
weekly_df['vol'] = weekly_df['returns'].rolling(window=20).std() * np.sqrt(252)  # Calculate volatility (annualized std dev, adjust window as needed)

console.print(Panel("Fingerprint Chain Test Starting", style="bold green", box=box.ROUNDED))
console.print(f"VERBOSE mode: {'On' if VERBOSE else 'Off'} - Clean summaries (toggle in settings.py for full details).", style="italic")

# Run on weekly
console.print(Panel("Running on Weekly Data (Persistence Context)", style="bold blue", box=box.ROUNDED))
edge_map_weekly = scan_for_edges(weekly_df)
tagged_map_weekly = classify_edges(edge_map_weekly)
evolved_map_weekly = evolve_edges(tagged_map_weekly, weekly_df, plot_enabled=PLOT_ENABLED)

# Display results
console.print("Weekly scan complete. Here's the raw edge map:", style="green")
table = Table(show_header=True, header_style="bold magenta", box=box.SIMPLE)
table.add_column("Category")
table.add_column("Broad Score")
table.add_column("Conditional Score")
table.add_column("Scopes")
for category, data in edge_map_weekly.items():
    table.add_row(category, str(data['broad_score']), str(data['conditional_score']), str(data['scopes']))
console.print(table)

console.print("Weekly classification complete. Here's the tagged map:", style="green")
table = Table(show_header=True, header_style="bold magenta", box=box.SIMPLE)
table.add_column("Category")
table.add_column("Desc")
table.add_column("Final Score")
table.add_column("Best Scope")
for category, data in tagged_map_weekly.items():
    table.add_row(category, data['primary_desc'], str(data['final_score']), data['sub']['best_scope'])
console.print(table)

console.print("Weekly evolution complete. Here's the final evolved map:", style="green")
table = Table(show_header=True, header_style="bold magenta", box=box.SIMPLE)
table.add_column("Category")
table.add_column("Final Score")
table.add_column("Rolling Avg")
table.add_column("Intensity Slope")
table.add_column("Persistence Days")
table.add_column("Break Detected")
for category, data in evolved_map_weekly.items():
    evolution = data['evolution']
    slope = str(evolution.get('intensity_slope', 'N/A'))
    break_d = str(evolution.get('break_detected', 'N/A'))
    table.add_row(category, str(data['final_score']), str(evolution['rolling_avg']), slope, str(evolution['persistence_days']), break_d)
console.print(table)
console.print("Explanation: Evolution = trends (rolling_avg = average strength, intensity_slope = change rate, persistence_days = how long it holds). Plots in docs/plots/ show lines (rising = strengthening) and bars (tall = strong categories).", style="dim")

console.print(Panel("Test Complete\nCheck logs/fingerprint_test/[name]_[yyyy-mm-dd_hh-mm-ss].log for details. Plots in docs/plots/. Toggle VERBOSE=True for more scan info. Real data next for stronger edges!", style="bold green", box=box.ROUNDED))

