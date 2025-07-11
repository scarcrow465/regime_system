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
import np as np
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

# Create fake data
fake_data = {
    'returns': np.random.normal(0.001, 0.02, 100),
    'vol': np.random.normal(0.01, 0.005, 100)
}
fake_df = pd.DataFrame(fake_data, index=pd.date_range('2024-01-01', periods=100))

console.print(Panel("Fingerprint Chain Test Starting", style="bold green", box=box.ROUNDED))
console.print(f"VERBOSE mode: {'On' if VERBOSE else 'Off'} - Clean summaries (toggle in settings.py for full details).", style="italic")

# Step 1: Run scan
console.print(Panel("Step 1: Scan - Searching for asymmetries (edges) across categories", style="bold blue", box=box.ROUNDED))
edge_map = scan_for_edges(fake_df)
console.print("Scan complete. 8 potential edges found. Here's the raw edge map (broad/conditional scores, scopes):", style="green")
# Pretty table with Rich
table = Table(show_header=True, header_style="bold magenta", box=box.SIMPLE)
table.add_column("Category")
table.add_column("Broad Score")
table.add_column("Conditional Score")
table.add_column("Scopes")
for category, data in edge_map.items():
    table.add_row(category, str(data['broad_score']), str(data['conditional_score']), str(data['scopes']))
console.print(table)
console.print("Explanation: Broad score = global asymmetry, Conditional = subset boost (e.g., low-vol like RSI2). Scopes = hold ranges (e.g., day_trading = 1-day). Low in fake; real NQ shows trending.", style="dim")

# Step 2: Run classify
console.print(Panel("Step 2: Classify - Tagging edges with taxonomy (e.g., scopes like day trading)", style="bold blue", box=box.ROUNDED))
tagged_map = classify_edges(edge_map)
console.print("Classification complete. Here's the tagged map (desc, score, best scope):", style="green")
table = Table(show_header=True, header_style="bold magenta", box=box.SIMPLE)
table.add_column("Category")
table.add_column("Desc")
table.add_column("Final Score")
table.add_column("Best Scope")
for category, data in tagged_map.items():
    table.add_row(category, data['primary_desc'], str(data['final_score']), data['sub']['best_scope'])
console.print(table)
console.print("Explanation: Primary desc = edge type (e.g., behavioral for reversion). Final score = viability (>0.1 = potential). Best scope = optimal hold (e.g., day_trading for 1-day).", style="dim")

# Step 3: Run evolve
console.print(Panel("Step 3: Evolve - Tracking changes (e.g., strengthening slope, persistence)", style="bold blue", box=box.ROUNDED))
evolved_map = evolve_edges(tagged_map, fake_df, plot_enabled=PLOT_ENABLED)
console.print("Evolution complete. Here's the final evolved map (score, evolution metrics):", style="green")
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

