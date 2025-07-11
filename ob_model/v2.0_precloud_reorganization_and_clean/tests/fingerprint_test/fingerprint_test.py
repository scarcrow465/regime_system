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

# Load daily fake (or your real daily NQ)
fake_data = {
    'returns': np.random.normal(0.001, 0.02, 100),
    'vol': np.random.normal(0.01, 0.005, 100)
}
daily_df = pd.DataFrame(fake_data, index=pd.date_range('2024-01-01', periods=100))

# Load your weekly data (replace with actual path/variable)
weekly_df = pd.read_csv('path/to/weekly_nq.csv')  # Or weekly_df = your_weekly_df_variable
weekly_df.index = pd.to_datetime(weekly_df.index)  # Ensure datetime index

console.print(Panel("Fingerprint Chain Test Starting", style="bold green", box=box.ROUNDED))
console.print(f"VERBOSE mode: {'On' if VERBOSE else 'Off'} - Clean summaries (toggle in settings.py for full details).", style="italic")

# Run on daily
console.print(Panel("Running on Daily Data", style="bold blue", box=box.ROUNDED))
edge_map_daily = scan_for_edges(daily_df)
tagged_map_daily = classify_edges(edge_map_daily)
evolved_map_daily = evolve_edges(tagged_map_daily, daily_df, plot_enabled=PLOT_ENABLED)

# Run on weekly
console.print(Panel("Running on Weekly Data (Persistence Context)", style="bold blue", box=box.ROUNDED))
edge_map_weekly = scan_for_edges(weekly_df)
tagged_map_weekly = classify_edges(edge_map_weekly)
evolved_map_weekly = evolve_edges(tagged_map_weekly, weekly_df, plot_enabled=PLOT_ENABLED)

# Comparison Table
console.print("Comparison: Daily vs. Weekly (High weekly = edge persists, like RSI2 glue)", style="green")
table = Table(show_header=True, header_style="bold magenta", box=box.SIMPLE)
table.add_column("Category")
table.add_column("Daily Score")
table.add_column("Weekly Score")
table.add_column("Weekly Persistence (Days)")
for category in evolved_map_daily:
    d_score = evolved_map_daily[category]['final_score']
    w_score = evolved_map_weekly.get(category, {'final_score': 'N/A'})['final_score']
    persistence = evolved_map_weekly.get(category, {'evolution': {'persistence_days': 'N/A'}})['evolution']['persistence_days']
    table.add_row(category, str(d_score), str(w_score), str(persistence))
console.print(table)
console.print("Explanation: Weekly checks long-hold (position scope). If weekly score close to daily, edge strong over weeks—OB 'why' for sizing up.", style="dim")

console.print(Panel("Test Complete\nCheck logs/fingerprint_test/[name]_[yyyy-mm-dd_hh-mm-ss].log for details. Plots in docs/plots/date/time. Real NQ weekly edges show persistence!", style="bold green", box=box.ROUNDED))

