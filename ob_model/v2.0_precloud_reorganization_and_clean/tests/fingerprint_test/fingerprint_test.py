#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
Test Script for Pattern Checker
Checks data for trade patterns on chosen time (1h, daily, weekly)—easy test to see wins.
Why: Shows simple "win chances" like "upward pull" or "bounce after drop", how they change.
"""

import pandas as pd
import numpy as np
from core.edge_scanner import scan_for_edges
from core.fingerprint_classifier import classify_edges
from core.fingerprint_evolver import evolve_edges
from utils.logger import get_logger
from config.settings import PLOT_ENABLED, VERBOSE  # Toggle for detail
from config.edge_taxonomy import SCOPES  # Add this import for hold times table
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box
import matplotlib.pyplot as plt  # For table PNG

console = Console()  # Pretty output
logger = get_logger('pattern_test')

# Choose time: '1h', 'daily', 'weekly'
TIMEFRAME = '1h'

# Load data (change path if needed)
data_path = r'C:\Users\rs\GitProjects\regime_system\ob_model\v2.0_precloud_reorganization_and_clean'
df = pd.read_csv(f'{data_path}\\combined_NQ_{TIMEFRAME}_data.csv')
df.index = pd.to_datetime(df.index)
df['returns'] = df['close'].pct_change()  # Win % change
df['vol'] = df['returns'].rolling(20).std() * np.sqrt(252)  # Market wildness

console.print(Panel(f"Pattern Check Starting on {TIMEFRAME.upper()} Data", style="bold green", box=box.ROUNDED))
console.print(f"Detail Mode: {'On' if VERBOSE else 'Off'} - Flip in settings.py for more.", style="italic")

# Quick Guide
console.print("Quick Guide:", style="bold yellow")
console.print("- Patterns: 8 types like 'Upward Pull' (prices go up more than down).")
console.print("- Strength: 0-1 number (higher = better win chance).")
console.print("- Hold Times: How long to keep trade (short/medium—best one highlighted).")
console.print("- Changes: If getting better/worse, how long it lasts, any sudden shift.")

# Run check
edge_map = scan_for_edges(df)
tagged_map = classify_edges(edge_map, TIMEFRAME)
evolved_map = evolve_edges(tagged_map, df, plot_enabled=PLOT_ENABLED)

# Patterns Found Table
console.print(f"{TIMEFRAME.upper()} Patterns Found: (Higher Strength = Better Win Chance)", style="green")
table = Table(show_header=True, header_style="bold magenta", box=box.SIMPLE)
table.add_column("Pattern Type")
table.add_column("Overall Strength")
table.add_column("Better Conditions Strength")
table.add_column("Hold Times")
for category, data in edge_map.items():
    table.add_row(category.capitalize(), str(data['broad_strength']), str(data['conditional_strength']), str(data['scopes']))
console.print(table)
console.print("What It Means: Overall = average win chance. Better Conditions = win chance in calm markets. Hold Times = strength for short/medium trades.", style="dim")

# Export table
pd.DataFrame(edge_map).T.to_csv(f'docs/tables/{TIMEFRAME}_patterns_found.csv')
fig, ax = plt.subplots()  # PNG of table
ax.axis('off')
ax.table(cellText=[ [category.capitalize(), str(data['broad_strength']), str(data['conditional_strength']), str(data['scopes'])] for category, data in edge_map.items()], colLabels=["Pattern Type", "Overall Strength", "Better Conditions Strength", "Hold Times"], loc='center')
fig.savefig(f'docs/plots/{TIMEFRAME}_patterns_found.png')

# Best Ways Table
console.print(f"Best Ways to Use Patterns: (Strength >0.1 = Good for Trades)", style="green")
table = Table(show_header=True, header_style="bold magenta", box=box.SIMPLE)
table.add_column("Pattern Type")
table.add_column("Simple Name")
table.add_column("Strength")
table.add_column("Best Hold Time")
for category, data in tagged_map.items():
    table.add_row(category.capitalize(), data['name'], str(data['strength']), data['best_hold'])
console.print(table)
console.print("What It Means: Simple Name = easy description. Strength = win chance (higher = better). Best Hold = time to keep trade for max win.", style="dim")

# Export
pd.DataFrame(tagged_map).T.to_csv(f'docs/tables/{TIMEFRAME}_best_ways.csv')
fig, ax = plt.subplots()
ax.axis('off')
ax.table(cellText=[ [category.capitalize(), data['name'], str(data['strength']), data['best_hold']] for category, data in tagged_map.items()], colLabels=["Pattern Type", "Simple Name", "Strength", "Best Hold Time"], loc='center')
fig.savefig(f'docs/plots/{TIMEFRAME}_best_ways.png')

# Hold Times Table (New)
console.print(f"Hold Times for Patterns: (Higher = Better for That Length)", style="green")
table = Table(show_header=True, header_style="bold magenta", box=box.SIMPLE)
table.add_column("Pattern Type")
for hold in SCOPES[TIMEFRAME]:
    table.add_column(hold.capitalize() + " Strength")
for category, data in tagged_map.items():
    row = [category.capitalize()]
    for hold in SCOPES[TIMEFRAME]:
        row.append(str(data['all_holds'].get(hold, 0)))
    table.add_row(*row)
console.print(table)
console.print("What It Means: Shows win chance for short vs. medium holds—pick highest for your style.", style="dim")

# Export
pd.DataFrame([data['all_holds'] for data in tagged_map.values()], index=tagged_map.keys()).to_csv(f'docs/tables/{TIMEFRAME}_hold_times.csv')
fig, ax = plt.subplots()
ax.axis('off')
ax.table(cellText=[ [category.capitalize()] + [str(data['all_holds'].get(hold, 0)) for hold in SCOPES[TIMEFRAME]] for category, data in tagged_map.items()], colLabels=["Pattern Type"] + [hold.capitalize() + " Strength" for hold in SCOPES[TIMEFRAME]], loc='center')
fig.savefig(f'docs/plots/{TIMEFRAME}_hold_times.png')

# Changes Table
console.print(f"How Patterns Change: (Positive Trend = Getting Better)", style="green")
table = Table(show_header=True, header_style="bold magenta", box=box.SIMPLE)
table.add_column("Pattern Type")
table.add_column("Average Strength")
table.add_column("Change Trend")
table.add_column("Lasts (Days)")
table.add_column("Sudden Shift")
for category, data in evolved_map.items():
    changes = data['changes']
    table.add_row(category.capitalize(), str(changes['avg_strength']), str(changes['change_trend']), str(changes['lasts_days']), changes['change_date'])
console.print(table)
console.print("What It Means: Average Strength = typical win chance over time. Change Trend = if improving (positive) or weakening (negative). Lasts = how many days reliable. Sudden Shift = date it changed big (or None).", style="dim")

# Export
pd.DataFrame([data['changes'] for data in evolved_map.values()], index=evolved_map.keys()).to_csv(f'docs/tables/{TIMEFRAME}_changes.csv')
fig, ax = plt.subplots()
ax.axis('off')
ax.table(cellText=[ [category.capitalize(), str(changes['avg_strength']), str(changes['change_trend']), str(changes['lasts_days']), changes['change_date']] for category, data in evolved_map.items() for changes in [data['changes']]], colLabels=["Pattern Type", "Average Strength", "Change Trend", "Lasts (Days)", "Sudden Shift"], loc='center')
fig.savefig(f'docs/plots/{TIMEFRAME}_changes.png')

console.print(Panel("Check Complete—See tables/plots in docs/ for saves. Flip VERBOSE for details. Next: Add patterns like 'Bounce After Drop' for higher strengths!", style="bold green", box=box.ROUNDED))

