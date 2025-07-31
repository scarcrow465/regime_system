#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import sys
import os
BASE_DIR = r"C:\Users\rs\GitProjects\regime_system\ob_model\v3.0"  # Hardcode if not importing settings yet
sys.path.append(BASE_DIR)

# Load the CSV data
df = pd.read_csv('exports\csv\20250722_140152_enhanced_strategy_probe_results.csv')

print("=== REGIME DISTRIBUTION ANALYSIS ===")
print(f"\nTotal regime-strategy combinations: {len(df)}")
print(f"\nUnique regimes found: {sorted(df['regime'].unique())}")
print(f"Regime counts:\n{df['regime'].value_counts().sort_index()}")

print("\n=== TRADES BY REGIME ===")
regime_trades = df.groupby('regime')['total_trades'].sum()
print(regime_trades)
print(f"\nTotal trades across all regimes: {regime_trades.sum()}")

print("\n=== PROFITABLE STRATEGIES BY REGIME ===")
profitable = df[df['total_pnl'] > 0]
profit_by_regime = profitable.groupby('regime').size()
print(f"Strategies with positive PnL by regime:\n{profit_by_regime}")

print("\n=== SESSION DISTRIBUTION BY REGIME ===")
session_dist = pd.crosstab(df['regime'], df['session'])
print(session_dist)

print("\n=== STRATEGY PERFORMANCE SUMMARY ===")
strategy_summary = df.groupby('strategy').agg({
    'total_trades': 'sum',
    'total_pnl': 'sum',
    'win_rate': 'mean',
    'sharpe_ratio': 'mean'
}).round(2)
print(strategy_summary)

print("\n=== TOP 10 PERFORMING COMBINATIONS ===")
top_10 = df.nlargest(10, 'sharpe_ratio')[['regime', 'session', 'strategy', 'total_trades', 'win_rate', 'sharpe_ratio', 'total_pnl']]
print(top_10)

print("\n=== REGIME CHARACTERISTICS ===")
regime_chars = df.groupby('regime').agg({
    'avg_trend': 'mean',
    'avg_volatility': 'mean'
}).round(4)
print(regime_chars)

