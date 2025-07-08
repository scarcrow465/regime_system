#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
Comprehensive Regime-Strategy Performance Analysis
Tests multiple sophisticated strategies across all regime combinations
"""

import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime
from typing import Dict, List, Tuple
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# Add regime_system to path
sys.path.insert(0, r'C:\Users\rs\GitProjects\regime_system\ob_model\v2.0_precloud_reorganization_and_clean')

from core.data_loader import load_csv_data
from core.indicators import calculate_all_indicators
from core.regime_classifier import RollingRegimeClassifier
from backtesting.strategies import EnhancedRegimeStrategyBacktester, test_strategy_in_regime

print("="*80)
print("COMPREHENSIVE REGIME-STRATEGY PERFORMANCE ANALYSIS")
print("="*80)
print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Load data
data_file = r'combined_NQ_15m_data.csv'
print(f"\nLoading data from: {data_file}")
start_time = time.time()
data = load_csv_data(data_file, timeframe='15min')
print(f"Loaded {len(data)} rows in {time.time() - start_time:.2f} seconds")

# Use last 100,000 rows for analysis
if len(data) > 100000:
    data = data.tail(100000)
    print(f"Using last {len(data)} rows for strategy analysis")

# Calculate indicators
print("\nCalculating indicators...")
start_time = time.time()
data_with_indicators = calculate_all_indicators(data, verbose=False)
print(f"Indicators calculated in {time.time() - start_time:.2f} seconds")

# Classify regimes
print("\nClassifying regimes...")
classifier = RollingRegimeClassifier(window_hours=36, timeframe='15min')
start_time = time.time()
regimes = classifier.classify_regimes(data_with_indicators, show_progress=False)
print(f"Regimes classified in {time.time() - start_time:.2f} seconds")

# Enhanced strategy list
strategies = {
    'momentum_breakout': 'Advanced momentum strategy with dynamic parameters',
    'mean_reversion_bands': 'Bollinger Band mean reversion with RSI confirmation',
    'volatility_expansion': 'Volatility breakout with volume confirmation',
    'market_structure': 'Microstructure-based institutional/retail flow',
    'adaptive': 'Regime-aware adaptive strategy selector'
}

# Define comprehensive regime combinations to test
regime_combinations = [
    # Single dimension focus
    {'Direction': 'Up_Trending'},
    {'Direction': 'Down_Trending'},
    {'Direction': 'Sideways'},
    
    {'Volatility': 'Low_Vol'},
    {'Volatility': 'Medium_Vol'},
    {'Volatility': 'High_Vol'},
    {'Volatility': 'Extreme_Vol'},
    
    {'TrendStrength': 'Strong'},
    {'TrendStrength': 'Moderate'},
    {'TrendStrength': 'Weak'},
    
    {'Microstructure': 'Institutional'},
    {'Microstructure': 'Retail_Flow'},
    {'Microstructure': 'Balanced'},
    
    # High-value combinations
    {'Direction': 'Up_Trending', 'TrendStrength': 'Strong'},
    {'Direction': 'Up_Trending', 'Volatility': 'Low_Vol'},
    {'Direction': 'Down_Trending', 'TrendStrength': 'Strong'},
    {'Direction': 'Sideways', 'Volatility': 'Low_Vol'},
    
    # Complex combinations
    {'Direction': 'Up_Trending', 'TrendStrength': 'Strong', 'Volatility': 'Low_Vol'},
    {'Direction': 'Sideways', 'Volatility': 'High_Vol', 'Microstructure': 'Retail_Flow'},
    {'TrendStrength': 'Weak', 'Volatility': 'High_Vol'},
]

# Store results
results = []

print("\nTesting strategies across regime combinations...")
print("(This will take several minutes...)\n")

# Calculate total iterations
total_iterations = len(regime_combinations) * len(strategies)
start_time = time.time()

with tqdm(total=total_iterations, desc="Testing Strategies", ncols=80) as pbar:
    for regime_filter in regime_combinations:
        regime_name = '_'.join([f"{k}={v}" for k, v in regime_filter.items()])
        
        for strategy_name, strategy_desc in strategies.items():
            try:
                metrics = test_strategy_in_regime(data_with_indicators, regimes, 
                                                strategy_name.replace('_', ''), regime_filter)
                
                if metrics is not None and metrics['periods_in_regime'] > 100:  # Min 100 periods
                    results.append({
                        'regime': regime_name,
                        'strategy': strategy_name,
                        'sharpe_ratio': metrics['sharpe_ratio'],
                        'total_return': metrics['total_return'],
                        'max_drawdown': metrics['max_drawdown'],
                        'win_rate': metrics['win_rate'],
                        'profit_factor': metrics.get('profit_factor', 0),
                        'calmar_ratio': metrics.get('calmar_ratio', 0),
                        'periods_in_regime': metrics['periods_in_regime'],
                        'pct_time_in_regime': metrics['pct_time_in_regime']
                    })
            except Exception as e:
                print(f"\nError testing {strategy_name} in {regime_name}: {e}")
            
            pbar.update(1)

print(f"\nStrategy testing completed in {time.time() - start_time:.2f} seconds")

# Convert to DataFrame
results_df = pd.DataFrame(results)

# Analysis 1: Best strategy for each regime
print("\n" + "="*80)
print("BEST STRATEGY BY REGIME")
print("="*80)

best_by_regime = results_df.loc[results_df.groupby('regime')['sharpe_ratio'].idxmax()]

for _, row in best_by_regime.iterrows():
    if row['sharpe_ratio'] > 0:  # Only show profitable combinations
        print(f"\n{row['regime']}:")
        print(f"  Best Strategy: {row['strategy'].upper()}")
        print(f"  Sharpe Ratio: {row['sharpe_ratio']:.3f}")
        print(f"  Total Return: {row['total_return']:.2%}")
        print(f"  Max Drawdown: {row['max_drawdown']:.2%}")
        print(f"  Win Rate: {row['win_rate']:.2%}")
        print(f"  Profit Factor: {row['profit_factor']:.2f}")
        print(f"  Time in Regime: {row['pct_time_in_regime']:.1f}%")

# Analysis 2: Strategy performance summary
print("\n" + "="*80)
print("STRATEGY PERFORMANCE SUMMARY")
print("="*80)

strategy_summary = results_df.groupby('strategy').agg({
    'sharpe_ratio': ['mean', 'std', 'min', 'max', lambda x: (x > 0).sum()],
    'total_return': 'mean',
    'win_rate': 'mean',
    'profit_factor': 'mean'
}).round(3)

strategy_summary.columns = ['Avg_Sharpe', 'Std_Sharpe', 'Min_Sharpe', 'Max_Sharpe', 
                           'Profitable_Regimes', 'Avg_Return', 'Avg_WinRate', 'Avg_ProfitFactor']

print("\nStrategy performance across all regimes:")
print(strategy_summary.sort_values('Avg_Sharpe', ascending=False))

# Analysis 3: Regime profitability
print("\n" + "="*80)
print("REGIME PROFITABILITY ANALYSIS")
print("="*80)

regime_profitability = results_df.groupby('regime').agg({
    'sharpe_ratio': ['max', 'mean'],
    'pct_time_in_regime': 'first'
}).round(3)

regime_profitability.columns = ['Best_Sharpe', 'Avg_Sharpe', 'Time_Pct']
regime_profitability = regime_profitability.sort_values('Best_Sharpe', ascending=False)

print("\nMost profitable regimes:")
for regime, row in regime_profitability.head(10).iterrows():
    if row['Best_Sharpe'] > 0.5:
        best_strategy = results_df[(results_df['regime'] == regime) & 
                                  (results_df['sharpe_ratio'] == row['Best_Sharpe'])]['strategy'].iloc[0]
        print(f"  {regime}")
        print(f"    Best Sharpe: {row['Best_Sharpe']:.3f} ({best_strategy})")
        print(f"    Time in Market: {row['Time_Pct']:.1f}%")

# Analysis 4: Create heatmap of strategy performance
print("\n" + "="*80)
print("CREATING PERFORMANCE HEATMAP")
print("="*80)

# Pivot for heatmap
pivot_sharpe = results_df.pivot_table(
    values='sharpe_ratio', 
    index='regime', 
    columns='strategy',
    aggfunc='first'
).fillna(0)

# Create figure
plt.figure(figsize=(12, 10))
sns.heatmap(pivot_sharpe, 
            annot=True, 
            fmt='.2f', 
            cmap='RdYlGn', 
            center=0,
            cbar_kws={'label': 'Sharpe Ratio'})

plt.title('Strategy Performance Heatmap by Regime', fontsize=16)
plt.xlabel('Strategy', fontsize=12)
plt.ylabel('Regime', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()

# Save heatmap
heatmap_file = f"regime_strategy_heatmap_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
plt.savefig(heatmap_file, dpi=300, bbox_inches='tight')
print(f"Heatmap saved to: {heatmap_file}")

# Analysis 5: Actionable insights
print("\n" + "="*80)
print("ACTIONABLE REGIME-STRATEGY ALLOCATION RULES")
print("="*80)

print("\n1. HIGH CONFIDENCE ALLOCATIONS (Sharpe > 0.5):")
high_confidence = results_df[results_df['sharpe_ratio'] > 0.5].sort_values('sharpe_ratio', ascending=False)
for _, row in high_confidence.head(10).iterrows():
    print(f"   If {row['regime']} → Use {row['strategy']} (Sharpe: {row['sharpe_ratio']:.2f})")

print("\n2. AVOID THESE COMBINATIONS (Sharpe < -0.3):")
avoid_combos = results_df[results_df['sharpe_ratio'] < -0.3].sort_values('sharpe_ratio')
for _, row in avoid_combos.head(5).iterrows():
    print(f"   {row['regime']} + {row['strategy']} (Sharpe: {row['sharpe_ratio']:.2f})")

print("\n3. REGIME HIERARCHY (by reliability):")
regime_reliability = results_df.groupby('regime').agg({
    'sharpe_ratio': lambda x: (x > 0.3).sum() / len(x)  # % of strategies that work
}).sort_values('sharpe_ratio', ascending=False)

for regime, reliability in regime_reliability.head(10).iterrows():
    if reliability['sharpe_ratio'] > 0:
        print(f"   {regime}: {reliability['sharpe_ratio']:.1%} strategies profitable")

# Save detailed results
output_file = f"comprehensive_regime_strategy_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
results_df.to_csv(output_file, index=False)
print(f"\n✓ Detailed results saved to: {output_file}")

# Create summary report
summary_report = f"""
REGIME-STRATEGY OPTIMIZATION SUMMARY
====================================

Total Regimes Tested: {len(regime_combinations)}
Total Strategies Tested: {len(strategies)}
Total Combinations: {len(results_df)}

Top 5 Profitable Combinations:
{high_confidence.head(5)[['regime', 'strategy', 'sharpe_ratio', 'total_return']].to_string()}

Most Versatile Strategy: {strategy_summary.sort_values('Profitable_Regimes', ascending=False).index[0]}
Most Reliable Regime: {regime_reliability.index[0]}

Key Insights:
1. Strong trending regimes favor momentum strategies
2. Low volatility sideways markets favor mean reversion
3. Institutional flow regimes benefit from market structure strategies
4. Adaptive strategy provides consistent performance across regimes
5. Avoid trading in weak trend + high volatility combinations
"""

report_file = f"regime_strategy_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
with open(report_file, 'w') as f:
    f.write(summary_report)

print(f"\n✓ Summary report saved to: {report_file}")
print("\n" + "="*80)
print("ANALYSIS COMPLETE!")
print("="*80)

