#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
Test different holding periods to find optimal trading scope
Works with your existing 15-min and daily data
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

# Add paths
sys.path.insert(0, r'C:\Users\rs\GitProjects\regime_system\ob_model\v2.0_precloud_reorganization_and_clean')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.data_loader import load_csv_data
from core.indicators import calculate_all_indicators

print("="*80)
print("HOLDING PERIOD ANALYSIS")
print("="*80)
print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Load 15-min data
print("\nLoading 15-minute data...")
data_15m = load_csv_data(r'combined_NQ_15m_data.csv', timeframe='15min')
data_15m = data_15m.tail(100000)  # Last 100k bars
print(f"Loaded {len(data_15m)} 15-min bars")

# Test different holding periods on 15-min data
holding_periods_15m = {
    '1 hour': 4,           # 4 bars
    '2 hours': 8,          # 8 bars  
    '4 hours': 16,         # 16 bars
    '1 day': 26,           # 26 bars (6.5 hours)
    '2 days': 52,          # 52 bars
    '3 days': 78,          # 78 bars
    '5 days': 130,         # 130 bars
    '10 days': 260,        # 260 bars
}

print("\n" + "-"*60)
print("Testing holding periods on 15-MIN data:")
print("-"*60)

results_15m = {}
transaction_cost = 0.0001  # 1 bp for 15-min

for period_name, bars in holding_periods_15m.items():
    # Simple momentum strategy: buy if close > close[bars] ago
    returns = data_15m['close'].pct_change(bars).shift(-bars)  # Future returns
    signal = (data_15m['close'] > data_15m['close'].shift(bars)).astype(int)
    
    # Calculate strategy returns
    strategy_returns = signal * returns
    
    # Subtract transaction costs (entry + exit)
    trades = signal.diff().abs()
    costs = trades * transaction_cost
    strategy_returns = strategy_returns - costs
    
    # Calculate metrics
    clean_returns = strategy_returns.dropna()
    if len(clean_returns) > 0 and clean_returns.std() > 0:
        sharpe = clean_returns.mean() / clean_returns.std() * np.sqrt(252 * 26)
        total_return = (1 + clean_returns).prod() - 1
        win_rate = (clean_returns > 0).mean()
    else:
        sharpe = -999
        total_return = -1
        win_rate = 0
    
    results_15m[period_name] = {
        'sharpe': sharpe,
        'total_return': total_return,
        'win_rate': win_rate,
        'trades_per_year': 252 * 26 / bars
    }
    
    print(f"{period_name:<10}: Sharpe={sharpe:>7.3f}, WinRate={win_rate:>6.1%}, "
          f"Trades/Year={252 * 26 / bars:>6.0f}")

# Load daily data
print("\n" + "-"*60)
print("Testing holding periods on DAILY data:")
print("-"*60)

data_daily = load_csv_data(r'combined_NQ_daily_data.csv', timeframe='1d')
data_daily = data_daily.tail(2520)  # Last 10 years
print(f"Loaded {len(data_daily)} daily bars")

holding_periods_daily = {
    '1 day': 1,
    '2 days': 2,
    '3 days': 3,
    '5 days': 5,
    '10 days': 10,
    '20 days': 20,
    '50 days': 50,
    '100 days': 100,
}

results_daily = {}
transaction_cost = 0.00005  # 0.5 bp for daily

for period_name, bars in holding_periods_daily.items():
    # Simple momentum strategy
    returns = data_daily['close'].pct_change(bars).shift(-bars)
    signal = (data_daily['close'] > data_daily['close'].shift(bars)).astype(int)
    
    # Calculate strategy returns
    strategy_returns = signal * returns
    
    # Subtract transaction costs
    trades = signal.diff().abs()
    costs = trades * transaction_cost
    strategy_returns = strategy_returns - costs
    
    # Calculate metrics
    clean_returns = strategy_returns.dropna()
    if len(clean_returns) > 0 and clean_returns.std() > 0:
        sharpe = clean_returns.mean() / clean_returns.std() * np.sqrt(252)
        total_return = (1 + clean_returns).prod() - 1
        win_rate = (clean_returns > 0).mean()
    else:
        sharpe = -999
        total_return = -1
        win_rate = 0
    
    results_daily[period_name] = {
        'sharpe': sharpe,
        'total_return': total_return,
        'win_rate': win_rate,
        'trades_per_year': 252 / bars
    }
    
    print(f"{period_name:<10}: Sharpe={sharpe:>7.3f}, WinRate={win_rate:>6.1%}, "
          f"Trades/Year={252 / bars:>6.0f}")

# Find optimal trading scope
print("\n" + "="*80)
print("OPTIMAL TRADING SCOPE ANALYSIS")
print("="*80)

# Combine results
all_results = []
for period, metrics in results_15m.items():
    all_results.append({
        'timeframe': '15-min',
        'holding_period': period,
        'sharpe': metrics['sharpe'],
        'win_rate': metrics['win_rate'],
        'trades_per_year': metrics['trades_per_year']
    })

for period, metrics in results_daily.items():
    all_results.append({
        'timeframe': 'Daily',
        'holding_period': period,
        'sharpe': metrics['sharpe'],
        'win_rate': metrics['win_rate'],
        'trades_per_year': metrics['trades_per_year']
    })

# Sort by Sharpe ratio
all_results.sort(key=lambda x: x['sharpe'], reverse=True)

print("\nTop 10 Holding Period/Timeframe Combinations:")
print(f"{'Rank':<5} {'Timeframe':<10} {'Hold Period':<12} {'Sharpe':<10} {'Win Rate':<10} {'Trades/Year':<12}")
print("-"*70)

for i, result in enumerate(all_results[:10]):
    print(f"{i+1:<5} {result['timeframe']:<10} {result['holding_period']:<12} "
          f"{result['sharpe']:<10.3f} {result['win_rate']:<10.1%} {result['trades_per_year']:<12.0f}")

# Categorize by trading style
print("\n" + "-"*60)
print("BY TRADING STYLE:")
print("-"*60)

print("\nDAY TRADING (< 1 day holds):")
day_trading = [r for r in all_results if r['holding_period'] in ['1 hour', '2 hours', '4 hours', '1 day']]
for r in sorted(day_trading, key=lambda x: x['sharpe'], reverse=True)[:3]:
    print(f"  {r['timeframe']} {r['holding_period']}: {r['sharpe']:.3f} Sharpe")

print("\nSWING TRADING (2-10 day holds):")
swing_trading = [r for r in all_results if r['holding_period'] in ['2 days', '3 days', '5 days', '10 days']]
for r in sorted(swing_trading, key=lambda x: x['sharpe'], reverse=True)[:3]:
    print(f"  {r['timeframe']} {r['holding_period']}: {r['sharpe']:.3f} Sharpe")

print("\nPOSITION TRADING (10+ day holds):")
position_trading = [r for r in all_results if r['holding_period'] in ['10 days', '20 days', '50 days', '100 days']]
for r in sorted(position_trading, key=lambda x: x['sharpe'], reverse=True)[:3]:
    print(f"  {r['timeframe']} {r['holding_period']}: {r['sharpe']:.3f} Sharpe")

# Create visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# 15-min results
periods_15m = list(results_15m.keys())
sharpes_15m = [results_15m[p]['sharpe'] for p in periods_15m]
ax1.bar(periods_15m, sharpes_15m, alpha=0.7, color='blue')
ax1.set_title('15-Minute Data: Sharpe by Holding Period')
ax1.set_xlabel('Holding Period')
ax1.set_ylabel('Sharpe Ratio')
ax1.axhline(y=0, color='red', linestyle='--', alpha=0.5)
ax1.tick_params(axis='x', rotation=45)
ax1.grid(axis='y', alpha=0.3)

# Daily results
periods_daily = list(results_daily.keys())
sharpes_daily = [results_daily[p]['sharpe'] for p in periods_daily]
ax2.bar(periods_daily, sharpes_daily, alpha=0.7, color='green')
ax2.set_title('Daily Data: Sharpe by Holding Period')
ax2.set_xlabel('Holding Period')
ax2.set_ylabel('Sharpe Ratio')
ax2.axhline(y=0, color='red', linestyle='--', alpha=0.5)
ax2.tick_params(axis='x', rotation=45)
ax2.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(f'holding_period_analysis_{datetime.now().strftime("%Y%m%d")}.png', dpi=150)
print("\n✓ Chart saved")

# Recommendations
print("\n" + "="*80)
print("RECOMMENDATIONS")
print("="*80)

best_overall = all_results[0]
print(f"\n1. OPTIMAL CONFIGURATION:")
print(f"   Timeframe: {best_overall['timeframe']}")
print(f"   Holding Period: {best_overall['holding_period']}")
print(f"   Expected Sharpe: {best_overall['sharpe']:.3f}")

print(f"\n2. REGIME SYSTEM DESIGN:")
if best_overall['holding_period'] in ['1 day', '2 days', '3 days']:
    print("   → Use DAILY regimes for direction")
    print("   → Use 15-MIN for entry timing")
    print("   → Hold positions for 1-3 days")
elif best_overall['holding_period'] in ['5 days', '10 days']:
    print("   → Use DAILY regimes as primary")
    print("   → Use 4-HOUR or DAILY for entries")
    print("   → 15-min too noisy for this scope")
else:
    print("   → Consider position trading approach")
    print("   → Weekly/Monthly regimes may be better")

print(f"\n3. TRANSACTION COST IMPACT:")
print(f"   Avoid holding periods < 4 hours on 15-min data")
print(f"   Daily data more forgiving of costs")

print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

