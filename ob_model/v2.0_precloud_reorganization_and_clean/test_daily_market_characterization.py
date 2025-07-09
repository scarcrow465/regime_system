#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
Test Daily NQ Market Characterization for Day Trading
Optimized for short-term strategies with higher trade frequency
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
from core.market_characterizer import MarketCharacterizer

print("="*80)
print("DAILY NQ MARKET CHARACTERIZATION FOR DAY TRADING")
print("="*80)
print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

data_file = r'combined_NQ_daily_data.csv'
print(f"\nLoading DAILY data from: {data_file}")

try:
    data = load_csv_data(data_file, timeframe='1d')
except:
    print("Trying alternative filenames...")
    for alt_file in ['NQ_daily_data.csv', 'NQ_1d_data.csv', 'combined_NQ_1d_data.csv']:
        try:
            data = load_csv_data(alt_file, timeframe='1d')
            print(f"Successfully loaded from: {alt_file}")
            break
        except:
            continue

print(f"Loaded {len(data)} daily bars")
print(f"Date range: {data.index[0]} to {data.index[-1]}")
print(f"Price range: ${data['close'].iloc[0]:.2f} to ${data['close'].iloc[-1]:.2f}")

print("\nPrice Evolution by Decade:")
for year in range(2000, 2030, 10):
    year_data = data[data.index.year >= year]
    if len(year_data) > 0:
        year_data = year_data[year_data.index.year < year + 10]
        if len(year_data) > 0:
            print(f"  {year}s: ${year_data['close'].iloc[0]:.2f} → ${year_data['close'].iloc[-1]:.2f} "
                  f"({(year_data['close'].iloc[-1]/year_data['close'].iloc[0]-1)*100:.1f}% gain)")

print("\nCalculating daily indicators...")
data_with_indicators = calculate_all_indicators(data, verbose=False)

print("\n" + "="*80)
print("TESTING MULTIPLE TIME PERIODS")
print("="*80)

periods_to_test = [
    ("Full History", None, None),
    ("Last 10 Years", -252*10, None),
    ("Last 5 Years", -252*5, None),
    ("Last 3 Years", -252*3, None),
    ("2020-2023 (COVID Era)", "2020-01-01", "2023-12-31"),
    ("2008-2009 (Financial Crisis)", "2008-01-01", "2009-12-31"),
]

all_profiles = {}

for period_name, start, end in periods_to_test:
    print(f"\n{'-'*60}")
    print(f"Testing: {period_name}")
    
    if isinstance(start, int):
        period_data = data_with_indicators.iloc[start:end]
    elif isinstance(start, str):
        period_data = data_with_indicators[start:end]
    else:
        period_data = data_with_indicators
    
    if len(period_data) < 100:
        print(f"  Skipping - insufficient data ({len(period_data)} bars)")
        continue
    
    print(f"  Analyzing {len(period_data)} daily bars")
    
    characterizer = MarketCharacterizer(
        transaction_cost=0.0005,  # 5 bps for frequent trading
        trend_period=5,           # Shorter SMA for more signals
        rsi_period=2,             # RSI(2) for higher frequency
        rsi_buy_threshold=25,
        rsi_sell_threshold=75,
        breakout_period=5,        # Shorter breakout period
        holding_period=1          # 1-day hold for day trading
    )
    profile = characterizer.characterize_market(
        period_data, 
        instrument='NQ',
        timeframe='Daily'
    )
    
    all_profiles[period_name] = profile
    
    print(f"  Primary Behavior: {profile.primary_behavior.upper()}")
    print(f"  Directional Bias: {profile.directional_bias.upper()}")
    print(f"  Long Edge: {profile.long_edge:.3f}")
    print(f"  Trend Score: {profile.trend_persistence:.3f}")
    print(f"  Mean Rev Score: {profile.mean_reversion:.3f}")
    print(f"  Breakout Score: {profile.volatility_expansion:.3f}")
    print(f"  Optimal Hold: {profile.optimal_holding_period} days")

print("\n" + "="*80)
print("MARKET REGIME EVOLUTION SUMMARY")
print("="*80)

print("\nBehavior Changes Over Time:")
print(f"{'Period':<25} {'Behavior':<15} {'Direction':<10} {'Long Sharpe':<12} {'Trend Score':<12}")
print("-"*80)
for period_name, profile in all_profiles.items():
    print(f"{period_name:<25} {profile.primary_behavior:<15} {profile.directional_bias:<10} "
          f"{profile.long_edge:<12.3f} {profile.trend_persistence:<12.3f}")

recent_profile = all_profiles.get("Last 5 Years") or all_profiles.get("Last 3 Years") or list(all_profiles.values())[0]

print("\n" + "="*80)
print("RECOMMENDATIONS BASED ON RECENT MARKET证明BEHAVIOR")
print("="*80)

print(f"\nUsing {[k for k,v in all_profiles.items() if v == recent_profile][0]} as baseline:")
print(f"\n1. Market Type: {recent_profile.primary_behavior.upper()}")
if recent_profile.primary_behavior == 'trending':
    print("   → Daily trends persist - use momentum entries")
elif recent_profile.primary_behavior == 'mean_reverting':
    print("   → Daily moves reverse - fade extremes")
else:
    print("   → Focus on breakout opportunities")

print(f"\n2. Directional Bias: {recent_profile.directional_bias.upper()}")
if recent_profile.long_edge > 0.5:
    print("   → Prefer long positions")
elif recent_profile.short_edge > 0.5:
    print("   → Favor short positions")
else:
    print("   → No strong edge, trade both sides")

print(f"\n3. Optimal Daily Holding: {recent_profile.optimal_holding_period} days")

min_sharpe = max(recent_profile.random_long_sharpe + 0.3, 0.5)
print(f"\n4. Strategy Performance Target: >{min_sharpe:.2f} Sharpe")

if len(all_profiles) > 2:
    plt.figure(figsize=(12, 6))
    
    periods = list(all_profiles.keys())
    long_edges = [p.long_edge for p in all_profiles.values()]
    trend_scores = [p.trend_persistence for p in all_profiles.values()]
    
    x = range(len(periods))
    width = 0.35
    
    plt.subplot(1, 2, 1)
    plt.bar([i - width/2 for i in x], long_edges, width, label='Long Edge', alpha=0.8)
    plt.bar([i + width/2 for i in x], trend_scores, width, label='Trend Score', alpha=0.8)
    plt.xlabel('Time Period')
    plt.ylabel('Sharpe Ratio')
    plt.title('Market Character Evolution')
    plt.xticks(x, periods, rotation=45, ha='right')
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    
    plt.subplot(1, 2, 2)
    behaviors = [p.primary_behavior for p in all_profiles.values()]
    behavior_counts = {b: behaviors.count(b) for b in set(behaviors)}
    plt.pie(behavior_counts.values(), labels=behavior_counts.keys(), autopct='%1.1f%%')
    plt.title('Market Behavior Distribution')
    
    plt.tight_layout()
    plt.savefig(f'daily_market_evolution_{datetime.now().strftime("%Y%m%d")}.png', dpi=150)
    print(f"\n✓ Evolution chart saved")

print("\n" + "="*80)
print("NEXT STEPS FOR DAY TRADING STRATEGY")
print("="*80)
print("\n1. Use short-term signals (RSI(2), 5-day SMA, 5-day breakouts)")
print("2. Target ~1,000 trades over full history")
print("3. Adjust transaction costs based on broker fees")

output_file = f"daily_market_profile_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
with open(output_file, 'w') as f:
    f.write("Daily NQ Market Characterization Results\n")
    f.write("="*50 + "\n\n")
    for period_name, profile in all_profiles.items():
        f.write(f"\n{period_name}:\n")
        f.write(f"  Behavior: {profile.primary_behavior}\n")
        f.write(f"  Direction: {profile.directional_bias}\n")
        f.write(f"  Long Edge: {profile.long_edge:.3f}\n")
        f.write(f"  Optimal Hold: {profile.optimal_holding_period} days\n")

print(f"\n✓ Detailed results saved to: {output_file}")
print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


# In[ ]:




