#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
Test script for market characterization
Save this as: test_market_characterization.py
Run this FIRST to see market profile before regime analysis
"""

import sys
import os
import pandas as pd
from datetime import datetime

# Add your project path
sys.path.insert(0, r'C:\Users\rs\GitProjects\regime_system\ob_model\v2.0_precloud_reorganization_and_clean')

# Import your existing modules
from core.data_loader import load_csv_data
from core.indicators import calculate_all_indicators
from core.market_characterizer import MarketCharacterizer

print("="*80)
print("MARKET CHARACTERIZATION TEST")
print("="*80)
print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Load your data (same as in strategy_performance_mapping.py)
data_file = r'combined_NQ_15m_data.csv'
print(f"\nLoading data from: {data_file}")
data = load_csv_data(data_file, timeframe='15min')
print(f"Loaded {len(data)} rows")

# Use last 100,000 rows (same as your strategy test)
if len(data) > 100000:
    data = data.tail(100000)
    print(f"Using last {len(data)} rows for analysis")

# Calculate indicators
print("\nCalculating indicators...")
data_with_indicators = calculate_all_indicators(data, verbose=False)

# Run market characterization
print("\n" + "="*80)
print("RUNNING MARKET CHARACTERIZATION")
print("="*80)

characterizer = MarketCharacterizer(transaction_cost=0.0001)
market_profile = characterizer.characterize_market(
    data_with_indicators, 
    instrument='NQ',
    timeframe='15min'
)

# Display results
print("\n" + "="*80)
print("MARKET PROFILE RESULTS")
print("="*80)

print(f"\nInstrument: {market_profile.instrument} {market_profile.timeframe}")
print(f"Sample Size: {market_profile.sample_size:,} bars")
print(f"Confidence Level: {market_profile.confidence_level:.1%}")

print("\n1. DIRECTIONAL BIAS:")
print(f"   Primary Bias: {market_profile.directional_bias.upper()}")
print(f"   Long Edge (Buy & Hold): {market_profile.long_edge:.3f} Sharpe")
print(f"   Short Edge (Short & Hold): {market_profile.short_edge:.3f} Sharpe")

print("\n2. BEHAVIORAL TYPE:")
print(f"   Primary Behavior: {market_profile.primary_behavior.upper()}")
print(f"   - Trend Persistence Score: {market_profile.trend_persistence:.3f}")
print(f"   - Mean Reversion Score: {market_profile.mean_reversion:.3f}")
print(f"   - Volatility Breakout Score: {market_profile.volatility_expansion:.3f}")

print("\n3. OPTIMAL PARAMETERS:")
print(f"   Optimal Holding Period: {market_profile.optimal_holding_period} bars ({market_profile.optimal_holding_period * 15} minutes)")
print(f"   Edge Half-Life: {market_profile.edge_half_life:,} bars ({market_profile.edge_half_life * 15 / 60:.1f} hours)")

print("\n4. RANDOM BASELINE (Must Beat This):")
print(f"   Random Long: {market_profile.random_long_sharpe:.3f} Sharpe")
print(f"   Random Short: {market_profile.random_short_sharpe:.3f} Sharpe")

print("\n" + "="*80)
print("INTERPRETATION & RECOMMENDATIONS")
print("="*80)

# Provide interpretation
if market_profile.primary_behavior == 'trending':
    print("\n✓ This is a TRENDING market")
    print("  - Momentum strategies should work best")
    print("  - Mean reversion likely to underperform")
    print("  - Focus on trend-following approaches")
elif market_profile.primary_behavior == 'mean_reverting':
    print("\n✓ This is a MEAN REVERTING market")
    print("  - Fade extreme moves")
    print("  - Momentum strategies may struggle")
    print("  - Focus on overbought/oversold conditions")
else:
    print("\n✓ This is a BREAKOUT market")
    print("  - Volatility expansion strategies preferred")
    print("  - Trade range breakouts")
    print("  - Avoid tight ranges")

if market_profile.directional_bias == 'long':
    print("\n✓ Market has LONG bias")
    print("  - Prefer long positions")
    print("  - Be cautious with shorts")
elif market_profile.directional_bias == 'short':
    print("\n✓ Market has SHORT bias")
    print("  - Prefer short positions")
    print("  - Be cautious with longs")
else:
    print("\n✓ Market is NEUTRAL")
    print("  - No strong directional edge")
    print("  - Focus on volatility or regime-based strategies")

# Minimum performance thresholds
min_sharpe = max(
    market_profile.random_long_sharpe + 0.2,
    market_profile.random_short_sharpe + 0.2,
    0.3
)

print(f"\n📊 MINIMUM SHARPE RATIO TARGET: {min_sharpe:.3f}")
print("   (Any strategy must beat this to be considered viable)")

print("\n" + "="*80)
print("NEXT STEPS")
print("="*80)
print("\n1. If primary behavior score < 0.3 Sharpe:")
print("   → This timeframe may not have a tradeable edge")
print("   → Consider testing daily timeframe instead")
print("\n2. If directional bias is strong (>0.5 Sharpe):")
print("   → Focus strategies on that direction")
print("   → Regime system should enhance, not fight this bias")
print("\n3. Use these results to:")
print("   → Filter which strategies to test")
print("   → Set realistic performance expectations")
print("   → Adjust regime thresholds accordingly")

print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Save results
output_file = f"market_profile_NQ_15min_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
with open(output_file, 'w') as f:
    f.write(f"Market Profile for {market_profile.instrument} {market_profile.timeframe}\n")
    f.write("="*50 + "\n")
    f.write(f"Primary Behavior: {market_profile.primary_behavior}\n")
    f.write(f"Directional Bias: {market_profile.directional_bias}\n")
    f.write(f"Long Edge: {market_profile.long_edge:.3f}\n")
    f.write(f"Short Edge: {market_profile.short_edge:.3f}\n")
    f.write(f"Optimal Holding: {market_profile.optimal_holding_period} bars\n")
    f.write(f"Minimum Target Sharpe: {min_sharpe:.3f}\n")

print(f"\n✓ Results saved to: {output_file}")

