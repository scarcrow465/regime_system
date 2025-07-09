#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
Test 1-Hour NQ Market Characterization
This could be the sweet spot for day trading
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime

# Add paths
sys.path.insert(0, r'C:\Users\rs\GitProjects\regime_system\ob_model\v2.0_precloud_reorganization_and_clean')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.data_loader import load_csv_data
from core.indicators import calculate_all_indicators
from core.market_characterizer import MarketCharacterizer

print("="*80)
print("1-HOUR NQ MARKET CHARACTERIZATION")
print("="*80)
print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Try different possible filenames for 1-hour data
possible_files = [
    r'combined_NQ_60m_data.csv',
    r'combined_NQ_1h_data.csv',
    r'NQ_60m_data.csv',
    r'NQ_1h_data.csv',
    r'combined_NQ_60min_data.csv'
]

data = None
for filename in possible_files:
    try:
        print(f"\nTrying to load: {filename}")
        data = load_csv_data(filename, timeframe='60min')
        print(f"✓ Successfully loaded from: {filename}")
        break
    except Exception as e:
        print(f"✗ Failed: {str(e)}")
        continue

if data is None:
    print("\nERROR: Could not find 1-hour data file")
    print("Please update the script with the correct filename")
    sys.exit(1)

print(f"\nLoaded {len(data)} hourly bars")
print(f"Date range: {data.index[0]} to {data.index[-1]}")

# Use last 50,000 bars for analysis (about 8 years of hourly data)
if len(data) > 50000:
    data = data.tail(50000)
    print(f"Using last {len(data)} bars for analysis")

# Calculate indicators
print("\nCalculating indicators...")
data_with_indicators = calculate_all_indicators(data, verbose=False)

# Run market characterization with appropriate transaction costs
print("\n" + "="*80)
print("MARKET CHARACTERIZATION")
print("="*80)

characterizer = MarketCharacterizer(transaction_cost=0.00007)  # 0.7 bps for hourly
profile = characterizer.characterize_market(
    data_with_indicators, 
    instrument='NQ',
    timeframe='1-Hour'
)

# Display results
print(f"\nMarket Profile Summary:")
print(f"  Primary Behavior: {profile.primary_behavior.upper()}")
print(f"  Directional Bias: {profile.directional_bias.upper()}")
print(f"  Long Edge (Buy & Hold): {profile.long_edge:.3f} Sharpe")
print(f"  Short Edge: {profile.short_edge:.3f} Sharpe")

print(f"\nStrategy Performance:")
print(f"  Trend Following: {profile.trend_persistence:.3f} Sharpe")
print(f"  Mean Reversion: {profile.mean_reversion:.3f} Sharpe")
print(f"  Volatility Breakout: {profile.volatility_expansion:.3f} Sharpe")

print(f"\nOptimal Parameters:")
print(f"  Holding Period: {profile.optimal_holding_period} hours")
print(f"  Holding Period (days): {profile.optimal_holding_period / 6.5:.1f} trading days")
print(f"  Edge Half-Life: {profile.edge_half_life} hours")

print(f"\nRandom Baseline:")
print(f"  Random Long: {profile.random_long_sharpe:.3f} Sharpe")
print(f"  Random Short: {profile.random_short_sharpe:.3f} Sharpe")

# Test specific holding periods on hourly data
print("\n" + "="*80)
print("HOLDING PERIOD ANALYSIS FOR 1-HOUR DATA")
print("="*80)

holding_periods = {
    '2 hours': 2,
    '4 hours': 4,
    '6.5 hours (1 day)': 7,
    '13 hours (2 days)': 13,
    '32.5 hours (5 days)': 33,
    '65 hours (10 days)': 65,
}

results = {}
transaction_cost = 0.00007

for period_name, bars in holding_periods.items():
    if bars > len(data_with_indicators) / 10:
        continue
        
    # Simple momentum strategy
    returns = data_with_indicators['close'].pct_change(bars).shift(-bars)
    signal = (data_with_indicators['close'] > data_with_indicators['close'].shift(bars)).astype(int)
    
    # Calculate strategy returns
    strategy_returns = signal * returns
    trades = signal.diff().abs()
    costs = trades * transaction_cost
    strategy_returns = strategy_returns - costs
    
    # Calculate metrics
    clean_returns = strategy_returns.dropna()
    if len(clean_returns) > 0 and clean_returns.std() > 0:
        # Annualize for hourly data (252 * 6.5 hours per year)
        sharpe = clean_returns.mean() / clean_returns.std() * np.sqrt(252 * 6.5)
        win_rate = (clean_returns > 0).mean()
    else:
        sharpe = -999
        win_rate = 0
    
    results[period_name] = {'sharpe': sharpe, 'win_rate': win_rate}
    print(f"{period_name:<20}: Sharpe={sharpe:>7.3f}, WinRate={win_rate:>6.1%}")

# Compare with your 15-min and daily results
print("\n" + "="*80)
print("TIMEFRAME COMPARISON")
print("="*80)

print("\nFor 1-Day Holding Period:")
print("  15-min data: 2.464 Sharpe (your result)")
print(f"  1-hour data: {results.get('6.5 hours (1 day)', {}).get('sharpe', 'N/A'):.3f} Sharpe")
print("  Daily data: 0.240 Sharpe (your result)")

print("\n" + "="*80)
print("HIERARCHICAL REGIME RECOMMENDATIONS")
print("="*80)

if profile.primary_behavior == 'trending' and profile.trend_persistence > 0.5:
    print("\n✓ 1-Hour shows TRENDING behavior with positive edge!")
    print("\nRecommended Hierarchy:")
    print("1. DAILY: Directional bias (already confirmed strong LONG)")
    print("2. 1-HOUR: Primary regime classification")
    print("3. 15-MIN: Entry timing within hourly regime")
    
elif profile.primary_behavior == 'mean_reverting' and profile.mean_reversion > 0.5:
    print("\n✓ 1-Hour shows MEAN REVERTING behavior!")
    print("\nRecommended Hierarchy:")
    print("1. DAILY: Trend context")
    print("2. 1-HOUR: Identify overbought/oversold")
    print("3. 15-MIN: Time reversal entries")
    
else:
    print("\n⚠ 1-Hour may not have clear edge")
    print("Stick with 15-min data for 1-day holds")

print(f"\nMinimum Performance Target: {max(profile.random_long_sharpe + 0.3, 0.5):.3f} Sharpe")

# Save results
output_file = f"market_profile_NQ_1hour_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
with open(output_file, 'w') as f:
    f.write(f"1-Hour Market Profile\n")
    f.write("="*50 + "\n")
    f.write(f"Primary Behavior: {profile.primary_behavior}\n")
    f.write(f"Directional Bias: {profile.directional_bias}\n")
    f.write(f"Best Strategy Score: {max(profile.trend_persistence, profile.mean_reversion, profile.volatility_expansion):.3f}\n")
    f.write(f"Optimal Holding: {profile.optimal_holding_period} hours\n")
    
print(f"\n✓ Results saved to: {output_file}")
print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

