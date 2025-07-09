#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
Test the NQ Daily Regime Classifier
Validates regime classification and persistence
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# Add paths
sys.path.insert(0, r'C:\Users\rs\GitProjects\regime_system\ob_model\v2.0_precloud_reorganization_and_clean')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.data_loader import load_csv_data
from core.indicators import calculate_all_indicators
from daily_regime_classifier import NQDailyRegimeClassifier, DailyRegime
# Verify imports
print("Imported NQDailyRegimeClassifier:", NQDailyRegimeClassifier)
print("Imported DailyRegime:", DailyRegime)

print("="*80)
print("NQ DAILY REGIME CLASSIFIER TEST")
print("="*80)
print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Load daily data
print("\nLoading NQ daily data...")
data = load_csv_data(r'combined_NQ_daily_data.csv', timeframe='1d')
print(f"Loaded {len(data)} daily bars")
print(f"Date range: {data.index[0]} to {data.index[-1]}")

# Focus on recent 5 years for testing
data = data.tail(252 * 10)  # Last 5 years
print(f"\nUsing last 5 years: {data.index[0]} to {data.index[-1]}")

# Calculate indicators
print("\nCalculating indicators...")
data_with_indicators = calculate_all_indicators(data, verbose=False)

# Initialize classifier
classifier = NQDailyRegimeClassifier(lookback_days=252)

# Classify regimes
print("\nClassifying daily regimes...")
regime_data = classifier.classify_regimes(data_with_indicators)

# Get current regime
current_regime = classifier.get_current_regime(data_with_indicators)

print("\n" + "="*80)
print("CURRENT REGIME STATE")
print("="*80)
print(f"\nAs of: {current_regime.timestamp.strftime('%Y-%m-%d')}")
print(f"\nPrimary Classifications:")
print(f"  Direction: {current_regime.direction} (confidence: {current_regime.direction_confidence:.2%})")
print(f"  Strength: {current_regime.strength} (confidence: {current_regime.strength_confidence:.2%})")
print(f"  Volatility: {current_regime.volatility} (confidence: {current_regime.volatility_confidence:.2%})")
print(f"  Character: {current_regime.character} (confidence: {current_regime.character_confidence:.2%})")
print(f"\nComposite Regime: {current_regime.composite_regime}")
print(f"Regime Age: {current_regime.regime_age} days")
print(f"\nSupporting Metrics:")
print(f"  Trend Score: {current_regime.trend_score:.3f}")
print(f"  Efficiency Ratio: {current_regime.efficiency_ratio:.3f}")
print(f"  Volatility Percentile: {current_regime.volatility_percentile:.1f}%")

# Analyze regime distribution
print("\n" + "="*80)
print("REGIME DISTRIBUTION ANALYSIS")
print("="*80)

# Direction distribution
direction_dist = regime_data['direction_regime'].value_counts()
print("\nDirection Regimes:")
for regime, count in direction_dist.items():
    print(f"  {regime}: {count} days ({count/len(regime_data)*100:.1f}%)")

# Strength distribution
strength_dist = regime_data['strength_regime'].value_counts()
print("\nStrength Regimes:")
for regime, count in strength_dist.items():
    print(f"  {regime}: {count} days ({count/len(regime_data)*100:.1f}%)")

# Volatility distribution
vol_dist = regime_data['volatility_regime'].value_counts()
print("\nVolatility Regimes:")
for regime, count in vol_dist.items():
    print(f"  {regime}: {count} days ({count/len(regime_data)*100:.1f}%)")

# Character distribution
char_dist = regime_data['character_regime'].value_counts()
print("\nCharacter Regimes:")
for regime, count in char_dist.items():
    print(f"  {regime}: {count} days ({count/len(regime_data)*100:.1f}%)")

# Analyze regime persistence
print("\n" + "="*80)
print("REGIME PERSISTENCE ANALYSIS")
print("="*80)

# Find regime transitions
regime_changes = regime_data['composite_regime'] != regime_data['composite_regime'].shift(1)
transition_dates = regime_data[regime_changes].index

print(f"\nTotal regime changes: {len(transition_dates) - 1}")
print(f"Average regime duration: {len(regime_data) / (len(transition_dates) - 1):.1f} days")

# Show recent regime transitions
print("\nRecent Regime Transitions (last 10):")
recent_transitions = regime_data[regime_changes].tail(10)
for i in range(len(recent_transitions) - 1):
    from_regime = recent_transitions.iloc[i]['composite_regime']
    to_regime = recent_transitions.iloc[i+1]['composite_regime']
    date = recent_transitions.index[i+1]
    days = (recent_transitions.index[i+1] - recent_transitions.index[i]).days
    print(f"  {date.strftime('%Y-%m-%d')}: {from_regime} → {to_regime} (lasted {days} days)")

# Analyze regime performance
print("\n" + "="*80)
print("REGIME PERFORMANCE ANALYSIS")
print("="*80)

# Calculate returns for each regime
regime_data['returns'] = data_with_indicators['close'].pct_change()
regime_data['log_returns'] = np.log(data_with_indicators['close'] / data_with_indicators['close'].shift(1))

# Performance by direction regime
print("\nPerformance by Direction Regime:")
for regime in regime_data['direction_regime'].unique():
    mask = regime_data['direction_regime'] == regime
    regime_returns = regime_data.loc[mask, 'returns']
    
    if len(regime_returns) > 0:
        avg_return = regime_returns.mean() * 252 * 100  # Annualized
        volatility = regime_returns.std() * np.sqrt(252) * 100
        sharpe = avg_return / volatility if volatility > 0 else 0
        
        print(f"  {regime}:")
        print(f"    Avg Return: {avg_return:.1f}% annualized")
        print(f"    Volatility: {volatility:.1f}%")
        print(f"    Sharpe: {sharpe:.3f}")

# Performance by composite regime (top 10)
print("\nTop 10 Composite Regimes by Frequency:")
top_regimes = regime_data['composite_regime'].value_counts().head(10)
for regime, count in top_regimes.items():
    mask = regime_data['composite_regime'] == regime
    regime_returns = regime_data.loc[mask, 'returns']
    
    if len(regime_returns) > 20:  # Only if enough data
        avg_return = regime_returns.mean() * 252 * 100
        sharpe = avg_return / (regime_returns.std() * np.sqrt(252) * 100) if regime_returns.std() > 0 else 0
        
        print(f"  {regime}: {count} days")
        print(f"    Return: {avg_return:.1f}% ann., Sharpe: {sharpe:.3f}")

# Create visualizations
print("\nCreating regime visualizations...")

# 1. Regime Timeline
fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)

# Map regimes to colors
direction_colors = {'Uptrend': 'green', 'Downtrend': 'red', 'Sideways': 'gray'}
strength_colors = {'Strong': 'darkblue', 'Moderate': 'blue', 'Weak': 'lightblue'}
vol_colors = {'Low': 'lightgreen', 'Normal': 'yellow', 'High': 'orange', 'Extreme': 'red'}
char_colors = {'Trending': 'purple', 'Ranging': 'gray', 'Volatile': 'red', 'Transitioning': 'orange'}

# Plot price with regime colors
ax = axes[0]
ax.plot(data_with_indicators.index, data_with_indicators['close'], 'k-', linewidth=0.5, alpha=0.7)

regime_changes = regime_data['direction_regime'] != regime_data['direction_regime'].shift(1)
regime_changes.iloc[0] = True  # Ensure first point is included
change_indices = regime_data.index[regime_changes]
change_indices = regime_data['direction_regime'][regime_changes]

for i in range(len(change_indices) - 1):
    start_idx = change_indices.index[i]
    end_idx = change_indices.index[i + 1]
    regime = change_indices.iloc[i]
    mask = (regime_data.index >= start_idx) & (regime_data.index < end_idx)
    ax.fill_between(
        regime_data.index[mask],
        data_with_indicators['close'].min() * 0.9,
        data_with_indicators['close'].max() * 1.1,
        alpha=0.3,
        color=direction_colors[regime],
        label=regime if i == 0 or regime not in ax.get_legend_handles_labels()[1] else ""
    )

    

ax.set_ylabel('Price')
ax.set_title('NQ Daily Price with Direction Regimes')
ax.legend(loc='upper left')
ax.set_yscale('log')

# Plot regime indicators
axes[1].plot(regime_data.index, regime_data['direction_score'], label='Direction Score')
axes[1].axhline(0, color='black', linestyle='--', alpha=0.5)
axes[1].set_ylabel('Score')
axes[1].set_title('Regime Scores')
axes[1].legend()

axes[2].plot(regime_data.index, regime_data['volatility_score'] * 100, label='Volatility Percentile', color='orange')
axes[2].set_ylabel('Percentile')
axes[2].set_title('Volatility Regime')

axes[3].plot(regime_data.index, regime_data['regime_confidence'], label='Regime Confidence', color='purple')
axes[3].set_ylabel('Confidence')
axes[3].set_title('Overall Regime Confidence')
axes[3].set_xlabel('Date')

plt.tight_layout()
plt.savefig(f'nq_daily_regime_timeline_{datetime.now().strftime("%Y%m%d")}.png', dpi=150)
print("✓ Saved regime timeline chart")

# 2. Regime Transition Matrix
plt.figure(figsize=(10, 8))

# Create transition matrix for direction regimes
regimes = regime_data['direction_regime'].unique()
transition_matrix = pd.DataFrame(0, index=regimes, columns=regimes)

for i in range(1, len(regime_data)):
    from_regime = regime_data['direction_regime'].iloc[i-1]
    to_regime = regime_data['direction_regime'].iloc[i]
    transition_matrix.loc[from_regime, to_regime] += 1

# Normalize to probabilities
transition_matrix = transition_matrix.div(transition_matrix.sum(axis=1), axis=0)

# Plot heatmap
sns.heatmap(transition_matrix, annot=True, fmt='.2f', cmap='Blues', 
            cbar_kws={'label': 'Transition Probability'})
plt.title('Daily Regime Transition Probabilities')
plt.xlabel('To Regime')
plt.ylabel('From Regime')
plt.tight_layout()
plt.savefig(f'nq_regime_transitions_{datetime.now().strftime("%Y%m%d")}.png', dpi=150)
print("✓ Saved transition matrix")

# Save regime data
output_file = f'nq_daily_regimes_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
regime_data.to_csv(output_file)
print(f"\n✓ Regime data saved to: {output_file}")

# Summary recommendations
print("\n" + "="*80)
print("REGIME INSIGHTS & RECOMMENDATIONS")
print("="*80)

# Check if regimes are meaningful
avg_duration = len(regime_data) / (len(transition_dates) - 1)
if avg_duration < 5:
    print("\n⚠ WARNING: Regimes changing too frequently (avg < 5 days)")
    print("  Consider increasing smoothing parameters")
elif avg_duration > 50:
    print("\n⚠ WARNING: Regimes too persistent (avg > 50 days)")
    print("  Consider decreasing smoothing parameters")
else:
    print(f"\n✓ Regime persistence looks good: {avg_duration:.1f} days average")

# Check regime balance
uptrend_pct = (regime_data['direction_regime'] == 'Uptrend').mean() * 100
if uptrend_pct > 70:
    print(f"\n✓ Strong upward bias confirmed: {uptrend_pct:.1f}% in uptrend")
    print("  Aligns with market characterization (4.65 Sharpe)")

# Volatility insights
low_vol_pct = (regime_data['volatility_regime'] == 'Low').mean() * 100
if low_vol_pct > 40:
    print(f"\n✓ Market often in low volatility: {low_vol_pct:.1f}%")
    print("  Good for trend-following strategies")

print("\nNext Steps:")
print("1. Test how 1-hour and 15-min regimes predict daily regime changes")
print("2. Validate that trading with daily regime alignment improves performance")
print("3. Create early warning system using intraday divergences")

print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

