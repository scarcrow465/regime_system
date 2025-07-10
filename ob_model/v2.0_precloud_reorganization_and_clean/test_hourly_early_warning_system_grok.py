#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python
# coding: utf-8

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import argparse  # For parameterization

# Add paths (make configurable)
sys.path.insert(0, r'C:\Users\rs\GitProjects\regime_system\ob_model\v2.0_precloud_reorganization_and_clean')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.data_loader import load_csv_data
from core.indicators import calculate_all_indicators
from daily_regime_classifier import NQDailyRegimeClassifier
from hourly_early_warning_system_grok import LowerTimeframeEarlyWarningSystem  # Updated import

# Argument parser for configs
parser = argparse.ArgumentParser()
parser.add_argument('--daily_path', default='combined_NQ_daily_data.csv')
parser.add_argument('--ltf_path', default='combined_NQ_1h_data.csv')  # Assume 1H input
parser.add_argument('--timeframes', nargs='+', default=['1H', '4H', '8H'])  # Multiple TFs
parser.add_argument('--lookback_days', type=int, default=252)
args = parser.parse_args()

print("="*80)
print("LTF EARLY WARNING SYSTEM TEST")
print("="*80)
print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Load data
print("\nLoading daily data...")
daily_data = load_csv_data(args.daily_path, timeframe='1d').tail(args.lookback_days * 2)
print(f"Loaded {len(daily_data)} daily bars")

print("\nLoading LTF (1H) data...")
ltf_data = load_csv_data(args.ltf_path, timeframe='60min')
ltf_data = ltf_data[ltf_data.index.date >= daily_data.index[0].date()]
ltf_data = ltf_data[ltf_data.index.date <= daily_data.index[-1].date()]
print(f"Loaded {len(ltf_data)} 1H bars")

# Calculate indicators
print("\nCalculating daily indicators...")
daily_with_indicators = calculate_all_indicators(daily_data, verbose=False)

# Initialize classifier
daily_classifier = NQDailyRegimeClassifier(lookback_days=args.lookback_days)
daily_regimes = daily_classifier.classify_regimes(daily_with_indicators)

# Run for multiple timeframes and ensemble
ensemble_divergences = []
for tf in args.timeframes:
    print(f"\nProcessing {tf} timeframe...")
    ews = LowerTimeframeEarlyWarningSystem(daily_classifier, timeframe=tf)
    
    print(f"Calculating {tf} indicators...")
    ltf_with_indicators = calculate_all_indicators(ltf_data, verbose=False)  # Base on 1H, resample inside
    
    divergences = ews.detect_divergences(daily_regimes, ltf_with_indicators)
    ensemble_divergences.append(divergences['divergence_score'])
    
    # Save per TF
    divergences.to_csv(f'{tf}_divergences_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')

# Ensemble: Average scores for higher confidence
avg_divergence = pd.concat(ensemble_divergences, axis=1).mean(axis=1)
print("\nEnsemble Average Divergence Score (across TFs):")
print(avg_divergence.tail())

# Use divergences from '1H' or avg for analysis/visualization (adjust as needed)
# For simplicity, use the first TF's divergences for detailed analysis
divergences = ews.detect_divergences(daily_regimes, ltf_with_indicators)  # Reuse last one or average if preferred

# Analyze divergence patterns
print("\n" + "="*80)
print("DIVERGENCE ANALYSIS")
print("="*80)

# Overall divergence statistics
total_hours = len(divergences)
direction_div_pct = divergences['direction_divergence'].mean() * 100
strength_div_pct = divergences['strength_divergence'].mean() * 100
volatility_div_pct = divergences['volatility_divergence'].mean() * 100
character_div_pct = divergences['character_divergence'].mean() * 100

print(f"\nOverall Divergence Rates:")
print(f"  Direction: {direction_div_pct:.1f}% of periods")
print(f"  Strength: {strength_div_pct:.1f}% of periods")
print(f"  Volatility: {volatility_div_pct:.1f}% of periods")
print(f"  Character: {character_div_pct:.1f}% of periods")

# Summarize high divergence periods
high_div_threshold = 0.5  # 50% divergence
recent_window = 24 * 7  # Last week, adjusted for TF if needed

print(f"\nSummary of High Divergence Periods (>{high_div_threshold*100}% in {recent_window} period window):")
divergences['rolling_div_score'] = divergences['divergence_score'].rolling(recent_window).mean()
high_div_periods = divergences[divergences['rolling_div_score'] > high_div_threshold]

if len(high_div_periods) > 0:
    # Group consecutive periods
    high_div_periods['group'] = (high_div_periods.index.to_series().diff() > pd.Timedelta(hours=1)).cumsum()
    
    # Calculate summary statistics
    num_periods = high_div_periods['group'].nunique()
    avg_div_score = high_div_periods['divergence_score'].mean() * 100
    div_score_std = high_div_periods['divergence_score'].std() * 100
    div_score_percentiles = high_div_periods['divergence_score'].quantile([0.25, 0.5, 0.75]) * 100
    
    # Count successful predictions
    successful_predictions = 0
    for group_id, group in high_div_periods.groupby('group'):
        start = group.index[0]
        daily_date = start.date()
        if daily_date in daily_regimes.index.date:
            daily_idx = daily_regimes.index.get_loc(pd.Timestamp(daily_date))
            if daily_idx < len(daily_regimes) - 1:
                current_regime = daily_regimes.iloc[daily_idx]['composite_regime']
                next_regime = daily_regimes.iloc[daily_idx + 1]['composite_regime']
                if current_regime != next_regime:
                    successful_predictions += 1
    
    success_rate = (successful_predictions / num_periods * 100) if num_periods > 0 else 0
    
    print(f"  Total High Divergence Periods: {num_periods}")
    print(f"  Average Divergence Score: {avg_div_score:.1f}%")
    print(f"  Divergence Score Std Dev: {div_score_std:.1f}%")
    print(f"  Divergence Score Percentiles: 25%={div_score_percentiles[0.25]:.1f}%, 50%={div_score_percentiles[0.5]:.1f}%, 75%={div_score_percentiles[0.75]:.1f}%")
    print(f"  Periods Leading to Regime Change: {successful_predictions} ({success_rate:.1f}%)")
else:
    print("  No high divergence periods found.")

# Analyze regime change prediction accuracy
print("\n" + "="*80)
print("REGIME CHANGE PREDICTION ANALYSIS")
print("="*80)

# Find all daily regime changes
daily_regime_changes = daily_regimes['composite_regime'] != daily_regimes['composite_regime'].shift(1)
change_dates = daily_regimes[daily_regime_changes].index[1:]  # Skip first

print(f"\nFound {len(change_dates)} daily regime changes")

# Check if LTF divergences preceded each change
lead_times = []
prediction_success = []

for change_date in change_dates[-10:]:  # Last 10 changes
    # Look at 48 periods before the change (adjusted for TF)
    start_check = change_date - pd.Timedelta(hours=48)
    end_check = change_date
    
    # Get divergences in this window
    window_div = divergences[(divergences.index >= start_check) & (divergences.index < end_check)]
    
    if len(window_div) > 0:
        # Calculate average divergence in windows
        div_24h = window_div.iloc[-24:]['divergence_score'].mean() if len(window_div) >= 24 else 0
        div_48h = window_div['divergence_score'].mean()
        
        # Find first significant divergence
        significant_div = window_div[window_div['divergence_score'] > 0.4]
        if len(significant_div) > 0:
            first_warning = significant_div.index[0]
            lead_time = (change_date - first_warning).total_seconds() / 3600
            lead_times.append(lead_time)
            prediction_success.append(True)
            
            print(f"\n  {change_date.strftime('%Y-%m-%d')}:")
            print(f"    Lead time: {lead_time:.1f} hours")
            print(f"    24h divergence: {div_24h*100:.0f}%")
            print(f"    48h divergence: {div_48h*100:.0f}%")
        else:
            prediction_success.append(False)
            print(f"\n  {change_date.strftime('%Y-%m-%d')}: No significant warning")

if lead_times:
    print(f"\nPrediction Statistics:")
    print(f"  Success rate: {sum(prediction_success)/len(prediction_success)*100:.0f}%")
    print(f"  Average lead time: {np.mean(lead_times):.1f} hours")
    print(f"  Median lead time: {np.median(lead_times):.1f} hours")

# Generate current warnings
print("\n" + "="*80)
print("CURRENT WARNINGS")
print("="*80)

current_warnings = ews.generate_warnings(divergences, lookback_periods=24)  # Adjust lookback for TF

if current_warnings:
    for warning in current_warnings:
        print(f"\n{warning['level']} WARNING - {warning['type'].upper()}:")
        print(f"  {warning['message']}")
        if 'divergence_pct' in warning:
            print(f"  Divergence: {warning['divergence_pct']:.0f}%")
else:
    print("\nNo significant warnings at this time")

# Create visualization
print("\nCreating visualization...")

fig, axes = plt.subplots(5, 1, figsize=(14, 12), sharex=True)

# Plot 1: Price with regime changes
ax1 = axes[0]
ax1.plot(ltf_data.index, ltf_data['close'], 'k-', linewidth=0.5, alpha=0.7)

# Mark daily regime changes
for change_date in change_dates:
    ax1.axvline(x=change_date, color='red', linestyle='--', alpha=0.5)

ax1.set_ylabel('Price')
ax1.set_title('NQ LTF Price with Daily Regime Changes (Red Lines)')
ax1.set_yscale('log')

# Plot 2: Direction divergence
ax2 = axes[1]
ax2.fill_between(divergences.index, 0, divergences['direction_divergence'], 
                 alpha=0.5, color='blue', label='Direction Divergence')
ax2.set_ylabel('Divergence')
ax2.set_title('Direction Regime Divergence (LTF vs Daily)')
ax2.set_ylim(-0.1, 1.1)

# Plot 3: Strength divergence
ax3 = axes[2]
ax3.fill_between(divergences.index, 0, divergences['strength_divergence'], 
                 alpha=0.5, color='orange', label='Strength Divergence')
ax3.set_ylabel('Divergence')
ax3.set_title('Strength Regime Divergence')
ax3.set_ylim(-0.1, 1.1)

# Plot 4: Volatility divergence
ax4 = axes[3]
ax4.fill_between(divergences.index, 0, divergences['volatility_divergence'], 
                 alpha=0.5, color='red', label='Volatility Divergence')
ax4.set_ylabel('Divergence')
ax4.set_title('Volatility Regime Divergence')
ax4.set_ylim(-0.1, 1.1)

# Plot 5: Composite divergence score
ax5 = axes[4]
ax5.plot(divergences.index, divergences['divergence_score'], 'purple', linewidth=1)
ax5.fill_between(divergences.index, 0, divergences['divergence_score'], 
                 alpha=0.3, color='purple')

# Add warning level lines
ax5.axhline(y=0.3, color='yellow', linestyle='--', alpha=0.5, label='Weak Warning')
ax5.axhline(y=0.5, color='orange', linestyle='--', alpha=0.5, label='Moderate Warning')
ax5.axhline(y=0.7, color='red', linestyle='--', alpha=0.5, label='Strong Warning')

ax5.set_ylabel('Score')
ax5.set_xlabel('Date')
ax5.set_title('Composite Divergence Score')
ax5.set_ylim(0, 1)
ax5.legend()

plt.tight_layout()
plt.savefig(f'ltf_early_warning_{datetime.now().strftime("%Y%m%d")}.png', dpi=150)
print("✓ Saved divergence chart")

# Save divergence data
output_file = f'ltf_divergences_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
divergences.to_csv(output_file)
print(f"\n✓ Divergence data saved to: {output_file}")

print("\n" + "="*80)
print("EARLY WARNING SYSTEM INSIGHTS")
print("="*80)

print("\n1. DIVERGENCE PATTERNS:")
print("   - Normal divergence rate: 20-30% is healthy")
print("   - >50% sustained divergence often precedes regime change")
print("   - Direction divergence is most predictive")

print("\n2. TYPICAL LEAD TIMES:")
print("   - Minor regime adjustments: 6-12 hours warning")
print("   - Major regime changes: 24-48 hours warning")
print("   - Crisis/volatile transitions: Can be sudden (<6 hours)")

print("\n3. USAGE RECOMMENDATIONS:")
print("   - Monitor composite score >0.5 for potential changes")
print("   - Direction divergence >70% = high probability of trend change")
print("   - Multiple divergences = higher confidence signal")

print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

