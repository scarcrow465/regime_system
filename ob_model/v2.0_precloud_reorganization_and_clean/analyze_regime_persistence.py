#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
Analyze regime persistence - how long do regimes typically last?
This helps verify the 3-period smoothing is appropriate
"""

import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime
import matplotlib.pyplot as plt

# Add regime_system to path
sys.path.insert(0, r'C:\Users\rs\GitProjects\regime_system\ob_model\v2.0_precloud_reorganization_and_clean')

from core.data_loader import load_csv_data
from core.indicators import calculate_all_indicators
from core.regime_classifier import RollingRegimeClassifier

print("="*80)
print("REGIME PERSISTENCE ANALYSIS")
print("="*80)
print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Load data
data_file = r'combined_NQ_15m_data.csv'
print(f"\nLoading data from: {data_file}")
data = load_csv_data(data_file, timeframe='15min')
print(f"Loaded {len(data)} rows")

# Use last 50,000 rows for faster analysis but still meaningful
if len(data) > 50000:
    data = data.tail(50000)
    print(f"Using last {len(data)} rows for persistence analysis")

# Calculate indicators
print("\nCalculating indicators...")
data_with_indicators = calculate_all_indicators(data, verbose=False)

# Classify regimes
print("\nClassifying regimes...")
classifier = RollingRegimeClassifier(window_hours=36, timeframe='15min')
regimes = classifier.classify_regimes(data_with_indicators, show_progress=True)

# Function to analyze regime persistence
def analyze_regime_persistence(regimes_df, dimension_name):
    """Analyze how long regimes persist in a given dimension"""
    regime_col = f'{dimension_name}_Regime'
    
    # Track regime changes
    regime_changes = []
    current_regime = None
    regime_start = 0
    
    for i in range(len(regimes_df)):
        regime = regimes_df[regime_col].iloc[i]
        
        if regime == 'Undefined':
            continue
            
        if current_regime is None:
            current_regime = regime
            regime_start = i
        elif regime != current_regime:
            # Regime changed
            duration = i - regime_start
            regime_changes.append({
                'regime': current_regime,
                'duration_bars': duration,
                'duration_hours': duration * 0.25,  # 15-min bars
                'duration_days': duration * 0.25 / 24,
                'start_idx': regime_start,
                'end_idx': i
            })
            current_regime = regime
            regime_start = i
    
    # Don't forget the last regime
    if current_regime is not None:
        duration = len(regimes_df) - regime_start
        regime_changes.append({
            'regime': current_regime,
            'duration_bars': duration,
            'duration_hours': duration * 0.25,
            'duration_days': duration * 0.25 / 24,
            'start_idx': regime_start,
            'end_idx': len(regimes_df)
        })
    
    return pd.DataFrame(regime_changes)

# Analyze persistence for each dimension
print("\n" + "="*80)
print("REGIME PERSISTENCE RESULTS")
print("="*80)

dimensions = ['Direction', 'TrendStrength', 'Velocity', 'Volatility', 'Microstructure']
persistence_summary = {}

for dim in dimensions:
    print(f"\n{dim} Dimension:")
    persistence_df = analyze_regime_persistence(regimes, dim)
    
    if len(persistence_df) == 0:
        print("  No regime changes found")
        continue
    
    # Calculate statistics
    stats = {
        'total_changes': len(persistence_df),
        'avg_duration_hours': persistence_df['duration_hours'].mean(),
        'median_duration_hours': persistence_df['duration_hours'].median(),
        'min_duration_hours': persistence_df['duration_hours'].min(),
        'max_duration_hours': persistence_df['duration_hours'].max(),
        'std_duration_hours': persistence_df['duration_hours'].std()
    }
    
    persistence_summary[dim] = stats
    
    print(f"  Total regime changes: {stats['total_changes']}")
    print(f"  Average duration: {stats['avg_duration_hours']:.1f} hours ({stats['avg_duration_hours']/24:.1f} days)")
    print(f"  Median duration: {stats['median_duration_hours']:.1f} hours ({stats['median_duration_hours']/24:.1f} days)")
    print(f"  Min duration: {stats['min_duration_hours']:.1f} hours")
    print(f"  Max duration: {stats['max_duration_hours']:.1f} hours ({stats['max_duration_hours']/24:.1f} days)")
    print(f"  Std deviation: {stats['std_duration_hours']:.1f} hours")
    
    # Show distribution by regime type
    print(f"\n  Duration by regime type (average hours):")
    regime_durations = persistence_df.groupby('regime')['duration_hours'].agg(['mean', 'count'])
    for regime, row in regime_durations.iterrows():
        print(f"    {regime}: {row['mean']:.1f} hours (occurred {row['count']} times)")

# Analyze smoothing effectiveness
print("\n" + "="*80)
print("SMOOTHING ANALYSIS (3-period confirmation)")
print("="*80)

# Count very short regimes (less than 3 periods = 0.75 hours)
for dim in dimensions:
    persistence_df = analyze_regime_persistence(regimes, dim)
    if len(persistence_df) > 0:
        short_regimes = persistence_df[persistence_df['duration_hours'] < 0.75]
        pct_short = len(short_regimes) / len(persistence_df) * 100
        print(f"{dim}: {len(short_regimes)} regimes < 0.75 hours ({pct_short:.1f}% of all regimes)")

# Check for regime flipping (A->B->A patterns)
print("\n" + "="*80)
print("REGIME FLIPPING ANALYSIS")
print("="*80)

for dim in dimensions:
    regime_col = f'{dim}_Regime'
    regime_values = regimes[regime_col].values
    
    flips = 0
    for i in range(2, len(regime_values)):
        if (regime_values[i] == regime_values[i-2] and 
            regime_values[i] != regime_values[i-1] and
            regime_values[i] != 'Undefined'):
            flips += 1
    
    print(f"{dim}: {flips} flip patterns (A->B->A) found")

# Summary and recommendations
print("\n" + "="*80)
print("SUMMARY & RECOMMENDATIONS")
print("="*80)

# Calculate average regime duration across all dimensions
all_durations = []
for dim in dimensions:
    if dim in persistence_summary:
        all_durations.append(persistence_summary[dim]['avg_duration_hours'])

if all_durations:
    overall_avg = np.mean(all_durations)
    print(f"\nOverall average regime duration: {overall_avg:.1f} hours ({overall_avg/24:.1f} days)")
    
    # Check if 3-period smoothing is appropriate
    if overall_avg < 1:  # Less than 1 hour average
        print("⚠️ WARNING: Regimes are changing very rapidly!")
        print("  Consider increasing smoothing periods from 3 to 5-7")
    elif overall_avg > 100:  # More than 4 days average
        print("⚠️ WARNING: Regimes are very persistent!")
        print("  Consider decreasing smoothing periods from 3 to 1-2")
    else:
        print("✓ Current 3-period smoothing appears appropriate")
        print(f"  Regimes last {overall_avg:.1f} hours on average")

# Save detailed results
output_file = f"regime_persistence_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
all_persistence = pd.DataFrame()
for dim in dimensions:
    df = analyze_regime_persistence(regimes, dim)
    df['dimension'] = dim
    all_persistence = pd.concat([all_persistence, df])

all_persistence.to_csv(output_file, index=False)
print(f"\n✓ Detailed results saved to: {output_file}")

print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

