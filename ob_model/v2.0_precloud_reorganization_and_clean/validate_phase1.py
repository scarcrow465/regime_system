#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
Script to update indicators with full calculations and run validation
Phase 1 - Step 1: Fix indicator calculations and validate
"""

import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime

# Add regime_system to path
sys.path.insert(0, 'regime_system')

# First, backup the existing indicators.py
print("="*80)
print("PHASE 1: INDICATOR RESTORATION AND VALIDATION")
print("="*80)
print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Step 1: Backup existing file
print("\nStep 1: Creating backup of existing indicators.py...")
backup_path = f"regime_system/core/indicators_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.py"
try:
    import shutil
    shutil.copy2('regime_system/core/indicators.py', backup_path)
    print(f"✓ Backup created: {backup_path}")
except Exception as e:
    print(f"✗ Backup failed: {e}")

# Step 2: Import the restored indicators from our artifact
print("\nStep 2: Loading restored indicator calculations...")
# In practice, you would copy the restored_indicators content to indicators.py
# For now, let's validate what we have

from regime_system.core.data_loader import load_csv_data
from regime_system.core.indicators import calculate_all_indicators, validate_indicators
from regime_system.core.regime_classifier import RollingRegimeClassifier

# Step 3: Run indicator validation
print("\nStep 3: Running indicator validation...")

def run_indicator_validation(data_file: str = None):
    """Run comprehensive indicator validation"""
    
    # Load test data
    if data_file and os.path.exists(data_file):
        print(f"Loading data from: {data_file}")
        data = load_csv_data(data_file, timeframe='15min')
    else:
        print("Generating synthetic test data...")
        # Generate synthetic data for testing
        dates = pd.date_range(start='2023-01-01', periods=1000, freq='15min')
        data = pd.DataFrame({
            'Date': dates,
            'open': 100 + np.cumsum(np.random.randn(1000) * 0.5),
            'high': 0,
            'low': 0,
            'close': 0,
            'volume': np.random.randint(1000, 10000, 1000)
        })
        data['high'] = data['open'] + np.abs(np.random.randn(1000) * 0.3)
        data['low'] = data['open'] - np.abs(np.random.randn(1000) * 0.3)
        data['close'] = data['low'] + (data['high'] - data['low']) * np.random.rand(1000)
        data.set_index('Date', inplace=True)
    
    print(f"Data shape: {data.shape}")
    
    # Calculate indicators
    print("\nCalculating indicators...")
    try:
        data_with_indicators = calculate_all_indicators(data, verbose=True)
        indicator_count = len(data_with_indicators.columns) - len(data.columns)
        print(f"\n✓ Successfully calculated {indicator_count} indicators")
    except Exception as e:
        print(f"\n✗ Error calculating indicators: {e}")
        return None
    
    # Validate indicators
    print("\nValidating indicators...")
    validation_results = validate_indicators(data_with_indicators)
    
    print("\nValidation Results:")
    print(f"  Valid indicators: {len(validation_results['valid'])}")
    print(f"  Missing indicators: {len(validation_results['missing'])}")
    print(f"  All NaN indicators: {len(validation_results['all_nan'])}")
    print(f"  Mostly NaN indicators: {len(validation_results['mostly_nan'])}")
    print(f"  Constant indicators: {len(validation_results['constant'])}")
    
    if validation_results['missing']:
        print(f"\n  Missing: {validation_results['missing'][:5]}{'...' if len(validation_results['missing']) > 5 else ''}")
    
    if validation_results['all_nan']:
        print(f"\n  All NaN: {validation_results['all_nan'][:5]}{'...' if len(validation_results['all_nan']) > 5 else ''}")
    
    return data_with_indicators, validation_results

def check_confidence_scores(data_with_indicators):
    """Check if confidence scores are properly varied"""
    print("\nStep 4: Testing regime classification confidence scores...")
    
    # Create classifier
    classifier = RollingRegimeClassifier(window_hours=36, timeframe='15min')
    
    # Classify regimes
    print("Classifying regimes...")
    regimes = classifier.classify_regimes(data_with_indicators, show_progress=True)
    
    # Check confidence scores
    print("\nConfidence Score Analysis:")
    for dim in ['Direction', 'TrendStrength', 'Velocity', 'Volatility', 'Microstructure']:
        conf_col = f'{dim}_Confidence'
        if conf_col in regimes.columns:
            conf_values = regimes[conf_col].dropna()
            print(f"\n{dim} Confidence:")
            print(f"  Mean: {conf_values.mean():.3f}")
            print(f"  Std: {conf_values.std():.3f}")
            print(f"  Min: {conf_values.min():.3f}")
            print(f"  Max: {conf_values.max():.3f}")
            print(f"  Unique values: {len(conf_values.unique())}")
            
            # Check if all values are 1.0
            if (conf_values == 1.0).all():
                print(f"  ⚠️ WARNING: All confidence scores are 1.0!")
            elif conf_values.std() < 0.01:
                print(f"  ⚠️ WARNING: Very low variance in confidence scores!")
            else:
                print(f"  ✓ Confidence scores show proper variation")
    
    return regimes

def analyze_indicator_correlations(data_with_indicators):
    """Analyze correlations between indicators"""
    print("\nStep 5: Analyzing indicator correlations...")
    
    # Select only indicator columns (exclude OHLCV)
    base_cols = ['open', 'high', 'low', 'close', 'volume']
    indicator_cols = [col for col in data_with_indicators.columns if col not in base_cols]
    
    # Calculate correlation matrix
    print(f"Calculating correlations for {len(indicator_cols)} indicators...")
    corr_matrix = data_with_indicators[indicator_cols].corr()
    
    # Find highly correlated pairs
    high_corr_threshold = 0.90
    high_corr_pairs = []
    
    for i in range(len(indicator_cols)):
        for j in range(i+1, len(indicator_cols)):
            corr_value = abs(corr_matrix.iloc[i, j])
            if corr_value > high_corr_threshold and not pd.isna(corr_value):
                high_corr_pairs.append({
                    'indicator1': indicator_cols[i],
                    'indicator2': indicator_cols[j],
                    'correlation': corr_value
                })
    
    print(f"\nFound {len(high_corr_pairs)} indicator pairs with correlation > {high_corr_threshold}")
    
    if high_corr_pairs:
        # Sort by correlation
        high_corr_pairs.sort(key=lambda x: x['correlation'], reverse=True)
        print("\nTop 10 highly correlated pairs:")
        for i, pair in enumerate(high_corr_pairs[:10]):
            print(f"  {i+1}. {pair['indicator1']} <-> {pair['indicator2']}: {pair['correlation']:.3f}")
        
        # Group by indicator
        redundant_indicators = {}
        for pair in high_corr_pairs:
            for ind in [pair['indicator1'], pair['indicator2']]:
                if ind not in redundant_indicators:
                    redundant_indicators[ind] = 0
                redundant_indicators[ind] += 1
        
        # Find most redundant indicators
        most_redundant = sorted(redundant_indicators.items(), key=lambda x: x[1], reverse=True)[:10]
        print("\nMost redundant indicators (appear in many high-correlation pairs):")
        for ind, count in most_redundant:
            print(f"  {ind}: appears in {count} high-correlation pairs")
    
    return corr_matrix, high_corr_pairs

def create_validation_report(validation_results, regimes, high_corr_pairs):
    """Create a comprehensive validation report"""
    print("\n" + "="*80)
    print("VALIDATION REPORT")
    print("="*80)
    
    report = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'indicators': {
            'total_calculated': len(validation_results['valid']) + len(validation_results['all_nan']) + 
                              len(validation_results['mostly_nan']) + len(validation_results['constant']),
            'valid': len(validation_results['valid']),
            'problematic': len(validation_results['all_nan']) + len(validation_results['mostly_nan']) + 
                          len(validation_results['constant']),
            'missing_expected': len(validation_results['missing'])
        },
        'confidence_scores': {
            'all_ones_issue': False,
            'low_variance_issue': False,
            'dimensions_checked': 5
        },
        'correlations': {
            'high_correlation_pairs': len(high_corr_pairs),
            'threshold_used': 0.90
        },
        'recommendations': []
    }
    
    # Check confidence score issues
    for dim in ['Direction', 'TrendStrength', 'Velocity', 'Volatility', 'Microstructure']:
        conf_col = f'{dim}_Confidence'
        if conf_col in regimes.columns:
            conf_values = regimes[conf_col].dropna()
            if (conf_values == 1.0).all():
                report['confidence_scores']['all_ones_issue'] = True
            elif conf_values.std() < 0.01:
                report['confidence_scores']['low_variance_issue'] = True
    
    # Generate recommendations
    if report['confidence_scores']['all_ones_issue']:
        report['recommendations'].append("CRITICAL: Update indicator calculations to use full formulas instead of simplified versions")
    
    if report['indicators']['problematic'] > 10:
        report['recommendations'].append("HIGH: Review and fix indicators that are returning NaN or constant values")
    
    if report['correlations']['high_correlation_pairs'] > 20:
        report['recommendations'].append("MEDIUM: Consider removing redundant indicators with correlation > 0.90")
    
    if report['indicators']['missing_expected'] > 0:
        report['recommendations'].append("LOW: Some expected indicators are missing from calculations")
    
    # Print summary
    print("\nSUMMARY:")
    print(f"✓ Total indicators calculated: {report['indicators']['total_calculated']}")
    print(f"{'✓' if not report['confidence_scores']['all_ones_issue'] else '✗'} Confidence scores: {'Properly varied' if not report['confidence_scores']['all_ones_issue'] else 'All showing 1.0 (needs fix)'}")
    print(f"⚠️  High correlation pairs: {report['correlations']['high_correlation_pairs']}")
    
    print("\nRECOMMENDATIONS:")
    for i, rec in enumerate(report['recommendations'], 1):
        print(f"{i}. {rec}")
    
    # Save report
    report_file = f"validation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(report_file, 'w') as f:
        f.write("REGIME SYSTEM VALIDATION REPORT\n")
        f.write("="*80 + "\n")
        f.write(f"Generated: {report['timestamp']}\n\n")
        
        f.write("INDICATOR ANALYSIS:\n")
        f.write(f"  Total Calculated: {report['indicators']['total_calculated']}\n")
        f.write(f"  Valid: {report['indicators']['valid']}\n")
        f.write(f"  Problematic: {report['indicators']['problematic']}\n")
        f.write(f"  Missing Expected: {report['indicators']['missing_expected']}\n\n")
        
        f.write("CONFIDENCE SCORES:\n")
        f.write(f"  All 1.0 Issue: {'Yes' if report['confidence_scores']['all_ones_issue'] else 'No'}\n")
        f.write(f"  Low Variance Issue: {'Yes' if report['confidence_scores']['low_variance_issue'] else 'No'}\n\n")
        
        f.write("CORRELATIONS:\n")
        f.write(f"  High Correlation Pairs (>{report['correlations']['threshold_used']}): {report['correlations']['high_correlation_pairs']}\n\n")
        
        f.write("RECOMMENDATIONS:\n")
        for rec in report['recommendations']:
            f.write(f"  - {rec}\n")
    
    print(f"\n✓ Report saved to: {report_file}")
    
    return report

# Main execution
if __name__ == "__main__":
    print("\nRunning comprehensive validation...")
    
    # Step 1: Validate indicators
    data_with_indicators, validation_results = run_indicator_validation()
    
    if data_with_indicators is not None:
        # Step 2: Check confidence scores
        regimes = check_confidence_scores(data_with_indicators)
        
        # Step 3: Analyze correlations
        corr_matrix, high_corr_pairs = analyze_indicator_correlations(data_with_indicators)
        
        # Step 4: Create report
        report = create_validation_report(validation_results, regimes, high_corr_pairs)
        
        print("\n" + "="*80)
        print("NEXT STEPS:")
        print("="*80)
        print("1. Replace regime_system/core/indicators.py with the full implementation")
        print("2. Re-run this validation to confirm confidence scores are fixed")
        print("3. Remove highly correlated indicators based on the analysis")
        print("4. Proceed with regime distribution validation")
        print("5. Run performance attribution analysis")
    else:
        print("\n✗ Validation failed - please check indicator calculations")
    
    print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

