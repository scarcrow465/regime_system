#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
Script to validate indicators using REAL market data
Phase 1 - Testing with actual futures data
"""

import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime

# Add regime_system to path
sys.path.insert(0, r'C:\Users\rs\GitProjects\regime_system\ob_model\v2.0_precloud_reorganization_and_clean')

# Import required modules
from validation.indicator_analysis import IndicatorValidator
from core.data_loader import load_csv_data
from core.indicators import calculate_all_indicators
from core.regime_classifier import RollingRegimeClassifier

print("="*80)
print("PHASE 1: VALIDATION WITH REAL MARKET DATA")
print("="*80)
print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

def run_indicator_validation_real_data():
    """Run validation with real market data"""
    
    # Path to your actual data file
    # Update this path to your actual CSV file location
    data_files = [
        r"C:\Users\rs\GitProjects\regime_system\ob_model\v2.0_precloud_reorganization_and_clean\combined_NQ_15m_data.csv"
    ]
    
    # Find the first existing file
    data_file = None
    for file_path in data_files:
        if os.path.exists(file_path):
            data_file = file_path
            break
    
    if data_file is None:
        print("\n⚠️ No data file found! Please update the path to your actual data file.")
        print("Looking for files in these locations:")
        for fp in data_files:
            print(f"  - {fp}")
        return None, None
    
    print(f"\n✓ Found data file: {data_file}")
    
    # Load the data
    print("\nLoading real market data...")
    try:
        data = load_csv_data(data_file, timeframe='15min')
        print(f"Loaded {len(data)} rows of data")
        print(f"Date range: {data.index[0]} to {data.index[-1]}")
        
        # Use a subset for faster testing (last 10,000 rows)
        if len(data) > 10000:
            print(f"\nUsing last 10,000 rows for validation (full dataset has {len(data)} rows)")
            data = data.tail(10000)
        
        print(f"Data shape: {data.shape}")
        print(f"Columns: {list(data.columns)}")
        print(f"Data types: {data.dtypes.value_counts()}")
        
        # Show any non-numeric columns
        non_numeric = data.select_dtypes(exclude=[np.number]).columns.tolist()
        if non_numeric:
            print(f"Non-numeric columns found: {non_numeric}")
        
        # Check for required columns
        required = ['open', 'high', 'low', 'close']
        missing = [col for col in required if col not in data.columns]
        if missing:
            print(f"\n⚠️ Missing required columns: {missing}")
            return None, None
            
    except Exception as e:
        print(f"\n✗ Error loading data: {e}")
        return None, None
    
    # Calculate indicators
    print("\nCalculating indicators on real data...")
    try:
        data_with_indicators = calculate_all_indicators(data, verbose=True)
        indicator_count = len(data_with_indicators.columns) - len(data.columns)
        print(f"\n✓ Successfully calculated {indicator_count} indicators")
    except Exception as e:
        print(f"\n✗ Error calculating indicators: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame(), {'error': str(e)}
    
    # Validate indicators
    print("\nValidating indicators...")
    validator = IndicatorValidator()
    validation_results = validator.validate_indicators(data_with_indicators)
    
    print("\nValidation Results:")
    print(f"  Overall missing %: {validation_results['missing_data']['overall_missing_pct']:.2f}%")
    print(f"  Quality score: {validation_results['data_quality']['quality_score']:.2f}")
    print(f"  Range issues: {len(validation_results['indicator_ranges']['out_of_range'])}")
    print(f"  Overall score: {validation_results['overall_score']:.2f}")
    
    # Show indicator count by type
    numeric_indicators = data_with_indicators.select_dtypes(include=[np.number]).columns
    non_base_numeric = [col for col in numeric_indicators if col not in ['open', 'high', 'low', 'close', 'volume']]
    print(f"\nIndicator Summary:")
    print(f"  Numeric indicators: {len(non_base_numeric)}")
    print(f"  Total columns: {len(data_with_indicators.columns)}")
    
    return data_with_indicators, validation_results

def check_confidence_scores_real_data(data_with_indicators):
    """Check confidence scores with real market data"""
    print("\n" + "="*60)
    print("REGIME CLASSIFICATION WITH REAL DATA")
    print("="*60)
    
    # Create classifier with 36-hour window (optimal for 15-min NQ)
    classifier = RollingRegimeClassifier(window_hours=36, timeframe='15min')
    
    # Classify regimes
    print("Classifying regimes...")
    regimes = classifier.classify_regimes(data_with_indicators, show_progress=True)
    
    # Get regime statistics
    stats = classifier.get_regime_statistics(regimes)
    
    # Print regime distributions
    print("\nREGIME DISTRIBUTIONS:")
    for dimension, dim_stats in stats.items():
        if dimension != 'Composite' and 'percentages' in dim_stats:
            print(f"\n{dimension} Dimension:")
            for regime, pct in sorted(dim_stats['percentages'].items()):
                if regime != 'Undefined':
                    print(f"  {regime}: {pct:.1f}%")
    
    # Check confidence scores
    print("\n" + "="*60)
    print("CONFIDENCE SCORE ANALYSIS:")
    print("="*60)
    
    confidence_summary = {}
    
    for dim in ['Direction', 'TrendStrength', 'Velocity', 'Volatility', 'Microstructure']:
        conf_col = f'{dim}_Confidence'
        if conf_col in regimes.columns:
            conf_values = regimes[conf_col].dropna()
            
            # Calculate statistics
            stats_dict = {
                'mean': conf_values.mean(),
                'std': conf_values.std(),
                'min': conf_values.min(),
                'max': conf_values.max(),
                'unique': len(conf_values.unique()),
                'zeros': (conf_values == 0).sum(),
                'ones': (conf_values == 1).sum()
            }
            
            confidence_summary[dim] = stats_dict
            
            print(f"\n{dim} Confidence:")
            print(f"  Mean: {stats_dict['mean']:.3f}")
            print(f"  Std: {stats_dict['std']:.3f}")
            print(f"  Min: {stats_dict['min']:.3f}")
            print(f"  Max: {stats_dict['max']:.3f}")
            print(f"  Unique values: {stats_dict['unique']}")
            
            # Check for issues
            if stats_dict['mean'] == 0:
                print(f"  ⚠️ WARNING: All zeros - no confident classifications!")
            elif stats_dict['mean'] == 1:
                print(f"  ⚠️ WARNING: All ones - perfect agreement (suspicious)!")
            elif stats_dict['std'] < 0.01:
                print(f"  ⚠️ WARNING: Very low variance!")
            else:
                print(f"  ✓ Confidence scores show proper variation")
            
            # Show distribution
            if stats_dict['unique'] <= 5:
                print(f"  Distribution: {conf_values.value_counts().sort_index().to_dict()}")
    
    return regimes, confidence_summary

def analyze_correlations_focused(data_with_indicators):
    """Analyze correlations with focus on problematic pairs"""
    print("\n" + "="*60)
    print("CORRELATION ANALYSIS")
    print("="*60)
    
    # Select only indicator columns (numeric only)
    base_cols = ['open', 'high', 'low', 'close', 'volume', 'Symbol', 'Date']
    numeric_cols = data_with_indicators.select_dtypes(include=[np.number]).columns.tolist()
    indicator_cols = [col for col in numeric_cols if col not in base_cols]
    
    # Additional check for any string columns that might have slipped through
    indicator_cols = [col for col in indicator_cols if data_with_indicators[col].dtype in ['float64', 'int64', 'float32', 'int32']]
    
    print(f"Analyzing correlations for {len(indicator_cols)} numeric indicators...")
    
    # Debug: Show first few indicators
    print(f"First 5 indicators: {indicator_cols[:5]}")
    
    try:
        corr_matrix = data_with_indicators[indicator_cols].corr()
    except Exception as e:
        print(f"\n⚠️ Error calculating correlations: {e}")
        print("Checking for problematic columns...")
        for col in indicator_cols:
            try:
                test = data_with_indicators[col].astype(float)
            except:
                print(f"  Problem column: {col} - {data_with_indicators[col].dtype}")
        return None, [], []
    
    # Find perfect correlations (>0.99)
    perfect_corr_pairs = []
    high_corr_pairs = []
    seen_pairs = set()  # Track unique pairs
    
    for i in range(len(indicator_cols)):
        for j in range(i+1, len(indicator_cols)):
            pair = tuple(sorted([indicator_cols[i], indicator_cols[j]]))  # Ensure consistent ordering
            if pair not in seen_pairs:
                corr_value = abs(corr_matrix.iloc[i, j])
                if corr_value > 0.99 and not pd.isna(corr_value):
                    perfect_corr_pairs.append({
                        'indicator1': indicator_cols[i],
                        'indicator2': indicator_cols[j],
                        'correlation': corr_value
                    })
                elif corr_value > 0.90 and not pd.isna(corr_value):
                    high_corr_pairs.append({
                        'indicator1': indicator_cols[i],
                        'indicator2': indicator_cols[j],
                        'correlation': corr_value
                    })
                seen_pairs.add(pair)
    
    print(f"\nPerfect correlations (>0.99): {len(perfect_corr_pairs)}")
    print(f"High correlations (0.90-0.99): {len(high_corr_pairs)}")
    
    # Show perfect correlations
    if perfect_corr_pairs:
        print("\nPERFECT CORRELATIONS (Consider removing one from each pair):")
        for pair in perfect_corr_pairs[:10]:
            print(f"  - {pair['indicator1']} = {pair['indicator2']} ({pair['correlation']:.4f})")
    
    return corr_matrix, perfect_corr_pairs, high_corr_pairs

# Main execution
if __name__ == "__main__":
    print("\nRunning validation with REAL market data...")
    
    # Step 1: Validate indicators with real data
    data_with_indicators, validation_results = run_indicator_validation_real_data()
    
    if data_with_indicators is not None and not data_with_indicators.empty:
        # Step 2: Check confidence scores
        regimes, confidence_summary = check_confidence_scores_real_data(data_with_indicators)
        
        # Step 3: Analyze correlations
        corr_matrix, perfect_corr_pairs, high_corr_pairs = analyze_correlations_focused(data_with_indicators)
        
        # Summary
        print("\n" + "="*80)
        print("SUMMARY - REAL DATA VALIDATION")
        print("="*80)
        
        # Check if Direction and Velocity improved
        if 'Direction' in confidence_summary and confidence_summary['Direction']['mean'] > 0:
            print("✓ Direction confidence improved with real data!")
        else:
            print("✗ Direction confidence still showing issues")
            
        if 'Velocity' in confidence_summary and confidence_summary['Velocity']['mean'] > 0:
            print("✓ Velocity confidence improved with real data!")
        else:
            print("✗ Velocity confidence still showing issues")
        
        if corr_matrix is not None:
            print(f"\n⚠️ Found {len(perfect_corr_pairs)} perfect correlations to remove")
            print(f"⚠️ Found {len(high_corr_pairs)} high correlations to review")
        else:
            print("\n⚠️ Correlation analysis failed - check for non-numeric columns")
        
        print("\nNEXT STEPS:")
        print("1. Review confidence score distributions")
        print("2. Remove perfectly correlated indicators")
        print("3. Test on different time periods")
        print("4. Proceed with regime distribution validation")
        
    else:
        print("\n✗ Validation failed - please check data file path and format")
    
    print(f"\n\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

