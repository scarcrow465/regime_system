#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
Strategy Mapping Analysis - Which strategies work best in which regimes?
Tests multiple strategy types across different regime combinations
"""

import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime
from typing import Dict, List, Tuple
import time
from tqdm import tqdm
import pandas_ta as ta

# Add regime_system to path
sys.path.insert(0, r'C:\Users\rs\GitProjects\regime_system\ob_model\v2.0_precloud_reorganization_and_clean')

from core.data_loader import load_csv_data
from core.indicators import calculate_all_indicators
from core.regime_classifier import RollingRegimeClassifier
from backtesting.strategies import EnhancedRegimeStrategyBacktester

def preprocess_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure all required columns for strategies are present.
    Adds RSI, MACD, ATR, ADX, and Bollinger Bands if missing.
    """
    try:
        data = data.copy()
        if 'close' not in data.columns:
            raise ValueError("Input data must contain 'close' column")
        
        # Calculate RSI if missing
        if 'RSI' not in data.columns:
            data['RSI'] = ta.rsi(data['close'], length=14)
        
        # Calculate MACD if missing
        if 'MACD' not in data.columns or 'MACD_signal' not in data.columns:
            macd = ta.macd(data['close'], fast=12, slow=26, signal=9)
            data['MACD'] = macd['MACD_12_26_9']
            data['MACD_signal'] = macd['MACDs_12_26_9']
        
        # Calculate ATR if missing
        if 'ATR' not in data.columns:
            if 'high' in data.columns and 'low' in data.columns:
                data['ATR'] = ta.atr(data['high'], data['low'], data['close'], length=14)
            else:
                data['ATR'] = data['close'].rolling(14).std()  # Fallback
        
        # Calculate ADX if missing
        if 'ADX' not in data.columns:
            if 'high' in data.columns and 'low' in data.columns:
                adx = ta.adx(data['high'], data['low'], data['close'], length=14)
                data['ADX'] = adx['ADX_14']
            else:
                data['ADX'] = pd.Series(0, index=data.index)  # Fallback
        
        # Calculate Bollinger Bands if missing
        if 'BB_lower' not in data.columns:
            bb = ta.bbands(data['close'], length=20, std=2.0)
            data['BB_lower'] = bb['BBL_20_2.0']
            data['BB_middle'] = bb['BBM_20_2.0']
            data['BB_upper'] = bb['BBU_20_2.0']
        
        return data
    except Exception as e:
        raise ValueError(f"Error in preprocessing data: {e}")

print("="*80)
print("REGIME-STRATEGY PERFORMANCE MAPPING")
print("="*80)
print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Load data
data_file = r'combined_NQ_15m_data.csv'
print(f"\nLoading data from: {data_file}")
start_time = time.time()
with tqdm(total=1, desc="Loading Data", ncols=80, mininterval=1) as pbar:
    data = load_csv_data(data_file, timeframe='15min')
    pbar.update(1)
print(f"Loaded {len(data)} rows in {time.time() - start_time:.2f} seconds")

# Use last 100,000 rows for meaningful but manageable analysis
if len(data) > 100000:
    data = data.tail(100000)
    print(f"Using last {len(data)} rows for strategy analysis")

# Calculate indicators and preprocess
print("\nCalculating indicators...")
start_time = time.time()
with tqdm(total=1, desc="Calculating Indicators", ncols=80, mininterval=1) as pbar:
    data_with_indicators = calculate_all_indicators(data, verbose=False)
    data_with_indicators = preprocess_data(data_with_indicators)
    pbar.update(1)
print(f"Indicators calculated and preprocessed in {time.time() - start_time:.2f} seconds")

# Classify regimes
print("\nClassifying regimes...")
start_time = time.time()
classifier = RollingRegimeClassifier(window_hours=36, timeframe='15min')
with tqdm(total=1, desc="Classifying Regimes", ncols=80, mininterval=1) as pbar:
    regimes = classifier.classify_regimes(data_with_indicators, show_progress=True)
    pbar.update(1)
print(f"Regimes classified in {time.time() - start_time:.2f} seconds")

# Initialize backtester
backtester = EnhancedRegimeStrategyBacktester()

# Define strategies to test
strategies = {
    'momentum': 'Trend following - buys strength, sells weakness',
    'mean_reversion': 'Counter-trend - buys oversold, sells overbought',
    'volatility_breakout': 'Trades range expansions',
    'adaptive': 'Switches strategy based on regime'
}

print("\n" + "="*80)
print("TESTING STRATEGIES ACROSS REGIMES")
print("="*80)

# Function to test a specific strategy in specific regime conditions
def test_strategy_in_regime(data, regimes, strategy_type, regime_filter):
    """Test a strategy when specific regime conditions are met"""
    start_time = time.time()
    print(f"\nTesting {strategy_type} in regime {regime_filter}")
    
    # Create mask for when regime conditions are met
    mask = pd.Series(True, index=regimes.index)
    for dim, regime_value in regime_filter.items():
        col_name = f'{dim}_Regime'
        if col_name in regimes.columns:
            mask &= (regimes[col_name] == regime_value)
    
    if mask.sum() < 50:  # Reduced from 100
        print(f"Skipping {strategy_type} in {regime_filter}: Insufficient periods ({mask.sum()})")
        return None, f"Regime {regime_filter} has insufficient periods: {mask.sum()}"
    
    # Run strategy
    if strategy_type == 'momentum':
        # Enhanced momentum strategy
        direction_up = regimes['Direction_Regime'] == 'Up_Trending'
        direction_down = regimes['Direction_Regime'] == 'Down_Trending'
        returns = backtester.momentum_strategy_enhanced(
            data, 
            (direction_up | direction_down) & mask, 
            regimes
        )
    
    elif strategy_type == 'mean_reversion':
        # Mean reversion in sideways markets
        sideways = regimes['Direction_Regime'] == 'Sideways'
        returns = backtester.mean_reversion_strategy_enhanced(
            data, 
            sideways & mask, 
            regimes
        )
    
    elif strategy_type == 'volatility_breakout':
        # Volatility breakout strategy
        high_vol = regimes['Volatility_Regime'].isin(['High_Vol', 'Extreme_Vol'])
        returns = backtester.volatility_breakout_strategy_enhanced(
            data, 
            high_vol & mask, 
            regimes
        )
    
    elif strategy_type == 'adaptive':
        # Use the full adaptive strategy
        returns = backtester.adaptive_regime_strategy_enhanced(data, regimes)
        # Apply mask to returns
        returns = returns * mask
    
    else:
        print(f"Invalid strategy type: {strategy_type}")
        return None, f"Invalid strategy type: {strategy_type}"
    
    # Calculate performance metrics
    if returns is not None and len(returns[returns != 0]) > 10:  # Reduced from 20
        metrics = backtester.calculate_performance_metrics(returns)
        metrics['periods_in_regime'] = mask.sum()
        metrics['pct_time_in_regime'] = mask.sum() / len(mask) * 100
        print(f"Completed {strategy_type} in {regime_filter} in {time.time() - start_time:.2f} seconds")
        return metrics, None
    
    return None, f"No valid returns for {strategy_type} in {regime_filter}"

# Test each strategy in major regime combinations
regime_combinations = [
    # Single dimension focus
    {'Direction': 'Up_Trending'},
    {'Direction': 'Down_Trending'},
    {'Direction': 'Sideways'},
    {'Volatility': 'Low_Vol'},
    {'Volatility': 'High_Vol'},
    {'Volatility': 'Extreme_Vol'},
    {'TrendStrength': 'Strong'},
    {'TrendStrength': 'Weak'},
    
    # Useful combinations
    {'Direction': 'Up_Trending', 'TrendStrength': 'Strong'},
    {'Direction': 'Up_Trending', 'TrendStrength': 'Weak'},
    {'Direction': 'Down_Trending', 'TrendStrength': 'Strong'},
    {'Direction': 'Down_Trending', 'TrendStrength': 'Weak'},
    {'Direction': 'Sideways', 'Volatility': 'Low_Vol'},
    {'Direction': 'Sideways', 'Volatility': 'High_Vol'},
    {'TrendStrength': 'Strong', 'Volatility': 'Low_Vol'},
    {'TrendStrength': 'Strong', 'Volatility': 'High_Vol'},
]

# Store results
results = []

print("\nTesting strategies in different regime combinations...")
print("(This may take a few minutes...)\n")

# Calculate total iterations for progress bar
total_iterations = len(regime_combinations) * len(strategies)
start_time = time.time()
with tqdm(total=total_iterations, desc="Testing Strategies", ncols=80, mininterval=1) as pbar:
    for regime_filter in regime_combinations:
        regime_name = '_'.join([f"{k}={v}" for k, v in regime_filter.items()])
        
        for strategy_name, strategy_desc in strategies.items():
            metrics, error_msg = test_strategy_in_regime(data_with_indicators, regimes, 
                                                       strategy_name, regime_filter)
            
            if metrics is not None:
                results.append({
                    'regime': regime_name,
                    'strategy': strategy_name,
                    'sharpe_ratio': metrics['sharpe_ratio'],
                    'total_return': metrics['total_return'],
                    'max_drawdown': metrics['max_drawdown'],
                    'win_rate': metrics['win_rate'],
                    'periods_in_regime': metrics['periods_in_regime'],
                    'pct_time_in_regime': metrics['pct_time_in_regime']
                })
            else:
                print(error_msg)
            pbar.update(1)
print(f"Strategy testing completed in {time.time() - start_time:.2f} seconds")

# Convert to DataFrame for analysis
results_df = pd.DataFrame(results)

# Find best strategy for each regime
print("="*80)
print("BEST STRATEGY BY REGIME")
print("="*80)

best_by_regime = results_df.loc[results_df.groupby('regime')['sharpe_ratio'].idxmax()]

for _, row in best_by_regime.iterrows():
    print(f"\n{row['regime']}:")
    print(f"  Best Strategy: {row['strategy'].upper()}")
    print(f"  Sharpe Ratio: {row['sharpe_ratio']:.3f}")
    print(f"  Total Return: {row['total_return']:.2%}")
    print(f"  Max Drawdown: {row['max_drawdown']:.2%}")
    print(f"  Win Rate: {row['win_rate']:.2%}")
    print(f"  Time in Regime: {row['pct_time_in_regime']:.1f}%")

# Strategy performance summary
print("\n" + "="*80)
print("STRATEGY PERFORMANCE SUMMARY")
print("="*80)

strategy_summary = results_df.groupby('strategy').agg({
    'sharpe_ratio': ['mean', 'std', 'min', 'max'],
    'total_return': 'mean',
    'win_rate': 'mean'
}).round(3)

print("\nAverage performance across all regimes:")
print(strategy_summary)

# Key insights
print("\n" + "="*80)
print("KEY INSIGHTS & RECOMMENDATIONS")
print("="*80)

# Find which regimes favor each strategy
for strategy in strategies.keys():
    strategy_results = results_df[results_df['strategy'] == strategy]
    if len(strategy_results) > 0:
        best_regimes = strategy_results.nlargest(3, 'sharpe_ratio')
        worst_regimes = strategy_results.nsmallest(3, 'sharpe_ratio')
        
        print(f"\n{strategy.upper()} Strategy:")
        print("  Best regimes:")
        for _, row in best_regimes.iterrows():
            print(f"    - {row['regime']} (Sharpe: {row['sharpe_ratio']:.3f})")
        print("  Worst regimes:")
        for _, row in worst_regimes.iterrows():
            print(f"    - {row['regime']} (Sharpe: {row['sharpe_ratio']:.3f})")

# Identify regime combinations that are particularly profitable
print("\n" + "="*80)
print("MOST PROFITABLE REGIME COMBINATIONS")
print("="*80)

profitable_regimes = results_df[results_df['sharpe_ratio'] > 0.5].groupby('regime').agg({
    'sharpe_ratio': 'max',
    'strategy': lambda x: x.iloc[x.values.argmax()]
}).sort_values('sharpe_ratio', ascending=False).head(10)

print("\nTop 10 profitable regime-strategy combinations:")
for regime, row in profitable_regimes.iterrows():
    print(f"  {regime} → {row['strategy']} (Sharpe: {row['sharpe_ratio']:.3f})")

# Identify challenging regimes
challenging_regimes = results_df.groupby('regime')['sharpe_ratio'].max()
challenging_regimes = challenging_regimes[challenging_regimes < 0].sort_values()

if len(challenging_regimes) > 0:
    print("\n" + "="*80)
    print("CHALLENGING REGIMES (Avoid or use defensive strategies)")
    print("="*80)
    for regime, best_sharpe in challenging_regimes.items():
        print(f"  {regime} (Best Sharpe: {best_sharpe:.3f})")

# Save detailed results
output_file = f"regime_strategy_mapping_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
results_df.to_csv(output_file, index=False)
print(f"\n✓ Detailed results saved to: {output_file}")

# Create strategy allocation rules
print("\n" + "="*80)
print("SUGGESTED STRATEGY ALLOCATION RULES")
print("="*80)

print("\n1. PRIMARY RULES (Single dimension):")
print("   - Direction = Up_Trending → MOMENTUM")
print("   - Direction = Down_Trending → MOMENTUM (short)")
print("   - Direction = Sideways → MEAN REVERSION")
print("   - Volatility = High/Extreme → VOLATILITY BREAKOUT")
print("   - Volatility = Low → MEAN REVERSION")

print("\n2. ENHANCED RULES (Multi-dimensional):")
print("   - Strong Trend + Low Volatility → MOMENTUM (high confidence)")
print("   - Weak Trend + High Volatility → REDUCE POSITION SIZE")
print("   - Sideways + Low Volatility → MEAN REVERSION (high confidence)")
print("   - Any + Extreme Volatility → VOLATILITY BREAKOUT or CASH")

print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

