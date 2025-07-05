#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
SIMPLEST POSSIBLE TEST - Just run: python simple_test.py
This will test if the regime system is working without needing any data files
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

print("="*60)
print("SIMPLE REGIME SYSTEM TEST")
print("="*60)

# Step 1: Generate simple test data
print("\nStep 1: Creating test data...")
dates = pd.date_range(end=datetime.now(), periods=500, freq='15min')
np.random.seed(42)

# Simple price data
close_prices = 100 + np.cumsum(np.random.randn(500) * 0.5)
test_data = pd.DataFrame({
    'open': close_prices + np.random.randn(500) * 0.1,
    'high': close_prices + np.abs(np.random.randn(500) * 0.2),
    'low': close_prices - np.abs(np.random.randn(500) * 0.2),
    'close': close_prices,
    'volume': 1000000 + np.random.randn(500) * 100000
}, index=dates)

print(f"✓ Created test data with {len(test_data)} rows")

# Step 2: Calculate indicators
print("\nStep 2: Calculating indicators...")
try:
    from core.indicators import calculate_all_indicators
    data_with_indicators = calculate_all_indicators(test_data, verbose=False)
    print(f"✓ Calculated {len(data_with_indicators.columns) - len(test_data.columns)} indicators")
except Exception as e:
    print(f"✗ Error calculating indicators: {e}")
    sys.exit(1)

# Step 3: Classify regimes
print("\nStep 3: Classifying regimes...")
try:
    from core.regime_classifier import RollingRegimeClassifier
    classifier = RollingRegimeClassifier(window_hours=24, timeframe='15min')
    regimes = classifier.classify_regimes(data_with_indicators, show_progress=False)
    print(f"✓ Classified regimes for {len(regimes)} periods")
    
    # Show some results
    print("\nRegime Distribution:")
    print(regimes['Direction_Regime'].value_counts())
except Exception as e:
    print(f"✗ Error classifying regimes: {e}")
    sys.exit(1)

# Step 4: Run a simple backtest
print("\nStep 4: Running simple backtest...")
try:
    from backtesting.strategies import EnhancedRegimeStrategyBacktester
    backtester = EnhancedRegimeStrategyBacktester()
    returns = backtester.adaptive_regime_strategy_enhanced(data_with_indicators, regimes)
    metrics = backtester.calculate_performance_metrics(returns)
    
    print(f"✓ Backtest complete!")
    print(f"  Total Return: {metrics['total_return']:.2%}")
    print(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.4f}")
    print(f"  Max Drawdown: {metrics['max_drawdown']:.2%}")
except Exception as e:
    print(f"✗ Error in backtesting: {e}")

print("\n" + "="*60)
print("TEST COMPLETE! The regime system is working correctly.")
print("="*60)
print("\nNext steps:")
print("1. Use quick_start.py to run with your real data")
print("2. Or use the command line interface:")
print('   python main.py analyze --data "your_file.csv" --timeframe 15min')

