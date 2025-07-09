#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
Example: Using Daily Regime for 1-Day Holding Strategy
Shows how regime alignment can improve your edge
"""

import pandas as pd
import numpy as np

def regime_aware_trading_strategy(data: pd.DataFrame, regime_data: pd.DataFrame) -> pd.DataFrame:
    """
    Example strategy using daily regime to guide 1-day holds
    
    Strategy Rules:
    1. Only trade in favorable regimes
    2. Long only in Uptrends
    3. Short only in Downtrends  
    4. Reduce size in weak trends
    5. Avoid volatile regimes
    """
    
    results = pd.DataFrame(index=data.index)
    results['signal'] = 0
    results['position_size'] = 0
    
    for i in range(1, len(data)):
        # Get current regime
        direction = regime_data['direction_regime'].iloc[i]
        strength = regime_data['strength_regime'].iloc[i]
        volatility = regime_data['volatility_regime'].iloc[i]
        character = regime_data['character_regime'].iloc[i]
        confidence = regime_data['regime_confidence'].iloc[i]
        
        # Base position sizing
        base_size = 1.0
        
        # Regime-based rules
        if direction == 'Uptrend' and character == 'Trending':
            # Long signal
            results.loc[results.index[i], 'signal'] = 1
            
            # Size based on strength
            if strength == 'Strong':
                size = base_size * 1.5  # Increase size in strong trends
            elif strength == 'Moderate':
                size = base_size * 1.0
            else:  # Weak
                size = base_size * 0.5  # Reduce in weak trends
                
        elif direction == 'Downtrend' and character == 'Trending':
            # Short signal
            results.loc[results.index[i], 'signal'] = -1
            
            # Size based on strength (more conservative on shorts)
            if strength == 'Strong':
                size = base_size * 1.0
            elif strength == 'Moderate':
                size = base_size * 0.7
            else:  # Weak
                size = base_size * 0.3
                
        else:
            # No signal in sideways or transitioning markets
            results.loc[results.index[i], 'signal'] = 0
            size = 0
        
        # Volatility adjustment
        if volatility == 'Extreme':
            size *= 0.5  # Half size in extreme volatility
        elif volatility == 'High':
            size *= 0.7
        elif volatility == 'Low':
            size *= 1.2  # Increase in low volatility
            
        # Confidence adjustment
        size *= confidence  # Scale by regime confidence
        
        # Final position size
        results.loc[results.index[i], 'position_size'] = size
    
    # Calculate returns (1-day holding)
    results['returns'] = results['signal'].shift(1) * data['returns']
    results['sized_returns'] = results['position_size'].shift(1) * data['returns']
    
    return results


# Example usage in the test script:
print("\nTesting Regime-Aware Strategy...")

# Simple buy-and-hold benchmark
data_with_indicators['returns'] = data_with_indicators['close'].pct_change()
benchmark_returns = data_with_indicators['returns'].fillna(0)
benchmark_cumulative = (1 + benchmark_returns).cumprod()

# Regime-aware strategy
strategy_results = regime_aware_trading_strategy(data_with_indicators, regime_data)
strategy_cumulative = (1 + strategy_results['sized_returns'].fillna(0)).cumprod()

# Compare performance
benchmark_sharpe = benchmark_returns.mean() / benchmark_returns.std() * np.sqrt(252)
strategy_sharpe = strategy_results['sized_returns'].mean() / strategy_results['sized_returns'].std() * np.sqrt(252)

print(f"\nPerformance Comparison (Last 5 Years):")
print(f"Buy & Hold Sharpe: {benchmark_sharpe:.3f}")
print(f"Regime Strategy Sharpe: {strategy_sharpe:.3f}")
print(f"Improvement: {((strategy_sharpe / benchmark_sharpe) - 1) * 100:.1f}%")

# Show when strategy is active
active_days = (strategy_results['signal'] != 0).sum()
total_days = len(strategy_results)
print(f"\nStrategy Active: {active_days}/{total_days} days ({active_days/total_days*100:.1f}%)")

# Average position size when active
avg_position = strategy_results[strategy_results['signal'] != 0]['position_size'].mean()
print(f"Average Position Size When Active: {avg_position:.2f}x")

