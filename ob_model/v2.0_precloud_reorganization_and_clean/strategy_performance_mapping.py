#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""
Strategy performance mapping across market regimes
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
import time
from datetime import datetime
import os
import sys

sys.path.insert(0, r'C:\Users\rs\GitProjects\regime_system\ob_model\v2.0_precloud_reorganization_and_clean')

from config.settings import INDICATOR_WEIGHTS
from backtesting.strategies import EnhancedRegimeStrategyBacktester, compare_strategies

logger = logging.getLogger(__name__)

def calculate_all_indicators(data: pd.DataFrame) -> pd.DataFrame:
    try:
        import pandas_ta as ta
        data = data.copy()
        # Attempt to compute all indicators listed in INDICATOR_WEIGHTS
        for indicator in INDICATOR_WEIGHTS.keys():
            if indicator == 'RSI':
                data['RSI'] = ta.rsi(data['close'], length=14)
            elif indicator == 'MACD':
                data['MACD'], data['MACD_signal'], _ = ta.macd(data['close'], fast=12, slow=26, signal=9)
            elif indicator == 'ATR':
                data['ATR'] = ta.atr(data['high'], data['low'], data['close'], length=14)
            elif indicator == 'ADX':
                data['ADX'] = ta.adx(data['high'], data['low'], data['close'], length=14)['ADX_14']
            elif indicator == 'BBANDS':
                bb = ta.bbands(data['close'], length=20, std=2.0)
                data['BB_lower'] = bb['BBL_20_2.0']
                data['BB_middle'] = bb['BBM_20_2.0']
                data['BB_upper'] = bb['BBU_20_2.0']
            # Add other indicators from INDICATOR_WEIGHTS (e.g., SMA, EMA, etc.)
            # This likely includes all 87 indicators, but some may fail or overwrite columns
        return data
    except Exception as e:
        logger.error(f"Error calculating indicators: {e}")
        return data

def identify_regimes(data: pd.DataFrame) -> pd.DataFrame:
    try:
        regimes = pd.DataFrame(index=data.index)
        # Simplified regime identification for example
        if 'close' in data.columns:
            regimes['Direction_Regime'] = np.where(
                data['close'].pct_change(20).rolling(20).mean() > 0, 'Up_Trending',
                np.where(data['close'].pct_change(20).rolling(20).mean() < 0, 'Down_Trending', 'Sideways')
            )
            regimes['TrendStrength_Regime'] = np.where(
                data['ADX'] > 25 if 'ADX' in data.columns else False, 'Strong', 'Weak'
            )
            regimes['Volatility_Regime'] = np.where(
                data['ATR'] > data['ATR'].rolling(20).mean() * 1.5 if 'ATR' in data.columns else False, 'High_Vol',
                np.where(data['ATR'] < data['ATR'].rolling(20).mean() * 0.5, 'Low_Vol', 'Medium_Vol')
            )
            regimes['Direction_Confidence'] = 0.5  # Placeholder
            regimes['Volatility_Confidence'] = 0.5  # Placeholder
        else:
            logger.warning("Missing 'close' column for regime identification")
            regimes['Direction_Regime'] = 'Undefined'
            regimes['TrendStrength_Regime'] = 'Undefined'
            regimes['Volatility_Regime'] = 'Undefined'
            regimes['Direction_Confidence'] = 0.5
            regimes['Volatility_Confidence'] = 0.5
        return regimes
    except Exception as e:
        logger.error(f"Error identifying regimes: {e}")
        return pd.DataFrame(index=data.index)

def test_strategy_in_regime(data: pd.DataFrame, regimes: pd.DataFrame, strategy_name: str, 
                           regime_name: str, regime_mask: pd.Series) -> Tuple[Optional[Dict], Optional[str]]:
    try:
        backtester = EnhancedRegimeStrategyBacktester()
        strategy_func = {
            'momentum': backtester.momentum_strategy_enhanced,
            'mean_reversion': backtester.mean_reversion_strategy_enhanced,
            'volatility_breakout': backtester.volatility_breakout_strategy_enhanced,
            'adaptive': backtester.adaptive_regime_strategy_enhanced
        }.get(strategy_name.lower())
        
        if strategy_func is None:
            return None, f"Unknown strategy: {strategy_name}"
        
        logger.info(f"Testing {strategy_name} in regime {regime_name}")
        if strategy_name.lower() == 'adaptive':
            returns = strategy_func(data, regimes)
        else:
            returns = strategy_func(data, regime_mask, regimes)
        
        if returns is None or returns.empty or returns.isna().all():
            return None, f"No valid returns for {strategy_name} in {regime_name}"
        
        if regime_mask.sum() < 100:
            return None, f"Regime {regime_name} has insufficient periods: {regime_mask.sum()}"
        
        if len(returns[returns != 0]) <= 20:
            return None, f"No valid returns for {strategy_name} in {regime_name}"
        
        metrics = backtester.calculate_performance_metrics(returns[regime_mask])
        metrics['periods_in_regime'] = regime_mask.sum()
        metrics['pct_time_in_regime'] = regime_mask.sum() / len(regime_mask) * 100
        return metrics, None
    except Exception as e:
        logger.error(f"Error testing {strategy_name} in regime {regime_name}: {e}")
        return None, str(e)

def map_strategy_performance(data: pd.DataFrame) -> pd.DataFrame:
    start_time = time.time()
    logger.info("Starting strategy performance mapping")
    
    # Calculate indicators
    data = calculate_all_indicators(data)
    
    # Classify regimes
    regimes = identify_regimes(data)
    
    # Define regime combinations
    regime_combinations = [
        {'Direction': 'Up_Trending'},
        {'Direction': 'Down_Trending'},
        {'Direction': 'Sideways'},
        {'Volatility': 'Low_Vol'},
        {'Volatility': 'High_Vol'},
        {'Volatility': 'Extreme_Vol'},
        {'TrendStrength': 'Strong'},
        {'TrendStrength': 'Weak'},
        {'Direction': 'Up_Trending', 'TrendStrength': 'Strong'},
        {'Direction': 'Up_Trending', 'TrendStrength': 'Weak'},
        {'Direction': 'Down_Trending', 'TrendStrength': 'Strong'},
        {'Direction': 'Down_Trending', 'TrendStrength': 'Weak'},
        {'Direction': 'Sideways', 'Volatility': 'High_Vol'},
        {'TrendStrength': 'Strong', 'Volatility': 'High_Vol'},
        {'TrendStrength': 'Strong', 'Volatility': 'Low_Vol'}
    ]
    
    strategies = ['momentum', 'mean_reversion', 'volatility_breakout', 'adaptive']
    results = []
    
    for regime in regime_combinations:
        regime_name = '_'.join(f"{k}={v}" for k, v in regime.items())
        regime_mask = pd.Series(True, index=data.index)
        for key, value in regime.items():
            regime_mask = regime_mask & (regimes[f"{key}_Regime"] == value)
        
        for strategy in strategies:
            metrics, error = test_strategy_in_regime(data, regimes, strategy, regime_name, regime_mask)
            if error:
                logger.warning(error)
                continue
            result = {
                'Regime': regime_name,
                'Strategy': strategy,
                'Sharpe_Ratio': metrics['sharpe_ratio'],
                'Total_Return': metrics['total_return'],
                'Max_Drawdown': metrics['max_drawdown'],
                'Win_Rate': metrics['win_rate'],
                'Time_in_Regime': metrics['pct_time_in_regime'] / 100
            }
            results.append(result)
    
    results_df = pd.DataFrame(results)
    
    # Generate summary
    best_strategies = results_df.groupby('Regime').apply(
        lambda x: x.loc[x['Sharpe_Ratio'].idxmax()]
    ).reset_index(drop=True)
    
    # Overall strategy performance
    performance_summary = compare_strategies(data, regimes)
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = f'regime_strategy_mapping_{timestamp}.csv'
    results_df.to_csv(output_file, index=False)
    
    # Log results
    logger.info(f"Strategy testing completed in {time.time() - start_time:.2f} seconds")
    logger.info("="*80)
    logger.info("BEST STRATEGY BY REGIME")
    logger.info("="*80)
    for _, row in best_strategies.iterrows():
        logger.info(f"\n{row['Regime']}:")
        logger.info(f"  Best Strategy: {row['Strategy'].upper()}")
        logger.info(f"  Sharpe Ratio: {row['Sharpe_Ratio']:.3f}")
        logger.info(f"  Total Return: {row['Total_Return']*100:.2f}%")
        logger.info(f"  Max Drawdown: {row['Max_Drawdown']*100:.2f}%")
        logger.info(f"  Win Rate: {row['Win_Rate']*100:.2f}%")
        logger.info(f"  Time in Regime: {row['Time_in_Regime']*100:.1f}%")
    
    logger.info("\n" + "="*80)
    logger.info("STRATEGY PERFORMANCE SUMMARY")
    logger.info("="*80)
    logger.info("\nAverage performance across all regimes:")
    logger.info(performance_summary.to_string())
    
    logger.info("\n" + "="*80)
    logger.info("KEY INSIGHTS & RECOMMENDATIONS")
    logger.info("="*80)
    
    for strategy in strategies:
        strategy_results = results_df[results_df['Strategy'] == strategy]
        if len(strategy_results) > 0:
            best_regimes = strategy_results.nlargest(3, 'Sharpe_Ratio')
            worst_regimes = strategy_results.nsmallest(3, 'Sharpe_Ratio')
            logger.info(f"\n{strategy.upper()} Strategy:")
            logger.info("  Best regimes:")
            for _, row in best_regimes.iterrows():
                logger.info(f"    - {row['Regime']} (Sharpe: {row['Sharpe_Ratio']:.3f})")
            logger.info("  Worst regimes:")
            for _, row in worst_regimes.iterrows():
                logger.info(f"    - {row['Regime']} (Sharpe: {row['Sharpe_Ratio']:.3f})")
    
    logger.info("\n" + "="*80)
    logger.info("MOST PROFITABLE REGIME COMBINATIONS")
    logger.info("="*80)
    profitable_regimes = results_df[results_df['Sharpe_Ratio'] > 0.5].groupby('Regime').agg({
        'Sharpe_Ratio': 'max',
        'Strategy': lambda x: x.iloc[x.values.argmax()]
    }).sort_values('Sharpe_Ratio', ascending=False).head(10)
    logger.info("\nTop 10 profitable regime-strategy combinations:")
    for regime, row in profitable_regimes.iterrows():
        logger.info(f"  {regime} → {row['Strategy']} (Sharpe: {row['Sharpe_Ratio']:.3f})")
    
    challenging_regimes = results_df.groupby('Regime')['Sharpe_Ratio'].max()
    challenging_regimes = challenging_regimes[challenging_regimes < 0].sort_values()
    if len(challenging_regimes) > 0:
        logger.info("\n" + "="*80)
        logger.info("CHALLENGING REGIMES (Avoid or use defensive strategies)")
        logger.info("="*80)
        for regime, best_sharpe in challenging_regimes.items():
            logger.info(f"  {regime} (Best Sharpe: {best_sharpe:.3f})")
    
    logger.info(f"\n✓ Detailed results saved to: {output_file}")
    
    logger.info("\n" + "="*80)
    logger.info("SUGGESTED STRATEGY ALLOCATION RULES")
    logger.info("="*80)
    logger.info("\n1. PRIMARY RULES (Single dimension):")
    logger.info("   - Direction = Up_Trending → MOMENTUM")
    logger.info("   - Direction = Down_Trending → MOMENTUM (short)")
    logger.info("   - Direction = Sideways → MEAN REVERSION")
    logger.info("   - Volatility = High/Extreme → VOLATILITY BREAKOUT")
    logger.info("   - Volatility = Low → MEAN REVERSION")
    logger.info("\n2. ENHANCED RULES (Multi-dimensional):")
    logger.info("   - Strong Trend + Low Volatility → MOMENTUM (high confidence)")
    logger.info("   - Weak Trend + High Volatility → REDUCE POSITION SIZE")
    logger.info("   - Sideways + Low Volatility → MEAN REVERSION (high confidence)")
    logger.info("   - Any + Extreme Volatility → VOLATILITY BREAKOUT or CASH")
    
    logger.info(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return results_df

