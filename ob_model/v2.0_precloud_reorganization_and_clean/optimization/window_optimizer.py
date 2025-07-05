# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
"""
Window optimization module
Finds optimal rolling window size for regime classification
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import (
    DEFAULT_WINDOW_HOURS, TIMEFRAMES, OPTIMIZATION_ITERATIONS
)
from core.regime_classifier import RollingRegimeClassifier
from optimization.multi_objective import MultiObjectiveRegimeOptimizer

logger = logging.getLogger(__name__)

# =============================================================================
# WINDOW SIZE OPTIMIZER
# =============================================================================

def evaluate_window_size(data: pd.DataFrame,
                        window_hours: float,
                        timeframe: str,
                        iterations: int = 20) -> Dict[str, float]:
    """
    Evaluate performance for a specific window size
    
    Args:
        data: DataFrame with indicators
        window_hours: Window size to test
        timeframe: Data timeframe
        iterations: Optimization iterations
        
    Returns:
        Dictionary with performance metrics
    """
    try:
        logger.info(f"Evaluating window size: {window_hours} hours")
        
        # Create classifier with specific window
        classifier = RollingRegimeClassifier(
            window_hours=window_hours,
            timeframe=timeframe
        )
        
        # Run optimization
        optimizer = MultiObjectiveRegimeOptimizer(classifier, data)
        results = optimizer.optimize_regime_thresholds(
            method='differential_evolution',
            max_iterations=iterations
        )
        
        return {
            'window_hours': window_hours,
            'sharpe_ratio': results.sharpe_ratio,
            'max_drawdown': results.max_drawdown,
            'total_return': results.total_return,
            'regime_persistence': results.regime_persistence,
            'win_rate': results.win_rate,
            'calmar_ratio': results.calmar_ratio,
            'sortino_ratio': results.sortino_ratio,
            'score': results.best_score,
            'best_params': results.best_params
        }
        
    except Exception as e:
        logger.error(f"Error evaluating window size {window_hours}: {e}")
        return {
            'window_hours': window_hours,
            'sharpe_ratio': -99,
            'max_drawdown': -0.99,
            'total_return': -0.99,
            'regime_persistence': 0,
            'win_rate': 0,
            'calmar_ratio': -99,
            'sortino_ratio': -99,
            'score': -99,
            'best_params': {}
        }

def optimize_window_size(data: pd.DataFrame,
                        timeframe: str = '15min',
                        window_sizes_hours: Optional[List[float]] = None,
                        iterations_per_window: int = 20,
                        parallel: bool = False) -> Dict[str, any]:
    """
    Optimize rolling window size for regime classification
    
    Args:
        data: DataFrame with indicators
        timeframe: Data timeframe
        window_sizes_hours: List of window sizes to test (in hours)
        iterations_per_window: Optimization iterations per window
        parallel: Whether to use parallel processing
        
    Returns:
        Dictionary with optimization results
    """
    logger.info("="*80)
    logger.info("WINDOW SIZE OPTIMIZATION")
    logger.info("="*80)
    
    # Default window sizes based on timeframe
    if window_sizes_hours is None:
        if timeframe in ['5min', '15min']:
            window_sizes_hours = [12, 24, 36, 48, 72]  # 0.5 to 3 days
        elif timeframe in ['30min', '1H']:
            window_sizes_hours = [24, 48, 72, 96, 120]  # 1 to 5 days
        elif timeframe in ['4H', 'Daily']:
            window_sizes_hours = [120, 240, 360, 480, 720]  # 5 to 30 days
        else:
            window_sizes_hours = [24, 48, 72, 96, 120]
    
    logger.info(f"Testing window sizes: {window_sizes_hours} hours")
    logger.info(f"Timeframe: {timeframe}")
    logger.info(f"Iterations per window: {iterations_per_window}")
    
    results = []
    
    if parallel and len(window_sizes_hours) > 1:
        # Parallel execution
        with ProcessPoolExecutor(max_workers=min(4, len(window_sizes_hours))) as executor:
            # Submit all tasks
            future_to_window = {
                executor.submit(
                    evaluate_window_size, 
                    data, 
                    window_hours, 
                    timeframe, 
                    iterations_per_window
                ): window_hours 
                for window_hours in window_sizes_hours
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_window):
                window_hours = future_to_window[future]
                try:
                    result = future.result()
                    results.append(result)
                    logger.info(f"Completed window {window_hours}h: Sharpe={result['sharpe_ratio']:.4f}")
                except Exception as e:
                    logger.error(f"Window {window_hours}h failed: {e}")
    else:
        # Sequential execution
        for window_hours in window_sizes_hours:
            result = evaluate_window_size(data, window_hours, timeframe, iterations_per_window)
            results.append(result)
            
            # Print progress
            print(f"\nWindow {window_hours}h Results:")
            print(f"  Sharpe Ratio: {result['sharpe_ratio']:.4f}")
            print(f"  Max Drawdown: {result['max_drawdown']:.2%}")
            print(f"  Win Rate: {result['win_rate']:.2%}")
            print(f"  Persistence: {result['regime_persistence']:.4f}")
    
    # Find best window
    best_result = max(results, key=lambda x: x['sharpe_ratio'])
    
    # Create summary
    summary = {
        'best_window': best_result['window_hours'],
        'best_sharpe': best_result['sharpe_ratio'],
        'best_params': best_result['best_params'],
        'all_results': results,
        'window_performance': {r['window_hours']: r['sharpe_ratio'] for r in results},
        'recommendation': generate_window_recommendation(results, timeframe)
    }
    
    # Print summary
    print_window_optimization_summary(summary)
    
    return summary

def generate_window_recommendation(results: List[Dict], timeframe: str) -> str:
    """Generate recommendation based on window optimization results"""
    
    # Sort by sharpe ratio
    sorted_results = sorted(results, key=lambda x: x['sharpe_ratio'], reverse=True)
    best = sorted_results[0]
    
    # Check if there's a clear winner
    if len(sorted_results) > 1:
        second_best = sorted_results[1]
        improvement = (best['sharpe_ratio'] - second_best['sharpe_ratio']) / abs(second_best['sharpe_ratio'])
        
        if improvement < 0.1:  # Less than 10% improvement
            # Check persistence
            if best['regime_persistence'] < second_best['regime_persistence']:
                return (f"Consider {second_best['window_hours']}h window. "
                       f"While {best['window_hours']}h has slightly better Sharpe, "
                       f"{second_best['window_hours']}h has better regime stability.")
    
    # Check if window is appropriate for timeframe
    bars_per_day = TIMEFRAMES.get(timeframe, 26)
    window_days = best['window_hours'] / 24
    window_bars = window_days * bars_per_day
    
    if window_bars < 100:
        return (f"Warning: {best['window_hours']}h window may be too short "
               f"({window_bars:.0f} bars). Consider longer windows.")
    elif window_bars > 2000:
        return (f"Warning: {best['window_hours']}h window may be too long "
               f"({window_bars:.0f} bars). Consider shorter windows.")
    
    return f"Recommended: {best['window_hours']}h window for {timeframe} data."

def print_window_optimization_summary(summary: Dict):
    """Print window optimization summary"""
    print("\n" + "="*80)
    print("WINDOW OPTIMIZATION SUMMARY")
    print("="*80)
    
    print(f"\nBest Window: {summary['best_window']} hours")
    print(f"Best Sharpe: {summary['best_sharpe']:.4f}")
    
    print("\nAll Windows Tested:")
    for window_hours, sharpe in sorted(summary['window_performance'].items()):
        print(f"  {window_hours:>4.0f}h: Sharpe = {sharpe:>7.4f}")
    
    print(f"\nRecommendation: {summary['recommendation']}")
    print("="*80)

# =============================================================================
# ADAPTIVE WINDOW SIZING
# =============================================================================

class AdaptiveWindowOptimizer:
    """
    Adaptive window sizing based on market conditions
    Adjusts window size dynamically based on volatility regime
    """
    
    def __init__(self, 
                 base_window_hours: float = DEFAULT_WINDOW_HOURS,
                 timeframe: str = '15min'):
        """
        Initialize adaptive window optimizer
        
        Args:
            base_window_hours: Base window size
            timeframe: Data timeframe
        """
        self.base_window_hours = base_window_hours
        self.timeframe = timeframe
        self.window_adjustments = {
            'Low_Vol': 1.5,      # Longer window in low volatility
            'Medium_Vol': 1.0,   # Base window
            'High_Vol': 0.75,    # Shorter window in high volatility
            'Extreme_Vol': 0.5   # Very short window in extreme volatility
        }
        
    def calculate_adaptive_window(self, 
                                 volatility_regime: str,
                                 market_conditions: Optional[Dict] = None) -> float:
        """
        Calculate adaptive window size based on conditions
        
        Args:
            volatility_regime: Current volatility regime
            market_conditions: Optional additional market conditions
            
        Returns:
            Adjusted window size in hours
        """
        # Base adjustment from volatility
        adjustment = self.window_adjustments.get(volatility_regime, 1.0)
        window_hours = self.base_window_hours * adjustment
        
        # Additional adjustments based on market conditions
        if market_conditions:
            # Adjust for trend strength
            if market_conditions.get('trend_strength') == 'Strong':
                window_hours *= 1.2  # Longer window in strong trends
            elif market_conditions.get('trend_strength') == 'Weak':
                window_hours *= 0.9  # Shorter window in weak trends
            
            # Adjust for regime stability
            if market_conditions.get('regime_persistence', 0) > 0.9:
                window_hours *= 1.1  # Longer window if regimes are stable
            elif market_conditions.get('regime_persistence', 0) < 0.7:
                window_hours *= 0.9  # Shorter window if regimes are unstable
        
        # Ensure window is within reasonable bounds
        min_window = 6 if self.timeframe in ['5min', '15min'] else 12
        max_window = 168  # 1 week
        
        return np.clip(window_hours, min_window, max_window)
    
    def optimize_adaptive_parameters(self,
                                   data: pd.DataFrame,
                                   regime_data: pd.DataFrame) -> Dict[str, float]:
        """
        Optimize adaptive window parameters
        
        Returns:
            Optimized adjustment factors
        """
        logger.info("Optimizing adaptive window parameters...")
        
        best_adjustments = self.window_adjustments.copy()
        best_score = -np.inf
        
        # Test different adjustment factors
        for low_adj in [1.2, 1.5, 1.8]:
            for high_adj in [0.5, 0.75, 1.0]:
                for extreme_adj in [0.3, 0.5, 0.7]:
                    
                    test_adjustments = {
                        'Low_Vol': low_adj,
                        'Medium_Vol': 1.0,
                        'High_Vol': high_adj,
                        'Extreme_Vol': extreme_adj
                    }
                    
                    # Evaluate performance with these adjustments
                    score = self._evaluate_adaptive_performance(
                        data, regime_data, test_adjustments
                    )
                    
                    if score > best_score:
                        best_score = score
                        best_adjustments = test_adjustments
        
        logger.info(f"Best adaptive adjustments: {best_adjustments}")
        return best_adjustments
    
    def _evaluate_adaptive_performance(self,
                                     data: pd.DataFrame,
                                     regime_data: pd.DataFrame,
                                     adjustments: Dict[str, float]) -> float:
        """Evaluate performance with specific adjustment factors"""
        # This is a simplified evaluation
        # In practice, would run full backtesting with adaptive windows
        
        # For now, return a score based on regime stability
        vol_regimes = regime_data['Volatility_Regime']
        
        score = 0
        for vol_regime, adjustment in adjustments.items():
            mask = vol_regimes == vol_regime
            if mask.sum() > 0:
                # Favor larger adjustments for extreme regimes
                if vol_regime == 'Low_Vol':
                    score += adjustment * 0.3
                elif vol_regime == 'Extreme_Vol':
                    score += (1 / adjustment) * 0.3
        
        return score

# =============================================================================
# TIMEFRAME-SPECIFIC OPTIMIZATION
# =============================================================================

def get_recommended_window_range(timeframe: str) -> Tuple[float, float]:
    """Get recommended window range for a timeframe"""
    
    recommendations = {
        '5min': (6, 48),      # 6 hours to 2 days
        '15min': (12, 72),    # 0.5 to 3 days
        '30min': (24, 96),    # 1 to 4 days
        '1H': (48, 168),      # 2 to 7 days
        '4H': (96, 336),      # 4 to 14 days
        'Daily': (240, 720),  # 10 to 30 days
        'Weekly': (720, 2160) # 30 to 90 days
    }
    
    return recommendations.get(timeframe, (24, 168))

def calculate_effective_lookback(window_hours: float, timeframe: str) -> Dict[str, any]:
    """Calculate effective lookback period for a window"""
    
    bars_per_day = TIMEFRAMES.get(timeframe, 26)
    window_days = window_hours / 24
    window_bars = int(window_days * bars_per_day)
    
    return {
        'window_hours': window_hours,
        'window_days': window_days,
        'window_bars': window_bars,
        'min_bars_required': window_bars // 2,
        'memory_days': window_days,
        'responsiveness': 1 / window_days  # Higher = more responsive
    }
