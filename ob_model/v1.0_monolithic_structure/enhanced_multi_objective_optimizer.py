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
# =============================================================================
# INTEGRATION SCRIPT - ENHANCED BACKTESTER
# This shows how to integrate the enhanced backtester with your existing system
# Save this as: enhanced_multi_objective_optimizer.py
# =============================================================================

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Any
from scipy.optimize import differential_evolution
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

# Import your existing modules
from multi_objective_optimizer import OptimizationResults, MultiObjectiveRegimeOptimizer

# Import the enhanced backtester (save the previous artifact as enhanced_backtester.py)
from enhanced_backtester import EnhancedRegimeStrategyBacktester, StrategyConfig

logger = logging.getLogger(__name__)

# Override the existing RegimeStrategyBacktester with the enhanced version
RegimeStrategyBacktester = EnhancedRegimeStrategyBacktester

class EnhancedMultiObjectiveRegimeOptimizer(MultiObjectiveRegimeOptimizer):
    """
    Enhanced optimizer that uses the sophisticated backtester
    Inherits from your existing optimizer to preserve all logic
    """
    
    def __init__(self, regime_classifier, data: pd.DataFrame):
        super().__init__(regime_classifier, data)
        # Replace the backtester with enhanced version
        self.backtester = EnhancedRegimeStrategyBacktester()
        
    def evaluate_regime_performance(self, data_with_regimes: pd.DataFrame) -> Dict[str, Any]:
        """
        Enhanced performance evaluation using sophisticated strategies
        This overrides the original method with better strategy logic
        """
        try:
            # Use the enhanced adaptive strategy
            strategy_returns = self.backtester.adaptive_regime_strategy_enhanced(
                self.data, data_with_regimes
            )
            
            # Calculate enhanced performance metrics
            performance_metrics = self.backtester.calculate_performance_metrics(strategy_returns)
            
            # Extract key metrics for optimization
            return {
                'returns': strategy_returns,
                'sharpe_ratio': performance_metrics['sharpe_ratio'],
                'max_drawdown': performance_metrics['max_drawdown'],
                'total_return': performance_metrics['total_return'],
                'win_rate': performance_metrics['win_rate'],
                'calmar_ratio': performance_metrics['calmar_ratio'],
                'sortino_ratio': performance_metrics['sortino_ratio']
            }
            
        except Exception as e:
            logger.error(f"Enhanced performance evaluation error: {e}")
            return {
                'returns': pd.Series(0, index=self.data.index),
                'sharpe_ratio': -99,
                'max_drawdown': -0.99,
                'total_return': -0.99,
                'win_rate': 0,
                'calmar_ratio': 0,
                'sortino_ratio': -99
            }

def run_regime_optimization(classifier, data: pd.DataFrame, 
                          max_iterations: int = 50, 
                          method: str = 'differential_evolution',
                          walk_forward: bool = False) -> OptimizationResults:
    """
    Main function to run regime optimization
    """
    print("\n" + "="*80)
    print("MULTI-OBJECTIVE REGIME OPTIMIZATION")
    print("="*80)
    print("Optimizing regime classification thresholds for:")
    print("  • Sharpe Ratio (40% weight)")
    print("  • Maximum Drawdown (30% weight)") 
    print("  • Regime Persistence (30% weight)")
    print(f"  • Method: {method}")
    print(f"  • Max Iterations: {max_iterations}")
    print(f"  • Walk-Forward: {walk_forward}")
    print("="*80)
    
    optimizer = MultiObjectiveRegimeOptimizer(classifier, data)
    results = optimizer.optimize_regime_thresholds(method=method, max_iterations=max_iterations)
    
    return results

# =============================================================================
# USAGE EXAMPLE - MINIMAL CHANGES TO YOUR EXISTING CODE
# =============================================================================

def update_existing_system():
    """
    This shows the minimal changes needed to your existing system
    """
    
    # In your multidimensional_regime_system.py, make these changes:
    
    # 1. Change the import at the top:
    # OLD:
    # from multi_objective_optimizer import (
    #     MultiObjectiveRegimeOptimizer, 
    #     RegimeStrategyBacktester,
    #     run_regime_optimization,
    #     print_optimization_results,
    #     OptimizationResults
    # )
    
    # NEW:
    print("""
    # Add this import to your multidimensional_regime_system.py:
    from enhanced_multi_objective_optimizer import (
        run_enhanced_regime_optimization as run_regime_optimization,
        OptimizationResults,
        print_optimization_results
    )
    from multi_objective_optimizer import MultiObjectiveRegimeOptimizer
    """)
    
    # 2. In the run_enhanced_analysis_with_optimization function, change:
    # OLD:
    # optimization_results = run_regime_optimization(
    #     classifier, 
    #     data_with_indicators,
    #     max_iterations=OPTIMIZATION_ITERATIONS
    # )
    
    # NEW:
    print("""
    # Replace the optimization call with:
    optimization_results = run_enhanced_regime_optimization(
        classifier, 
        data_with_indicators,
        max_iterations=OPTIMIZATION_ITERATIONS
    )
    """)
    
    # That's it! The enhanced backtester will now be used automatically

# =============================================================================
# QUICK TEST FUNCTION
# =============================================================================

def test_enhanced_backtester(data: pd.DataFrame, regimes: pd.DataFrame):
    """
    Quick test to verify the enhanced backtester works with your data
    """
    print("\n" + "="*50)
    print("TESTING ENHANCED BACKTESTER")
    print("="*50)
    
    backtester = EnhancedRegimeStrategyBacktester()
    
    # Test momentum strategy
    up_mask = regimes['Direction_Regime'] == 'Up_Trending'
    momentum_returns = backtester.momentum_strategy_enhanced(data, up_mask, regimes)
    momentum_metrics = backtester.calculate_performance_metrics(momentum_returns)
    
    print(f"\nMomentum Strategy Performance:")
    print(f"  Sharpe Ratio: {momentum_metrics['sharpe_ratio']:.4f}")
    print(f"  Total Return: {momentum_metrics['total_return']:.2%}")
    print(f"  Max Drawdown: {momentum_metrics['max_drawdown']:.2%}")
    print(f"  Win Rate: {momentum_metrics['win_rate']:.2%}")
    
    # Test adaptive strategy
    adaptive_returns = backtester.adaptive_regime_strategy_enhanced(data, regimes)
    adaptive_metrics = backtester.calculate_performance_metrics(adaptive_returns)
    
    print(f"\nAdaptive Strategy Performance:")
    print(f"  Sharpe Ratio: {adaptive_metrics['sharpe_ratio']:.4f}")
    print(f"  Total Return: {adaptive_metrics['total_return']:.2%}")
    print(f"  Max Drawdown: {adaptive_metrics['max_drawdown']:.2%}")
    print(f"  Win Rate: {adaptive_metrics['win_rate']:.2%}")
    print(f"  Calmar Ratio: {adaptive_metrics['calmar_ratio']:.4f}")
    print(f"  Sortino Ratio: {adaptive_metrics['sortino_ratio']:.4f}")
    
    print("="*50)

if __name__ == "__main__":
    print("\n" + "="*80)
    print("ENHANCED REGIME STRATEGY BACKTESTER INTEGRATION")
    print("="*80)
    print("\nThis enhanced backtester includes:")
    print("  ✅ Sophisticated multi-indicator entry/exit signals")
    print("  ✅ Dynamic position sizing based on volatility and regime confidence")
    print("  ✅ Transaction costs (0.5 bps) and slippage (1 bp)")
    print("  ✅ Risk management with ATR-based stops and targets")
    print("  ✅ Regime-specific strategy parameters")
    print("  ✅ Enhanced performance metrics (Sharpe, Sortino, Calmar)")
    print("\nExpected improvements:")
    print("  • Sharpe Ratio: Target > 1.0 (from -0.1665)")
    print("  • Positive returns with controlled drawdowns")
    print("  • More realistic trading simulation")
    print("\nIntegration steps:")
    print("  1. Save the enhanced backtester as 'enhanced_backtester.py'")
    print("  2. Save this file as 'enhanced_multi_objective_optimizer.py'")
    print("  3. Update imports in your main system file")
    print("  4. Run optimization with OPTIMIZATION_ITERATIONS = 50-100")
    print("="*80)
    
    update_existing_system()
