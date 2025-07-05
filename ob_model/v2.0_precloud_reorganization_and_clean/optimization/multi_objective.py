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
Multi-objective optimization module
Handles regime threshold optimization with multiple objectives
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from scipy.optimize import differential_evolution, minimize
import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import (
    OBJECTIVE_WEIGHTS, PARAMETER_BOUNDS, OPTIMIZATION_METHOD,
    COMMISSION_RATE, SLIPPAGE_RATE
)
from core.regime_classifier import RollingRegimeClassifier
from backtesting.strategies import EnhancedRegimeStrategyBacktester

logger = logging.getLogger(__name__)

# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class OptimizationResults:
    """Container for optimization results"""
    best_params: Dict[str, float]
    best_score: float
    sharpe_ratio: float
    max_drawdown: float
    regime_persistence: float
    total_return: float
    win_rate: float
    calmar_ratio: float
    sortino_ratio: float
    strategy_returns: pd.Series
    regime_classifications: pd.DataFrame
    optimization_history: List[Dict]
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for saving"""
        return {
            'best_params': self.best_params,
            'best_score': self.best_score,
            'sharpe_ratio': self.sharpe_ratio,
            'max_drawdown': self.max_drawdown,
            'regime_persistence': self.regime_persistence,
            'total_return': self.total_return,
            'win_rate': self.win_rate,
            'calmar_ratio': self.calmar_ratio,
            'sortino_ratio': self.sortino_ratio,
            'optimization_iterations': len(self.optimization_history)
        }

# =============================================================================
# MULTI-OBJECTIVE OPTIMIZER
# =============================================================================

class MultiObjectiveRegimeOptimizer:
    """
    Multi-objective optimization for regime classification thresholds
    Optimizes for Sharpe ratio, drawdown, and regime persistence
    """
    
    def __init__(self, 
                 regime_classifier: RollingRegimeClassifier,
                 data: pd.DataFrame,
                 objective_weights: Optional[Dict[str, float]] = None):
        """
        Initialize optimizer
        
        Args:
            regime_classifier: Configured regime classifier
            data: DataFrame with indicators
            objective_weights: Weights for objectives (sharpe, drawdown, persistence)
        """
        self.classifier = regime_classifier
        self.data = data
        self.backtester = EnhancedRegimeStrategyBacktester()
        
        # Objective weights
        self.objective_weights = objective_weights or OBJECTIVE_WEIGHTS
        
        # Parameter bounds
        self.param_bounds = PARAMETER_BOUNDS
        
        # Optimization history
        self.optimization_history = []
        self.iteration_count = 0
        self.max_iterations = 100  # Will be updated
        
        logger.info(f"Initialized optimizer with {len(self.param_bounds)} parameters")
    
    def update_classifier_thresholds(self, params: Dict[str, float]):
        """Update classifier thresholds with new parameters"""
        try:
            # Update direction thresholds
            self.classifier.dimension_thresholds['direction']['strong_trend_threshold'] = params['direction_strong_trend']
            self.classifier.dimension_thresholds['direction']['weak_trend_threshold'] = params['direction_weak_trend']
            
            # Update trend strength thresholds
            self.classifier.dimension_thresholds['trend_strength']['strong_alignment'] = params['trend_strong_alignment']
            self.classifier.dimension_thresholds['trend_strength']['moderate_alignment'] = params['trend_moderate_alignment']
            
            # Update velocity thresholds
            self.classifier.dimension_thresholds['velocity']['acceleration_threshold'] = params['velocity_acceleration']
            self.classifier.dimension_thresholds['velocity']['stable_range'] = params['velocity_stable_range']
            
            # Update volatility thresholds
            self.classifier.dimension_thresholds['volatility']['high_vol_percentile'] = params['volatility_high_percentile']
            self.classifier.dimension_thresholds['volatility']['low_vol_percentile'] = params['volatility_low_percentile']
            
            # Update microstructure thresholds
            self.classifier.dimension_thresholds['microstructure']['institutional_volume_threshold'] = params['microstructure_institutional']
            self.classifier.dimension_thresholds['microstructure']['retail_volume_threshold'] = params['microstructure_retail']
            
        except Exception as e:
            logger.error(f"Error updating classifier thresholds: {e}")
    
    def calculate_regime_persistence(self, regimes: pd.DataFrame) -> float:
        """Calculate regime persistence score (higher is better)"""
        try:
            persistence_scores = []
            
            # Calculate persistence for each dimension
            for dimension in ['Direction', 'TrendStrength', 'Velocity', 'Volatility', 'Microstructure']:
                col = f'{dimension}_Regime'
                if col in regimes.columns:
                    # Count regime changes
                    regime_changes = (regimes[col] != regimes[col].shift()).sum()
                    total_periods = len(regimes[regimes[col] != 'Undefined'])
                    
                    if total_periods > 0:
                        # Persistence = 1 - (changes / total_periods)
                        persistence = max(0, 1 - (regime_changes / total_periods))
                        persistence_scores.append(persistence)
            
            # Return average persistence across all dimensions
            return np.mean(persistence_scores) if persistence_scores else 0.0
            
        except Exception as e:
            logger.warning(f"Error calculating regime persistence: {e}")
            return 0.0
    
    def evaluate_performance(self, params_array: np.ndarray) -> Tuple[Dict[str, float], pd.DataFrame]:
        """
        Evaluate performance for given parameters
        
        Returns:
            Tuple of (performance_metrics, regime_classifications)
        """
        # Convert array to parameter dictionary
        param_names = list(self.param_bounds.keys())
        params = dict(zip(param_names, params_array))
        
        # Update classifier with new parameters
        self.update_classifier_thresholds(params)
        
        # Classify regimes with new parameters
        regimes = self.classifier.classify_regimes(self.data, show_progress=False)
        
        # Run backtesting
        strategy_returns = self.backtester.adaptive_regime_strategy_enhanced(self.data, regimes)
        
        # Calculate performance metrics
        performance = self.backtester.calculate_performance_metrics(strategy_returns)
        
        # Calculate regime persistence
        persistence = self.calculate_regime_persistence(regimes)
        
        # Add persistence to metrics
        performance['regime_persistence'] = persistence
        
        return performance, regimes
    
    def objective_function(self, params_array: np.ndarray) -> float:
        """
        Objective function for optimization (to minimize)
        
        Args:
            params_array: Array of parameter values
            
        Returns:
            Combined objective score (lower is better)
        """
        try:
            self.iteration_count += 1
            
            # Log progress periodically
            if self.iteration_count % 5 == 0:
                print(f"Function call {self.iteration_count}/{self.max_iterations}")
            
            # Evaluate performance
            performance, regimes = self.evaluate_performance(params_array)
            
            # Extract metrics
            sharpe_ratio = performance['sharpe_ratio']
            max_drawdown = performance['max_drawdown']
            persistence = performance['regime_persistence']
            
            # Multi-objective score (convert to minimization)
            sharpe_component = -sharpe_ratio  # Negative because we want to maximize
            drawdown_component = -max_drawdown  # Already negative, but we want less negative
            persistence_component = -persistence  # Negative because we want to maximize
            
            # Weighted combination
            total_score = (
                self.objective_weights['sharpe_ratio'] * sharpe_component +
                self.objective_weights['max_drawdown'] * drawdown_component +
                self.objective_weights['regime_persistence'] * persistence_component
            )
            
            # Store in history
            param_dict = dict(zip(self.param_bounds.keys(), params_array))
            self.optimization_history.append({
                'iteration': self.iteration_count,
                'params': param_dict.copy(),
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'regime_persistence': persistence,
                'total_score': -total_score,  # Convert back for reporting
                'total_return': performance['total_return'],
                'win_rate': performance['win_rate']
            })
            
            # Log best so far
            if self.iteration_count % 10 == 0:
                best_score = min(h['total_score'] for h in self.optimization_history)
                logger.info(f"Iteration {self.iteration_count}: Current score={-total_score:.4f}, "
                          f"Best={-best_score:.4f}, Sharpe={sharpe_ratio:.4f}")
            
            return total_score
            
        except Exception as e:
            logger.error(f"Error in objective function: {e}")
            return 1000.0  # High penalty for errors
    
    def optimize_regime_thresholds(self, 
                                  method: str = OPTIMIZATION_METHOD,
                                  max_iterations: int = 100) -> OptimizationResults:
        """
        Run multi-objective optimization
        
        Args:
            method: Optimization method ('differential_evolution' or 'L-BFGS-B')
            max_iterations: Maximum iterations
            
        Returns:
            OptimizationResults object
        """
        logger.info("Starting multi-objective regime optimization...")
        logger.info(f"Method: {method}, Max iterations: {max_iterations}")
        logger.info(f"Objectives: {list(self.objective_weights.keys())}")
        logger.info(f"Weights: {list(self.objective_weights.values())}")
        
        # Reset counters
        self.iteration_count = 0
        self.max_iterations = max_iterations
        self.optimization_history = []
        
        try:
            # Prepare bounds
            bounds = list(self.param_bounds.values())
            param_names = list(self.param_bounds.keys())
            
            # Initial guess (midpoint of bounds)
            initial_guess = [(b[0] + b[1]) / 2 for b in bounds]
            
            logger.info(f"Optimizing {len(bounds)} parameters...")
            
            # Run optimization
            if method == 'differential_evolution':
                result = differential_evolution(
                    self.objective_function,
                    bounds,
                    maxiter=max_iterations,
                    popsize=5,  # Small population for speed
                    seed=42,
                    disp=True,
                    workers=1,
                    updating='immediate',
                    strategy='best1bin',
                    recombination=0.7,
                    mutation=(0.5, 1.0)
                )
            else:
                # L-BFGS-B
                result = minimize(
                    self.objective_function,
                    initial_guess,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': max_iterations, 'disp': True}
                )
            
            # Extract best parameters
            best_params = dict(zip(param_names, result.x))
            
            # Get final performance with best parameters
            final_performance, final_regimes = self.evaluate_performance(result.x)
            
            logger.info("Optimization completed!")
            logger.info(f"Best Score: {-result.fun:.4f}")
            logger.info(f"Best Sharpe: {final_performance['sharpe_ratio']:.4f}")
            logger.info(f"Best Drawdown: {final_performance['max_drawdown']:.4f}")
            logger.info(f"Best Persistence: {final_performance['regime_persistence']:.4f}")
            
            # Create results object
            results = OptimizationResults(
                best_params=best_params,
                best_score=-result.fun,
                sharpe_ratio=final_performance['sharpe_ratio'],
                max_drawdown=final_performance['max_drawdown'],
                regime_persistence=final_performance['regime_persistence'],
                total_return=final_performance['total_return'],
                win_rate=final_performance['win_rate'],
                calmar_ratio=final_performance['calmar_ratio'],
                sortino_ratio=final_performance['sortino_ratio'],
                strategy_returns=final_performance['returns'],
                regime_classifications=final_regimes,
                optimization_history=self.optimization_history
            )
            
            return results
            
        except Exception as e:
            logger.error(f"Optimization failed: {e}")
            # Return default results
            return self._create_default_results()
    
    def _create_default_results(self) -> OptimizationResults:
        """Create default results for failed optimization"""
        return OptimizationResults(
            best_params={},
            best_score=0.0,
            sharpe_ratio=0.0,
            max_drawdown=-1.0,
            regime_persistence=0.0,
            total_return=-1.0,
            win_rate=0.0,
            calmar_ratio=0.0,
            sortino_ratio=0.0,
            strategy_returns=pd.Series(dtype=float),
            regime_classifications=pd.DataFrame(),
            optimization_history=self.optimization_history
        )

# =============================================================================
# OPTIMIZATION UTILITIES
# =============================================================================

def print_optimization_results(results: OptimizationResults):
    """Pretty print optimization results"""
    print("\n" + "="*80)
    print("OPTIMIZATION RESULTS")
    print("="*80)
    
    print("\nPerformance Metrics:")
    print(f"  Best Score: {results.best_score:.4f}")
    print(f"  Sharpe Ratio: {results.sharpe_ratio:.4f}")
    print(f"  Max Drawdown: {results.max_drawdown:.2%}")
    print(f"  Total Return: {results.total_return:.2%}")
    print(f"  Win Rate: {results.win_rate:.2%}")
    print(f"  Calmar Ratio: {results.calmar_ratio:.4f}")
    print(f"  Sortino Ratio: {results.sortino_ratio:.4f}")
    print(f"  Regime Persistence: {results.regime_persistence:.4f}")
    
    print("\nOptimized Parameters:")
    for param, value in results.best_params.items():
        print(f"  {param}: {value:.4f}")
    
    print(f"\nOptimization completed in {len(results.optimization_history)} iterations")
    print("="*80)

def compare_optimizations(results_list: List[OptimizationResults], 
                         labels: Optional[List[str]] = None) -> pd.DataFrame:
    """Compare multiple optimization results"""
    if labels is None:
        labels = [f"Opt_{i+1}" for i in range(len(results_list))]
    
    comparison_data = []
    
    for i, (result, label) in enumerate(zip(results_list, labels)):
        comparison_data.append({
            'Label': label,
            'Score': result.best_score,
            'Sharpe': result.sharpe_ratio,
            'Drawdown': result.max_drawdown,
            'Return': result.total_return,
            'Win_Rate': result.win_rate,
            'Calmar': result.calmar_ratio,
            'Sortino': result.sortino_ratio,
            'Persistence': result.regime_persistence
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df = comparison_df.set_index('Label')
    
    return comparison_df

def save_optimization_history(history: List[Dict], filepath: str):
    """Save optimization history to CSV"""
    history_df = pd.DataFrame(history)
    history_df.to_csv(filepath, index=False)
    logger.info(f"Optimization history saved to: {filepath}")
