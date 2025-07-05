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
Walk-forward optimization module
Implements walk-forward validation to prevent overfitting
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import (
    WALK_FORWARD_WINDOWS, WALK_FORWARD_TRAIN_RATIO,
    OPTIMIZATION_ITERATIONS, RESULTS_DIR
)
from core.regime_classifier import RollingRegimeClassifier
from core.indicators import calculate_all_indicators
from optimization.multi_objective import (
    MultiObjectiveRegimeOptimizer, OptimizationResults
)

logger = logging.getLogger(__name__)

# =============================================================================
# WALK-FORWARD OPTIMIZER
# =============================================================================

class WalkForwardOptimizer:
    """
    Walk-forward optimization to validate regime parameters
    Prevents overfitting by testing on out-of-sample data
    """
    
    def __init__(self, 
                 n_windows: int = WALK_FORWARD_WINDOWS,
                 train_ratio: float = WALK_FORWARD_TRAIN_RATIO):
        """
        Initialize walk-forward optimizer
        
        Args:
            n_windows: Number of walk-forward windows
            train_ratio: Ratio of data for training (rest for testing)
        """
        self.n_windows = n_windows
        self.train_ratio = train_ratio
        self.window_results = []
        
        logger.info(f"Initialized walk-forward optimizer with {n_windows} windows, "
                   f"{train_ratio:.1%} train ratio")
    
    def split_data_windows(self, data: pd.DataFrame) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Split data into walk-forward windows
        
        Returns:
            List of (train_data, test_data) tuples
        """
        total_len = len(data)
        window_size = total_len // self.n_windows
        train_size = int(window_size * self.train_ratio)
        test_size = window_size - train_size
        
        windows = []
        
        for i in range(self.n_windows):
            start_idx = i * test_size
            train_end_idx = start_idx + train_size
            test_end_idx = min(train_end_idx + test_size, total_len)
            
            # Skip if not enough data for test
            if test_end_idx <= train_end_idx:
                break
            
            train_data = data.iloc[start_idx:train_end_idx]
            test_data = data.iloc[train_end_idx:test_end_idx]
            
            windows.append((train_data, test_data))
            
            logger.info(f"Window {i+1}: Train {train_data.index[0]} to {train_data.index[-1]} "
                       f"({len(train_data)} periods), Test {test_data.index[0]} to {test_data.index[-1]} "
                       f"({len(test_data)} periods)")
        
        return windows
    
    def run_single_window(self, 
                         train_data: pd.DataFrame,
                         test_data: pd.DataFrame,
                         window_hours: float,
                         timeframe: str,
                         max_iterations: int) -> Dict[str, any]:
        """
        Run optimization on single window
        
        Returns:
            Dictionary with train and test results
        """
        try:
            # Create classifier for this window
            classifier = RollingRegimeClassifier(window_hours=window_hours, timeframe=timeframe)
            
            # Optimize on training data
            optimizer = MultiObjectiveRegimeOptimizer(classifier, train_data)
            train_results = optimizer.optimize_regime_thresholds(
                method='differential_evolution',
                max_iterations=max_iterations
            )
            
            # Apply best parameters to test data
            classifier_test = RollingRegimeClassifier(window_hours=window_hours, timeframe=timeframe)
            optimizer_test = MultiObjectiveRegimeOptimizer(classifier_test, test_data)
            
            # Update test classifier with trained parameters
            optimizer_test.update_classifier_thresholds(train_results.best_params)
            
            # Evaluate on test data
            test_performance, test_regimes = optimizer_test.evaluate_performance(
                np.array(list(train_results.best_params.values()))
            )
            
            return {
                'train_sharpe': train_results.sharpe_ratio,
                'test_sharpe': test_performance['sharpe_ratio'],
                'train_drawdown': train_results.max_drawdown,
                'test_drawdown': test_performance['max_drawdown'],
                'train_return': train_results.total_return,
                'test_return': test_performance['total_return'],
                'train_persistence': train_results.regime_persistence,
                'test_persistence': test_performance['regime_persistence'],
                'best_params': train_results.best_params,
                'overfit_ratio': (train_results.sharpe_ratio - test_performance['sharpe_ratio']) / 
                                abs(train_results.sharpe_ratio) if train_results.sharpe_ratio != 0 else 0
            }
            
        except Exception as e:
            logger.error(f"Error in window optimization: {e}")
            return {
                'train_sharpe': 0,
                'test_sharpe': 0,
                'train_drawdown': -1,
                'test_drawdown': -1,
                'train_return': -1,
                'test_return': -1,
                'train_persistence': 0,
                'test_persistence': 0,
                'best_params': {},
                'overfit_ratio': 1.0
            }
    
    def run_walk_forward(self,
                        data: pd.DataFrame,
                        window_hours: float,
                        timeframe: str = '15min',
                        max_iterations: int = OPTIMIZATION_ITERATIONS) -> Dict[str, any]:
        """
        Run complete walk-forward optimization
        
        Returns:
            Dictionary with aggregated results
        """
        logger.info("="*80)
        logger.info("STARTING WALK-FORWARD OPTIMIZATION")
        logger.info("="*80)
        
        # Split data into windows
        windows = self.split_data_windows(data)
        logger.info(f"Created {len(windows)} walk-forward windows")
        
        # Run optimization on each window
        self.window_results = []
        
        for i, (train_data, test_data) in enumerate(windows):
            logger.info(f"\n--- Window {i+1}/{len(windows)} ---")
            logger.info(f"Training: {train_data.index[0]} to {train_data.index[-1]} ({len(train_data)} periods)")
            logger.info(f"Testing: {test_data.index[0]} to {test_data.index[-1]} ({len(test_data)} periods)")
            
            window_result = self.run_single_window(
                train_data, test_data, window_hours, timeframe, max_iterations
            )
            
            window_result['window_id'] = i + 1
            self.window_results.append(window_result)
            
            # Log results
            logger.info(f"Window {i+1} Results:")
            logger.info(f"  Train Sharpe: {window_result['train_sharpe']:.4f}")
            logger.info(f"  Test Sharpe: {window_result['test_sharpe']:.4f}")
            logger.info(f"  Overfit Ratio: {window_result['overfit_ratio']:.2%}")
        
        # Aggregate results
        aggregated = self.aggregate_results()
        
        # Print summary
        self.print_summary(aggregated)
        
        return aggregated
    
    def aggregate_results(self) -> Dict[str, any]:
        """Aggregate results across all windows"""
        if not self.window_results:
            return {}
        
        # Calculate averages
        avg_train_sharpe = np.mean([r['train_sharpe'] for r in self.window_results])
        avg_test_sharpe = np.mean([r['test_sharpe'] for r in self.window_results])
        avg_train_drawdown = np.mean([r['train_drawdown'] for r in self.window_results])
        avg_test_drawdown = np.mean([r['test_drawdown'] for r in self.window_results])
        avg_train_return = np.mean([r['train_return'] for r in self.window_results])
        avg_test_return = np.mean([r['test_return'] for r in self.window_results])
        avg_overfit_ratio = np.mean([r['overfit_ratio'] for r in self.window_results])
        
        # Find most consistent parameters (appear most frequently)
        all_params = {}
        for result in self.window_results:
            for param, value in result['best_params'].items():
                if param not in all_params:
                    all_params[param] = []
                all_params[param].append(value)
        
        # Use median of each parameter
        median_params = {param: np.median(values) for param, values in all_params.items()}
        
        return {
            'n_windows': len(self.window_results),
            'avg_train_sharpe': avg_train_sharpe,
            'avg_test_sharpe': avg_test_sharpe,
            'avg_train_drawdown': avg_train_drawdown,
            'avg_test_drawdown': avg_test_drawdown,
            'avg_train_return': avg_train_return,
            'avg_test_return': avg_test_return,
            'avg_overfit_ratio': avg_overfit_ratio,
            'sharpe_degradation': (avg_train_sharpe - avg_test_sharpe) / abs(avg_train_sharpe) if avg_train_sharpe != 0 else 0,
            'median_params': median_params,
            'window_results': self.window_results,
            'is_overfit': avg_overfit_ratio > 0.3,  # Flag if >30% degradation
            'best_score': avg_test_sharpe,  # Use test sharpe as score
            'sharpe_ratio': avg_test_sharpe,
            'max_drawdown': avg_test_drawdown,
            'total_return': avg_test_return,
            'regime_persistence': np.mean([r['test_persistence'] for r in self.window_results]),
            'best_params': median_params  # For compatibility
        }
    
    def print_summary(self, results: Dict[str, any]):
        """Print walk-forward summary"""
        print("\n" + "="*80)
        print("WALK-FORWARD OPTIMIZATION SUMMARY")
        print("="*80)
        
        print(f"\nWindows analyzed: {results['n_windows']}")
        
        print("\nAverage Performance:")
        print(f"  Train Sharpe: {results['avg_train_sharpe']:.4f}")
        print(f"  Test Sharpe: {results['avg_test_sharpe']:.4f}")
        print(f"  Sharpe Degradation: {results['sharpe_degradation']:.1%}")
        
        print(f"\n  Train Drawdown: {results['avg_train_drawdown']:.2%}")
        print(f"  Test Drawdown: {results['avg_test_drawdown']:.2%}")
        
        print(f"\n  Train Return: {results['avg_train_return']:.2%}")
        print(f"  Test Return: {results['avg_test_return']:.2%}")
        
        print(f"\nOverfitting Analysis:")
        print(f"  Average Overfit Ratio: {results['avg_overfit_ratio']:.1%}")
        print(f"  Is Overfit: {'YES' if results['is_overfit'] else 'NO'}")
        
        print("\nMedian Optimized Parameters:")
        for param, value in results['median_params'].items():
            print(f"  {param}: {value:.4f}")
        
        print("\nIndividual Window Results:")
        for window in results['window_results']:
            print(f"  Window {window['window_id']}: "
                  f"Train Sharpe={window['train_sharpe']:.3f}, "
                  f"Test Sharpe={window['test_sharpe']:.3f}, "
                  f"Overfit={window['overfit_ratio']:.1%}")
        
        print("="*80)
    
    def save_results(self, results: Dict[str, any], filename: Optional[str] = None):
        """Save walk-forward results"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"walk_forward_results_{timestamp}.csv"
        
        filepath = os.path.join(RESULTS_DIR, filename)
        
        # Convert window results to DataFrame
        window_df = pd.DataFrame(results['window_results'])
        window_df.to_csv(filepath, index=False)
        
        # Save summary
        summary_file = filepath.replace('.csv', '_summary.txt')
        with open(summary_file, 'w') as f:
            f.write("WALK-FORWARD OPTIMIZATION SUMMARY\n")
            f.write("="*80 + "\n\n")
            f.write(f"Windows: {results['n_windows']}\n")
            f.write(f"Avg Train Sharpe: {results['avg_train_sharpe']:.4f}\n")
            f.write(f"Avg Test Sharpe: {results['avg_test_sharpe']:.4f}\n")
            f.write(f"Sharpe Degradation: {results['sharpe_degradation']:.1%}\n")
            f.write(f"Is Overfit: {'YES' if results['is_overfit'] else 'NO'}\n")
            f.write("\nMedian Parameters:\n")
            for param, value in results['median_params'].items():
                f.write(f"  {param}: {value:.4f}\n")
        
        logger.info(f"Walk-forward results saved to: {filepath}")
        logger.info(f"Summary saved to: {summary_file}")

# =============================================================================
# VALIDATION UTILITIES
# =============================================================================

def validate_no_forward_bias(data: pd.DataFrame, 
                           regimes: pd.DataFrame,
                           window_bars: int) -> bool:
    """
    Validate that regime classifications don't use future data
    
    Returns:
        True if no forward bias detected
    """
    logger.info("Validating no forward-looking bias...")
    
    # Check that early periods have undefined regimes
    undefined_count = (regimes.iloc[:window_bars]['Direction_Regime'] == 'Undefined').sum()
    
    if undefined_count < window_bars * 0.8:  # Should be mostly undefined
        logger.warning(f"Potential forward bias: Only {undefined_count}/{window_bars} "
                      f"undefined regimes in initial window")
        return False
    
    # Check regime stability
    for col in regimes.columns:
        if 'Regime' in col and col != 'Composite_Regime':
            # Check if regimes are too stable (might indicate using full data)
            regime_changes = (regimes[col] != regimes[col].shift()).sum()
            if regime_changes < len(regimes) * 0.01:  # Less than 1% changes
                logger.warning(f"Potential forward bias in {col}: "
                              f"Only {regime_changes} regime changes")
                return False
    
    logger.info("No forward-looking bias detected")
    return True

def calculate_stability_metrics(window_results: List[Dict]) -> Dict[str, float]:
    """Calculate stability metrics across walk-forward windows"""
    
    # Parameter stability (std dev of each parameter)
    param_stability = {}
    all_params = {}
    
    for result in window_results:
        for param, value in result['best_params'].items():
            if param not in all_params:
                all_params[param] = []
            all_params[param].append(value)
    
    for param, values in all_params.items():
        param_stability[param] = np.std(values) / np.mean(values) if np.mean(values) != 0 else 0
    
    # Performance stability
    train_sharpes = [r['train_sharpe'] for r in window_results]
    test_sharpes = [r['test_sharpe'] for r in window_results]
    
    return {
        'param_stability': param_stability,
        'avg_param_stability': np.mean(list(param_stability.values())),
        'train_sharpe_std': np.std(train_sharpes),
        'test_sharpe_std': np.std(test_sharpes),
        'performance_consistency': 1 - (np.std(test_sharpes) / np.mean(test_sharpes)) 
                                  if np.mean(test_sharpes) != 0 else 0
    }
