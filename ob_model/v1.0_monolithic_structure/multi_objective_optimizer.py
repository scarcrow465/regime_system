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
# MULTI-OBJECTIVE REGIME OPTIMIZATION FRAMEWORK
# Optimize regime classification thresholds for: Sharpe + Drawdown + Persistence
# Based on your original Week 1 plan for institutional-level optimization
# =============================================================================

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Any
from scipy.optimize import differential_evolution, minimize
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

@dataclass
class OptimizationResults:
    """Results from multi-objective optimization"""
    best_params: Dict[str, float]
    best_score: float
    sharpe_ratio: float
    max_drawdown: float
    regime_persistence: float
    strategy_returns: pd.Series
    regime_classifications: pd.DataFrame
    optimization_history: List[Dict]

class RegimeStrategyBacktester:
    """
    Simple strategy backtesting for regime validation
    Tests basic strategies on each regime to measure performance
    """
    
    def __init__(self):
        self.strategies = {
            # Direction-based strategies
            'up_trending_momentum': self.momentum_strategy,
            'down_trending_short': self.short_momentum_strategy,
            'sideways_mean_reversion': self.mean_reversion_strategy,
            
            # Volatility-based strategies
            'high_vol_breakout': self.volatility_breakout_strategy,
            'low_vol_trend': self.trend_following_strategy,
            
            # Composite strategies
            'regime_adaptive': self.adaptive_regime_strategy
        }
    
    def momentum_strategy(self, data: pd.DataFrame, regime_mask: pd.Series) -> pd.Series:
        """Simple momentum strategy for trending regimes"""
        try:
            returns = data['close'].pct_change()
            
            # Simple momentum: buy when EMA12 > EMA26
            signal = (data['EMA_12'] > data['EMA_26']).astype(int)
            
            # Apply only during specified regime
            signal = signal * regime_mask
            
            # Calculate strategy returns
            strategy_returns = signal.shift(1) * returns
            return strategy_returns.fillna(0)
            
        except Exception as e:
            logger.warning(f"Momentum strategy error: {e}")
            return pd.Series(0, index=data.index)
    
    def short_momentum_strategy(self, data: pd.DataFrame, regime_mask: pd.Series) -> pd.Series:
        """Short momentum for down trending periods"""
        try:
            returns = data['close'].pct_change()
            
            # Short when EMA12 < EMA26
            signal = (data['EMA_12'] < data['EMA_26']).astype(int) * -1
            
            # Apply only during specified regime
            signal = signal * regime_mask
            
            strategy_returns = signal.shift(1) * returns
            return strategy_returns.fillna(0)
            
        except Exception as e:
            logger.warning(f"Short momentum strategy error: {e}")
            return pd.Series(0, index=data.index)
    
    def mean_reversion_strategy(self, data: pd.DataFrame, regime_mask: pd.Series) -> pd.Series:
        """Mean reversion strategy for sideways markets"""
        try:
            returns = data['close'].pct_change()
            
            # Mean reversion using RSI
            signal = np.where(data['RSI'] < 30, 1,  # Buy oversold
                             np.where(data['RSI'] > 70, -1, 0))  # Sell overbought
            
            signal = pd.Series(signal, index=data.index) * regime_mask
            
            strategy_returns = signal.shift(1) * returns
            return strategy_returns.fillna(0)
            
        except Exception as e:
            logger.warning(f"Mean reversion strategy error: {e}")
            return pd.Series(0, index=data.index)
    
    def volatility_breakout_strategy(self, data: pd.DataFrame, regime_mask: pd.Series) -> pd.Series:
        """Volatility breakout for high volatility periods"""
        try:
            returns = data['close'].pct_change()
            
            # Buy breakouts during high volatility
            price_change = data['close'].pct_change()
            signal = np.where(abs(price_change) > data['ATR'] / data['close'] * 0.5, 
                            np.sign(price_change), 0)
            
            signal = pd.Series(signal, index=data.index) * regime_mask
            
            strategy_returns = signal.shift(1) * returns
            return strategy_returns.fillna(0)
            
        except Exception as e:
            logger.warning(f"Volatility breakout strategy error: {e}")
            return pd.Series(0, index=data.index)
    
    def trend_following_strategy(self, data: pd.DataFrame, regime_mask: pd.Series) -> pd.Series:
        """Trend following for low volatility periods"""
        try:
            returns = data['close'].pct_change()
            
            # Follow longer-term trend during low volatility
            signal = np.where(data['SMA_20'] > data['SMA_50'], 1,
                            np.where(data['SMA_20'] < data['SMA_50'], -1, 0))
            
            signal = pd.Series(signal, index=data.index) * regime_mask
            
            strategy_returns = signal.shift(1) * returns
            return strategy_returns.fillna(0)
            
        except Exception as e:
            logger.warning(f"Trend following strategy error: {e}")
            return pd.Series(0, index=data.index)
    
    def adaptive_regime_strategy(self, data: pd.DataFrame, regimes: pd.DataFrame) -> pd.Series:
        """Adaptive strategy that switches based on regime combination"""
        try:
            total_returns = pd.Series(0, index=data.index)
            
            # Define regime-specific strategies
            regime_strategies = {
                'Up_Trending': ('momentum', 1.0),
                'Down_Trending': ('short_momentum', 1.0),
                'Sideways': ('mean_reversion', 0.5),
                'High_Vol': ('volatility_breakout', 0.8),
                'Low_Vol': ('trend_following', 1.0)
            }
            
            # Apply strategies based on regimes
            for regime_type, (strategy_name, weight) in regime_strategies.items():
                if regime_type in ['Up_Trending', 'Down_Trending', 'Sideways']:
                    mask = regimes['Direction_Regime'] == regime_type
                elif regime_type in ['High_Vol', 'Low_Vol']:
                    mask = regimes['Volatility_Regime'].str.contains(regime_type.replace('_', '_Vol'))
                else:
                    continue
                
                if strategy_name == 'momentum':
                    strategy_returns = self.momentum_strategy(data, mask) * weight
                elif strategy_name == 'short_momentum':
                    strategy_returns = self.short_momentum_strategy(data, mask) * weight
                elif strategy_name == 'mean_reversion':
                    strategy_returns = self.mean_reversion_strategy(data, mask) * weight
                elif strategy_name == 'volatility_breakout':
                    strategy_returns = self.volatility_breakout_strategy(data, mask) * weight
                elif strategy_name == 'trend_following':
                    strategy_returns = self.trend_following_strategy(data, mask) * weight
                
                total_returns += strategy_returns
            
            return total_returns
            
        except Exception as e:
            logger.warning(f"Adaptive regime strategy error: {e}")
            return pd.Series(0, index=data.index)

class MultiObjectiveRegimeOptimizer:
    """
    Multi-Objective Optimization Framework for Regime Classification
    Optimizes thresholds for: Sharpe Ratio + Max Drawdown + Regime Persistence
    """
    
    def __init__(self, regime_classifier, data: pd.DataFrame):
        self.classifier = regime_classifier
        self.data = data
        self.backtester = RegimeStrategyBacktester()
        self.optimization_history = []
        self.max_iterations = 100  # Default value
        self.function_call_count = 0  # Add this counter
        
        # Define parameter bounds for optimization
        self.param_bounds = {
            # Direction thresholds
            'direction_strong_slope': (0.01, 0.05),  # 1% to 5%
            'direction_weak_slope': (0.002, 0.01),   # 0.2% to 1%
            
            # Trend strength thresholds
            'trend_strong_alignment': (0.005, 0.02), # 0.5% to 2%
            'trend_moderate_alignment': (0.002, 0.01), # 0.2% to 1%
            
            # Velocity thresholds
            'velocity_acceleration': (0.01, 0.03),   # 1% to 3%
            'velocity_stable_range': (0.002, 0.01),  # 0.2% to 1%
            
            # Volatility percentile thresholds
            'volatility_high_percentile': (70, 90),  # 70th to 90th percentile
            'volatility_low_percentile': (10, 40),   # 10th to 40th percentile
            
            # Microstructure thresholds
            'microstructure_institutional': (1.2, 2.0), # Volume ratio thresholds
            'microstructure_retail': (0.5, 0.9)
        }
        
        # Optimization weights
        self.objective_weights = {
            'sharpe_ratio': 0.4,
            'max_drawdown': 0.3,  # Want to minimize this
            'regime_persistence': 0.3
        }
    
    def update_classifier_thresholds(self, params: Dict[str, float]):
        """Update classifier thresholds with new parameters"""
        try:
            # Update direction thresholds
            self.classifier.dimension_thresholds['direction']['strong_trend_slope'] = params['direction_strong_slope']
            self.classifier.dimension_thresholds['direction']['weak_trend_slope'] = params['direction_weak_slope']
            
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
        """Calculate regime persistence score - higher is better"""
        try:
            persistence_scores = []
            
            # Calculate persistence for each dimension
            for dimension in ['Direction_Regime', 'TrendStrength_Regime', 'Velocity_Regime', 
                            'Volatility_Regime', 'Microstructure_Regime']:
                if dimension in regimes.columns:
                    # Count regime changes
                    regime_changes = (regimes[dimension] != regimes[dimension].shift()).sum()
                    total_periods = len(regimes)
                    
                    # Persistence = 1 - (changes / total_periods)
                    persistence = 1 - (regime_changes / total_periods)
                    persistence_scores.append(persistence)
            
            # Return average persistence across all dimensions
            return np.mean(persistence_scores) if persistence_scores else 0.0
            
        except Exception as e:
            logger.warning(f"Error calculating regime persistence: {e}")
            return 0.0
    
    def calculate_performance_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """Calculate strategy performance metrics"""
        try:
            if len(returns) == 0 or returns.sum() == 0:
                return {'sharpe_ratio': 0.0, 'max_drawdown': -1.0, 'total_return': 0.0}
            
            # Remove NaN values
            clean_returns = returns.dropna()
            
            if len(clean_returns) == 0:
                return {'sharpe_ratio': 0.0, 'max_drawdown': -1.0, 'total_return': 0.0}
            
            # Calculate Sharpe ratio (annualized)
            mean_return = clean_returns.mean()
            std_return = clean_returns.std()
            sharpe_ratio = (mean_return / std_return) * np.sqrt(252 * 24 * 4) if std_return > 0 else 0  # 15-min periods
            
            # Calculate maximum drawdown
            cumulative_returns = (1 + clean_returns).cumprod()
            rolling_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - rolling_max) / rolling_max
            max_drawdown = drawdown.min()
            
            # Total return
            total_return = cumulative_returns.iloc[-1] - 1 if len(cumulative_returns) > 0 else 0
            
            return {
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'total_return': total_return
            }
            
        except Exception as e:
            logger.warning(f"Error calculating performance metrics: {e}")
            return {'sharpe_ratio': 0.0, 'max_drawdown': -1.0, 'total_return': 0.0}
    
    def objective_function(self, params_array: np.ndarray) -> float:
        try:
            # Increment and check function call count
            self.function_call_count += 1
            if self.function_call_count > self.max_iterations:
                print(f"STOPPING: Reached {self.max_iterations} function calls")
                return 1.0
            
            print(f"Function call {self.function_call_count}/{self.max_iterations}")
            # Convert parameter array to dictionary
            param_names = list(self.param_bounds.keys())
            params = dict(zip(param_names, params_array))
            
            # Update classifier with new parameters
            self.update_classifier_thresholds(params)
            
            # Re-classify regimes with new thresholds
            regimes_df = self.classifier.classify_multidimensional_regime(self.data.copy())
            
            # Test adaptive regime strategy
            strategy_returns = self.backtester.adaptive_regime_strategy(self.data, regimes_df)
            
            # Calculate performance metrics
            performance = self.calculate_performance_metrics(strategy_returns)
            
            # Calculate regime persistence
            persistence = self.calculate_regime_persistence(regimes_df)
            
            # Multi-objective score (convert to minimization problem)
            sharpe_component = -performance['sharpe_ratio']  # Negative because we want to maximize
            drawdown_component = -performance['max_drawdown']  # Negative drawdown is good
            persistence_component = -persistence  # Negative because we want to maximize
            
            # Weighted combination
            total_score = (
                self.objective_weights['sharpe_ratio'] * sharpe_component +
                self.objective_weights['max_drawdown'] * drawdown_component +
                self.objective_weights['regime_persistence'] * persistence_component
            )
            
            # Store optimization history
            self.optimization_history.append({
                'params': params.copy(),
                'sharpe_ratio': performance['sharpe_ratio'],
                'max_drawdown': performance['max_drawdown'],
                'regime_persistence': persistence,
                'total_score': -total_score,  # Convert back to maximization for reporting
                'total_return': performance['total_return']
            })
            
            # Log progress periodically and check for early stopping
            if len(self.optimization_history) % 5 == 0:  # Log every 5 iterations
                print(f"Iteration {len(self.optimization_history)}: Score={-total_score:.4f}, Sharpe={performance['sharpe_ratio']:.4f}")
                logger.info(f"Optimization iteration {len(self.optimization_history)}: "
                          f"Score={-total_score:.4f}, Sharpe={performance['sharpe_ratio']:.4f}, "
                          f"DD={performance['max_drawdown']:.4f}, Persistence={persistence:.4f}")
            
            # Early stopping if we exceed expected iterations
            if len(self.optimization_history) > 200:  # Fixed limit instead of max_iterations * 2
                print(f"WARNING: Exceeded 200 iterations, stopping optimization...")
                return 10.0
            
            return total_score
            
        except Exception as e:
            logger.error(f"Error in objective function: {e}")
            return 1000.0  # Return high penalty for errors
    
    def optimize_regime_thresholds(self, method: str = 'differential_evolution', 
                                 max_iterations: int = 100) -> OptimizationResults:
        """Run multi-objective optimization to find best regime thresholds"""
        
        # Store max_iterations for use in objective function
        self.max_iterations = max_iterations
        
        logger.info("Starting multi-objective regime threshold optimization...")
        logger.info(f"Optimizing for: {list(self.objective_weights.keys())} with weights {list(self.objective_weights.values())}")
        
        try:
            # Prepare bounds for scipy
            bounds = list(self.param_bounds.values())
            param_names = list(self.param_bounds.keys())
            
            # Initial guess (midpoint of bounds)
            initial_guess = [(b[0] + b[1]) / 2 for b in bounds]
            
            logger.info(f"Parameter bounds: {dict(zip(param_names, bounds))}")
            logger.info(f"Starting optimization with {len(bounds)} parameters...")
            
            # Run optimization with strict limits
            if method == 'differential_evolution':
                result = differential_evolution(
                    self.objective_function,
                    bounds,
                    maxiter=max_iterations,
                    popsize=5,  # Reduced from 10 to 5
                    seed=42,
                    disp=True,
                    workers=1,  # Force single-threaded
                    updating='immediate'  # Faster convergence
                )
            else:
                # Alternative: L-BFGS-B
                result = minimize(
                    self.objective_function,
                    initial_guess,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': max_iterations}
                )
            
            # Extract best parameters
            best_params = dict(zip(param_names, result.x))
            
            # Update classifier with best parameters
            self.update_classifier_thresholds(best_params)
            
            # Generate final results with best parameters
            final_regimes = self.classifier.classify_multidimensional_regime(self.data.copy())
            final_returns = self.backtester.adaptive_regime_strategy(self.data, final_regimes)
            final_performance = self.calculate_performance_metrics(final_returns)
            final_persistence = self.calculate_regime_persistence(final_regimes)
            
            logger.info("Optimization completed!")
            logger.info(f"Best Score: {-result.fun:.4f}")
            logger.info(f"Best Sharpe: {final_performance['sharpe_ratio']:.4f}")
            logger.info(f"Best Max Drawdown: {final_performance['max_drawdown']:.4f}")
            logger.info(f"Best Regime Persistence: {final_persistence:.4f}")
            
            return OptimizationResults(
                best_params=best_params,
                best_score=-result.fun,
                sharpe_ratio=final_performance['sharpe_ratio'],
                max_drawdown=final_performance['max_drawdown'],
                regime_persistence=final_persistence,
                strategy_returns=final_returns,
                regime_classifications=final_regimes,
                optimization_history=self.optimization_history
            )
            
        except Exception as e:
            logger.error(f"Optimization failed: {e}")
            # Return default results
            return OptimizationResults(
                best_params={},
                best_score=0.0,
                sharpe_ratio=0.0,
                max_drawdown=-1.0,
                regime_persistence=0.0,
                strategy_returns=pd.Series(),
                regime_classifications=pd.DataFrame(),
                optimization_history=self.optimization_history
            )

def optimize_window_size(data: pd.DataFrame, classifier, timeframe: str = '15min', 
                        window_sizes_hours: List[float] = None) -> Dict:
    """
    Optimize rolling window size for a specific timeframe
    """
    import multidimensional_regime_system as mds
    
    # Convert hours to periods based on timeframe
    timeframe_multipliers = {
        '5min': 12,   # 12 periods per hour
        '15min': 4,   # 4 periods per hour  
        '1H': 1,      # 1 period per hour
        'D': 1/24     # 1 period per 24 hours
    }
    
    if window_sizes_hours is None:
        if timeframe == '5min':
            window_sizes_hours = [1, 2, 3, 4, 6, 8]
        elif timeframe == '15min':
            window_sizes_hours = [2, 4, 6, 8, 12, 16, 24, 36]
        elif timeframe == '1H':
            window_sizes_hours = [24, 36, 48, 72, 96]
        else:  # Daily
            window_sizes_hours = [240, 360, 480, 720, 960]
    
    results = {}
    multiplier = timeframe_multipliers.get(timeframe, 4)
    
    print(f"\n{'='*60}")
    print(f"WINDOW SIZE OPTIMIZATION FOR {timeframe}")
    print(f"{'='*60}")
    
    # Save original window configuration
    original_window_hours = mds.ROLLING_WINDOW_HOURS if hasattr(mds, 'ROLLING_WINDOW_HOURS') else 8
    
    for window_hours in window_sizes_hours:
        window_periods = int(window_hours * multiplier)
        print(f"\nTesting {window_hours} hour window ({window_periods} periods)...")
        
        try:
            # Update the global configuration
            mds.ROLLING_WINDOW_HOURS = window_hours
            
            # Create a new classifier with the new window size
            new_classifier = mds.MultiDimensionalRegimeClassifier(window_hours=window_hours)
            print(f"  Created classifier with {window_hours} hour window")
            
            # Run classification
            data_with_regimes = new_classifier.classify_multidimensional_regime(
                data.copy(), 
                symbol='NQ'
            )
            
            # Run simple backtest
            backtester = RegimeStrategyBacktester()
            strategy_returns = backtester.adaptive_regime_strategy(data, data_with_regimes)
            
            # Calculate metrics directly
            # Sharpe Ratio
            returns_mean = strategy_returns.mean()
            returns_std = strategy_returns.std()
            sharpe = (returns_mean / returns_std * np.sqrt(252 * 96)) if returns_std > 0 else 0  # Annualized for 15min data
            
            # Max Drawdown
            cumulative_returns = (1 + strategy_returns).cumprod()
            running_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - running_max) / running_max
            max_dd = drawdown.min()
            
            # Total Return
            total_return = (1 + strategy_returns).prod() - 1
            
            # Calculate regime stability
            regime_changes = 0
            for col in data_with_regimes.columns:
                if '_Regime' in col:
                    regime_changes += (data_with_regimes[col] != data_with_regimes[col].shift()).sum()
            
            avg_regime_changes = regime_changes / 5  # 5 dimensions
            regime_persistence = 1 - (avg_regime_changes / len(data_with_regimes))
            
            results[window_hours] = {
                'window_periods': window_periods,
                'sharpe_ratio': sharpe,
                'max_drawdown': max_dd,
                'total_return': total_return,
                'regime_persistence': regime_persistence,
                'combined_score': sharpe * 0.4 - max_dd * 0.3 + regime_persistence * 0.3
            }
            
            print(f"  Sharpe: {sharpe:.3f}, MaxDD: {max_dd:.1%}, Persistence: {regime_persistence:.1%}")
            
        except Exception as e:
            print(f"  Error with {window_hours}h window: {e}")
            import traceback
            traceback.print_exc()
            results[window_hours] = {'combined_score': -999}
    
    # Restore original configuration
    mds.ROLLING_WINDOW_HOURS = original_window_hours
    
    # Find best window
    valid_results = {k: v for k, v in results.items() if v['combined_score'] != -999}
    if valid_results:
        best_window = max(valid_results.keys(), key=lambda x: valid_results[x]['combined_score'])
        print(f"\n{'='*60}")
        print(f"OPTIMAL WINDOW: {best_window} hours")
        print(f"Performance: {valid_results[best_window]}")
        print(f"{'='*60}\n")
        
        return {
            'optimal_window_hours': best_window,
            'optimal_window_periods': valid_results[best_window]['window_periods'],
            'all_results': results
        }
    else:
        print("\nERROR: No valid results obtained!")
        return {
            'optimal_window_hours': original_window_hours,
            'optimal_window_periods': int(original_window_hours * multiplier),
            'all_results': results
        }

def apply_params_to_classifier(classifier, params: Dict[str, float]):
    """Apply optimized parameters to a classifier instance"""
    try:
        # Update direction thresholds
        if 'direction_strong_slope' in params:
            classifier.dimension_thresholds['direction']['strong_trend_slope'] = params['direction_strong_slope']
        if 'direction_weak_slope' in params:
            classifier.dimension_thresholds['direction']['weak_trend_slope'] = params['direction_weak_slope']
        
        # Update trend strength thresholds
        if 'trend_strong_alignment' in params:
            classifier.dimension_thresholds['trend_strength']['strong_alignment'] = params['trend_strong_alignment']
        if 'trend_moderate_alignment' in params:
            classifier.dimension_thresholds['trend_strength']['moderate_alignment'] = params['trend_moderate_alignment']
        
        # Update velocity thresholds
        if 'velocity_acceleration' in params:
            classifier.dimension_thresholds['velocity']['acceleration_threshold'] = params['velocity_acceleration']
        if 'velocity_stable_range' in params:
            classifier.dimension_thresholds['velocity']['stable_range'] = params['velocity_stable_range']
        
        # Update volatility thresholds
        if 'volatility_high_percentile' in params:
            classifier.dimension_thresholds['volatility']['percentile']['high'] = params['volatility_high_percentile']
        if 'volatility_low_percentile' in params:
            classifier.dimension_thresholds['volatility']['percentile']['low'] = params['volatility_low_percentile']
        
        # Update microstructure thresholds
        if 'microstructure_institutional' in params:
            classifier.dimension_thresholds['microstructure']['institutional_volume_threshold'] = params['microstructure_institutional']
        if 'microstructure_retail' in params:
            classifier.dimension_thresholds['microstructure']['retail_volume_threshold'] = params['microstructure_retail']
            
    except Exception as e:
        logger.error(f"Error applying parameters to classifier: {e}")
    
class WalkForwardOptimizer:
    """
    Walk-Forward Analysis for regime optimization
    Prevents overfitting by training on past data and testing on future data
    """
    
    def __init__(self, data: pd.DataFrame, train_periods: int = 252*2, test_periods: int = 252):
        """
        Args:
            data: Full dataset
            train_periods: Number of periods for training (default 2 years for daily)
            test_periods: Number of periods for testing (default 1 year for daily)
        """
        self.data = data
        self.train_periods = train_periods
        self.test_periods = test_periods
        self.results = []
        
    def run_walk_forward_optimization(self, classifier, optimization_method: str = 'differential_evolution',
                                    max_iterations: int = 20) -> pd.DataFrame:
        """
        Run walk-forward optimization
        """
        print("\n" + "="*80)
        print("WALK-FORWARD OPTIMIZATION")
        print("="*80)
        print(f"Train periods: {self.train_periods}, Test periods: {self.test_periods}")
        print(f"Total windows: {self._calculate_windows()}")
        print("="*80)
        
        results = []
        window_num = 1
        
        # Calculate total possible windows
        total_periods = len(self.data)
        step_size = self.test_periods  # Non-overlapping test windows
        
        start_idx = 0
        while start_idx + self.train_periods + self.test_periods <= total_periods:
            print(f"\n--- Window {window_num} ---")
            
            # Split data
            train_end = start_idx + self.train_periods
            test_end = train_end + self.test_periods
            
            train_data = self.data.iloc[start_idx:train_end].copy()
            test_data = self.data.iloc[train_end:test_end].copy()
            
            train_dates = f"{train_data.index[0].strftime('%Y-%m-%d')} to {train_data.index[-1].strftime('%Y-%m-%d')}"
            test_dates = f"{test_data.index[0].strftime('%Y-%m-%d')} to {test_data.index[-1].strftime('%Y-%m-%d')}"
            
            print(f"Training: {train_dates} ({len(train_data)} periods)")
            print(f"Testing: {test_dates} ({len(test_data)} periods)")
            
            try:
                # Optimize on training data
                optimizer = MultiObjectiveRegimeOptimizer(classifier, train_data)
                opt_results = optimizer.optimize_regime_thresholds(
                    method=optimization_method,
                    max_iterations=max_iterations
                )
                
                # Test on out-of-sample data
                classifier_copy = self._copy_classifier(classifier)
                self._apply_parameters(classifier_copy, opt_results.best_params)

                from multidimensional_regime_system import calculate_all_indicators
                
                # Calculate test performance
                test_data_with_indicators = calculate_all_indicators(test_data.copy())
                test_data_with_regimes = classifier_copy.classify_multidimensional_regime(test_data_with_indicators, symbol='NQ')  # or pass the appropriate symbol
                
                backtester = RegimeStrategyBacktester()
                test_returns = backtester.adaptive_regime_strategy(test_data_with_regimes, test_data_with_regimes)
                
                optimizer = MultiObjectiveRegimeOptimizer(classifier, test_data_with_indicators)
                performance_metrics = optimizer.calculate_performance_metrics(test_returns)
                test_sharpe = performance_metrics['sharpe_ratio']
                test_max_dd = performance_metrics['max_drawdown']
                test_return = performance_metrics['total_return']
                
                window_result = {
                    'window': window_num,
                    'train_start': train_data.index[0],
                    'train_end': train_data.index[-1],
                    'test_start': test_data.index[0],
                    'test_end': test_data.index[-1],
                    'train_sharpe': opt_results.sharpe_ratio,
                    'train_max_dd': opt_results.max_drawdown,
                    'test_sharpe': test_sharpe,
                    'test_max_dd': test_max_dd,
                    'test_return': test_return,
                    'overfit_ratio': opt_results.sharpe_ratio / test_sharpe if test_sharpe != 0 else 999
                }
                
                results.append(window_result)
                
                print(f"Train Sharpe: {opt_results.sharpe_ratio:.3f}, Test Sharpe: {test_sharpe:.3f}")
                print(f"Overfit Ratio: {window_result['overfit_ratio']:.2f}")
                
            except Exception as e:
                print(f"Error in window {window_num}: {e}")
                import traceback
                traceback.print_exc()
            
            # Move to next window
            start_idx += step_size
            window_num += 1
        
        # Create summary
        results_df = pd.DataFrame(results)
        
        print("\n" + "="*80)
        print("WALK-FORWARD SUMMARY")
        print("="*80)
        
        if len(results_df) > 0:
            print(f"Successful windows: {len(results_df)}")
            print(f"Average Train Sharpe: {results_df['train_sharpe'].mean():.3f}")
            print(f"Average Test Sharpe: {results_df['test_sharpe'].mean():.3f}")
            print(f"Average Overfit Ratio: {results_df['overfit_ratio'].mean():.2f}")
            print(f"Consistency (std of test sharpe): {results_df['test_sharpe'].std():.3f}")
        else:
            print("No successful windows completed!")
            print("Check error messages above for issues.")
            
        print("="*80)
        
        return results_df
    
    def _calculate_windows(self) -> int:
        """Calculate number of walk-forward windows"""
        total_periods = len(self.data)
        return (total_periods - self.train_periods) // self.test_periods
    
    def _copy_classifier(self, classifier):
        """Create a copy of the classifier"""
        import copy
        return copy.deepcopy(classifier)
    
    def _apply_parameters(self, classifier, params):
        """Apply optimized parameters to classifier"""
        apply_params_to_classifier(classifier, params)

def run_regime_optimization(classifier, data: pd.DataFrame, 
                          max_iterations: int = 50, 
                          method: str = 'differential_evolution') -> OptimizationResults:
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
    print("="*80)
    
    optimizer = MultiObjectiveRegimeOptimizer(classifier, data)
    results = optimizer.optimize_regime_thresholds(method=method, max_iterations=max_iterations)
    
    return results

def print_optimization_results(results: OptimizationResults):
    """Print comprehensive optimization results"""
    print("\n" + "="*80)
    print("OPTIMIZATION RESULTS")
    print("="*80)
    
    print(f"📊 PERFORMANCE METRICS:")
    print(f"   Final Score: {results.best_score:.4f}")
    print(f"   Sharpe Ratio: {results.sharpe_ratio:.4f}")
    print(f"   Max Drawdown: {results.max_drawdown:.2%}")
    print(f"   Regime Persistence: {results.regime_persistence:.2%}")
    
    if len(results.strategy_returns) > 0:
        total_return = (1 + results.strategy_returns).prod() - 1
        print(f"   Total Return: {total_return:.2%}")
    
    print(f"\n🎛️  OPTIMIZED PARAMETERS:")
    for param, value in results.best_params.items():
        print(f"   {param}: {value:.6f}")
    
    print(f"\n📈 OPTIMIZATION HISTORY:")
    if results.optimization_history:
        history_df = pd.DataFrame(results.optimization_history)
        print(f"   Total Iterations: {len(history_df)}")
        print(f"   Best Score Achieved: {history_df['total_score'].max():.4f}")
        print(f"   Score Improvement: {history_df['total_score'].max() - history_df['total_score'].iloc[0]:.4f}")
    
    print("="*80)

# %%
