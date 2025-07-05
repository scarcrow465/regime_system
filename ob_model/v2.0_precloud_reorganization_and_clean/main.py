#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
Main entry point for the Regime System
Demonstrates usage of the reorganized package structure
"""

import pandas as pd
import numpy as np
import logging
import os
import sys
from datetime import datetime
from typing import Dict, List, Optional, Tuple

# Add regime_system to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import from our organized modules
from core.data_loader import load_csv_data, prepare_data_for_analysis, get_data_info
from core.indicators import calculate_all_indicators, validate_indicators
from core.regime_classifier import RollingRegimeClassifier
from config.settings import (
    DEFAULT_WINDOW_HOURS, OPTIMIZATION_ITERATIONS, 
    RESULTS_DIR, LOG_FILE, DEFAULT_SYMBOLS, TIMEFRAMES,
    WALK_FORWARD_WINDOWS, OBJECTIVE_WEIGHTS
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# =============================================================================
# MAIN ANALYSIS FUNCTIONS
# =============================================================================

# 1

def run_regime_analysis(filepath: str, 
                       window_hours: float = DEFAULT_WINDOW_HOURS,
                       timeframe: str = '15min',
                       save_results: bool = True) -> pd.DataFrame:
    """
    Run complete regime analysis on data
    
    Args:
        filepath: Path to CSV data file
        window_hours: Rolling window size in hours
        timeframe: Data timeframe
        save_results: Whether to save results to file
        
    Returns:
        DataFrame with regime classifications
    """
    logger.info("="*80)
    logger.info("STARTING REGIME ANALYSIS")
    logger.info("="*80)
    logger.info(f"Data file: {filepath}")
    logger.info(f"Window: {window_hours} hours")
    logger.info(f"Timeframe: {timeframe}")
    
    try:
        # Step 1: Load data
        logger.info("\nStep 1: Loading data...")
        data = load_csv_data(filepath, timeframe=timeframe)
        data = prepare_data_for_analysis(data)
        
        data_info = get_data_info(data)
        logger.info(f"Loaded {data_info['rows']} rows from {data_info['start_date']} to {data_info['end_date']}")
        
        # Step 2: Calculate indicators
        logger.info("\nStep 2: Calculating indicators...")
        data_with_indicators = calculate_all_indicators(data, verbose=True)
        
        # Validate indicators
        validation = validate_indicators(data_with_indicators)
        logger.info(f"Valid indicators: {len(validation['valid'])}")
        if validation['missing']:
            logger.warning(f"Missing indicators: {validation['missing']}")
        
        # Step 3: Classify regimes
        logger.info("\nStep 3: Classifying regimes...")
        classifier = RollingRegimeClassifier(window_hours=window_hours, timeframe=timeframe)
        regimes = classifier.classify_regimes(data_with_indicators, show_progress=True)
        
        # Get regime statistics
        stats = classifier.get_regime_statistics(regimes)
        
        # Print summary
        print("\n" + "="*80)
        print("REGIME CLASSIFICATION SUMMARY")
        print("="*80)
        
        for dimension, dim_stats in stats.items():
            if dimension != 'Composite':
                print(f"\n{dimension} Dimension:")
                for regime, pct in dim_stats['percentages'].items():
                    if regime != 'Undefined':
                        print(f"  {regime}: {pct:.1f}%")
                print(f"  Average Confidence: {dim_stats['average_confidence']:.3f}")
        
        if 'Composite' in stats:
            print(f"\nComposite Regimes: {stats['Composite']['unique_regimes']} unique combinations")
            print("Top 5 Composite Regimes:")
            for regime, count in list(stats['Composite']['top_10_regimes'].items())[:5]:
                pct = (count / len(regimes)) * 100
                print(f"  {regime}: {pct:.1f}%")
        
        # Step 4: Save results
        if save_results:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = os.path.join(RESULTS_DIR, f"regime_analysis_{timestamp}.csv")
            
            # Combine data with regimes
            results = pd.concat([data_with_indicators, regimes], axis=1)
            results.to_csv(output_file)
            logger.info(f"\nResults saved to: {output_file}")
        
        return regimes
        
    except Exception as e:
        logger.error(f"Error in regime analysis: {e}")
        raise

def run_optimization(filepath: str,
                    timeframe: str = '15min',
                    optimize_window: bool = True,
                    walk_forward: bool = True) -> Dict:
    """
    Run regime optimization
    
    Args:
        filepath: Path to CSV data file
        timeframe: Data timeframe
        optimize_window: Whether to optimize window size
        walk_forward: Whether to use walk-forward validation
        
    Returns:
        Optimization results
    """
    logger.info("="*80)
    logger.info("STARTING REGIME OPTIMIZATION")
    logger.info("="*80)
    
    try:
        # Import optimization modules (lazy import to avoid circular dependencies)
        from optimization.multi_objective import MultiObjectiveRegimeOptimizer
        from optimization.window_optimizer import optimize_window_size
        from optimization.walk_forward import WalkForwardOptimizer
        
        # Load and prepare data
        data = load_csv_data(filepath, timeframe=timeframe)
        data = prepare_data_for_analysis(data)
        data_with_indicators = calculate_all_indicators(data)
        
        # Step 1: Window optimization (if requested)
        optimal_window = DEFAULT_WINDOW_HOURS
        if optimize_window:
            logger.info("\nStep 1: Optimizing window size...")
            window_results = optimize_window_size(
                data_with_indicators,
                timeframe=timeframe,
                window_sizes_hours=[12, 24, 36, 48, 72]
            )
            optimal_window = window_results['best_window']
            logger.info(f"Optimal window: {optimal_window} hours")
        
        # Step 2: Regime optimization
        if walk_forward:
            logger.info("\nStep 2: Running walk-forward optimization...")
            wf_optimizer = WalkForwardOptimizer(
                n_windows=WALK_FORWARD_WINDOWS,
                train_ratio=0.67
            )
            results = wf_optimizer.run_walk_forward(
                data_with_indicators,
                window_hours=optimal_window,
                timeframe=timeframe,
                max_iterations=OPTIMIZATION_ITERATIONS
            )
        else:
            logger.info("\nStep 2: Running single optimization...")
            classifier = RollingRegimeClassifier(window_hours=optimal_window, timeframe=timeframe)
            optimizer = MultiObjectiveRegimeOptimizer(classifier, data_with_indicators)
            results = optimizer.optimize_regime_thresholds(
                method='differential_evolution',
                max_iterations=OPTIMIZATION_ITERATIONS
            )
        
        # Print results
        print("\n" + "="*80)
        print("OPTIMIZATION RESULTS")
        print("="*80)
        print(f"Best Score: {results.get('best_score', 0):.4f}")
        print(f"Sharpe Ratio: {results.get('sharpe_ratio', 0):.4f}")
        print(f"Max Drawdown: {results.get('max_drawdown', 0):.4f}")
        print(f"Regime Persistence: {results.get('regime_persistence', 0):.4f}")
        
        if 'best_params' in results:
            print("\nOptimized Parameters:")
            for param, value in results['best_params'].items():
                print(f"  {param}: {value:.4f}")
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = os.path.join(RESULTS_DIR, f"optimization_results_{timestamp}.json")
        
        import json
        with open(results_file, 'w') as f:
            # Convert non-serializable objects
            save_results = {k: v for k, v in results.items() 
                          if not isinstance(v, (pd.DataFrame, pd.Series))}
            json.dump(save_results, f, indent=2, default=str)
        
        logger.info(f"\nResults saved to: {results_file}")
        
        return results
        
    except Exception as e:
        logger.error(f"Error in optimization: {e}")
        raise

def run_backtesting(filepath: str,
                   regime_file: Optional[str] = None,
                   timeframe: str = '15min') -> Dict:
    """
    Run backtesting with regime-based strategies
    
    Args:
        filepath: Path to CSV data file
        regime_file: Path to saved regime classifications (optional)
        timeframe: Data timeframe
        
    Returns:
        Backtesting results
    """
    logger.info("="*80)
    logger.info("STARTING REGIME BACKTESTING")
    logger.info("="*80)
    
    try:
        # Import backtesting module
        from backtesting.strategies import EnhancedRegimeStrategyBacktester
        
        # Load data
        data = load_csv_data(filepath, timeframe=timeframe)
        data = prepare_data_for_analysis(data)
        data_with_indicators = calculate_all_indicators(data)
        
        # Load or calculate regimes
        if regime_file and os.path.exists(regime_file):
            logger.info(f"Loading regimes from: {regime_file}")
            all_data = pd.read_csv(regime_file, index_col=0, parse_dates=True)
            regime_columns = [col for col in all_data.columns if 'Regime' in col or 'Confidence' in col]
            regimes = all_data[regime_columns]
        else:
            logger.info("Calculating regimes...")
            classifier = RollingRegimeClassifier(window_hours=DEFAULT_WINDOW_HOURS, timeframe=timeframe)
            regimes = classifier.classify_regimes(data_with_indicators)
        
        # Run backtesting
        logger.info("\nRunning backtesting...")
        backtester = EnhancedRegimeStrategyBacktester()
        
        # Test adaptive strategy
        returns = backtester.adaptive_regime_strategy_enhanced(data_with_indicators, regimes)
        metrics = backtester.calculate_performance_metrics(returns)
        
        # Print results
        print("\n" + "="*80)
        print("BACKTESTING RESULTS - Adaptive Regime Strategy")
        print("="*80)
        print(f"Total Return: {metrics['total_return']:.2%}")
        print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.4f}")
        print(f"Max Drawdown: {metrics['max_drawdown']:.2%}")
        print(f"Win Rate: {metrics['win_rate']:.2%}")
        print(f"Calmar Ratio: {metrics['calmar_ratio']:.4f}")
        print(f"Sortino Ratio: {metrics['sortino_ratio']:.4f}")
        
        return metrics
        
    except Exception as e:
        logger.error(f"Error in backtesting: {e}")
        raise

# =============================================================================
# COMMAND LINE INTERFACE
# =============================================================================

def main():
    """Main CLI entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Regime System - Institutional Grade Analysis')
    parser.add_argument('command', choices=['analyze', 'optimize', 'backtest'],
                       help='Command to run')
    parser.add_argument('--data', required=True, help='Path to CSV data file')
    parser.add_argument('--window', type=float, default=DEFAULT_WINDOW_HOURS,
                       help='Window size in hours')
    parser.add_argument('--timeframe', default='15min',
                       choices=['5min', '15min', '30min', '1H', '4H', 'Daily'],
                       help='Data timeframe')
    parser.add_argument('--optimize-window', action='store_true',
                       help='Optimize window size')
    parser.add_argument('--walk-forward', action='store_true',
                       help='Use walk-forward validation')
    parser.add_argument('--regime-file', help='Path to saved regime classifications')
    parser.add_argument('--no-save', action='store_true',
                       help='Do not save results')
    
    args = parser.parse_args()
    
    try:
        if args.command == 'analyze':
            run_regime_analysis(
                args.data,
                window_hours=args.window,
                timeframe=args.timeframe,
                save_results=not args.no_save
            )
        
        elif args.command == 'optimize':
            run_optimization(
                args.data,
                timeframe=args.timeframe,
                optimize_window=args.optimize_window,
                walk_forward=args.walk_forward
            )
        
        elif args.command == 'backtest':
            run_backtesting(
                args.data,
                regime_file=args.regime_file,
                timeframe=args.timeframe
            )
        
        logger.info("\nProcess completed successfully!")
        
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)

# =============================================================================
# EXAMPLE USAGE
# =============================================================================

def example_usage():
    """Show example usage of the regime system"""
    print("""
    # Example Usage:
    
    # 1. Basic regime analysis
    python main.py analyze --data "path/to/data.csv" --timeframe 15min
    
    # 2. Optimize with window size optimization
    python main.py optimize --data "path/to/data.csv" --optimize-window
    
    # 3. Walk-forward optimization
    python main.py optimize --data "path/to/data.csv" --walk-forward
    
    # 4. Backtest with saved regimes
    python main.py backtest --data "path/to/data.csv" --regime-file "results/regimes.csv"
    
    # 5. Custom window analysis
    python main.py analyze --data "path/to/data.csv" --window 48 --timeframe 1H
    """)

if __name__ == "__main__":
    # Check if no arguments provided
    if len(sys.argv) == 1:
        example_usage()
    else:
        main()

