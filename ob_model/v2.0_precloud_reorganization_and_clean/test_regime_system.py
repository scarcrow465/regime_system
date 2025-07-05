#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
Test script for the reorganized regime system
Run this to verify all components work correctly
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import traceback

# Add regime_system to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Track test results
test_results = {
    'passed': 0,
    'failed': 0,
    'errors': []
}

def test_function(test_name):
    """Decorator for test functions"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            print(f"\n{'='*60}")
            print(f"Testing: {test_name}")
            print('='*60)
            try:
                result = func(*args, **kwargs)
                test_results['passed'] += 1
                print(f"✓ {test_name} PASSED")
                return result
            except Exception as e:
                test_results['failed'] += 1
                test_results['errors'].append({
                    'test': test_name,
                    'error': str(e),
                    'traceback': traceback.format_exc()
                })
                print(f"✗ {test_name} FAILED: {e}")
                return None
        return wrapper
    return decorator

# =============================================================================
# TEST DATA GENERATION
# =============================================================================

def generate_test_data(n_periods=1000):
    """Generate synthetic test data"""
    # Create date range
    dates = pd.date_range(end=datetime.now(), periods=n_periods, freq='15min')
    
    # Generate synthetic price data
    np.random.seed(42)
    returns = np.random.normal(0.0001, 0.01, n_periods)
    close = 100 * np.exp(np.cumsum(returns))
    
    # Generate OHLC
    high = close * (1 + np.abs(np.random.normal(0, 0.002, n_periods)))
    low = close * (1 - np.abs(np.random.normal(0, 0.002, n_periods)))
    open_ = np.roll(close, 1)
    open_[0] = close[0]
    
    # Generate volume
    volume = np.random.lognormal(10, 0.5, n_periods)
    
    # Create DataFrame
    data = pd.DataFrame({
        'open': open_,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    }, index=dates)
    
    return data

# =============================================================================
# COMPONENT TESTS
# =============================================================================

@test_function("Core Imports")
def test_core_imports():
    """Test that core modules can be imported"""
    from regime_system.core.data_loader import load_csv_data
    from regime_system.core.indicators import calculate_all_indicators
    from regime_system.core.regime_classifier import RollingRegimeClassifier
    print("All core imports successful")

@test_function("Data Loader")
def test_data_loader():
    """Test data loading functionality"""
    from regime_system.core.data_loader import (
        prepare_data_for_analysis, 
        get_data_info,
        check_data_quality
    )
    
    # Generate test data
    data = generate_test_data(1000)
    
    # Prepare data
    prepared_data = prepare_data_for_analysis(data)
    assert 'returns' in prepared_data.columns
    assert 'log_returns' in prepared_data.columns
    
    # Get data info
    info = get_data_info(prepared_data)
    assert info['rows'] == 1000
    assert 'timeframe' in info
    
    # Check quality
    quality = check_data_quality(prepared_data)
    assert 'quality_score' in quality
    
    print(f"Data prepared with {len(prepared_data)} rows")
    print(f"Data quality score: {quality['quality_score']:.2f}")

@test_function("Indicator Calculations")
def test_indicators():
    """Test indicator calculation"""
    from regime_system.core.indicators import (
        calculate_all_indicators,
        validate_indicators,
        get_indicator_info
    )
    
    # Generate test data
    data = generate_test_data(500)
    
    # Calculate indicators
    data_with_indicators = calculate_all_indicators(data, verbose=False)
    
    # Validate
    validation = validate_indicators(data_with_indicators)
    
    # Get info
    info = get_indicator_info()
    
    print(f"Calculated {len(data_with_indicators.columns) - len(data.columns)} indicators")
    print(f"Valid indicators: {len(validation['valid'])}")
    print(f"Missing indicators: {len(validation['missing'])}")
    print(f"Total indicators in system: {info['total_indicators']}")
    
    assert len(validation['valid']) > 50  # Should have many valid indicators

@test_function("Regime Classification")
def test_regime_classification():
    """Test regime classification"""
    from regime_system.core.regime_classifier import RollingRegimeClassifier
    from regime_system.core.indicators import calculate_all_indicators
    
    # Generate test data
    data = generate_test_data(1000)
    data_with_indicators = calculate_all_indicators(data)
    
    # Create classifier
    classifier = RollingRegimeClassifier(window_hours=24, timeframe='15min')
    
    # Classify regimes
    regimes = classifier.classify_regimes(data_with_indicators, show_progress=False)
    
    # Check results
    assert 'Direction_Regime' in regimes.columns
    assert 'TrendStrength_Regime' in regimes.columns
    assert 'Volatility_Regime' in regimes.columns
    assert 'Composite_Regime' in regimes.columns
    
    # Get statistics
    stats = classifier.get_regime_statistics(regimes)
    
    print(f"Classified {len(regimes)} periods")
    print(f"Unique composite regimes: {stats['Composite']['unique_regimes']}")
    
    # Print distribution for one dimension
    if 'Direction' in stats:
        print("\nDirection regime distribution:")
        for regime, pct in stats['Direction']['percentages'].items():
            print(f"  {regime}: {pct:.1f}%")

@test_function("Optimization")
def test_optimization():
    """Test optimization functionality"""
    from regime_system.core.regime_classifier import RollingRegimeClassifier
    from regime_system.core.indicators import calculate_all_indicators
    from regime_system.optimization.multi_objective import MultiObjectiveRegimeOptimizer
    
    # Generate test data (smaller for faster test)
    data = generate_test_data(500)
    data_with_indicators = calculate_all_indicators(data)
    
    # Create classifier
    classifier = RollingRegimeClassifier(window_hours=24, timeframe='15min')
    
    # Create optimizer
    optimizer = MultiObjectiveRegimeOptimizer(classifier, data_with_indicators)
    
    # Run short optimization
    results = optimizer.optimize_regime_thresholds(
        method='differential_evolution',
        max_iterations=5  # Very short for testing
    )
    
    print(f"Optimization complete")
    print(f"Best score: {results.best_score:.4f}")
    print(f"Sharpe ratio: {results.sharpe_ratio:.4f}")
    print(f"Max drawdown: {results.max_drawdown:.2%}")
    
    assert hasattr(results, 'best_params')
    assert hasattr(results, 'optimization_history')

@test_function("Backtesting")
def test_backtesting():
    """Test backtesting functionality"""
    from regime_system.core.regime_classifier import RollingRegimeClassifier
    from regime_system.core.indicators import calculate_all_indicators
    from regime_system.backtesting.strategies import (
        EnhancedRegimeStrategyBacktester,
        compare_strategies
    )
    
    # Generate test data
    data = generate_test_data(1000)
    data_with_indicators = calculate_all_indicators(data)
    
    # Create classifier and get regimes
    classifier = RollingRegimeClassifier(window_hours=24, timeframe='15min')
    regimes = classifier.classify_regimes(data_with_indicators, show_progress=False)
    
    # Create backtester
    backtester = EnhancedRegimeStrategyBacktester()
    
    # Test adaptive strategy
    returns = backtester.adaptive_regime_strategy_enhanced(data_with_indicators, regimes)
    metrics = backtester.calculate_performance_metrics(returns)
    
    print(f"Backtesting complete")
    print(f"Total return: {metrics['total_return']:.2%}")
    print(f"Sharpe ratio: {metrics['sharpe_ratio']:.4f}")
    print(f"Max drawdown: {metrics['max_drawdown']:.2%}")
    print(f"Win rate: {metrics['win_rate']:.2%}")
    
    # Compare strategies
    comparison = compare_strategies(data_with_indicators, regimes)
    print(f"\nStrategy comparison completed for {len(comparison)} strategies")

@test_function("Walk-Forward Validation")
def test_walk_forward():
    """Test walk-forward validation"""
    from regime_system.optimization.walk_forward import WalkForwardOptimizer
    from regime_system.core.indicators import calculate_all_indicators
    
    # Generate test data
    data = generate_test_data(1000)
    data_with_indicators = calculate_all_indicators(data)
    
    # Create walk-forward optimizer
    wf_optimizer = WalkForwardOptimizer(n_windows=3, train_ratio=0.7)
    
    # Test data splitting
    windows = wf_optimizer.split_data_windows(data_with_indicators)
    
    print(f"Created {len(windows)} walk-forward windows")
    for i, (train, test) in enumerate(windows):
        print(f"  Window {i+1}: Train={len(train)}, Test={len(test)}")
    
    assert len(windows) == 3

@test_function("Window Optimization")
def test_window_optimization():
    """Test window size optimization"""
    from regime_system.optimization.window_optimizer import (
        optimize_window_size,
        get_recommended_window_range,
        calculate_effective_lookback
    )
    from regime_system.core.indicators import calculate_all_indicators
    
    # Generate test data
    data = generate_test_data(1000)
    data_with_indicators = calculate_all_indicators(data)
    
    # Get recommended range
    min_window, max_window = get_recommended_window_range('15min')
    print(f"Recommended window range for 15min: {min_window}-{max_window} hours")
    
    # Calculate effective lookback
    lookback = calculate_effective_lookback(24, '15min')
    print(f"24-hour window = {lookback['window_bars']} bars")
    
    # Note: Full window optimization takes time, so we skip it in quick test
    print("Window optimization functions verified")

@test_function("Checkpointing")
def test_checkpointing():
    """Test checkpoint functionality"""
    from regime_system.utils.checkpoint import (
        OptimizationCheckpoint,
        OptimizationStateManager,
        CloudCostMonitor
    )
    
    # Test checkpoint manager
    checkpoint_mgr = OptimizationCheckpoint()
    
    # Test state manager
    state_mgr = OptimizationStateManager(checkpoint_interval=10)
    
    # Update state
    state_mgr.update_state(
        iteration=1,
        best_params={'test': 0.5},
        best_score=1.0,
        history=[{'iteration': 1, 'score': 1.0}]
    )
    
    # Test cost monitor
    cost_monitor = CloudCostMonitor(max_cost_usd=50)
    should_stop, reason = cost_monitor.should_stop(1, 1.0)
    
    print("Checkpoint system verified")
    print(f"Cost monitoring active (limit: ${cost_monitor.max_cost_usd})")

@test_function("Validation Tools")
def test_validation():
    """Test validation functionality"""
    from regime_system.core.regime_classifier import RollingRegimeClassifier
    from regime_system.core.indicators import calculate_all_indicators
    from regime_system.validation.indicator_analysis import run_indicator_analysis
    from regime_system.validation.regime_distribution import validate_regime_distributions
    
    # Generate test data
    data = generate_test_data(1000)
    data_with_indicators = calculate_all_indicators(data)
    
    # Get regimes
    classifier = RollingRegimeClassifier(window_hours=24, timeframe='15min')
    regimes = classifier.classify_regimes(data_with_indicators, show_progress=False)
    
    # Run indicator analysis (without saving)
    indicator_results = run_indicator_analysis(
        data_with_indicators, regimes, save_report=False
    )
    
    print(f"Found {indicator_results['correlation_analysis']['redundant_count']} "
          f"redundant indicator pairs")
    
    # Run regime validation (without saving)
    regime_results = validate_regime_distributions(
        regimes, data_with_indicators, save_report=False
    )
    
    print(f"Regime validation: {'PASSED' if regime_results['validation_passed'] else 'FAILED'}")
    print(f"Issues found: {len(regime_results['issues'])}")

@test_function("Logging System")
def test_logging():
    """Test logging functionality"""
    from regime_system.utils.logger import (
        get_logger,
        PerformanceLogger,
        TradingLogger
    )
    
    # Test basic logger
    logger = get_logger('test_module')
    logger.info("Test log message")
    
    # Test performance logger
    perf_logger = PerformanceLogger('test_performance')
    perf_logger.log_metric('test_metric', 1.5, {'context': 'test'})
    
    # Test trading logger
    trade_logger = TradingLogger('test_trading')
    trade_logger.log_trade('BUY', 'NQ', 1, 15000, 'Test trade')
    
    print("Logging system verified")

@test_function("Main Entry Points")
def test_main_entry_points():
    """Test main entry point functions"""
    from regime_system import (
        run_regime_analysis,
        run_optimization,
        run_backtesting
    )
    
    print("Main entry points imported successfully")
    print("Available functions:")
    print("  - run_regime_analysis()")
    print("  - run_optimization()")
    print("  - run_backtesting()")

# =============================================================================
# INTEGRATION TESTS
# =============================================================================

@test_function("End-to-End Workflow")
def test_end_to_end_workflow():
    """Test complete workflow"""
    from regime_system.core.data_loader import prepare_data_for_analysis
    from regime_system.core.indicators import calculate_all_indicators
    from regime_system.core.regime_classifier import RollingRegimeClassifier
    from regime_system.optimization.multi_objective import MultiObjectiveRegimeOptimizer
    from regime_system.backtesting.strategies import EnhancedRegimeStrategyBacktester
    
    # 1. Generate data
    data = generate_test_data(500)
    
    # 2. Prepare data
    data = prepare_data_for_analysis(data)
    
    # 3. Calculate indicators
    data_with_indicators = calculate_all_indicators(data, verbose=False)
    
    # 4. Classify regimes
    classifier = RollingRegimeClassifier(window_hours=24, timeframe='15min')
    regimes = classifier.classify_regimes(data_with_indicators, show_progress=False)
    
    # 5. Optimize (very short)
    optimizer = MultiObjectiveRegimeOptimizer(classifier, data_with_indicators)
    results = optimizer.optimize_regime_thresholds(max_iterations=2)
    
    # 6. Backtest
    backtester = EnhancedRegimeStrategyBacktester()
    returns = backtester.adaptive_regime_strategy_enhanced(data_with_indicators, regimes)
    metrics = backtester.calculate_performance_metrics(returns)
    
    print("Complete workflow executed successfully!")
    print(f"Final Sharpe Ratio: {metrics['sharpe_ratio']:.4f}")

# =============================================================================
# MAIN TEST RUNNER
# =============================================================================

def run_all_tests():
    """Run all tests"""
    print("="*80)
    print("REGIME SYSTEM TEST SUITE")
    print("="*80)
    print(f"Testing reorganized package structure")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Run tests
    test_core_imports()
    test_data_loader()
    test_indicators()
    test_regime_classification()
    test_optimization()
    test_backtesting()
    test_walk_forward()
    test_window_optimization()
    test_checkpointing()
    test_validation()
    test_logging()
    test_main_entry_points()
    test_end_to_end_workflow()
    
    # Print summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    print(f"Total tests: {test_results['passed'] + test_results['failed']}")
    print(f"Passed: {test_results['passed']} ✓")
    print(f"Failed: {test_results['failed']} ✗")
    
    if test_results['failed'] > 0:
        print("\nFAILED TESTS:")
        for error in test_results['errors']:
            print(f"\n{error['test']}:")
            print(f"  Error: {error['error']}")
            if 'traceback' in error:
                print(f"  Traceback:\n{error['traceback']}")
    
    print("\n" + "="*80)
    if test_results['failed'] == 0:
        print("ALL TESTS PASSED! ✓")
        print("The regime system is working correctly.")
    else:
        print("SOME TESTS FAILED! ✗")
        print("Please check the errors above.")
    print("="*80)
    
    return test_results['failed'] == 0

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

