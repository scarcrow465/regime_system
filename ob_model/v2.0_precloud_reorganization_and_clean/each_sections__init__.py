#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
Package initialization files for regime_system
Save each section as a separate __init__.py file in the appropriate directory
"""

# =============================================================================
# regime_system/__init__.py
# =============================================================================
"""
Regime System - Institutional Grade Trading Analysis
A sophisticated regime classification and optimization system
"""

__version__ = '1.0.0'
__author__ = 'Regime System Team'

# Import main components for easy access
from .core.regime_classifier import RollingRegimeClassifier
from .core.indicators import calculate_all_indicators
from .core.data_loader import load_csv_data
from .optimization.multi_objective import MultiObjectiveRegimeOptimizer
from .backtesting.strategies import EnhancedRegimeStrategyBacktester

# Main entry points
from .main import (
    run_regime_analysis,
    run_optimization,
    run_backtesting
)

__all__ = [
    'RollingRegimeClassifier',
    'calculate_all_indicators',
    'load_csv_data',
    'MultiObjectiveRegimeOptimizer',
    'EnhancedRegimeStrategyBacktester',
    'run_regime_analysis',
    'run_optimization',
    'run_backtesting'
]

# =============================================================================
# regime_system/core/__init__.py
# =============================================================================
"""
Core regime system components
"""

from .regime_classifier import (
    RollingRegimeClassifier,
    RegimeSmoother,
    DirectionRegime,
    TrendStrengthRegime,
    VelocityRegime,
    VolatilityRegime,
    MicrostructureRegime
)

from .indicators import (
    calculate_all_indicators,
    validate_indicators,
    get_indicator_info
)

from .data_loader import (
    load_csv_data,
    prepare_data_for_analysis,
    resample_data,
    get_data_info,
    check_data_quality
)

__all__ = [
    # Classifier
    'RollingRegimeClassifier',
    'RegimeSmoother',
    'DirectionRegime',
    'TrendStrengthRegime',
    'VelocityRegime',
    'VolatilityRegime',
    'MicrostructureRegime',
    # Indicators
    'calculate_all_indicators',
    'validate_indicators',
    'get_indicator_info',
    # Data
    'load_csv_data',
    'prepare_data_for_analysis',
    'resample_data',
    'get_data_info',
    'check_data_quality'
]

# =============================================================================
# regime_system/optimization/__init__.py
# =============================================================================
"""
Optimization components for regime system
"""

from .multi_objective import (
    MultiObjectiveRegimeOptimizer,
    OptimizationResults,
    print_optimization_results,
    compare_optimizations
)

from .walk_forward import (
    WalkForwardOptimizer,
    validate_no_forward_bias,
    calculate_stability_metrics
)

from .window_optimizer import (
    optimize_window_size,
    AdaptiveWindowOptimizer,
    get_recommended_window_range,
    calculate_effective_lookback
)

__all__ = [
    # Multi-objective
    'MultiObjectiveRegimeOptimizer',
    'OptimizationResults',
    'print_optimization_results',
    'compare_optimizations',
    # Walk-forward
    'WalkForwardOptimizer',
    'validate_no_forward_bias',
    'calculate_stability_metrics',
    # Window optimization
    'optimize_window_size',
    'AdaptiveWindowOptimizer',
    'get_recommended_window_range',
    'calculate_effective_lookback'
]

# =============================================================================
# regime_system/backtesting/__init__.py
# =============================================================================
"""
Backtesting components for regime strategies
"""

from .strategies import (
    EnhancedRegimeStrategyBacktester,
    StrategyConfig,
    compare_strategies,
    optimize_strategy_parameters
)

__all__ = [
    'EnhancedRegimeStrategyBacktester',
    'StrategyConfig',
    'compare_strategies',
    'optimize_strategy_parameters'
]

# =============================================================================
# regime_system/utils/__init__.py
# =============================================================================
"""
Utility functions for regime system
"""

from .checkpoint import (
    OptimizationCheckpoint,
    OptimizationStateManager,
    CloudCostMonitor
)

from .logger import (
    setup_logger,
    get_logger,
    PerformanceLogger,
    TradingLogger,
    log_execution_time,
    log_errors
)

__all__ = [
    # Checkpoint
    'OptimizationCheckpoint',
    'OptimizationStateManager',
    'CloudCostMonitor',
    # Logger
    'setup_logger',
    'get_logger',
    'PerformanceLogger',
    'TradingLogger',
    'log_execution_time',
    'log_errors'
]

# =============================================================================
# regime_system/validation/__init__.py
# =============================================================================
"""
Validation components for regime system
"""

from .indicator_analysis import (
    IndicatorCorrelationAnalyzer,
    IndicatorImportanceAnalyzer,
    IndicatorValidator,
    run_indicator_analysis
)

from .regime_distribution import (
    RegimeDistributionAnalyzer,
    RegimeTransitionAnalyzer,
    RegimeTemporalAnalyzer,
    validate_regime_distributions
)

from .performance_attribution import (
    PerformanceAttributionAnalyzer,
    FactorAttributionAnalyzer,
    run_performance_attribution
)

__all__ = [
    # Indicator analysis
    'IndicatorCorrelationAnalyzer',
    'IndicatorImportanceAnalyzer',
    'IndicatorValidator',
    'run_indicator_analysis',
    # Regime distribution
    'RegimeDistributionAnalyzer',
    'RegimeTransitionAnalyzer',
    'RegimeTemporalAnalyzer',
    'validate_regime_distributions',
    # Performance attribution
    'PerformanceAttributionAnalyzer',
    'FactorAttributionAnalyzer',
    'run_performance_attribution'
]

# =============================================================================
# regime_system/config/__init__.py
# =============================================================================
"""
Configuration settings for regime system
"""

from .settings import *

# Export all settings
__all__ = [
    # Paths
    'BASE_DIR',
    'DATA_DIR',
    'RESULTS_DIR',
    'LOG_DIR',
    # Regime parameters
    'DEFAULT_WINDOW_HOURS',
    'MIN_WINDOW_DAYS',
    'MAX_WINDOW_DAYS',
    'REGIME_SMOOTHING_PERIODS',
    'REGIME_DIMENSIONS',
    # Weights and thresholds
    'INDICATOR_WEIGHTS',
    'DEFAULT_DIMENSION_THRESHOLDS',
    'OBJECTIVE_WEIGHTS',
    'PARAMETER_BOUNDS',
    # Optimization
    'OPTIMIZATION_ITERATIONS',
    'OPTIMIZATION_METHOD',
    'WALK_FORWARD_WINDOWS',
    'WALK_FORWARD_TRAIN_RATIO',
    # Trading
    'COMMISSION_RATE',
    'SLIPPAGE_RATE',
    'DEFAULT_STOP_LOSS_ATR',
    'DEFAULT_TAKE_PROFIT_ATR',
    'MAX_POSITION_SIZE',
    'MIN_POSITION_SIZE',
    # Data
    'TIMEFRAMES',
    'DEFAULT_SYMBOLS',
    # Cloud
    'AWS_REGION',
    'S3_BUCKET',
    'EC2_INSTANCE_TYPE',
    'MAX_CLOUD_COST_USD',
    'COST_CHECK_INTERVAL',
    # Logging
    'LOG_LEVEL',
    'LOG_FORMAT',
    'LOG_FILE',
    # Reporting
    'SAVE_FORMATS',
    'REPORT_DECIMAL_PLACES',
    'PLOT_STYLE',
    'FIGURE_DPI',
    'FIGURE_SIZE'
]

# =============================================================================
# regime_system/cloud/__init__.py
# =============================================================================
"""
Cloud deployment components (placeholder for future development)
"""

# This module is reserved for future cloud deployment functionality
# Will include AWS deployment, monitoring, and scaling features

__all__ = []

# =============================================================================
# regime_system/future/__init__.py
# =============================================================================
"""
Future enhancements module (placeholder)
"""

# This module is reserved for future enhancements including:
# - Multi-timeframe integration
# - Market structure regimes
# - Global macro context
# - Sentiment analysis
# - Machine learning enhancements

__all__ = []


# In[ ]:




