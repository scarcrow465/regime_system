#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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

