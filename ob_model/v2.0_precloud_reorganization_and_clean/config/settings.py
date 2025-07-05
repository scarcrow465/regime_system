#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
Configuration settings for the Regime System
All constants and configuration parameters in one place
"""

import os
from datetime import datetime
from typing import Dict, List, Any

# =============================================================================
# SYSTEM CONFIGURATION
# =============================================================================

# Base paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "..", "data")
RESULTS_DIR = os.path.join(BASE_DIR, "..", "multidimensional_regime_results")
LOG_DIR = os.path.join(BASE_DIR, "..", "logs")

# Create directories if they don't exist
for directory in [DATA_DIR, RESULTS_DIR, LOG_DIR]:
    os.makedirs(directory, exist_ok=True)

# =============================================================================
# REGIME CLASSIFICATION PARAMETERS
# =============================================================================

# Window parameters for rolling calculations
DEFAULT_WINDOW_HOURS = 36  # Optimal for 15-min NQ data
MIN_WINDOW_DAYS = 2  # Minimum 2 days of data
MAX_WINDOW_DAYS = 10  # Maximum 10 days for rolling windows

# Regime smoothing
REGIME_SMOOTHING_PERIODS = 3  # Confirmations needed for regime change

# Classification dimensions
REGIME_DIMENSIONS = [
    'Direction',
    'TrendStrength', 
    'Velocity',
    'Volatility',
    'Microstructure'
]

# =============================================================================
# INDICATOR WEIGHTS
# =============================================================================

INDICATOR_WEIGHTS = {
    'direction': {
        'SMA': 1.0,
        'EMA': 1.0,
        'MACD': 1.2,
        'ADX': 0.8,
        'Aroon': 0.9,
        'CCI': 0.7,
        'Ichimoku': 1.1,
        'PSAR': 0.8,
        'Vortex': 0.9,
        'SuperTrend': 1.2,
        'DPO': 0.7,
        'KST': 0.8
    },
    'trend_strength': {
        'ADX': 1.2,
        'Aroon': 0.9,
        'CCI': 0.8,
        'MACD_histogram': 1.0,
        'RSI': 0.7,
        'TSI': 0.8,
        'LinearReg_Slope': 1.1,
        'Correlation': 0.9
    },
    'velocity': {
        'ROC': 1.0,
        'RSI': 0.8,
        'TSI': 0.9,
        'MACD_histogram': 1.0,
        'Acceleration': 1.2,
        'Jerk': 0.7
    },
    'volatility': {
        'ATR': 1.0,
        'BB_width': 1.1,
        'KC_width': 0.9,
        'DC_width': 0.8,
        'NATR': 1.0,
        'UI': 0.7,
        'Historical_Vol': 1.2,
        'Parkinson': 0.9,
        'GarmanKlass': 0.9,
        'RogersSatchell': 0.8,
        'YangZhang': 1.0
    },
    'microstructure': {
        'Volume': 1.0,
        'OBV': 0.9,
        'CMF': 1.0,
        'MFI': 1.1,
        'ADI': 0.8,
        'EOM': 0.7,
        'FI': 0.8,
        'VPT': 0.8,
        'VWAP': 1.2,
        'CVD': 1.1,
        'Delta': 1.0
    }
}

# =============================================================================
# DIMENSION THRESHOLDS
# =============================================================================

DEFAULT_DIMENSION_THRESHOLDS = {
    'direction': {
        'strong_trend_threshold': 0.65,
        'weak_trend_threshold': 0.35
    },
    'trend_strength': {
        'strong_alignment': 0.70,
        'moderate_alignment': 0.40
    },
    'velocity': {
        'acceleration_threshold': 0.65,
        'stable_range': 0.35
    },
    'volatility': {
        'high_vol_percentile': 75,
        'low_vol_percentile': 25
    },
    'microstructure': {
        'institutional_volume_threshold': 1.5,
        'retail_volume_threshold': 0.7
    }
}

# =============================================================================
# OPTIMIZATION PARAMETERS
# =============================================================================

# Optimization settings
OPTIMIZATION_ITERATIONS = 50
OPTIMIZATION_METHOD = 'differential_evolution'
WALK_FORWARD_WINDOWS = 6
WALK_FORWARD_TRAIN_RATIO = 0.67

# Multi-objective weights
OBJECTIVE_WEIGHTS = {
    'sharpe_ratio': 0.4,
    'max_drawdown': 0.3,
    'regime_persistence': 0.3
}

# Parameter bounds for optimization
PARAMETER_BOUNDS = {
    'direction_strong_trend': (0.5, 0.8),
    'direction_weak_trend': (0.2, 0.5),
    'trend_strong_alignment': (0.6, 0.85),
    'trend_moderate_alignment': (0.3, 0.6),
    'velocity_acceleration': (0.5, 0.8),
    'velocity_stable_range': (0.2, 0.5),
    'volatility_high_percentile': (65, 85),
    'volatility_low_percentile': (15, 35),
    'microstructure_institutional': (1.2, 2.0),
    'microstructure_retail': (0.5, 0.8)
}

# =============================================================================
# BACKTESTING PARAMETERS
# =============================================================================

# Transaction costs
COMMISSION_RATE = 0.00005  # 0.5 bps per side
SLIPPAGE_RATE = 0.00010   # 1 bp slippage

# Risk management
DEFAULT_STOP_LOSS_ATR = 2.0
DEFAULT_TAKE_PROFIT_ATR = 3.0
MAX_POSITION_SIZE = 1.0
MIN_POSITION_SIZE = 0.1

# =============================================================================
# DATA PARAMETERS
# =============================================================================

# Supported timeframes
TIMEFRAMES = {
    '5min': 78,    # bars per day
    '15min': 26,   # bars per day
    '30min': 13,   # bars per day
    '1H': 6.5,     # bars per day
    '4H': 1.625,   # bars per day
    'Daily': 1     # bars per day
}

# Default symbols
DEFAULT_SYMBOLS = ['NQ'] # 'ES', 'YM', 'RTY', 'GC', 'CL', 'ZB', 'ZW', 'ZS']

# =============================================================================
# CLOUD DEPLOYMENT
# =============================================================================

# AWS settings
AWS_REGION = 'us-east-1'
S3_BUCKET = 'regime-system-data'
EC2_INSTANCE_TYPE = 't3.xlarge'

# Cost monitoring
MAX_CLOUD_COST_USD = 50  # Maximum cost per optimization run
COST_CHECK_INTERVAL = 300  # Check cost every 5 minutes

# =============================================================================
# LOGGING
# =============================================================================

# Logging configuration
LOG_LEVEL = 'INFO'
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_FILE = os.path.join(LOG_DIR, f'regime_system_{datetime.now().strftime("%Y%m%d")}.log')

# =============================================================================
# REPORTING
# =============================================================================

# Output formats
SAVE_FORMATS = ['csv', 'json', 'pickle']
REPORT_DECIMAL_PLACES = 4

# Visualization
PLOT_STYLE = 'seaborn'
FIGURE_DPI = 300
FIGURE_SIZE = (12, 8)

