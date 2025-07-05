#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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

