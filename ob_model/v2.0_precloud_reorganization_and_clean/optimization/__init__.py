#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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

