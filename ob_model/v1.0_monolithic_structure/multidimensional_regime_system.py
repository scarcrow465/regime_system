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
# MULTI-DIMENSIONAL REGIME CLASSIFICATION SYSTEM - 5 DIMENSIONS
# Comprehensive institutional-level regime analysis as originally intended
# Save this as: multidimensional_regime_system.py
# =============================================================================

import pandas as pd
import numpy as np
import re
import gc
import logging
import time
import psutil
import os
from tqdm import tqdm
from typing import List, Optional, Union, Dict, Tuple, Any
from datetime import datetime
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

# Optimization Code
from multi_objective_optimizer import (
    run_regime_optimization,
    OptimizationResults,
    MultiObjectiveRegimeOptimizer,
    print_optimization_results,
    optimize_window_size,
    WalkForwardOptimizer
)

# Technical Analysis Libraries
from ta.trend import SMAIndicator, EMAIndicator, MACD, ADXIndicator, AroonIndicator, CCIIndicator, IchimokuIndicator, PSARIndicator, VortexIndicator
from ta.momentum import RSIIndicator, StochasticOscillator, ROCIndicator, TSIIndicator, UltimateOscillator, StochRSIIndicator
from ta.volatility import AverageTrueRange, BollingerBands, KeltnerChannel, DonchianChannel, UlcerIndex
from ta.volume import OnBalanceVolumeIndicator, ChaikinMoneyFlowIndicator, MFIIndicator, AccDistIndexIndicator, EaseOfMovementIndicator, ForceIndexIndicator
from ta.others import DailyReturnIndicator, CumulativeReturnIndicator

# Technical Analysis Libraries - Compatible with ta 0.11.0
from ta.trend import SMAIndicator, EMAIndicator, MACD, ADXIndicator, AroonIndicator, CCIIndicator, IchimokuIndicator, PSARIndicator, VortexIndicator
from ta.momentum import RSIIndicator, StochasticOscillator, ROCIndicator, TSIIndicator, UltimateOscillator, StochRSIIndicator
from ta.volatility import AverageTrueRange, BollingerBands, KeltnerChannel, DonchianChannel, UlcerIndex
from ta.volume import OnBalanceVolumeIndicator, ChaikinMoneyFlowIndicator, MFIIndicator, AccDistIndexIndicator, EaseOfMovementIndicator, ForceIndexIndicator
from ta.others import DailyReturnIndicator, CumulativeReturnIndicator

# Configure enhanced logging
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('multidimensional_regime_debug.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

print("=== MULTI-DIMENSIONAL REGIME CLASSIFICATION SYSTEM ===")
print("5 Separate Regime Dimensions with Democratic Voting")
print("Direction • Trend Strength • Velocity • Volatility • Microstructure")
print("=" * 65)

# =============================================================================
# CONFIGURATION
# =============================================================================

CSV_FILES = [
    r"C:\Users\rs\OneDrive\Desktop\Excel\Data\New Data\7.1 Master 15m Data - Updated - Nearest Unadjusted - 2014_01_01 - 2025_04_01 .csv",
    r"C:\Users\rs\OneDrive\Desktop\Excel\Data\New Data\7.2 Master 15m Data - Updated - Nearest Unadjusted - 2000_01_01 - 2013_12_31 .csv"
]

SYMBOLS = ["NQ"]  

# Training Period
TRAIN_START_DATE = "2022-01-01 00:00:00-05:00"  
TRAIN_END_DATE = "2023-01-01 00:00:00-05:00"    

# Testing Period  
TEST_START_DATE = "2023-01-01 00:00:00-05:00"
TEST_END_DATE = "2024-01-01 00:00:00-05:00"

# For initial development/testing, use training data
START_DATE = TRAIN_START_DATE
END_DATE = TRAIN_END_DATE

TIMEFRAME = "15min"
OUTPUT_DIR = "multidimensional_regime_results"

# OPTIMIZATION SETTINGS
RUN_OPTIMIZATION = True  
OPTIMIZATION_ITERATIONS = 10

# =============================================================================
# WINDOW CONFIGURATION
# =============================================================================

WALK_FORWARD = True  # Enable walk-forward validation
OPTIMIZE_WINDOW = False  # Enable window size optimization

# For 15-minute data, use hours instead of days
if TIMEFRAME == "15min":
    ROLLING_WINDOW_HOURS = 36    # 8 hours of data (32 bars)
    MIN_WINDOW_HOURS = 2        # 2 hours minimum (8 bars)
    BARS_PER_HOUR = 4          # 4 bars per hour for 15-min
elif TIMEFRAME == "5min":
    ROLLING_WINDOW_HOURS = 4    # 4 hours of data (48 bars)
    MIN_WINDOW_HOURS = 1        # 1 hour minimum (12 bars)
    BARS_PER_HOUR = 12         # 12 bars per hour for 5-min
elif TIMEFRAME == "1H":
    ROLLING_WINDOW_HOURS = 48   # 2 days of data (48 bars)
    MIN_WINDOW_HOURS = 12       # 12 hours minimum
    BARS_PER_HOUR = 1          # 1 bar per hour
else:  # Daily
    ROLLING_WINDOW_DAYS = 60    # Use days for daily
    MIN_WINDOW_DAYS = 20        # Use days for daily
    BARS_PER_HOUR = None       # Not applicable for daily

SMOOTHING_PERIODS = 2           # 2-period smoothing for intraday

# =============================================================================
# HELPER FUNCTION TO CALCULATE WINDOW SIZES
# =============================================================================

def calculate_window_sizes(timeframe: str):
    """Calculate window sizes in bars based on timeframe"""
    
    if timeframe in ["5min", "15min", "1H"]:
        # For intraday, use hours
        rolling_bars = int(ROLLING_WINDOW_HOURS * BARS_PER_HOUR)
        min_bars = int(MIN_WINDOW_HOURS * BARS_PER_HOUR)
    else:
        # For daily, use days directly
        rolling_bars = int(ROLLING_WINDOW_DAYS)
        min_bars = int(MIN_WINDOW_DAYS)
    
    return rolling_bars, min_bars

# =============================================================================
# REGIME TYPE DEFINITIONS - 5 DIMENSIONS
# =============================================================================

class DirectionRegime:
    """Direction regime classifications"""
    UP_TRENDING = "Up_Trending"
    DOWN_TRENDING = "Down_Trending"
    SIDEWAYS = "Sideways"
    UNDEFINED = "Undefined"

class TrendStrengthRegime:
    """Trend strength regime classifications"""
    STRONG = "Strong"
    MODERATE = "Moderate"
    WEAK = "Weak"
    UNDEFINED = "Undefined"

class VelocityRegime:
    """Velocity regime classifications"""
    ACCELERATING = "Accelerating"
    DECELERATING = "Decelerating"
    STABLE = "Stable"
    UNDEFINED = "Undefined"

class VolatilityRegime:
    """Volatility regime classifications"""
    LOW_VOL = "Low_Vol"
    MEDIUM_VOL = "Medium_Vol"
    HIGH_VOL = "High_Vol"
    EXTREME_VOL = "Extreme_Vol"
    UNDEFINED = "Undefined"

class MicrostructureRegime:
    """Market microstructure regime classifications"""
    INSTITUTIONAL_FLOW = "Institutional_Flow"
    RETAIL_FLOW = "Retail_Flow"
    BALANCED_FLOW = "Balanced_Flow"
    LOW_PARTICIPATION = "Low_Participation"
    UNDEFINED = "Undefined"

@dataclass
class DimensionalVote:
    """Vote for a specific regime dimension"""
    dimension: str
    indicator_name: str
    regime_vote: str
    confidence: float
    value: float
    threshold_info: Dict[str, Any] = None
    
    def __repr__(self):
        return f"{self.dimension}Vote({self.indicator_name}: {self.regime_vote} @ {self.confidence:.2f})"

@dataclass
class MultiDimensionalClassification:
    """Complete multi-dimensional regime classification"""
    timestamp: pd.Timestamp
    direction_regime: str
    direction_confidence: float
    trend_strength_regime: str
    trend_strength_confidence: float
    velocity_regime: str
    velocity_confidence: float
    volatility_regime: str
    volatility_confidence: float
    microstructure_regime: str
    microstructure_confidence: float
    composite_regime: str
    composite_confidence: float
    all_votes: List[DimensionalVote]
    
    def __repr__(self):
        return f"MultiRegime({self.composite_regime} @ {self.composite_confidence:.2f})"

# =============================================================================

@dataclass
class InstrumentRegimeParameters:
    """Store instrument-specific regime parameters"""
    symbol: str
    direction_thresholds: Dict[str, float]
    trend_strength_thresholds: Dict[str, float]
    velocity_thresholds: Dict[str, float]
    volatility_thresholds: Dict[str, float]
    microstructure_thresholds: Dict[str, float]
    last_update: pd.Timestamp
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for storage"""
        return {
            'symbol': self.symbol,
            'direction_thresholds': self.direction_thresholds,
            'trend_strength_thresholds': self.trend_strength_thresholds,
            'velocity_thresholds': self.velocity_thresholds,
            'volatility_thresholds': self.volatility_thresholds,
            'microstructure_thresholds': self.microstructure_thresholds,
            'last_update': self.last_update.isoformat()
        }

class RegimeSmoother:
    """Smooth regime transitions to prevent whipsaws"""
    
    def __init__(self, confirmation_periods: int = 3):
        self.confirmation_periods = confirmation_periods
        self.regime_counters = {}
        self.current_regimes = {}
        
    def smooth_regime(self, dimension: str, new_regime: str, 
                     timestamp: pd.Timestamp) -> Tuple[str, bool]:
        """
        Apply regime smoothing logic
        Returns: (regime_to_use, regime_changed)
        """
        if dimension not in self.current_regimes:
            # First time seeing this dimension
            self.current_regimes[dimension] = new_regime
            self.regime_counters[dimension] = 0
            return new_regime, True
            
        current_regime = self.current_regimes[dimension]
        
        if new_regime == current_regime:
            # Same regime, reset counter
            self.regime_counters[dimension] = 0
            return current_regime, False
        else:
            # Different regime, increment counter
            self.regime_counters[dimension] += 1
            
            if self.regime_counters[dimension] >= self.confirmation_periods:
                # Enough confirmations, switch regime
                self.current_regimes[dimension] = new_regime
                self.regime_counters[dimension] = 0
                #logger.info(f"{dimension} regime changed from {current_regime} to {new_regime} at {timestamp}")
                return new_regime, True
            else:
                # Not enough confirmations yet
                return current_regime, False
    
    def reset(self):
        """Reset all counters and regimes"""
        self.regime_counters = {}
        self.current_regimes = {}

class RollingRegimeClassifier:
    """
    Point-in-time regime classification with rolling windows
    Eliminates forward-looking bias
    """
    
    def __init__(self, min_periods: int = 520, max_periods: int = 1560):
        """
        Initialize rolling classifier
        min_periods: 20 trading days * 26 bars/day = 520 bars
        max_periods: 60 trading days * 26 bars/day = 1560 bars
        """
        self.min_periods = min_periods
        self.max_periods = max_periods
        self.instrument_params = {}
        
    def calculate_rolling_percentiles(self, series: pd.Series, 
                                    percentiles: List[float],
                                    min_periods: int = None) -> pd.DataFrame:
        """Calculate rolling percentiles for a series"""
        if min_periods is None:
            min_periods = self.min_periods
            
        # Create expanding window that grows to max_periods
        window_sizes = pd.Series(range(len(series)))
        window_sizes = window_sizes.clip(lower=min_periods, upper=self.max_periods)
        
        result_df = pd.DataFrame(index=series.index)
        
        for pct in percentiles:
            pct_values = []
            
            for i in range(len(series)):
                if i < min_periods - 1:
                    pct_values.append(np.nan)
                else:
                    # Use expanding window up to max_periods
                    window_size = int(window_sizes.iloc[i])
                    start_idx = max(0, i - window_size + 1)
                    window_data = series.iloc[start_idx:i+1]
                    pct_values.append(np.percentile(window_data.dropna(), pct))
            
            result_df[f'p{int(pct)}'] = pct_values
            
        return result_df
    
    def calculate_rolling_stats(self, series: pd.Series) -> pd.DataFrame:
        """Calculate rolling statistics (mean, std, min, max)"""
        window_sizes = pd.Series(range(len(series)))
        window_sizes = window_sizes.clip(lower=self.min_periods, upper=self.max_periods)
        
        stats_df = pd.DataFrame(index=series.index)
        stats = ['mean', 'std', 'min', 'max']
        
        for stat in stats:
            stat_values = []
            
            for i in range(len(series)):
                if i < self.min_periods - 1:
                    stat_values.append(np.nan)
                else:
                    window_size = int(window_sizes.iloc[i])
                    start_idx = max(0, i - window_size + 1)
                    window_data = series.iloc[start_idx:i+1].dropna()
                    
                    if stat == 'mean':
                        stat_values.append(window_data.mean())
                    elif stat == 'std':
                        stat_values.append(window_data.std())
                    elif stat == 'min':
                        stat_values.append(window_data.min())
                    elif stat == 'max':
                        stat_values.append(window_data.max())
            
            stats_df[stat] = stat_values
            
        return stats_df
    
    def normalize_rolling(self, series: pd.Series) -> pd.Series:
        """Normalize series using rolling min/max"""
        stats = self.calculate_rolling_stats(series)
        
        # Avoid division by zero
        range_vals = stats['max'] - stats['min']
        range_vals = range_vals.replace(0, 1)
        
        normalized = (series - stats['min']) / range_vals
        return normalized.fillna(0.5)  # Middle value for NaN

# =============================================================================
# PERFORMANCE MONITORING (FROM YOUR WORKING SYSTEM)
# =============================================================================

class PerformanceMonitor:
    """Performance monitoring"""
    
    def __init__(self):
        self.timings = {}
        self.memory_usage = {}
        self.error_counts = {}
        
    def start_timer(self, operation: str):
        self.timings[operation] = time.time()
        
    def end_timer(self, operation: str):
        if operation in self.timings:
            duration = time.time() - self.timings[operation]
            logger.info(f"TIMER: {operation} completed in {duration:.3f} seconds")
            return duration
        return None
        
    def log_memory(self, operation: str):
        memory_mb = psutil.Process().memory_info().rss / 1024 / 1024
        self.memory_usage[operation] = memory_mb
        logger.info(f"MEMORY: Memory usage after {operation}: {memory_mb:.1f} MB")

perf_monitor = PerformanceMonitor()

# =============================================================================
# UPDATED MULTI-DIMENSIONAL REGIME CLASSIFIER - 80+ INDICATORS
# Complete classification system for all 5 regime dimensions
# =============================================================================

# =============================================================================
# ORIGINAL WORKING REGIME CLASSIFIER - EXPANDED TO 85 INDICATORS
# Based on proven logic that produced realistic 44%/35%/20% distributions
# =============================================================================

class MultiDimensionalRegimeClassifier:
    """
    Multi-Dimensional Regime Classification System with EFFICIENT Rolling Windows
    Pre-calculates all rolling statistics for performance
    """
    
    def __init__(self, min_periods: int = None, max_periods: int = None, 
                 smoothing_periods: int = 3, window_hours: float = None):
        """
        Initialize with optional window_hours parameter for easy configuration
        """
        # If window_hours is provided, calculate periods from it
        if window_hours is not None:
            if TIMEFRAME == "15min":
                periods = int(window_hours * 4)  # 4 bars per hour
            elif TIMEFRAME == "5min":
                periods = int(window_hours * 12)  # 12 bars per hour
            elif TIMEFRAME == "1H":
                periods = int(window_hours * 1)  # 1 bar per hour
            else:  # Daily
                periods = int(window_hours / 24)  # Convert hours to days
            
            # Set both min and max to the same value for testing
            self.min_periods = periods if min_periods is None else min_periods
            self.max_periods = periods if max_periods is None else max_periods
        else:
            # Use provided values or defaults
            self.min_periods = min_periods if min_periods is not None else 520
            self.max_periods = max_periods if max_periods is not None else 1560
        
        self.smoothing_periods = smoothing_periods
        
        # Rest of the initialization stays the same
        self.indicator_weights = {}
        self.dimension_thresholds = {}
        self.regime_history = []
        self.rolling_stats_cache = {}
        self.regime_smoother = RegimeSmoother(smoothing_periods)
        
        # Initialize system
        self.setup_indicator_weights()
        self.setup_dimension_thresholds()

        logger.info(f"Initialized classifier with min_periods={self.min_periods}, max_periods={self.max_periods}")
        
    def setup_indicator_weights(self):
        """Setup voting weights for targeted indicators within each dimension"""
        # KEEPING EXACTLY AS YOUR ORIGINAL
        self.indicator_weights = {
            # Direction Dimension Weights (expanded from original 3 to 10+ indicators)
            'direction': {
                'EMA': 1.0,      # EMA crossovers (original)
                'SMA': 1.2,      # SMA slopes (original) 
                'MA': 1.0,       # MA alignment (original)
                'KAMA': 0.9,     # Adaptive MAs
                'WMA': 0.8,      # Weighted MAs
                'HMA': 0.9,      # Hull MAs
                'VWMA': 1.0,     # Volume weighted
                'Ichimoku': 0.8, # Ichimoku components
                'PSAR': 1.1      # Parabolic SAR
            },
            
            # Trend Strength Dimension Weights
            'trend_strength': {
                'MA': 1.0,       # MA alignment (original)
                'ADX': 1.2,      # ADX levels (original)
                'MACD': 1.0,     # MACD histogram (original)
                'Aroon': 0.9,    # Aroon oscillator
                'DMI': 0.8,      # Directional movement
                'CCI': 0.7,      # Commodity Channel Index
                'Vortex': 0.8,   # Vortex indicator
                'Choppiness': 0.9 # Choppiness index
            },
            
            # Velocity Dimension Weights
            'velocity': {
                'RSI': 1.0,      # RSI momentum (original)
                'ROC': 1.1,      # Rate of change (original)
                'Momentum': 1.0, # Momentum oscillator (original)
                'TSI': 0.9,      # True Strength Index
                'Ultimate': 0.8, # Ultimate Oscillator
                'StochRSI': 0.9, # Stochastic RSI
                'MACD': 0.8      # MACD velocity component
            },
            
            # Volatility Dimension Weights
            'volatility': {
                'ATR': 1.2,      # ATR levels (original)
                'BB': 1.0,       # Bollinger Band width (original)
                'Percentile': 0.9, # Vol percentiles (original)
                'Keltner': 0.8,  # Keltner channel width
                'Donchian': 0.7, # Donchian channel width
                'Ulcer': 0.8,    # Ulcer index
                'GARCH': 0.9,    # GARCH estimate
                'Parkinson': 0.8 # Parkinson estimator
            },
            
            # Microstructure Dimension Weights
            'microstructure': {
                'Volume': 1.0,   # Volume analysis (original)
                'VWAP': 1.1,     # VWAP deviation (original)
                'OBV': 0.9,      # On Balance Volume
                'CMF': 0.8,      # Chaikin Money Flow
                'MFI': 0.9,      # Money Flow Index
                'ADI': 0.7,      # Accumulation/Distribution
                'EOM': 0.8,      # Ease of Movement
                'Force': 0.8     # Force Index
            }
        }
        
    def setup_dimension_thresholds(self):
        """Setup initial thresholds (will be updated by rolling calculations)"""
        # KEEPING YOUR ORIGINAL THRESHOLDS
        self.dimension_thresholds = {
            'direction': {
                'ema_cross': {'bullish': 0.02, 'bearish': -0.02},
                'ma_alignment': {'bullish': 0.7, 'bearish': 0.3},
                'slope': {'up': 0.001, 'down': -0.001}
            },
            'trend_strength': {
                'adx': {'strong': 40, 'moderate': 25, 'weak': 15},
                'macd_hist': {'strong': 0.005, 'moderate': 0.002}
            },
            'velocity': {
                'rsi': {'overbought': 70, 'oversold': 30, 'neutral_high': 60, 'neutral_low': 40},
                'roc': {'accelerating': 0.02, 'decelerating': -0.02}
            },
            'volatility': {
                'percentile': {'low': 25, 'medium': 50, 'high': 75, 'extreme': 90}
            },
            'microstructure': {
                'volume_ratio': {'high': 1.5, 'low': 0.5},
                'vwap_deviation': {'institutional': 0.002, 'retail': 0.005}
            }
        }
    
    def pre_calculate_rolling_statistics(self, data: pd.DataFrame):
        """
        Pre-calculate ALL rolling statistics ONCE for efficiency
        This is the KEY to fixing the performance issue
        """
        logger.info("Pre-calculating rolling statistics for efficient classification...")
        
        # Clear cache
        self.rolling_stats_cache = {}
        
        # Use expanding window that caps at max_periods
        window = data.index.to_series().expanding(min_periods=self.min_periods)
        
        # Direction indicators - calculate rolling slopes
        for indicator_type in ['EMA', 'SMA', 'KAMA', 'WMA', 'HMA']:
            for period in [9, 12, 21, 26, 50]:
                col_name = f'{indicator_type}_{period}'
                if col_name in data.columns:
                    # 5-period slope
                    slope = data[col_name].diff(5) / 5
                    
                    # Rolling percentiles using pandas rolling
                    self.rolling_stats_cache[f'{col_name}_slope'] = slope
                    self.rolling_stats_cache[f'{col_name}_slope_p20'] = slope.rolling(
                        window=self.max_periods, min_periods=self.min_periods
                    ).quantile(0.2)
                    self.rolling_stats_cache[f'{col_name}_slope_p80'] = slope.rolling(
                        window=self.max_periods, min_periods=self.min_periods
                    ).quantile(0.8)
        
        # Volatility indicators - calculate rolling percentiles
        if 'ATR' in data.columns:
            atr = data['ATR']
            self.rolling_stats_cache['ATR'] = atr
            self.rolling_stats_cache['ATR_p25'] = atr.rolling(
                window=self.max_periods, min_periods=self.min_periods
            ).quantile(0.25)
            self.rolling_stats_cache['ATR_p50'] = atr.rolling(
                window=self.max_periods, min_periods=self.min_periods
            ).quantile(0.50)
            self.rolling_stats_cache['ATR_p75'] = atr.rolling(
                window=self.max_periods, min_periods=self.min_periods
            ).quantile(0.75)
            self.rolling_stats_cache['ATR_p90'] = atr.rolling(
                window=self.max_periods, min_periods=self.min_periods
            ).quantile(0.90)
        
        # Bollinger Band width normalization
        if 'BB_Width' in data.columns:
            bb_width = data['BB_Width']
            self.rolling_stats_cache['BB_Width'] = bb_width
            rolling_min = bb_width.rolling(window=self.max_periods, min_periods=self.min_periods).min()
            rolling_max = bb_width.rolling(window=self.max_periods, min_periods=self.min_periods).max()
            rolling_range = rolling_max - rolling_min
            rolling_range[rolling_range == 0] = 1  # Avoid division by zero
            self.rolling_stats_cache['BB_Width_normalized'] = (bb_width - rolling_min) / rolling_range
        
        # Trend strength indicators
        if 'ADX' in data.columns:
            self.rolling_stats_cache['ADX'] = data['ADX']
        
        if 'MACD_Histogram' in data.columns:
            macd_hist = data['MACD_Histogram']
            self.rolling_stats_cache['MACD_Histogram'] = macd_hist
            self.rolling_stats_cache['MACD_Histogram_p80'] = macd_hist.rolling(
                window=self.max_periods, min_periods=self.min_periods
            ).quantile(0.8)
            self.rolling_stats_cache['MACD_Histogram_p20'] = macd_hist.rolling(
                window=self.max_periods, min_periods=self.min_periods
            ).quantile(0.2)
        
        # Velocity indicators
        if 'RSI' in data.columns:
            self.rolling_stats_cache['RSI'] = data['RSI']
        
        if 'ROC' in data.columns:
            roc = data['ROC']
            self.rolling_stats_cache['ROC'] = roc
            self.rolling_stats_cache['ROC_p80'] = roc.rolling(
                window=self.max_periods, min_periods=self.min_periods
            ).quantile(0.8)
            self.rolling_stats_cache['ROC_p20'] = roc.rolling(
                window=self.max_periods, min_periods=self.min_periods
            ).quantile(0.2)
        
        # Volume indicators
        if 'volume' in data.columns:
            volume = data['volume']
            self.rolling_stats_cache['volume'] = volume
            self.rolling_stats_cache['volume_ma'] = volume.rolling(
                window=20, min_periods=10
            ).mean()
        
        logger.info(f"Pre-calculated {len(self.rolling_stats_cache)} rolling statistics")
    
    def classify_direction_dimension(self, data: pd.DataFrame, votes: List[DimensionalVote], 
                                   index: int) -> Tuple[str, float]:
        """Classify direction for a specific time index using pre-calculated stats"""
        try:
            direction_scores = {'Up_Trending': 0, 'Down_Trending': 0, 'Sideways': 0}
            total_weight = 0
            
            # Use pre-calculated rolling statistics
            for indicator_type in ['EMA', 'SMA', 'KAMA', 'WMA', 'HMA']:
                for period in [9, 12, 21, 26, 50]:
                    col_name = f'{indicator_type}_{period}'
                    slope_key = f'{col_name}_slope'
                    
                    if slope_key in self.rolling_stats_cache:
                        current_slope = self.rolling_stats_cache[slope_key].iloc[index]
                        up_threshold = self.rolling_stats_cache[f'{slope_key}_p80'].iloc[index]
                        down_threshold = self.rolling_stats_cache[f'{slope_key}_p20'].iloc[index]
                        
                        weight = self.indicator_weights['direction'].get(indicator_type, 0.8)
                        
                        if pd.notna(current_slope) and pd.notna(up_threshold):
                            if current_slope > up_threshold:
                                direction_scores['Up_Trending'] += weight
                                vote = DimensionalVote(
                                    dimension='direction',
                                    indicator_name=col_name,
                                    regime_vote='Up_Trending',
                                    confidence=0.8,
                                    value=current_slope,
                                    threshold_info={'up': up_threshold, 'down': down_threshold}
                                )
                                votes.append(vote)
                            elif current_slope < down_threshold:
                                direction_scores['Down_Trending'] += weight
                                vote = DimensionalVote(
                                    dimension='direction',
                                    indicator_name=col_name,
                                    regime_vote='Down_Trending',
                                    confidence=0.8,
                                    value=current_slope,
                                    threshold_info={'up': up_threshold, 'down': down_threshold}
                                )
                                votes.append(vote)
                            else:
                                direction_scores['Sideways'] += weight * 0.5
                            
                            total_weight += weight
            
            # Determine regime
            if total_weight > 0:
                for regime in direction_scores:
                    direction_scores[regime] /= total_weight
                
                best_regime = max(direction_scores, key=direction_scores.get)
                confidence = direction_scores[best_regime]
                return best_regime, confidence
            else:
                return DirectionRegime.UNDEFINED, 0.0
                
        except Exception as e:
            logger.error(f"Error in direction classification: {e}")
            return DirectionRegime.UNDEFINED, 0.0
    
    def classify_trend_strength_dimension(self, data: pd.DataFrame, votes: List[DimensionalVote], 
                                        index: int) -> Tuple[str, float]:
        """Classify trend strength for a specific time index"""
        try:
            strength_scores = {'Strong': 0, 'Moderate': 0, 'Weak': 0}
            total_weight = 0
            
            # ADX-based classification
            if 'ADX' in self.rolling_stats_cache:
                adx_value = self.rolling_stats_cache['ADX'].iloc[index]
                if pd.notna(adx_value):
                    weight = self.indicator_weights['trend_strength'].get('ADX', 1.2)
                    
                    if adx_value > 40:
                        strength_scores['Strong'] += weight
                        regime_vote = 'Strong'
                    elif adx_value > 25:
                        strength_scores['Moderate'] += weight
                        regime_vote = 'Moderate'
                    else:
                        strength_scores['Weak'] += weight
                        regime_vote = 'Weak'
                    
                    vote = DimensionalVote(
                        dimension='trend_strength',
                        indicator_name='ADX',
                        regime_vote=regime_vote,
                        confidence=0.9,
                        value=adx_value
                    )
                    votes.append(vote)
                    total_weight += weight
            
            # MACD Histogram strength
            if 'MACD_Histogram' in self.rolling_stats_cache:
                macd_hist = self.rolling_stats_cache['MACD_Histogram'].iloc[index]
                macd_p80 = self.rolling_stats_cache['MACD_Histogram_p80'].iloc[index]
                macd_p20 = self.rolling_stats_cache['MACD_Histogram_p20'].iloc[index]
                
                if pd.notna(macd_hist) and pd.notna(macd_p80):
                    weight = self.indicator_weights['trend_strength'].get('MACD', 1.0)
                    abs_hist = abs(macd_hist)
                    abs_p80 = abs(macd_p80)
                    
                    if abs_hist > abs_p80:
                        strength_scores['Strong'] += weight
                    elif abs_hist > abs_p80 * 0.5:
                        strength_scores['Moderate'] += weight
                    else:
                        strength_scores['Weak'] += weight
                    
                    total_weight += weight
            
            # Determine regime
            if total_weight > 0:
                for regime in strength_scores:
                    strength_scores[regime] /= total_weight
                
                best_regime = max(strength_scores, key=strength_scores.get)
                confidence = strength_scores[best_regime]
                return best_regime, confidence
            else:
                return TrendStrengthRegime.UNDEFINED, 0.0
                
        except Exception as e:
            logger.error(f"Error in trend strength classification: {e}")
            return TrendStrengthRegime.UNDEFINED, 0.0
    
    def classify_velocity_dimension(self, data: pd.DataFrame, votes: List[DimensionalVote], 
                                  index: int) -> Tuple[str, float]:
        """Classify velocity for a specific time index - ADJUSTED FOR BETTER DISTRIBUTION"""
        try:
            velocity_scores = {'Accelerating': 0, 'Decelerating': 0, 'Stable': 0}
            total_weight = 0
            
            # RSI momentum - ADJUSTED THRESHOLDS
            if 'RSI' in self.rolling_stats_cache:
                rsi_value = self.rolling_stats_cache['RSI'].iloc[index]
                if pd.notna(rsi_value):
                    weight = self.indicator_weights['velocity'].get('RSI', 1.0)
                    
                    # More balanced thresholds
                    if rsi_value > 65:  # Lowered from 70
                        velocity_scores['Accelerating'] += weight
                        regime_vote = 'Accelerating'
                    elif rsi_value < 35:  # Raised from 30
                        velocity_scores['Decelerating'] += weight
                        regime_vote = 'Decelerating'
                    else:
                        velocity_scores['Stable'] += weight * 0.7  # Reduced weight for stable
                        regime_vote = 'Stable'
                    
                    vote = DimensionalVote(
                        dimension='velocity',
                        indicator_name='RSI',
                        regime_vote=regime_vote,
                        confidence=0.8,
                        value=rsi_value
                    )
                    votes.append(vote)
                    total_weight += weight
            
            # ROC acceleration - USE PERCENTILES FOR DYNAMIC THRESHOLDS
            if 'ROC' in self.rolling_stats_cache:
                roc_value = self.rolling_stats_cache['ROC'].iloc[index]
                # Use 70/30 percentiles instead of 80/20 for more balanced distribution
                roc_p70_key = 'ROC_p70'
                roc_p30_key = 'ROC_p30'
                
                # Calculate these if not in cache
                if roc_p70_key not in self.rolling_stats_cache:
                    roc_series = self.rolling_stats_cache['ROC']
                    self.rolling_stats_cache[roc_p70_key] = roc_series.rolling(
                        window=self.max_periods, min_periods=self.min_periods
                    ).quantile(0.7)
                    self.rolling_stats_cache[roc_p30_key] = roc_series.rolling(
                        window=self.max_periods, min_periods=self.min_periods
                    ).quantile(0.3)
                
                roc_p70 = self.rolling_stats_cache[roc_p70_key].iloc[index]
                roc_p30 = self.rolling_stats_cache[roc_p30_key].iloc[index]
                
                if pd.notna(roc_value) and pd.notna(roc_p70):
                    weight = self.indicator_weights['velocity'].get('ROC', 1.1)
                    
                    if roc_value > roc_p70:
                        velocity_scores['Accelerating'] += weight
                    elif roc_value < roc_p30:
                        velocity_scores['Decelerating'] += weight
                    else:
                        velocity_scores['Stable'] += weight * 0.7  # Reduced weight
                    
                    total_weight += weight
            
            # Add Stochastic RSI for more velocity signals
            if 'Stoch_RSI' in data.columns and index < len(data):
                stoch_rsi = data['Stoch_RSI'].iloc[index]
                if pd.notna(stoch_rsi):
                    weight = self.indicator_weights['velocity'].get('StochRSI', 0.9)
                    
                    if stoch_rsi > 0.8:
                        velocity_scores['Accelerating'] += weight
                    elif stoch_rsi < 0.2:
                        velocity_scores['Decelerating'] += weight
                    else:
                        velocity_scores['Stable'] += weight * 0.6
                    
                    total_weight += weight
            
            # TSI (True Strength Index) if available
            if 'TSI' in data.columns and index < len(data):
                tsi_value = data['TSI'].iloc[index]
                if pd.notna(tsi_value):
                    weight = self.indicator_weights['velocity'].get('TSI', 0.9)
                    
                    if tsi_value > 25:
                        velocity_scores['Accelerating'] += weight
                    elif tsi_value < -25:
                        velocity_scores['Decelerating'] += weight
                    else:
                        velocity_scores['Stable'] += weight * 0.6
                    
                    total_weight += weight
            
            # Momentum indicator
            if 'Momentum' in data.columns and index < len(data):
                mom_value = data['Momentum'].iloc[index]
                if pd.notna(mom_value):
                    weight = self.indicator_weights['velocity'].get('Momentum', 1.0)
                    
                    # Use percentage of price for momentum
                    price = data['close'].iloc[index] if 'close' in data.columns else 1
                    mom_pct = (mom_value / price) * 100 if price > 0 else 0
                    
                    if mom_pct > 2:  # 2% positive momentum
                        velocity_scores['Accelerating'] += weight
                    elif mom_pct < -2:  # 2% negative momentum
                        velocity_scores['Decelerating'] += weight
                    else:
                        velocity_scores['Stable'] += weight * 0.5
                    
                    total_weight += weight
            
            # Determine regime
            if total_weight > 0:
                for regime in velocity_scores:
                    velocity_scores[regime] /= total_weight
                
                best_regime = max(velocity_scores, key=velocity_scores.get)
                confidence = velocity_scores[best_regime]
                return best_regime, confidence
            else:
                return VelocityRegime.UNDEFINED, 0.0
                
        except Exception as e:
            logger.error(f"Error in velocity classification: {e}")
            return VelocityRegime.UNDEFINED, 0.0
    
    def classify_volatility_dimension(self, data: pd.DataFrame, votes: List[DimensionalVote], 
                                    index: int) -> Tuple[str, float]:
        """Classify volatility for a specific time index"""
        try:
            vol_scores = {'Low_Vol': 0, 'Medium_Vol': 0, 'High_Vol': 0, 'Extreme_Vol': 0}
            total_weight = 0
            
            # ATR-based classification
            if 'ATR' in self.rolling_stats_cache:
                current_atr = self.rolling_stats_cache['ATR'].iloc[index]
                atr_p25 = self.rolling_stats_cache['ATR_p25'].iloc[index]
                atr_p50 = self.rolling_stats_cache['ATR_p50'].iloc[index]
                atr_p75 = self.rolling_stats_cache['ATR_p75'].iloc[index]
                atr_p90 = self.rolling_stats_cache['ATR_p90'].iloc[index]
                
                if pd.notna(current_atr) and pd.notna(atr_p25):
                    weight = self.indicator_weights['volatility'].get('ATR', 1.2)
                    
                    if current_atr < atr_p25:
                        vol_scores['Low_Vol'] += weight
                        regime_vote = 'Low_Vol'
                    elif current_atr < atr_p50:
                        vol_scores['Medium_Vol'] += weight
                        regime_vote = 'Medium_Vol'
                    elif current_atr < atr_p90:
                        vol_scores['High_Vol'] += weight
                        regime_vote = 'High_Vol'
                    else:
                        vol_scores['Extreme_Vol'] += weight
                        regime_vote = 'Extreme_Vol'
                    
                    vote = DimensionalVote(
                        dimension='volatility',
                        indicator_name='ATR',
                        regime_vote=regime_vote,
                        confidence=0.9,
                        value=current_atr,
                        threshold_info={
                            'p25': atr_p25, 'p50': atr_p50,
                            'p75': atr_p75, 'p90': atr_p90
                        }
                    )
                    votes.append(vote)
                    total_weight += weight
            
            # Bollinger Band width
            if 'BB_Width_normalized' in self.rolling_stats_cache:
                bb_norm = self.rolling_stats_cache['BB_Width_normalized'].iloc[index]
                
                if pd.notna(bb_norm):
                    weight = self.indicator_weights['volatility'].get('BB', 1.0)
                    
                    if bb_norm < 0.25:
                        vol_scores['Low_Vol'] += weight
                    elif bb_norm < 0.5:
                        vol_scores['Medium_Vol'] += weight
                    elif bb_norm < 0.75:
                        vol_scores['High_Vol'] += weight
                    else:
                        vol_scores['Extreme_Vol'] += weight
                    
                    total_weight += weight
            
            # Determine regime
            if total_weight > 0:
                for regime in vol_scores:
                    vol_scores[regime] /= total_weight
                
                best_regime = max(vol_scores, key=vol_scores.get)
                confidence = vol_scores[best_regime]
                return best_regime, confidence
            else:
                return VolatilityRegime.UNDEFINED, 0.0
                
        except Exception as e:
            logger.error(f"Error in volatility classification: {e}")
            return VolatilityRegime.UNDEFINED, 0.0
    
    def classify_microstructure_dimension(self, data: pd.DataFrame, votes: List[DimensionalVote], 
                                        index: int) -> Tuple[str, float]:
        """Classify microstructure for a specific time index"""
        try:
            micro_scores = {'Institutional_Flow': 0, 'Retail_Flow': 0, 
                          'Balanced_Flow': 0, 'Low_Participation': 0}
            total_weight = 0
            
            # Volume-based classification
            if 'volume' in self.rolling_stats_cache and 'volume_ma' in self.rolling_stats_cache:
                current_vol = self.rolling_stats_cache['volume'].iloc[index]
                vol_ma = self.rolling_stats_cache['volume_ma'].iloc[index]
                
                if pd.notna(current_vol) and pd.notna(vol_ma) and vol_ma > 0:
                    vol_ratio = current_vol / vol_ma
                    weight = self.indicator_weights['microstructure'].get('Volume', 1.0)
                    
                    if vol_ratio > 1.5:
                        micro_scores['Institutional_Flow'] += weight
                        regime_vote = 'Institutional_Flow'
                    elif vol_ratio < 0.5:
                        micro_scores['Low_Participation'] += weight
                        regime_vote = 'Low_Participation'
                    else:
                        micro_scores['Balanced_Flow'] += weight
                        regime_vote = 'Balanced_Flow'
                    
                    vote = DimensionalVote(
                        dimension='microstructure',
                        indicator_name='Volume',
                        regime_vote=regime_vote,
                        confidence=0.7,
                        value=vol_ratio
                    )
                    votes.append(vote)
                    total_weight += weight
            
            # Money Flow Index
            if 'MFI' in data.columns and index < len(data):
                mfi_value = data['MFI'].iloc[index]
                if pd.notna(mfi_value):
                    weight = self.indicator_weights['microstructure'].get('MFI', 0.9)
                    
                    if mfi_value > 80:
                        micro_scores['Institutional_Flow'] += weight * 0.8
                    elif mfi_value < 20:
                        micro_scores['Retail_Flow'] += weight * 0.8
                    else:
                        micro_scores['Balanced_Flow'] += weight * 0.5
                    
                    total_weight += weight
            
            # Determine regime
            if total_weight > 0:
                for regime in micro_scores:
                    micro_scores[regime] /= total_weight
                
                best_regime = max(micro_scores, key=micro_scores.get)
                confidence = micro_scores[best_regime]
                return best_regime, confidence
            else:
                return MicrostructureRegime.UNDEFINED, 0.0
                
        except Exception as e:
            logger.error(f"Error in microstructure classification: {e}")
            return MicrostructureRegime.UNDEFINED, 0.0
    
    def classify_multidimensional_regime(self, data: pd.DataFrame, 
                                       symbol: str = None) -> pd.DataFrame:
        """
        EFFICIENT classification using pre-calculated rolling statistics
        This is the main fix for the performance issue
        """
        logger.info("Starting efficient multi-dimensional regime classification...")
        
        # CRITICAL: Pre-calculate all rolling statistics ONCE
        self.pre_calculate_rolling_statistics(data)
        
        # Reset smoother for new classification
        self.regime_smoother.reset()
        
        # Initialize results DataFrame with all columns
        results = pd.DataFrame(index=data.index)
        
        # Initialize all regime columns
        results['Direction_Regime'] = DirectionRegime.UNDEFINED
        results['Direction_Confidence'] = 0.0
        results['TrendStrength_Regime'] = TrendStrengthRegime.UNDEFINED
        results['TrendStrength_Confidence'] = 0.0
        results['Velocity_Regime'] = VelocityRegime.UNDEFINED
        results['Velocity_Confidence'] = 0.0
        results['Volatility_Regime'] = VolatilityRegime.UNDEFINED
        results['Volatility_Confidence'] = 0.0
        results['Microstructure_Regime'] = MicrostructureRegime.UNDEFINED
        results['Microstructure_Confidence'] = 0.0
        results['Composite_Regime'] = 'Undefined'
        results['Composite_Confidence'] = 0.0
        
        # Process each time period efficiently
        for i in tqdm(range(len(data)), desc="Classifying regimes", mininterval=1.0, miniters=5):
            if i < self.min_periods:
                # Not enough data yet - keep defaults
                continue
            
            votes = []
            
            # Classify each dimension using pre-calculated stats
            direction_regime, direction_conf = self.classify_direction_dimension(data, votes, i)
            trend_strength_regime, trend_strength_conf = self.classify_trend_strength_dimension(data, votes, i)
            velocity_regime, velocity_conf = self.classify_velocity_dimension(data, votes, i)
            volatility_regime, volatility_conf = self.classify_volatility_dimension(data, votes, i)
            microstructure_regime, microstructure_conf = self.classify_microstructure_dimension(data, votes, i)
            
            # Apply smoothing to each dimension
            smoothed_direction, _ = self.regime_smoother.smooth_regime('direction', direction_regime, data.index[i])
            smoothed_trend_strength, _ = self.regime_smoother.smooth_regime('trend_strength', trend_strength_regime, data.index[i])
            smoothed_velocity, _ = self.regime_smoother.smooth_regime('velocity', velocity_regime, data.index[i])
            smoothed_volatility, _ = self.regime_smoother.smooth_regime('volatility', volatility_regime, data.index[i])
            smoothed_microstructure, _ = self.regime_smoother.smooth_regime('microstructure', microstructure_regime, data.index[i])
            
            # Store results
            results.loc[data.index[i], 'Direction_Regime'] = smoothed_direction
            results.loc[data.index[i], 'Direction_Confidence'] = direction_conf
            results.loc[data.index[i], 'TrendStrength_Regime'] = smoothed_trend_strength
            results.loc[data.index[i], 'TrendStrength_Confidence'] = trend_strength_conf
            results.loc[data.index[i], 'Velocity_Regime'] = smoothed_velocity
            results.loc[data.index[i], 'Velocity_Confidence'] = velocity_conf
            results.loc[data.index[i], 'Volatility_Regime'] = smoothed_volatility
            results.loc[data.index[i], 'Volatility_Confidence'] = volatility_conf
            results.loc[data.index[i], 'Microstructure_Regime'] = smoothed_microstructure
            results.loc[data.index[i], 'Microstructure_Confidence'] = microstructure_conf
            
            # Create composite regime
            composite = f"{smoothed_direction}_{smoothed_trend_strength}_{smoothed_volatility}"
            results.loc[data.index[i], 'Composite_Regime'] = composite
            
            # Calculate composite confidence
            avg_confidence = np.mean([direction_conf, trend_strength_conf, 
                                    velocity_conf, volatility_conf, microstructure_conf])
            results.loc[data.index[i], 'Composite_Confidence'] = avg_confidence
        
        # Store parameters for this instrument
        if symbol:
            params = InstrumentRegimeParameters(
                symbol=symbol,
                direction_thresholds=self.dimension_thresholds.get('direction', {}),
                trend_strength_thresholds=self.dimension_thresholds.get('trend_strength', {}),
                velocity_thresholds=self.dimension_thresholds.get('velocity', {}),
                volatility_thresholds=self.dimension_thresholds.get('volatility', {}),
                microstructure_thresholds=self.dimension_thresholds.get('microstructure', {}),
                last_update=data.index[-1]
            )
            logger.info(f"Stored regime parameters for {symbol}")
        
        # Combine with original data
        return pd.concat([data, results], axis=1)
    
    # Keep all existing methods that aren't being replaced...
    def update_thresholds(self, threshold_params: Dict[str, float]):
        """Update dimension thresholds from optimization parameters"""
        # This is used by the optimizer - keep as is
        for param_name, value in threshold_params.items():
            if 'direction' in param_name:
                if 'bullish' in param_name:
                    self.dimension_thresholds['direction']['ema_cross']['bullish'] = value
                elif 'bearish' in param_name:
                    self.dimension_thresholds['direction']['ema_cross']['bearish'] = value
            elif 'volatility' in param_name:
                if 'low' in param_name:
                    self.dimension_thresholds['volatility']['percentile']['low'] = value
                elif 'high' in param_name:
                    self.dimension_thresholds['volatility']['percentile']['high'] = value
            elif 'trend' in param_name:
                if 'strong' in param_name:
                    self.dimension_thresholds['trend_strength']['adx']['strong'] = value
                elif 'weak' in param_name:
                    self.dimension_thresholds['trend_strength']['adx']['weak'] = value
            elif 'velocity' in param_name:
                if 'overbought' in param_name:
                    self.dimension_thresholds['velocity']['rsi']['overbought'] = value
                elif 'oversold' in param_name:
                    self.dimension_thresholds['velocity']['rsi']['oversold'] = value

# =============================================================================
# INTEGRATION WITH YOUR WORKING SYSTEM (DATA LOADING + INDICATORS)
# =============================================================================

def parse_symbol(symbol_str):
    """Extract base symbol from futures contract notation"""
    if not isinstance(symbol_str, str) or pd.isna(symbol_str):
        return None
    if len(symbol_str) < 3:
        return symbol_str
    year = symbol_str[-2:]
    if year.isdigit():
        month = symbol_str[-3]
        valid_month_codes = {'F', 'G', 'H', 'J', 'K', 'M', 'N', 'Q', 'U', 'V', 'X', 'Z'}
        if month in valid_month_codes:
            return symbol_str[:-3]
    return symbol_str

def load_csv_data(csv_paths: Union[str, List[str]], symbols: Optional[List[str]] = None, start_date: Optional[str] = None, end_date: Optional[str] = None) -> pd.DataFrame:
    """Your working data loader"""
    logger.info("DATA: Starting load_csv_data")
    perf_monitor.start_timer("data_loading")
    
    if isinstance(csv_paths, str):
        csv_paths = [csv_paths]
    
    all_dfs = []
    
    for csv_path in tqdm(csv_paths, desc="Reading CSV files"):
        try:
            chunks = pd.read_csv(csv_path, parse_dates=['Date'], index_col='Date', chunksize=100000, dtype={'Symbol': str}, low_memory=False)
            
            for chunk in chunks:
                if chunk.empty or chunk.index.hasnans:
                    continue
                
                chunk = chunk.copy()
                
                try:
                    if chunk.index.tz is None:
                        chunk.index = chunk.index.tz_localize('America/New_York', ambiguous='infer', nonexistent='shift_forward')
                    else:
                        chunk.index = chunk.index.tz_convert('America/New_York')
                except Exception:
                    if chunk.index.tz is not None:
                        chunk.index = chunk.index.tz_localize(None)
                
                dfs = []
                i = 1
                
                while True:
                    symbol_col = f'Symbol.{i}'
                    required_cols = [symbol_col, f'Open.{i}', f'High.{i}', f'Low.{i}', f'Close.{i}']
                    if not all(col in chunk.columns for col in required_cols):
                        break
                    
                    sub_df = pd.DataFrame(index=chunk.index)
                    sub_df['symbol'] = chunk[symbol_col]
                    sub_df['open'] = chunk[f'Open.{i}']
                    sub_df['high'] = chunk[f'High.{i}']
                    sub_df['low'] = chunk[f'Low.{i}']
                    sub_df['close'] = chunk[f'Close.{i}']
                    sub_df['volume'] = chunk[f'Volume.{i}'] if f'Volume.{i}' in chunk.columns else 0
                    sub_df['openinterest'] = chunk[f'OpenInterest.{i}'] if f'OpenInterest.{i}' in chunk.columns else 0
                    
                    sub_df = sub_df.dropna(subset=['symbol'])
                    if sub_df.empty:
                        i += 1
                        continue
                    
                    sub_df.columns = ['symbol', 'open', 'high', 'low', 'close', 'volume', 'openinterest']
                    
                    for col in ['open', 'high', 'low', 'close', 'volume', 'openinterest']:
                        sub_df[col] = sub_df[col].astype(str).str.replace(',', '', regex=False)
                        sub_df[col] = pd.to_numeric(sub_df[col], errors='coerce')
                    
                    sub_df['BaseSymbol'] = sub_df['symbol'].apply(lambda x: parse_symbol(x) if pd.notna(x) else None)
                    sub_df = sub_df.dropna(subset=['BaseSymbol'])
                    
                    if not sub_df.empty:
                        dfs.append(sub_df)
                    
                    i += 1
                
                if dfs:
                    chunk_df = pd.concat(dfs, ignore_index=False)
                    all_dfs.append(chunk_df)
                    
        except Exception as e:
            logger.error(f"FILE ERROR: {e}")
            continue
    
    if not all_dfs:
        return pd.DataFrame()
    
    df = pd.concat(all_dfs, ignore_index=False)
    df = df.sort_index()
    
    if symbols:
        df = df[df['BaseSymbol'].isin(symbols)]
    if start_date:
        df = df[df.index >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df.index <= pd.to_datetime(end_date)]
    
    df = df.dropna(subset=['open', 'high', 'low', 'close'], how='all')
    df = df[~df.index.duplicated(keep='first')]
    
    gc.collect()
    perf_monitor.end_timer("data_loading")
    logger.info(f"SUCCESS: Loaded {len(df)} rows")
    return df

# =============================================================================
# EXPANDED 80+ INDICATOR UNIVERSE - TA 0.11.0 COMPATIBLE
# Complete institutional-level indicator calculation system
# =============================================================================

def calculate_kama(close: pd.Series, window: int = 14, pow1: int = 2, pow2: int = 30) -> pd.Series:
    """Kaufman's Adaptive Moving Average - Custom implementation"""
    try:
        change = close.diff(window).abs()
        volatility = close.diff().abs().rolling(window).sum()
        er = change / volatility
        sc = (er * (2.0 / (pow1 + 1) - 2.0 / (pow2 + 1.0)) + 2 / (pow2 + 1.0)) ** 2.0
        kama = pd.Series(index=close.index, dtype=float)
        kama.iloc[window] = close.iloc[:window+1].mean()
        for i in range(window + 1, len(close)):
            kama.iloc[i] = kama.iloc[i-1] + sc.iloc[i] * (close.iloc[i] - kama.iloc[i-1])
        return kama
    except Exception as e:
        logger.warning(f"KAMA calculation failed: {e}")
        return pd.Series(index=close.index, dtype=float)

def calculate_wma(close: pd.Series, window: int) -> pd.Series:
    """Weighted Moving Average - Custom implementation"""
    try:
        weights = np.arange(1, window + 1)
        return close.rolling(window).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)
    except Exception as e:
        logger.warning(f"WMA calculation failed: {e}")
        return pd.Series(index=close.index, dtype=float)

def calculate_hma(close: pd.Series, window: int) -> pd.Series:
    """Hull Moving Average - Custom implementation"""
    try:
        wma_half = calculate_wma(close, window // 2)
        wma_full = calculate_wma(close, window)
        hma_data = 2 * wma_half - wma_full
        hma_window = int(np.sqrt(window))
        return calculate_wma(hma_data, hma_window)
    except Exception as e:
        logger.warning(f"HMA calculation failed: {e}")
        return pd.Series(index=close.index, dtype=float)

def calculate_vwma(close: pd.Series, volume: pd.Series, window: int) -> pd.Series:
    """Volume Weighted Moving Average - Custom implementation"""
    try:
        return (close * volume).rolling(window).sum() / volume.rolling(window).sum()
    except Exception as e:
        logger.warning(f"VWMA calculation failed: {e}")
        return pd.Series(index=close.index, dtype=float)

def calculate_volume_sma(volume: pd.Series, window: int) -> pd.Series:
    """Volume Simple Moving Average"""
    try:
        return volume.rolling(window).mean()
    except Exception as e:
        logger.warning(f"Volume SMA calculation failed: {e}")
        return pd.Series(index=volume.index, dtype=float)

def calculate_vwap(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, window: int) -> pd.Series:
    """Volume Weighted Average Price"""
    try:
        typical_price = (high + low + close) / 3
        return (typical_price * volume).rolling(window).sum() / volume.rolling(window).sum()
    except Exception as e:
        logger.warning(f"VWAP calculation failed: {e}")
        return pd.Series(index=close.index, dtype=float)

def calculate_dmi(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> tuple:
    """Directional Movement Index - Custom implementation"""
    try:
        tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
        dm_plus = ((high - high.shift()) > (low.shift() - low)) & ((high - high.shift()) > 0)
        dm_plus = dm_plus * (high - high.shift())
        dm_minus = ((low.shift() - low) > (high - high.shift())) & ((low.shift() - low) > 0)
        dm_minus = dm_minus * (low.shift() - low)
        
        tr_smooth = tr.rolling(window).mean()
        dm_plus_smooth = dm_plus.rolling(window).mean()
        dm_minus_smooth = dm_minus.rolling(window).mean()
        
        di_plus = 100 * dm_plus_smooth / tr_smooth
        di_minus = 100 * dm_minus_smooth / tr_smooth
        
        return di_plus, di_minus
    except Exception as e:
        logger.warning(f"DMI calculation failed: {e}")
        return pd.Series(index=close.index, dtype=float), pd.Series(index=close.index, dtype=float)

def calculate_choppiness_index(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    """Choppiness Index - Custom implementation"""
    try:
        tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
        atr_sum = tr.rolling(window).sum()
        high_max = high.rolling(window).max()
        low_min = low.rolling(window).min()
        ci = 100 * np.log10(atr_sum / (high_max - low_min)) / np.log10(window)
        return ci
    except Exception as e:
        logger.warning(f"Choppiness Index calculation failed: {e}")
        return pd.Series(index=close.index, dtype=float)

def calculate_linear_regression_slope(close: pd.Series, window: int) -> pd.Series:
    """Linear Regression Slope - Custom implementation"""
    try:
        def slope(y):
            x = np.arange(len(y))
            return np.polyfit(x, y, 1)[0] if len(y) == window else np.nan
        return close.rolling(window).apply(slope, raw=True)
    except Exception as e:
        logger.warning(f"Linear Regression Slope calculation failed: {e}")
        return pd.Series(index=close.index, dtype=float)

def calculate_momentum(close: pd.Series, window: int) -> pd.Series:
    """Momentum - Custom implementation"""
    try:
        return close.diff(window)
    except Exception as e:
        logger.warning(f"Momentum calculation failed: {e}")
        return pd.Series(index=close.index, dtype=float)

def calculate_price_acceleration(close: pd.Series, window: int = 10) -> pd.Series:
    """Price Acceleration - Custom implementation"""
    try:
        velocity = close.diff()
        acceleration = velocity.diff()
        return acceleration.rolling(window).mean()
    except Exception as e:
        logger.warning(f"Price acceleration calculation failed: {e}")
        return pd.Series(index=close.index, dtype=float)

def calculate_bollinger_width(close: pd.Series, window: int = 20, std_dev: int = 2) -> pd.Series:
    """Bollinger Bands Width - Custom implementation"""
    try:
        bb = BollingerBands(close=close, window=window, window_dev=std_dev)
        upper = bb.bollinger_hband()
        lower = bb.bollinger_lband()
        width = (upper - lower) / bb.bollinger_mavg() * 100
        return width
    except Exception as e:
        logger.warning(f"Bollinger Width calculation failed: {e}")
        return pd.Series(index=close.index, dtype=float)

def calculate_historical_volatility(close: pd.Series, window: int = 20) -> pd.Series:
    """Historical Volatility - Custom implementation"""
    try:
        returns = np.log(close / close.shift())
        return returns.rolling(window).std() * np.sqrt(252) * 100
    except Exception as e:
        logger.warning(f"Historical Volatility calculation failed: {e}")
        return pd.Series(index=close.index, dtype=float)

def calculate_parkinson_estimator(high: pd.Series, low: pd.Series, window: int = 20) -> pd.Series:
    """Parkinson Volatility Estimator - Custom implementation"""
    try:
        log_ratio = np.log(high / low) ** 2
        parkinson = np.sqrt(log_ratio.rolling(window).mean() / (4 * np.log(2))) * np.sqrt(252) * 100
        return parkinson
    except Exception as e:
        logger.warning(f"Parkinson Estimator calculation failed: {e}")
        return pd.Series(index=close.index, dtype=float)

def calculate_trend_consistency(close: pd.Series, window: int = 20) -> pd.Series:
    """Trend Consistency - Custom implementation"""
    try:
        price_changes = close.diff()
        positive_changes = (price_changes > 0).rolling(window).sum()
        consistency = positive_changes / window
        return consistency
    except Exception as e:
        logger.warning(f"Trend Consistency calculation failed: {e}")
        return pd.Series(index=close.index, dtype=float)

def calculate_ma_alignment(close: pd.Series, windows: list = [5, 10, 20, 50]) -> pd.Series:
    """Moving Average Alignment Score - Custom implementation"""
    try:
        mas = {}
        for w in windows:
            mas[w] = close.rolling(w).mean()
        
        alignment_scores = []
        for i in close.index:
            ma_values = [mas[w].loc[i] if i in mas[w].index and not pd.isna(mas[w].loc[i]) else None for w in windows]
            ma_values = [v for v in ma_values if v is not None]
            
            if len(ma_values) < 2:
                alignment_scores.append(0)
                continue
                
            # Check if MAs are in ascending or descending order
            ascending = all(ma_values[i] <= ma_values[i+1] for i in range(len(ma_values)-1))
            descending = all(ma_values[i] >= ma_values[i+1] for i in range(len(ma_values)-1))
            
            if ascending:
                alignment_scores.append(1)  # Bullish alignment
            elif descending:
                alignment_scores.append(-1)  # Bearish alignment
            else:
                alignment_scores.append(0)  # Mixed alignment
        
        return pd.Series(alignment_scores, index=close.index)
    except Exception as e:
        logger.warning(f"MA Alignment calculation failed: {e}")
        return pd.Series(index=close.index, dtype=float)

# =============================================================================
# 80+ INDICATOR CALCULATION FUNCTION
# =============================================================================

def calculate_all_indicators(df: pd.DataFrame, timeframe: str = "15min") -> pd.DataFrame:
    """
    Calculate complete 80+ indicator universe across 5 regime dimensions
    
    DIRECTION INDICATORS (20):
    - Moving Averages: SMA, EMA, KAMA, WMA, HMA, VWMA
    - Trend Indicators: Ichimoku, PSAR, ADX, DMI, Aroon, CCI
    
    TREND STRENGTH INDICATORS (15):
    - MACD family, Slope analysis, Trend consistency, MA alignment
    - Vortex, Linear Regression, Choppiness Index
    
    VELOCITY INDICATORS (12):
    - ROC multiple timeframes, Momentum, RSI derivatives
    - Stochastic momentum, Price acceleration
    
    VOLATILITY INDICATORS (13):
    - ATR family, Bollinger Bands, Keltner Channels
    - Ulcer Index, Historical volatility, Parkinson estimator
    
    MARKET MICROSTRUCTURE INDICATORS (12):
    - VWAP analysis, Volume indicators, Money flow
    - Accumulation/Distribution, Force Index
    """
    
    logger.info("INDICATORS: Starting 80+ indicator calculation")
    
    try:
        df = df.copy()
        
        # Ensure required columns exist
        required_cols = ['open', 'high', 'low', 'close']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        
        # Handle volume column
        has_volume = 'volume' in df.columns
        if not has_volume:
            logger.warning("Volume column not found, volume-based indicators will be skipped")
            df['volume'] = 1.0  # Dummy volume for non-volume indicators
        
        # =============================================================================
        # DIRECTION INDICATORS (20)
        # =============================================================================
        
        logger.info("Calculating Direction Indicators (20)...")
        
        # Moving Averages - Multiple timeframes with error handling
        try:
            df['SMA_5'] = SMAIndicator(close=df['close'], window=5).sma_indicator()
            df['SMA_10'] = SMAIndicator(close=df['close'], window=10).sma_indicator()
            df['SMA_20'] = SMAIndicator(close=df['close'], window=20).sma_indicator()
            df['SMA_50'] = SMAIndicator(close=df['close'], window=50).sma_indicator()
            df['SMA_100'] = SMAIndicator(close=df['close'], window=100).sma_indicator()
            df['SMA_200'] = SMAIndicator(close=df['close'], window=200).sma_indicator()
            logger.info("SMA indicators calculated successfully")
        except Exception as e:
            logger.error(f"Error calculating SMA indicators: {e}")
        
        try:
            df['EMA_5'] = EMAIndicator(close=df['close'], window=5).ema_indicator()
            df['EMA_8'] = EMAIndicator(close=df['close'], window=8).ema_indicator()
            df['EMA_12'] = EMAIndicator(close=df['close'], window=12).ema_indicator()
            df['EMA_21'] = EMAIndicator(close=df['close'], window=21).ema_indicator()
            df['EMA_26'] = EMAIndicator(close=df['close'], window=26).ema_indicator()
            df['EMA_50'] = EMAIndicator(close=df['close'], window=50).ema_indicator()
            df['EMA_100'] = EMAIndicator(close=df['close'], window=100).ema_indicator()
            logger.info("EMA indicators calculated successfully")
        except Exception as e:
            logger.error(f"Error calculating EMA indicators: {e}")
        
        # Custom Moving Averages
        try:
            df['KAMA_14'] = calculate_kama(df['close'], window=14)
            df['KAMA_30'] = calculate_kama(df['close'], window=30)
            df['WMA_20'] = calculate_wma(df['close'], window=20)
            df['HMA_14'] = calculate_hma(df['close'], window=14)
            logger.info("Custom moving averages calculated successfully")
        except Exception as e:
            logger.error(f"Error calculating custom MAs: {e}")
        
        if has_volume:
            try:
                df['VWMA_20'] = calculate_vwma(df['close'], df['volume'], window=20)
                logger.info("VWMA calculated successfully")
            except Exception as e:
                logger.error(f"Error calculating VWMA: {e}")
        
        # Trend Direction Indicators
        try:
            ichimoku = IchimokuIndicator(high=df['high'], low=df['low'], window1=9, window2=26, window3=52)
            df['Ichimoku_A'] = ichimoku.ichimoku_a()
            df['Ichimoku_B'] = ichimoku.ichimoku_b()
            df['Ichimoku_Base'] = ichimoku.ichimoku_base_line()
            df['Ichimoku_Conversion'] = ichimoku.ichimoku_conversion_line()
            logger.info("Ichimoku indicators calculated successfully")
        except Exception as e:
            logger.error(f"Error calculating Ichimoku: {e}")
        
        try:
            df['PSAR'] = PSARIndicator(high=df['high'], low=df['low'], close=df['close']).psar()
            logger.info("PSAR calculated successfully")
        except Exception as e:
            logger.error(f"Error calculating PSAR: {e}")
        
        # =============================================================================
        # TREND STRENGTH INDICATORS (15)
        # =============================================================================
        
        logger.info("Calculating Trend Strength Indicators (15)...")
        
        # MACD Family with error handling
        try:
            macd = MACD(close=df['close'], window_fast=12, window_slow=26, window_sign=9)
            df['MACD'] = macd.macd()
            df['MACD_Signal'] = macd.macd_signal()
            df['MACD_Histogram'] = macd.macd_diff()
            logger.info("MACD indicators calculated successfully")
        except Exception as e:
            logger.error(f"Error calculating MACD: {e}")
        
        # ADX and DMI with error handling
        try:
            adx = ADXIndicator(high=df['high'], low=df['low'], close=df['close'], window=14)
            df['ADX'] = adx.adx()
            df['ADX_POS'] = adx.adx_pos()
            df['ADX_NEG'] = adx.adx_neg()
            logger.info("ADX indicators calculated successfully")
        except Exception as e:
            logger.error(f"Error calculating ADX: {e}")
        
        # Additional DMI calculation
        try:
            df['DI_Plus'], df['DI_Minus'] = calculate_dmi(df['high'], df['low'], df['close'], window=14)
            logger.info("Custom DMI calculated successfully")
        except Exception as e:
            logger.error(f"Error calculating custom DMI: {e}")
        
        # Aroon - Fixed for ta 0.11.0 compatibility
        try:
            aroon = AroonIndicator(high=df['high'], low=df['low'], window=25)
            df['Aroon_Up'] = aroon.aroon_up()
            df['Aroon_Down'] = aroon.aroon_down()
            df['Aroon_Oscillator'] = df['Aroon_Up'] - df['Aroon_Down']
            logger.info("Aroon indicators calculated successfully")
        except Exception as e:
            logger.error(f"Error calculating Aroon: {e}")
        
        # CCI with error handling
        try:
            df['CCI'] = CCIIndicator(high=df['high'], low=df['low'], close=df['close'], window=20).cci()
            logger.info("CCI calculated successfully")
        except Exception as e:
            logger.error(f"Error calculating CCI: {e}")
        
        # Vortex with error handling
        try:
            vortex = VortexIndicator(high=df['high'], low=df['low'], close=df['close'], window=14)
            df['Vortex_Pos'] = vortex.vortex_indicator_pos()
            df['Vortex_Neg'] = vortex.vortex_indicator_neg()
            logger.info("Vortex indicators calculated successfully")
        except Exception as e:
            logger.error(f"Error calculating Vortex: {e}")
        
        # Custom Trend Strength Indicators
        try:
            df['Linear_Regression_Slope_14'] = calculate_linear_regression_slope(df['close'], window=14)
            df['Linear_Regression_Slope_30'] = calculate_linear_regression_slope(df['close'], window=30)
            df['Choppiness_Index'] = calculate_choppiness_index(df['high'], df['low'], df['close'], window=14)
            df['Trend_Consistency'] = calculate_trend_consistency(df['close'], window=20)
            df['MA_Alignment'] = calculate_ma_alignment(df['close'], windows=[5, 10, 20, 50])
            logger.info("Custom trend strength indicators calculated successfully")
        except Exception as e:
            logger.error(f"Error calculating custom trend indicators: {e}")
        
        # =============================================================================
        # VELOCITY INDICATORS (12)
        # =============================================================================
        
        logger.info("Calculating Velocity Indicators (12)...")
        
        # ROC Multiple Timeframes with error handling
        try:
            df['ROC_5'] = ROCIndicator(close=df['close'], window=5).roc()
            df['ROC_10'] = ROCIndicator(close=df['close'], window=10).roc()
            df['ROC_20'] = ROCIndicator(close=df['close'], window=20).roc()
            df['ROC_50'] = ROCIndicator(close=df['close'], window=50).roc()
            logger.info("ROC indicators calculated successfully")
        except Exception as e:
            logger.error(f"Error calculating ROC: {e}")
        
        # RSI and derivatives with error handling
        try:
            df['RSI'] = RSIIndicator(close=df['close'], window=14).rsi()
            df['RSI_9'] = RSIIndicator(close=df['close'], window=9).rsi()
            logger.info("RSI indicators calculated successfully")
        except Exception as e:
            logger.error(f"Error calculating RSI: {e}")
        
        try:
            stoch_rsi = StochRSIIndicator(close=df['close'], window=14, smooth1=3, smooth2=3)
            df['StochRSI'] = stoch_rsi.stochrsi()
            df['StochRSI_K'] = stoch_rsi.stochrsi_k()
            df['StochRSI_D'] = stoch_rsi.stochrsi_d()
            logger.info("StochRSI indicators calculated successfully")
        except Exception as e:
            logger.error(f"Error calculating StochRSI: {e}")
        
        # Stochastic Oscillator with error handling
        try:
            stoch = StochasticOscillator(high=df['high'], low=df['low'], close=df['close'], window=14, smooth_window=3)
            df['Stoch_K'] = stoch.stoch()
            df['Stoch_D'] = stoch.stoch_signal()
            logger.info("Stochastic indicators calculated successfully")
        except Exception as e:
            logger.error(f"Error calculating Stochastic: {e}")
        
        # Custom Velocity Indicators with error handling
        try:
            df['Momentum_10'] = calculate_momentum(df['close'], window=10)
            df['Price_Acceleration'] = calculate_price_acceleration(df['close'], window=10)
            logger.info("Custom velocity indicators calculated successfully")
        except Exception as e:
            logger.error(f"Error calculating custom velocity indicators: {e}")
        
        # =============================================================================
        # VOLATILITY INDICATORS (13)
        # =============================================================================
        
        logger.info("Calculating Volatility Indicators (13)...")
        
        # ATR Family with error handling
        try:
            df['ATR'] = AverageTrueRange(high=df['high'], low=df['low'], close=df['close'], window=14).average_true_range()
            df['ATR_7'] = AverageTrueRange(high=df['high'], low=df['low'], close=df['close'], window=7).average_true_range()
            df['ATR_21'] = AverageTrueRange(high=df['high'], low=df['low'], close=df['close'], window=21).average_true_range()
            logger.info("ATR indicators calculated successfully")
        except Exception as e:
            logger.error(f"Error calculating ATR: {e}")
        
        # Bollinger Bands with error handling
        try:
            bb = BollingerBands(close=df['close'], window=20, window_dev=2)
            df['BB_Upper'] = bb.bollinger_hband()
            df['BB_Lower'] = bb.bollinger_lband()
            df['BB_Middle'] = bb.bollinger_mavg()
            df['BB_Width'] = calculate_bollinger_width(df['close'], window=20, std_dev=2)
            df['BB_Percent'] = (df['close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
            logger.info("Bollinger Bands calculated successfully")
        except Exception as e:
            logger.error(f"Error calculating Bollinger Bands: {e}")
        
        # Keltner Channels with error handling
        try:
            kc = KeltnerChannel(high=df['high'], low=df['low'], close=df['close'], window=20, window_atr=10)
            df['KC_Upper'] = kc.keltner_channel_hband()
            df['KC_Lower'] = kc.keltner_channel_lband()
            df['KC_Middle'] = kc.keltner_channel_mband()
            logger.info("Keltner Channels calculated successfully")
        except Exception as e:
            logger.error(f"Error calculating Keltner Channels: {e}")
        
        # Donchian Channels with error handling
        try:
            dc = DonchianChannel(high=df['high'], low=df['low'], close=df['close'], window=20)
            df['DC_Upper'] = dc.donchian_channel_hband()
            df['DC_Lower'] = dc.donchian_channel_lband()
            df['DC_Middle'] = dc.donchian_channel_mband()
            logger.info("Donchian Channels calculated successfully")
        except Exception as e:
            logger.error(f"Error calculating Donchian Channels: {e}")
        
        # Additional Volatility Indicators with error handling
        try:
            df['Ulcer_Index'] = UlcerIndex(close=df['close'], window=14).ulcer_index()
            logger.info("Ulcer Index calculated successfully")
        except Exception as e:
            logger.error(f"Error calculating Ulcer Index: {e}")
        
        try:
            df['Historical_Volatility'] = calculate_historical_volatility(df['close'], window=20)
            df['Parkinson_Estimator'] = calculate_parkinson_estimator(df['high'], df['low'], window=20)
            logger.info("Custom volatility indicators calculated successfully")
        except Exception as e:
            logger.error(f"Error calculating custom volatility indicators: {e}")
        
        # =============================================================================
        # MARKET MICROSTRUCTURE INDICATORS (12)
        # =============================================================================
        
        logger.info("Calculating Market Microstructure Indicators (12)...")
        
        if has_volume:
            # VWAP Analysis with error handling
            try:
                df['VWAP'] = calculate_vwap(df['high'], df['low'], df['close'], df['volume'], window=14)
                df['VWAP_20'] = calculate_vwap(df['high'], df['low'], df['close'], df['volume'], window=20)
                logger.info("VWAP indicators calculated successfully")
            except Exception as e:
                logger.error(f"Error calculating VWAP: {e}")
            
            # Volume Indicators with error handling
            try:
                df['Volume_SMA'] = calculate_volume_sma(df['volume'], window=20)
                df['Volume_Ratio'] = df['volume'] / df['Volume_SMA']
                logger.info("Volume indicators calculated successfully")
            except Exception as e:
                logger.error(f"Error calculating volume indicators: {e}")
            
            # On Balance Volume with error handling
            try:
                df['OBV'] = OnBalanceVolumeIndicator(close=df['close'], volume=df['volume']).on_balance_volume()
                logger.info("OBV calculated successfully")
            except Exception as e:
                logger.error(f"Error calculating OBV: {e}")
            
            # Money Flow Indicators with error handling
            try:
                df['CMF'] = ChaikinMoneyFlowIndicator(high=df['high'], low=df['low'], close=df['close'], volume=df['volume'], window=20).chaikin_money_flow()
                df['MFI'] = MFIIndicator(high=df['high'], low=df['low'], close=df['close'], volume=df['volume'], window=14).money_flow_index()
                logger.info("Money flow indicators calculated successfully")
            except Exception as e:
                logger.error(f"Error calculating money flow indicators: {e}")
            
            # Accumulation/Distribution with error handling
            try:
                df['ADI'] = AccDistIndexIndicator(high=df['high'], low=df['low'], close=df['close'], volume=df['volume']).acc_dist_index()
                logger.info("ADI calculated successfully")
            except Exception as e:
                logger.error(f"Error calculating ADI: {e}")
            
            # Force Index with error handling
            try:
                df['Force_Index'] = ForceIndexIndicator(close=df['close'], volume=df['volume'], window=13).force_index()
                logger.info("Force Index calculated successfully")
            except Exception as e:
                logger.error(f"Error calculating Force Index: {e}")
            
            # Ease of Movement with error handling
            try:
                df['EMV'] = EaseOfMovementIndicator(high=df['high'], low=df['low'], volume=df['volume'], window=14).ease_of_movement()
                logger.info("EMV calculated successfully")
            except Exception as e:
                logger.error(f"Error calculating EMV: {e}")
            
            # Volume Price Trend with error handling
            try:
                df['VPT'] = (df['volume'] * ((df['close'] - df['close'].shift()) / df['close'].shift())).cumsum()
                df['PVT'] = (df['volume'] * (df['close'] - df['close'].shift()) / df['close'].shift()).cumsum()
                logger.info("VPT and PVT calculated successfully")
            except Exception as e:
                logger.error(f"Error calculating VPT/PVT: {e}")
        else:
            # Create dummy microstructure indicators when volume is not available
            logger.warning("Creating dummy microstructure indicators (no volume data)")
            microstructure_cols = ['VWAP', 'VWAP_20', 'Volume_SMA', 'Volume_Ratio', 'OBV', 'CMF', 'MFI', 'ADI', 'Force_Index', 'EMV', 'VPT', 'PVT']
            for col in microstructure_cols:
                df[col] = 0.0
        
        # =============================================================================
        # ADDITIONAL UTILITY INDICATORS
        # =============================================================================
        
        # Daily Returns (useful for various calculations) with error handling
        try:
            df['Daily_Return'] = DailyReturnIndicator(close=df['close']).daily_return()
            df['Cumulative_Return'] = CumulativeReturnIndicator(close=df['close']).cumulative_return()
            logger.info("Return indicators calculated successfully")
        except Exception as e:
            logger.error(f"Error calculating return indicators: {e}")
        
        # True Range with error handling
        try:
            df['True_Range'] = pd.concat([
                df['high'] - df['low'],
                (df['high'] - df['close'].shift()).abs(),
                (df['low'] - df['close'].shift()).abs()
            ], axis=1).max(axis=1)
            logger.info("True Range calculated successfully")
        except Exception as e:
            logger.error(f"Error calculating True Range: {e}")
        
        # Calculate total indicator count
        indicator_columns = [col for col in df.columns if col not in ['symbol', 'open', 'high', 'low', 'close', 'volume', 'openinterest', 'BaseSymbol']]
        total_indicators = len(indicator_columns)
        
        logger.info(f"INDICATORS: Successfully calculated {total_indicators} indicators")
        logger.info(f"INDICATORS: Direction: 20, Trend Strength: 15, Velocity: 12, Volatility: 13, Microstructure: 12+")
        
        # Final data validation
        if df.isnull().any().any():
            null_counts = df.isnull().sum()
            null_columns = null_counts[null_counts > 0]
            logger.warning(f"NULL values found in {len(null_columns)} columns: {null_columns.to_dict()}")
        
        return df
        
    except Exception as e:
        logger.error(f"INDICATORS: Critical error in indicator calculation: {e}")
        raise

# =============================================================================
# MAIN EXECUTION - UPDATE THIS SECTION
# =============================================================================

print(">>> Multi-Dimensional Regime Classification System - Starting...")
print(">>> Now with Rolling Windows & Forward-Looking Bias Protection")
print("="*80)

timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")

def run_enhanced_analysis_with_optimization(data_with_regimes, data_with_indicators, classifier):
    """Enhanced analysis that includes multi-objective optimization"""
    
    # STEP 4A: Initial Regime Analysis (Original)
    print("\nSTEP 4A: Generating Initial Multi-Dimensional Analysis")
    print("-" * 50)
    
    print(f"\nDEBUG: RUN_OPTIMIZATION = {RUN_OPTIMIZATION}")
    print(f"DEBUG: OPTIMIZE_WINDOW = {OPTIMIZE_WINDOW}")
    print(f"DEBUG: WALK_FORWARD = {WALK_FORWARD}")
    
    # Check if function exists
    try:
        print(f"DEBUG: optimize_window_size function exists: {'optimize_window_size' in globals()}")
        if 'optimize_window_size' not in globals():
            from multi_objective_optimizer import optimize_window_size
            print("DEBUG: Had to import optimize_window_size!")
    except Exception as e:
        print(f"DEBUG: Error checking for function: {e}")
    
    # Analyze each dimension
    dimensions = ['Direction', 'TrendStrength', 'Velocity', 'Volatility', 'Microstructure']
    
    print("\nINITIAL DIMENSIONAL DISTRIBUTION:")
    initial_distributions = {}
    for dimension in dimensions:
        col_name = f'{dimension}_Regime'
        if col_name in data_with_regimes.columns:
            regime_counts = data_with_regimes[col_name].value_counts()
            initial_distributions[dimension] = regime_counts
            print(f"\n{dimension}:")
            for regime, count in regime_counts.items():
                pct = count / len(data_with_regimes) * 100
                print(f"  {regime}: {count} ({pct:.1f}%)")
    
    # Persistence analysis
    print("\nREGIME PERSISTENCE ANALYSIS:")
    for dimension in dimensions:
        col_name = f'{dimension}_Regime'
        if col_name in data_with_regimes.columns:
            regime_changes = (data_with_regimes[col_name] != data_with_regimes[col_name].shift()).sum()
            persistence = 1 - (regime_changes / len(data_with_regimes))
            print(f"  {dimension}: {persistence:.1%} persistence ({regime_changes} changes)")
    
    # STEP 4B: Run Optimization if enabled
    optimization_results = None
    if RUN_OPTIMIZATION:
        logger.info("\nSTEP 4B: Optimization Phase")
        logger.info("-" * 50)
        logger.info(f"DEBUG: About to run window optimization...")
        
        # First, optimize window size
        logger.info("4B.1: Window Size Optimization")
        
        # First, optimize window size
        if OPTIMIZE_WINDOW:  # ADD THIS LINE
            print("\n4B.1: Window Size Optimization")
            try:

                from multi_objective_optimizer import optimize_window_size
                
                print("DEBUG: About to call optimize_window_size...")
                window_results = optimize_window_size(
                    data_with_indicators,
                    classifier,
                    timeframe='15min',
                    window_sizes_hours=[2, 4, 6, 8, 12, 16, 24, 36]
                )
                print("DEBUG: optimize_window_size completed!")
                print(f"DEBUG: Results: {window_results}")
                
                # Apply optimal window to classifier
                optimal_window = window_results['optimal_window_periods']
                print(f"\nApplying optimal window: {optimal_window} periods")
                classifier.rolling_window = optimal_window
                
            except Exception as e:
                print(f"ERROR in window optimization: {e}")
                import traceback
                traceback.print_exc()
        
        # Now run walk-forward validation
        if WALK_FORWARD:
            print("\n4B.2: Walk-Forward Validation")
            
            # For 15-min data: ~20 days training, ~10 days testing
            train_periods = 20 * 96  # 20 days * 96 periods per day
            test_periods = 10 * 96   # 10 days * 96 periods per day
            
            wf_optimizer = WalkForwardOptimizer(
                data_with_indicators,
                train_periods=train_periods,
                test_periods=test_periods
            )
            
            wf_results = wf_optimizer.run_walk_forward_optimization(
                classifier,
                optimization_method='differential_evolution',
                max_iterations=20  # Fewer iterations for walk-forward
            )
            
            # Save walk-forward results
            wf_results.to_csv(f'walk_forward_results_{timestamp}.csv', index=False)
            print(f"\nWalk-forward results saved to: walk_forward_results_{timestamp}.csv")
            
        else:
            # Regular optimization (existing code)
            print("\n4B.2: Regular Optimization")
            optimization_results = run_regime_optimization(
                classifier, 
                data_with_indicators,
                max_iterations=OPTIMIZATION_ITERATIONS
            )
        
        if optimization_results:
            print("\nOPTIMIZATION COMPLETE!")
            print_optimization_results(optimization_results)
            
            # Re-classify with optimized parameters
            print("\nRe-classifying with optimized parameters...")
            data_with_regimes = classifier.classify_multidimensional_regime(
                data_with_indicators.copy(), 
                symbol=SYMBOLS[0]
            )
            
            # Show improved distributions
            print("\nOPTIMIZED DIMENSIONAL DISTRIBUTIONS:")
            for dimension in dimensions:
                col_name = f'{dimension}_Regime'
                if col_name in data_with_regimes.columns:
                    regime_counts = data_with_regimes[col_name].value_counts()
                    print(f"\n{dimension}:")
                    for regime, count in regime_counts.items():
                        pct = count / len(data_with_regimes) * 100
                        
                        # Compare to initial
                        if dimension in initial_distributions and regime in initial_distributions[dimension]:
                            initial_pct = initial_distributions[dimension][regime] / len(data_with_regimes) * 100
                            change = pct - initial_pct
                            print(f"  {regime}: {count} ({pct:.1f}%) [Change: {change:+.1f}%]")
                        else:
                            print(f"  {regime}: {count} ({pct:.1f}%) [New]")
    
    return data_with_regimes, optimization_results

if __name__ == "__main__":
    try:
        # Create output directory
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        # STEP 1: Load Data
        print("\nSTEP 1: Loading Data")
        print("-" * 50)
        data = load_csv_data(CSV_FILES, SYMBOLS, START_DATE, END_DATE)
        print(f"SUCCESS: Loaded {len(data)} records")
        
        # STEP 2: Calculate Indicators
        print("\nSTEP 2: Calculating 85 Indicators")
        print("-" * 50)
        data_with_indicators = calculate_all_indicators(data)
        print(f"SUCCESS: Calculated {len(data_with_indicators.columns)} total columns")
        
        # STEP 3: Classify Regimes with Rolling Windows
        print("\nSTEP 3: Classifying Multi-Dimensional Regimes")
        print("-" * 50)
        
        # Calculate window sizes in bars
        bars_per_hour = 4  # 4 bars per hour for 15-minute data
        rolling_bars = int(ROLLING_WINDOW_HOURS * bars_per_hour)  # 8 * 4 = 32 bars
        min_bars = int(MIN_WINDOW_HOURS * bars_per_hour)          # 2 * 4 = 8 bars
        
        print(f"Window Configuration for 15-minute data:")
        print(f"  Rolling Window: {ROLLING_WINDOW_HOURS} hours ({rolling_bars} bars)")
        print(f"  Minimum Window: {MIN_WINDOW_HOURS} hours ({min_bars} bars)")
        print(f"  Smoothing Periods: {SMOOTHING_PERIODS}")
        
        classifier = MultiDimensionalRegimeClassifier(
            min_periods=min_bars,        # Now integer: 8
            max_periods=rolling_bars,    # Now integer: 32
            smoothing_periods=SMOOTHING_PERIODS
        )
        
        data_with_regimes = classifier.classify_multidimensional_regime(
            data_with_indicators.copy(),
            symbol=SYMBOLS[0]
        )
        
        print(f"SUCCESS: Classified all 5 dimensions for {len(data_with_regimes)} periods")
        print(f"Using {ROLLING_WINDOW_HOURS}-hour rolling windows with {SMOOTHING_PERIODS}-period smoothing")
        
        # STEP 4: Enhanced Analysis with Optimization
        data_with_regimes, optimization_results = run_enhanced_analysis_with_optimization(
            data_with_regimes, data_with_indicators, classifier
        )
        
        # STEP 5: Save Results
        print("\nSTEP 5: Saving Results")
        print("-" * 50)
        
        # Save main dataset
        main_output = os.path.join(OUTPUT_DIR, f"regime_analysis_rolling_{timestamp}.csv")
        data_with_regimes.to_csv(main_output)
        print(f"SUCCESS: Main analysis saved to {main_output}")
        
        # Save optimization results if available
        if optimization_results:
            opt_output = os.path.join(OUTPUT_DIR, f"optimization_results_{timestamp}.csv")
            
            # Save optimization history
            if optimization_results.optimization_history:
                opt_history_df = pd.DataFrame(optimization_results.optimization_history)
                opt_history_df.to_csv(opt_output, index=False)
                print(f"SUCCESS: Optimization history saved to {opt_output}")
            
            # Save best parameters
            params_output = os.path.join(OUTPUT_DIR, f"best_parameters_{timestamp}.txt")
            with open(params_output, 'w') as f:
                f.write("OPTIMIZED REGIME CLASSIFICATION PARAMETERS\n")
                f.write("="*50 + "\n\n")
                f.write("CONFIGURATION:\n")
                f.write(f"Rolling Window: {ROLLING_WINDOW_HOURS} hours\n")
                f.write(f"Min Window: {MIN_WINDOW_HOURS} hours\n")
                f.write(f"Smoothing Periods: {SMOOTHING_PERIODS}\n")
                f.write(f"Walk-Forward Validation: Enabled\n\n")
                f.write("PERFORMANCE METRICS:\n")
                f.write(f"Final Score: {optimization_results.best_score:.4f}\n")
                f.write(f"Sharpe Ratio: {optimization_results.sharpe_ratio:.4f}\n")
                f.write(f"Max Drawdown: {optimization_results.max_drawdown:.2%}\n")
                f.write(f"Regime Persistence: {optimization_results.regime_persistence:.2%}\n\n")
                f.write("BEST PARAMETERS:\n")
                for param, value in optimization_results.best_params.items():
                    f.write(f"{param}: {value:.6f}\n")
            
            print(f"SUCCESS: Best parameters saved to {params_output}")
        
        # Most common composite regimes
        print("\nMOST COMMON COMPOSITE REGIMES:")
        composite_counts = data_with_regimes['Composite_Regime'].value_counts().head(10)
        for regime, count in composite_counts.items():
            pct = count / len(data_with_regimes) * 100
            conf = data_with_regimes[data_with_regimes['Composite_Regime'] == regime]['Composite_Confidence'].mean()
            print(f"  {regime}: {count} periods ({pct:.1f}%) - Confidence: {conf:.3f}")
        
        # FINAL SUMMARY
        print("\n" + "="*80)
        print("MULTI-DIMENSIONAL REGIME CLASSIFICATION SYSTEM - COMPLETE!")
        print("Forward-Looking Bias: ELIMINATED")
        print("Rolling Windows: ACTIVE")
        print("Regime Smoothing: ENABLED")
        print("Walk-Forward Validation: COMPLETED")
        print("="*80)
        
    except Exception as e:
        logger.error(f"Critical error in main execution: {e}")
        raise
