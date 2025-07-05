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
# RUNONSAVE TEST

"""
Core regime classification components
Includes RollingRegimeClassifier, RegimeSmoother, and regime definitions
"""

import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Tuple, Any, Optional
from dataclasses import dataclass
from datetime import datetime
from tqdm import tqdm
import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import (
    DEFAULT_WINDOW_HOURS, REGIME_SMOOTHING_PERIODS,
    INDICATOR_WEIGHTS, DEFAULT_DIMENSION_THRESHOLDS,
    TIMEFRAMES, LOG_LEVEL, LOG_FORMAT
)

# Configure logging
logging.basicConfig(level=getattr(logging, LOG_LEVEL), format=LOG_FORMAT)
logger = logging.getLogger(__name__)

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

# =============================================================================
# DATA CLASSES
# =============================================================================

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

# =============================================================================
# REGIME SMOOTHER
# =============================================================================

class RegimeSmoother:
    """Smooth regime transitions to prevent whipsaws"""
    
    def __init__(self, confirmation_periods: int = REGIME_SMOOTHING_PERIODS):
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
                return new_regime, True
            else:
                # Not enough confirmations yet
                return current_regime, False
    
    def reset(self):
        """Reset all counters and regimes"""
        self.regime_counters = {}
        self.current_regimes = {}

# =============================================================================
# ROLLING REGIME CLASSIFIER
# =============================================================================

class RollingRegimeClassifier:
    """
    Point-in-time regime classification with rolling windows
    Eliminates forward-looking bias
    """
    
    def __init__(self, 
                 window_hours: float = DEFAULT_WINDOW_HOURS,
                 timeframe: str = '15min'):
        """
        Initialize rolling classifier
        
        Args:
            window_hours: Rolling window size in hours
            timeframe: Data timeframe for bar calculations
        """
        self.window_hours = window_hours
        self.timeframe = timeframe
        
        # Calculate window size in bars
        bars_per_day = TIMEFRAMES.get(timeframe, 26)
        self.window_bars = int((window_hours / 24) * bars_per_day)
        self.min_periods = max(self.window_bars // 2, 50)
        
        # Initialize components
        self.regime_smoother = RegimeSmoother()
        self.indicator_weights = INDICATOR_WEIGHTS
        self.dimension_thresholds = DEFAULT_DIMENSION_THRESHOLDS.copy()
        
        # Pre-calculated statistics storage
        self.rolling_stats = {}
        
        logger.info(f"Initialized RollingRegimeClassifier with {window_hours}h window ({self.window_bars} bars)")
    
    def update_window_size(self, new_window_hours: float):
        """Update the rolling window size"""
        self.window_hours = new_window_hours
        bars_per_day = TIMEFRAMES.get(self.timeframe, 26)
        self.window_bars = int((new_window_hours / 24) * bars_per_day)
        self.min_periods = max(self.window_bars // 2, 50)
        logger.info(f"Updated window size to {new_window_hours}h ({self.window_bars} bars)")
    
    def pre_calculate_rolling_statistics(self, data: pd.DataFrame):
        """
        Pre-calculate all rolling statistics for efficiency
        This is the key to avoiding repeated calculations
        """
        logger.info("Pre-calculating rolling statistics...")
        
        # Direction indicators
        if 'SMA_Signal' in data.columns:
            self.rolling_stats['SMA_pct_rank'] = data['SMA_Signal'].rolling(
                self.window_bars).rank(pct=True)
        
        if 'EMA_Signal' in data.columns:
            self.rolling_stats['EMA_pct_rank'] = data['EMA_Signal'].rolling(
                self.window_bars).rank(pct=True)
            
        if 'MACD_Signal' in data.columns:
            self.rolling_stats['MACD_pct_rank'] = data['MACD_Signal'].rolling(
                self.window_bars).rank(pct=True)
        
        # Trend strength indicators
        if 'ADX' in data.columns:
            self.rolling_stats['ADX_pct_rank'] = data['ADX'].rolling(
                self.window_bars).rank(pct=True)
        
        # Velocity indicators
        if 'ROC' in data.columns:
            self.rolling_stats['ROC_pct_rank'] = data['ROC'].rolling(
                self.window_bars).rank(pct=True)
            
        if 'Acceleration' in data.columns:
            self.rolling_stats['Acceleration_pct_rank'] = data['Acceleration'].rolling(
                self.window_bars).rank(pct=True)
        
        # Volatility indicators - using actual percentiles
        if 'ATR' in data.columns:
            self.rolling_stats['ATR_percentile'] = data['ATR'].rolling(
                self.window_bars).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100)
            
        if 'Historical_Vol' in data.columns:
            self.rolling_stats['Historical_Vol_percentile'] = data['Historical_Vol'].rolling(
                self.window_bars).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100)
        
        # Volume indicators
        if 'volume' in data.columns:
            self.rolling_stats['volume_ratio'] = data['volume'] / data['volume'].rolling(
                self.window_bars).mean()
            
        logger.info(f"Pre-calculated {len(self.rolling_stats)} rolling statistics")
    
    def classify_direction_dimension(self, data: pd.DataFrame, votes: List[DimensionalVote], 
                                   index: int) -> Tuple[str, float]:
        """Classify direction regime for a specific point in time"""
        try:
            direction_scores = {
                'Up_Trending': 0.0,
                'Down_Trending': 0.0,
                'Sideways': 0.0
            }
            total_weight = 0.0
            
            # Use pre-calculated rolling statistics
            if 'SMA_pct_rank' in self.rolling_stats and index < len(self.rolling_stats['SMA_pct_rank']):
                pct_rank = self.rolling_stats['SMA_pct_rank'].iloc[index]
                if pd.notna(pct_rank):
                    weight = self.indicator_weights['direction'].get('SMA', 1.0)
                    
                    if pct_rank > self.dimension_thresholds['direction']['strong_trend_threshold']:
                        direction_scores['Up_Trending'] += weight
                        regime_vote = 'Up_Trending'
                    elif pct_rank < self.dimension_thresholds['direction']['weak_trend_threshold']:
                        direction_scores['Down_Trending'] += weight
                        regime_vote = 'Down_Trending'
                    else:
                        direction_scores['Sideways'] += weight
                        regime_vote = 'Sideways'
                    
                    vote = DimensionalVote(
                        dimension='direction',
                        indicator_name='SMA',
                        regime_vote=regime_vote,
                        confidence=weight,
                        value=pct_rank
                    )
                    votes.append(vote)
                    total_weight += weight
            
            # Add more indicators as available
            if 'MACD_pct_rank' in self.rolling_stats and index < len(self.rolling_stats['MACD_pct_rank']):
                pct_rank = self.rolling_stats['MACD_pct_rank'].iloc[index]
                if pd.notna(pct_rank):
                    weight = self.indicator_weights['direction'].get('MACD', 1.2)
                    
                    if pct_rank > 0.65:
                        direction_scores['Up_Trending'] += weight
                        regime_vote = 'Up_Trending'
                    elif pct_rank < 0.35:
                        direction_scores['Down_Trending'] += weight
                        regime_vote = 'Down_Trending'
                    else:
                        direction_scores['Sideways'] += weight
                        regime_vote = 'Sideways'
                    
                    vote = DimensionalVote(
                        dimension='direction',
                        indicator_name='MACD',
                        regime_vote=regime_vote,
                        confidence=weight,
                        value=pct_rank
                    )
                    votes.append(vote)
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
        """Classify trend strength regime"""
        try:
            strength_scores = {
                'Strong': 0.0,
                'Moderate': 0.0,
                'Weak': 0.0
            }
            total_weight = 0.0
            
            # ADX-based strength
            if 'ADX_pct_rank' in self.rolling_stats and index < len(self.rolling_stats['ADX_pct_rank']):
                pct_rank = self.rolling_stats['ADX_pct_rank'].iloc[index]
                if pd.notna(pct_rank):
                    weight = self.indicator_weights['trend_strength'].get('ADX', 1.2)
                    
                    if pct_rank > self.dimension_thresholds['trend_strength']['strong_alignment']:
                        strength_scores['Strong'] += weight
                        regime_vote = 'Strong'
                    elif pct_rank > self.dimension_thresholds['trend_strength']['moderate_alignment']:
                        strength_scores['Moderate'] += weight
                        regime_vote = 'Moderate'
                    else:
                        strength_scores['Weak'] += weight
                        regime_vote = 'Weak'
                    
                    vote = DimensionalVote(
                        dimension='trend_strength',
                        indicator_name='ADX',
                        regime_vote=regime_vote,
                        confidence=weight,
                        value=pct_rank
                    )
                    votes.append(vote)
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
        """Classify velocity regime"""
        try:
            velocity_scores = {
                'Accelerating': 0.0,
                'Decelerating': 0.0,
                'Stable': 0.0
            }
            total_weight = 0.0
            
            # ROC-based velocity
            if 'ROC_pct_rank' in self.rolling_stats and index < len(self.rolling_stats['ROC_pct_rank']):
                pct_rank = self.rolling_stats['ROC_pct_rank'].iloc[index]
                if pd.notna(pct_rank):
                    weight = self.indicator_weights['velocity'].get('ROC', 1.0)
                    
                    if pct_rank > self.dimension_thresholds['velocity']['acceleration_threshold']:
                        velocity_scores['Accelerating'] += weight
                        regime_vote = 'Accelerating'
                    elif pct_rank < self.dimension_thresholds['velocity']['stable_range']:
                        velocity_scores['Decelerating'] += weight
                        regime_vote = 'Decelerating'
                    else:
                        velocity_scores['Stable'] += weight
                        regime_vote = 'Stable'
                    
                    vote = DimensionalVote(
                        dimension='velocity',
                        indicator_name='ROC',
                        regime_vote=regime_vote,
                        confidence=weight,
                        value=pct_rank
                    )
                    votes.append(vote)
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
        """Classify volatility regime"""
        try:
            vol_scores = {
                'Low_Vol': 0.0,
                'Medium_Vol': 0.0,
                'High_Vol': 0.0,
                'Extreme_Vol': 0.0
            }
            total_weight = 0.0
            
            # ATR-based volatility
            if 'ATR_percentile' in self.rolling_stats and index < len(self.rolling_stats['ATR_percentile']):
                percentile = self.rolling_stats['ATR_percentile'].iloc[index]
                if pd.notna(percentile):
                    weight = self.indicator_weights['volatility'].get('ATR', 1.0)
                    
                    if percentile > 90:
                        vol_scores['Extreme_Vol'] += weight
                        regime_vote = 'Extreme_Vol'
                    elif percentile > self.dimension_thresholds['volatility']['high_vol_percentile']:
                        vol_scores['High_Vol'] += weight
                        regime_vote = 'High_Vol'
                    elif percentile < self.dimension_thresholds['volatility']['low_vol_percentile']:
                        vol_scores['Low_Vol'] += weight
                        regime_vote = 'Low_Vol'
                    else:
                        vol_scores['Medium_Vol'] += weight
                        regime_vote = 'Medium_Vol'
                    
                    vote = DimensionalVote(
                        dimension='volatility',
                        indicator_name='ATR',
                        regime_vote=regime_vote,
                        confidence=weight,
                        value=percentile
                    )
                    votes.append(vote)
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
        """Classify microstructure regime"""
        try:
            micro_scores = {
                'Institutional_Flow': 0.0,
                'Retail_Flow': 0.0,
                'Balanced_Flow': 0.0,
                'Low_Participation': 0.0
            }
            total_weight = 0.0
            
            # Volume-based microstructure
            if 'volume_ratio' in self.rolling_stats and index < len(self.rolling_stats['volume_ratio']):
                vol_ratio = self.rolling_stats['volume_ratio'].iloc[index]
                if pd.notna(vol_ratio):
                    weight = self.indicator_weights['microstructure'].get('Volume', 1.0)
                    
                    if vol_ratio > self.dimension_thresholds['microstructure']['institutional_volume_threshold']:
                        micro_scores['Institutional_Flow'] += weight
                        regime_vote = 'Institutional_Flow'
                    elif vol_ratio < self.dimension_thresholds['microstructure']['retail_volume_threshold']:
                        micro_scores['Low_Participation'] += weight
                        regime_vote = 'Low_Participation'
                    else:
                        micro_scores['Balanced_Flow'] += weight
                        regime_vote = 'Balanced_Flow'
                    
                    vote = DimensionalVote(
                        dimension='microstructure',
                        indicator_name='Volume',
                        regime_vote=regime_vote,
                        confidence=weight,
                        value=vol_ratio
                    )
                    votes.append(vote)
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
    
    def classify_regimes(self, data: pd.DataFrame, 
                        show_progress: bool = True) -> pd.DataFrame:
        """
        Main method to classify all regimes
        
        Args:
            data: DataFrame with calculated indicators
            show_progress: Whether to show progress bar
            
        Returns:
            DataFrame with regime classifications
        """
        logger.info("Starting regime classification...")
        
        # Pre-calculate all rolling statistics
        self.pre_calculate_rolling_statistics(data)
        
        # Reset smoother
        self.regime_smoother.reset()
        
        # Initialize results
        results = pd.DataFrame(index=data.index)
        
        # Initialize columns
        for dim in ['Direction', 'TrendStrength', 'Velocity', 'Volatility', 'Microstructure']:
            results[f'{dim}_Regime'] = 'Undefined'
            results[f'{dim}_Confidence'] = 0.0
        results['Composite_Regime'] = 'Undefined'
        results['Composite_Confidence'] = 0.0
        
        # Progress bar setup
        iterator = range(len(data))
        if show_progress:
            iterator = tqdm(iterator, desc="Classifying regimes")
        
        # Process each time period
        for i in iterator:
            if i < self.min_periods:
                continue
            
            votes = []
            
            # Classify each dimension
            direction_regime, direction_conf = self.classify_direction_dimension(data, votes, i)
            trend_strength_regime, trend_strength_conf = self.classify_trend_strength_dimension(data, votes, i)
            velocity_regime, velocity_conf = self.classify_velocity_dimension(data, votes, i)
            volatility_regime, volatility_conf = self.classify_volatility_dimension(data, votes, i)
            microstructure_regime, microstructure_conf = self.classify_microstructure_dimension(data, votes, i)
            
            # Apply smoothing
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
            composite = f"{smoothed_direction}_{smoothed_trend_strength}_{smoothed_velocity}_{smoothed_volatility}_{smoothed_microstructure}"
            composite_conf = np.mean([direction_conf, trend_strength_conf, velocity_conf, 
                                     volatility_conf, microstructure_conf])
            
            results.loc[data.index[i], 'Composite_Regime'] = composite
            results.loc[data.index[i], 'Composite_Confidence'] = composite_conf
        
        logger.info(f"Regime classification complete for {len(results)} periods")
        
        return results
    
    def get_regime_statistics(self, regimes: pd.DataFrame) -> Dict[str, Any]:
        """Calculate statistics about regime classifications"""
        stats = {}
        
        # For each dimension
        for dim in ['Direction', 'TrendStrength', 'Velocity', 'Volatility', 'Microstructure']:
            col = f'{dim}_Regime'
            if col in regimes.columns:
                # Value counts
                counts = regimes[col].value_counts()
                total = len(regimes[regimes[col] != 'Undefined'])
                
                # Percentages
                percentages = (counts / total * 100).round(1) if total > 0 else counts * 0
                
                # Average confidence
                conf_col = f'{dim}_Confidence'
                avg_conf = regimes[conf_col].mean() if conf_col in regimes.columns else 0
                
                stats[dim] = {
                    'counts': counts.to_dict(),
                    'percentages': percentages.to_dict(),
                    'average_confidence': avg_conf,
                    'undefined_count': len(regimes[regimes[col] == 'Undefined'])
                }
        
        # Composite regime stats
        if 'Composite_Regime' in regimes.columns:
            composite_counts = regimes['Composite_Regime'].value_counts()
            stats['Composite'] = {
                'unique_regimes': len(composite_counts),
                'top_10_regimes': composite_counts.head(10).to_dict(),
                'average_confidence': regimes['Composite_Confidence'].mean()
            }
        
        return stats
