#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
1-Hour Early Warning System for Daily Regime Changes
Detects when hourly regimes diverge from daily, signaling potential transitions
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class RegimeDivergence:
    """Represents a divergence between hourly and daily regimes"""
    timestamp: pd.Timestamp
    daily_regime: str
    hourly_regime: str
    divergence_type: str  # 'direction', 'strength', 'volatility', 'character'
    divergence_strength: float  # 0-1 score
    hours_persisted: int
    confidence: float


class HourlyEarlyWarningSystem:
    """
    Detects early warning signals from 1-hour regime divergences
    Optimized for NQ futures based on market characterization
    """
    
    def __init__(self, daily_classifier, lookback_hours: int = 168):  # 168 hours = 7 days
        """
        Initialize early warning system
        
        Args:
            daily_classifier: Instance of NQDailyRegimeClassifier
            lookback_hours: Hours of history for hourly analysis
        """
        self.daily_classifier = daily_classifier
        self.lookback_hours = lookback_hours
        
        # NQ-specific parameters for hourly data
        self.hourly_thresholds = {
            # More sensitive than daily for early detection
            'direction_strong': 0.2,      # Lower than daily 0.3
            'direction_neutral': 0.1,     
            
            'strength_strong': 0.35,      # Lower than daily 0.38
            'strength_moderate': 0.2,     
            
            'vol_low': 20,               # More sensitive to vol changes
            'vol_normal': 70,
            'vol_high': 85,
            
            'efficiency_trending': 0.2,   # Lower than daily
            'efficiency_ranging': 0.12,   
            
            # Faster regime detection
            'smoothing_hours': 6,         # 6 hours vs 5-7 days
            'min_divergence_hours': 4,    # Minimum hours to confirm divergence
        }
        
        # Warning thresholds
        self.warning_levels = {
            'weak': 0.3,      # 30% of hourly periods diverging
            'moderate': 0.5,  # 50% diverging
            'strong': 0.7,    # 70% diverging
            'critical': 0.85  # 85% diverging = regime change imminent
        }
        
        # Store divergence history
        self.divergence_history = []
        
    def calculate_hourly_regimes(self, hourly_data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate regimes on hourly data with more sensitive parameters
        Similar to daily but adapted for faster timeframe
        """
        df = hourly_data.copy()
        
        # Calculate same indicators but on hourly timeframe
        df = self._calculate_hourly_indicators(df)
        
        # Initialize regime columns
        df['direction_regime'] = 'Sideways'
        df['strength_regime'] = 'Weak'
        df['volatility_regime'] = 'Normal'
        df['character_regime'] = 'Ranging'
        
        # Classify with hourly parameters
        df = self._classify_hourly_direction(df)
        df = self._classify_hourly_strength(df)
        df = self._classify_hourly_volatility(df)
        df = self._classify_hourly_character(df)
        
        # Apply faster smoothing
        df = self._smooth_hourly_regimes(df)
        
        # Create composite regime
        df['composite_regime'] = (
            df['strength_regime'] + '_' + 
            df['direction_regime'] + '_' + 
            df['volatility_regime'] + '_Vol'
        )
        
        return df
    
    def detect_divergences(self, daily_regimes: pd.DataFrame, 
                        hourly_data: pd.DataFrame) -> pd.DataFrame:
        """
        Detect divergences between daily and hourly regimes
        Returns dataframe with divergence signals
        """
        logger.info("Detecting regime divergences...")
        
        # Calculate hourly regimes
        hourly_regimes = self.calculate_hourly_regimes(hourly_data)
        
        # Ensure the index has a name for consistent column creation
        if hourly_regimes.index.name is None:
            hourly_regimes.index.name = 'date'
        
        # Reset index to move the datetime index to a column
        hourly_regimes = hourly_regimes.reset_index()
        
        # Align hourly to daily based on trading session (18:00 ET prior day to 16:00 ET current day)
        def get_trading_session_date(dt):
            if dt.hour >= 18:
                return dt.date() + pd.Timedelta(days=1)
            return dt.date()
        
        # Use the 'date' column (lowercase, as shown in error)
        hourly_regimes['session_date'] = hourly_regimes['date'].apply(get_trading_session_date)
        
        # Ensure daily_regimes index has a name
        if daily_regimes.index.name is None:
            daily_regimes.index.name = 'date'
        
        # Create a copy of daily regimes with date as a regular column
        daily_for_merge = daily_regimes.copy().reset_index()
        daily_for_merge['session_date'] = daily_for_merge['date'].dt.date
        
        # Merge to compare
        merged = hourly_regimes.merge(
            daily_for_merge[['session_date', 'direction_regime', 'strength_regime', 
                            'volatility_regime', 'character_regime', 'composite_regime']],
            on='session_date',
            how='left',
            suffixes=('_hourly', '_daily')
        )
        
        # Set index back to the date column
        merged.set_index('date', inplace=True)
        
        # Calculate divergences
        divergences = pd.DataFrame(index=merged.index)
        
        # Direction divergence
        divergences['direction_divergence'] = (
            merged['direction_regime_hourly'] != merged['direction_regime_daily']
        ).astype(int)
        
        # Strength divergence
        divergences['strength_divergence'] = (
            merged['strength_regime_hourly'] != merged['strength_regime_daily']
        ).astype(int)
        
        # Volatility divergence
        divergences['volatility_divergence'] = (
            merged['volatility_regime_hourly'] != merged['volatility_regime_daily']
        ).astype(int)
        
        # Character divergence
        divergences['character_divergence'] = (
            merged['character_regime_hourly'] != merged['character_regime_daily']
        ).astype(int)
        
        # Calculate divergence persistence (consecutive hours)
        for col in ['direction', 'strength', 'volatility', 'character']:
            div_col = f'{col}_divergence'
            pers_col = f'{col}_divergence_hours'
            
            # Count consecutive diverging hours
            divergences[pers_col] = divergences[div_col].groupby(
                (divergences[div_col] != divergences[div_col].shift()).cumsum()
            ).cumsum() * divergences[div_col]
        
        # Add regime information
        divergences['hourly_regime'] = merged['composite_regime_hourly']
        divergences['daily_regime'] = merged['composite_regime_daily']
        
        # Calculate overall divergence strength
        divergences['divergence_score'] = (
            divergences['direction_divergence'] * 0.4 +  # Direction most important
            divergences['strength_divergence'] * 0.2 +
            divergences['volatility_divergence'] * 0.2 +
            divergences['character_divergence'] * 0.2
        )
        
        return divergences
    
    def generate_warnings(self, divergences: pd.DataFrame, 
                         lookback_hours: int = 24) -> List[Dict]:
        """
        Generate early warning signals based on divergence patterns
        """
        warnings = []
        
        # Get latest data
        latest = divergences.iloc[-lookback_hours:]
        
        # Check each type of divergence
        for div_type in ['direction', 'strength', 'volatility', 'character']:
            div_col = f'{div_type}_divergence'
            pers_col = f'{div_type}_divergence_hours'
            
            # Calculate divergence percentage in lookback window
            div_pct = latest[div_col].mean()
            max_persistence = latest[pers_col].max()
            
            # Determine warning level
            warning_level = None
            if div_pct >= self.warning_levels['critical']:
                warning_level = 'CRITICAL'
            elif div_pct >= self.warning_levels['strong']:
                warning_level = 'STRONG'
            elif div_pct >= self.warning_levels['moderate']:
                warning_level = 'MODERATE'
            elif div_pct >= self.warning_levels['weak']:
                warning_level = 'WEAK'
            
            if warning_level:
                # Get specific regime details
                latest_hourly = latest['hourly_regime'].iloc[-1]
                latest_daily = latest['daily_regime'].iloc[-1]
                
                warning = {
                    'timestamp': latest.index[-1],
                    'type': div_type,
                    'level': warning_level,
                    'divergence_pct': div_pct * 100,
                    'max_persistence_hours': max_persistence,
                    'hourly_regime': latest_hourly,
                    'daily_regime': latest_daily,
                    'message': self._generate_warning_message(
                        div_type, warning_level, div_pct, 
                        latest_hourly, latest_daily, max_persistence
                    )
                }
                warnings.append(warning)
        
        # Check for composite warnings (multiple divergences)
        total_divergence = latest['divergence_score'].mean()
        if total_divergence >= 0.6:  # 60% overall divergence
            warnings.append({
                'timestamp': latest.index[-1],
                'type': 'composite',
                'level': 'CRITICAL',
                'divergence_pct': total_divergence * 100,
                'message': f"CRITICAL: Multiple regime divergences detected! "
                          f"Overall divergence: {total_divergence*100:.1f}%. "
                          f"Daily regime change likely within 1-2 days."
            })
        
        return warnings
    
    def _generate_warning_message(self, div_type: str, level: str, 
                                 div_pct: float, hourly: str, daily: str,
                                 persistence: int) -> str:
        """Generate human-readable warning message"""
        
        messages = {
            'direction': {
                'WEAK': f"Early direction divergence: Hourly showing {hourly.split('_')[1]} while daily remains {daily.split('_')[1]}",
                'MODERATE': f"Growing direction divergence: {div_pct*100:.0f}% of hours conflicting with daily trend",
                'STRONG': f"Strong direction warning: Hourly trend shift persisting {persistence} hours",
                'CRITICAL': f"CRITICAL direction change: Daily regime likely shifting to {hourly.split('_')[1]} soon"
            },
            'strength': {
                'WEAK': f"Trend strength diverging: Hourly {hourly.split('_')[0]} vs daily {daily.split('_')[0]}",
                'MODERATE': f"Trend strength weakening/strengthening: {div_pct*100:.0f}% divergence",
                'STRONG': f"Significant strength change developing over {persistence} hours",
                'CRITICAL': f"Trend strength regime change imminent"
            },
            'volatility': {
                'WEAK': f"Volatility regime shifting on hourly",
                'MODERATE': f"Volatility divergence: {div_pct*100:.0f}% of hours differ from daily",
                'STRONG': f"Major volatility shift detected over {persistence} hours",
                'CRITICAL': f"Volatility regime change imminent - adjust position sizing"
            },
            'character': {
                'WEAK': f"Market character starting to shift",
                'MODERATE': f"Character divergence: Hourly {hourly} vs daily {daily}",
                'STRONG': f"Market behavior changing significantly",
                'CRITICAL': f"Complete character regime change likely"
            }
        }
        
        return messages.get(div_type, {}).get(level, "Divergence detected")
    
    def _calculate_hourly_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate indicators adapted for hourly timeframe"""
        
        # Similar to daily but with faster periods
        # Price vs moving averages (8h and 40h instead of 50d and 200d)
        if 'close' in df.columns:
            df['SMA_8'] = df['close'].rolling(8).mean()  # 8 hours
            df['SMA_40'] = df['close'].rolling(40).mean()  # 40 hours ~ 1 week
            df['price_vs_sma8'] = (df['close'] - df['SMA_8']) / df['SMA_8']
            df['price_vs_sma40'] = (df['close'] - df['SMA_40']) / df['SMA_40']
            df['sma8_vs_sma40'] = (df['SMA_8'] - df['SMA_40']) / df['SMA_40']
        
        # Trend slopes (faster)
        df['trend_slope_4'] = (df['close'] - df['close'].shift(4)) / df['close'].shift(4)
        df['trend_slope_12'] = (df['close'] - df['close'].shift(12)) / df['close'].shift(12)
        
        # ADX for hourly (if available in indicators)
        # This would come from calculate_all_indicators()
        
        # Efficiency ratio (10 hour period)
        df['efficiency_ratio'] = self._calculate_efficiency_ratio(df['close'], 10)
        
        # Realized volatility (hourly)
        df['realized_vol'] = df['close'].pct_change().rolling(24).std() * np.sqrt(252 * 6.5) * 100
        
        return df
    
    def _calculate_efficiency_ratio(self, price_series: pd.Series, period: int) -> pd.Series:
        """Calculate Kaufman's Efficiency Ratio for hourly data"""
        net_change = abs(price_series - price_series.shift(period))
        total_change = price_series.diff().abs().rolling(period).sum()
        efficiency_ratio = net_change / total_change
        return efficiency_ratio.fillna(0.5)
    
    # Classification methods adapted for hourly...
    def _classify_hourly_direction(self, df: pd.DataFrame) -> pd.DataFrame:
        """Classify direction with hourly thresholds"""
        
        direction_score = 0
        signal_count = 0
        
        if 'price_vs_sma8' in df.columns:
            direction_score += np.sign(df['price_vs_sma8']) * 0.3
            signal_count += 1
        
        if 'price_vs_sma40' in df.columns:
            direction_score += np.sign(df['price_vs_sma40']) * 0.4
            signal_count += 1
            
        if 'trend_slope_4' in df.columns:
            direction_score += np.tanh(df['trend_slope_4'] * 20) * 0.3
            signal_count += 1
        
        if signal_count > 0:
            df['direction_score'] = direction_score
        
        # Classify with hourly thresholds
        df.loc[df['direction_score'] > self.hourly_thresholds['direction_strong'], 'direction_regime'] = 'Uptrend'
        df.loc[df['direction_score'] < -self.hourly_thresholds['direction_strong'], 'direction_regime'] = 'Downtrend'
        df.loc[abs(df['direction_score']) < self.hourly_thresholds['direction_neutral'], 'direction_regime'] = 'Sideways'
        
        return df
    
    def _classify_hourly_strength(self, df: pd.DataFrame) -> pd.DataFrame:
        """Simplified strength classification for hourly"""
        
        # Use trend consistency over shorter window
        df['trend_consistency'] = df['close'].pct_change().rolling(12).apply(
            lambda x: (x > 0).sum() / len(x) if len(x) > 0 else 0.5
        )
        
        strength_score = abs(df['trend_consistency'] - 0.5) * 2
        df['strength_score'] = strength_score
        
        df.loc[df['strength_score'] > self.hourly_thresholds['strength_strong'], 'strength_regime'] = 'Strong'
        df.loc[
            (df['strength_score'] > self.hourly_thresholds['strength_moderate']) & 
            (df['strength_score'] <= self.hourly_thresholds['strength_strong']), 
            'strength_regime'
        ] = 'Moderate'
        df.loc[df['strength_score'] <= self.hourly_thresholds['strength_moderate'], 'strength_regime'] = 'Weak'
        
        return df
    
    def _classify_hourly_volatility(self, df: pd.DataFrame) -> pd.DataFrame:
        """Classify volatility for hourly data"""
        
        if 'realized_vol' in df.columns:
            # Use 7-day rolling window for percentile
            df['vol_percentile'] = df['realized_vol'].rolling(
                24 * 7, min_periods=24
            ).rank(pct=True) * 100
            
            df['volatility_score'] = df['vol_percentile'] / 100
            
            df.loc[df['vol_percentile'] < self.hourly_thresholds['vol_low'], 'volatility_regime'] = 'Low'
            df.loc[
                (df['vol_percentile'] >= self.hourly_thresholds['vol_low']) & 
                (df['vol_percentile'] < self.hourly_thresholds['vol_normal']), 
                'volatility_regime'
            ] = 'Normal'
            df.loc[
                (df['vol_percentile'] >= self.hourly_thresholds['vol_normal']) & 
                (df['vol_percentile'] < self.hourly_thresholds['vol_high']), 
                'volatility_regime'
            ] = 'High'
            df.loc[df['vol_percentile'] >= self.hourly_thresholds['vol_high'], 'volatility_regime'] = 'Extreme'
        
        return df
    
    def _classify_hourly_character(self, df: pd.DataFrame) -> pd.DataFrame:
        """Classify market character for hourly"""
        
        if 'efficiency_ratio' in df.columns:
            df['character_score'] = df['efficiency_ratio']
            
            trending_mask = (
                (df['efficiency_ratio'] > self.hourly_thresholds['efficiency_trending']) & 
                (df['direction_regime'] != 'Sideways')
            )
            df.loc[trending_mask, 'character_regime'] = 'Trending'
            
            ranging_mask = df['efficiency_ratio'] < self.hourly_thresholds['efficiency_ranging']
            df.loc[ranging_mask, 'character_regime'] = 'Ranging'
            
            volatile_mask = df['volatility_regime'].isin(['High', 'Extreme'])
            df.loc[volatile_mask, 'character_regime'] = 'Volatile'
        
        return df
    
    def _smooth_hourly_regimes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply smoothing appropriate for hourly data"""
        
        smooth_window = self.hourly_thresholds['smoothing_hours']
        
        # Define regime mappings for smoothing
        regime_maps = {
            'direction_regime': {'Uptrend': 0, 'Downtrend': 1, 'Sideways': 2},
            'strength_regime': {'Strong': 0, 'Moderate': 1, 'Weak': 2},
            'volatility_regime': {'Low': 0, 'Normal': 1, 'High': 2, 'Extreme': 3},
            'character_regime': {'Trending': 0, 'Ranging': 1, 'Volatile': 2}
        }
        
        for col, mapping in regime_maps.items():
            if col in df.columns:
                # Map to numeric
                df[f'{col}_num'] = df[col].map(mapping)
                
                # Apply rolling mode
                df[f'{col}_smooth'] = df[f'{col}_num'].rolling(
                    window=smooth_window,
                    min_periods=1
                ).apply(lambda x: pd.Series(x).mode()[0] if len(pd.Series(x).mode()) > 0 else x.iloc[-1])
                
                # Map back to labels
                inv_mapping = {v: k for k, v in mapping.items()}
                df[col] = df[f'{col}_smooth'].map(inv_mapping)
                
                # Clean up
                df.drop([f'{col}_num', f'{col}_smooth'], axis=1, inplace=True)
        
        return df

