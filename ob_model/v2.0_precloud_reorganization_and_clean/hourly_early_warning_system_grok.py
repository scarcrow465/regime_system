#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python
# coding: utf-8

"""
Lower-Timeframe Early Warning System for Daily Regime Changes
Detects when LTF regimes diverge from daily, signaling potential transitions
Supports resampling 1H data to 4H, 8H, 12H, etc.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class RegimeDivergence:
    """Represents a divergence between LTF and daily regimes"""
    timestamp: pd.Timestamp
    daily_regime: str
    ltf_regime: str
    divergence_type: str  # 'direction', 'strength', 'volatility', 'character'
    divergence_strength: float  # 0-1 score
    periods_persisted: int
    confidence: float


class LowerTimeframeEarlyWarningSystem:
    """
    Detects early warning signals from LTF regime divergences
    Optimized for NQ futures; supports multiple timeframes via resampling
    """
    
    def __init__(
        self, 
        daily_classifier, 
        timeframe: str = '1H',  # e.g., '1H', '4H', '8H', '12H'
        lookback_periods: int = 168,  # Periods in LTF (e.g., 168 for 1H ~7 days)
        config: Optional[Dict] = None
    ):
        """
        Initialize early warning system
        
        Args:
            daily_classifier: Instance of NQDailyRegimeClassifier
            timeframe: LTF timeframe (e.g., '4H')
            lookback_periods: Periods of history for LTF analysis
            config: Optional dict to override thresholds/weights
        """
        self.daily_classifier = daily_classifier
        self.timeframe = timeframe
        self.lookback_periods = lookback_periods
        self.multiplier = int(timeframe[:-1]) if timeframe != '1H' else 1  # Hours per period
        
        # Default config (scalable by multiplier where needed)
        self.config = {
            'thresholds': {
                'direction_strong': 0.2,
                'direction_neutral': 0.1,
                'strength_strong': 0.35,
                'strength_moderate': 0.2,
                'vol_low': 20,
                'vol_normal': 70,
                'vol_high': 85,
                'efficiency_trending': 0.2,
                'efficiency_ranging': 0.12,
                'smoothing_periods': 6 * self.multiplier,  # Scale smoothing
                'min_divergence_periods': 4 * self.multiplier,
            },
            'warning_levels': {
                'weak': 0.3,
                'moderate': 0.5,
                'strong': 0.7,
                'critical': 0.85
            },
            'divergence_weights': {
                'direction': 0.4,
                'strength': 0.2,
                'volatility': 0.2,
                'character': 0.2
            },
            'indicator_periods': {
                'sma_short': 8 * self.multiplier,  # Scale periods
                'sma_long': 40 * self.multiplier,
                'trend_slope_short': 4 * self.multiplier,
                'trend_slope_long': 12 * self.multiplier,
                'efficiency_period': 10 * self.multiplier,
                'vol_window': 24 * self.multiplier,
            }
        }
        if config:
            self._update_config(self.config, config)
        
        # Stateful storage for walk-forward
        self.ltf_data: pd.DataFrame = pd.DataFrame()
        self.daily_regimes: pd.DataFrame = pd.DataFrame()
        self.divergence_history: List[RegimeDivergence] = []
    
    def _update_config(self, base: Dict, updates: Dict):
        """Recursive config update"""
        for key, value in updates.items():
            if isinstance(value, dict) and key in base:
                self._update_config(base[key], value)
            else:
                base[key] = value
    
    def resample_data(self, data: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """Resample 1H data to higher timeframe"""
        if timeframe == '1H':
            return data
        try:
            resampled = data.resample(timeframe).agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum' if 'volume' in data else None
            }).dropna()
            logger.info(f"Resampled to {timeframe}: {len(resampled)} bars")
            return resampled
        except Exception as e:
            logger.error(f"Resampling error: {e}")
            raise
    
    def calculate_ltf_regimes(self, ltf_data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate regimes on LTF data with sensitive parameters
        Handles resampling if needed
        """
        df = self.resample_data(ltf_data, self.timeframe).copy()
        
        # Calculate indicators
        df = self._calculate_ltf_indicators(df)
        
        # Initialize regime columns
        df['direction_regime'] = 'Sideways'
        df['strength_regime'] = 'Weak'
        df['volatility_regime'] = 'Normal'
        df['character_regime'] = 'Ranging'
        
        # Classify
        df = self._classify_ltf_direction(df)
        df = self._classify_ltf_strength(df)
        df = self._classify_ltf_volatility(df)
        df = self._classify_ltf_character(df)
        
        # Smooth
        df = self._smooth_ltf_regimes(df)
        
        # Composite
        df['composite_regime'] = (
            df['strength_regime'] + '_' + 
            df['direction_regime'] + '_' + 
            df['volatility_regime'] + '_Vol'
        )
        
        return df
    
    def detect_divergences(self, daily_regimes: pd.DataFrame, 
                           ltf_data: pd.DataFrame) -> pd.DataFrame:
        """
        Detect divergences between daily and LTF regimes
        """
        logger.info("Detecting regime divergences...")
        
        ltf_regimes = self.calculate_ltf_regimes(ltf_data)
        
        if ltf_regimes.index.name is None:
            ltf_regimes.index.name = 'date'
        
        ltf_regimes = ltf_regimes.reset_index()
        
        def get_trading_session_date(dt):
            if dt.hour >= 18:
                return dt.date() + pd.Timedelta(days=1)
            return dt.date()
        
        ltf_regimes['session_date'] = ltf_regimes['date'].apply(get_trading_session_date)
        
        if daily_regimes.index.name is None:
            daily_regimes.index.name = 'date'
        
        daily_for_merge = daily_regimes.copy().reset_index()
        daily_for_merge['session_date'] = daily_for_merge['date'].dt.date
        
        unmatched = ltf_regimes[~ltf_regimes['session_date'].isin(daily_for_merge['session_date'])]['session_date'].unique()
        if len(unmatched) > 0:
            logger.warning(f"Unmatched session dates: {unmatched}. Forward-filling daily regimes.")
            # Forward-fill daily for missing
            daily_for_merge = daily_for_merge.set_index('session_date').reindex(
                pd.date_range(daily_for_merge['session_date'].min(), daily_for_merge['session_date'].max())
            ).ffill().reset_index().rename(columns={'index': 'session_date'})
        
        merged = ltf_regimes.merge(
            daily_for_merge[['session_date', 'direction_regime', 'strength_regime', 
                             'volatility_regime', 'character_regime', 'composite_regime']],
            on='session_date',
            how='left',
            suffixes=('_ltf', '_daily')
        ).set_index('date')
        
        # Handle NaNs
        merged = merged.fillna('Unknown')
        
        divergences = pd.DataFrame(index=merged.index)
        
        for aspect in ['direction', 'strength', 'volatility', 'character']:
            divergences[f'{aspect}_divergence'] = (
                merged[f'{aspect}_regime_ltf'] != merged[f'{aspect}_regime_daily']
            ).astype(int)
            
            div_col = f'{aspect}_divergence'
            pers_col = f'{aspect}_divergence_periods'
            divergences[pers_col] = divergences[div_col].groupby(
                (divergences[div_col] != divergences[div_col].shift()).cumsum()
            ).cumsum() * divergences[div_col]
        
        divergences['ltf_regime'] = merged['composite_regime_ltf']
        divergences['daily_regime'] = merged['composite_regime_daily']
        
        weights = self.config['divergence_weights']
        divergences['divergence_score'] = (
            divergences['direction_divergence'] * weights['direction'] +
            divergences['strength_divergence'] * weights['strength'] +
            divergences['volatility_divergence'] * weights['volatility'] +
            divergences['character_divergence'] * weights['character']
        )
        
        return divergences
    
    def generate_warnings(self, divergences: pd.DataFrame, 
                          lookback_periods: int = 24) -> List[Dict]:
        """
        Generate early warning signals
        """
        warnings = []
        latest = divergences.iloc[-lookback_periods:]
        
        for div_type in ['direction', 'strength', 'volatility', 'character']:
            div_col = f'{div_type}_divergence'
            pers_col = f'{div_type}_divergence_periods'
            
            div_pct = latest[div_col].mean()
            max_persistence = latest[pers_col].max()
            
            levels = self.config['warning_levels']
            warning_level = None
            if div_pct >= levels['critical']:
                warning_level = 'CRITICAL'
            elif div_pct >= levels['strong']:
                warning_level = 'STRONG'
            elif div_pct >= levels['moderate']:
                warning_level = 'MODERATE'
            elif div_pct >= levels['weak']:
                warning_level = 'WEAK'
            
            if warning_level:
                latest_ltf = latest['ltf_regime'].iloc[-1]
                latest_daily = latest['daily_regime'].iloc[-1]
                
                warning = {
                    'timestamp': latest.index[-1],
                    'type': div_type,
                    'level': warning_level,
                    'divergence_pct': div_pct * 100,
                    'max_persistence_periods': max_persistence,
                    'ltf_regime': latest_ltf,
                    'daily_regime': latest_daily,
                    'message': self._generate_warning_message(
                        div_type, warning_level, div_pct, 
                        latest_ltf, latest_daily, max_persistence
                    )
                }
                warnings.append(warning)
        
        total_divergence = latest['divergence_score'].mean()
        if total_divergence >= 0.6:
            warnings.append({
                'timestamp': latest.index[-1],
                'type': 'composite',
                'level': 'CRITICAL',
                'divergence_pct': total_divergence * 100,
                'message': f"CRITICAL: Multiple divergences! Overall: {total_divergence*100:.1f}%. Change likely."
            })
        
        return warnings
    
    def _generate_warning_message(self, div_type: str, level: str, 
                                  div_pct: float, ltf: str, daily: str,
                                  persistence: int) -> str:
        if pd.isna(daily) or not isinstance(daily, str):
            daily = "Unknown"
        if pd.isna(ltf) or not isinstance(ltf, str):
            ltf = "Unknown"
        
        # Similar to original, abbreviated for space
        messages = {
            'direction': { 'WEAK': f"Early direction div: LTF {ltf.split('_')[1]} vs daily {daily.split('_')[1]}", ... },  # Fill as per original
            # (Omit full dict for brevity; copy from your original and adjust)
        }
        return messages.get(div_type, {}).get(level, "Divergence detected")
    
    def _calculate_ltf_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        periods = self.config['indicator_periods']
        
        if 'close' in df.columns:
            df['SMA_short'] = df['close'].rolling(periods['sma_short']).mean()
            df['SMA_long'] = df['close'].rolling(periods['sma_long']).mean()
            df['price_vs_sma_short'] = (df['close'] - df['SMA_short']) / df['SMA_short']
            df['price_vs_sma_long'] = (df['close'] - df['SMA_long']) / df['SMA_long']
            df['sma_short_vs_long'] = (df['SMA_short'] - df['SMA_long']) / df['SMA_long']
        
        df['trend_slope_short'] = (df['close'] - df['close'].shift(periods['trend_slope_short'])) / df['close'].shift(periods['trend_slope_short'])
        df['trend_slope_long'] = (df['close'] - df['close'].shift(periods['trend_slope_long'])) / df['close'].shift(periods['trend_slope_long'])
        
        df['efficiency_ratio'] = self._calculate_efficiency_ratio(df['close'], periods['efficiency_period'])
        
        df['realized_vol'] = df['close'].pct_change().rolling(periods['vol_window']).std() * np.sqrt(252 * 6.5) * 100
        
        return df
    
    def _calculate_efficiency_ratio(self, price_series: pd.Series, period: int) -> pd.Series:
        net_change = abs(price_series - price_series.shift(period))
        total_change = price_series.diff().abs().rolling(period).sum()
        return (net_change / total_change).fillna(0.5)
    
    # _classify_ltf_* methods: Similar to original, but use self.config['thresholds']
    def _classify_ltf_direction(self, df: pd.DataFrame) -> pd.DataFrame:
        thresh = self.config['thresholds']
        direction_score = 0
        signal_count = 0
        
        if 'price_vs_sma_short' in df.columns:
            direction_score += np.sign(df['price_vs_sma_short']) * 0.3
            signal_count += 1
        
        if 'price_vs_sma_long' in df.columns:
            direction_score += np.sign(df['price_vs_sma_long']) * 0.4
            signal_count += 1
            
        if 'trend_slope_short' in df.columns:
            direction_score += np.tanh(df['trend_slope_short'] * 20) * 0.3
            signal_count += 1
        
        if signal_count > 0:
            df['direction_score'] = direction_score
        
        df.loc[df['direction_score'] > thresh['direction_strong'], 'direction_regime'] = 'Uptrend'
        df.loc[df['direction_score'] < -thresh['direction_strong'], 'direction_regime'] = 'Downtrend'
        df.loc[abs(df['direction_score']) < thresh['direction_neutral'], 'direction_regime'] = 'Sideways'
        
        return df
    
    # (Omit other _classify_* for brevity; update similarly with self.config['thresholds'])
    
    def _smooth_ltf_regimes(self, df: pd.DataFrame) -> pd.DataFrame:
        smooth_window = self.config['thresholds']['smoothing_periods']
        
        regime_maps = {
            'direction_regime': {'Uptrend': 0, 'Downtrend': 1, 'Sideways': 2},
            # ... (same as original)
        }
        
        for col, mapping in regime_maps.items():
            if col in df.columns:
                df[f'{col}_num'] = df[col].map(mapping)
                df[f'{col}_smooth'] = df[f'{col}_num'].rolling(
                    window=smooth_window, min_periods=1, closed='left'  # Avoid look-ahead
                ).apply(lambda x: pd.Series(x).mode()[0] if len(pd.Series(x).mode()) > 0 else x.iloc[-1])
                inv_mapping = {v: k for k, v in mapping.items()}
                df[col] = df[f'{col}_smooth'].map(inv_mapping)
                df.drop([f'{col}_num', f'{col}_smooth'], axis=1, inplace=True)
        
        return df
    
    def update(self, new_ltf_bar: pd.Series, new_daily_bar: Optional[pd.Series] = None) -> tuple:
        """Incremental update for walk-forward"""
        self.ltf_data = pd.concat([self.ltf_data, new_ltf_bar.to_frame().T])
        
        recent_ltf = self.ltf_data.iloc[-self.lookback_periods:]
        ltf_regimes = self.calculate_ltf_regimes(recent_ltf)
        
        if new_daily_bar is not None:
            self.daily_regimes = pd.concat([self.daily_regimes, new_daily_bar.to_frame().T])
        
        divergences = self.detect_divergences(self.daily_regimes, recent_ltf)
        warnings = self.generate_warnings(divergences)
        
        return warnings, divergences

