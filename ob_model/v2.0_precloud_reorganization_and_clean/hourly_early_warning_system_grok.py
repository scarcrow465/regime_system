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
        self.multiplier = int(timeframe[:-1]) if timeframe != '1H' else 1  # Hours per period
        self.lookback_periods = lookback_periods * self.multiplier
        
        # Default config (scalable by multiplier where needed)
        self.config = {
            'thresholds': {
                'direction_strong': 0.25,
                'direction_neutral': 0.1,
                'strength_strong': 0.35,
                'strength_moderate': 0.2,
                'vol_low': 20,
                'vol_normal': 70,
                'vol_high': 85,
                'efficiency_trending': 0.2,
                'efficiency_ranging': 0.12,
                'smoothing_periods': 6 * self.multiplier,  # Scale smoothing
                'min_divergence_periods': 8 * self.multiplier,
            },
            'warning_levels': {
                'weak': 0.3,
                'moderate': 0.55,
                'strong': 0.7,
                'critical': 0.85
            },
            'divergence_weights': {
                'direction': 0.6,  # High for trend changes
                'strength': 0.3,  # Medium for mean reversion signals
                'volatility': 0.05,  # Low, as vol varies naturally
                'character': 0.05   # Low, as ranging is LTF noise
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
        
        # Calculate LTF regimes
        ltf_regimes = self.calculate_ltf_regimes(ltf_data)
        
        if ltf_regimes.index.name is None:
            ltf_regimes.index.name = 'date'
        
        ltf_regimes = ltf_regimes.reset_index()
        
        def get_trading_session_date(dt):
            if dt.hour >= 18:
                return dt.date() + pd.Timedelta(days=1)
            return dt.date()
        
        ltf_regimes['session_date'] = ltf_regimes['date'].apply(get_trading_session_date)
        
        # Holiday list for U.S./CME (NQ futures) - add more years as needed
        holidays = [
            # 2023
            pd.to_datetime('2023-01-02'),  # New Year's Day observed
            pd.to_datetime('2023-01-16'),  # MLK Day
            pd.to_datetime('2023-02-20'),  # Presidents' Day
            pd.to_datetime('2023-04-07'),  # Good Friday
            pd.to_datetime('2023-05-29'),  # Memorial Day
            pd.to_datetime('2023-06-19'),  # Juneteenth
            pd.to_datetime('2023-07-04'),  # Independence Day
            pd.to_datetime('2023-09-04'),  # Labor Day
            pd.to_datetime('2023-11-23'),  # Thanksgiving
            pd.to_datetime('2023-12-25'),  # Christmas
            # 2024
            pd.to_datetime('2024-01-01'),  # New Year's Day
            pd.to_datetime('2024-01-15'),  # MLK Day
            pd.to_datetime('2024-02-19'),  # Presidents' Day
            pd.to_datetime('2024-03-29'),  # Good Friday
            pd.to_datetime('2024-05-27'),  # Memorial Day
            pd.to_datetime('2024-06-19'),  # Juneteenth
            pd.to_datetime('2024-07-04'),  # Independence Day
            pd.to_datetime('2024-09-02'),  # Labor Day
            pd.to_datetime('2024-11-28'),  # Thanksgiving
            pd.to_datetime('2024-12-25'),  # Christmas
            # 2025 (from CME schedule)
            pd.to_datetime('2025-01-01'),  # New Year's Day
            pd.to_datetime('2025-01-20'),  # MLK Day
            pd.to_datetime('2025-02-17'),  # Presidents' Day
            pd.to_datetime('2025-04-18'),  # Good Friday
            pd.to_datetime('2025-05-26'),  # Memorial Day
            pd.to_datetime('2025-06-19'),  # Juneteenth
            pd.to_datetime('2025-07-04'),  # Independence Day
            pd.to_datetime('2025-09-01'),  # Labor Day
            pd.to_datetime('2025-11-27'),  # Thanksgiving
            pd.to_datetime('2025-12-25'),  # Christmas
        ]
        
        # Adjust for holidays: If session_date is holiday, shift to next non-holiday
        def adjust_for_holiday(session_date):
            while session_date in holidays:
                session_date += pd.Timedelta(days=1)
            return session_date
        
        ltf_regimes['session_date'] = ltf_regimes['session_date'].apply(adjust_for_holiday)
        
        if daily_regimes.index.name is None:
            daily_regimes.index.name = 'date'
        
        daily_for_merge = daily_regimes.copy().reset_index()
        daily_for_merge['session_date'] = daily_for_merge['date'].dt.date
        daily_for_merge['session_date'] = daily_for_merge['session_date'].apply(adjust_for_holiday)
        
        # Fix type mismatch: Convert both session_date to datetime64[ns]
        daily_for_merge['session_date'] = pd.to_datetime(daily_for_merge['session_date'])
        ltf_regimes['session_date'] = pd.to_datetime(ltf_regimes['session_date'])
        
        unmatched = ltf_regimes[~ltf_regimes['session_date'].isin(daily_for_merge['session_date'])]['session_date'].unique()
        if len(unmatched) > 0:
            logger.debug(f"Unmatched session dates: {unmatched}. Forward-filling daily regimes.")
            # Remove duplicates before reindex to avoid error
            daily_for_merge = daily_for_merge.drop_duplicates(subset='session_date', keep='last')
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
        """Generate human-readable warning message"""
        
        # Handle cases where daily or ltf regime is NaN
        if not isinstance(daily, str) or pd.isna(daily):
            daily = "Unknown"
        if not isinstance(ltf, str) or pd.isna(ltf):
            ltf = "Unknown"
        
        messages = {
            'direction': {
                'WEAK': f"Early direction divergence: LTF showing {ltf.split('_')[1] if '_' in ltf else ltf} while daily remains {daily.split('_')[1] if '_' in daily else daily}",
                'MODERATE': f"Growing direction divergence: {div_pct*100:.0f}% of periods conflicting with daily trend",
                'STRONG': f"Strong direction warning: LTF trend shift persisting {persistence} periods",
                'CRITICAL': f"CRITICAL direction change: Daily regime likely shifting to {ltf.split('_')[1] if '_' in ltf else ltf} soon"
            },
            'strength': {
                'WEAK': f"Trend strength diverging: LTF {ltf.split('_')[0] if '_' in ltf else ltf} vs daily {daily.split('_')[0] if '_' in daily else daily}",
                'MODERATE': f"Trend strength weakening/strengthening: {div_pct*100:.0f}% divergence",
                'STRONG': f"Significant strength change developing over {persistence} periods",
                'CRITICAL': f"Trend strength regime change imminent"
            },
            'volatility': {
                'WEAK': f"Volatility regime shifting on LTF",
                'MODERATE': f"Volatility divergence: {div_pct*100:.0f}% of periods differ from daily",
                'STRONG': f"Major volatility shift detected over {persistence} periods",
                'CRITICAL': f"Volatility regime change imminent - adjust position sizing"
            },
            'character': {
                'WEAK': f"Market character starting to shift",
                'MODERATE': f"Character divergence: LTF {ltf} vs daily {daily}",
                'STRONG': f"Market behavior changing significantly",
                'CRITICAL': f"Complete character regime change likely"
            }
        }
        
        return messages.get(div_type, {}).get(level, "Divergence detected")
    
    def _calculate_ltf_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        periods = self.config['indicator_periods']
        
        if 'close' in df.columns:
            df['SMA_short'] = df['close'].rolling(periods['sma_short'], min_periods=1).mean()
            df['SMA_long'] = df['close'].rolling(periods['sma_long'], min_periods=1).mean()
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
        total_change = price_series.diff().abs().rolling(period, min_periods=1).sum()
        return (net_change / total_change).fillna(0.5)
    
    # _classify_ltf_* methods: Similar to original, but use self.config['thresholds']
    def _classify_ltf_direction(self, df: pd.DataFrame) -> pd.DataFrame:
        thresh = self.config['thresholds']
        df['price_vs_sma_short'] = pd.to_numeric(df['price_vs_sma_short'], errors='coerce').fillna(0)
        df['price_vs_sma_long'] = pd.to_numeric(df['price_vs_sma_long'], errors='coerce').fillna(0)
        df['trend_slope_short'] = pd.to_numeric(df['trend_slope_short'], errors='coerce').fillna(0)
        df['trend_slope_long'] = pd.to_numeric(df['trend_slope_long'], errors='coerce').fillna(0)
        df['efficiency_ratio'] = pd.to_numeric(df['efficiency_ratio'], errors='coerce').fillna(0)
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
    
    def _classify_ltf_strength(self, df: pd.DataFrame) -> pd.DataFrame:
        """Classify strength with LTF thresholds"""
        
        thresh = self.config['thresholds']
        
        # Use trend consistency over shorter window
        df['trend_consistency'] = df['close'].pct_change().rolling(12 * self.multiplier, min_periods=1).apply(
            lambda x: (x > 0).sum() / len(x) if len(x) > 0 else 0.5
        )
        
        strength_score = abs(df['trend_consistency'] - 0.5) * 2
        df['strength_score'] = strength_score
        
        df.loc[df['strength_score'] > thresh['strength_strong'], 'strength_regime'] = 'Strong'
        df.loc[
            (df['strength_score'] > thresh['strength_moderate']) & 
            (df['strength_score'] <= thresh['strength_strong']), 
            'strength_regime'
        ] = 'Moderate'
        df.loc[df['strength_score'] <= thresh['strength_moderate'], 'strength_regime'] = 'Weak'
        
        return df

    def _classify_ltf_volatility(self, df: pd.DataFrame) -> pd.DataFrame:
        """Classify volatility for LTF data"""
        
        thresh = self.config['thresholds']
        
        if 'realized_vol' in df.columns:
            # Use rolling window scaled by multiplier for percentile
            df['vol_percentile'] = df['realized_vol'].rolling(
                24 * 7 * self.multiplier, min_periods=1
            ).rank(pct=True) * 100
            
            df['volatility_score'] = df['vol_percentile'] / 100
            
            df.loc[df['vol_percentile'] < thresh['vol_low'], 'volatility_regime'] = 'Low'
            df.loc[
                (df['vol_percentile'] >= thresh['vol_low']) & 
                (df['vol_percentile'] < thresh['vol_normal']), 
                'volatility_regime'
            ] = 'Normal'
            df.loc[
                (df['vol_percentile'] >= thresh['vol_normal']) & 
                (df['vol_percentile'] < thresh['vol_high']), 
                'volatility_regime'
            ] = 'High'
            df.loc[df['vol_percentile'] >= thresh['vol_high'], 'volatility_regime'] = 'Extreme'
        
        return df

    def _classify_ltf_character(self, df: pd.DataFrame) -> pd.DataFrame:
        """Classify market character for LTF"""
        
        thresh = self.config['thresholds']
        
        if 'efficiency_ratio' in df.columns:
            df['character_score'] = df['efficiency_ratio']
            
            trending_mask = (
                (df['efficiency_ratio'] > thresh['efficiency_trending']) & 
                (df['direction_regime'] != 'Sideways')
            )
            df.loc[trending_mask, 'character_regime'] = 'Trending'
            
            ranging_mask = df['efficiency_ratio'] < thresh['efficiency_ranging']
            df.loc[ranging_mask, 'character_regime'] = 'Ranging'
            
            volatile_mask = df['volatility_regime'].isin(['High', 'Extreme'])
            df.loc[volatile_mask, 'character_regime'] = 'Volatile'
        
        return df

    def _smooth_ltf_regimes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply smoothing appropriate for LTF data"""
        
        smooth_window = self.config['thresholds']['smoothing_periods']
        
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
                
                # Apply rolling mode (avoid look-ahead with closed='left')
                df[f'{col}_smooth'] = df[f'{col}_num'].rolling(
                    window=smooth_window,
                    min_periods=1,
                    closed='left'
                ).apply(lambda x: pd.Series(x).mode()[0] if len(pd.Series(x).mode()) > 0 else x.iloc[-1])
                
                # Map back to labels
                inv_mapping = {v: k for k, v in mapping.items()}
                df[col] = df[f'{col}_smooth'].map(inv_mapping)
                
                # Clean up
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

