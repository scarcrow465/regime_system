#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
NQ Daily Regime Classifier
Foundation of the hierarchical regime system
Optimized for NQ daily data characteristics
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, List
from dataclasses import dataclass
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DailyRegime:
    """Represents the current daily regime state"""
    timestamp: pd.Timestamp
    
    # Primary classifications
    direction: str  # 'Uptrend', 'Downtrend', 'Sideways'
    strength: str   # 'Strong', 'Moderate', 'Weak'
    volatility: str # 'Low', 'Normal', 'High', 'Extreme'
    character: str  # 'Trending', 'Ranging', 'Volatile', 'Transitioning'
    
    # Confidence scores (0-1)
    direction_confidence: float
    strength_confidence: float
    volatility_confidence: float
    character_confidence: float
    
    # Composite regime
    composite_regime: str  # e.g., "Strong_Uptrend_Low_Vol"
    regime_age: int  # Days in current regime
    
    # Supporting metrics
    trend_score: float
    momentum_score: float
    volatility_percentile: float
    efficiency_ratio: float


class NQDailyRegimeClassifier:
    """
    Daily regime classifier optimized for NQ futures
    Based on market characterization showing strong trending behavior
    """
    
    def __init__(self, lookback_days: int = 252):
        """
        Initialize classifier with NQ-specific parameters
        
        Args:
            lookback_days: Days for rolling calculations (default 252 = 1 year)
        """
        self.lookback_days = lookback_days

        self.thresholds = {
            # Direction - aim for ~10-15% sideways
            'direction_strong': 0.25,      # Middle ground
            'direction_neutral': 0.1,      # Wider sideways band
            
            # Strength - restore strong trend detection
            'strength_strong': 0.45,       # Lower than 0.6 but higher than 0.325
            'strength_moderate': 0.25,     
            
            # Volatility - keep as is
            'vol_low': 25,
            'vol_normal': 75,
            'vol_high': 90,
            
            # Character - fix ranging dominance
            'efficiency_trending': 0.25,   # Lower than 0.5 but higher than 0.15
            'efficiency_ranging': 0.15,    
            
            # Smoothing
            'min_regime_days': 3,
            'smoothing_days': 5,
        }
        
        # Define regime mappings
        self.regime_maps = {
            'direction_regime': {'Uptrend': 0, 'Downtrend': 1, 'Sideways': 2},
            'strength_regime': {'Strong': 0, 'Moderate': 1, 'Weak': 2},
            'volatility_regime': {'Low': 0, 'Normal': 1, 'High': 2, 'Extreme': 3},
            'character_regime': {'Trending': 0, 'Ranging': 1, 'Volatile': 2, 'Transitioning': 3}
        }
        
        # Store regime history
        self.regime_history = []
        self.current_regime = None
        self.regime_start_date = None
        
    def calculate_regime_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all indicators needed for regime classification
        Assumes data already has the 87 indicators from calculate_all_indicators()
        """
        df = data.copy()
        
        # 1. Direction indicators (composite of multiple signals)
        direction_signals = []
        
        # Price vs moving averages
        if 'SMA_50' in df.columns and 'SMA_200' in df.columns:
            df['price_vs_sma50'] = (df['close'] - df['SMA_50']) / df['SMA_50']
            df['price_vs_sma200'] = (df['close'] - df['SMA_200']) / df['SMA_200']
            df['sma50_vs_sma200'] = (df['SMA_50'] - df['SMA_200']) / df['SMA_200']
            direction_signals.extend(['price_vs_sma50', 'price_vs_sma200', 'sma50_vs_sma200'])
        
        # Trend slope
        df['trend_slope_20'] = (df['close'] - df['close'].shift(20)) / df['close'].shift(20)
        df['trend_slope_50'] = (df['close'] - df['close'].shift(50)) / df['close'].shift(50)
        direction_signals.extend(['trend_slope_20', 'trend_slope_50'])
        
        # MACD signal
        if 'MACD' in df.columns and 'MACD_signal' in df.columns:
            df['macd_signal_strength'] = (df['MACD'] - df['MACD_signal']) / df['close'] * 100
            direction_signals.append('macd_signal_strength')
        
        # 2. Strength indicators
        strength_signals = []
        
        # ADX for trend strength
        if 'ADX' in df.columns:
            strength_signals.append('ADX')
        
        # Directional movement
        if 'DI_plus' in df.columns and 'DI_minus' in df.columns:
            df['di_spread'] = df['DI_plus'] - df['DI_minus']
            strength_signals.append('di_spread')
        
        # Trend consistency
        df['trend_consistency'] = df['close'].pct_change().rolling(20).apply(
            lambda x: (x > 0).sum() / len(x) if len(x) > 0 else 0.5
        )
        strength_signals.append('trend_consistency')
        
        # 3. Volatility indicators
        volatility_signals = []
        
        # ATR-based volatility
        if 'ATR' in df.columns:
            df['atr_percent'] = df['ATR'] / df['close'] * 100
            volatility_signals.append('atr_percent')
        
        # Realized volatility
        df['realized_vol'] = df['close'].pct_change().rolling(20).std() * np.sqrt(252) * 100
        volatility_signals.append('realized_vol')
        
        # Bollinger Band width
        if 'BB_Width' in df.columns:
            volatility_signals.append('BB_Width')
        
        # 4. Market character indicators
        
        # Efficiency Ratio (trending vs choppy)
        df['efficiency_ratio'] = self._calculate_efficiency_ratio(df['close'], 20)
        
        # Fractal Dimension (trending vs ranging)
        df['fractal_dimension'] = self._calculate_fractal_dimension(df['close'], 30)
        
        # Volume patterns
        if 'Volume' in df.columns:
            df['volume_ratio'] = df['Volume'] / df['Volume'].rolling(50).mean()
        
        return df
    
    def classify_regimes(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Classify daily regimes for NQ
        Returns dataframe with regime classifications
        """
        logger.info("Starting NQ daily regime classification...")
        
        # Calculate regime indicators
        df = self.calculate_regime_indicators(data)
        
        # Initialize regime columns
        df['direction_regime'] = 'Sideways'
        df['strength_regime'] = 'Weak'
        df['volatility_regime'] = 'Normal'
        df['character_regime'] = 'Ranging'
        
        # Classification scores
        df['direction_score'] = 0.0
        df['strength_score'] = 0.0
        df['volatility_score'] = 0.0
        df['character_score'] = 0.0
        
        # 1. Classify Direction
        df = self._classify_direction(df)
        
        # 2. Classify Strength
        df = self._classify_strength(df)
        
        # 3. Classify Volatility
        df = self._classify_volatility(df)
        
        # 4. Classify Character
        df = self._classify_character(df)
        
        # 5. Apply regime smoothing
        df = self._smooth_regimes(df)
        
        # 6. Create composite regime
        df['composite_regime'] = (
            df['strength_regime'] + '_' + 
            df['direction_regime'] + '_' + 
            df['volatility_regime'] + '_Vol'
        )
        
        # 7. Calculate regime age
        df['regime_age'] = self._calculate_regime_age(df['composite_regime'])
        
        # 8. Calculate confidence scores
        df = self._calculate_confidence_scores(df)
        
        logger.info(f"Regime classification complete for {len(df)} days")

        self.print_threshold_diagnostics(df)
        
        return df
    
    def _classify_direction(self, df: pd.DataFrame) -> pd.DataFrame:
        """Classify market direction with NQ-specific logic"""
        
        # Composite direction score (-1 to 1)
        direction_score = 0
        signal_count = 0
        
        # Price vs moving averages
        if 'price_vs_sma50' in df.columns:
            direction_score += np.sign(df['price_vs_sma50']) * 0.2
            signal_count += 1
        
        if 'price_vs_sma200' in df.columns:
            direction_score += np.sign(df['price_vs_sma200']) * 0.3
            signal_count += 1
            
        if 'sma50_vs_sma200' in df.columns:
            direction_score += np.sign(df['sma50_vs_sma200']) * 0.3
            signal_count += 1
        
        # Trend slopes
        if 'trend_slope_20' in df.columns:
            direction_score += np.tanh(df['trend_slope_20'] * 10) * 0.1
            signal_count += 1
            
        if 'trend_slope_50' in df.columns:
            direction_score += np.tanh(df['trend_slope_50'] * 5) * 0.1
            signal_count += 1
        
        # Normalize score
        if signal_count > 0:
            df['direction_score'] = direction_score
        
        # Classify based on score
        df.loc[df['direction_score'] > self.thresholds['direction_strong'], 'direction_regime'] = 'Uptrend'
        df.loc[df['direction_score'] < -self.thresholds['direction_strong'], 'direction_regime'] = 'Downtrend'
        df.loc[abs(df['direction_score']) < self.thresholds['direction_neutral'], 'direction_regime'] = 'Sideways'
        
        return df
    
    def _classify_strength(self, df: pd.DataFrame) -> pd.DataFrame:
        """Classify trend strength"""
        
        strength_score = 0
        signal_count = 0
        
        # ADX-based strength
        if 'ADX' in df.columns:
            df['adx_normalized'] = df['ADX'] / 50  # Normalize to 0-1
            strength_score += df['adx_normalized'].clip(0, 1) * 0.4
            signal_count += 1
        
        # Directional movement spread
        if 'di_spread' in df.columns:
            df['di_strength'] = abs(df['di_spread']) / 50
            strength_score += df['di_strength'].clip(0, 1) * 0.3
            signal_count += 1
        
        # Trend consistency
        if 'trend_consistency' in df.columns:
            consistency_strength = abs(df['trend_consistency'] - 0.5) * 2
            strength_score += consistency_strength * 0.3
            signal_count += 1
        
        # Normalize and classify
        if signal_count > 0:
            df['strength_score'] = strength_score
            
        df.loc[df['strength_score'] > self.thresholds['strength_strong'], 'strength_regime'] = 'Strong'
        df.loc[(df['strength_score'] > self.thresholds['strength_moderate']) & (df['strength_score'] <= self.thresholds['strength_strong']), 'strength_regime'] = 'Moderate'
        df.loc[df['strength_score'] <= self.thresholds['strength_moderate'], 'strength_regime'] = 'Weak'
        
        return df
    
    def _classify_volatility(self, df: pd.DataFrame) -> pd.DataFrame:
        """Classify volatility regime using percentiles"""
        
        # Calculate volatility percentile
        if 'realized_vol' in df.columns:
            df['vol_percentile'] = df['realized_vol'].rolling(
                self.lookback_days, min_periods=20
            ).rank(pct=True) * 100
            
            df['volatility_score'] = df['vol_percentile'] / 100
            
            # Classify based on percentiles
            df.loc[df['vol_percentile'] < self.thresholds['vol_low'], 'volatility_regime'] = 'Low'
            df.loc[
                (df['vol_percentile'] >= self.thresholds['vol_low']) & 
                (df['vol_percentile'] < self.thresholds['vol_normal']), 
                'volatility_regime'
            ] = 'Normal'
            df.loc[
                (df['vol_percentile'] >= self.thresholds['vol_normal']) & 
                (df['vol_percentile'] < self.thresholds['vol_high']), 
                'volatility_regime'
            ] = 'High'
            df.loc[df['vol_percentile'] >= self.thresholds['vol_high'], 'volatility_regime'] = 'Extreme'
        
        return df
    
    def _classify_character(self, df: pd.DataFrame) -> pd.DataFrame:
        """Classify market character (trending/ranging/volatile)"""
        
        # Efficiency ratio determines trending vs ranging
        if 'efficiency_ratio' in df.columns:
            df['character_score'] = df['efficiency_ratio']
            
            # High efficiency + directional = Trending
            trending_mask = (
                (df['efficiency_ratio'] > self.thresholds['efficiency_trending']) & 
                (df['direction_regime'] != 'Sideways')
            )
            df.loc[trending_mask, 'character_regime'] = 'Trending'
            
            # Low efficiency = Ranging
            ranging_mask = df['efficiency_ratio'] < self.thresholds['efficiency_ranging']
            df.loc[ranging_mask, 'character_regime'] = 'Ranging'
            
            # High volatility overrides
            volatile_mask = df['volatility_regime'].isin(['High', 'Extreme'])
            df.loc[volatile_mask, 'character_regime'] = 'Volatile'
            
            # Transition detection (regime about to change)
            if 'regime_age' in df.columns:
                transition_mask = (
                    (df['strength_regime'] == 'Weak') & 
                    (df['regime_age'] > 10)
                )
                df.loc[transition_mask, 'character_regime'] = 'Transitioning'
        
        return df
    
    def _smooth_regimes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply smoothing to prevent regime whipsaws"""
        
        smooth_window = self.thresholds['smoothing_days']
        temp_cols = []
        
        for col in ['direction_regime', 'strength_regime', 'volatility_regime', 'character_regime']:
            # Get forward and inverse mappings
            col_map = self.regime_maps[col]
            inv_map = {v: k for k, v in col_map.items()}
            
            # Map strings to integers
            df[f'{col}_int'] = df[col].map(col_map)
            temp_cols.append(f'{col}_int')
            
            # Apply rolling mode on integers
            df[f'{col}_smooth_int'] = df[f'{col}_int'].rolling(
                window=smooth_window,
                min_periods=1
            ).apply(lambda x: pd.Series(x).mode()[0] if not pd.Series(x).mode().empty else x.iloc[-1])
            temp_cols.append(f'{col}_smooth_int')
            
            # Map integers back to strings
            df[col] = df[f'{col}_smooth_int'].map(inv_map)
        
        # Drop temporary columns
        df.drop(temp_cols, axis=1, inplace=True)
        
        return df

    def _calculate_regime_age(self, regime_series: pd.Series) -> pd.Series:
        """Calculate how many days the current regime has persisted"""
        
        regime_age = pd.Series(index=regime_series.index, dtype=int)
        current_regime = None
        age = 0
        
        for i, (idx, regime) in enumerate(regime_series.items()):
            if regime != current_regime:
                current_regime = regime
                age = 1
            else:
                age += 1
            regime_age.iloc[i] = age
        
        return regime_age
    
    def _calculate_confidence_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate confidence scores for each classification"""
        
        # Direction confidence based on score magnitude
        df['direction_confidence'] = abs(df['direction_score']).clip(0, 1)
        
        # Strength confidence
        df['strength_confidence'] = df['strength_score'].clip(0, 1)
        
        # Volatility confidence based on percentile extremes
        df['volatility_confidence'] = abs(df['vol_percentile'] - 50) / 50
        
        # Character confidence based on efficiency ratio clarity
        df['character_confidence'] = abs(df['character_score'] - 0.5) * 2
        
        # Overall regime confidence
        df['regime_confidence'] = (
            df['direction_confidence'] * 0.3 +
            df['strength_confidence'] * 0.3 +
            df['volatility_confidence'] * 0.2 +
            df['character_confidence'] * 0.2
        ).clip(0, 1)
        
        return df
    
    def _calculate_efficiency_ratio(self, price_series: pd.Series, period: int) -> pd.Series:
        """Calculate Kaufman's Efficiency Ratio"""
        
        # Net change over period
        net_change = abs(price_series - price_series.shift(period))
        
        # Sum of absolute changes
        total_change = price_series.diff().abs().rolling(period).sum()
        
        # Efficiency ratio
        efficiency_ratio = net_change / total_change
        efficiency_ratio = efficiency_ratio.fillna(0.5)
        
        return efficiency_ratio
    
    def _calculate_fractal_dimension(self, price_series: pd.Series, period: int) -> pd.Series:
        """Calculate fractal dimension (1 = trending, 2 = random walk)"""
        
        # Simplified fractal dimension calculation
        log_n = np.log(period)
        
        def fractal_calc(window):
            if len(window) < period:
                return 1.5
            
            # Range over period
            R = window.max() - window.min()
            
            # Average absolute change
            S = window.diff().abs().mean()
            
            if S == 0:
                return 1.5
                
            # Fractal dimension
            return 2 - np.log(R / S) / log_n
        
        return price_series.rolling(period).apply(fractal_calc)
    
    def get_current_regime(self, data: pd.DataFrame) -> DailyRegime:
        """Get the current regime state"""
        
        # Classify all regimes
        df = self.classify_regimes(data)
        
        # Get latest values
        latest = df.iloc[-1]
        
        current = DailyRegime(
            timestamp=df.index[-1],
            direction=latest['direction_regime'],
            strength=latest['strength_regime'],
            volatility=latest['volatility_regime'],
            character=latest['character_regime'],
            direction_confidence=latest['direction_confidence'],
            strength_confidence=latest['strength_confidence'],
            volatility_confidence=latest['volatility_confidence'],
            character_confidence=latest['character_confidence'],
            composite_regime=latest['composite_regime'],
            regime_age=latest['regime_age'],
            trend_score=latest['direction_score'],
            momentum_score=latest.get('strength_score', 0),
            volatility_percentile=latest.get('vol_percentile', 50),
            efficiency_ratio=latest.get('efficiency_ratio', 0.5)
        )
        
        return current
    
    def get_regime_history(self, data: pd.DataFrame, 
                          start_date: str = None, 
                          end_date: str = None) -> pd.DataFrame:
        """Get regime history for analysis"""
        
        df = self.classify_regimes(data)
        
        if start_date:
            df = df[df.index >= start_date]
        if end_date:
            df = df[df.index <= end_date]
            
        # Return key columns for analysis
        return df[[
            'direction_regime', 'strength_regime', 'volatility_regime', 'character_regime',
            'composite_regime', 'regime_age', 'regime_confidence',
            'direction_score', 'strength_score', 'volatility_score', 'character_score'
        ]]
    
    def print_threshold_diagnostics(self, df: pd.DataFrame):
        """Print diagnostics to help tune thresholds"""
        
        print("\n" + "="*60)
        print("THRESHOLD DIAGNOSTICS FOR TUNING")
        print("="*60)
        
        # Direction scores
        print(f"\nDirection Scores Distribution:")
        print(f"  Mean: {df['direction_score'].mean():.3f}")
        print(f"  Std: {df['direction_score'].std():.3f}")
        print(f"  Min: {df['direction_score'].min():.3f}")
        print(f"  Max: {df['direction_score'].max():.3f}")
        print(f"  Percentiles: 10%={df['direction_score'].quantile(0.1):.3f}, 25%={df['direction_score'].quantile(0.25):.3f}, 75%={df['direction_score'].quantile(0.75):.3f}, 90%={df['direction_score'].quantile(0.9):.3f}")
        print(f"  Current thresholds: strong={self.thresholds['direction_strong']}, neutral={self.thresholds['direction_neutral']}")
        
        # Show what percentages would result from different thresholds
        for threshold in [0.1, 0.15, 0.2, 0.25, 0.3, 0.35]:
            up_pct = (df['direction_score'] > threshold).mean() * 100
            down_pct = (df['direction_score'] < -threshold).mean() * 100
            sideways_pct = 100 - up_pct - down_pct
            print(f"    If strong={threshold}: Up={up_pct:.1f}%, Down={down_pct:.1f}%, Sideways={sideways_pct:.1f}%")
        
        # Strength scores  
        print(f"\nStrength Scores Distribution:")
        print(f"  Mean: {df['strength_score'].mean():.3f}")
        print(f"  Std: {df['strength_score'].std():.3f}")
        print(f"  Percentiles: 25%={df['strength_score'].quantile(0.25):.3f}, 50%={df['strength_score'].quantile(0.5):.3f}, 75%={df['strength_score'].quantile(0.75):.3f}, 90%={df['strength_score'].quantile(0.9):.3f}")
        print(f"  Current thresholds: strong={self.thresholds['strength_strong']}, moderate={self.thresholds['strength_moderate']}")
        
        # Efficiency ratio
        print(f"\nEfficiency Ratio Distribution:")
        print(f"  Mean: {df['efficiency_ratio'].mean():.3f}")
        print(f"  Std: {df['efficiency_ratio'].std():.3f}")
        print(f"  Percentiles: 25%={df['efficiency_ratio'].quantile(0.25):.3f}, 50%={df['efficiency_ratio'].quantile(0.5):.3f}, 75%={df['efficiency_ratio'].quantile(0.75):.3f}")
        print(f"  Current thresholds: trending={self.thresholds['efficiency_trending']}, ranging={self.thresholds['efficiency_ranging']}")
        
        # Show what percentages would result
        for threshold in [0.1, 0.15, 0.2, 0.25, 0.3, 0.35]:
            trending_pct = ((df['efficiency_ratio'] > threshold) & (df['direction_regime'] != 'Sideways')).mean() * 100
            ranging_pct = (df['efficiency_ratio'] < threshold * 0.7).mean() * 100  # Using 70% of threshold for ranging
            print(f"    If trending={threshold}: Trending={trending_pct:.1f}%, Ranging={ranging_pct:.1f}%")
    
def validate_regimes(self, regime_data: pd.DataFrame) -> Dict[str, float]:
    """Validate regime classifications are reasonable"""
    
    validations = {}
    
    # Check direction distribution
    dir_dist = regime_data['direction_regime'].value_counts(normalize=True)
    validations['uptrend_pct'] = dir_dist.get('Uptrend', 0) * 100
    validations['downtrend_pct'] = dir_dist.get('Downtrend', 0) * 100
    validations['sideways_pct'] = dir_dist.get('Sideways', 0) * 100
    
    # Check strength distribution
    strength_dist = regime_data['strength_regime'].value_counts(normalize=True)
    validations['strong_pct'] = strength_dist.get('Strong', 0) * 100
    validations['moderate_pct'] = strength_dist.get('Moderate', 0) * 100
    validations['weak_pct'] = strength_dist.get('Weak', 0) * 100
    
    # Check character distribution
    char_dist = regime_data['character_regime'].value_counts(normalize=True)
    validations['trending_pct'] = char_dist.get('Trending', 0) * 100
    validations['ranging_pct'] = char_dist.get('Ranging', 0) * 100
    
    # Check average regime duration
    regime_changes = regime_data['composite_regime'] != regime_data['composite_regime'].shift(1)
    validations['avg_duration'] = len(regime_data) / regime_changes.sum() if regime_changes.sum() > 0 else 0
    
    # Check if any regime dominates too much
    composite_dist = regime_data['composite_regime'].value_counts(normalize=True)
    validations['max_regime_pct'] = composite_dist.iloc[0] * 100 if len(composite_dist) > 0 else 0
    
    # Print warnings
    print("\nREGIME VALIDATION:")
    
    if validations['sideways_pct'] < 8:
        print(f"⚠ Warning: Sideways regime too low ({validations['sideways_pct']:.1f}%) - increase direction_neutral threshold")
    elif validations['sideways_pct'] > 20:
        print(f"⚠ Warning: Sideways regime too high ({validations['sideways_pct']:.1f}%) - decrease direction_neutral threshold")
    
    if validations['strong_pct'] < 15:
        print(f"⚠ Warning: Strong trends too rare ({validations['strong_pct']:.1f}%) - decrease strength_strong threshold")
    elif validations['strong_pct'] > 50:
        print(f"⚠ Warning: Strong trends too common ({validations['strong_pct']:.1f}%) - increase strength_strong threshold")
    
    if validations['trending_pct'] < 30:
        print(f"⚠ Warning: Trending character too low ({validations['trending_pct']:.1f}%) - decrease efficiency_trending threshold")
    elif validations['trending_pct'] > 70:
        print(f"⚠ Warning: Trending character too high ({validations['trending_pct']:.1f}%) - increase efficiency_trending threshold")
    
    if validations['avg_duration'] < 5:
        print(f"⚠ Warning: Regimes changing too fast ({validations['avg_duration']:.1f} days) - increase smoothing")
    elif validations['avg_duration'] > 30:
        print(f"⚠ Warning: Regimes too persistent ({validations['avg_duration']:.1f} days) - decrease smoothing")
    
    if validations['max_regime_pct'] > 25:
        print(f"⚠ Warning: One regime dominates ({validations['max_regime_pct']:.1f}%) - check classification balance")
    
    # Print summary
    print(f"\nRegime Balance Summary:")
    print(f"  Direction: Up={validations['uptrend_pct']:.1f}%, Down={validations['downtrend_pct']:.1f}%, Sideways={validations['sideways_pct']:.1f}%")
    print(f"  Strength: Strong={validations['strong_pct']:.1f}%, Moderate={validations['moderate_pct']:.1f}%, Weak={validations['weak_pct']:.1f}%")
    print(f"  Character: Trending={validations['trending_pct']:.1f}%, Ranging={validations['ranging_pct']:.1f}%")
    print(f"  Persistence: {validations['avg_duration']:.1f} days average")
    
    return validations

