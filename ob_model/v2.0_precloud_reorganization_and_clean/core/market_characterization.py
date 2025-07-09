#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
Market Behavioral Fingerprint System
Save this as: market_characterizer.py
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple
from dataclasses import dataclass
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class MarketProfile:
    """Stores market behavioral characteristics"""
    instrument: str
    timeframe: str
    
    # Directional bias
    long_edge: float  # Sharpe of buy & hold
    short_edge: float  # Sharpe of short & hold
    directional_bias: str  # 'long', 'short', 'neutral'
    
    # Behavioral type
    trend_persistence: float  # Momentum strategy sharpe
    mean_reversion: float  # Fade strategy sharpe
    volatility_expansion: float  # Breakout strategy sharpe
    primary_behavior: str  # 'trending', 'mean_reverting', 'breakout'
    
    # Optimal parameters
    optimal_holding_period: int  # Bars
    edge_half_life: int  # Bars until edge decays 50%
    
    # Random baseline
    random_long_sharpe: float
    random_short_sharpe: float
    
    # Statistical significance
    sample_size: int
    confidence_level: float


class MarketCharacterizer:
    """Characterizes market behavior to establish baseline edge"""
    
    def __init__(self, transaction_cost: float = 0.0001):
        self.transaction_cost = transaction_cost
    
    def characterize_market(self, data: pd.DataFrame, 
                          instrument: str, 
                          timeframe: str) -> MarketProfile:
        """Complete market characterization"""
        logger.info(f"Characterizing {instrument} {timeframe}")
        
        # 1. Test directional bias
        long_edge, short_edge, bias = self._test_directional_bias(data)
        
        # 2. Test behavioral patterns
        trend_score = self._test_trend_persistence(data)
        mr_score = self._test_mean_reversion(data)
        breakout_score = self._test_volatility_expansion(data)
        
        # 3. Determine primary behavior
        behaviors = {
            'trending': trend_score,
            'mean_reverting': mr_score,
            'breakout': breakout_score
        }
        primary_behavior = max(behaviors, key=behaviors.get)
        
        # 4. Find optimal holding period
        optimal_period = self._find_optimal_holding_period(data, bias)
        
        # 5. Test edge decay
        edge_half_life = self._test_edge_decay(data, primary_behavior)
        
        # 6. Establish random baseline
        random_long, random_short = self._test_random_baseline(data)
        
        # 7. Calculate confidence
        confidence = self._calculate_confidence(data, behaviors[primary_behavior])
        
        return MarketProfile(
            instrument=instrument,
            timeframe=timeframe,
            long_edge=long_edge,
            short_edge=short_edge,
            directional_bias=bias,
            trend_persistence=trend_score,
            mean_reversion=mr_score,
            volatility_expansion=breakout_score,
            primary_behavior=primary_behavior,
            optimal_holding_period=optimal_period,
            edge_half_life=edge_half_life,
            random_long_sharpe=random_long,
            random_short_sharpe=random_short,
            sample_size=len(data),
            confidence_level=confidence
        )
    
    def _test_directional_bias(self, data: pd.DataFrame) -> Tuple[float, float, str]:
        """Test if market has inherent directional bias"""
        returns = data['close'].pct_change().dropna()
        
        # Buy and hold
        long_returns = returns
        long_sharpe = self._calculate_sharpe(long_returns)
        
        # Short and hold
        short_returns = -returns
        short_sharpe = self._calculate_sharpe(short_returns)
        
        # Determine bias
        if long_sharpe > 0.5 and long_sharpe > short_sharpe + 0.3:
            bias = 'long'
        elif short_sharpe > 0.5 and short_sharpe > long_sharpe + 0.3:
            bias = 'short'
        else:
            bias = 'neutral'
        
        logger.info(f"Directional bias: {bias} (Long: {long_sharpe:.3f}, Short: {short_sharpe:.3f})")
        return long_sharpe, short_sharpe, bias
    
    def _test_trend_persistence(self, data: pd.DataFrame) -> float:
        """Test if trends persist (momentum works)"""
        # Simple momentum: buy if price > 20-period SMA
        sma = data['close'].rolling(20).mean()
        signal = (data['close'] > sma).astype(int).diff()
        
        # Calculate returns
        position = signal.fillna(0).cumsum()
        returns = position.shift(1) * data['close'].pct_change()
        returns = returns - abs(signal) * self.transaction_cost
        
        return self._calculate_sharpe(returns)
    
    def _test_mean_reversion(self, data: pd.DataFrame) -> float:
        """Test if market mean reverts"""
        # Simple mean reversion: buy when RSI < 30, sell when RSI > 70
        if 'RSI_14' in data.columns:
            rsi = data['RSI_14']
        else:
            # Calculate RSI if not present
            delta = data['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
        
        # Generate signals
        long_signal = (rsi < 30).astype(int)
        short_signal = (rsi > 70).astype(int)
        position = long_signal - short_signal
        
        # Calculate returns
        returns = position.shift(1) * data['close'].pct_change()
        returns = returns - abs(position.diff()) * self.transaction_cost
        
        return self._calculate_sharpe(returns)
    
    def _test_volatility_expansion(self, data: pd.DataFrame) -> float:
        """Test if volatility breakouts work"""
        # Simple breakout: buy on 20-bar high, sell on 20-bar low
        if 'high' in data.columns and 'low' in data.columns:
            high_20 = data['high'].rolling(20).max()
            low_20 = data['low'].rolling(20).min()
        else:
            # Use close if high/low not available
            high_20 = data['close'].rolling(20).max()
            low_20 = data['close'].rolling(20).min()
        
        long_signal = (data['close'] > high_20.shift(1)).astype(int)
        short_signal = (data['close'] < low_20.shift(1)).astype(int)
        position = long_signal - short_signal
        
        # Calculate returns
        returns = position.shift(1) * data['close'].pct_change()
        returns = returns - abs(position.diff()) * self.transaction_cost
        
        return self._calculate_sharpe(returns)
    
    def _find_optimal_holding_period(self, data: pd.DataFrame, bias: str) -> int:
        """Find optimal holding period for the market"""
        best_sharpe = -999
        best_period = 1
        
        for period in [1, 5, 10, 20, 50, 100, 200]:
            if period > len(data) / 10:  # Skip if period too large for data
                continue
                
            # Test buy/sell and hold for 'period' bars
            if bias == 'long' or bias == 'neutral':
                signal = data['close'] > data['close'].shift(period)
            else:
                signal = data['close'] < data['close'].shift(period)
            
            # Hold for 'period' bars
            returns = signal.shift(period) * data['close'].pct_change().rolling(period).sum()
            sharpe = self._calculate_sharpe(returns.dropna())
            
            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_period = period
        
        logger.info(f"Optimal holding period: {best_period} bars (Sharpe: {best_sharpe:.3f})")
        return best_period
    
    def _test_edge_decay(self, data: pd.DataFrame, behavior: str) -> int:
        """Test how quickly edge decays over time"""
        # Split data into chunks and test strategy performance
        chunk_size = max(1000, len(data) // 10)  # At least 1000 bars per chunk
        sharpes = []
        
        for i in range(min(10, len(data) // chunk_size)):
            chunk = data.iloc[i*chunk_size:(i+1)*chunk_size]
            if len(chunk) < 100:  # Skip tiny chunks
                continue
                
            if behavior == 'trending':
                sharpe = self._test_trend_persistence(chunk)
            elif behavior == 'mean_reverting':
                sharpe = self._test_mean_reversion(chunk)
            else:
                sharpe = self._test_volatility_expansion(chunk)
            sharpes.append(sharpe)
        
        if not sharpes:
            return len(data) // 2  # Default to half the data
        
        # Find where performance drops 50%
        peak_sharpe = max(sharpes)
        half_sharpe = peak_sharpe / 2
        
        half_life_chunks = len(sharpes)  # Default if no decay
        for i, sharpe in enumerate(sharpes):
            if sharpe < half_sharpe:
                half_life_chunks = i + 1
                break
        
        half_life_bars = half_life_chunks * chunk_size
        logger.info(f"Edge half-life: {half_life_bars} bars")
        return half_life_bars
    
    def _test_random_baseline(self, data: pd.DataFrame) -> Tuple[float, float]:
        """Establish random entry baseline"""
        np.random.seed(42)
        
        # Random long entries
        random_long = np.random.choice([0, 1], size=len(data), p=[0.5, 0.5])
        long_returns = random_long * data['close'].pct_change()
        long_returns = long_returns - abs(np.diff(random_long, prepend=0)) * self.transaction_cost
        random_long_sharpe = self._calculate_sharpe(long_returns)
        
        # Random short entries
        random_short = np.random.choice([0, -1], size=len(data), p=[0.5, 0.5])
        short_returns = random_short * data['close'].pct_change()
        short_returns = short_returns - abs(np.diff(random_short, prepend=0)) * self.transaction_cost
        random_short_sharpe = self._calculate_sharpe(short_returns)
        
        logger.info(f"Random baseline - Long: {random_long_sharpe:.3f}, Short: {random_short_sharpe:.3f}")
        return random_long_sharpe, random_short_sharpe
    
    def _calculate_sharpe(self, returns: pd.Series) -> float:
        """Calculate Sharpe ratio"""
        if len(returns) < 20:
            return -999
        
        clean_returns = returns.replace([np.inf, -np.inf], np.nan).dropna()
        if len(clean_returns) == 0 or clean_returns.std() == 0:
            return -999
        
        # Annualized Sharpe for 15-min bars
        periods_per_year = 252 * 26  # 26 fifteen-minute periods per day
        sharpe = clean_returns.mean() / clean_returns.std() * np.sqrt(periods_per_year)
        return sharpe
    
    def _calculate_confidence(self, data: pd.DataFrame, score: float) -> float:
        """Calculate statistical confidence in the edge"""
        # Simple confidence based on sample size and score magnitude
        sample_factor = min(1.0, len(data) / 10000)  # Full confidence at 10k+ samples
        score_factor = min(1.0, abs(score) / 0.5)  # Full confidence at 0.5+ Sharpe
        
        confidence = sample_factor * score_factor
        return confidence

