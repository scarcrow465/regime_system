#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
from typing import Dict, Tuple, List
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
    long_edge: float
    short_edge: float
    directional_bias: str
    trend_persistence: float
    mean_reversion: float
    volatility_expansion: float
    primary_behavior: str
    optimal_holding_period: int
    edge_half_life: int
    random_long_sharpe: float
    random_short_sharpe: float
    sample_size: int
    confidence_level: float

class MarketCharacterizer:
    """Characterizes market behavior for trading strategies"""
    
    def __init__(self, transaction_cost: float = 0.0001, trend_period: int = 20, 
                 rsi_period: int = 14, rsi_buy_threshold: float = 30, 
                 rsi_sell_threshold: float = 70, breakout_period: int = 20, 
                 holding_period: int = 1):
        self.transaction_cost = transaction_cost
        self.trend_period = trend_period
        self.rsi_period = rsi_period
        self.rsi_buy_threshold = rsi_buy_threshold
        self.rsi_sell_threshold = rsi_sell_threshold
        self.breakout_period = breakout_period
        self.holding_period = holding_period
    
    def characterize_market(self, data: pd.DataFrame, 
                          instrument: str, 
                          timeframe: str) -> MarketProfile:
        """Complete market characterization"""
        logger.info(f"Characterizing {instrument} {timeframe}")
        
        long_edge, short_edge, bias = self._test_directional_bias(data)
        trend_score = self._test_trend_persistence(data)
        mr_score = self._test_mean_reversion(data)
        breakout_score = self._test_volatility_expansion(data)
        
        behaviors = {
            'trending': trend_score,
            'mean_reverting': mr_score,
            'breakout': breakout_score
        }
        primary_behavior = max(behaviors, key=behaviors.get)
        
        optimal_period = self._find_optimal_holding_period(data, bias)
        edge_half_life = self._test_edge_decay(data, primary_behavior)
        random_long, random_short = self._test_random_baseline(data)
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
        long_sharpe = self._calculate_sharpe(returns)
        short_sharpe = self._calculate_sharpe(-returns)
        
        if long_sharpe > 0.5 and long_sharpe > short_sharpe + 0.3:
            bias = 'long'
        elif short_sharpe > 0.5 and short_sharpe > long_sharpe + 0.3:
            bias = 'short'
        else:
            bias = 'neutral'
        
        logger.info(f"Directional bias: {bias} (Long: {long_sharpe:.3f}, Short: {short_sharpe:.3f})")
        return long_sharpe, short_sharpe, bias
    
    def _simulate_strategy(self, data, long_entries, short_entries, holding_period):
        positions = pd.Series(0, index=data.index)
        for entry_date in long_entries:
            if entry_date in data.index:
                exit_idx = min(data.index.get_loc(entry_date) + holding_period, len(data) - 1)
                exit_date = data.index[exit_idx]
                positions.loc[entry_date:exit_date] += 1
        for entry_date in short_entries:
            if entry_date in data.index:
                exit_idx = min(data.index.get_loc(entry_date) + holding_period, len(data) - 1)
                exit_date = data.index[exit_idx]
                positions.loc[entry_date:exit_date] -= 1

        returns = positions.shift(1).fillna(0) * data['close'].pct_change()
        
        # Combine long and short entries and count occurrences
        all_entries = list(long_entries) + list(short_entries)
        if all_entries:
            entries_count = pd.Series(all_entries).value_counts()
            entries = entries_count.reindex(data.index, fill_value=0)
        else:
            entries = pd.Series(0, index=data.index)
        
        transaction_costs = entries * self.transaction_cost
        net_returns = returns - transaction_costs
        
        return net_returns.mean() / net_returns.std() if net_returns.std() != 0 else 0
    
    def _test_trend_persistence(self, data: pd.DataFrame) -> float:
        """Test trend persistence with configurable SMA period"""
        sma = data['close'].rolling(self.trend_period).mean()
        long_entries = data.index[data['close'] > sma]
        short_entries = data.index[data['close'] < sma]
        
        return self._simulate_strategy(data, long_entries, short_entries, self.holding_period)
    
    def _test_mean_reversion(self, data: pd.DataFrame) -> float:
        """Test mean reversion with configurable RSI"""
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(self.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(self.rsi_period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        long_entries = data.index[rsi < self.rsi_buy_threshold]
        short_entries = data.index[rsi > self.rsi_sell_threshold]
        
        return self._simulate_strategy(data, long_entries, short_entries, self.holding_period)
    
    def _test_volatility_expansion(self, data: pd.DataFrame) -> float:
        """Test volatility breakouts with configurable period"""
        high_n = data['high'].rolling(self.breakout_period).max() if 'high' in data.columns else data['close'].rolling(self.breakout_period).max()
        low_n = data['low'].rolling(self.breakout_period).min() if 'low' in data.columns else data['close'].rolling(self.breakout_period).min()
        
        long_entries = data.index[data['close'] > high_n.shift(1)]
        short_entries = data.index[data['close'] < low_n.shift(1)]
        
        return self._simulate_strategy(data, long_entries, short_entries, self.holding_period)
    
    def _find_optimal_holding_period(self, data: pd.DataFrame, bias: str) -> int:
        """Find optimal holding period with focus on short-term"""
        best_sharpe = -999
        best_period = 1
        test_periods = [1, 2, 3, 5]
        
        for period in test_periods:
            if period > len(data) / 10:
                continue
                
            if bias == 'long' or bias == 'neutral':
                signal = data['close'] > data['close'].shift(period)
            else:
                signal = data['close'] < data['close'].shift(period)
            
            returns = signal.shift(period) * data['close'].pct_change().rolling(period).sum()
            sharpe = self._calculate_sharpe(returns.dropna())
            
            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_period = period
        
        logger.info(f"Optimal holding period: {best_period} bars (Sharpe: {best_sharpe:.3f})")
        return best_period
    
    def _test_edge_decay(self, data: pd.DataFrame, behavior: str) -> int:
        """Test edge decay over time"""
        chunk_size = max(1000, len(data) // 10)
        sharpes = []
        
        for i in range(min(10, len(data) // chunk_size)):
            chunk = data.iloc[i*chunk_size:(i+1)*chunk_size]
            if len(chunk) < 100:
                continue
                
            if behavior == 'trending':
                sharpe = self._test_trend_persistence(chunk)
            elif behavior == 'mean_reverting':
                sharpe = self._test_mean_reversion(chunk)
            else:
                sharpe = self._test_volatility_expansion(chunk)
            sharpes.append(sharpe)
        
        if not sharpes:
            return len(data) // 2
        
        peak_sharpe = max(sharpes)
        half_sharpe = peak_sharpe / 2
        half_life_chunks = next((i + 1 for i, s in enumerate(sharpes) if s < half_sharpe), len(sharpes))
        half_life_bars = half_life_chunks * chunk_size
        logger.info(f"Edge half-life: {half_life_bars} bars")
        return half_life_bars
    
    def _test_random_baseline(self, data: pd.DataFrame) -> Tuple[float, float]:
        """Establish random entry baseline"""
        np.random.seed(42)
        random_long = np.random.choice([0, 1], size=len(data), p=[0.5, 0.5])
        long_returns = random_long * data['close'].pct_change()
        long_returns = long_returns - abs(np.diff(random_long, prepend=0)) * self.transaction_cost
        random_long_sharpe = self._calculate_sharpe(long_returns)
        
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
        
        periods_per_year = 252  # Daily data
        sharpe = clean_returns.mean() / clean_returns.std() * np.sqrt(periods_per_year)
        return sharpe
    
    def _calculate_confidence(self, data: pd.DataFrame, score: float) -> float:
        """Calculate statistical confidence"""
        sample_factor = min(1.0, len(data) / 10000)
        score_factor = min(1.0, abs(score) / 0.5)
        return sample_factor * score_factor

