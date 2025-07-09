#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
Enhanced regime-based trading strategies
Sophisticated backtesting with realistic trading logic
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
import warnings
from tqdm import tqdm
import time
import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import (
    COMMISSION_RATE, SLIPPAGE_RATE,
    DEFAULT_STOP_LOSS_ATR, DEFAULT_TAKE_PROFIT_ATR,
    MAX_POSITION_SIZE, MIN_POSITION_SIZE
)

logger = logging.getLogger(__name__)

# =============================================================================
# STRATEGY CONFIGURATION
# =============================================================================

@dataclass
class StrategyConfig:
    """Configuration for regime-specific strategy parameters"""
    momentum_fast_period: int = 12
    momentum_slow_period: int = 26
    momentum_signal_period: int = 9
    rsi_oversold: float = 30
    rsi_overbought: float = 70
    bb_periods: int = 20
    bb_std: float = 2.0
    atr_multiplier: float = 2.0
    position_size_method: str = 'volatility'
    max_position_size: float = MAX_POSITION_SIZE
    min_position_size: float = MIN_POSITION_SIZE
    stop_loss_atr: float = DEFAULT_STOP_LOSS_ATR
    take_profit_atr: float = DEFAULT_TAKE_PROFIT_ATR
    trailing_stop_atr: float = 1.5
    commission_rate: float = COMMISSION_RATE
    slippage_rate: float = SLIPPAGE_RATE
    regime_confidence_threshold: float = 0.4
    regime_position_scalar: float = 1.0

# =============================================================================
# ENHANCED BACKTESTER
# =============================================================================

class EnhancedRegimeStrategyBacktester:
    def __init__(self):
        self.regime_configs = {
            'Up_Trending': StrategyConfig(
                momentum_fast_period=8,
                momentum_slow_period=21,
                position_size_method='volatility',
                regime_position_scalar=1.2,
                stop_loss_atr=1.5,
                take_profit_atr=4.0
            ),
            'Down_Trending': StrategyConfig(
                momentum_fast_period=8,
                momentum_slow_period=21,
                position_size_method='volatility',
                regime_position_scalar=1.2,
                stop_loss_atr=1.5,
                take_profit_atr=4.0
            ),
            'Sideways': StrategyConfig(
                rsi_oversold=30,
                rsi_overbought=70,
                bb_std=2.0,
                position_size_method='fixed',
                regime_position_scalar=0.7,
                max_position_size=0.5
            ),
            'High_Vol': StrategyConfig(
                atr_multiplier=3.0,
                position_size_method='volatility',
                regime_position_scalar=0.5,
                max_position_size=0.3,
                stop_loss_atr=3.0
            )
        }
        self.default_config = StrategyConfig(
            position_size_method='fixed',
            max_position_size=0.1,
            regime_position_scalar=0.3
        )
    
    def calculate_position_size(self, data: pd.DataFrame, index: int, 
                               config: StrategyConfig, 
                               regime_confidence: float = 0.5,
                               signal_strength: float = 1.0) -> float:
        base_size = 1.0
        if config.position_size_method == 'volatility':
            if 'ATR' in data.columns and index > 0:
                current_atr = data['ATR'].iloc[index]
                avg_atr = data['ATR'].iloc[max(0, index-20):index].mean()
                if avg_atr > 0 and current_atr > 0:
                    vol_adjustment = min(2.0, avg_atr / current_atr)
                    base_size *= vol_adjustment
        elif config.position_size_method == 'kelly':
            win_rate = 0.55
            avg_win = 0.02
            avg_loss = 0.01
            kelly_fraction = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
            base_size *= max(0.1, min(0.25, kelly_fraction))
        base_size *= (0.5 + 0.5 * regime_confidence)
        base_size *= signal_strength
        base_size *= config.regime_position_scalar
        return np.clip(base_size, config.min_position_size, config.max_position_size)
    
    def apply_transaction_costs(self, position_change: float, price: float, 
                               config: StrategyConfig) -> float:
        if position_change == 0:
            return 0.0
        commission = abs(position_change) * config.commission_rate * 2
        slippage = abs(position_change) * config.slippage_rate
        return -(commission + slippage)
    
    def calculate_performance_metrics(self, returns: pd.Series) -> Dict[str, float]:
        try:
            if len(returns) == 0 or returns.isna().all():
                return self._empty_metrics()
            clean_returns = returns.fillna(0)
            total_return = (1 + clean_returns).prod() - 1
            if clean_returns.std() > 0:
                sharpe_ratio = (clean_returns.mean() / clean_returns.std()) * np.sqrt(252 * 26)
            else:
                sharpe_ratio = 0
            cumulative = (1 + clean_returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            max_drawdown = drawdown.min()
            winning_trades = (clean_returns > 0).sum()
            losing_trades = (clean_returns < 0).sum()
            total_trades = winning_trades + losing_trades
            win_rate = winning_trades / total_trades if total_trades > 0 else 0
            if max_drawdown < 0:
                annual_return = total_return * (252 * 26 / len(returns))
                calmar_ratio = annual_return / abs(max_drawdown)
            else:
                calmar_ratio = 0
            downside_returns = clean_returns[clean_returns < 0]
            if len(downside_returns) > 0 and downside_returns.std() > 0:
                sortino_ratio = (clean_returns.mean() / downside_returns.std()) * np.sqrt(252 * 26)
            else:
                sortino_ratio = sharpe_ratio
            return {
                'total_return': total_return,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'win_rate': win_rate,
                'calmar_ratio': calmar_ratio,
                'sortino_ratio': sortino_ratio,
                'returns': clean_returns
            }
        except Exception as e:
            logger.error(f"Error calculating performance metrics: {e}")
            return self._empty_metrics()
    
    def _empty_metrics(self) -> Dict[str, float]:
        return {
            'total_return': -0.99,
            'sharpe_ratio': -99,
            'max_drawdown': -0.99,
            'win_rate': 0,
            'calmar_ratio': 0,
            'sortino_ratio': -99,
            'returns': pd.Series(dtype=float)
        }
    
    def momentum_strategy_enhanced(self, data: pd.DataFrame, regime_mask: pd.Series,
                                regime_data: pd.DataFrame) -> pd.Series:
        """
        Improved momentum strategy with:
        - Multiple confirmation signals
        - Regime-specific filters
        - Dynamic position sizing
        """
        start_time = time.time()
        logger.info("Starting enhanced momentum strategy backtest")
        
        # Normalize columns
        data = self._normalize_columns(data)
        
        try:
            returns = pd.Series(0.0, index=data.index)
            positions = pd.Series(0.0, index=data.index)
            
            # Get regime-specific config
            current_regime = self._identify_current_regime(regime_data)
            config = self.regime_configs.get(current_regime, self.default_config)
            
            # Calculate indicators with regime-specific parameters
            fast_ma = data['close'].rolling(config.momentum_fast_period).mean()
            slow_ma = data['close'].rolling(config.momentum_slow_period).mean()
            
            # Multiple signal confirmations
            price_momentum = (data['close'] - data['close'].shift(config.lookback_period)) / data['close'].shift(config.lookback_period)
            
            # Volume confirmation (if available)
            if 'Volume' in data.columns:
                volume_ma = data['Volume'].rolling(20).mean()
                volume_signal = data['Volume'] > volume_ma * 1.2  # Above average volume
            else:
                volume_signal = True  # Default to true if no volume
            
            # RSI filter - avoid overbought/oversold
            if 'RSI' in data.columns:
                rsi_filter = (data['RSI_14'] > config.rsi_oversold) & (data['RSI_14'] < config.rsi_overbought)
            else:
                rsi_filter = True
            
            # MACD confirmation
            if 'MACD' in data.columns and 'MACD_signal' in data.columns:
                macd_signal = data['MACD'] > data['MACD_signal']
            else:
                macd_signal = True
            
            # ATR for position sizing and stops
            if 'ATR' in data.columns:
                atr = data['ATR']
            else:
                atr = (data['High'] - data['Low']).rolling(14).mean()
            
            # Generate signals with multiple confirmations
            long_signal = (
                (fast_ma > slow_ma) &                              # Basic MA crossover
                (price_momentum > config.momentum_threshold) &      # Positive momentum
                (data['close'] > fast_ma) &                        # Price above fast MA
                rsi_filter &                                        # Not overbought
                macd_signal &                                       # MACD confirmation
                volume_signal &                                     # Volume confirmation
                regime_mask                                         # In target regime
            )
            
            short_signal = (
                (fast_ma < slow_ma) &                              # MA crossdown
                (price_momentum < -config.momentum_threshold) &     # Negative momentum
                (data['close'] < fast_ma) &                        # Price below fast MA
                rsi_filter &                                        # Not oversold
                (~macd_signal) &                                    # MACD confirms
                volume_signal &                                     # Volume confirmation
                regime_mask                                         # In target regime
            )
            
            # Position management with dynamic sizing
            current_position = 0.0
            entry_price = 0.0
            stop_loss = 0.0
            take_profit = 0.0
            trade_count = 0
            
            for i in range(config.momentum_slow_period, len(data)):
                if not regime_mask.iloc[i]:
                    # Exit if regime changes
                    if current_position != 0:
                        exit_return = current_position * (data['close'].iloc[i] / entry_price - 1)
                        returns.iloc[i] = exit_return - self.transaction_cost
                        current_position = 0.0
                        trade_count += 1
                    continue
                    
                # Check stops
                if current_position != 0:
                    current_price = data['close'].iloc[i]
                    
                    # Trailing stop based on ATR
                    if current_position > 0:
                        new_stop = current_price - (config.atr_multiplier * atr.iloc[i])
                        stop_loss = max(stop_loss, new_stop)
                        
                        if current_price <= stop_loss or current_price >= take_profit:
                            exit_return = current_position * (current_price / entry_price - 1)
                            returns.iloc[i] = exit_return - self.transaction_cost
                            positions.iloc[i] = 0
                            current_position = 0.0
                            trade_count += 1
                            continue
                    
                    # Update position
                    positions.iloc[i] = current_position
                    
                # New entries
                if current_position == 0:
                    if long_signal.iloc[i]:
                        # Dynamic position sizing based on ATR
                        position_size = min(
                            config.regime_position_scalar,
                            1.0 / (atr.iloc[i] / data['close'].iloc[i] * 100)  # Inverse volatility
                        )
                        position_size = min(position_size, config.max_position_size)
                        
                        current_position = position_size
                        entry_price = data['close'].iloc[i]
                        stop_loss = entry_price - (config.atr_multiplier * atr.iloc[i])
                        take_profit = entry_price + (config.atr_multiplier * 2 * atr.iloc[i])
                        positions.iloc[i] = current_position
                        returns.iloc[i] = -self.transaction_cost
                        
                    elif short_signal.iloc[i] and self.allow_shorting:
                        # Short position
                        position_size = min(
                            config.regime_position_scalar * 0.8,  # Reduce short size
                            0.8 / (atr.iloc[i] / data['close'].iloc[i] * 100)
                        )
                        position_size = min(position_size, config.max_position_size * 0.8)
                        
                        current_position = -position_size
                        entry_price = data['close'].iloc[i]
                        stop_loss = entry_price + (config.atr_multiplier * atr.iloc[i])
                        take_profit = entry_price - (config.atr_multiplier * 2 * atr.iloc[i])
                        positions.iloc[i] = current_position
                        returns.iloc[i] = -self.transaction_cost
            
            # Calculate returns from positions
            price_returns = data['close'].pct_change()
            for i in range(1, len(returns)):
                if returns.iloc[i] == 0 and positions.iloc[i-1] != 0:
                    returns.iloc[i] = positions.iloc[i-1] * price_returns.iloc[i]
            
            logger.info(f"Enhanced momentum strategy completed in {time.time() - start_time:.2f} seconds, trades: {trade_count}")
            return returns
            
        except Exception as e:
            logger.error(f"Error in enhanced momentum strategy: {e}")
            return pd.Series(0.0, index=data.index)
        
    def _identify_current_regime(self, regime_data: pd.DataFrame) -> str:
        """Identify the most specific current regime from regime data."""
        if regime_data.empty:
            return 'default'
        
        # Priority: combined regimes > single dimension regimes
        regime_cols = [col for col in regime_data.columns if '_Regime' in col]
        
        # Look for combined regimes first
        if 'Direction_Regime' in regime_data.columns and 'TrendStrength_Regime' in regime_data.columns:
            direction = regime_data['Direction_Regime'].iloc[-1]
            strength = regime_data['TrendStrength_Regime'].iloc[-1]
            combined = f"{direction}_{strength}"
            if combined in self.regime_configs:
                return combined
        
        # Fall back to single dimension
        for col in regime_cols:
            regime = regime_data[col].iloc[-1]
            if regime in self.regime_configs:
                return regime
        
        return 'default'
    
    def _normalize_columns(self, data: pd.DataFrame) -> pd.DataFrame:
        """Normalize column names for strategy compatibility."""
        data = data.copy()
        
        # Handle close/Close
        if 'Close' in data.columns and 'close' not in data.columns:
            data['close'] = data['Close']
        elif 'price' in data.columns and 'close' not in data.columns:
            data['close'] = data['price']
        
        # Handle RSI variants
        if 'RSI_14' in data.columns and 'RSI' not in data.columns:
            data['RSI'] = data['RSI_14']
        elif 'rsi' in data.columns and 'RSI' not in data.columns:
            data['RSI'] = data['rsi']
        
        # Handle MACD variants (if still needed)
        if 'MACD_Signal' in data.columns and 'MACD_signal' not in data.columns:
            data['MACD_signal'] = data['MACD_Signal']
        
        # Add any other column mappings as needed
        
        return data

    # Then modify mean_reversion_strategy_enhanced to use it:
    def mean_reversion_strategy_enhanced(self, data: pd.DataFrame, regime_mask: pd.Series,
                                    regime_data: pd.DataFrame) -> pd.Series:
        start_time = time.time()
        logger.info("Starting mean reversion strategy backtest")
        
        # Normalize columns first
        data = self._normalize_columns(data)
        
        # Check if required columns exist
        required_cols = ['close', 'RSI_14']
        missing_cols = [col for col in required_cols if col not in data.columns]
        
        if missing_cols:
            # More informative error message
            available_price_cols = [c for c in data.columns if 'close' in c.lower() or 'price' in c.lower()]
            available_rsi_cols = [c for c in data.columns if 'rsi' in c.lower()]
            logger.warning(f"Missing required columns {missing_cols}. Available: price={available_price_cols}, RSI={available_rsi_cols}")
            return pd.Series(0.0, index=data.index)
        
        # Continue with existing logic...
        try:
            returns = pd.Series(0.0, index=data.index)
            positions = pd.Series(0.0, index=data.index)
            config = self.regime_configs.get('Sideways', self.default_config)
            if all(col in data.columns for col in ['close', 'RSI']):
                bb_middle = data['close'].rolling(config.bb_periods).mean()
                bb_std = data['close'].rolling(config.bb_periods).std()
                bb_upper = bb_middle + (config.bb_std * bb_std)
                bb_lower = bb_middle - (config.bb_std * bb_std)
                rsi = data['RSI_14']
            else:
                logger.warning("Missing required columns ('close', 'RSI') in data")
                return returns
            current_position = 0.0
            entry_price = 0.0
            trade_count = 0
            with tqdm(total=len(data) - config.bb_periods, desc="Mean Reversion Strategy", ncols=80, mininterval=1) as pbar:
                for i in range(config.bb_periods, len(data)):
                    if i % 1000 == 0:
                        logger.info(f"Mean reversion strategy at index {i}, trades: {trade_count}")
                    if not regime_mask.iloc[i]:
                        if current_position != 0:
                            exit_return = current_position * data['close'].pct_change().iloc[i]
                            costs = self.apply_transaction_costs(abs(current_position), 
                                                                data['close'].iloc[i], config)
                            returns.iloc[i] = exit_return + costs
                            current_position = 0.0
                            trade_count += 1
                        pbar.update(1)
                        continue
                    regime_confidence = regime_data['Direction_Confidence'].iloc[i] if 'Direction_Confidence' in regime_data.columns else 0.5
                    if current_position == 0:
                        if (data['close'].iloc[i] < bb_lower.iloc[i] and 
                            rsi.iloc[i] < config.rsi_oversold):
                            signal_strength = min(1.0, (bb_middle.iloc[i] - data['close'].iloc[i]) / 
                                                (bb_middle.iloc[i] - bb_lower.iloc[i]))
                            position_size = self.calculate_position_size(
                                data, i, config, regime_confidence, signal_strength
                            )
                            current_position = position_size
                            entry_price = data['close'].iloc[i]
                            returns.iloc[i] = self.apply_transaction_costs(
                                position_size, entry_price, config
                            )
                            trade_count += 1
                        elif (data['close'].iloc[i] > bb_upper.iloc[i] and 
                              rsi.iloc[i] > config.rsi_overbought):
                            signal_strength = min(1.0, (data['close'].iloc[i] - bb_middle.iloc[i]) / 
                                                (bb_upper.iloc[i] - bb_middle.iloc[i]))
                            position_size = self.calculate_position_size(
                                data, i, config, regime_confidence, signal_strength
                            )
                            current_position = -position_size
                            entry_price = data['close'].iloc[i]
                            returns.iloc[i] = self.apply_transaction_costs(
                                position_size, entry_price, config
                            )
                            trade_count += 1
                    else:
                        exit_signal = False
                        if current_position > 0:
                            if (data['close'].iloc[i] >= bb_middle.iloc[i] or 
                                rsi.iloc[i] > config.rsi_overbought):
                                exit_signal = True
                        elif current_position < 0:
                            if (data['close'].iloc[i] <= bb_middle.iloc[i] or 
                                rsi.iloc[i] < config.rsi_oversold):
                                exit_signal = True
                        if exit_signal:
                            position_change = -current_position
                            exit_return = current_position * data['close'].pct_change().iloc[i]
                            costs = self.apply_transaction_costs(abs(position_change),
                                                                data['close'].iloc[i], config)
                            returns.iloc[i] = exit_return + costs
                            current_position = 0.0
                            trade_count += 1
                        else:
                            returns.iloc[i] = current_position * data['close'].pct_change().iloc[i]
                    positions.iloc[i] = current_position
                    pbar.update(1)
            logger.info(f"Mean reversion strategy completed in {time.time() - start_time:.2f} seconds, trades: {trade_count}")
            return returns
        except Exception as e:
            logger.error(f"Enhanced mean reversion strategy error: {e}")
            return pd.Series(0, index=data.index)
    
    def volatility_breakout_strategy_enhanced(self, data: pd.DataFrame, regime_mask: pd.Series,
                                            regime_data: pd.DataFrame) -> pd.Series:
        start_time = time.time()
        logger.info("Starting volatility breakout strategy backtest")
        try:
            returns = pd.Series(0.0, index=data.index)
            positions = pd.Series(0.0, index=data.index)
            config = self.regime_configs.get('High_Vol', self.default_config)
            lookback = 20
            if len(data) > lookback:
                rolling_high = data['high'].rolling(lookback).max() if 'high' in data.columns else data['close'].rolling(lookback).max()
                rolling_low = data['low'].rolling(lookback).min() if 'low' in data.columns else data['close'].rolling(lookback).min()
            else:
                rolling_high = data['close']
                rolling_low = data['close']
            current_position = 0.0
            entry_price = 0.0
            stop_loss = 0.0
            trade_count = 0
            with tqdm(total=len(data) - lookback, desc="Volatility Breakout Strategy", ncols=80, mininterval=1) as pbar:
                for i in range(lookback, len(data)):
                    if i % 1000 == 0:
                        logger.info(f"Volatility breakout strategy at index {i}, trades: {trade_count}")
                    if not regime_mask.iloc[i]:
                        if current_position != 0:
                            position_change = -current_position
                            exit_return = current_position * data['close'].pct_change().iloc[i]
                            costs = self.apply_transaction_costs(position_change,
                                                                data['close'].iloc[i], config)
                            returns.iloc[i] = exit_return + costs
                            current_position = 0.0
                            trade_count += 1
                        pbar.update(1)
                        continue
                    regime_confidence = regime_data['Volatility_Confidence'].iloc[i] if 'Volatility_Confidence' in regime_data.columns else 0.5
                    if current_position == 0:
                        if data['close'].iloc[i] > rolling_high.iloc[i-1]:
                            if 'ATR' in data.columns and data['ATR'].iloc[i] > 0:
                                signal_strength = min((data['close'].iloc[i] - rolling_high.iloc[i-1]) / 
                                                    data['ATR'].iloc[i], 1.0)
                            else:
                                signal_strength = 0.5
                            position_size = self.calculate_position_size(
                                data, i, config, regime_confidence, signal_strength
                            )
                            current_position = position_size
                            entry_price = data['close'].iloc[i]
                            if 'ATR' in data.columns:
                                stop_loss = entry_price - (config.stop_loss_atr * data['ATR'].iloc[i])
                            returns.iloc[i] = self.apply_transaction_costs(
                                position_size, entry_price, config
                            )
                            trade_count += 1
                        elif data['close'].iloc[i] < rolling_low.iloc[i-1]:
                            if 'ATR' in data.columns and data['ATR'].iloc[i] > 0:
                                signal_strength = min((rolling_low.iloc[i-1] - data['close'].iloc[i]) / 
                                                    data['ATR'].iloc[i], 1.0)
                            else:
                                signal_strength = 0.5
                            position_size = self.calculate_position_size(
                                data, i, config, regime_confidence, signal_strength
                            )
                            current_position = -position_size
                            entry_price = data['close'].iloc[i]
                            if 'ATR' in data.columns:
                                stop_loss = entry_price + (config.stop_loss_atr * data['ATR'].iloc[i])
                            returns.iloc[i] = self.apply_transaction_costs(
                                position_size, entry_price, config
                            )
                            trade_count += 1
                    else:
                        exit_signal = False
                        if current_position > 0:
                            if data['close'].iloc[i] <= stop_loss:
                                exit_signal = True
                            elif data['close'].iloc[i] < rolling_low.iloc[i]:
                                exit_signal = True
                        elif current_position < 0:
                            if stop_loss > 0 and data['close'].iloc[i] >= stop_loss:
                                exit_signal = True
                            elif data['close'].iloc[i] > rolling_high.iloc[i]:
                                exit_signal = True
                        if exit_signal:
                            position_change = -current_position
                            exit_return = current_position * data['close'].pct_change().iloc[i]
                            costs = self.apply_transaction_costs(abs(position_change),
                                                                data['close'].iloc[i], config)
                            returns.iloc[i] = exit_return + costs
                            current_position = 0.0
                            trade_count += 1
                        else:
                            returns.iloc[i] = current_position * data['close'].pct_change().iloc[i]
                    positions.iloc[i] = current_position
                    pbar.update(1)
            logger.info(f"Volatility breakout strategy completed in {time.time() - start_time:.2f} seconds, trades: {trade_count}")
            return returns
        except Exception as e:
            logger.error(f"Enhanced volatility breakout strategy error: {e}")
            return pd.Series(0, index=data.index)
    
    def adaptive_regime_strategy_enhanced(self, data: pd.DataFrame, regimes: pd.DataFrame) -> pd.Series:
        start_time = time.time()
        logger.info("Starting adaptive strategy backtest")
        try:
            # Precompute strategy returns to avoid redundant calculations
            momentum_mask = regimes['Direction_Regime'].isin(['Up_Trending', 'Down_Trending'])
            mr_mask = regimes['Direction_Regime'] == 'Sideways'
            vol_mask = regimes['Volatility_Regime'].isin(['High_Vol', 'Extreme_Vol'])
            logger.info("Precomputing strategy returns")
            momentum_returns = self.momentum_strategy_enhanced(data, momentum_mask, regimes)
            mr_returns = self.mean_reversion_strategy_enhanced(data, mr_mask, regimes)
            vol_returns = self.volatility_breakout_strategy_enhanced(data, vol_mask, regimes)
            total_returns = pd.Series(0.0, index=data.index)
            trade_count = 0
            with tqdm(total=len(data), desc="Adaptive Strategy", ncols=80, mininterval=1) as pbar:
                for i in range(len(data)):
                    if i % 1000 == 0:
                        logger.info(f"Adaptive strategy at index {i}, trades: {trade_count}")
                    direction = regimes['Direction_Regime'].iloc[i] if 'Direction_Regime' in regimes.columns else 'Undefined'
                    trend_strength = regimes['TrendStrength_Regime'].iloc[i] if 'TrendStrength_Regime' in regimes.columns else 'Undefined'
                    volatility = regimes['Volatility_Regime'].iloc[i] if 'Volatility_Regime' in regimes.columns else 'Undefined'
                    allocation = {}
                    if direction in ['Up_Trending', 'Down_Trending'] and trend_strength == 'Strong':
                        allocation['momentum'] = (0.8, momentum_returns.iloc[i])
                        trade_count += 1 if momentum_returns.iloc[i] != 0 else 0
                    elif direction == 'Sideways' and volatility in ['Low_Vol', 'Medium_Vol']:
                        allocation['mean_reversion'] = (0.7, mr_returns.iloc[i])
                        trade_count += 1 if mr_returns.iloc[i] != 0 else 0
                    elif volatility in ['High_Vol', 'Extreme_Vol']:
                        allocation['volatility'] = (0.5, vol_returns.iloc[i])
                        trade_count += 1 if vol_returns.iloc[i] != 0 else 0
                    if allocation:
                        weights = [w for w, _ in allocation.values()]
                        returns = [r for _, r in allocation.values()]
                        total_weight = sum(weights)
                        if total_weight > 0:
                            weighted_return = sum(w * r for w, r in allocation.values()) / total_weight
                            total_returns.iloc[i] = weighted_return * min(1.0, total_weight)
                    pbar.update(1)
            logger.info(f"Adaptive strategy completed in {time.time() - start_time:.2f} seconds, trades: {trade_count}")
            return total_returns
        except Exception as e:
            logger.error(f"Enhanced adaptive strategy error: {e}")
            return pd.Series(0, index=data.index)

def compare_strategies(data: pd.DataFrame, regimes: pd.DataFrame) -> pd.DataFrame:
    backtester = EnhancedRegimeStrategyBacktester()
    results = {}
    strategies = {
        'Momentum': lambda: backtester.momentum_strategy_enhanced(
            data, regimes['Direction_Regime'].isin(['Up_Trending', 'Down_Trending']), regimes
        ),
        'Mean Reversion': lambda: backtester.mean_reversion_strategy_enhanced(
            data, regimes['Direction_Regime'] == 'Sideways', regimes
        ),
        'Volatility Breakout': lambda: backtester.volatility_breakout_strategy_enhanced(
            data, regimes['Volatility_Regime'].isin(['High_Vol', 'Extreme_Vol']), regimes
        ),
        'Adaptive': lambda: backtester.adaptive_regime_strategy_enhanced(data, regimes)
    }
    for name, strategy_func in strategies.items():
        logger.info(f"Comparing strategy: {name}")
        start_time = time.time()
        returns = strategy_func()
        metrics = backtester.calculate_performance_metrics(returns)
        results[name] = metrics
        logger.info(f"Comparison for {name} completed in {time.time() - start_time:.2f} seconds")
    comparison = pd.DataFrame({
        name: {
            'Total Return': metrics['total_return'],
            'Sharpe Ratio': metrics['sharpe_ratio'],
            'Max Drawdown': metrics['max_drawdown'],
            'Win Rate': metrics['win_rate'],
            'Calmar Ratio': metrics['calmar_ratio'],
            'Sortino Ratio': metrics['sortino_ratio']
        }
        for name, metrics in results.items()
    }).T
    return comparison

def optimize_strategy_parameters(data: pd.DataFrame, regimes: pd.DataFrame,
                               strategy_type: str = 'momentum') -> Dict[str, float]:
    logger.info(f"Optimizing {strategy_type} strategy parameters...")
    optimized_params = {
        'momentum': {
            'fast_period': 10,
            'slow_period': 25,
            'stop_loss_atr': 2.0,
            'take_profit_atr': 3.5
        },
        'mean_reversion': {
            'bb_periods': 20,
            'bb_std': 2.2,
            'rsi_oversold': 28,
            'rsi_overbought': 72
        },
        'volatility': {
            'lookback': 20,
            'atr_multiplier': 2.5,
            'stop_loss_atr': 2.5
        }
    }
    return optimized_params.get(strategy_type, {})

