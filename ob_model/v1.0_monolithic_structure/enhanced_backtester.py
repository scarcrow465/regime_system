#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# =============================================================================
# ENHANCED REGIME STRATEGY BACKTESTER - INSTITUTIONAL GRADE
# Sophisticated strategy implementation with proper risk management
# Preserves your working regime classification while improving performance
# =============================================================================

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

@dataclass
class StrategyConfig:
    """Configuration for regime-specific strategy parameters"""
    # Entry/Exit parameters
    momentum_fast_period: int = 12
    momentum_slow_period: int = 26
    momentum_signal_period: int = 9
    rsi_oversold: float = 30
    rsi_overbought: float = 70
    bb_periods: int = 20
    bb_std: float = 2.0
    atr_multiplier: float = 2.0
    
    # Risk Management
    position_size_method: str = 'volatility'  # 'fixed', 'volatility', 'kelly'
    max_position_size: float = 1.0
    min_position_size: float = 0.1
    stop_loss_atr: float = 2.0
    take_profit_atr: float = 3.0
    trailing_stop_atr: float = 1.5
    
    # Transaction Costs
    commission_rate: float = 0.00005  # 0.5 bps per side
    slippage_rate: float = 0.00010   # 1 bp slippage
    
    # Regime-specific adjustments
    regime_confidence_threshold: float = 0.4
    regime_position_scalar: float = 1.0

class EnhancedRegimeStrategyBacktester:
    """
    Institutional-grade strategy backtesting with sophisticated risk management
    """
    
    def __init__(self):
        # Regime-specific configurations
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
                rsi_oversold=25,
                rsi_overbought=75,
                bb_std=2.5,
                position_size_method='fixed',
                regime_position_scalar=0.7,
                max_position_size=0.5
            ),
            'High_Vol': StrategyConfig(
                atr_multiplier=3.0,
                position_size_method='volatility',
                max_position_size=0.3,
                stop_loss_atr=3.0,
                regime_position_scalar=0.5
            ),
            'Low_Vol': StrategyConfig(
                position_size_method='volatility',
                max_position_size=1.5,
                regime_position_scalar=1.3,
                stop_loss_atr=1.0
            )
        }
        
        # Initialize performance tracking
        self.trade_history = []
        self.position_history = []
        
    def calculate_position_size(self, data: pd.DataFrame, idx: int, config: StrategyConfig, 
                              regime_confidence: float, signal_strength: float) -> float:
        """Calculate position size based on volatility and regime confidence"""
        
        if config.position_size_method == 'fixed':
            base_size = 1.0
        elif config.position_size_method == 'volatility':
            # Inverse volatility sizing
            current_vol = data['ATR'].iloc[idx] / data['close'].iloc[idx]
            target_vol = 0.02  # 2% target volatility
            base_size = min(target_vol / current_vol, 2.0) if current_vol > 0 else 1.0
        else:  # Kelly criterion approximation
            win_rate = 0.55  # Conservative estimate
            avg_win_loss_ratio = 1.5
            kelly_fraction = (win_rate * avg_win_loss_ratio - (1 - win_rate)) / avg_win_loss_ratio
            base_size = max(0, min(kelly_fraction * 4, 1.0))  # Scale and cap Kelly
        
        # Adjust for regime confidence
        if regime_confidence >= config.regime_confidence_threshold:
            confidence_scalar = 1.0 + (regime_confidence - config.regime_confidence_threshold)
        else:
            confidence_scalar = 0.5
        
        # Adjust for signal strength
        signal_scalar = 0.5 + (signal_strength * 0.5)  # Scale between 0.5 and 1.0
        
        # Apply regime-specific scalar
        regime_scalar = config.regime_position_scalar
        
        # Calculate final position size
        position_size = base_size * confidence_scalar * signal_scalar * regime_scalar
        
        # Apply limits
        position_size = max(config.min_position_size, 
                          min(config.max_position_size, position_size))
        
        return position_size
    
    def calculate_signal_strength(self, data: pd.DataFrame, idx: int, 
                                signal_type: str, config: StrategyConfig) -> float:
        """Calculate signal strength from multiple indicators"""
        
        strength_components = []
        
        if signal_type == 'momentum_long':
            # MACD confirmation
            if 'MACD' in data.columns and 'MACD_signal' in data.columns:
                if data['MACD'].iloc[idx] > data['MACD_signal'].iloc[idx]:
                    macd_strength = min((data['MACD'].iloc[idx] - data['MACD_signal'].iloc[idx]) / 
                                      data['ATR'].iloc[idx], 1.0)
                    strength_components.append(macd_strength)
            
            # RSI not overbought
            if 'RSI' in data.columns:
                rsi_strength = max(0, (config.rsi_overbought - data['RSI'].iloc[idx]) / 
                                  (config.rsi_overbought - 50))
                strength_components.append(rsi_strength)
            
            # Price above moving average
            if 'SMA_20' in data.columns:
                ma_strength = min((data['close'].iloc[idx] - data['SMA_20'].iloc[idx]) / 
                                data['ATR'].iloc[idx], 1.0)
                if ma_strength > 0:
                    strength_components.append(ma_strength)
        
        elif signal_type == 'momentum_short':
            # Inverse of long signals
            if 'MACD' in data.columns and 'MACD_signal' in data.columns:
                if data['MACD'].iloc[idx] < data['MACD_signal'].iloc[idx]:
                    macd_strength = min((data['MACD_signal'].iloc[idx] - data['MACD'].iloc[idx]) / 
                                      data['ATR'].iloc[idx], 1.0)
                    strength_components.append(macd_strength)
            
            if 'RSI' in data.columns:
                rsi_strength = max(0, (data['RSI'].iloc[idx] - config.rsi_oversold) / 
                                  (50 - config.rsi_oversold))
                strength_components.append(rsi_strength)
        
        elif signal_type == 'mean_reversion':
            # Bollinger Band position
            if 'BB_upper' in data.columns and 'BB_lower' in data.columns:
                bb_width = data['BB_upper'].iloc[idx] - data['BB_lower'].iloc[idx]
                if bb_width > 0:
                    if data['close'].iloc[idx] < data['BB_lower'].iloc[idx]:
                        bb_strength = min((data['BB_lower'].iloc[idx] - data['close'].iloc[idx]) / 
                                        (bb_width * 0.5), 1.0)
                        strength_components.append(bb_strength)
                    elif data['close'].iloc[idx] > data['BB_upper'].iloc[idx]:
                        bb_strength = min((data['close'].iloc[idx] - data['BB_upper'].iloc[idx]) / 
                                        (bb_width * 0.5), 1.0)
                        strength_components.append(bb_strength)
        
        # Return average strength
        return np.mean(strength_components) if strength_components else 0.5
    
    def apply_transaction_costs(self, position_change: float, price: float, 
                              config: StrategyConfig) -> float:
        """Calculate transaction costs including commission and slippage"""
        
        if position_change == 0:
            return 0
        
        # Commission (both sides)
        commission = abs(position_change) * price * config.commission_rate * 2
        
        # Slippage (worse fill)
        slippage = abs(position_change) * price * config.slippage_rate
        
        return -(commission + slippage) / price  # Return as percentage

    def adaptive_regime_strategy(self, data: pd.DataFrame, regimes: pd.DataFrame) -> pd.Series:
        """Wrapper for compatibility with original optimizer"""
        return self.adaptive_regime_strategy_enhanced(data, regimes)
    
    def momentum_strategy_enhanced(self, data: pd.DataFrame, regime_mask: pd.Series,
                                 regime_data: pd.DataFrame) -> pd.Series:
        """Enhanced momentum strategy with sophisticated entry/exit logic"""
        
        try:
            returns = pd.Series(0.0, index=data.index)
            positions = pd.Series(0.0, index=data.index)
            
            # Get regime-specific config
            config = self.regime_configs.get('Up_Trending', StrategyConfig())
            
            # Calculate indicators if not present
            if f'EMA_{config.momentum_fast_period}' not in data.columns:
                logger.warning(f"EMA_{config.momentum_fast_period} not found, using defaults")
                fast_ma = data['EMA_12'] if 'EMA_12' in data.columns else data['close'].rolling(12).mean()
                slow_ma = data['EMA_26'] if 'EMA_26' in data.columns else data['close'].rolling(26).mean()
            else:
                fast_ma = data[f'EMA_{config.momentum_fast_period}']
                slow_ma = data[f'EMA_{config.momentum_slow_period}']
            
            # Initialize tracking variables
            current_position = 0.0
            entry_price = 0.0
            stop_loss = 0.0
            take_profit = 0.0
            
            for i in range(1, len(data)):
                if not regime_mask.iloc[i]:
                    # Exit if not in regime
                    if current_position != 0:
                        position_change = -current_position
                        returns.iloc[i] = (current_position * data['close'].pct_change().iloc[i] + 
                                         self.apply_transaction_costs(position_change, 
                                                                    data['close'].iloc[i], config))
                        current_position = 0.0
                    continue
                
                # Get regime confidence
                regime_confidence = regime_data['Direction_Confidence'].iloc[i] if 'Direction_Confidence' in regime_data.columns else 0.5
                
                # Entry logic
                if current_position == 0:
                    # Long entry conditions
                    if (fast_ma.iloc[i] > slow_ma.iloc[i] and 
                        fast_ma.iloc[i-1] <= slow_ma.iloc[i-1]):  # Crossover
                        
                        signal_strength = self.calculate_signal_strength(data, i, 'momentum_long', config)
                        position_size = self.calculate_position_size(data, i, config, 
                                                                   regime_confidence, signal_strength)
                        
                        if position_size > config.min_position_size:
                            current_position = position_size
                            entry_price = data['close'].iloc[i]
                            stop_loss = entry_price - (config.stop_loss_atr * data['ATR'].iloc[i])
                            take_profit = entry_price + (config.take_profit_atr * data['ATR'].iloc[i])
                            
                            returns.iloc[i] = self.apply_transaction_costs(current_position, 
                                                                         entry_price, config)
                
                # Exit logic
                elif current_position > 0:
                    current_price = data['close'].iloc[i]
                    
                    # Update trailing stop
                    new_stop = current_price - (config.trailing_stop_atr * data['ATR'].iloc[i])
                    stop_loss = max(stop_loss, new_stop)
                    
                    exit_signal = False
                    
                    # Check exit conditions
                    if current_price <= stop_loss:  # Stop loss hit
                        exit_signal = True
                    elif current_price >= take_profit:  # Take profit hit
                        exit_signal = True
                    elif fast_ma.iloc[i] < slow_ma.iloc[i]:  # Trend reversal
                        exit_signal = True
                    
                    if exit_signal:
                        position_change = -current_position
                        returns.iloc[i] = (current_position * data['close'].pct_change().iloc[i] + 
                                         self.apply_transaction_costs(position_change, 
                                                                    current_price, config))
                        current_position = 0.0
                    else:
                        # Hold position
                        returns.iloc[i] = current_position * data['close'].pct_change().iloc[i]
                
                positions.iloc[i] = current_position
            
            return returns
            
        except Exception as e:
            logger.error(f"Enhanced momentum strategy error: {e}")
            return pd.Series(0, index=data.index)
    
    def mean_reversion_strategy_enhanced(self, data: pd.DataFrame, regime_mask: pd.Series,
                                       regime_data: pd.DataFrame) -> pd.Series:
        """Enhanced mean reversion with Bollinger Bands and RSI confirmation"""
        
        try:
            returns = pd.Series(0.0, index=data.index)
            positions = pd.Series(0.0, index=data.index)
            
            config = self.regime_configs.get('Sideways', StrategyConfig())
            
            # Initialize tracking
            current_position = 0.0
            entry_price = 0.0
            
            for i in range(1, len(data)):
                if not regime_mask.iloc[i]:
                    if current_position != 0:
                        position_change = -current_position
                        returns.iloc[i] = (current_position * data['close'].pct_change().iloc[i] + 
                                         self.apply_transaction_costs(position_change, 
                                                                    data['close'].iloc[i], config))
                        current_position = 0.0
                    continue
                
                regime_confidence = regime_data['Direction_Confidence'].iloc[i] if 'Direction_Confidence' in regime_data.columns else 0.5
                
                # Entry logic
                if current_position == 0:
                    # Long entry - oversold conditions
                    if ('RSI' in data.columns and data['RSI'].iloc[i] < config.rsi_oversold and
                        'BB_lower' in data.columns and data['close'].iloc[i] < data['BB_lower'].iloc[i]):
                        
                        signal_strength = self.calculate_signal_strength(data, i, 'mean_reversion', config)
                        position_size = self.calculate_position_size(data, i, config,
                                                                   regime_confidence, signal_strength)
                        
                        if position_size > config.min_position_size:
                            current_position = position_size
                            entry_price = data['close'].iloc[i]
                            returns.iloc[i] = self.apply_transaction_costs(current_position,
                                                                         entry_price, config)
                    
                    # Short entry - overbought conditions
                    elif ('RSI' in data.columns and data['RSI'].iloc[i] > config.rsi_overbought and
                          'BB_upper' in data.columns and data['close'].iloc[i] > data['BB_upper'].iloc[i]):
                        
                        signal_strength = self.calculate_signal_strength(data, i, 'mean_reversion', config)
                        position_size = self.calculate_position_size(data, i, config,
                                                                   regime_confidence, signal_strength)
                        
                        if position_size > config.min_position_size:
                            current_position = -position_size
                            entry_price = data['close'].iloc[i]
                            returns.iloc[i] = self.apply_transaction_costs(abs(current_position),
                                                                         entry_price, config)
                
                # Exit logic
                else:
                    exit_signal = False
                    
                    if current_position > 0:  # Long position
                        # Exit at middle band or RSI neutral
                        if ('BB_middle' in data.columns and data['close'].iloc[i] > data['BB_middle'].iloc[i]) or \
                           ('RSI' in data.columns and data['RSI'].iloc[i] > 50):
                            exit_signal = True
                        # Stop loss
                        elif data['close'].iloc[i] < entry_price * 0.98:
                            exit_signal = True
                            
                    elif current_position < 0:  # Short position
                        # Exit at middle band or RSI neutral
                        if ('BB_middle' in data.columns and data['close'].iloc[i] < data['BB_middle'].iloc[i]) or \
                           ('RSI' in data.columns and data['RSI'].iloc[i] < 50):
                            exit_signal = True
                        # Stop loss
                        elif data['close'].iloc[i] > entry_price * 1.02:
                            exit_signal = True
                    
                    if exit_signal:
                        position_change = -current_position
                        returns.iloc[i] = (current_position * data['close'].pct_change().iloc[i] + 
                                         self.apply_transaction_costs(abs(position_change),
                                                                    data['close'].iloc[i], config))
                        current_position = 0.0
                    else:
                        returns.iloc[i] = current_position * data['close'].pct_change().iloc[i]
                
                positions.iloc[i] = current_position
            
            return returns
            
        except Exception as e:
            logger.error(f"Enhanced mean reversion strategy error: {e}")
            return pd.Series(0, index=data.index)
    
    def volatility_breakout_strategy_enhanced(self, data: pd.DataFrame, regime_mask: pd.Series,
                                            regime_data: pd.DataFrame) -> pd.Series:
        """Enhanced volatility breakout with dynamic thresholds"""
        
        try:
            returns = pd.Series(0.0, index=data.index)
            positions = pd.Series(0.0, index=data.index)
            
            config = self.regime_configs.get('High_Vol', StrategyConfig())
            
            # Calculate dynamic breakout thresholds
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
            
            for i in range(lookback, len(data)):
                if not regime_mask.iloc[i]:
                    if current_position != 0:
                        position_change = -current_position
                        returns.iloc[i] = (current_position * data['close'].pct_change().iloc[i] + 
                                         self.apply_transaction_costs(position_change,
                                                                    data['close'].iloc[i], config))
                        current_position = 0.0
                    continue
                
                regime_confidence = regime_data['Volatility_Confidence'].iloc[i] if 'Volatility_Confidence' in regime_data.columns else 0.5
                
                # Entry logic
                if current_position == 0:
                    # Breakout detection
                    if data['close'].iloc[i] > rolling_high.iloc[i-1]:
                        # Upside breakout
                        signal_strength = min((data['close'].iloc[i] - rolling_high.iloc[i-1]) / 
                                            data['ATR'].iloc[i], 1.0) if data['ATR'].iloc[i] > 0 else 0.5
                        
                        position_size = self.calculate_position_size(data, i, config,
                                                                   regime_confidence, signal_strength)
                        
                        if position_size > config.min_position_size:
                            current_position = position_size
                            entry_price = data['close'].iloc[i]
                            stop_loss = entry_price - (config.stop_loss_atr * data['ATR'].iloc[i])
                            returns.iloc[i] = self.apply_transaction_costs(current_position,
                                                                         entry_price, config)
                    
                    elif data['close'].iloc[i] < rolling_low.iloc[i-1]:
                        # Downside breakout
                        signal_strength = min((rolling_low.iloc[i-1] - data['close'].iloc[i]) / 
                                            data['ATR'].iloc[i], 1.0) if data['ATR'].iloc[i] > 0 else 0.5
                        
                        position_size = self.calculate_position_size(data, i, config,
                                                                   regime_confidence, signal_strength)
                        
                        if position_size > config.min_position_size:
                            current_position = -position_size
                            entry_price = data['close'].iloc[i]
                            stop_loss = entry_price + (config.stop_loss_atr * data['ATR'].iloc[i])
                            returns.iloc[i] = self.apply_transaction_costs(abs(current_position),
                                                                         entry_price, config)
                
                # Exit logic
                elif current_position != 0:
                    exit_signal = False
                    
                    if current_position > 0:  # Long position
                        if data['close'].iloc[i] <= stop_loss:
                            exit_signal = True
                        # Take profit at 3 ATR
                        elif data['close'].iloc[i] >= entry_price + (config.take_profit_atr * data['ATR'].iloc[i]):
                            exit_signal = True
                        # Exit if momentum dies (price back below breakout level)
                        elif data['close'].iloc[i] < rolling_high.iloc[i]:
                            exit_signal = True
                    
                    else:  # Short position
                        if data['close'].iloc[i] >= stop_loss:
                            exit_signal = True
                        elif data['close'].iloc[i] <= entry_price - (config.take_profit_atr * data['ATR'].iloc[i]):
                            exit_signal = True
                        elif data['close'].iloc[i] > rolling_low.iloc[i]:
                            exit_signal = True
                    
                    if exit_signal:
                        position_change = -current_position
                        returns.iloc[i] = (current_position * data['close'].pct_change().iloc[i] + 
                                         self.apply_transaction_costs(abs(position_change),
                                                                    data['close'].iloc[i], config))
                        current_position = 0.0
                    else:
                        returns.iloc[i] = current_position * data['close'].pct_change().iloc[i]
                
                positions.iloc[i] = current_position
            
            return returns
            
        except Exception as e:
            logger.error(f"Enhanced volatility breakout strategy error: {e}")
            return pd.Series(0, index=data.index)
    
    def adaptive_regime_strategy_enhanced(self, data: pd.DataFrame, regimes: pd.DataFrame) -> pd.Series:
        """Enhanced adaptive strategy with sophisticated regime-based allocation"""
        
        try:
            total_returns = pd.Series(0.0, index=data.index)
            
            # Create composite regime score
            regime_scores = pd.DataFrame(index=data.index)
            
            # Calculate regime strength scores
            if 'Direction_Confidence' in regimes.columns:
                regime_scores['direction_score'] = regimes['Direction_Confidence']
            else:
                regime_scores['direction_score'] = 0.5
            
            if 'TrendStrength_Confidence' in regimes.columns:
                regime_scores['trend_score'] = regimes['TrendStrength_Confidence']
            else:
                regime_scores['trend_score'] = 0.5
            
            if 'Volatility_Confidence' in regimes.columns:
                regime_scores['volatility_score'] = regimes['Volatility_Confidence']
            else:
                regime_scores['volatility_score'] = 0.5
            
            # Define sophisticated regime strategy mapping
            strategy_allocations = []
            
            for i in range(len(data)):
                allocation = {}
                
                # Get current regimes
                direction = regimes['Direction_Regime'].iloc[i] if 'Direction_Regime' in regimes.columns else 'Undefined'
                trend_strength = regimes['TrendStrength_Regime'].iloc[i] if 'TrendStrength_Regime' in regimes.columns else 'Undefined'
                volatility = regimes['Volatility_Regime'].iloc[i] if 'Volatility_Regime' in regimes.columns else 'Undefined'
                
                # Strong trending market
                if direction in ['Up_Trending', 'Down_Trending'] and trend_strength == 'Strong':
                    if volatility == 'Low_Vol':
                        allocation['momentum'] = 0.8
                        allocation['trend_following'] = 0.2
                    else:
                        allocation['momentum'] = 0.6
                        allocation['volatility_breakout'] = 0.4
                
                # Sideways market
                elif direction == 'Sideways':
                    if volatility in ['Low_Vol', 'Medium_Vol']:
                        allocation['mean_reversion'] = 0.7
                        allocation['range_trading'] = 0.3
                    else:
                        allocation['mean_reversion'] = 0.4
                        allocation['volatility_breakout'] = 0.6
                
                # Weak trend
                elif trend_strength == 'Weak':
                    allocation['mean_reversion'] = 0.5
                    allocation['momentum'] = 0.3
                    allocation['neutral'] = 0.2
                
                # Default balanced allocation
                else:
                    allocation['momentum'] = 0.3
                    allocation['mean_reversion'] = 0.3
                    allocation['volatility_breakout'] = 0.2
                    allocation['neutral'] = 0.2
                
                strategy_allocations.append(allocation)
            
            # Apply strategies based on allocations
            for strategy_name, default_weight in [('momentum', 0.3), ('mean_reversion', 0.3), 
                                                 ('volatility_breakout', 0.2)]:
                
                if strategy_name == 'momentum':
                    # Use appropriate regime mask
                    mask = regimes['Direction_Regime'].isin(['Up_Trending', 'Down_Trending'])
                    strategy_returns = self.momentum_strategy_enhanced(data, mask, regimes)
                    
                elif strategy_name == 'mean_reversion':
                    mask = regimes['Direction_Regime'] == 'Sideways'
                    strategy_returns = self.mean_reversion_strategy_enhanced(data, mask, regimes)
                    
                elif strategy_name == 'volatility_breakout':
                    mask = regimes['Volatility_Regime'].str.contains('High_Vol')
                    strategy_returns = self.volatility_breakout_strategy_enhanced(data, mask, regimes)
                
                # Apply dynamic weights
                for i in range(len(data)):
                    weight = strategy_allocations[i].get(strategy_name, default_weight)
                    total_returns.iloc[i] += strategy_returns.iloc[i] * weight
            
            return total_returns
            
        except Exception as e:
            logger.error(f"Enhanced adaptive strategy error: {e}")
            return pd.Series(0, index=data.index)
    
    def calculate_performance_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """Calculate comprehensive performance metrics"""
        
        try:
            # Remove any NaN values
            clean_returns = returns.fillna(0)
            
            # Basic metrics
            total_return = (1 + clean_returns).prod() - 1
            
            # Sharpe ratio (annualized for 15-min data)
            if clean_returns.std() > 0:
                periods_per_year = 252 * 26  # 26 fifteen-minute periods per trading day
                sharpe_ratio = np.sqrt(periods_per_year) * clean_returns.mean() / clean_returns.std()
            else:
                sharpe_ratio = 0
            
            # Maximum drawdown
            cum_returns = (1 + clean_returns).cumprod()
            running_max = cum_returns.expanding().max()
            drawdown = (cum_returns - running_max) / running_max
            max_drawdown = drawdown.min()
            
            # Additional metrics
            win_rate = (clean_returns > 0).sum() / (clean_returns != 0).sum() if (clean_returns != 0).sum() > 0 else 0
            
            # Calmar ratio
            calmar_ratio = total_return / abs(max_drawdown) if max_drawdown != 0 else 0
            
            # Sortino ratio (downside deviation)
            negative_returns = clean_returns[clean_returns < 0]
            if len(negative_returns) > 0 and negative_returns.std() > 0:
                sortino_ratio = np.sqrt(periods_per_year) * clean_returns.mean() / negative_returns.std()
            else:
                sortino_ratio = sharpe_ratio
            
            return {
                'total_return': total_return,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'win_rate': win_rate,
                'calmar_ratio': calmar_ratio,
                'sortino_ratio': sortino_ratio,
                'volatility': clean_returns.std() * np.sqrt(periods_per_year)
            }
            
        except Exception as e:
            logger.error(f"Performance calculation error: {e}")
            return {
                'total_return': 0,
                'sharpe_ratio': 0,
                'max_drawdown': -1.0,
                'win_rate': 0,
                'calmar_ratio': 0,
                'sortino_ratio': 0,
                'volatility': 0
            }

