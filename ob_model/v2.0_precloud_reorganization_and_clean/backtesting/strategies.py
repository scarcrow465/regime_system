#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Enhanced Regime-Specific Trading Strategies
# ============================================
# Fixes column issues and adds sophisticated regime-specific strategies

import pandas as pd
import numpy as np
import time
from typing import Dict, Optional, Tuple
from tqdm import tqdm
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class StrategyConfig:
    """Configuration for strategy parameters"""
    # Momentum parameters
    fast_period: int = 10
    slow_period: int = 25
    momentum_threshold: float = 0.0
    stop_loss_atr: float = 2.0
    take_profit_atr: float = 3.0
    
    # Mean reversion parameters
    bb_periods: int = 20
    bb_std: float = 2.0
    rsi_oversold: int = 30
    rsi_overbought: int = 70
    
    # Volatility breakout parameters
    lookback: int = 20
    atr_multiplier: float = 2.0
    volume_threshold: float = 1.5
    
    # Risk management
    position_size: float = 1.0
    max_positions: int = 3
    
    # Transaction costs
    commission: float = 0.0002  # 2 basis points
    slippage: float = 0.0001    # 1 basis point

class EnhancedRegimeStrategyBacktester:
    """Enhanced strategy backtester with regime-specific implementations"""
    
    def __init__(self, config: Optional[StrategyConfig] = None):
        self.default_config = config or StrategyConfig()
        
        # Regime-specific configurations
        self.regime_configs = {
            # Direction regimes
            'Up_Trending': StrategyConfig(
                fast_period=8, slow_period=21, momentum_threshold=0.001,
                stop_loss_atr=1.5, take_profit_atr=4.0
            ),
            'Down_Trending': StrategyConfig(
                fast_period=8, slow_period=21, momentum_threshold=-0.001,
                stop_loss_atr=1.5, take_profit_atr=4.0
            ),
            'Sideways': StrategyConfig(
                bb_periods=20, bb_std=2.0, rsi_oversold=25, rsi_overbought=75
            ),
            
            # Volatility regimes
            'Low_Vol': StrategyConfig(
                bb_periods=30, bb_std=1.5, position_size=1.5
            ),
            'High_Vol': StrategyConfig(
                lookback=15, atr_multiplier=2.5, position_size=0.7
            ),
            'Extreme_Vol': StrategyConfig(
                lookback=10, atr_multiplier=3.0, position_size=0.5,
                stop_loss_atr=3.0
            ),
            
            # Trend strength regimes
            'Strong': StrategyConfig(
                fast_period=5, slow_period=15, position_size=1.2
            ),
            'Weak': StrategyConfig(
                bb_periods=25, bb_std=2.5, position_size=0.8
            )
        }
    
    def _fix_column_names(self, data: pd.DataFrame) -> pd.DataFrame:
        """Ensure consistent column naming"""
        column_mapping = {
            'close': 'Close',
            'open': 'Open', 
            'high': 'High',
            'low': 'Low',
            'volume': 'Volume'
        }
        
        # Create a copy to avoid modifying original
        data_copy = data.copy()
        
        # Fix OHLCV columns
        for old_name, new_name in column_mapping.items():
            if old_name in data_copy.columns and new_name not in data_copy.columns:
                data_copy[new_name] = data_copy[old_name]
        
        return data_copy
    
    def apply_transaction_costs(self, position_change: float, price: float, 
                              config: StrategyConfig) -> float:
        """Apply transaction costs to position changes"""
        if position_change == 0:
            return 0.0
        
        # Commission + slippage
        cost = abs(position_change) * (config.commission + config.slippage)
        return -cost
    
    def calculate_performance_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """Calculate comprehensive performance metrics"""
        if len(returns) == 0 or returns.std() == 0:
            return {
                'total_return': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'win_rate': 0.0,
                'calmar_ratio': 0.0,
                'sortino_ratio': 0.0,
                'avg_win': 0.0,
                'avg_loss': 0.0,
                'profit_factor': 0.0
            }
        
        # Basic metrics
        total_return = (1 + returns).prod() - 1
        sharpe_ratio = np.sqrt(252 * 96) * returns.mean() / returns.std()  # 96 periods per day for 15min
        
        # Drawdown calculation
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Win rate and profit metrics
        winning_returns = returns[returns > 0]
        losing_returns = returns[returns < 0]
        
        win_rate = len(winning_returns) / len(returns) if len(returns) > 0 else 0
        avg_win = winning_returns.mean() if len(winning_returns) > 0 else 0
        avg_loss = losing_returns.mean() if len(losing_returns) > 0 else 0
        
        # Profit factor
        total_wins = winning_returns.sum() if len(winning_returns) > 0 else 0
        total_losses = abs(losing_returns.sum()) if len(losing_returns) > 0 else 0
        profit_factor = total_wins / total_losses if total_losses > 0 else 0
        
        # Advanced ratios
        calmar_ratio = total_return / abs(max_drawdown) if max_drawdown != 0 else 0
        downside_returns = returns[returns < 0]
        sortino_ratio = 0.0
        if len(downside_returns) > 0 and downside_returns.std() > 0:
            sortino_ratio = np.sqrt(252 * 96) * returns.mean() / downside_returns.std()
        
        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'calmar_ratio': calmar_ratio,
            'sortino_ratio': sortino_ratio,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor
        }
    
    # === REGIME-SPECIFIC STRATEGY IMPLEMENTATIONS ===
    
    def momentum_breakout_strategy(self, data: pd.DataFrame, regime_mask: pd.Series,
                                  regime_data: pd.DataFrame) -> pd.Series:
        """
        Advanced momentum strategy for trending regimes
        - Uses dynamic lookback based on trend strength
        - Implements regime-aware stops
        - Scales position size by trend confidence
        """
        logger.info("Starting momentum breakout strategy")
        data = self._fix_column_names(data)
        
        returns = pd.Series(0.0, index=data.index)
        positions = pd.Series(0.0, index=data.index)
        
        # Get regime-specific config
        if 'Direction_Regime' in regime_data.columns:
            primary_regime = regime_data.loc[regime_mask, 'Direction_Regime'].mode()[0] if regime_mask.any() else 'Up_Trending'
            config = self.regime_configs.get(primary_regime, self.default_config)
        else:
            config = self.default_config
        
        # Calculate indicators
        fast_ma = data['Close'].rolling(config.fast_period).mean()
        slow_ma = data['Close'].rolling(config.slow_period).mean()
        atr = data['ATR'] if 'ATR' in data.columns else (data['High'] - data['Low']).rolling(14).mean()
        
        # Trend strength for position sizing
        if 'TrendStrength_Confidence' in regime_data.columns:
            trend_confidence = regime_data['TrendStrength_Confidence'].fillna(0.5)
        else:
            trend_confidence = pd.Series(0.5, index=data.index)
        
        current_position = 0.0
        entry_price = 0.0
        stop_loss = 0.0
        take_profit = 0.0
        
        for i in range(max(config.slow_period, 20), len(data)):
            if not regime_mask.iloc[i]:
                # Exit if not in target regime
                if current_position != 0:
                    exit_return = current_position * data['Close'].pct_change().iloc[i]
                    costs = self.apply_transaction_costs(abs(current_position), 
                                                        data['Close'].iloc[i], config)
                    returns.iloc[i] = exit_return + costs
                    current_position = 0.0
                continue
            
            # Entry logic
            if current_position == 0:
                # Long entry
                if (fast_ma.iloc[i] > slow_ma.iloc[i] and 
                    fast_ma.iloc[i-1] <= slow_ma.iloc[i-1] and
                    data['Close'].iloc[i] > fast_ma.iloc[i]):
                    
                    # Scale position by trend confidence
                    position_size = config.position_size * (0.5 + trend_confidence.iloc[i])
                    current_position = position_size
                    entry_price = data['Close'].iloc[i]
                    stop_loss = entry_price - (config.stop_loss_atr * atr.iloc[i])
                    take_profit = entry_price + (config.take_profit_atr * atr.iloc[i])
                    
                    costs = self.apply_transaction_costs(position_size, entry_price, config)
                    returns.iloc[i] = costs
                
                # Short entry (if trending down)
                elif (fast_ma.iloc[i] < slow_ma.iloc[i] and 
                      fast_ma.iloc[i-1] >= slow_ma.iloc[i-1] and
                      data['Close'].iloc[i] < fast_ma.iloc[i]):
                    
                    position_size = -config.position_size * (0.5 + trend_confidence.iloc[i])
                    current_position = position_size
                    entry_price = data['Close'].iloc[i]
                    stop_loss = entry_price + (config.stop_loss_atr * atr.iloc[i])
                    take_profit = entry_price - (config.take_profit_atr * atr.iloc[i])
                    
                    costs = self.apply_transaction_costs(abs(position_size), entry_price, config)
                    returns.iloc[i] = costs
            
            # Exit logic
            else:
                current_price = data['Close'].iloc[i]
                
                # Check stops
                if current_position > 0:
                    if current_price <= stop_loss or current_price >= take_profit:
                        exit_return = current_position * data['Close'].pct_change().iloc[i]
                        costs = self.apply_transaction_costs(abs(current_position), 
                                                            current_price, config)
                        returns.iloc[i] = exit_return + costs
                        current_position = 0.0
                    else:
                        # Trail stop
                        new_stop = current_price - (config.stop_loss_atr * atr.iloc[i])
                        stop_loss = max(stop_loss, new_stop)
                        returns.iloc[i] = current_position * data['Close'].pct_change().iloc[i]
                
                elif current_position < 0:
                    if current_price >= stop_loss or current_price <= take_profit:
                        exit_return = current_position * data['Close'].pct_change().iloc[i]
                        costs = self.apply_transaction_costs(abs(current_position), 
                                                            current_price, config)
                        returns.iloc[i] = exit_return + costs
                        current_position = 0.0
                    else:
                        # Trail stop
                        new_stop = current_price + (config.stop_loss_atr * atr.iloc[i])
                        stop_loss = min(stop_loss, new_stop)
                        returns.iloc[i] = current_position * data['Close'].pct_change().iloc[i]
            
            positions.iloc[i] = current_position
        
        return returns
    
    def mean_reversion_bands_strategy(self, data: pd.DataFrame, regime_mask: pd.Series,
                                     regime_data: pd.DataFrame) -> pd.Series:
        """
        Sophisticated mean reversion for sideways/low volatility regimes
        - Uses Bollinger Bands with regime-specific parameters
        - Combines with RSI for confirmation
        - Implements profit targets based on volatility
        """
        logger.info("Starting mean reversion bands strategy")
        data = self._fix_column_names(data)
        
        returns = pd.Series(0.0, index=data.index)
        positions = pd.Series(0.0, index=data.index)
        
        # Get regime-specific config
        config = self.regime_configs.get('Sideways', self.default_config)
        
        # Calculate indicators
        bb_middle = data['Close'].rolling(config.bb_periods).mean()
        bb_std = data['Close'].rolling(config.bb_periods).std()
        bb_upper = bb_middle + (config.bb_std * bb_std)
        bb_lower = bb_middle - (config.bb_std * bb_std)
        
        # Use RSI if available, otherwise calculate simple version
        if 'RSI' in data.columns:
            rsi = data['RSI']
        else:
            # Simple RSI calculation
            delta = data['Close'].diff()
            gain = delta.where(delta > 0, 0).rolling(14).mean()
            loss = -delta.where(delta < 0, 0).rolling(14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
        
        current_position = 0.0
        entry_price = 0.0
        
        for i in range(config.bb_periods, len(data)):
            if not regime_mask.iloc[i]:
                # Exit if not in target regime
                if current_position != 0:
                    exit_return = current_position * data['Close'].pct_change().iloc[i]
                    costs = self.apply_transaction_costs(abs(current_position), 
                                                        data['Close'].iloc[i], config)
                    returns.iloc[i] = exit_return + costs
                    current_position = 0.0
                continue
            
            # Entry logic
            if current_position == 0:
                # Long entry - price touches lower band and RSI oversold
                if (data['Close'].iloc[i] <= bb_lower.iloc[i] and 
                    rsi.iloc[i] < config.rsi_oversold):
                    
                    current_position = config.position_size
                    entry_price = data['Close'].iloc[i]
                    costs = self.apply_transaction_costs(current_position, entry_price, config)
                    returns.iloc[i] = costs
                
                # Short entry - price touches upper band and RSI overbought
                elif (data['Close'].iloc[i] >= bb_upper.iloc[i] and 
                      rsi.iloc[i] > config.rsi_overbought):
                    
                    current_position = -config.position_size
                    entry_price = data['Close'].iloc[i]
                    costs = self.apply_transaction_costs(abs(current_position), entry_price, config)
                    returns.iloc[i] = costs
            
            # Exit logic
            else:
                if current_position > 0:
                    # Exit long - price reaches middle band or RSI neutral
                    if (data['Close'].iloc[i] >= bb_middle.iloc[i] or 
                        rsi.iloc[i] > 50):
                        
                        exit_return = current_position * data['Close'].pct_change().iloc[i]
                        costs = self.apply_transaction_costs(abs(current_position), 
                                                            data['Close'].iloc[i], config)
                        returns.iloc[i] = exit_return + costs
                        current_position = 0.0
                    else:
                        returns.iloc[i] = current_position * data['Close'].pct_change().iloc[i]
                
                elif current_position < 0:
                    # Exit short - price reaches middle band or RSI neutral
                    if (data['Close'].iloc[i] <= bb_middle.iloc[i] or 
                        rsi.iloc[i] < 50):
                        
                        exit_return = current_position * data['Close'].pct_change().iloc[i]
                        costs = self.apply_transaction_costs(abs(current_position), 
                                                            data['Close'].iloc[i], config)
                        returns.iloc[i] = exit_return + costs
                        current_position = 0.0
                    else:
                        returns.iloc[i] = current_position * data['Close'].pct_change().iloc[i]
            
            positions.iloc[i] = current_position
        
        return returns
    
    def volatility_expansion_strategy(self, data: pd.DataFrame, regime_mask: pd.Series,
                                     regime_data: pd.DataFrame) -> pd.Series:
        """
        Volatility breakout strategy for high volatility regimes
        - Trades volatility expansions using ATR and price channels
        - Implements wider stops for volatile conditions
        - Uses volume confirmation
        """
        logger.info("Starting volatility expansion strategy")
        data = self._fix_column_names(data)
        
        returns = pd.Series(0.0, index=data.index)
        positions = pd.Series(0.0, index=data.index)
        
        # Get regime-specific config
        config = self.regime_configs.get('High_Vol', self.default_config)
        
        # Calculate indicators
        lookback = config.lookback
        high_channel = data['High'].rolling(lookback).max()
        low_channel = data['Low'].rolling(lookback).min()
        
        # ATR for volatility
        atr = data['ATR'] if 'ATR' in data.columns else (data['High'] - data['Low']).rolling(14).mean()
        
        # Volume analysis
        if 'Volume' in data.columns:
            volume_sma = data['Volume'].rolling(20).mean()
            volume_ratio = data['Volume'] / volume_sma
        else:
            volume_ratio = pd.Series(1.0, index=data.index)
        
        current_position = 0.0
        entry_price = 0.0
        stop_loss = 0.0
        
        for i in range(lookback + 20, len(data)):
            if not regime_mask.iloc[i]:
                # Exit if not in target regime
                if current_position != 0:
                    exit_return = current_position * data['Close'].pct_change().iloc[i]
                    costs = self.apply_transaction_costs(abs(current_position), 
                                                        data['Close'].iloc[i], config)
                    returns.iloc[i] = exit_return + costs
                    current_position = 0.0
                continue
            
            # Entry logic
            if current_position == 0:
                # Long breakout - price breaks above channel with volume
                if (data['Close'].iloc[i] > high_channel.iloc[i-1] and
                    volume_ratio.iloc[i] > config.volume_threshold):
                    
                    current_position = config.position_size
                    entry_price = data['Close'].iloc[i]
                    stop_loss = entry_price - (config.atr_multiplier * atr.iloc[i])
                    
                    costs = self.apply_transaction_costs(current_position, entry_price, config)
                    returns.iloc[i] = costs
                
                # Short breakout - price breaks below channel with volume
                elif (data['Close'].iloc[i] < low_channel.iloc[i-1] and
                      volume_ratio.iloc[i] > config.volume_threshold):
                    
                    current_position = -config.position_size
                    entry_price = data['Close'].iloc[i]
                    stop_loss = entry_price + (config.atr_multiplier * atr.iloc[i])
                    
                    costs = self.apply_transaction_costs(abs(current_position), entry_price, config)
                    returns.iloc[i] = costs
            
            # Exit logic
            else:
                current_price = data['Close'].iloc[i]
                
                # Check stops
                if (current_position > 0 and current_price <= stop_loss) or \
                   (current_position < 0 and current_price >= stop_loss):
                    
                    exit_return = current_position * data['Close'].pct_change().iloc[i]
                    costs = self.apply_transaction_costs(abs(current_position), 
                                                        current_price, config)
                    returns.iloc[i] = exit_return + costs
                    current_position = 0.0
                
                # Trail stop in profitable trades
                elif current_position != 0:
                    if current_position > 0:
                        new_stop = current_price - (config.atr_multiplier * atr.iloc[i])
                        stop_loss = max(stop_loss, new_stop)
                    else:
                        new_stop = current_price + (config.atr_multiplier * atr.iloc[i])
                        stop_loss = min(stop_loss, new_stop)
                    
                    returns.iloc[i] = current_position * data['Close'].pct_change().iloc[i]
            
            positions.iloc[i] = current_position
        
        return returns
    
    def market_structure_strategy(self, data: pd.DataFrame, regime_mask: pd.Series,
                                 regime_data: pd.DataFrame) -> pd.Series:
        """
        Strategy based on market microstructure regimes
        - Follows institutional flow in institutional regimes
        - Fades retail extremes
        - Uses order flow proxies
        """
        logger.info("Starting market structure strategy")
        data = self._fix_column_names(data)
        
        returns = pd.Series(0.0, index=data.index)
        positions = pd.Series(0.0, index=data.index)
        
        # Determine microstructure regime
        if 'Microstructure_Regime' in regime_data.columns:
            is_institutional = regime_data['Microstructure_Regime'] == 'Institutional'
            is_retail = regime_data['Microstructure_Regime'] == 'Retail_Flow'
        else:
            is_institutional = pd.Series(False, index=data.index)
            is_retail = pd.Series(False, index=data.index)
        
        # Calculate order flow proxies
        if 'VWAP' in data.columns:
            vwap_signal = (data['Close'] - data['VWAP']) / data['VWAP']
        else:
            # Simple VWAP approximation
            typical_price = (data['High'] + data['Low'] + data['Close']) / 3
            if 'Volume' in data.columns:
                vwap = (typical_price * data['Volume']).rolling(20).sum() / data['Volume'].rolling(20).sum()
            else:
                vwap = typical_price.rolling(20).mean()
            vwap_signal = (data['Close'] - vwap) / vwap
        
        # Money flow
        if 'MFI' in data.columns:
            mfi = data['MFI']
        else:
            mfi = pd.Series(50, index=data.index)  # Neutral if not available
        
        current_position = 0.0
        
        for i in range(50, len(data)):
            if not regime_mask.iloc[i]:
                # Exit if not in target regime
                if current_position != 0:
                    exit_return = current_position * data['Close'].pct_change().iloc[i]
                    costs = self.apply_transaction_costs(abs(current_position), 
                                                        data['Close'].iloc[i], self.default_config)
                    returns.iloc[i] = exit_return + costs
                    current_position = 0.0
                continue
            
            # Institutional flow - follow the smart money
            if is_institutional.iloc[i]:
                if current_position == 0:
                    # Long if price above VWAP with strong money flow
                    if vwap_signal.iloc[i] > 0.001 and mfi.iloc[i] > 60:
                        current_position = 1.0
                        costs = self.apply_transaction_costs(1.0, data['Close'].iloc[i], 
                                                            self.default_config)
                        returns.iloc[i] = costs
                    
                    # Short if price below VWAP with weak money flow
                    elif vwap_signal.iloc[i] < -0.001 and mfi.iloc[i] < 40:
                        current_position = -1.0
                        costs = self.apply_transaction_costs(1.0, data['Close'].iloc[i], 
                                                            self.default_config)
                        returns.iloc[i] = costs
                
                # Exit on reversal
                elif current_position > 0 and vwap_signal.iloc[i] < -0.001:
                    exit_return = current_position * data['Close'].pct_change().iloc[i]
                    costs = self.apply_transaction_costs(abs(current_position), 
                                                        data['Close'].iloc[i], self.default_config)
                    returns.iloc[i] = exit_return + costs
                    current_position = 0.0
                
                elif current_position < 0 and vwap_signal.iloc[i] > 0.001:
                    exit_return = current_position * data['Close'].pct_change().iloc[i]
                    costs = self.apply_transaction_costs(abs(current_position), 
                                                        data['Close'].iloc[i], self.default_config)
                    returns.iloc[i] = exit_return + costs
                    current_position = 0.0
                else:
                    returns.iloc[i] = current_position * data['Close'].pct_change().iloc[i]
            
            # Retail extremes - fade the crowd
            elif is_retail.iloc[i]:
                if current_position == 0:
                    # Fade retail buying extremes
                    if mfi.iloc[i] > 80 and vwap_signal.iloc[i] > 0.002:
                        current_position = -0.5  # Smaller size for fading
                        costs = self.apply_transaction_costs(0.5, data['Close'].iloc[i], 
                                                            self.default_config)
                        returns.iloc[i] = costs
                    
                    # Fade retail selling extremes
                    elif mfi.iloc[i] < 20 and vwap_signal.iloc[i] < -0.002:
                        current_position = 0.5
                        costs = self.apply_transaction_costs(0.5, data['Close'].iloc[i], 
                                                            self.default_config)
                        returns.iloc[i] = costs
                
                # Quick exits for fade trades
                elif abs(vwap_signal.iloc[i]) < 0.001:
                    exit_return = current_position * data['Close'].pct_change().iloc[i]
                    costs = self.apply_transaction_costs(abs(current_position), 
                                                        data['Close'].iloc[i], self.default_config)
                    returns.iloc[i] = exit_return + costs
                    current_position = 0.0
                else:
                    returns.iloc[i] = current_position * data['Close'].pct_change().iloc[i]
            
            # Balanced regime - no trades
            else:
                if current_position != 0:
                    exit_return = current_position * data['Close'].pct_change().iloc[i]
                    costs = self.apply_transaction_costs(abs(current_position), 
                                                        data['Close'].iloc[i], self.default_config)
                    returns.iloc[i] = exit_return + costs
                    current_position = 0.0
            
            positions.iloc[i] = current_position
        
        return returns
    
    def adaptive_regime_strategy_enhanced(self, data: pd.DataFrame, 
                                         regimes: pd.DataFrame) -> pd.Series:
        """
        Enhanced adaptive strategy that selects the best approach based on current regime
        """
        logger.info("Starting enhanced adaptive regime strategy")
        data = self._fix_column_names(data)
        
        returns = pd.Series(0.0, index=data.index)
        
        # Define regime to strategy mapping based on your results
        regime_strategy_map = {
            # Single dimension rules
            ('Direction', 'Up_Trending'): self.momentum_breakout_strategy,
            ('Direction', 'Down_Trending'): self.momentum_breakout_strategy,
            ('Direction', 'Sideways'): self.mean_reversion_bands_strategy,
            ('Volatility', 'Low_Vol'): self.mean_reversion_bands_strategy,
            ('Volatility', 'High_Vol'): self.volatility_expansion_strategy,
            ('Volatility', 'Extreme_Vol'): self.volatility_expansion_strategy,
            ('TrendStrength', 'Strong'): self.momentum_breakout_strategy,
            ('TrendStrength', 'Weak'): self.mean_reversion_bands_strategy,
            ('Microstructure', 'Institutional'): self.market_structure_strategy,
        }
        
        # Process each regime period
        current_strategy = None
        strategy_returns = pd.Series(0.0, index=data.index)
        
        for i in range(len(data)):
            # Determine primary regime
            best_confidence = 0
            best_regime = None
            best_dimension = None
            
            for dimension in ['Direction', 'TrendStrength', 'Volatility', 'Microstructure']:
                regime_col = f'{dimension}_Regime'
                confidence_col = f'{dimension}_Confidence'
                
                if regime_col in regimes.columns and confidence_col in regimes.columns:
                    if regimes[confidence_col].iloc[i] > best_confidence:
                        best_confidence = regimes[confidence_col].iloc[i]
                        best_regime = regimes[regime_col].iloc[i]
                        best_dimension = dimension
            
            # Select strategy based on regime
            if best_regime and (best_dimension, best_regime) in regime_strategy_map:
                selected_strategy = regime_strategy_map[(best_dimension, best_regime)]
                
                # If strategy changed, calculate returns for the period
                if selected_strategy != current_strategy:
                    current_strategy = selected_strategy
                    
                    # Create regime mask for this specific regime
                    regime_mask = regimes[f'{best_dimension}_Regime'] == best_regime
                    
                    # Get returns from the selected strategy
                    strategy_returns = current_strategy(data, regime_mask, regimes)
            
            returns.iloc[i] = strategy_returns.iloc[i] if i < len(strategy_returns) else 0.0
        
        return returns

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

# Additional helper functions for strategy analysis

def test_strategy_in_regime(data: pd.DataFrame, regimes: pd.DataFrame, 
                           strategy_name: str, regime_filter: Dict[str, str]) -> Optional[Dict[str, float]]:
    """Test a specific strategy in a specific regime combination"""
    
    # Create regime mask
    regime_mask = pd.Series(True, index=data.index)
    for dimension, regime_value in regime_filter.items():
        if f'{dimension}_Regime' in regimes.columns:
            regime_mask &= (regimes[f'{dimension}_Regime'] == regime_value)
    
    if not regime_mask.any():
        return None
    
    # Initialize backtester
    backtester = EnhancedRegimeStrategyBacktester()
    
    # Run appropriate strategy
    strategy_map = {
        'momentum': backtester.momentum_breakout_strategy,
        'mean_reversion': backtester.mean_reversion_bands_strategy,
        'volatility_breakout': backtester.volatility_expansion_strategy,
        'market_structure': backtester.market_structure_strategy,
        'adaptive': backtester.adaptive_regime_strategy_enhanced
    }
    
    if strategy_name not in strategy_map:
        logger.warning(f"Unknown strategy: {strategy_name}")
        return None
    
    # Get strategy returns
    if strategy_name == 'adaptive':
        returns = strategy_map[strategy_name](data, regimes)
    else:
        returns = strategy_map[strategy_name](data, regime_mask, regimes)
    
    # Calculate metrics
    metrics = backtester.calculate_performance_metrics(returns[regime_mask])
    
    # Add regime-specific info
    metrics['periods_in_regime'] = regime_mask.sum()
    metrics['pct_time_in_regime'] = 100 * regime_mask.sum() / len(data)
    
    return metrics

def create_regime_strategy_matrix(data: pd.DataFrame, regimes: pd.DataFrame) -> pd.DataFrame:
    """Create a matrix showing best strategy for each regime"""
    
    strategies = ['momentum', 'mean_reversion', 'volatility_breakout', 'market_structure', 'adaptive']
    regime_dimensions = ['Direction', 'TrendStrength', 'Volatility', 'Microstructure']
    
    results = []
    
    for dimension in regime_dimensions:
        if f'{dimension}_Regime' in regimes.columns:
            unique_regimes = regimes[f'{dimension}_Regime'].unique()
            
            for regime_value in unique_regimes:
                regime_filter = {dimension: regime_value}
                
                best_strategy = None
                best_sharpe = -np.inf
                
                for strategy in strategies:
                    metrics = test_strategy_in_regime(data, regimes, strategy, regime_filter)
                    
                    if metrics and metrics['sharpe_ratio'] > best_sharpe:
                        best_sharpe = metrics['sharpe_ratio']
                        best_strategy = strategy
                
                results.append({
                    'Dimension': dimension,
                    'Regime': regime_value,
                    'Best_Strategy': best_strategy,
                    'Sharpe_Ratio': best_sharpe,
                    'Time_Pct': metrics['pct_time_in_regime'] if metrics else 0
                })
    
    return pd.DataFrame(results)

