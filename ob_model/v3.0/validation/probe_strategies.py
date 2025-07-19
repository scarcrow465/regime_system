#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import sys
import os
BASE_DIR = r"C:\Users\rs\GitProjects\regime_system\ob_model\v3.0"  # Hardcode if not importing settings yet
sys.path.append(BASE_DIR)
import pandas as pd
import numpy as np
import pandas_ta as ta
from utils.logger import log_message, progress_bar
from config.settings import DEBUG_LEVEL, BASE_DIR, DATA_PATH, TEST_SLICE
from core.regime_classifier import fit_gmm, add_session_labels, smooth_regime_labels
from core.indicators import select_and_compute_indicators_live
from core.data_loader import load_csv_data
from rich.table import Table
from datetime import datetime
from core.helpers import console
from utils.metrics import compute_persistence
from rich.console import Console
from rich.panel import Panel

# Account and risk configuration
ACCOUNT_SIZE = 50000  # Starting account size for drawdown calculations
CONTRACT_SIZE = 1  # Fixed number of contracts per trade
POINT_VALUE = 20  # Dollar value per point for NQ
TICK_SIZE = 0.25  # Minimum price movement
TICK_VALUE = 5  # Dollar value per tick ($20/point * 0.25)

# Strategy parameters for different scopes
STRATEGY_PARAMS = {
    'trend': {
        'normal': {'break_len': 20, 'hold_bars': 5, 'stop_mult': 2, 'atr_len': 14},
        'fast': {'break_len': 10, 'hold_bars': 2, 'stop_mult': 1, 'atr_len': 7},
        'slow': {'break_len': 50, 'hold_bars': 20, 'stop_mult': 3, 'atr_len': 20}
    },
    'reversion': {
        'normal': {'rsi_len': 14, 'rsi_low': 40, 'rsi_high': 60, 'rsi_exit_low': 50, 'rsi_exit_high': 50, 'hold_bars': 3, 'bb_std': 2, 'atr_len': 14},
        'fast': {'rsi_len': 7, 'rsi_low': 30, 'rsi_high': 70, 'rsi_exit_low': 40, 'rsi_exit_high': 60, 'hold_bars': 2, 'bb_std': 2, 'atr_len': 7},
        'slow': {'rsi_len': 20, 'rsi_low': 45, 'rsi_high': 55, 'rsi_exit_low': 55, 'rsi_exit_high': 45, 'hold_bars': 10, 'bb_std': 2.5, 'atr_len': 20}
    },
    'ma_cross': {
        'normal': {'fast_len': 50, 'slow_len': 200, 'hold_bars': 5, 'atr_len': 14},
        'fast': {'fast_len': 20, 'slow_len': 50, 'hold_bars': 3, 'atr_len': 7},
        'slow': {'fast_len': 100, 'slow_len': 200, 'hold_bars': 20, 'atr_len': 20}
    },
    'bb_fade': {
        'normal': {'bb_len': 20, 'bb_std': 2, 'hold_bars': 3, 'atr_len': 14},
        'fast': {'bb_len': 10, 'bb_std': 1.5, 'hold_bars': 2, 'atr_len': 7},
        'slow': {'bb_len': 50, 'bb_std': 2.5, 'hold_bars': 10, 'atr_len': 20}
    }
}

def ma_crossover_strategy(df, entry_bar, scope='normal'):
    if not isinstance(df, pd.DataFrame) or not all(col in df.columns for col in ['high', 'low', 'close']):
        return 0, 0, 0
    params = STRATEGY_PARAMS['ma_cross'][scope]
    if entry_bar < params['slow_len'] or entry_bar + params['hold_bars'] >= len(df):
        return 0, 0, 0
    
    # Calculate ATR
    tr_list = []
    for j in range(max(1, entry_bar-params['atr_len']), entry_bar):
        high_low = df['high'].iloc[j] - df['low'].iloc[j]
        high_close = abs(df['high'].iloc[j] - df['close'].iloc[j-1])
        low_close = abs(df['low'].iloc[j] - df['close'].iloc[j-1])
        tr_list.append(max(high_low, high_close, low_close))
    atr_value = sum(tr_list) / len(tr_list) if tr_list else 0
    
    # Calculate MAs using ONLY data up to entry_bar
    ma_fast = df['close'].iloc[max(0, entry_bar-params['fast_len']):entry_bar].mean()
    ma_slow = df['close'].iloc[max(0, entry_bar-params['slow_len']):entry_bar].mean()
    
    if df['close'].iloc[entry_bar] > ma_fast > ma_slow:
        entry_price = df['close'].iloc[entry_bar]
        exit_bar = min(entry_bar + params['hold_bars'], len(df) - 1)
        exit_price = df['close'].iloc[exit_bar]
        pnl = (exit_price - entry_price) * POINT_VALUE * CONTRACT_SIZE
        return pnl, atr_value, entry_price
    return 0, 0, 0

def bb_fade_strategy(df, entry_bar, scope='normal'):
    if not isinstance(df, pd.DataFrame) or not all(col in df.columns for col in ['high', 'low', 'close']):
        return 0, 0, 0
    params = STRATEGY_PARAMS['bb_fade'][scope]
    if entry_bar < params['bb_len'] or entry_bar + params['hold_bars'] >= len(df):
        return 0, 0, 0
    
    # Calculate ATR
    tr_list = []
    for j in range(max(1, entry_bar-params['atr_len']), entry_bar):
        high_low = df['high'].iloc[j] - df['low'].iloc[j]
        high_close = abs(df['high'].iloc[j] - df['close'].iloc[j-1])
        low_close = abs(df['low'].iloc[j] - df['close'].iloc[j-1])
        tr_list.append(max(high_low, high_close, low_close))
    atr_value = sum(tr_list) / len(tr_list) if tr_list else 0
    
    # Calculate BB using ONLY historical data
    close_slice = df['close'].iloc[max(0, entry_bar-params['bb_len']):entry_bar]
    sma = close_slice.mean()
    std = close_slice.std()
    lower_bb = sma - (params['bb_std'] * std)
    
    if df['low'].iloc[entry_bar] <= lower_bb:
        entry_price = df['close'].iloc[entry_bar]
        exit_bar = min(entry_bar + params['hold_bars'], len(df) - 1)
        exit_price = df['close'].iloc[exit_bar]
        pnl = (exit_price - entry_price) * POINT_VALUE * CONTRACT_SIZE
        return pnl, atr_value, entry_price
    return 0, 0, 0

def trend_following_strategy(df, entry_bar, scope='normal'):
    if not isinstance(df, pd.DataFrame) or not all(col in df.columns for col in ['high', 'low', 'close']):
        return 0, 0, 0
    params = STRATEGY_PARAMS['trend'][scope]
    if entry_bar < params['break_len'] or entry_bar + params['hold_bars'] >= len(df):
        return 0, 0, 0
    
    # Calculate ATR
    tr_list = []
    for j in range(max(1, entry_bar-params['atr_len']), entry_bar):
        high_low = df['high'].iloc[j] - df['low'].iloc[j]
        high_close = abs(df['high'].iloc[j] - df['close'].iloc[j-1])
        low_close = abs(df['low'].iloc[j] - df['close'].iloc[j-1])
        tr_list.append(max(high_low, high_close, low_close))
    atr_value = sum(tr_list) / len(tr_list) if tr_list else 0
    
    # Entry logic
    high_break = df['high'].iloc[entry_bar-params['break_len']:entry_bar].max()
    if df['close'].iloc[entry_bar] > high_break:
        entry_price = df['close'].iloc[entry_bar]
        exit_bar = min(entry_bar + params['hold_bars'], len(df) - 1)
        exit_price = df['close'].iloc[exit_bar]
        stop_loss = entry_price - params['stop_mult'] * atr_value
        for i in range(entry_bar + 1, exit_bar + 1):
            if df['low'].iloc[i] <= stop_loss:
                exit_price = stop_loss
                break
        pnl = (exit_price - entry_price) * POINT_VALUE * CONTRACT_SIZE
        return pnl, atr_value, entry_price
    return 0, 0, 0

def mean_reversion_strategy(df, entry_bar, scope='normal'):
    if not isinstance(df, pd.DataFrame) or not all(col in df.columns for col in ['high', 'low', 'close']):
        return 0, 0, 0
    params = STRATEGY_PARAMS['reversion'][scope]
    if entry_bar < params['rsi_len'] +1 or entry_bar + params['hold_bars'] >= len(df):
        return 0, 0, 0
    
    # Calculate ATR
    tr_list = []
    for j in range(max(1, entry_bar-params['atr_len']), entry_bar):
        high_low = df['high'].iloc[j] - df['low'].iloc[j]
        high_close = abs(df['high'].iloc[j] - df['close'].iloc[j-1])
        low_close = abs(df['low'].iloc[j] - df['close'].iloc[j-1])
        tr_list.append(max(high_low, high_close, low_close))
    atr_value = sum(tr_list) / len(tr_list) if tr_list else 0
    
    # Calculate RSI manually using only historical data
    close_slice = df['close'].iloc[max(0, entry_bar-params['rsi_len']):entry_bar+1]
    deltas = close_slice.diff()
    gains = deltas.where(deltas > 0, 0)
    losses = -deltas.where(deltas < 0, 0)
    avg_gain = gains.iloc[1:].mean()
    avg_loss = losses.iloc[1:].mean()
    rs = avg_gain / avg_loss if avg_loss != 0 else 0
    rsi_value = 100 - (100 / (1 + rs))
    
    # Calculate BB using only historical data
    bb_slice = df['close'].iloc[max(0, entry_bar-params.get('bb_len', 20)):entry_bar]
    sma = bb_slice.mean()
    std = bb_slice.std()
    lower_bb = sma - (params['bb_std'] * std)
    middle_bb = sma
    
    # Entry conditions
    price_pct_from_lower = (df['close'].iloc[entry_bar] - lower_bb) / (middle_bb - lower_bb) if middle_bb != lower_bb else 1
    
    if (rsi_value < params['rsi_low'] and price_pct_from_lower < 0.2):
        entry_price = df['close'].iloc[entry_bar]
        
        # Exit - calculate RSI for each exit bar
        for i in range(entry_bar + 1, min(entry_bar + params['hold_bars'] +1, len(df))):
            exit_close_slice = df['close'].iloc[max(0, i-params['rsi_len']):i+1]
            exit_deltas = exit_close_slice.diff()
            exit_gains = exit_deltas.where(exit_deltas > 0, 0)
            exit_losses = -exit_deltas.where(exit_deltas < 0, 0)
            exit_avg_gain = exit_gains.iloc[1:].mean()
            exit_avg_loss = exit_losses.iloc[1:].mean()
            exit_rs = exit_avg_gain / exit_avg_loss if exit_avg_loss != 0 else 0
            exit_rsi = 100 - (100 / (1 + exit_rs))
            
            if exit_rsi > params['rsi_exit_low']:
                pnl = (df['close'].iloc[i] - entry_price) * POINT_VALUE * CONTRACT_SIZE
                return pnl, atr_value, entry_price
        
        # Exit after hold_bars
        pnl = (df['close'].iloc[min(entry_bar + params['hold_bars'], len(df) - 1)] - entry_price) * POINT_VALUE * CONTRACT_SIZE
        return pnl, atr_value, entry_price
    
    return 0, 0, 0

def trend_following_strategy_short(df, entry_bar, scope='normal'):
    if not isinstance(df, pd.DataFrame) or not all(col in df.columns for col in ['high', 'low', 'close']):
        return 0, 0, 0
    params = STRATEGY_PARAMS['trend'][scope]
    if entry_bar < params['break_len'] or entry_bar + params['hold_bars'] >= len(df):
        return 0, 0, 0
    
    # Calculate ATR
    tr_list = []
    for j in range(max(1, entry_bar-params['atr_len']), entry_bar):
        high_low = df['high'].iloc[j] - df['low'].iloc[j]
        high_close = abs(df['high'].iloc[j] - df['close'].iloc[j-1])
        low_close = abs(df['low'].iloc[j] - df['close'].iloc[j-1])
        tr_list.append(max(high_low, high_close, low_close))
    atr_value = sum(tr_list) / len(tr_list) if tr_list else 0
    
    # Entry logic - opposite of long
    low_break = df['low'].iloc[entry_bar-params['break_len']:entry_bar].min()
    if df['close'].iloc[entry_bar] < low_break:
        entry_price = df['close'].iloc[entry_bar]
        exit_bar = min(entry_bar + params['hold_bars'], len(df) - 1)
        exit_price = df['close'].iloc[exit_bar]
        stop_loss = entry_price + params['stop_mult'] * atr_value
        for i in range(entry_bar + 1, exit_bar + 1):
            if df['high'].iloc[i] >= stop_loss:
                exit_price = stop_loss
                break
        pnl = (entry_price - exit_price) * POINT_VALUE * CONTRACT_SIZE
        return pnl, atr_value, entry_price
    return 0, 0, 0

def mean_reversion_strategy_short(df, entry_bar, scope='normal'):
    if not isinstance(df, pd.DataFrame) or not all(col in df.columns for col in ['high', 'low', 'close']):
        return 0, 0, 0
    params = STRATEGY_PARAMS['reversion'][scope]
    if entry_bar < params['rsi_len'] +1 or entry_bar + params['hold_bars'] >= len(df):
        return 0, 0, 0
    
    # Calculate ATR
    tr_list = []
    for j in range(max(1, entry_bar-params['atr_len']), entry_bar):
        high_low = df['high'].iloc[j] - df['low'].iloc[j]
        high_close = abs(df['high'].iloc[j] - df['close'].iloc[j-1])
        low_close = abs(df['low'].iloc[j] - df['close'].iloc[j-1])
        tr_list.append(max(high_low, high_close, low_close))
    atr_value = sum(tr_list) / len(tr_list) if tr_list else 0
    
    # Calculate RSI manually using only historical data
    close_slice = df['close'].iloc[max(0, entry_bar-params['rsi_len']):entry_bar+1]
    deltas = close_slice.diff()
    gains = deltas.where(deltas > 0, 0)
    losses = -deltas.where(deltas < 0, 0)
    avg_gain = gains.iloc[1:].mean()
    avg_loss = losses.iloc[1:].mean()
    rs = avg_gain / avg_loss if avg_loss != 0 else 0
    rsi_value = 100 - (100 / (1 + rs))
    
    # Calculate BB using only historical data
    bb_slice = df['close'].iloc[max(0, entry_bar-params.get('bb_len', 20)):entry_bar]
    sma = bb_slice.mean()
    std = bb_slice.std()
    upper_bb = sma + (params['bb_std'] * std)
    middle_bb = sma
    
    # Entry conditions - opposite of long
    price_pct_from_upper = (upper_bb - df['close'].iloc[entry_bar]) / (upper_bb - middle_bb) if upper_bb != middle_bb else 1
    
    if (rsi_value > params['rsi_high'] and price_pct_from_upper < 0.2):
        entry_price = df['close'].iloc[entry_bar]
        
        # Exit - calculate RSI for each exit bar
        for i in range(entry_bar + 1, min(entry_bar + params['hold_bars'] +1, len(df))):
            exit_close_slice = df['close'].iloc[max(0, i-params['rsi_len']):i+1]
            exit_deltas = exit_close_slice.diff()
            exit_gains = exit_deltas.where(exit_deltas > 0, 0)
            exit_losses = -exit_deltas.where(exit_deltas < 0, 0)
            exit_avg_gain = exit_gains.iloc[1:].mean()
            exit_avg_loss = exit_losses.iloc[1:].mean()
            exit_rs = exit_avg_gain / exit_avg_loss if exit_avg_loss != 0 else 0
            exit_rsi = 100 - (100 / (1 + exit_rs))
            
            if exit_rsi < params['rsi_exit_high']:
                pnl = (entry_price - df['close'].iloc[i]) * POINT_VALUE * CONTRACT_SIZE
                return pnl, atr_value, entry_price
        
        # Exit after hold_bars
        pnl = (entry_price - df['close'].iloc[min(entry_bar + params['hold_bars'], len(df) - 1)]) * POINT_VALUE * CONTRACT_SIZE
        return pnl, atr_value, entry_price
    
    return 0, 0, 0

def ma_crossover_strategy_short(df, entry_bar, scope='normal'):
    if not isinstance(df, pd.DataFrame) or not all(col in df.columns for col in ['high', 'low', 'close']):
        return 0, 0, 0
    params = STRATEGY_PARAMS['ma_cross'][scope]
    if entry_bar < params['slow_len'] or entry_bar + params['hold_bars'] >= len(df):
        return 0, 0, 0
    
    # Calculate ATR
    tr_list = []
    for j in range(max(1, entry_bar-params['atr_len']), entry_bar):
        high_low = df['high'].iloc[j] - df['low'].iloc[j]
        high_close = abs(df['high'].iloc[j] - df['close'].iloc[j-1])
        low_close = abs(df['low'].iloc[j] - df['close'].iloc[j-1])
        tr_list.append(max(high_low, high_close, low_close))
    atr_value = sum(tr_list) / len(tr_list) if tr_list else 0
    
    # Calculate MAs using ONLY data up to entry_bar
    ma_fast = df['close'].iloc[max(0, entry_bar-params['fast_len']):entry_bar].mean()
    ma_slow = df['close'].iloc[max(0, entry_bar-params['slow_len']):entry_bar].mean()
    
    if df['close'].iloc[entry_bar] < ma_fast < ma_slow:
        entry_price = df['close'].iloc[entry_bar]
        exit_bar = min(entry_bar + params['hold_bars'], len(df) - 1)
        exit_price = df['close'].iloc[exit_bar]
        pnl = (entry_price - exit_price) * POINT_VALUE * CONTRACT_SIZE
        return pnl, atr_value, entry_price
    return 0, 0, 0

def bb_fade_strategy_short(df, entry_bar, scope='normal'):
    if not isinstance(df, pd.DataFrame) or not all(col in df.columns for col in ['high', 'low', 'close']):
        return 0, 0, 0
    params = STRATEGY_PARAMS['bb_fade'][scope]
    if entry_bar < params['bb_len'] or entry_bar + params['hold_bars'] >= len(df):
        return 0, 0, 0
    
    # Calculate ATR
    tr_list = []
    for j in range(max(1, entry_bar-params['atr_len']), entry_bar):
        high_low = df['high'].iloc[j] - df['low'].iloc[j]
        high_close = abs(df['high'].iloc[j] - df['close'].iloc[j-1])
        low_close = abs(df['low'].iloc[j] - df['close'].iloc[j-1])
        tr_list.append(max(high_low, high_close, low_close))
    atr_value = sum(tr_list) / len(tr_list) if tr_list else 0
    
    # Calculate BB using ONLY historical data
    close_slice = df['close'].iloc[max(0, entry_bar-params['bb_len']):entry_bar]
    sma = close_slice.mean()
    std = close_slice.std()
    upper_bb = sma + (params['bb_std'] * std)
    
    if df['high'].iloc[entry_bar] >= upper_bb:
        entry_price = df['close'].iloc[entry_bar]
        exit_bar = min(entry_bar + params['hold_bars'], len(df) - 1)
        exit_price = df['close'].iloc[exit_bar]
        pnl = (entry_price - exit_price) * POINT_VALUE * CONTRACT_SIZE
        return pnl, atr_value, entry_price
    return 0, 0, 0

def calculate_regime_characteristics(df, model, features, raw_features):
    """Pre-calculate regime characteristics to avoid repeated calls."""
    if not isinstance(df, pd.DataFrame) or not all(col in df.columns for col in ['high', 'low', 'close']):
        return {}
    # Handle dict of models
    if isinstance(model, dict):
        # Use voting to get labels
        class_labels = pd.DataFrame(index=features.index)
        for cls, model_info in model.items():
            if isinstance(model_info, tuple):
                cls_model, fitted_cols = model_info
                available_cols = [col for col in fitted_cols if col in features.columns]
                if available_cols:
                    class_labels[cls] = pd.Series(cls_model.predict(features[available_cols]), index=features.index)
        labels = class_labels.mode(axis=1)[0].astype(int) if not class_labels.empty else pd.Series(0, index=features.index)
    else:
        labels = pd.Series(model.predict(features), index=features.index)
    
    # Calculate average indicators per regime
    regime_stats = {}
    for regime in labels.unique():
        regime_data = raw_features[labels == regime]
        if len(regime_data) > 0:
            trend_value = regime_data[['EMA_20', 'EMA_50']].diff().mean().mean() if 'EMA_20' in regime_data and 'EMA_50' in regime_data else 0
            regime_stats[regime] = {
                'avg_trend': trend_value if not pd.isna(trend_value) else 0,
                'avg_volatility': regime_data['ATR_14'].mean() if 'ATR_14' in regime_data.columns else 0,
                'avg_rsi': regime_data['RSI_14'].mean() if 'RSI_14' in regime_data.columns else 50,
                'count': len(regime_data)
            }
    return regime_stats

def calculate_trade_metrics(trades, use_normalized=True):
    """Calculate comprehensive metrics for a list of trades"""
    if len(trades) == 0:
        return {
            'num_trades': 0,
            'total_pnl': 0,
            'avg_pnl': 0,
            'normalized_pnl': 0,
            'win_rate': 0,
            'profit_factor': 0,
            'max_drawdown': 0,
            'max_drawdown_pct': 0,
            'sharpe_ratio': 0,
            'sortino_ratio': 0,
            'calmar_ratio': 0,
            'avg_duration': 0,
            'risk_reward': 0,
            'expectancy': 0,
            'consecutive_losses': 0,
            'time_in_drawdown': 0
        }
    
    # Get raw PnLs for drawdown calculation
    raw_pnls = [t['pnl'] for t in trades]
    
    # Calculate drawdown on raw dollar amounts
    cumulative_pnl = np.cumsum(raw_pnls)
    equity_curve = ACCOUNT_SIZE + cumulative_pnl
    running_max = np.maximum.accumulate(equity_curve)
    drawdown_dollars = equity_curve - running_max
    drawdown_pct = (drawdown_dollars / running_max) * 100 if np.any(running_max > 0) else np.zeros_like(drawdown_dollars)
    max_drawdown_dollars = abs(min(drawdown_dollars)) if len(drawdown_dollars) > 0 else 0
    max_drawdown_pct = abs(min(drawdown_pct)) if len(drawdown_pct) > 0 else 0
    
    # Calculate consecutive losses
    consecutive_losses = 0
    current_losses = 0
    for pnl in raw_pnls:
        if pnl < 0:
            current_losses += 1
            consecutive_losses = max(consecutive_losses, current_losses)
        else:
            current_losses = 0
    
    # Calculate time in drawdown (percentage of time below peak)
    time_in_dd = np.sum(drawdown_dollars < 0) / len(drawdown_dollars) * 100 if len(drawdown_dollars) > 0 else 0
    
    # Use normalized or raw PnL for other metrics
    if use_normalized:
        pnls = [t['normalized_pnl'] for t in trades]
    else:
        pnls = raw_pnls
        
    total_pnl = sum(pnls)
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    
    win_rate = len(wins) / len(trades) * 100 if len(trades) > 0 else 0
    profit_factor = sum(wins) / abs(sum(losses)) if losses else float('inf')
    
    # Sharpe ratio
    if len(pnls) > 1:
        returns = pd.Series(pnls)
        sharpe_ratio = (returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0
        
        # Sortino ratio (downside deviation)
        downside_returns = returns[returns < 0]
        if len(downside_returns) > 1 and downside_returns.std() > 0:
            sortino_ratio = (returns.mean() / downside_returns.std() * np.sqrt(252))
        else:
            sortino_ratio = 0
            
        # Calmar ratio (annual return / max drawdown)
        annual_return = returns.mean() * 252
        calmar_ratio = (annual_return / (max_drawdown_pct / 100)) if max_drawdown_pct > 0 else 0
    else:
        sharpe_ratio = 0
        sortino_ratio = 0
        calmar_ratio = 0
    
    # Risk reward
    avg_win = np.mean(wins) if wins else 0
    avg_loss = abs(np.mean(losses)) if losses else 0
    risk_reward = avg_win / avg_loss if avg_loss > 0 else 0
    
    # Expectancy (average profit per trade)
    expectancy = total_pnl / len(trades) if len(trades) > 0 else 0
    
    return {
        'num_trades': len(trades),
        'total_pnl': sum(raw_pnls),  # Always raw PnL
        'normalized_pnl': total_pnl,  # Normalized total
        'avg_pnl': sum(raw_pnls) / len(trades) if len(trades) > 0 else 0,
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'max_drawdown': max_drawdown_dollars,
        'max_drawdown_pct': max_drawdown_pct,
        'sharpe_ratio': sharpe_ratio,
        'sortino_ratio': sortino_ratio,
        'calmar_ratio': calmar_ratio,
        'avg_duration': np.mean([t['duration'] for t in trades]),
        'risk_reward': risk_reward,
        'expectancy': expectancy,
        'consecutive_losses': consecutive_losses,
        'time_in_drawdown': time_in_dd
    }

def normalize_metrics_by_time(metrics, session_hours):
    """Normalize metrics by session duration"""
    if session_hours == 0:
        return metrics
    
    normalized = metrics.copy()
    # Normalize per-hour metrics
    normalized['trades_per_hour'] = metrics['num_trades'] / session_hours
    normalized['pnl_per_hour'] = metrics['total_pnl'] / session_hours
    
    return normalized

def normalize_trade_risk(pnl, entry_price):
    """
    Normalize trade PnL to % return on position value (fixed contract size)
    
    Args:
        pnl: Raw PnL from trade
        entry_price: Entry price for position value calculation
    
    Returns:
        Normalized PnL as percentage of position value
    """
    if entry_price > 0:
        position_value = entry_price * POINT_VALUE * CONTRACT_SIZE
        return_pct = (pnl / position_value) * 100 if position_value > 0 else 0
    else:
        return_pct = 0
    return return_pct

def run_strategy_probes(df, model):
    """Run probe strategies to validate regimes with enhanced metrics"""
    
    # Get features and predict regimes
    ind_df, raw_ind_df = select_and_compute_indicators_live(df)
    features = ind_df.select_dtypes(include=[np.number]).dropna()
    
    if len(features) < 100:
        log_message("Insufficient data for probes", 'error')
        return pd.DataFrame()
    
    # Predict regimes
    class_labels = pd.DataFrame(index=features.index)
    for cls, model_info in model.items():
        if isinstance(model_info, tuple):
            cls_model, fitted_cols = model_info
            available_cols = [col for col in fitted_cols if col in features.columns]
            if available_cols:
                class_labels[cls] = pd.Series(cls_model.predict(features[available_cols]), index=features.index)
    labels = class_labels.mode(axis=1)[0].astype(int) if not class_labels.empty else pd.Series(0, index=features.index)
    labels = smooth_regime_labels(labels, min_persistence=3)
    
    # Add to dataframe
    df_filtered = df.loc[labels.index].copy()
    df_filtered['regime'] = labels
    df_filtered['session'] = ind_df.loc[labels.index, 'refined_session']
    
    # Calculate regime characteristics
    regime_stats = calculate_regime_characteristics(df_filtered, model, features, raw_ind_df)
    
    # Define session hours
    session_hours = {
        'NY_Open': 2.5,
        'NY_Midday': 2.0,
        'NY_Afternoon': 2.0,
        'Power_Hour': 1.0,
        'London_Open': 3.0,
        'Other': 2.0  # Average
    }
    
    results = []
    
    for regime in df_filtered['regime'].unique():
        regime_char = regime_stats.get(regime, {})
        
        for session in df_filtered['session'].unique():
            subset = df_filtered[(df_filtered['regime'] == regime) & 
                               (df_filtered['session'] == session)].reset_index(drop=False)
            
            if len(subset) < 100:  # Min data constraint
                continue
            
            # Store trades for each strategy AND scope
            strategy_trades = {
                # Normal scope (existing)
                'trend': {'all': [], 'long': [], 'short': []},
                'reversion': {'all': [], 'long': [], 'short': []},
                'ma_cross': {'all': [], 'long': [], 'short': []},
                'bb_fade': {'all': [], 'long': [], 'short': []},
                # Fast scope
                'trend_fast': {'all': [], 'long': [], 'short': []},
                'reversion_fast': {'all': [], 'long': [], 'short': []},
                'ma_cross_fast': {'all': [], 'long': [], 'short': []},
                'bb_fade_fast': {'all': [], 'long': [], 'short': []},
                # Slow scope
                'trend_slow': {'all': [], 'long': [], 'short': []},
                'reversion_slow': {'all': [], 'long': [], 'short': []},
                'ma_cross_slow': {'all': [], 'long': [], 'short': []},
                'bb_fade_slow': {'all': [], 'long': [], 'short': []},
            }
            
            # Always run all strategies for comprehensive testing
            use_trend = True
            use_reversion = True
            use_ma_cross = True
            use_bb_fade = True
            
            # Run strategies and collect trades
            for i in range(200, len(subset) - 10):  # Need 200 bars for MA strategies
                # Trend following - NORMAL LONG
                if use_trend:
                    pnl, atr, entry_price = trend_following_strategy(subset, i, 'normal')
                    if pnl != 0:
                        normalized_pnl = normalize_trade_risk(pnl, entry_price)
                        trade = {
                            'pnl': pnl,
                            'normalized_pnl': normalized_pnl,
                            'atr': atr,
                            'entry_price': entry_price,
                            'duration': STRATEGY_PARAMS['trend']['normal']['hold_bars'],
                            'direction': 'long'
                        }
                        strategy_trades['trend']['all'].append(trade)
                        strategy_trades['trend']['long'].append(trade)
                    
                    # Trend following - NORMAL SHORT
                    pnl, atr, entry_price = trend_following_strategy_short(subset, i, 'normal')
                    if pnl != 0:
                        normalized_pnl = normalize_trade_risk(pnl, entry_price)
                        trade = {
                            'pnl': pnl,
                            'normalized_pnl': normalized_pnl,
                            'atr': atr,
                            'entry_price': entry_price,
                            'duration': STRATEGY_PARAMS['trend']['normal']['hold_bars'],
                            'direction': 'short'
                        }
                        strategy_trades['trend']['all'].append(trade)
                        strategy_trades['trend']['short'].append(trade)
                    
                    # Trend following - FAST LONG
                    pnl, atr, entry_price = trend_following_strategy(subset, i, 'fast')
                    if pnl != 0:
                        normalized_pnl = normalize_trade_risk(pnl, entry_price)
                        trade = {
                            'pnl': pnl,
                            'normalized_pnl': normalized_pnl,
                            'atr': atr,
                            'entry_price': entry_price,
                            'duration': STRATEGY_PARAMS['trend']['fast']['hold_bars'],
                            'direction': 'long'
                        }
                        strategy_trades['trend_fast']['all'].append(trade)
                        strategy_trades['trend_fast']['long'].append(trade)
                    
                    # Trend following - FAST SHORT
                    pnl, atr, entry_price = trend_following_strategy_short(subset, i, 'fast')
                    if pnl != 0:
                        normalized_pnl = normalize_trade_risk(pnl, entry_price)
                        trade = {
                            'pnl': pnl,
                            'normalized_pnl': normalized_pnl,
                            'atr': atr,
                            'entry_price': entry_price,
                            'duration': STRATEGY_PARAMS['trend']['fast']['hold_bars'],
                            'direction': 'short'
                        }
                        strategy_trades['trend_fast']['all'].append(trade)
                        strategy_trades['trend_fast']['short'].append(trade)
                    
                    # Trend following - SLOW LONG
                    pnl, atr, entry_price = trend_following_strategy(subset, i, 'slow')
                    if pnl != 0:
                        normalized_pnl = normalize_trade_risk(pnl, entry_price)
                        trade = {
                            'pnl': pnl,
                            'normalized_pnl': normalized_pnl,
                            'atr': atr,
                            'entry_price': entry_price,
                            'duration': STRATEGY_PARAMS['trend']['slow']['hold_bars'],
                            'direction': 'long'
                        }
                        strategy_trades['trend_slow']['all'].append(trade)
                        strategy_trades['trend_slow']['long'].append(trade)
                    
                    # Trend following - SLOW SHORT
                    pnl, atr, entry_price = trend_following_strategy_short(subset, i, 'slow')
                    if pnl != 0:
                        normalized_pnl = normalize_trade_risk(pnl, entry_price)
                        trade = {
                            'pnl': pnl,
                            'normalized_pnl': normalized_pnl,
                            'atr': atr,
                            'entry_price': entry_price,
                            'duration': STRATEGY_PARAMS['trend']['slow']['hold_bars'],
                            'direction': 'short'
                        }
                        strategy_trades['trend_slow']['all'].append(trade)
                        strategy_trades['trend_slow']['short'].append(trade)
                
                # Mean reversion - NORMAL LONG
                if use_reversion:
                    pnl, atr, entry_price = mean_reversion_strategy(subset, i, 'normal')
                    if pnl != 0:
                        normalized_pnl = normalize_trade_risk(pnl, entry_price)
                        trade = {
                            'pnl': pnl,
                            'normalized_pnl': normalized_pnl,
                            'atr': atr,
                            'entry_price': entry_price,
                            'duration': STRATEGY_PARAMS['reversion']['normal']['hold_bars'],
                            'direction': 'long'
                        }
                        strategy_trades['reversion']['all'].append(trade)
                        strategy_trades['reversion']['long'].append(trade)
                    
                    # Mean reversion - NORMAL SHORT
                    pnl, atr, entry_price = mean_reversion_strategy_short(subset, i, 'normal')
                    if pnl != 0:
                        normalized_pnl = normalize_trade_risk(pnl, entry_price)
                        trade = {
                            'pnl': pnl,
                            'normalized_pnl': normalized_pnl,
                            'atr': atr,
                            'entry_price': entry_price,
                            'duration': STRATEGY_PARAMS['reversion']['normal']['hold_bars'],
                            'direction': 'short'
                        }
                        strategy_trades['reversion']['all'].append(trade)
                        strategy_trades['reversion']['short'].append(trade)
                    
                    # Mean reversion - FAST LONG
                    pnl, atr, entry_price = mean_reversion_strategy(subset, i, 'fast')
                    if pnl != 0:
                        normalized_pnl = normalize_trade_risk(pnl, entry_price)
                        trade = {
                            'pnl': pnl,
                            'normalized_pnl': normalized_pnl,
                            'atr': atr,
                            'entry_price': entry_price,
                            'duration': STRATEGY_PARAMS['reversion']['fast']['hold_bars'],
                            'direction': 'long'
                        }
                        strategy_trades['reversion_fast']['all'].append(trade)
                        strategy_trades['reversion_fast']['long'].append(trade)
                    
                    # Mean reversion - FAST SHORT
                    pnl, atr, entry_price = mean_reversion_strategy_short(subset, i, 'fast')
                    if pnl != 0:
                        normalized_pnl = normalize_trade_risk(pnl, entry_price)
                        trade = {
                            'pnl': pnl,
                            'normalized_pnl': normalized_pnl,
                            'atr': atr,
                            'entry_price': entry_price,
                            'duration': STRATEGY_PARAMS['reversion']['fast']['hold_bars'],
                            'direction': 'short'
                        }
                        strategy_trades['reversion_fast']['all'].append(trade)
                        strategy_trades['reversion_fast']['short'].append(trade)
                    
                    # Mean reversion - SLOW LONG
                    pnl, atr, entry_price = mean_reversion_strategy(subset, i, 'slow')
                    if pnl != 0:
                        normalized_pnl = normalize_trade_risk(pnl, entry_price)
                        trade = {
                            'pnl': pnl,
                            'normalized_pnl': normalized_pnl,
                            'atr': atr,
                            'entry_price': entry_price,
                            'duration': STRATEGY_PARAMS['reversion']['slow']['hold_bars'],
                            'direction': 'long'
                        }
                        strategy_trades['reversion_slow']['all'].append(trade)
                        strategy_trades['reversion_slow']['long'].append(trade)
                    
                    # Mean reversion - SLOW SHORT
                    pnl, atr, entry_price = mean_reversion_strategy_short(subset, i, 'slow')
                    if pnl != 0:
                        normalized_pnl = normalize_trade_risk(pnl, entry_price)
                        trade = {
                            'pnl': pnl,
                            'normalized_pnl': normalized_pnl,
                            'atr': atr,
                            'entry_price': entry_price,
                            'duration': STRATEGY_PARAMS['reversion']['slow']['hold_bars'],
                            'direction': 'short'
                        }
                        strategy_trades['reversion_slow']['all'].append(trade)
                        strategy_trades['reversion_slow']['short'].append(trade)
                
                # MA Crossover - NORMAL LONG
                if use_ma_cross:
                    pnl, atr, entry_price = ma_crossover_strategy(subset, i, 'normal')
                    if pnl != 0:
                        normalized_pnl = normalize_trade_risk(pnl, entry_price)
                        trade = {
                            'pnl': pnl,
                            'normalized_pnl': normalized_pnl,
                            'atr': atr,
                            'entry_price': entry_price,
                            'duration': STRATEGY_PARAMS['ma_cross']['normal']['hold_bars'],
                            'direction': 'long'
                        }
                        strategy_trades['ma_cross']['all'].append(trade)
                        strategy_trades['ma_cross']['long'].append(trade)
                    
                    # MA Crossover - NORMAL SHORT
                    pnl, atr, entry_price = ma_crossover_strategy_short(subset, i, 'normal')
                    if pnl != 0:
                        normalized_pnl = normalize_trade_risk(pnl, entry_price)
                        trade = {
                            'pnl': pnl,
                            'normalized_pnl': normalized_pnl,
                            'atr': atr,
                            'entry_price': entry_price,
                            'duration': STRATEGY_PARAMS['ma_cross']['normal']['hold_bars'],
                            'direction': 'short'
                        }
                        strategy_trades['ma_cross']['all'].append(trade)
                        strategy_trades['ma_cross']['short'].append(trade)
                    
                    # MA Crossover - FAST LONG
                    pnl, atr, entry_price = ma_crossover_strategy(subset, i, 'fast')
                    if pnl != 0:
                        normalized_pnl = normalize_trade_risk(pnl, entry_price)
                        trade = {
                            'pnl': pnl,
                            'normalized_pnl': normalized_pnl,
                            'atr': atr,
                            'entry_price': entry_price,
                            'duration': STRATEGY_PARAMS['ma_cross']['fast']['hold_bars'],
                            'direction': 'long'
                        }
                        strategy_trades['ma_cross_fast']['all'].append(trade)
                        strategy_trades['ma_cross_fast']['long'].append(trade)
                    
                    # MA Crossover - FAST SHORT
                    pnl, atr, entry_price = ma_crossover_strategy_short(subset, i, 'fast')
                    if pnl != 0:
                        normalized_pnl = normalize_trade_risk(pnl, entry_price)
                        trade = {
                            'pnl': pnl,
                            'normalized_pnl': normalized_pnl,
                            'atr': atr,
                            'entry_price': entry_price,
                            'duration': STRATEGY_PARAMS['ma_cross']['fast']['hold_bars'],
                            'direction': 'short'
                        }
                        strategy_trades['ma_cross_fast']['all'].append(trade)
                        strategy_trades['ma_cross_fast']['short'].append(trade)
                    
                    # MA Crossover - SLOW LONG
                    pnl, atr, entry_price = ma_crossover_strategy(subset, i, 'slow')
                    if pnl != 0:
                        normalized_pnl = normalize_trade_risk(pnl, entry_price)
                        trade = {
                            'pnl': pnl,
                            'normalized_pnl': normalized_pnl,
                            'atr': atr,
                            'entry_price': entry_price,
                            'duration': STRATEGY_PARAMS['ma_cross']['slow']['hold_bars'],
                            'direction': 'long'
                        }
                        strategy_trades['ma_cross_slow']['all'].append(trade)
                        strategy_trades['ma_cross_slow']['long'].append(trade)
                    
                    # MA Crossover - SLOW SHORT
                    pnl, atr, entry_price = ma_crossover_strategy_short(subset, i, 'slow')
                    if pnl != 0:
                        normalized_pnl = normalize_trade_risk(pnl, entry_price)
                        trade = {
                            'pnl': pnl,
                            'normalized_pnl': normalized_pnl,
                            'atr': atr,
                            'entry_price': entry_price,
                            'duration': STRATEGY_PARAMS['ma_cross']['slow']['hold_bars'],
                            'direction': 'short'
                        }
                        strategy_trades['ma_cross_slow']['all'].append(trade)
                        strategy_trades['ma_cross_slow']['short'].append(trade)
                
                # BB Fade - NORMAL LONG
                if use_bb_fade:
                    pnl, atr, entry_price = bb_fade_strategy(subset, i, 'normal')
                    if pnl != 0:
                        normalized_pnl = normalize_trade_risk(pnl, entry_price)
                        trade = {
                            'pnl': pnl,
                            'normalized_pnl': normalized_pnl,
                            'atr': atr,
                            'entry_price': entry_price,
                            'duration': STRATEGY_PARAMS['bb_fade']['normal']['hold_bars'],
                            'direction': 'long'
                        }
                        strategy_trades['bb_fade']['all'].append(trade)
                        strategy_trades['bb_fade']['long'].append(trade)
                    
                    # BB Fade - NORMAL SHORT
                    pnl, atr, entry_price = bb_fade_strategy_short(subset, i, 'normal')
                    if pnl != 0:
                        normalized_pnl = normalize_trade_risk(pnl, entry_price)
                        trade = {
                            'pnl': pnl,
                            'normalized_pnl': normalized_pnl,
                            'atr': atr,
                            'entry_price': entry_price,
                            'duration': STRATEGY_PARAMS['bb_fade']['normal']['hold_bars'],
                            'direction': 'short'
                        }
                        strategy_trades['bb_fade']['all'].append(trade)
                        strategy_trades['bb_fade']['short'].append(trade)
                    
                    # BB Fade - FAST LONG
                    pnl, atr, entry_price = bb_fade_strategy(subset, i, 'fast')
                    if pnl != 0:
                        normalized_pnl = normalize_trade_risk(pnl, entry_price)
                        trade = {
                            'pnl': pnl,
                            'normalized_pnl': normalized_pnl,
                            'atr': atr,
                            'entry_price': entry_price,
                            'duration': STRATEGY_PARAMS['bb_fade']['fast']['hold_bars'],
                            'direction': 'long'
                        }
                        strategy_trades['bb_fade_fast']['all'].append(trade)
                        strategy_trades['bb_fade_fast']['long'].append(trade)
                    
                    # BB Fade - FAST SHORT
                    pnl, atr, entry_price = bb_fade_strategy_short(subset, i, 'fast')
                    if pnl != 0:
                        normalized_pnl = normalize_trade_risk(pnl, entry_price)
                        trade = {
                            'pnl': pnl,
                            'normalized_pnl': normalized_pnl,
                            'atr': atr,
                            'entry_price': entry_price,
                            'duration': STRATEGY_PARAMS['bb_fade']['fast']['hold_bars'],
                            'direction': 'short'
                        }
                        strategy_trades['bb_fade_fast']['all'].append(trade)
                        strategy_trades['bb_fade_fast']['short'].append(trade)
                    
                    # BB Fade - SLOW LONG
                    pnl, atr, entry_price = bb_fade_strategy(subset, i, 'slow')
                    if pnl != 0:
                        normalized_pnl = normalize_trade_risk(pnl, entry_price)
                        trade = {
                            'pnl': pnl,
                            'normalized_pnl': normalized_pnl,
                            'atr': atr,
                            'entry_price': entry_price,
                            'duration': STRATEGY_PARAMS['bb_fade']['slow']['hold_bars'],
                            'direction': 'long'
                        }
                        strategy_trades['bb_fade_slow']['all'].append(trade)
                        strategy_trades['bb_fade_slow']['long'].append(trade)
                    
                    # BB Fade - SLOW SHORT
                    pnl, atr, entry_price = bb_fade_strategy_short(subset, i, 'slow')
                    if pnl != 0:
                        normalized_pnl = normalize_trade_risk(pnl, entry_price)
                        trade = {
                            'pnl': pnl,
                            'normalized_pnl': normalized_pnl,
                            'atr': atr,
                            'entry_price': entry_price,
                            'duration': STRATEGY_PARAMS['bb_fade']['slow']['hold_bars'],
                            'direction': 'short'
                        }
                        strategy_trades['bb_fade_slow']['all'].append(trade)
                        strategy_trades['bb_fade_slow']['short'].append(trade)
            
            # Calculate metrics for each strategy
            hours = session_hours.get(session, 2.0)
            
            # For each strategy, calculate comprehensive metrics
            for strategy_name, trades_dict in strategy_trades.items():
                # All trades metrics
                metrics = calculate_trade_metrics(trades_dict['all'])
                metrics = normalize_metrics_by_time(metrics, hours)
                
                # Long-only metrics
                long_metrics = calculate_trade_metrics(trades_dict['long'])
                
                # Short-only metrics
                short_metrics = calculate_trade_metrics(trades_dict['short'])
                
                # Minimum trades constraint
                if metrics['num_trades'] < 30 and metrics['num_trades'] > 0:
                    log_message(f"Warning: {strategy_name} in Regime {regime}/{session} has "
                              f"{metrics['num_trades']} trades < 30", 'warning')
                
                results.append({
                    'regime': regime,
                    'session': session,
                    'strategy': strategy_name,
                    'bars': len(subset),
                    'session_hours': hours,
                    # Overall metrics
                    'total_trades': metrics['num_trades'],
                    'total_pnl': metrics['total_pnl'],
                    'normalized_pnl': metrics['normalized_pnl'],
                    'avg_pnl': metrics['avg_pnl'],
                    'expectancy': metrics['expectancy'],
                    'win_rate': metrics['win_rate'],
                    'profit_factor': metrics['profit_factor'],
                    'sharpe_ratio': metrics['sharpe_ratio'],
                    'sortino_ratio': metrics['sortino_ratio'],  
                    'calmar_ratio': metrics['calmar_ratio'],    
                    'max_drawdown': metrics['max_drawdown'],
                    'max_drawdown_pct': metrics['max_drawdown_pct'],  
                    'consecutive_losses': metrics['consecutive_losses'],  
                    'time_in_drawdown': metrics['time_in_drawdown'],  
                    'risk_reward': metrics['risk_reward'],
                    'avg_duration': metrics['avg_duration'],
                    # Normalized metrics
                    'trades_per_hour': metrics.get('trades_per_hour', 0),
                    'pnl_per_hour': metrics.get('pnl_per_hour', 0),
                    # Long metrics
                    'long_trades': long_metrics['num_trades'],
                    'long_pnl': long_metrics['total_pnl'],
                    'long_win_rate': long_metrics['win_rate'],
                    'long_sharpe': long_metrics['sharpe_ratio'],
                    # Short metrics
                    'short_trades': short_metrics['num_trades'],
                    'short_pnl': short_metrics['total_pnl'],
                    'short_win_rate': short_metrics['win_rate'],
                    'short_sharpe': short_metrics['sharpe_ratio'],
                    # Regime characteristics
                    'avg_trend': regime_char.get('avg_trend', 0),
                    'avg_volatility': regime_char.get('avg_volatility', 0)
                })
    
    results_df = pd.DataFrame(results)
    
    # Display comprehensive results with rich tables
    if DEBUG_LEVEL in ['debug', 'verbose'] and not results_df.empty:
        display_comprehensive_results(results_df)
    
    # Save detailed results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_df.to_csv(
        os.path.join(BASE_DIR, 'exports', 'csv', f"{timestamp}_enhanced_strategy_probe_results.csv"), 
        index=False
    )
    
    # Check persistence
    persistence, _ = compute_persistence(labels)
    if persistence < 75:
        log_message(f"Warning: Overall persistence {persistence:.1f}% < 75%", 'warning')
    
    log_message(f"Strategy probes complete. Total results: {len(results_df)}", 'info')
    
    return results_df

def display_comprehensive_results(results_df):
    """Display comprehensive results using rich tables"""
    
    strategies = results_df['strategy'].unique()
    
    # For each strategy, show multiple views
    for strategy in strategies:
        strategy_data = results_df[results_df['strategy'] == strategy]
        
        # 1. Combined Performance Table (Long + Short)
        combined_table = Table(title=f"\n{strategy.upper()} Strategy - Combined Performance (Long + Short)")
        combined_table.add_column("Regime", style="cyan")
        combined_table.add_column("Session", style="magenta")
        combined_table.add_column("Trades", justify="right")
        combined_table.add_column("PnL", justify="right", style="green")
        combined_table.add_column("Norm%", justify="right", style="yellow")
        combined_table.add_column("Win%", justify="right")
        combined_table.add_column("Sharpe", justify="right")
        combined_table.add_column("Sortino", justify="right")
        combined_table.add_column("PF", justify="right")
        combined_table.add_column("DD%", justify="right", style="red")

        for _, row in strategy_data.iterrows():
            combined_table.add_row(
                str(row['regime']),
                row['session'],
                str(row['total_trades']),
                f"{row['total_pnl']:.0f}",
                f"{row['normalized_pnl']:.1f}%",
                f"{row['win_rate']:.1f}%",
                f"{row['sharpe_ratio']:.2f}",
                f"{row['sortino_ratio']:.2f}",
                f"{row['profit_factor']:.2f}",
                f"{row['max_drawdown_pct']:.1f}%"
            )
        console.print(combined_table)
        
        # 2. Long-Only Performance Table
        long_table = Table(title=f"{strategy.upper()} Strategy - Long Only")
        long_table.add_column("Regime", style="cyan")
        long_table.add_column("Session", style="magenta")
        long_table.add_column("Trades", justify="right")
        long_table.add_column("PnL", justify="right", style="green")
        long_table.add_column("Win%", justify="right")
        long_table.add_column("Sharpe", justify="right")
        
        for _, row in strategy_data.iterrows():
            if row['long_trades'] > 0:
                long_table.add_row(
                    str(row['regime']),
                    row['session'],
                    str(row['long_trades']),
                    f"{row['long_pnl']:.0f}",
                    f"{row['long_win_rate']:.1f}%",
                    f"{row['long_sharpe']:.2f}"
                )
        console.print(long_table)
        
        # 3. Short-Only Performance Table
        short_table = Table(title=f"{strategy.upper()} Strategy - Short Only")
        short_table.add_column("Regime", style="cyan")
        short_table.add_column("Session", style="magenta")
        short_table.add_column("Trades", justify="right")
        short_table.add_column("PnL", justify="right", style="green")
        short_table.add_column("Win%", justify="right")
        short_table.add_column("Sharpe", justify="right")
        
        for _, row in strategy_data.iterrows():
            if row['short_trades'] > 0:
                short_table.add_row(
                    str(row['regime']),
                    row['session'],
                    str(row['short_trades']),
                    f"{row['short_pnl']:.0f}",
                    f"{row['short_win_rate']:.1f}%",
                    f"{row['short_sharpe']:.2f}"
                )
        console.print(short_table)
        
        # 4. All Regimes Combined Summary
        regime_summary = strategy_data.groupby('strategy').agg({
            'total_trades': 'sum',
            'total_pnl': 'sum',
            'long_trades': 'sum',
            'long_pnl': 'sum',
            'short_trades': 'sum',
            'short_pnl': 'sum',
            'win_rate': 'mean',
            'sharpe_ratio': 'mean',
            'profit_factor': 'mean'
        }).reset_index()
        
        if not regime_summary.empty:
            all_regimes_table = Table(title=f"{strategy.upper()} Strategy - All Regimes Combined")
            all_regimes_table.add_column("Metric", style="yellow")
            all_regimes_table.add_column("Value", justify="right", style="white")
            
            all_regimes_table.add_row("Total Trades", str(int(regime_summary.iloc[0]['total_trades'])))
            all_regimes_table.add_row("Total PnL", f"{regime_summary.iloc[0]['total_pnl']:.0f}")
            all_regimes_table.add_row("Long Trades", str(int(regime_summary.iloc[0]['long_trades'])))
            all_regimes_table.add_row("Long PnL", f"{regime_summary.iloc[0]['long_pnl']:.0f}")
            all_regimes_table.add_row("Short Trades", str(int(regime_summary.iloc[0]['short_trades'])))
            all_regimes_table.add_row("Short PnL", f"{regime_summary.iloc[0]['short_pnl']:.0f}")
            all_regimes_table.add_row("Avg Win Rate", f"{regime_summary.iloc[0]['win_rate']:.1f}%")
            all_regimes_table.add_row("Avg Sharpe", f"{regime_summary.iloc[0]['sharpe_ratio']:.2f}")
            all_regimes_table.add_row("Avg Profit Factor", f"{regime_summary.iloc[0]['profit_factor']:.2f}")
            
            console.print(all_regimes_table)
        
        # 5. All Sessions Combined Summary
        session_summary = strategy_data.groupby('session').agg({
            'total_trades': 'sum',
            'total_pnl': 'sum',
            'win_rate': 'mean',
            'sharpe_ratio': 'mean'
        }).reset_index()
        
        if not session_summary.empty:
            session_table = Table(title=f"{strategy.upper()} Strategy - By Session")
            session_table.add_column("Session", style="magenta")
            session_table.add_column("Trades", justify="right")
            session_table.add_column("PnL", justify="right", style="green")
            session_table.add_column("Avg Win%", justify="right")
            session_table.add_column("Avg Sharpe", justify="right")
            
            for _, row in session_summary.iterrows():
                session_table.add_row(
                    row['session'],
                    str(int(row['total_trades'])),
                    f"{row['total_pnl']:.0f}",
                    f"{row['win_rate']:.1f}%",
                    f"{row['sharpe_ratio']:.2f}"
                )
            console.print(session_table)
    
    # Global Summary Tables
    console.print("\n" + "="*80 + "\n")
    
    # Strategy Comparison Table
    strategy_comparison = results_df.groupby('strategy').agg({
        'total_trades': 'sum',
        'total_pnl': 'sum',
        'long_pnl': 'sum',
        'short_pnl': 'sum',
        'win_rate': 'mean',
        'sharpe_ratio': 'mean',
        'profit_factor': 'mean',
        'max_drawdown': 'min'
    }).reset_index()
    
    comparison_table = Table(title="Strategy Comparison - All Strategies")
    comparison_table.add_column("Strategy", style="cyan")
    comparison_table.add_column("Total Trades", justify="right")
    comparison_table.add_column("Total PnL", justify="right", style="green")
    comparison_table.add_column("Long PnL", justify="right")
    comparison_table.add_column("Short PnL", justify="right")
    comparison_table.add_column("Avg Win%", justify="right")
    comparison_table.add_column("Avg Sharpe", justify="right")
    comparison_table.add_column("Worst DD", justify="right", style="red")
    
    for _, row in strategy_comparison.iterrows():
        comparison_table.add_row(
            row['strategy'],
            str(int(row['total_trades'])),
            f"{row['total_pnl']:.0f}",
            f"{row['long_pnl']:.0f}",
            f"{row['short_pnl']:.0f}",
            f"{row['win_rate']:.1f}%",
            f"{row['sharpe_ratio']:.2f}",
            f"{row['max_drawdown']:.0f}"
        )
    console.print(comparison_table)
    
    # Per-Regime Performance Summary
    regime_perf = results_df.groupby('regime').agg({
        'total_pnl': 'sum',
        'total_trades': 'sum',
        'win_rate': 'mean',
        'sharpe_ratio': 'mean'
    }).reset_index()
    
    regime_table = Table(title="Regime Performance Summary")
    regime_table.add_column("Regime", style="cyan")
    regime_table.add_column("Total Trades", justify="right")
    regime_table.add_column("Total PnL", justify="right", style="green")
    regime_table.add_column("Avg Win%", justify="right")
    regime_table.add_column("Avg Sharpe", justify="right")
    
    for _, row in regime_perf.iterrows():
        regime_table.add_row(
            str(int(row['regime'])),
            str(int(row['total_trades'])),
            f"{row['total_pnl']:.0f}",
            f"{row['win_rate']:.1f}%",
            f"{row['sharpe_ratio']:.2f}"
        )
    console.print(regime_table)
    
    # Top 10 Regime-Strategy Combinations
    top_combinations = results_df.nlargest(10, 'sharpe_ratio')[['regime', 'session', 'strategy', 'sharpe_ratio', 'total_pnl', 'total_trades']]
    
    top_table = Table(title="Top 10 Regime-Strategy Combinations (by Sharpe)")
    top_table.add_column("Rank", style="yellow")
    top_table.add_column("Regime", style="cyan")
    top_table.add_column("Session", style="magenta")
    top_table.add_column("Strategy", style="white")
    top_table.add_column("Sharpe", justify="right", style="green")
    top_table.add_column("PnL", justify="right")
    top_table.add_column("Trades", justify="right")
    
    for i, (_, row) in enumerate(top_combinations.iterrows(), 1):
        top_table.add_row(
            str(i),
            str(int(row['regime'])),
            row['session'],
            row['strategy'],
            f"{row['sharpe_ratio']:.2f}",
            f"{row['total_pnl']:.0f}",
            str(int(row['total_trades']))
        )
    console.print(top_table)
    
    # Worst 5 Regime-Strategy Combinations
    worst_combinations = results_df[results_df['total_trades'] > 0].nsmallest(5, 'sharpe_ratio')[['regime', 'session', 'strategy', 'sharpe_ratio', 'total_pnl', 'total_trades']]
    
    if not worst_combinations.empty:
        worst_table = Table(title="Worst 5 Regime-Strategy Combinations (by Sharpe)")
        worst_table.add_column("Rank", style="yellow")
        worst_table.add_column("Regime", style="cyan")
        worst_table.add_column("Session", style="magenta")
        worst_table.add_column("Strategy", style="white")
        worst_table.add_column("Sharpe", justify="right", style="red")
        worst_table.add_column("PnL", justify="right")
        worst_table.add_column("Trades", justify="right")
        
        for i, (_, row) in enumerate(worst_combinations.iterrows(), 1):
            worst_table.add_row(
                str(i),
                str(int(row['regime'])),
                row['session'],
                row['strategy'],
                f"{row['sharpe_ratio']:.2f}",
                f"{row['total_pnl']:.0f}",
                str(int(row['total_trades']))
            )
        console.print(worst_table)

def main():
    df = load_csv_data(DATA_PATH)
    if TEST_SLICE > 0: df = df.iloc[:TEST_SLICE]
    
    # Get features and fit model
    ind_df, _ = select_and_compute_indicators_live(df)
    features = ind_df.select_dtypes(include=[np.number]).dropna()
    
    model, _ = fit_gmm(features)
    if model is None:
        log_message("GMM fit failed", 'error')
        return
    
    results = run_strategy_probes(df, model)
    
    # Summary statistics
    # The display is now handled within run_strategy_probes via display_comprehensive_results
    if not results.empty:
        log_message(f"Analysis complete. {len(results)} strategy-regime combinations tested.", 'info')

if __name__ == "__main__":
    main()

