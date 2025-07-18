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

def ma_crossover_strategy(df, entry_bar):
    """MA crossover for trend: Buy on fast > slow MA."""
    if entry_bar < 200 or entry_bar + 5 >= len(df):
        return 0
    
    # Calculate MAs using ONLY data up to entry_bar
    ma_fast = df['close'].iloc[max(0, entry_bar-50):entry_bar].mean()
    ma_slow = df['close'].iloc[max(0, entry_bar-200):entry_bar].mean()
    
    if df['close'].iloc[entry_bar] > ma_fast > ma_slow:
        entry_price = df['close'].iloc[entry_bar]
        exit_bar = min(entry_bar + 5, len(df) - 1)
        exit_price = df['close'].iloc[exit_bar]
        point_value = 20
        return (exit_price - entry_price) * point_value
    return 0

def bb_fade_strategy(df, entry_bar):
    """BB fade for range/low vol: Buy lower band touch."""
    if entry_bar < 20 or entry_bar + 3 >= len(df):
        return 0
    
    # Calculate BB using ONLY historical data
    close_slice = df['close'].iloc[max(0, entry_bar-20):entry_bar]
    sma = close_slice.mean()
    std = close_slice.std()
    lower_bb = sma - (2 * std)
    
    if df['low'].iloc[entry_bar] <= lower_bb:
        entry_price = df['close'].iloc[entry_bar]
        exit_bar = min(entry_bar + 3, len(df) - 1)
        exit_price = df['close'].iloc[exit_bar]
        point_value = 20
        return (exit_price - entry_price) * point_value
    return 0

def calculate_regime_characteristics(df, model, features, raw_features):
    """Pre-calculate regime characteristics to avoid repeated calls."""
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
            regime_stats[regime] = {
                'avg_volatility': regime_data['ATR_14'].mean() if 'ATR_14' in regime_data.columns else 0,
                'avg_trend': regime_data['ADX_14'].mean() if 'ADX_14' in regime_data.columns else 0,
                'avg_rsi': regime_data['RSI_14'].mean() if 'RSI_14' in regime_data.columns else 50,
                'count': len(regime_data)
            }
    
    return regime_stats

def trend_following_strategy(df, entry_bar):
    """
    Trend following: Enter when price breaks above 20-bar high
    Exit after 5 bars or stop loss hit
    """
    if entry_bar < 20 or entry_bar + 5 >= len(df):
        return 0
    
    # Entry logic
    high_20 = df['high'].iloc[entry_bar-20:entry_bar].max()
    if df['close'].iloc[entry_bar] > high_20:
        entry_price = df['close'].iloc[entry_bar]
        
        # Hold for up to 5 bars
        exit_bar = min(entry_bar + 5, len(df) - 1)
        exit_price = df['close'].iloc[exit_bar]
        
        # Check stop loss (2 ATR)
        # Calculate ATR using only historical data
        tr_list = []
        for j in range(max(1, entry_bar-14), entry_bar):
            high_low = df['high'].iloc[j] - df['low'].iloc[j]
            high_close = abs(df['high'].iloc[j] - df['close'].iloc[j-1])
            low_close = abs(df['low'].iloc[j] - df['close'].iloc[j-1])
            tr_list.append(max(high_low, high_close, low_close))

        atr_value = sum(tr_list) / len(tr_list) if tr_list else 0
        stop_loss = entry_price - 2 * atr_value

        for i in range(entry_bar + 1, exit_bar + 1):
            if df['low'].iloc[i] <= stop_loss:
                exit_price = stop_loss
                break
        
        return exit_price - entry_price
    
    return 0

def mean_reversion_strategy(df, entry_bar):
    """
    Mean reversion: Buy when RSI < 40 and near lower Bollinger Band
    Exit when RSI > 50 or after 3 bars
    """
    if entry_bar < 20 or entry_bar + 3 >= len(df):
        return 0
    
    # Calculate RSI manually using only historical data
    close_slice = df['close'].iloc[max(0, entry_bar-14):entry_bar+1]
    deltas = close_slice.diff()
    gains = deltas.where(deltas > 0, 0)
    losses = -deltas.where(deltas < 0, 0)
    avg_gain = gains.iloc[1:].mean()  # Skip first NaN
    avg_loss = losses.iloc[1:].mean()
    rs = avg_gain / avg_loss if avg_loss != 0 else 0
    rsi_value = 100 - (100 / (1 + rs))
    
    # Calculate BB using only historical data
    bb_slice = df['close'].iloc[max(0, entry_bar-20):entry_bar]
    sma = bb_slice.mean()
    std = bb_slice.std()
    lower_bb = sma - (2 * std)
    middle_bb = sma
    
    # Entry conditions
    price_pct_from_lower = (df['close'].iloc[entry_bar] - lower_bb) / (middle_bb - lower_bb) if middle_bb != lower_bb else 1
    
    if (rsi_value < 40 and price_pct_from_lower < 0.2):
        entry_price = df['close'].iloc[entry_bar]
        
        # Exit - calculate RSI for each exit bar
        for i in range(entry_bar + 1, min(entry_bar + 4, len(df))):
            # Recalculate RSI at exit bar
            exit_close_slice = df['close'].iloc[max(0, i-14):i+1]
            exit_deltas = exit_close_slice.diff()
            exit_gains = exit_deltas.where(exit_deltas > 0, 0)
            exit_losses = -exit_deltas.where(exit_deltas < 0, 0)
            exit_avg_gain = exit_gains.iloc[1:].mean()
            exit_avg_loss = exit_losses.iloc[1:].mean()
            exit_rs = exit_avg_gain / exit_avg_loss if exit_avg_loss != 0 else 0
            exit_rsi = 100 - (100 / (1 + exit_rs))
            
            if exit_rsi > 50:
                point_value = 20
                return (df['close'].iloc[i] - entry_price) * point_value
        
        # Exit after 3 bars
        point_value = 20
        return (df['close'].iloc[min(entry_bar + 3, len(df) - 1)] - entry_price) * point_value
    
    return 0

def calculate_trade_metrics(trades):
    """Calculate comprehensive metrics for a list of trades"""
    if len(trades) == 0:
        return {
            'num_trades': 0,
            'total_pnl': 0,
            'avg_pnl': 0,
            'win_rate': 0,
            'profit_factor': 0,
            'max_drawdown': 0,
            'sharpe_ratio': 0,
            'avg_duration': 0,
            'risk_reward': 0
        }
    
    pnls = [t['pnl'] for t in trades]
    total_pnl = sum(pnls)
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    
    win_rate = len(wins) / len(trades) * 100 if len(trades) > 0 else 0
    profit_factor = sum(wins) / abs(sum(losses)) if losses else float('inf')
    
    # Calculate drawdown
    cumulative = np.cumsum(pnls)
    running_max = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - running_max)
    max_drawdown = abs(min(drawdown)) if len(drawdown) > 0 else 0
    
    # Sharpe ratio (simplified)
    if len(pnls) > 1:
        returns = pd.Series(pnls)
        sharpe_ratio = (returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0
    else:
        sharpe_ratio = 0
    
    # Risk reward
    avg_win = np.mean(wins) if wins else 0
    avg_loss = abs(np.mean(losses)) if losses else 0
    risk_reward = avg_win / avg_loss if avg_loss > 0 else 0
    
    return {
        'num_trades': len(trades),
        'total_pnl': total_pnl,
        'avg_pnl': total_pnl / len(trades),
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'max_drawdown': max_drawdown,
        'sharpe_ratio': sharpe_ratio,
        'avg_duration': np.mean([t['duration'] for t in trades]),
        'risk_reward': risk_reward
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
    regime_stats = {}
    for regime in df_filtered['regime'].unique():
        regime_data = df_filtered[df_filtered['regime'] == regime]
        regime_feats = raw_ind_df.loc[regime_data.index]
        
        regime_stats[regime] = {
            'avg_trend': regime_feats[['EMA_20', 'EMA_50']].diff().mean().mean(),
            'avg_volatility': regime_feats[['ATR_14']].mean().values[0] if 'ATR_14' in regime_feats else 0,
            'count': len(regime_data)
        }
    
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
        
        # Store trades for each strategy
        all_trades = {
            'trend': {'all': [], 'long': [], 'short': []},
            'reversion': {'all': [], 'long': [], 'short': []},
            'ma_cross': {'all': [], 'long': [], 'short': []},
            'bb_fade': {'all': [], 'long': [], 'short': []}
        }
        
        for session in df_filtered['session'].unique():
            subset = df_filtered[(df_filtered['regime'] == regime) & 
                               (df_filtered['session'] == session)].reset_index(drop=False)
            
            if len(subset) < 100:  # Min data constraint
                continue
            
            # Determine which strategies to use
            use_trend = regime_char.get('avg_trend', 0) > 15
            use_reversion = regime_char.get('avg_volatility', 0) < 0.01
            
            # Run strategies and collect trades
            for i in range(20, len(subset) - 10):
                # Trend following
                if use_trend:
                    pnl = trend_following_strategy(subset, i)
                    if pnl != 0:
                        trade = {
                            'pnl': pnl,
                            'duration': 5,  # bars
                            'direction': 'long' if pnl > 0 else 'short'
                        }
                        all_trades['trend']['all'].append(trade)
                        all_trades['trend'][trade['direction']].append(trade)
                
                # MA Crossover
                pnl = ma_crossover_strategy(subset, i)
                if pnl != 0:
                    trade = {
                        'pnl': pnl,
                        'duration': 5,
                        'direction': 'long'  # MA cross is long-only
                    }
                    all_trades['ma_cross']['all'].append(trade)
                    all_trades['ma_cross']['long'].append(trade)
                
                # Mean reversion
                if use_reversion:
                    pnl = mean_reversion_strategy(subset, i)
                    if pnl != 0:
                        trade = {
                            'pnl': pnl,
                            'duration': 3,
                            'direction': 'short' if pnl > 0 else 'long'
                        }
                        all_trades['reversion']['all'].append(trade)
                        all_trades['reversion'][trade['direction']].append(trade)
                
                # BB Fade
                pnl = bb_fade_strategy(subset, i)
                if pnl != 0:
                    trade = {
                        'pnl': pnl,
                        'duration': 3,
                        'direction': 'long'  # BB fade is long-only
                    }
                    all_trades['bb_fade']['all'].append(trade)
                    all_trades['bb_fade']['long'].append(trade)
            
            # Calculate metrics for each strategy
            hours = session_hours.get(session, 2.0)
            
            # For each strategy, calculate comprehensive metrics
            for strategy_name, trades_dict in all_trades.items():
                # All trades metrics
                metrics = calculate_trade_metrics(trades_dict['all'])
                metrics = normalize_metrics_by_time(metrics, hours)
                
                # Long-only metrics
                long_metrics = calculate_trade_metrics(trades_dict['long'])
                
                # Short-only metrics
                short_metrics = calculate_trade_metrics(trades_dict['short'])
                
                # Minimum trades constraint
                if metrics['num_trades'] < 30:
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
                    'avg_pnl': metrics['avg_pnl'],
                    'win_rate': metrics['win_rate'],
                    'profit_factor': metrics['profit_factor'],
                    'sharpe_ratio': metrics['sharpe_ratio'],
                    'max_drawdown': metrics['max_drawdown'],
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
    
    # Display results based on debug level
    if DEBUG_LEVEL in ['debug', 'verbose']:
        # Create strategy-specific tables
        for strategy in results_df['strategy'].unique():
            strategy_data = results_df[results_df['strategy'] == strategy]
            
            table = Table(title=f"{strategy.upper()} Strategy Results")
            table.add_column("Regime")
            table.add_column("Session")
            table.add_column("Trades")
            table.add_column("PnL")
            table.add_column("Win%")
            table.add_column("Sharpe")
            table.add_column("Max DD")
            table.add_column("PF")
            
            for _, row in strategy_data.iterrows():
                table.add_row(
                    str(row['regime']),
                    row['session'],
                    str(row['total_trades']),
                    f"{row['total_pnl']:.2f}",
                    f"{row['win_rate']:.1f}%",
                    f"{row['sharpe_ratio']:.2f}",
                    f"{row['max_drawdown']:.2f}",
                    f"{row['profit_factor']:.2f}"
                )
            console.print(table)
        
        # Summary table
        summary_table = Table(title="Strategy Summary - All Regimes")
        summary_table.add_column("Strategy")
        summary_table.add_column("Total Trades")
        summary_table.add_column("Total PnL")
        summary_table.add_column("Avg Win%")
        summary_table.add_column("Avg Sharpe")
        
        for strategy in results_df['strategy'].unique():
            strategy_data = results_df[results_df['strategy'] == strategy]
            summary_table.add_row(
                strategy,
                str(strategy_data['total_trades'].sum()),
                f"{strategy_data['total_pnl'].sum():.2f}",
                f"{strategy_data['win_rate'].mean():.1f}%",
                f"{strategy_data['sharpe_ratio'].mean():.2f}"
            )
        console.print(summary_table)
    
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
    
    # Print regime-specific analysis
    if not results_df.empty:
        log_message("Regime Performance Summary:", 'info')
        
        regime_summary = results_df.groupby('regime').agg({
            'total_pnl': ['sum', 'mean'],
            'total_trades': 'sum',
            'win_rate': 'mean',
            'sharpe_ratio': 'mean'
        })
        
        print("\nPer-Regime Performance:")
        print(regime_summary)
        
        # Identify best regime-strategy pairs
        best_pairs = results_df.nlargest(5, 'sharpe_ratio')[['regime', 'session', 'strategy', 'sharpe_ratio', 'total_pnl']]
        print("\nTop 5 Regime-Strategy Combinations:")
        print(best_pairs)
    
    return results_df

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
    if not results.empty:
        log_message(f"Average PnL by regime:", 'info')
        regime_summary = results.groupby('regime')['total_pnl'].agg(['mean', 'sum', 'count'])
        print(regime_summary)

if __name__ == "__main__":
    main()

