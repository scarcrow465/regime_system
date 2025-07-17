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
from config.settings import DEBUG_LEVEL, BASE_DIR, DATA_PATH
from core.regime_classifier import fit_gmm, add_session_labels
from core.indicators import select_and_compute_indicators
from core.data_loader import load_csv_data
from rich.table import Table
from datetime import datetime
from core.helpers import console
from utils.metrics import compute_persistence

def ma_crossover_strategy(df, entry_bar):
    """MA crossover for trend: Buy on fast > slow MA."""
    if entry_bar < 50 or entry_bar + 5 >= len(df):
        return 0
    ma_fast = ta.sma(df['close'], length=50).iloc[entry_bar]
    ma_slow = ta.sma(df['close'], length=200).iloc[entry_bar]
    if df['close'].iloc[entry_bar] > ma_fast > ma_slow:
        entry_price = df['close'].iloc[entry_bar]
        exit_bar = min(entry_bar + 5, len(df) - 1)
        exit_price = df['close'].iloc[exit_bar]
        return exit_price - entry_price
    return 0

def bb_fade_strategy(df, entry_bar):
    """BB fade for range/low vol: Buy lower band touch."""
    if entry_bar < 20 or entry_bar + 3 >= len(df):
        return 0
    bb = ta.bbands(df['close'], length=20)
    lower_bb = bb['BBL_20_2.0'].iloc[entry_bar]
    if df['low'].iloc[entry_bar] <= lower_bb:
        entry_price = df['close'].iloc[entry_bar]
        exit_bar = min(entry_bar + 3, len(df) - 1)
        exit_price = df['close'].iloc[exit_bar]
        return exit_price - entry_price
    return 0

def calculate_regime_characteristics(df, model, features):
    """Pre-calculate regime characteristics to avoid repeated calls."""
    labels = pd.Series(model.predict(features), index=features.index)
    
    # Calculate average indicators per regime
    regime_stats = {}
    for regime in range(model.n_components):
        regime_data = features[labels == regime]
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
        if 'ATR_14' in df.columns:
            stop_loss = entry_price - 2 * df['ATR_14'].iloc[entry_bar]
            for i in range(entry_bar + 1, exit_bar + 1):
                if df['low'].iloc[i] <= stop_loss:
                    exit_price = stop_loss
                    break
        
        return exit_price - entry_price
    
    return 0

def mean_reversion_strategy(df, entry_bar):
    """
    Mean reversion: Buy when RSI < 30 and at lower Bollinger Band
    Exit when RSI > 50 or after 3 bars
    """
    if entry_bar < 20 or entry_bar + 3 >= len(df):
        return 0
    
    # Calculate indicators if not present
    if 'RSI_14' not in df.columns:
        df['RSI_14'] = ta.rsi(df['close'], length=14)
    
    bb = ta.bbands(df['close'], length=20)
    lower_bb = bb['BBL_20_2.0'].iloc[entry_bar]
    
    # Entry logic
    if (df['RSI_14'].iloc[entry_bar] < 30 and 
        df['low'].iloc[entry_bar] <= lower_bb):
        
        entry_price = df['close'].iloc[entry_bar]
        
        # Exit logic
        for i in range(entry_bar + 1, min(entry_bar + 4, len(df))):
            if df['RSI_14'].iloc[i] > 50:
                return df['close'].iloc[i] - entry_price
        
        # Exit after 3 bars if no RSI exit
        return df['close'].iloc[min(entry_bar + 3, len(df) - 1)] - entry_price
    
    return 0

def run_strategy_probes(df, model):
    df_filtered = df[(df.index.hour >= 4) & (df.index.hour < 16)].copy()  # Session window
    
    ind_df = select_and_compute_indicators(df_filtered)
    ind_df = add_session_labels(ind_df)
    features = ind_df.select_dtypes(include=[np.number]).dropna()
    
    from core.regime_classifier import smooth_regime_labels

    class_groups = {
            'direction': [col for col in features if col.startswith('EMA') or col in ['ADX_14', 'MACD']],
            'volatility': [col for col in features if col.startswith('ATR') or col in ['BB_width', 'Hist_Vol', 'KC_width']],
            'trend_strength': [col for col in features if col.startswith('RSI') or col in ['Stoch', 'DMI_Plus', 'DMI_Minus']],
            'momentum': [col for col in features if col in ['ROC_12', 'PPO', 'PPO_Hist', 'PPO_Signal', 'CCI_20']],
            'session': [col for col in features if 'session' in col.lower()],
            'structure': [col for col in features if col in ['OBV', 'VWAP_14', 'CMF_20']]
        }

    if not isinstance(model, dict):
        raw_labels = pd.Series(model.predict(features), index=features.index)
    else:
        class_labels = pd.DataFrame(index=features.index)
        for cls, model_info in model.items():
            cls_model, fitted_cols = model_info
            class_labels[cls] = pd.Series(cls_model.predict(features[fitted_cols]), index=features.index)
        raw_labels = class_labels.mode(axis=1)[0].astype(int)  # Compute voting here
    labels = smooth_regime_labels(raw_labels, min_persistence=3)
    
    regime_stats = calculate_regime_characteristics(df_filtered, model, features)
    
    df_filtered['regime'] = labels
    df_filtered['session'] = ind_df['refined_session']
    
    results = []
    
    for regime in df_filtered['regime'].unique():
        regime_char = regime_stats.get(regime, {})
        
        for session in df_filtered['session'].unique():
            subset = df_filtered[(df_filtered['regime'] == regime) & (df_filtered['session'] == session)].reset_index(drop=False)
            
            if len(subset) < 100:  # #2 Min data
                continue
            
            # Run probes (existing + new)
            trend_pnl = reversion_pnl = ma_pnl = bb_pnl = 0
            trend_trades = reversion_trades = ma_trades = bb_trades = 0
            
            use_trend = regime_char.get('avg_trend', 0) > 25
            use_reversion = regime_char.get('avg_volatility', 0) < subset['ATR_14'].median()
            
            for i in range(20, len(subset) - 10):
                if use_trend:
                    pnl = trend_following_strategy(subset, i)
                    if pnl != 0:
                        trend_pnl += pnl
                        trend_trades += 1
                    pnl = ma_crossover_strategy(subset, i)  # New
                    if pnl != 0:
                        ma_pnl += pnl
                        ma_trades += 1
                if use_reversion:
                    pnl = mean_reversion_strategy(subset, i)
                    if pnl != 0:
                        reversion_pnl += pnl
                        reversion_trades += 1
                    pnl = bb_fade_strategy(subset, i)  # New
                    if pnl != 0:
                        bb_pnl += pnl
                        bb_trades += 1
            
            # #1 Per-Regime Metrics (risk/reward, frequency)
            total_pnl = trend_pnl + reversion_pnl + ma_pnl + bb_pnl
            total_trades = trend_trades + reversion_trades + ma_trades + bb_trades
            if total_trades > 0:
                wins = total_pnl > 0  # Simple proxy
                risk_reward = (total_pnl if wins else 0) / abs(total_pnl) if total_pnl < 0 else 1
            else:
                risk_reward = 1
            frequency = total_trades / len(subset) * 100 if len(subset) > 0 else 0
            
            results.append({
                'regime': regime,
                'session': session,
                'bars': len(subset),
                'trend_pnl': trend_pnl,
                'reversion_pnl': reversion_pnl,
                'ma_pnl': ma_pnl,
                'bb_pnl': bb_pnl,
                'total_pnl': total_pnl,
                'risk_reward': risk_reward,  # #1
                'frequency': frequency,  # #1
                'avg_trend': regime_char.get('avg_trend', 0),
                'avg_volatility': regime_char.get('avg_volatility', 0)
            })
            
            # #2 Constraints (min trades, persistence)
            if total_trades < 30:
                log_message(f"Warning: Regime {regime}/{session} has {total_trades} trades <30—low reliability", 'warning')
    
    results_df = pd.DataFrame(results)
    
    if DEBUG_LEVEL in ['debug', 'verbose']:
        table = Table(title="Strategy Probe Results")
        table.add_column("Regime")
        table.add_column("Session") 
        table.add_column("Bars")
        table.add_column("Trend PnL")
        table.add_column("Rev PnL")
        table.add_column("Total PnL")
        
        for _, row in results_df.iterrows():
            table.add_row(
                str(row['regime']),
                row['session'],
                str(row['bars']),
                f"{row['trend_pnl']:.2f}",
                f"{row['reversion_pnl']:.2f}",
                f"{row['total_pnl']:.2f}"
            )
        console.print(table)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_df.to_csv(
        os.path.join(BASE_DIR, 'exports', 'csv', f"{timestamp}_strategy_probe_results.csv"), 
        index=False
    )

    # Global #2 persistence (from labels)
    persistence, _ = compute_persistence(labels)
    if persistence < 75:
        log_message(f"Warning: Overall persistence {persistence:.1f}% <75%—iterate", 'warning')
    
    log_message(f"Strategy probes complete. Total results: {len(results_df)}", 'info')
    return results_df

def main():
    df = load_csv_data(DATA_PATH)
    
    # Get features and fit model
    ind_df = select_and_compute_indicators(df)
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

