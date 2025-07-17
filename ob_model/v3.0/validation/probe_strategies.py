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

def run_strategy_probes(df, model):
    df_filtered = df[(df.index.hour >= 4) & (df.index.hour < 16)].copy()
    
    # CRITICAL FIX: Compute indicators on FULL data first to match training
    ind_df_full, raw_ind_df_full = select_and_compute_indicators(df)  # Full data
    features_full = ind_df_full.select_dtypes(include=[np.number]).dropna()
    
    # Now filter AFTER computing features
    filter_mask = (df.index.hour >= 4) & (df.index.hour < 16)
    features = features_full[filter_mask].copy()
    raw_features = raw_ind_df_full.select_dtypes(include=[np.number])[filter_mask].copy()
    df_filtered = df[filter_mask].copy()
    
    # Get session info for filtered data
    session_info = ind_df_full[['refined_session']][filter_mask].copy()
    
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
        # Single model case (backward compatibility)
        raw_labels = pd.Series(model.predict(features), index=features.index)
    else:
        # Multi-model voting case
        class_labels = pd.DataFrame(index=features.index)

        log_message(f"Feature columns: {list(features.columns)[:10]}...", 'debug')

        for cls, model_info in model.items():
            try:
                if isinstance(model_info, tuple):
                    cls_model, fitted_cols = model_info
                    log_message(f"{cls}: Expected {len(fitted_cols)} cols, got {len([c for c in fitted_cols if c in features.columns])}", 'debug')
                    # Ensure columns exist
                    available_cols = [col for col in fitted_cols if col in features.columns]
                    if not available_cols:
                        log_message(f"Warning: No features available for {cls}", 'warning')
                        continue
                    class_labels[cls] = pd.Series(cls_model.predict(features[available_cols]), index=features.index)
                else:
                    # Fallback for old format
                    log_message(f"Warning: {cls} model not in expected format", 'warning')
                    continue
            except Exception as e:
                log_message(f"Error processing {cls}: {e}", 'error')
                continue
        
        if class_labels.empty:
            log_message("No valid class predictions, falling back to single model", 'warning')
            # Fallback - use first valid model
            for cls, model_info in model.items():
                if isinstance(model_info, tuple):
                    cls_model, _ = model_info
                    raw_labels = pd.Series(cls_model.predict(features), index=features.index)
                    break
        else:
            raw_labels = class_labels.mode(axis=1)[0].astype(int)
    labels = smooth_regime_labels(raw_labels, min_persistence=3)
    
    regime_stats = calculate_regime_characteristics(df_filtered, model, features, raw_features)
    
    df_filtered['regime'] = labels
    df_filtered['session'] = session_info['refined_session']
    
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
            
            # Use regime characteristics only (these are OK as they're aggregate stats)
            use_trend = regime_char.get('avg_trend', 0) > 15
            use_reversion = regime_char.get('avg_volatility', 0) < 0.01  # Use absolute threshold
            
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
    ind_df, _ = select_and_compute_indicators(df)
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

