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
    """Run strategies on each regime/session combination."""
    # Filter to trading hours
    df_filtered = df[(df.index.hour >= 4) & (df.index.hour < 16)].copy()
    
    # Get features and labels
    ind_df = select_and_compute_indicators(df_filtered)
    ind_df = add_session_labels(ind_df)
    features = ind_df.select_dtypes(include=[np.number]).dropna()
    
    # Get regime labels with smoothing
    from core.regime_classifier import smooth_regime_labels
    raw_labels = pd.Series(model.predict(features), index=features.index)
    labels = smooth_regime_labels(raw_labels, min_persistence=3)
    
    # Calculate regime characteristics once
    regime_stats = calculate_regime_characteristics(df_filtered, model, features)
    
    # Merge everything
    df_filtered['regime'] = labels
    df_filtered['session'] = ind_df['refined_session']
    
    results = []
    
    for regime in df_filtered['regime'].unique():
        regime_char = regime_stats.get(regime, {})
        
        for session in df_filtered['session'].unique():
            subset = df_filtered[(df_filtered['regime'] == regime) & 
                               (df_filtered['session'] == session)]
            
            if len(subset) < 100:  # Need minimum data
                continue
            
            # Reset index for easier iteration
            subset = subset.reset_index(drop=False)
            
            trend_pnl = 0
            reversion_pnl = 0
            trend_trades = 0
            reversion_trades = 0
            
            # Choose strategy based on regime characteristics
            use_trend = regime_char.get('avg_trend', 0) > 25  # Strong trend
            use_reversion = regime_char.get('avg_volatility', 0) < subset['ATR_14'].median()  # Low vol
            
            # Run through bars
            for i in range(20, len(subset) - 10):  # Leave room for exits
                if use_trend:
                    pnl = trend_following_strategy(subset, i)
                    if pnl != 0:
                        trend_pnl += pnl
                        trend_trades += 1
                
                if use_reversion:
                    pnl = mean_reversion_strategy(subset, i)
                    if pnl != 0:
                        reversion_pnl += pnl
                        reversion_trades += 1
            
            results.append({
                'regime': regime,
                'session': session,
                'bars': len(subset),
                'trend_pnl': trend_pnl,
                'trend_trades': trend_trades,
                'reversion_pnl': reversion_pnl,
                'reversion_trades': reversion_trades,
                'total_pnl': trend_pnl + reversion_pnl,
                'avg_trend': regime_char.get('avg_trend', 0),
                'avg_volatility': regime_char.get('avg_volatility', 0)
            })
    
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

