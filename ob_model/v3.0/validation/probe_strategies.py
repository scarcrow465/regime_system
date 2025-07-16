#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import pandas_ta as ta
from utils.logger import log_message, progress_bar
from config.settings import DEBUG_LEVEL, BASE_DIR, DATA_PATH
from core.regime_classifier import fit_gmm, add_session_labels, name_clusters
from core.indicators import select_and_compute_indicators
from core.data_loader import load_csv_data
from rich.table import Table
import os
from datetime import datetime
from core.helpers import console

def simple_ma_crossover_probe(df):
    """Simple probe for trend: Buy on EMA50 > EMA200 cross."""
    df['EMA50'] = df['close'].rolling(50).mean()
    df['EMA200'] = df['close'].rolling(200).mean()
    df['signal'] = np.where(df['EMA50'] > df['EMA200'], 1, 0)  # Buy if cross
    df['pnl'] = df['signal'].shift(1) * (df['close'] - df['open'])  # Sim pnl
    return df['pnl'].sum()  # Total profit

def simple_bb_fade_probe(df):
    """Simple probe for range: Buy lower BB touch."""
    bb = ta.bbands(df['close'], length=20)
    df['lower_bb'] = bb['BBL_20_2.0']
    df['signal'] = np.where(df['low'] <= df['lower_bb'], 1, 0)  # Buy touch
    df['pnl'] = df['signal'].shift(1) * (df['close'] - df['open'])
    return df['pnl'].sum()

def run_probes(df, model):
    """Run probes per regime/session in 4am-16:00 window."""
    df = df[(df.index.hour >= 4) & (df.index.hour < 16)]  # Filter window
    features = select_and_compute_indicators(df)
    features = add_session_labels(features)
    features = features.select_dtypes(include=[np.number]).dropna()
    labels = pd.Series(model.predict(features), index=features.index)
    df['regime'] = labels
    df['session'] = features['refined_session']  # Use refined
    
    results = []
    for regime in df['regime'].unique():
        for session in df['session'].unique():
            subset = df[(df['regime'] == regime) & (df['session'] == session)]
            if len(subset) < 50:
                continue
            ma_profit = simple_ma_crossover_probe(subset) if "trend" in name_clusters(model, features)[regime] else 0
            bb_profit = simple_bb_fade_probe(subset) if "range" in name_clusters(model, features)[regime] or "low vol" in name_clusters(model, features)[regime] else 0
            pf = ma_profit + bb_profit  # Sim profit factor >0 if match
            results.append({'regime': regime, 'session': session, 'ma_profit': ma_profit, 'bb_profit': bb_profit, 'pf': pf})
    
    results_df = pd.DataFrame(results)
    if DEBUG_LEVEL in ['debug', 'verbose']:
        table = Table(title="Probe Profits")
        for col in results_df.columns:
            table.add_column(col.capitalize())
        for _, row in results_df.iterrows():
            table.add_row(*[str(v) for v in row])
        console.print(table)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_df.to_csv(os.path.join(BASE_DIR, 'exports', 'csv', f"{timestamp}_probe_results.csv"), index=False)
    log_message("Probes complete", 'info')
    return results_df

def main():
    df = load_csv_data(DATA_PATH)
    model, _ = fit_gmm(select_and_compute_indicators(df))
    run_probes(df, model)

if __name__ == "__main__":
    main()

