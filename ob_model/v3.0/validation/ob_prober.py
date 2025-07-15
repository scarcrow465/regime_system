#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from utils.logger import log_message, progress_bar
from config.settings import DEBUG_LEVEL, BASE_DIR, OB_PATH, DATA_PATH  # Add OB_PATH = r"your/sample.csv" in settings
from core.regime_classifier import fit_gmm  # For labels (assume from Phase 1B model load if needed)
from core.indicators import select_and_compute_indicators
from core.data_loader import load_csv_data
from rich.table import Table
import os
from datetime import datetime
from core.helpers import console

def load_ob_csv(ob_path):
    """Load OB CSV, parse dates."""
    ob_df = pd.read_csv(ob_path, parse_dates=['entry_time', 'exit_time'])
    ob_df['entry_time'] = pd.to_datetime(ob_df['entry_time'], utc=True).dt.tz_convert('America/New_York').dt.floor('15min')  # TZ + floor for merge
    ob_df['outcome_win'] = (ob_df['outcome'] == 'WIN').astype(int)  # For win_rate
    # Handle pnl_by_candle cols if present (optional)
    pnl_cols = [col for col in ob_df.columns if col.startswith('pnl_') and col.endswith('m')]
    if pnl_cols and DEBUG_LEVEL == 'verbose':
        log_message(f"Found {len(pnl_cols)} pnl_by_candle cols", 'info')
    if DEBUG_LEVEL == 'verbose':
        log_message(f"OB sample: {ob_df.head(1).to_dict()}", 'info')
    return ob_df

def merge_regimes(ob_df, df, model):
    """Merge regimes to OB via entry_time."""
    ind_df = select_and_compute_indicators(df)
    features = ind_df.select_dtypes(include=[np.number]).dropna()
    labels = pd.Series(model.predict(features), index=features.index, name='regime')
    merged = ob_df.merge(labels, left_on='entry_time', right_index=True, how='left')
    merged.dropna(subset=['regime'], inplace=True)  # Drop unmatched
    if len(merged) < len(ob_df):
        log_message(f"{len(ob_df) - len(merged)} unmatched timestamps", 'info')
    return merged

def probe_filtering(merged):
    """Probe lifts by regime combo."""
    if merged.empty:
        log_message("Empty merged DF—no probes possible", 'error')
        return pd.DataFrame(), None
    baseline_win = merged['outcome_win'].mean() * 100
    combos = merged.groupby('regime').agg(
        trades=('pnl', 'count'),
        win_rate=('outcome_win', 'mean'),
        avg_pnl=('pnl', 'mean')
    )
    combos['win_rate'] *= 100
    combos['lift'] = combos['win_rate'] - baseline_win
    if 'session' in merged:  # Session crosstab
        crosstab = pd.crosstab(merged['regime'], merged['session'], normalize='index') * 100
    else:
        crosstab = None
    if DEBUG_LEVEL in ['debug', 'verbose']:
        table = Table(title="Probe Results")
        for col in combos.columns:
            table.add_column(col.capitalize())
        for idx, row in combos.iterrows():
            table.add_row(str(idx), *[f"{v:.1f}" for v in row])
        console.print(table)
        if not combos.empty:
            sns.heatmap(combos[['lift']], annot=True)
            plt.savefig(os.path.join(BASE_DIR, 'exports', 'plots', 'probes_heatmap.png'))
        else:
            log_message("No probed trades—skipping heatmap", 'info')
    return combos, crosstab

def validate_oos(merged):
    """OOS validation on probes."""
    if merged.empty:
        log_message("Empty merged DF—no OOS validation", 'error')
        return 0.0
    train_size = int(len(merged) * 0.8)
    train, test = merged.iloc[:train_size], merged.iloc[train_size:]
    train_combos = probe_filtering(train)[0]
    test_combos = probe_filtering(test)[0]
    if train_combos.empty or test_combos.empty:
        log_message("Empty combos in OOS—skipping", 'info')
        return 0.0
    train_lift = train_combos['lift'].mean()
    test_lift = test_combos['lift'].mean()
    delta = abs(train_lift - test_lift)
    if DEBUG_LEVEL != 'none':
        log_message(f"OOS lift delta: {delta:.1f}%", 'info')
    return delta

def main():
    df = load_csv_data(DATA_PATH)  # Use the list directly
    if df.empty:
        log_message("No OHLC data loaded—check paths/columns", 'error')
        return
    ob_df = load_ob_csv(OB_PATH)  # OB trades
    ind_df = select_and_compute_indicators(df)
    features = ind_df.select_dtypes(include=[np.number]).dropna()
    if len(features) < 5:
        log_message("Insufficient features for GMM", 'error')
        return
    model, _ = fit_gmm(features)  # Sim from Phase 1B
    if model is None:
        log_message("GMM fit failed—no model", 'error')
        return
    merged = merge_regimes(ob_df, df, model)
    combos, crosstab = probe_filtering(merged)
    delta = validate_oos(merged)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    combos.to_csv(os.path.join(BASE_DIR, 'exports', 'csv', f"{timestamp}_probed_trades.csv"))

if __name__ == "__main__":
    main()

