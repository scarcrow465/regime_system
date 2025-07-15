#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from utils.logger import log_message, progress_bar
from config.settings import DEBUG_LEVEL, BASE_DIR, OB_PATH  # Add OB_PATH = r"your/sample.csv" in settings
from core.regime_classifier import fit_gmm  # For labels (assume from Phase 1B model load if needed)
from core.indicators import select_and_compute_indicators
from core.data_loader import load_csv_data
from rich.table import Table
import os
from datetime import datetime
from core.helpers import console

def load_ob_csv(ob_path):
    """Load OB CSV, parse dates."""
    ob_df = pd.read_csv(ob_path, parse_dates=['entry_date_time', 'exit_date_time'])
    ob_df['entry_time'] = ob_df['entry_date_time'].dt.floor('15min')  # For merge
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
        sns.heatmap(combos[['lift']], annot=True)
        plt.savefig(os.path.join(BASE_DIR, 'exports', 'plots', 'probes_heatmap.png'))
    return combos, crosstab

def validate_oos(merged):
    """OOS validation on probes."""
    train_size = int(len(merged) * 0.8)
    train, test = merged.iloc[:train_size], merged.iloc[train_size:]
    train_lift = probe_filtering(train)['lift'].mean()
    test_lift = probe_filtering(test)['lift'].mean()
    delta = abs(train_lift - test_lift)
    if DEBUG_LEVEL != 'none':
        log_message(f"OOS lift delta: {delta:.1f}%", 'info')
    return delta

def main():
    df = load_csv_data([DATA_PATH])  # OHLC for regimes
    ob_df = load_ob_csv(OB_PATH)  # OB trades
    model, _ = fit_gmm(select_and_compute_indicators(df))  # Sim from Phase 1B
    merged = merge_regimes(ob_df, df, model)
    combos, crosstab = probe_filtering(merged)
    delta = validate_oos(merged)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    combos.to_csv(os.path.join(BASE_DIR, 'exports', 'csv', f"{timestamp}_probed_trades.csv"))

if __name__ == "__main__":
    main()

