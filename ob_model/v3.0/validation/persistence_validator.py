#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
from sklearn.metrics import silhouette_score  # For KS alt if needed
from utils.logger import log_message, progress_bar
from config.settings import DEBUG_LEVEL, BASE_DIR, DATA_PATH
from core.regime_classifier import fit_gmm, add_session_labels
from core.indicators import select_and_compute_indicators
from core.data_loader import load_csv_data
from core.regime_classifier import export_model  # Assuming this is defined in regime_classifier
from core.helpers import console  # Assuming console is defined in utils or similar
import os
from datetime import datetime
import matplotlib.pyplot as plt  # For plots if debug
from rich.table import Table  # For rich tables

MAX_ITERS = 5

def compute_persistence(labels):
    """Calculate persistence % and transitions."""
    changes = (labels != labels.shift(1)).cumsum()
    persistence = labels.groupby(changes).size().mean() / len(labels) * 100
    transitions = pd.crosstab(labels.shift(1), labels, normalize='index')
    if DEBUG_LEVEL == 'verbose':
        log_message(f"Persistence: {persistence:.2f}%", 'info')
    return persistence, transitions

def analyze_distributions(labels, sessions=None):
    """Distributions with session breakdown."""
    dist = labels.value_counts(normalize=True) * 100
    if sessions is not None:
        crosstab = pd.crosstab(labels, sessions, normalize='index') * 100
    else:
        crosstab = None
    if DEBUG_LEVEL in ['debug', 'verbose']:
        table = Table(title="Distributions")
        table.add_column("Regime")
        table.add_column("%")
        for r, p in dist.items():
            table.add_row(str(r), f"{p:.1f}%")
        console.print(table)
    return dist, crosstab

def compare_is_oos(features, model):
    """IS vs OOS comparison."""
    train_size = int(len(features) * 0.8)
    train, test = features.iloc[:train_size], features.iloc[train_size:]
    train_labels = model.predict(train)
    test_labels = model.predict(test)
    train_dist = pd.Series(train_labels).value_counts(normalize=True)
    test_dist = pd.Series(test_labels).value_counts(normalize=True)
    delta = abs(train_dist - test_dist).mean() * 100
    ks = np.max(np.abs(np.cumsum(train_dist.sort_index()) - np.cumsum(test_dist.sort_index())))  # Sim KS
    if DEBUG_LEVEL != 'none':
        log_message(f"OOS delta: {delta:.1f}%, KS: {ks:.2f}", 'info')
    return delta, ks

def run_validation_iteration(df, iter_num):
    """Single iteration."""
    ind_df = select_and_compute_indicators(df)
    ind_df = add_session_labels(ind_df)
    features = ind_df.select_dtypes(include=[np.number]).dropna()
    model, n = fit_gmm(features)
    labels = pd.Series(model.predict(features), index=features.index)
    persistence, transitions = compute_persistence(labels)
    dist, crosstab = analyze_distributions(labels, ind_df['session'])
    delta, ks = compare_is_oos(features, model)
    if DEBUG_LEVEL == 'debug':
        plt.hist(labels)
        plt.savefig(os.path.join(BASE_DIR, 'exports', 'plots', f"dist_iter{iter_num}.png"))
    return persistence, dist, delta, ks, model

def main():
    df = load_csv_data([DATA_PATH])
    changes = []
    for i in progress_bar(range(1, MAX_ITERS+1), desc="Validation iterations"):
        log_message(f"Iteration {i}", 'info')
        persistence, dist, delta, ks, model = run_validation_iteration(df, i)
        changes.append(f"Iter {i}: Persistence {persistence:.1f}%, Delta {delta:.1f}%, KS {ks:.2f}")
        if persistence > 75 and delta < 10 and ks > 0.5:
            log_message("Criteria met—exiting loop", 'info')
            break
        # Modify (sim: retune n_range or drop col)
        log_message("Modifying for next iter (e.g., adjust params)", 'info')
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(os.path.join(BASE_DIR, 'exports', 'csv', f"{timestamp}_changes.csv"), 'w') as f:
        f.write("\n".join(changes))
    export_model(model, timestamp)  # From regime_classifier

if __name__ == "__main__":
    main()


# In[ ]:




