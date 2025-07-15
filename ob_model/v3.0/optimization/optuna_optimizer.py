#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import sys
import os
BASE_DIR = r"C:\Users\rs\GitProjects\regime_system\ob_model\v3.0"  # Hardcode if not importing settings yet
sys.path.append(BASE_DIR)
import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
import optuna
from optuna.pruners import HyperbandPruner
from utils.logger import log_message, progress_bar
from config.settings import DEBUG_LEVEL, BASE_DIR, OB_PATH, DATA_PATH
from validation.ob_prober import merge_regimes, probe_filtering, load_ob_csv  # Reuse from Phase 1C
from validation.persistence_validator import compute_persistence, compare_is_oos  # Reuse from Phase 1B
from core.indicators import select_and_compute_indicators
from core.data_loader import load_csv_data
from core.regime_classifier import add_session_labels
from rich.table import Table
from datetime import datetime
import joblib
from core.helpers import console
from core.regime_classifier import export_model

MAX_LOOPS = 4
N_TRIALS = 10

def optuna_objective(trial, df, features):
    n_components = trial.suggest_int('n_components', 2, 5)
    cov_type = trial.suggest_categorical('cov_type', ['full', 'diag'])
    
    model = GaussianMixture(n_components=n_components, covariance_type=cov_type, random_state=42)
    model.fit(features)
    
    labels = pd.Series(model.predict(features), index=features.index)
    
    persistence, _ = compute_persistence(labels)
    _, ks = compare_is_oos(features, model)
    merged = merge_regimes(load_ob_csv(OB_PATH), df, model)
    combos = probe_filtering(merged)[0]
    lift = combos['lift'].mean() if not combos.empty else 0
    
    # New metrics from pnl in combos
    pnl = merged['pnl']
    returns = pnl / 100  # Sim, adjust to % if needed
    sharpe = returns.mean() / returns.std() if returns.std() != 0 else 0
    wins = pnl[pnl > 0].sum()
    losses = abs(pnl[pnl < 0].sum())
    profit_factor = wins / losses if losses > 0 else 0
    cum_pnl = pnl.cumsum()
    dd = (cum_pnl - cum_pnl.cummax()).min()
    dd_penalty = abs(dd) / 15 if dd < 0 else 0  # Normalize <15% =0 penalty
    
    score = 0.3 * lift + 0.3 * sharpe + 0.2 * profit_factor + 0.1 * (1 - dd_penalty) + 0.1 * persistence
    if DEBUG_LEVEL == 'verbose':
        log_message(f"Trial: n={n_components}, cov={cov_type}, score={score} (lift={lift}, sharpe={sharpe}, pf={profit_factor}, dd_pen={dd_penalty}, pers={persistence})", 'info')
    
    return score

def run_optuna_loop(df, features, loop_num):
    """Single Optuna loop."""
    study = optuna.create_study(direction='maximize', pruner=HyperbandPruner())
    study.optimize(lambda trial: optuna_objective(trial, df, features), n_trials=N_TRIALS)
    best_params = study.best_params
    best_score = study.best_value
    if DEBUG_LEVEL in ['debug', 'verbose']:
        table = Table(title=f"Loop {loop_num} Best")
        table.add_column("Param")
        table.add_column("Value")
        for k, v in best_params.items():
            table.add_row(k, str(v))
        console.print(table)
    return best_params, best_score, study

def main():
    df = load_csv_data(DATA_PATH)
    ind_df = select_and_compute_indicators(df)
    ind_df = add_session_labels(ind_df)
    features = ind_df.select_dtypes(include=[np.number]).dropna()
    changes = []
    best_model = None
    for i in progress_bar(range(1, MAX_LOOPS+1), desc="Optuna loops"):
        log_message(f"Loop {i}", 'info')
        params, score, study = run_optuna_loop(df, features, i)  # Pass df
        changes.append(f"Loop {i}: Score {score:.2f}, Params {params}")
        if score > 3 and sharpe > 1.2 and profit_factor > 1.3 and dd_penalty < 0.15 and persistence > 75:
            log_message("Criteria met—exiting", 'info')
            model = GaussianMixture(**params, random_state=42)
            model.fit(features)
            best_model = model
            break
        log_message("Retuning for next loop", 'info')
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(os.path.join(BASE_DIR, 'exports', 'csv', f"{timestamp}_opt_changes.csv"), 'w') as f:
        f.write("\n".join(changes))
    if best_model:
        joblib.dump(study, os.path.join(BASE_DIR, 'exports', 'study.pkl'))
        export_model(best_model, timestamp)

if __name__ == "__main__":
    main()

