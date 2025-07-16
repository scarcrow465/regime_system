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
    
    # Get OB data and merge
    ob_df = load_ob_csv(OB_PATH)
    merged = merge_regimes(ob_df, df, model)
    
    if merged.empty:
        return 0.0, 0, 0, 0, 0, 0  # Defaults
    
    # Calculate metrics from merged data
    persistence, _ = compute_persistence(labels)
    _, ks = compare_is_oos(features, model)
    
    # Get probe results
    combos = probe_filtering(merged)[0]
    baseline_win = merged['outcome_win'].mean()
    lift = combos['lift'].mean() if not combos.empty else 0
    
    # Calculate trading metrics from merged OB data
    pnl = merged['pnl']
    returns = pnl / 100  # Convert to percentage
    sharpe = (returns.mean() / returns.std()) * np.sqrt(252*26) if returns.std() != 0 else 0  # Annualized
    
    wins = pnl[pnl > 0].sum()
    losses = abs(pnl[pnl < 0].sum())
    profit_factor = wins / losses if losses > 0 else 0
    
    cum_pnl = pnl.cumsum()
    dd = (cum_pnl - cum_pnl.cummax()).min()
    dd_penalty = abs(dd) / 1000 if dd < -1000 else abs(dd) / 10000  # Normalized
    
    # Hybrid score as discussed
    score = (0.3 * (lift/10) +  # Normalize lift to 0-1 range (10% target)
             0.3 * min(sharpe/1.5, 1) +  # Normalize sharpe (1.5 target)
             0.2 * min(profit_factor/2, 1) +  # Normalize PF (2.0 target)
             0.1 * (1 - dd_penalty) +
             0.1 * (persistence/100))  # Normalize persistence
    
    if DEBUG_LEVEL == 'verbose':
        log_message(f"Trial: n={n_components}, cov={cov_type}, score={score:.3f} "
                   f"(lift={lift:.1f}%, sharpe={sharpe:.2f}, pf={profit_factor:.2f}, "
                   f"dd={dd:.0f}, pers={persistence:.1f}%)", 'info')
    
    return score, lift, sharpe, profit_factor, dd_penalty, persistence

def run_optuna_loop(df, features, loop_num):
    """Single Optuna loop."""
    study = optuna.create_study(direction='maximize', pruner=HyperbandPruner())
    study.optimize(lambda trial: optuna_objective(trial, df, features)[0], n_trials=N_TRIALS)  # Optimize on score only
    best_params = study.best_params
    best_score = study.best_value
    
    # Unpack full metrics from best
    best_score, lift, sharpe, profit_factor, dd_penalty, persistence = optuna_objective(optuna.trial.FixedTrial(best_params), df, features)
    
    if DEBUG_LEVEL in ['debug', 'verbose']:
        table = Table(title=f"Loop {loop_num} Best")
        table.add_column("Param")
        table.add_column("Value")
        for k, v in best_params.items():
            table.add_row(k, str(v))
        console.print(table)
    return best_params, best_score, study, lift, sharpe, profit_factor, dd_penalty, persistence  # Return extras

def main():
    df = load_csv_data(DATA_PATH)
    ind_df = select_and_compute_indicators(df)
    ind_df = add_session_labels(ind_df)
    features = ind_df.select_dtypes(include=[np.number]).dropna()
    changes = []
    best_model = None
    for i in progress_bar(range(1, MAX_LOOPS+1), desc="Optuna loops"):
        log_message(f"Loop {i}", 'info')
        params, score, study, lift, sharpe, profit_factor, dd_penalty, persistence = run_optuna_loop(df, features, i)  # Unpack extras
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

