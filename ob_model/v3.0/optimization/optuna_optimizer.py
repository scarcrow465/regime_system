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

def optuna_objective(trial, features):
    """Optuna objective: GMM params for OB lift + persistence + KS."""
    n_components = trial.suggest_int('n_components', 2, 5)
    cov_type = trial.suggest_categorical('cov_type', ['full', 'diag'])
    
    model = GaussianMixture(n_components=n_components, covariance_type=cov_type, random_state=42)
    model.fit(features)
    
    labels = pd.Series(model.predict(features), index=features.index)
    
    persistence, _ = compute_persistence(labels)  # From Phase 1B
    _, ks = compare_is_oos(features, model)  # From Phase 1B
    merged = merge_regimes(load_ob_csv(OB_PATH), features.index.to_frame(), model)  # Sim DF for merge
    combos = probe_filtering(merged)[0]
    lift = combos['lift'].mean() if not combos.empty else 0
    
    score = 0.4 * lift + 0.3 * persistence + 0.3 * ks
    if DEBUG_LEVEL == 'verbose':
        log_message(f"Trial params: n={n_components}, cov={cov_type}, score={score}", 'info')
    
    return score

def run_optuna_loop(features, loop_num):
    """Single Optuna loop."""
    study = optuna.create_study(direction='maximize', pruner=HyperbandPruner())
    study.optimize(lambda trial: optuna_objective(trial, features), n_trials=N_TRIALS)
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
        params, score, study = run_optuna_loop(features, i)
        changes.append(f"Loop {i}: Score {score:.2f}, Params {params}")
        if score > 20:  # Sim criteria (lift proxy)
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

