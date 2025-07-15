#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score  # For alt to BIC if needed
from utils.logger import log_message, progress_bar
from config.settings import DEBUG_LEVEL
from core.indicators import select_and_compute_indicators
import joblib  # For export
import os
import sys
from config.settings import BASE_DIR
sys.path.append(BASE_DIR)
from core.data_loader import load_csv_data

def add_session_labels(df):
    """Incorporate session labels (e.g., NY Open)."""
    df['hour'] = df.index.hour
    df['session'] = 'Other'
    df.loc[(df['hour'] >= 8) & (df['hour'] < 11), 'session'] = 'NY_Open'  # For OB focus
    if DEBUG_LEVEL == 'verbose':
        log_message(f"Session distribution: {df['session'].value_counts()}", 'info')
    return df

def fit_gmm(features, n_components_range=[2,5], walk_forward=True):
    """Fit GMM with tuning (BIC/grid), walk-forward."""
    best_model = None
    best_bic = float('inf')
    best_n = None
    
    if walk_forward:
        train_size = int(len(features) * 0.8)
        train, test = features.iloc[:train_size], features.iloc[train_size:]
    else:
        train, test = features, None
    
    for n in progress_bar(range(n_components_range[0], n_components_range[1]+1), desc="GMM tuning"):
        gmm = GaussianMixture(n_components=n, covariance_type='diag', random_state=42)
        gmm.fit(train)
        bic = gmm.bic(train)
        if bic < best_bic:
            best_bic = bic
            best_model = gmm
            best_n = n
        if DEBUG_LEVEL == 'debug':
            log_message(f"n={n}, BIC={bic}", 'info')
    
    if test is not None:
        test_score = silhouette_score(test, best_model.predict(test))  # Validate OOS
        if DEBUG_LEVEL != 'none':
            log_message(f"OOS silhouette: {test_score}", 'info')
    
    return best_model, best_n

def export_model(model, timestamp):
    export_path = os.path.join(BASE_DIR, 'exports', 'models', f"{timestamp}_gmm.pkl")
    joblib.dump(model, export_path)
    log_message(f"Exported model to {export_path}", 'info')

if __name__ == "__main__":
    df = select_and_compute_indicators(load_csv_data(["path/to/csv.csv"]))  # Chain
    df = add_session_labels(df)
    features = df.dropna()  # For GMM
    model, n = fit_gmm(features)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    export_model(model, timestamp)

