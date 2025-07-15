#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from utils.logger import log_message, progress_bar
from config.settings import DEBUG_LEVEL, BASE_DIR, DATA_PATH
from core.indicators import select_and_compute_indicators
import joblib
import os
from datetime import datetime
from core.data_loader import load_csv_data
import numpy as np

def add_session_labels(df):
    """Incorporate session labels (e.g., NY Open)."""
    if len(df) == 0:
        log_message("Empty DF for session labels", 'error')
        return df
    df['hour'] = df.index.hour
    df['session'] = 'Other'
    df.loc[(df['hour'] >= 8) & (df['hour'] < 11), 'session'] = 'NY_Open'
    if DEBUG_LEVEL == 'verbose':
        log_message(f"Session distribution: {df['session'].value_counts()}", 'info')
    return df

def fit_gmm(features, n_components_range=[2,5], walk_forward=True):
    # Select only numeric columns for GMM
    numeric_features = features.select_dtypes(include=[np.number])
    
    if len(numeric_features) < n_components_range[1]:
        log_message("Insufficient numeric data for GMM—need > max n_components rows", 'error')
        return None, None
    
    best_model = None
    best_bic = float('inf')
    best_n = None
    
    if walk_forward:
        train_size = int(len(numeric_features) * 0.8)
        train, test = numeric_features.iloc[:train_size], numeric_features.iloc[train_size:]
    else:
        train, test = numeric_features, None
    
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
    
    if test is not None and len(test) > 0:
        test_score = silhouette_score(test, best_model.predict(test))
        if DEBUG_LEVEL != 'none':
            log_message(f"OOS silhouette: {test_score}", 'info')
    
    return best_model, best_n

def export_model(model, timestamp):
    if model is None:
        return
    export_path = os.path.join(BASE_DIR, 'exports', 'models', f"{timestamp}_gmm.pkl")
    joblib.dump(model, export_path)
    log_message(f"Exported model to {export_path}", 'info')

if __name__ == "__main__":
    df = load_csv_data([DATA_PATH])
    if df.empty:
        log_message("No data loaded—check path", 'error')
        exit(1)
    df = select_and_compute_indicators(df)
    df = add_session_labels(df)
    features = df.dropna()
    if len(features) < 5:
        log_message("Insufficient data—need 5+ rows", 'error')
        exit(1)
    model, n = fit_gmm(features)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    export_model(model, timestamp)

