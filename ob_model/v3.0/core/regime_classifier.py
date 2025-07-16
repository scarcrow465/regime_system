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
    """Add full (overlapping) and refined (non-overlapping) session labels."""
    if len(df) == 0:
        log_message("Empty DF for session labels", 'error')
        return df
    
    df['hour'] = df.index.hour
    
    # CORRECTED Full sessions (Eastern Time)
    df['full_session'] = 'Other'
    df.loc[(df['hour'] >= 18) | (df['hour'] < 2), 'full_session'] = 'Sydney'  # 5pm-2am ET
    df.loc[(df['hour'] >= 20) | (df['hour'] < 4), 'full_session'] = 'Tokyo'   # 7pm-4am ET
    df.loc[(df['hour'] >= 3) & (df['hour'] < 12), 'full_session'] = 'London'  # 3am-12pm ET
    df.loc[(df['hour'] >= 8) & (df['hour'] < 17), 'full_session'] = 'NY'      # 8am-5pm ET
    
    # CORRECTED Refined sessions (your key trading windows)
    df['refined_session'] = 'Other'
    df.loc[(df['hour'] >= 3) & (df['hour'] < 5), 'refined_session'] = 'London_Open'    # 3-5am ET
    df.loc[(df['hour'] >= 8) & (df['hour'] < 11), 'refined_session'] = 'NY_Open'       # 8-11am ET
    df.loc[(df['hour'] >= 11) & (df['hour'] < 14), 'refined_session'] = 'NY_Midday'    # 11am-2pm ET
    df.loc[(df['hour'] >= 14) & (df['hour'] < 15), 'refined_session'] = 'NY_Afternoon' # 2-3pm ET
    df.loc[(df['hour'] >= 15) & (df['hour'] < 16), 'refined_session'] = 'Power_Hour'   # 3-4pm ET
    
    if DEBUG_LEVEL == 'verbose':
        log_message(f"Full session dist: {df['full_session'].value_counts()}", 'info')
        log_message(f"Refined session dist: {df['refined_session'].value_counts()}", 'info')
    
    return df

def fit_gmm(features, n_components_range=[2,5], walk_forward=True):
    if len(features) < n_components_range[1]:
        log_message("Insufficient data for GMM—need > max n_components rows", 'error')
        return None, None
    numeric_features = features.select_dtypes(include=[np.number]).fillna(0)  # Extra fill for safety
    
    best_model = None
    best_bic = float('inf')
    best_n = None
    
    if walk_forward:
        train_size = int(len(numeric_features) * 0.8)
        train, test = numeric_features.iloc[:train_size], numeric_features.iloc[train_size:]
    else:
        train, test = numeric_features, None
    
    for n in progress_bar(range(n_components_range[0], n_components_range[1]+1), desc="GMM tuning"):
        # In fit_gmm function, wrap the GMM fit in try-except:
        for attempt in range(3):  # Try up to 3 times
            try:
                gmm = GaussianMixture(n_components=n, covariance_type='diag', 
                                    random_state=42, n_init=3, max_iter=200)
                gmm.fit(train)
                break
            except Exception as e:
                if attempt == 2:
                    log_message(f"GMM failed after 3 attempts: {e}", 'error')
                    continue
                log_message(f"GMM attempt {attempt+1} failed, retrying...", 'warning')
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

def name_clusters(model, features):
    labels = model.predict(features)
    cluster_means = features.groupby(labels).mean()
    names = []
    for i in range(model.n_components):
        mean = cluster_means.loc[i]
        if mean['ADX_14'] > 30 and mean['BB_width'] < 0.05:
            names.append("Strong Trend Low Vol")
        elif mean['ADX_14'] < 20 and mean['BB_width'] > 0.1:
            names.append("Sideways High Vol")
        # Add rules for other indicators/classes
        else:
            names.append(f"Regime {i}")
    return dict(enumerate(names))

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
    # In main, after fit
    cluster_names = name_clusters(model, features)
    log_message(f"Cluster Names: {cluster_names}", 'info')
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    export_model(model, timestamp)

