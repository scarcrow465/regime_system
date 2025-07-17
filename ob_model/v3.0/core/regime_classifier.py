#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import sys
import os
BASE_DIR = r"C:\Users\rs\GitProjects\regime_system\ob_model\v3.0"  # Hardcode if not importing settings yet
sys.path.append(BASE_DIR)
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import OneHotEncoder
from utils.logger import log_message, progress_bar
from config.settings import DEBUG_LEVEL, BASE_DIR, DATA_PATH
from core.indicators import select_and_compute_indicators_live
import joblib
from datetime import datetime
from core.data_loader import load_csv_data
import numpy as np
from utils.metrics import compute_persistence

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
        full_dist = df['full_session'].value_counts(normalize=True) * 100
        refined_dist = df['refined_session'].value_counts(normalize=True) * 100
        log_message(f"Full session dist: {full_dist}", 'info')
        log_message(f"Refined session dist: {refined_dist}", 'info')
        
        # #2 Balance Check (warn if any <10%)
        for sess, pct in full_dist.items():
            if pct < 10:
                log_message(f"Warning: Full session {sess} only {pct:.1f}%—imbalanced", 'warning')
        for sess, pct in refined_dist.items():
            if pct < 10:
                log_message(f"Warning: Refined session {sess} only {pct:.1f}%—imbalanced", 'warning')
    
    # One-hot encode for GMM features
    from sklearn.preprocessing import OneHotEncoder
    full_cats = ['Other', 'Sydney', 'Tokyo', 'London', 'NY']
    refined_cats = ['Other', 'London_Open', 'NY_Open', 'NY_Midday', 'NY_Afternoon', 'Power_Hour']
    encoder = OneHotEncoder(sparse_output=False, categories=[full_cats, refined_cats])
    session_enc = encoder.fit_transform(df[['full_session', 'refined_session']])
    session_df = pd.DataFrame(session_enc, index=df.index, columns=encoder.get_feature_names_out(['full_session', 'refined_session']))  # Pass input_features
    df = pd.concat([df, session_df], axis=1)
    
    return df

def fit_gmm(features, n_components_range=[2,5], walk_forward=True):
    if len(features) < n_components_range[1]:
        log_message("Insufficient data for GMM—need > max n_components rows", 'error')
        return None, None
    numeric_features = features.select_dtypes(include=[np.number]).fillna(0)
    
    # Define class groups (from indicators.py logic)
    class_groups = {
        'direction': [col for col in numeric_features if col.startswith('EMA') or col in ['ADX_14', 'MACD']],
        'volatility': [col for col in numeric_features if col.startswith('ATR') or col in ['BB_width', 'Hist_Vol', 'KC_width']],
        'trend_strength': [col for col in numeric_features if col.startswith('RSI') or col in ['Stoch', 'DMI_Plus', 'DMI_Minus']],
        'momentum': [col for col in numeric_features if col in ['ROC_12', 'PPO', 'PPO_Hist', 'PPO_Signal', 'CCI_20']],
        'session': [col for col in numeric_features if 'session' in col.lower()],
        'structure': [col for col in numeric_features if col in ['OBV', 'VWAP_14', 'CMF_20']]
    }
    
    class_models = {}
    class_labels = pd.DataFrame(index=numeric_features.index)
    
    for cls, cols in class_groups.items():
        if not cols:
            log_message(f"No features for class {cls}", 'warning')
            continue
        
        class_feats = numeric_features[cols]
        
        best_model = None
        best_bic = float('inf')
        best_n = None
        
        if walk_forward:
            train_size = int(len(class_feats) * 0.8)
            train, test = class_feats.iloc[:train_size], class_feats.iloc[train_size:]
        else:
            train, test = class_feats, None
        
        for n in progress_bar(range(n_components_range[0], n_components_range[1]+1), desc=f"{cls} GMM"):
            for attempt in range(3):
                try:
                    gmm = GaussianMixture(n_components=n, covariance_type='diag', random_state=42, n_init=3, max_iter=200)
                    gmm.fit(train)
                    break
                except Exception as e:
                    if attempt == 2:
                        log_message(f"{cls} GMM failed: {e}", 'error')
                        continue
                    log_message(f"{cls} attempt {attempt+1} failed", 'warning')
            bic = gmm.bic(train)
            if bic < best_bic:
                best_bic = bic
                best_model = gmm
                best_n = n
            if DEBUG_LEVEL == 'debug':
                log_message(f"{cls} n={n}, BIC={bic}", 'info')
        
        # Store model and columns AFTER finding best
        if best_model is not None:
            class_models[cls] = (best_model, cols)  # Store actual column names used

        if test is not None and len(test) > 0 and best_model is not None:
            test_score = silhouette_score(test, best_model.predict(test))
            if DEBUG_LEVEL != 'none':
                log_message(f"{cls} OOS silhouette: {test_score}", 'info')
        
        # Generate labels using the stored model
        if best_model is not None:
            class_labels[cls] = pd.Series(best_model.predict(class_feats), index=class_feats.index)
    
    if class_labels.empty:
        log_message("No class labels generated", 'error')
        return None, None
    
    # Voting for final labels
    final_labels = class_labels.mode(axis=1, dropna=True)[0].astype(int)
    
    # #1 Per-Regime Metrics (use close pct as pnl proxy)
    unique_regimes = final_labels.unique()
    pnl_proxy = features.get('pnl', pd.Series(0, index=features.index))  # Change to "pnl_proxy = features.get('pnl', pd.Series(0, index=features.index))" (uses real 'pnl' if added to data_loader later; 0 otherwise). Real #1 happens in probes/ob_prober with actual trades.
    for regime in unique_regimes:
        regime_mask = final_labels == regime
        regime_pnl = pnl_proxy[regime_mask]
        avg_profit = regime_pnl.mean()
        win_rate = (regime_pnl > 0).mean() * 100
        wins = regime_pnl[regime_pnl > 0].mean()
        losses = abs(regime_pnl[regime_pnl < 0].mean())
        risk_reward = wins / losses if losses > 0 else 1
        frequency = regime_mask.sum()
        log_message(f"Regime {regime}: Avg Profit {avg_profit:.2f}, Win% {win_rate:.1f}, Risk/Reward {risk_reward:.1f}, Frequency {frequency}", 'info')
    
    # #2 Constraints
    persistence, _ = compute_persistence(final_labels)
    if persistence < 75:
        log_message(f"Warning: Persistence {persistence:.1f}% <75%—unstable", 'warning')
    for regime in unique_regimes:
        if (final_labels == regime).sum() < 30:
            log_message(f"Warning: Regime {regime} <30 points—low reliability", 'warning')
    
    # Test smoothing on sample (keep as is)
    test_labels = final_labels.head(100)
    raw_persistence, _ = compute_persistence(test_labels)
    if raw_persistence < 1:
        log_message(f"WARNING: Raw persistence {raw_persistence:.2f}% - unstable!", 'warning')
    
    return class_models, final_labels.nunique()  # Return models, num regimes

def smooth_regime_labels(labels, min_persistence=3):
    """
    Smooth regime labels to prevent single-bar flips.
    min_persistence: minimum bars a regime must persist before changing
    """
    if len(labels) < min_persistence * 2:
        log_message(f"Not enough data for smoothing (need {min_persistence * 2} bars)", 'warning')
        return labels
    
    smoothed = labels.copy()
    current_regime = labels.iloc[0]
    persistence_counter = 0
    candidate_regime = None
    
    for i in range(1, len(labels)):
        if labels.iloc[i] == current_regime:
            # Same regime, reset counter
            persistence_counter = 0
            candidate_regime = None
        else:
            if candidate_regime == labels.iloc[i]:
                # Continue counting same candidate
                persistence_counter += 1
                if persistence_counter >= min_persistence:
                    # Confirmed regime change
                    current_regime = candidate_regime
                    persistence_counter = 0
                    candidate_regime = None
            else:
                # New candidate regime
                candidate_regime = labels.iloc[i]
                persistence_counter = 1
        
        smoothed.iloc[i] = current_regime
    
    # Log the improvement
    if DEBUG_LEVEL == 'verbose':
        raw_changes = (labels != labels.shift(1)).sum()
        smooth_changes = (smoothed != smoothed.shift(1)).sum()
        log_message(f"Smoothing reduced regime changes from {raw_changes} to {smooth_changes}", 'info')
    
    return smoothed

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
    df = load_csv_data(DATA_PATH)
    ind_df, _ = select_and_compute_indicators_live(df)
    df = add_session_labels(df)
    features = ind_df.dropna()
    if len(features) < 5:
        log_message("Insufficient data—need 5+ rows", 'error')
        exit(1)
    model, n = fit_gmm(features)
    # In main, after fit
    # cluster_names = name_clusters(model, features)
    # log_message(f"Cluster Names: {cluster_names}", 'info')
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    export_model(model, timestamp)

