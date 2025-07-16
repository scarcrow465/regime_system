#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np

def compute_persistence(labels: pd.Series):
    """Calculate average persistence % and regime transition matrix."""
    labels = labels.dropna()
    if len(labels) < 2:
        return 0.0, pd.DataFrame()

    # Identify regime change points
    change_points = (labels != labels.shift(1)).cumsum()
    run_lengths = labels.groupby(change_points).size()

    # Average run length as percent of total length
    persistence = run_lengths.mean() / len(labels) * 100

    # Transition matrix
    transitions = pd.crosstab(labels.shift(1), labels, normalize='index')
    return persistence, transitions


def compare_is_oos(features, model):
    """IS vs OOS comparison."""
    train_size = int(len(features) * 0.8)
    train, test = features.iloc[:train_size], features.iloc[train_size:]
    train_labels = model.predict(train)
    test_labels = model.predict(test)
    train_dist = pd.Series(train_labels).value_counts(normalize=True)
    test_dist = pd.Series(test_labels).value_counts(normalize=True)
    delta = abs(train_dist - test_dist).mean() * 100
    ks = np.max(np.abs(np.cumsum(train_dist.sort_index()) - np.cumsum(test_dist.sort_index())))
    return delta, ks

