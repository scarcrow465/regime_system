# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import pandas as pd
results = pd.read_csv('walk_forward_results_20250703_170115.csv')
print(results.describe())
print("\nBest 5 windows:")
print(results.nlargest(5, 'test_sharpe')[['window', 'test_sharpe', 'train_sharpe']])
print("\nWorst 5 windows:")
print(results.nsmallest(5, 'test_sharpe')[['window', 'test_sharpe', 'train_sharpe']])

# %%
