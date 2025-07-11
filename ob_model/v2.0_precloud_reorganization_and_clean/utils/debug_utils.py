#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import logging
import pandas as pd

def check_data_sanity(df, logger, module_name):
    """
    Runs sanity checks on data—e.g., no NaNs, valid scopes.
    - Why: Catches issues early (like past NaNs), logs warnings/errors.
    - Ties to vision: Ensures clean data for accurate edge extraction.
    """
    if df.empty:
        logger.error(f"{module_name}: Empty dataframe—check data_loader.py")
        raise ValueError("Empty data")
    if df.isnull().any().any():
        logger.warning(f"{module_name}: NaNs found—filling with 0")
        df = df.fillna(0)
    # Add more, e.g., for scopes: if hold <0: logger.error("Invalid scope")
    return df

def log_var_state(var_name, var_value, logger, level='DEBUG'):
    """
    Logs variable states for deep debug—only if level=DEBUG.
    - Why: Helps trace "what went wrong" without manual prints.
    - Example: log_var_state('edge_map', edge_map, logger)
    """
    if logger.level <= logging.DEBUG:
        logger.debug(f"{var_name}: {str(var_value)[:200]}...")  # Truncate long vars

# Usage: In edge_scanner.py, df = check_data_sanity(df, logger, 'edge_scanner')
# log_var_state('scores', edge_scores, logger)

