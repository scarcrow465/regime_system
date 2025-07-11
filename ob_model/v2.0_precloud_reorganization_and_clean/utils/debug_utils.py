#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import logging
import pandas as pd
from typing import Any  # Added: For Any in function hints (fixes NameError)—like opening the toolbox for labels

def check_data_sanity(df: pd.DataFrame, logger: logging.Logger, module_name: str) -> pd.DataFrame:
    """
    Runs sanity checks on data—e.g., no NaNs, valid scopes.
    - Why: Catches issues early (like past NaNs), logs warnings.
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

def log_var_state(var_name: str, var_value: Any, logger: logging.Logger, level: str = 'DEBUG') -> None:
    """
    Logs variable states for deep debug—only if level=DEBUG.
    - Why: Helps trace "what went wrong" without manual prints.
    - Example: log_var_state('edge_map', edge_map, logger)
    """
    if logger.level <= logging.DEBUG:
        logger.debug(f"{var_name}: {str(var_value)[:200]}...")  # Truncate long vars

def safe_save(fig: Any, base_path: str, extension: str = 'png') -> str:
    """
    Safe save for files (PNG, CSV, TXT)—creates dir, adds timestamp, logs path.
    - Input: fig (plt.figure for PNG, df for CSV, str for TXT), base_path (e.g., 'docs/plots/category_evolution').
    - Output: file_path saved.
    - Why: Future-proof—no errors on missing dir/overwrite; timestamp for versions (e.g., _2025-07-11_10-45.png).
    - Use: In evolver: safe_save(plt.gcf(), 'docs/plots/category_evolution')—gcf gets current figure.
    """
    from datetime import datetime  # For timestamp
    import os  # For dir creation
    
    dir_name = os.path.dirname(base_path)
    os.makedirs(dir_name, exist_ok=True)  # Create dir if missing
    
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M')
    file_path = f"{base_path}_{timestamp}.{extension}"
    
    if extension == 'png':
        fig.savefig(file_path)
    elif extension == 'csv':
        fig.to_csv(file_path)  # Assume fig is df
    elif extension == 'txt':
        with open(file_path, 'w') as f:
            f.write(fig)  # Assume fig is str
    # Add more for other types
    
    logger.info(f"Saved {extension.upper()} file: {file_path}")
    return file_path

