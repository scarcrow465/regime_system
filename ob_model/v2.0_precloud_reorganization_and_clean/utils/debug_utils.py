#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import logging
import pandas as pd
from typing import Any
from utils.logger import get_logger  # For logger in safe_save

def check_data_sanity(df: pd.DataFrame, logger: logging.Logger, module_name: str) -> pd.DataFrame:
    if df.empty:
        logger.error(f"{module_name}: Empty dataframe—check data_loader.py")
        raise ValueError("Empty data")
    if df.isnull().any().any():
        logger.warning(f"{module_name}: NaNs found—filling with 0")
        df = df.fillna(0)
    return df

def log_var_state(var_name: str, var_value: Any, logger: logging.Logger, level: str = 'DEBUG') -> None:
    if logger.level <= logging.DEBUG:
        logger.debug(f"{var_name}: {str(var_value)[:200]}...")

def safe_save(fig: Any, base_path: str, extension: str = 'png') -> str:
    from datetime import datetime
    import os
    
    logger = get_logger('safe_save')  # Fixed: Define logger for safe_save
    
    dir_name = os.path.dirname(base_path)
    os.makedirs(dir_name, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    file_path = f"{base_path}_{timestamp}.{extension}"
    
    if extension == 'png':
        fig.savefig(file_path)
    elif extension == 'csv':
        fig.to_csv(file_path)
    elif extension == 'txt':
        with open(file_path, 'w') as f:
            f.write(fig)
    
    logger.info(f"Saved {extension.upper()} file: {file_path}")
    return file_path

