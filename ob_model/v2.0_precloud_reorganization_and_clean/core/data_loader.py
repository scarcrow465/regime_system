#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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


"""
Data loading and preprocessing functionality
Handles CSV loading, data validation, and preprocessing
"""

import pandas as pd
import numpy as np
import logging
from typing import List, Optional, Union, Dict, Tuple
from datetime import datetime
import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import (
    DATA_DIR, TIMEFRAMES, DEFAULT_SYMBOLS,
    LOG_LEVEL, LOG_FORMAT
)

# Configure logging
logging.basicConfig(level=getattr(logging, LOG_LEVEL), format=LOG_FORMAT)
logger = logging.getLogger(__name__)

def load_csv_data(filepath: str, 
                  parse_dates: bool = True,
                  date_column: str = 'Date',
                  timeframe: Optional[str] = None) -> pd.DataFrame:
    """
    Load CSV data with automatic date parsing, column standardization, and reshaping from wide to long format
    
    Args:
        filepath: Path to CSV file
        parse_dates: Whether to parse date column
        date_column: Name of date column
        timeframe: Optional timeframe hint (e.g., '15min', '1H', 'Daily')
        
    Returns:
        DataFrame with standardized columns and datetime index in long format
    """
    try:
        logger.info(f"Loading data from: {filepath}")
        
        # Load the CSV
        if parse_dates:
            df = pd.read_csv(filepath, parse_dates=[date_column])
        else:
            df = pd.read_csv(filepath)
        
        # Standardize column names
        df.columns = [col.strip().lower() for col in df.columns]
        
        # Handle date column
        date_col = date_column.lower()
        if date_col in df.columns:
            df['date'] = pd.to_datetime(df[date_col])
            df.set_index('date', inplace=True)
            if date_col != 'date':
                df.drop(columns=[date_col], inplace=True)
        
        # Reshape wide-format data to long format
        dfs = []
        i = 0
        while True:
            suffix = f'.{i}' if i > 0 else ''
            symbol_col = f'symbol{suffix}'
            required_cols = [symbol_col, f'open{suffix}', f'high{suffix}', f'low{suffix}', f'close{suffix}']
            if not all(col in df.columns for col in required_cols):
                break
            
            temp_df = pd.DataFrame(index=df.index)
            temp_df['symbol'] = df[symbol_col]
            temp_df['open'] = df[f'open{suffix}']
            temp_df['high'] = df[f'high{suffix}']
            temp_df['low'] = df[f'low{suffix}']
            temp_df['close'] = df[f'close{suffix}']
            temp_df['volume'] = df[f'volume{suffix}'] if f'volume{suffix}' in df.columns else 0
            temp_df['openinterest'] = df[f'openinterest{suffix}'] if f'openinterest{suffix}' in df.columns else 0
            
            # Drop rows where symbol is NaN
            temp_df = temp_df.dropna(subset=['symbol'])
            if not temp_df.empty:
                dfs.append(temp_df)
            
            i += 1
        
        if dfs:
            df = pd.concat(dfs, ignore_index=False)
        else:
            raise ValueError("No valid symbol data found after reshaping")
        
        # Sort by date
        df.sort_index(inplace=True)
        
        # Check required columns
        required_columns = ['open', 'high', 'low', 'close']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            logger.warning(f"Missing columns: {missing_columns}")
            # Try to handle missing high/low
            if 'high' not in df.columns and 'close' in df.columns:
                df['high'] = df['close']
            if 'low' not in df.columns and 'close' in df.columns:
                df['low'] = df['close']
        
        # Infer timeframe if not provided
        if timeframe is None and len(df) > 1:
            timeframe = infer_timeframe(df)
            logger.info(f"Inferred timeframe: {timeframe}")
        
        # Add timeframe info
        df.attrs['timeframe'] = timeframe
        
        # Basic data quality checks
        df = validate_and_clean_data(df)
        
        logger.info(f"Loaded {len(df)} rows from {df.index[0]} to {df.index[-1]}")
        logger.info(f"Columns: {list(df.columns)}")
        
        return df
        
    except Exception as e:
        logger.error(f"Error loading CSV data: {e}")
        raise

def infer_timeframe(df: pd.DataFrame) -> str:
    """
    Infer timeframe from data frequency
    
    Args:
        df: DataFrame with datetime index
        
    Returns:
        Timeframe string (e.g., '15min', '1H', 'Daily')
    """
    if len(df) < 2:
        return 'Unknown'
    
    # Calculate median time difference
    time_diffs = df.index[1:] - df.index[:-1]
    median_diff = pd.Timedelta(np.median(time_diffs.total_seconds()), unit='s')
    
    # Map to standard timeframes
    if median_diff <= pd.Timedelta(minutes=6):
        return '5min'
    elif median_diff <= pd.Timedelta(minutes=20):
        return '15min'
    elif median_diff <= pd.Timedelta(minutes=35):
        return '30min'
    elif median_diff <= pd.Timedelta(hours=1.5):
        return '1H'
    elif median_diff <= pd.Timedelta(hours=5):
        return '4H'
    elif median_diff <= pd.Timedelta(days=2):
        return 'Daily'
    else:
        return 'Weekly'

def validate_and_clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate and clean price data
    
    Args:
        df: DataFrame with OHLC data
        
    Returns:
        Cleaned DataFrame
    """
    initial_rows = len(df)
    
    # Remove duplicates
    df = df[~df.index.duplicated(keep='first')]
    
    # Remove rows with any NaN in OHLC
    ohlc_cols = ['open', 'high', 'low', 'close']
    existing_ohlc = [col for col in ohlc_cols if col in df.columns]
    df = df.dropna(subset=existing_ohlc)
    
    # Validate price relationships
    if all(col in df.columns for col in ['open', 'high', 'low', 'close']):
        # High should be >= Low
        invalid_hl = df['high'] < df['low']
        if invalid_hl.any():
            logger.warning(f"Found {invalid_hl.sum()} rows with high < low, fixing...")
            df.loc[invalid_hl, 'high'] = df.loc[invalid_hl, ['open', 'close']].max(axis=1)
            df.loc[invalid_hl, 'low'] = df.loc[invalid_hl, ['open', 'close']].min(axis=1)
    
    # Remove rows with zero or negative prices
    for col in existing_ohlc:
        df = df[df[col] > 0]
    
    final_rows = len(df)
    if final_rows < initial_rows:
        logger.info(f"Data cleaning removed {initial_rows - final_rows} rows")
    
    return df

def load_multiple_files(filepaths: List[str], 
                       concat: bool = True) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    Load multiple CSV files
    
    Args:
        filepaths: List of file paths
        concat: Whether to concatenate into single DataFrame
        
    Returns:
        Single concatenated DataFrame or dictionary of DataFrames
    """
    data_dict = {}
    
    for filepath in filepaths:
        filename = os.path.basename(filepath).split('.')[0]
        try:
            df = load_csv_data(filepath)
            data_dict[filename] = df
        except Exception as e:
            logger.error(f"Failed to load {filepath}: {e}")
    
    if concat and len(data_dict) > 1:
        # Concatenate all DataFrames
        all_data = pd.concat(data_dict.values(), sort=True)
        all_data.sort_index(inplace=True)
        return all_data
    else:
        return data_dict

def resample_data(df: pd.DataFrame, 
                  target_timeframe: str,
                  method: str = 'ohlc') -> pd.DataFrame:
    """
    Resample data to different timeframe
    
    Args:
        df: DataFrame with OHLC data
        target_timeframe: Target timeframe (e.g., '1H', '4H', 'D')
        method: Resampling method ('ohlc' or 'last')
        
    Returns:
        Resampled DataFrame
    """
    # Convert timeframe to pandas frequency
    freq_map = {
        '5min': '5T',
        '15min': '15T',
        '30min': '30T',
        '1H': '1H',
        '4H': '4H',
        'Daily': 'D',
        'Weekly': 'W'
    }
    
    if target_timeframe not in freq_map:
        raise ValueError(f"Unknown timeframe: {target_timeframe}")
    
    freq = freq_map[target_timeframe]
    
    if method == 'ohlc':
        # Resample OHLC properly
        resampled = df.resample(freq).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum' if 'volume' in df.columns else None
        }).dropna()
    else:
        # Simple last value resampling
        resampled = df.resample(freq).last().dropna()
    
    return resampled

def prepare_data_for_analysis(df: pd.DataFrame,
                            min_periods: int = 100) -> pd.DataFrame:
    """
    Prepare data for regime analysis
    
    Args:
        df: Raw OHLC DataFrame
        min_periods: Minimum periods required
        
    Returns:
        Prepared DataFrame
    """
    if len(df) < min_periods:
        raise ValueError(f"Insufficient data: {len(df)} rows, need at least {min_periods}")
    
    # Ensure we have all required columns
    if 'high' not in df.columns:
        df['high'] = df['close']
    if 'low' not in df.columns:
        df['low'] = df['close']
    
    # Add basic calculated fields
    df['returns'] = df['close'].pct_change()
    df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
    
    # Add session info if intraday
    if df.attrs.get('timeframe') in ['5min', '15min', '30min', '1H']:
        df['hour'] = df.index.hour
        df['day_of_week'] = df.index.dayofweek
    
    return df

def save_processed_data(df: pd.DataFrame, 
                       filepath: str,
                       format: str = 'csv') -> None:
    """
    Save processed data to file
    
    Args:
        df: DataFrame to save
        filepath: Output file path
        format: Output format ('csv', 'parquet', 'pickle')
    """
    try:
        if format == 'csv':
            df.to_csv(filepath)
        elif format == 'parquet':
            df.to_parquet(filepath)
        elif format == 'pickle':
            df.to_pickle(filepath)
        else:
            raise ValueError(f"Unknown format: {format}")
        
        logger.info(f"Saved {len(df)} rows to {filepath}")
        
    except Exception as e:
        logger.error(f"Error saving data: {e}")
        raise

# Utility functions for data information
def get_data_info(df: pd.DataFrame) -> Dict[str, any]:
    """Get summary information about the data"""
    info = {
        'rows': len(df),
        'columns': list(df.columns),
        'start_date': df.index[0],
        'end_date': df.index[-1],
        'timeframe': df.attrs.get('timeframe', 'Unknown'),
        'missing_values': df.isnull().sum().to_dict(),
        'memory_usage': df.memory_usage().sum() / 1024**2  # MB
    }
    return info

def check_data_quality(df: pd.DataFrame) -> Dict[str, any]:
    """Run data quality checks"""
    issues = []
    
    # Check for gaps
    if df.attrs.get('timeframe') in ['5min', '15min', '30min', '1H']:
        expected_freq = df.index.to_series().diff().mode()[0]
        gaps = df.index.to_series().diff()[1:] > expected_freq * 2
        if gaps.any():
            issues.append(f"Found {gaps.sum()} time gaps in data")
    
    # Check for outliers
    for col in ['open', 'high', 'low', 'close']:
        if col in df.columns:
            z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
            outliers = z_scores > 5
            if outliers.any():
                issues.append(f"Found {outliers.sum()} outliers in {col}")
    
    return {
        'issues': issues,
        'quality_score': 1.0 - len(issues) / 10.0
    }


# In[ ]:




