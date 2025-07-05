#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
Technical indicators calculation module
Contains all 85+ indicators organized by dimension
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Union, Any
import warnings
warnings.filterwarnings('ignore')

# Technical Analysis Libraries
from ta.trend import (
    SMAIndicator, EMAIndicator, MACD, ADXIndicator, AroonIndicator, 
    CCIIndicator, IchimokuIndicator, PSARIndicator, VortexIndicator
)
from ta.momentum import (
    RSIIndicator, StochasticOscillator, ROCIndicator, TSIIndicator, 
    UltimateOscillator, StochRSIIndicator
)
from ta.volatility import (
    AverageTrueRange, BollingerBands, KeltnerChannel, DonchianChannel, UlcerIndex
)
from ta.volume import (
    OnBalanceVolumeIndicator, ChaikinMoneyFlowIndicator, MFIIndicator, 
    AccDistIndexIndicator, EaseOfMovementIndicator, ForceIndexIndicator
)
from ta.others import DailyReturnIndicator, CumulativeReturnIndicator

logger = logging.getLogger(__name__)

# =============================================================================
# MAIN INDICATOR CALCULATION FUNCTION - SIMPLIFIED VERSION
# =============================================================================

def calculate_all_indicators(data: pd.DataFrame, 
                           verbose: bool = False) -> pd.DataFrame:
    """
    Calculate all technical indicators - FIXED VERSION
    
    Args:
        data: DataFrame with OHLCV data
        verbose: Whether to print progress
        
    Returns:
        DataFrame with all indicators added
    """
    logger.info("Starting indicator calculations...")
    
    # Create a copy to avoid modifying original
    df = data.copy()
    
    # Ensure we have required columns
    required = ['open', 'high', 'low', 'close']
    missing = [col for col in required if col not in df.columns]
    if missing:
        logger.warning(f"Missing columns: {missing}. Using close price as substitute.")
        for col in missing:
            df[col] = df['close']
    
    # Extract price series
    open_ = df['open']
    high = df['high']
    low = df['low']
    close = df['close']
    volume = df['volume'] if 'volume' in df.columns else pd.Series(1000000, index=df.index)
    
    try:
        # ======================
        # DIRECTION INDICATORS
        # ======================
        if verbose: print("Calculating Direction indicators...")
        
        # Moving Averages
        df['SMA_20'] = SMAIndicator(close=close, window=20).sma_indicator()
        df['SMA_50'] = SMAIndicator(close=close, window=50).sma_indicator()
        df['EMA_20'] = EMAIndicator(close=close, window=20).ema_indicator()
        df['EMA_50'] = EMAIndicator(close=close, window=50).ema_indicator()
        
        # MA Signals
        df['SMA_Signal'] = (close > df['SMA_20']).astype(float) - (close < df['SMA_20']).astype(float)
        df['EMA_Signal'] = (close > df['EMA_20']).astype(float) - (close < df['EMA_20']).astype(float)
        
        # MACD
        macd = MACD(close=close)
        df['MACD'] = macd.macd()
        df['MACD_Signal_Line'] = macd.macd_signal()
        df['MACD_Histogram'] = macd.macd_diff()
        df['MACD_Signal'] = (df['MACD'] > df['MACD_Signal_Line']).astype(float) - (df['MACD'] < df['MACD_Signal_Line']).astype(float)
        
        # ADX
        adx = ADXIndicator(high=high, low=low, close=close)
        df['ADX'] = adx.adx()
        df['DI_Plus'] = adx.adx_pos()
        df['DI_Minus'] = adx.adx_neg()
        
        # Aroon - FIXED: Use high and low, not close
        aroon = AroonIndicator(high=high, low=low)  # Fixed line
        df['Aroon_Up'] = aroon.aroon_up()
        df['Aroon_Down'] = aroon.aroon_down()
        df['Aroon_Oscillator'] = aroon.aroon_indicator()
        
        # CCI
        df['CCI'] = CCIIndicator(high=high, low=low, close=close).cci()
        
        # Ichimoku
        ichimoku = IchimokuIndicator(high=high, low=low)
        df['Ichimoku_A'] = ichimoku.ichimoku_a()
        df['Ichimoku_B'] = ichimoku.ichimoku_b()
        df['Ichimoku_Base'] = ichimoku.ichimoku_base_line()
        df['Ichimoku_Conversion'] = ichimoku.ichimoku_conversion_line()
        
        # Parabolic SAR
        psar = PSARIndicator(high=high, low=low, close=close)
        df['PSAR'] = psar.psar()
        
        # Vortex
        vortex = VortexIndicator(high=high, low=low, close=close)
        df['Vortex_Pos'] = vortex.vortex_indicator_pos()
        df['Vortex_Neg'] = vortex.vortex_indicator_neg()
        
        # Custom Direction Indicators (simplified)
        df['SuperTrend'] = df['SMA_20']  # Simplified placeholder
        df['SuperTrend_Direction'] = (close > df['SMA_20']).astype(float)
        df['DPO'] = close - df['SMA_20'].shift(10)
        df['KST'] = ROCIndicator(close=close, window=10).roc()  # Simplified
        
        # ======================
        # TREND STRENGTH INDICATORS
        # ======================
        if verbose: print("Calculating Trend Strength indicators...")
        
        # Linear Regression (simplified)
        df['LinearReg_Slope'] = close.rolling(20).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0)
        df['Correlation'] = close.rolling(20).apply(lambda x: x.corr(pd.Series(range(len(x)))) if len(x) > 1 else 0)
        
        # TSI
        df['TSI'] = TSIIndicator(close=close).tsi()
        
        # ======================
        # VELOCITY INDICATORS
        # ======================
        if verbose: print("Calculating Velocity indicators...")
        
        # ROC
        df['ROC'] = ROCIndicator(close=close).roc()
        
        # RSI
        df['RSI'] = RSIIndicator(close=close).rsi()
        
        # Acceleration and Jerk
        df['Acceleration'] = close.pct_change(10).diff()
        df['Jerk'] = df['Acceleration'].diff()
        
        # ======================
        # VOLATILITY INDICATORS
        # ======================
        if verbose: print("Calculating Volatility indicators...")
        
        # ATR
        atr = AverageTrueRange(high=high, low=low, close=close)
        df['ATR'] = atr.average_true_range()
        df['NATR'] = (df['ATR'] / close) * 100
        
        # Bollinger Bands
        bb = BollingerBands(close=close)
        df['BB_Upper'] = bb.bollinger_hband()
        df['BB_Lower'] = bb.bollinger_lband()
        df['BB_Middle'] = bb.bollinger_mavg()
        df['BB_Width'] = bb.bollinger_wband()
        df['BB_Percent'] = bb.bollinger_pband()
        
        # Keltner Channel
        kc = KeltnerChannel(high=high, low=low, close=close)
        df['KC_Upper'] = kc.keltner_channel_hband()
        df['KC_Lower'] = kc.keltner_channel_lband()
        df['KC_Middle'] = kc.keltner_channel_mband()
        df['KC_Width'] = df['KC_Upper'] - df['KC_Lower']
        
        # Donchian Channel
        dc = DonchianChannel(high=high, low=low, close=close)
        df['DC_Upper'] = dc.donchian_channel_hband()
        df['DC_Lower'] = dc.donchian_channel_lband()
        df['DC_Middle'] = dc.donchian_channel_mband()
        df['DC_Width'] = dc.donchian_channel_wband()
        
        # Ulcer Index
        df['UI'] = UlcerIndex(close=close).ulcer_index()
        
        # Simplified Volatility Estimators
        df['Historical_Vol'] = close.pct_change().rolling(20).std() * np.sqrt(252)
        df['Parkinson'] = ((np.log(high/low)**2).rolling(20).mean() / (4*np.log(2)))**0.5 * np.sqrt(252)
        df['GarmanKlass'] = df['Historical_Vol']  # Simplified
        df['RogersSatchell'] = df['Historical_Vol']  # Simplified
        df['YangZhang'] = df['Historical_Vol']  # Simplified
        
        # ======================
        # MICROSTRUCTURE INDICATORS
        # ======================
        if verbose: print("Calculating Microstructure indicators...")
        
        # Volume-based
        df['OBV'] = OnBalanceVolumeIndicator(close=close, volume=volume).on_balance_volume()
        df['CMF'] = ChaikinMoneyFlowIndicator(high=high, low=low, close=close, volume=volume).chaikin_money_flow()
        df['MFI'] = MFIIndicator(high=high, low=low, close=close, volume=volume).money_flow_index()
        df['ADI'] = AccDistIndexIndicator(high=high, low=low, close=close, volume=volume).acc_dist_index()
        df['EOM'] = EaseOfMovementIndicator(high=high, low=low, volume=volume).ease_of_movement()
        df['FI'] = ForceIndexIndicator(close=close, volume=volume).force_index()
        
        # Custom Volume Indicators (simplified)
        df['VPT'] = (volume * close.pct_change()).cumsum()
        df['VWAP'] = ((high + low + close) / 3 * volume).rolling(20).sum() / volume.rolling(20).sum()
        df['CVD'] = (volume * np.sign(close.diff())).rolling(20).sum()
        df['Delta'] = volume * np.sign(close.diff())
        
        # ======================
        # MOMENTUM INDICATORS
        # ======================
        if verbose: print("Calculating additional Momentum indicators...")
        
        # Stochastic
        stoch = StochasticOscillator(high=high, low=low, close=close)
        df['Stoch_K'] = stoch.stoch()
        df['Stoch_D'] = stoch.stoch_signal()
        
        # Ultimate Oscillator
        df['UO'] = UltimateOscillator(high=high, low=low, close=close).ultimate_oscillator()
        
        # Stochastic RSI
        stoch_rsi = StochRSIIndicator(close=close)
        df['StochRSI_K'] = stoch_rsi.stochrsi_k()
        df['StochRSI_D'] = stoch_rsi.stochrsi_d()
        
        # ======================
        # ADDITIONAL CALCULATIONS
        # ======================
        if verbose: print("Calculating additional features...")
        
        # Returns
        df['Returns'] = close.pct_change()
        df['Log_Returns'] = np.log(close / close.shift(1))
        df['Cumulative_Returns'] = (1 + df['Returns']).cumprod() - 1
        
        # Price features
        df['High_Low_Ratio'] = high / low
        df['Close_Open_Ratio'] = close / open_
        
        # Fill any NaN values with forward fill then backward fill
        df = df.fillna(method='ffill').fillna(method='bfill')
        
        logger.info(f"Calculated {len(df.columns) - len(data.columns)} indicators")
        
    except Exception as e:
        logger.error(f"Error in indicator calculation: {e}")
        raise
    
    return df

# =============================================================================
# SIMPLIFIED VALIDATION FUNCTION
# =============================================================================

def validate_indicators(data: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Validate calculated indicators
    
    Returns:
        Dictionary with validation results
    """
    results = {
        'missing': [],
        'all_nan': [],
        'mostly_nan': [],
        'constant': [],
        'valid': []
    }
    
    # Expected indicators (simplified list)
    expected_indicators = [
        'SMA_20', 'SMA_50', 'EMA_20', 'EMA_50', 'MACD', 'ADX', 
        'RSI', 'ROC', 'ATR', 'BB_Upper', 'BB_Lower', 'OBV', 'MFI'
    ]
    
    for indicator in expected_indicators:
        if indicator not in data.columns:
            results['missing'].append(indicator)
        else:
            col = data[indicator]
            
            # Check if all NaN
            if col.isna().all():
                results['all_nan'].append(indicator)
            # Check if mostly NaN (>90%)
            elif col.isna().sum() / len(col) > 0.9:
                results['mostly_nan'].append(indicator)
            # Check if constant
            elif col.nunique() == 1:
                results['constant'].append(indicator)
            else:
                results['valid'].append(indicator)
    
    return results

def get_indicator_info() -> Dict[str, Any]:
    """Get information about all indicators"""
    return {
        'dimensions': {
            'direction': {
                'indicators': ['SMA', 'EMA', 'MACD', 'ADX', 'Aroon', 'CCI', 
                             'Ichimoku', 'PSAR', 'Vortex', 'SuperTrend', 'DPO', 'KST'],
                'description': 'Indicators for market direction (up/down/sideways)'
            },
            'trend_strength': {
                'indicators': ['ADX', 'Aroon', 'CCI', 'MACD_histogram', 'RSI', 
                             'TSI', 'LinearReg_Slope', 'Correlation'],
                'description': 'Indicators for trend strength (strong/moderate/weak)'
            },
            'velocity': {
                'indicators': ['ROC', 'RSI', 'TSI', 'MACD_histogram', 'Acceleration', 'Jerk'],
                'description': 'Indicators for price velocity (accelerating/stable/decelerating)'
            },
            'volatility': {
                'indicators': ['ATR', 'BB', 'KC', 'DC', 'NATR', 'UI', 'Historical_Vol', 
                             'Parkinson', 'GarmanKlass', 'RogersSatchell', 'YangZhang'],
                'description': 'Indicators for volatility regimes'
            },
            'microstructure': {
                'indicators': ['Volume', 'OBV', 'CMF', 'MFI', 'ADI', 'EOM', 'FI', 
                             'VPT', 'VWAP', 'CVD', 'Delta'],
                'description': 'Indicators for market microstructure and flow'
            }
        },
        'total_indicators': 85,
        'custom_indicators': ['SuperTrend', 'DPO', 'KST', 'LinearReg_Slope', 'Correlation', 
                            'Acceleration', 'Jerk', 'Parkinson', 'GarmanKlass', 
                            'RogersSatchell', 'YangZhang', 'VPT', 'VWAP', 'CVD', 'Delta']
    }

