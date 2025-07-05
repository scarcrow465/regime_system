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
"""
Technical indicators calculation module
Contains all 85+ indicators organized by dimension
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Union
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
# CUSTOM INDICATOR FUNCTIONS
# =============================================================================

def calculate_supertrend(high: pd.Series, low: pd.Series, close: pd.Series, 
                        period: int = 10, multiplier: float = 3.0) -> pd.DataFrame:
    """Calculate SuperTrend indicator"""
    try:
        atr = AverageTrueRange(high=high, low=low, close=close, window=period).average_true_range()
        hl_avg = (high + low) / 2
        
        upper_band = hl_avg + (multiplier * atr)
        lower_band = hl_avg - (multiplier * atr)
        
        supertrend = pd.Series(index=close.index, dtype=float)
        direction = pd.Series(index=close.index, dtype=float)
        
        for i in range(period, len(close)):
            if close.iloc[i] <= upper_band.iloc[i]:
                if i > 0 and supertrend.iloc[i-1] == lower_band.iloc[i-1]:
                    supertrend.iloc[i] = max(lower_band.iloc[i], lower_band.iloc[i-1])
                else:
                    supertrend.iloc[i] = lower_band.iloc[i]
                direction.iloc[i] = 1
            else:
                if i > 0 and supertrend.iloc[i-1] == upper_band.iloc[i-1]:
                    supertrend.iloc[i] = min(upper_band.iloc[i], upper_band.iloc[i-1])
                else:
                    supertrend.iloc[i] = upper_band.iloc[i]
                direction.iloc[i] = -1
        
        return pd.DataFrame({
            'SuperTrend': supertrend,
            'SuperTrend_Direction': direction
        })
    except Exception as e:
        logger.error(f"Error calculating SuperTrend: {e}")
        return pd.DataFrame({
            'SuperTrend': pd.Series(index=close.index, dtype=float),
            'SuperTrend_Direction': pd.Series(index=close.index, dtype=float)
        })

def calculate_dpo(close: pd.Series, period: int = 20) -> pd.Series:
    """Calculate Detrended Price Oscillator"""
    try:
        sma = close.rolling(window=period).mean()
        dpo = close.shift(int(period/2 + 1)) - sma
        return dpo
    except Exception as e:
        logger.error(f"Error calculating DPO: {e}")
        return pd.Series(index=close.index, dtype=float)

def calculate_kst(close: pd.Series) -> pd.Series:
    """Calculate Know Sure Thing (KST) indicator"""
    try:
        rocma1 = ROCIndicator(close, window=10).roc().rolling(10).mean()
        rocma2 = ROCIndicator(close, window=15).roc().rolling(10).mean()
        rocma3 = ROCIndicator(close, window=20).roc().rolling(10).mean()
        rocma4 = ROCIndicator(close, window=30).roc().rolling(15).mean()
        
        kst = (rocma1 * 1) + (rocma2 * 2) + (rocma3 * 3) + (rocma4 * 4)
        return kst
    except Exception as e:
        logger.error(f"Error calculating KST: {e}")
        return pd.Series(index=close.index, dtype=float)

def calculate_linear_regression_slope(close: pd.Series, period: int = 20) -> pd.Series:
    """Calculate linear regression slope"""
    try:
        def lr_slope(values):
            if len(values) < 2:
                return np.nan
            x = np.arange(len(values))
            slope, _ = np.polyfit(x, values, 1)
            return slope
        
        return close.rolling(window=period).apply(lr_slope, raw=True)
    except Exception as e:
        logger.error(f"Error calculating Linear Regression Slope: {e}")
        return pd.Series(index=close.index, dtype=float)

def calculate_price_correlation(close: pd.Series, period: int = 20) -> pd.Series:
    """Calculate price correlation coefficient"""
    try:
        def correlation(values):
            if len(values) < 2:
                return np.nan
            x = np.arange(len(values))
            return np.corrcoef(x, values)[0, 1]
        
        return close.rolling(window=period).apply(correlation, raw=True)
    except Exception as e:
        logger.error(f"Error calculating Price Correlation: {e}")
        return pd.Series(index=close.index, dtype=float)

def calculate_acceleration(close: pd.Series, period: int = 10) -> pd.Series:
    """Calculate price acceleration"""
    try:
        velocity = close.pct_change(period)
        acceleration = velocity.diff()
        return acceleration
    except Exception as e:
        logger.error(f"Error calculating Acceleration: {e}")
        return pd.Series(index=close.index, dtype=float)

def calculate_jerk(close: pd.Series, period: int = 10) -> pd.Series:
    """Calculate price jerk (rate of change of acceleration)"""
    try:
        velocity = close.pct_change(period)
        acceleration = velocity.diff()
        jerk = acceleration.diff()
        return jerk
    except Exception as e:
        logger.error(f"Error calculating Jerk: {e}")
        return pd.Series(index=close.index, dtype=float)

def calculate_historical_volatility(close: pd.Series, period: int = 20) -> pd.Series:
    """Calculate historical volatility"""
    try:
        returns = close.pct_change()
        hvol = returns.rolling(window=period).std() * np.sqrt(252)
        return hvol
    except Exception as e:
        logger.error(f"Error calculating Historical Volatility: {e}")
        return pd.Series(index=close.index, dtype=float)

def calculate_parkinson_volatility(high: pd.Series, low: pd.Series, period: int = 20) -> pd.Series:
    """Calculate Parkinson volatility estimator"""
    try:
        hl_ratio = np.log(high / low)
        parkinson = hl_ratio.rolling(window=period).apply(
            lambda x: np.sqrt(np.sum(x**2) / (4 * len(x) * np.log(2)))
        ) * np.sqrt(252)
        return parkinson
    except Exception as e:
        logger.error(f"Error calculating Parkinson Volatility: {e}")
        return pd.Series(index=high.index, dtype=float)

def calculate_garman_klass_volatility(open_: pd.Series, high: pd.Series, 
                                    low: pd.Series, close: pd.Series, 
                                    period: int = 20) -> pd.Series:
    """Calculate Garman-Klass volatility estimator"""
    try:
        log_hl = np.log(high / low)
        log_co = np.log(close / open_)
        
        gk = pd.Series(index=close.index, dtype=float)
        
        for i in range(period-1, len(close)):
            window_hl = log_hl.iloc[i-period+1:i+1]
            window_co = log_co.iloc[i-period+1:i+1]
            
            term1 = np.sum(0.5 * window_hl**2)
            term2 = np.sum((2 * np.log(2) - 1) * window_co**2)
            
            gk.iloc[i] = np.sqrt((term1 - term2) / period) * np.sqrt(252)
        
        return gk
    except Exception as e:
        logger.error(f"Error calculating Garman-Klass Volatility: {e}")
        return pd.Series(index=close.index, dtype=float)

def calculate_rogers_satchell_volatility(open_: pd.Series, high: pd.Series, 
                                       low: pd.Series, close: pd.Series, 
                                       period: int = 20) -> pd.Series:
    """Calculate Rogers-Satchell volatility estimator"""
    try:
        log_ho = np.log(high / open_)
        log_hc = np.log(high / close)
        log_lo = np.log(low / open_)
        log_lc = np.log(low / close)
        
        rs = pd.Series(index=close.index, dtype=float)
        
        for i in range(period-1, len(close)):
            window_calc = (log_ho.iloc[i-period+1:i+1] * log_hc.iloc[i-period+1:i+1] + 
                          log_lo.iloc[i-period+1:i+1] * log_lc.iloc[i-period+1:i+1])
            rs.iloc[i] = np.sqrt(np.sum(window_calc) / period) * np.sqrt(252)
        
        return rs
    except Exception as e:
        logger.error(f"Error calculating Rogers-Satchell Volatility: {e}")
        return pd.Series(index=close.index, dtype=float)

def calculate_yang_zhang_volatility(open_: pd.Series, high: pd.Series, 
                                  low: pd.Series, close: pd.Series, 
                                  period: int = 20) -> pd.Series:
    """Calculate Yang-Zhang volatility estimator"""
    try:
        k = 0.34 / (1.34 + (period + 1) / (period - 1))
        
        log_ho = np.log(high / open_)
        log_lo = np.log(low / open_)
        log_co = np.log(close / open_)
        
        log_oc = np.log(open_ / close.shift(1))
        log_oc_mean = log_oc.rolling(window=period).mean()
        
        log_cc = np.log(close / close.shift(1))
        log_cc_mean = log_cc.rolling(window=period).mean()
        
        yz = pd.Series(index=close.index, dtype=float)
        
        for i in range(period-1, len(close)):
            window_oc = log_oc.iloc[i-period+1:i+1]
            window_cc = log_cc.iloc[i-period+1:i+1]
            
            overnight_var = np.sum((window_oc - log_oc_mean.iloc[i])**2) / (period - 1)
            close_var = np.sum((window_cc - log_cc_mean.iloc[i])**2) / (period - 1)
            
            window_rs = (log_ho.iloc[i-period+1:i+1] * (log_ho.iloc[i-period+1:i+1] - log_co.iloc[i-period+1:i+1]) + 
                        log_lo.iloc[i-period+1:i+1] * (log_lo.iloc[i-period+1:i+1] - log_co.iloc[i-period+1:i+1]))
            rs_var = np.sum(window_rs) / period
            
            yz.iloc[i] = np.sqrt(overnight_var + k * close_var + (1 - k) * rs_var) * np.sqrt(252)
        
        return yz
    except Exception as e:
        logger.error(f"Error calculating Yang-Zhang Volatility: {e}")
        return pd.Series(index=close.index, dtype=float)

def calculate_vpt(close: pd.Series, volume: pd.Series) -> pd.Series:
    """Calculate Volume Price Trend"""
    try:
        price_change = close.pct_change()
        vpt = (volume * price_change).cumsum()
        return vpt
    except Exception as e:
        logger.error(f"Error calculating VPT: {e}")
        return pd.Series(index=close.index, dtype=float)

def calculate_vwap(high: pd.Series, low: pd.Series, close: pd.Series, 
                   volume: pd.Series, period: int = 20) -> pd.Series:
    """Calculate Volume Weighted Average Price"""
    try:
        typical_price = (high + low + close) / 3
        vwap = (typical_price * volume).rolling(window=period).sum() / volume.rolling(window=period).sum()
        return vwap
    except Exception as e:
        logger.error(f"Error calculating VWAP: {e}")
        return pd.Series(index=close.index, dtype=float)

def calculate_cvd(close: pd.Series, volume: pd.Series, period: int = 20) -> pd.Series:
    """Calculate Cumulative Volume Delta"""
    try:
        price_change = close.diff()
        volume_delta = volume * np.sign(price_change)
        cvd = volume_delta.rolling(window=period).sum()
        return cvd
    except Exception as e:
        logger.error(f"Error calculating CVD: {e}")
        return pd.Series(index=close.index, dtype=float)

def calculate_delta(close: pd.Series, volume: pd.Series) -> pd.Series:
    """Calculate simple delta (buying vs selling pressure)"""
    try:
        price_change = close.diff()
        delta = volume * np.sign(price_change)
        return delta
    except Exception as e:
        logger.error(f"Error calculating Delta: {e}")
        return pd.Series(index=close.index, dtype=float)

# =============================================================================
# MAIN INDICATOR CALCULATION FUNCTION
# =============================================================================

def calculate_all_indicators(data: pd.DataFrame, 
                           verbose: bool = False) -> pd.DataFrame:
    """
    Calculate all 85+ technical indicators
    
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
    volume = df['volume'] if 'volume' in df.columns else pd.Series(1, index=df.index)
    
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
        
        # Aroon
        aroon = AroonIndicator(close=close)
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
        df['PSAR_Up'] = psar.psar_up()
        df['PSAR_Down'] = psar.psar_down()
        
        # Vortex
        vortex = VortexIndicator(high=high, low=low, close=close)
        df['Vortex_Pos'] = vortex.vortex_indicator_pos()
        df['Vortex_Neg'] = vortex.vortex_indicator_neg()
        
        # Custom Direction Indicators
        supertrend_df = calculate_supertrend(high, low, close)
        df['SuperTrend'] = supertrend_df['SuperTrend']
        df['SuperTrend_Direction'] = supertrend_df['SuperTrend_Direction']
        
        df['DPO'] = calculate_dpo(close)
        df['KST'] = calculate_kst(close)
        
        # ======================
        # TREND STRENGTH INDICATORS
        # ======================
        if verbose: print("Calculating Trend Strength indicators...")
        
        # ADX is already calculated
        # Linear Regression
        df['LinearReg_Slope'] = calculate_linear_regression_slope(close)
        df['Correlation'] = calculate_price_correlation(close)
        
        # TSI
        df['TSI'] = TSIIndicator(close=close).tsi()
        
        # ======================
        # VELOCITY INDICATORS
        # ======================
        if verbose: print("Calculating Velocity indicators...")
        
        # ROC
        df['ROC'] = ROCIndicator(close=close).roc()
        
        # RSI (also used for velocity)
        df['RSI'] = RSIIndicator(close=close).rsi()
        
        # Acceleration and Jerk
        df['Acceleration'] = calculate_acceleration(close)
        df['Jerk'] = calculate_jerk(close)
        
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
        
        # Advanced Volatility Estimators
        df['Historical_Vol'] = calculate_historical_volatility(close)
        df['Parkinson'] = calculate_parkinson_volatility(high, low)
        df['GarmanKlass'] = calculate_garman_klass_volatility(open_, high, low, close)
        df['RogersSatchell'] = calculate_rogers_satchell_volatility(open_, high, low, close)
        df['YangZhang'] = calculate_yang_zhang_volatility(open_, high, low, close)
        
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
        
        # Custom Volume Indicators
        df['VPT'] = calculate_vpt(close, volume)
        df['VWAP'] = calculate_vwap(high, low, close, volume)
        df['CVD'] = calculate_cvd(close, volume, period=20)
        df['Delta'] = calculate_delta(close, volume)
        
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
        
        logger.info(f"Calculated {len(df.columns) - len(data.columns)} indicators")
        
    except Exception as e:
        logger.error(f"Error in indicator calculation: {e}")
        raise
    
    return df

# =============================================================================
# INDICATOR VALIDATION
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
    
    # Expected indicators by dimension
    expected_indicators = {
        'direction': ['SMA_Signal', 'EMA_Signal', 'MACD_Signal', 'ADX', 'Aroon_Oscillator', 
                     'CCI', 'PSAR', 'Vortex_Pos', 'SuperTrend_Direction', 'DPO', 'KST'],
        'trend_strength': ['ADX', 'Aroon_Up', 'CCI', 'MACD_Histogram', 'RSI', 'TSI', 
                          'LinearReg_Slope', 'Correlation'],
        'velocity': ['ROC', 'RSI', 'TSI', 'MACD_Histogram', 'Acceleration', 'Jerk'],
        'volatility': ['ATR', 'BB_Width', 'KC_Width', 'DC_Width', 'NATR', 'UI', 
                      'Historical_Vol', 'Parkinson', 'GarmanKlass', 'RogersSatchell', 'YangZhang'],
        'microstructure': ['OBV', 'CMF', 'MFI', 'ADI', 'EOM', 'FI', 'VPT', 'VWAP', 'CVD', 'Delta']
    }
    
    # Check all expected indicators
    all_expected = set()
    for indicators in expected_indicators.values():
        all_expected.update(indicators)
    
    for indicator in all_expected:
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

# =============================================================================
# INDICATOR INFORMATION
# =============================================================================

def get_indicator_info() -> Dict[str, Dict[str, List[str]]]:
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
