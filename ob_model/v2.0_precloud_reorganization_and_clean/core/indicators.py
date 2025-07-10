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

logger = logging.getLogger(__name__)

# =============================================================================
# CUSTOM INDICATOR IMPLEMENTATIONS
# =============================================================================

def calculate_supertrend(high: pd.Series, low: pd.Series, close: pd.Series, 
                        period: int = 10, multiplier: float = 3.0) -> pd.DataFrame:
    """Calculate SuperTrend indicator"""
    try:
        # Calculate ATR
        hl_avg = (high + low) / 2
        atr = AverageTrueRange(high=high, low=low, close=close, window=period).average_true_range()
        
        # Calculate basic bands
        upper_basic = hl_avg + (multiplier * atr)
        lower_basic = hl_avg - (multiplier * atr)
        
        # Calculate final bands
        upper_band = upper_basic.copy()
        lower_band = lower_basic.copy()
        supertrend = pd.Series(index=close.index, dtype=float)
        direction = pd.Series(index=close.index, dtype=float)
        
        for i in range(period, len(close)):
            # Upper band
            if upper_basic.iloc[i] < upper_band.iloc[i-1] or close.iloc[i-1] > upper_band.iloc[i-1]:
                upper_band.iloc[i] = upper_basic.iloc[i]
            else:
                upper_band.iloc[i] = upper_band.iloc[i-1]
            
            # Lower band
            if lower_basic.iloc[i] > lower_band.iloc[i-1] or close.iloc[i-1] < lower_band.iloc[i-1]:
                lower_band.iloc[i] = lower_basic.iloc[i]
            else:
                lower_band.iloc[i] = lower_band.iloc[i-1]
            
            # SuperTrend
            if i == period:
                if close.iloc[i] <= upper_band.iloc[i]:
                    supertrend.iloc[i] = upper_band.iloc[i]
                    direction.iloc[i] = -1
                else:
                    supertrend.iloc[i] = lower_band.iloc[i]
                    direction.iloc[i] = 1
            else:
                if supertrend.iloc[i-1] == upper_band.iloc[i-1]:
                    if close.iloc[i] <= upper_band.iloc[i]:
                        supertrend.iloc[i] = upper_band.iloc[i]
                        direction.iloc[i] = -1
                    else:
                        supertrend.iloc[i] = lower_band.iloc[i]
                        direction.iloc[i] = 1
                else:
                    if close.iloc[i] >= lower_band.iloc[i]:
                        supertrend.iloc[i] = lower_band.iloc[i]
                        direction.iloc[i] = 1
                    else:
                        supertrend.iloc[i] = upper_band.iloc[i]
                        direction.iloc[i] = -1
        
        return pd.DataFrame({
            'SuperTrend': supertrend,
            'SuperTrend_Direction': direction
        })
    except Exception as e:
        logger.warning(f"SuperTrend calculation failed: {e}")
        return pd.DataFrame({
            'SuperTrend': pd.Series(index=close.index, dtype=float),
            'SuperTrend_Direction': pd.Series(index=close.index, dtype=float)
        })

def calculate_kst(close: pd.Series) -> pd.Series:
    """Calculate Know Sure Thing (KST) indicator"""
    try:
        # ROC periods and weights
        roc_periods = [10, 15, 20, 30]
        sma_periods = [10, 10, 10, 15]
        weights = [1, 2, 3, 4]
        
        # Calculate weighted ROC
        kst = pd.Series(0, index=close.index, dtype=float)
        for period, sma_period, weight in zip(roc_periods, sma_periods, weights):
            roc = close.pct_change(periods=period) * 100
            roc_sma = roc.rolling(window=sma_period).mean()
            kst += roc_sma * weight
        
        # Signal line (9-period SMA of KST)
        kst_signal = kst.rolling(window=9).mean()
        
        return kst
    except Exception as e:
        logger.warning(f"KST calculation failed: {e}")
        return pd.Series(index=close.index, dtype=float)

def calculate_dpo(close: pd.Series, period: int = 20) -> pd.Series:
    """Calculate Detrended Price Oscillator"""
    try:
        # Calculate SMA shifted back
        shift_period = int(period / 2) + 1
        sma = close.rolling(window=period).mean()
        dpo = close - sma.shift(-shift_period)
        return dpo
    except Exception as e:
        logger.warning(f"DPO calculation failed: {e}")
        return pd.Series(index=close.index, dtype=float)

def calculate_linear_regression_slope(close: pd.Series, window: int = 20) -> pd.Series:
    """Calculate Linear Regression Slope"""
    try:
        def linreg_slope(values):
            if len(values) < 2:
                return np.nan
            x = np.arange(len(values))
            try:
                slope, _ = np.polyfit(x, values, 1)
                return slope
            except:
                return np.nan
        
        return close.rolling(window).apply(linreg_slope, raw=True)
    except Exception as e:
        logger.warning(f"Linear Regression Slope calculation failed: {e}")
        return pd.Series(index=close.index, dtype=float)

def calculate_correlation_coefficient(close: pd.Series, window: int = 20) -> pd.Series:
    """Calculate Correlation Coefficient (price vs time)"""
    try:
        def correlation(values):
            if len(values) < 2:
                return np.nan
            x = np.arange(len(values))
            return np.corrcoef(x, values)[0, 1]
        
        return close.rolling(window).apply(correlation, raw=True)
    except Exception as e:
        logger.warning(f"Correlation calculation failed: {e}")
        return pd.Series(index=close.index, dtype=float)

def calculate_acceleration(close: pd.Series, period: int = 10) -> pd.Series:
    """Calculate price acceleration (second derivative)"""
    try:
        velocity = close.pct_change(periods=period)
        acceleration = velocity.diff(periods=period)
        return acceleration
    except Exception as e:
        logger.warning(f"Acceleration calculation failed: {e}")
        return pd.Series(index=close.index, dtype=float)

def calculate_jerk(close: pd.Series, period: int = 10) -> pd.Series:
    """Calculate price jerk (third derivative)"""
    try:
        velocity = close.pct_change(periods=period)
        acceleration = velocity.diff(periods=period)
        jerk = acceleration.diff(periods=period)
        return jerk
    except Exception as e:
        logger.warning(f"Jerk calculation failed: {e}")
        return pd.Series(index=close.index, dtype=float)

# =============================================================================
# VOLATILITY ESTIMATORS
# =============================================================================

def calculate_parkinson_estimator(high: pd.Series, low: pd.Series, window: int = 20) -> pd.Series:
    """Parkinson Volatility Estimator"""
    try:
        log_ratio = np.log(high / low) ** 2
        parkinson = np.sqrt(log_ratio.rolling(window).mean() / (4 * np.log(2))) * np.sqrt(252)
        return parkinson
    except Exception as e:
        logger.warning(f"Parkinson Estimator calculation failed: {e}")
        return pd.Series(index=high.index, dtype=float)

def calculate_garman_klass(open_: pd.Series, high: pd.Series, low: pd.Series, 
                          close: pd.Series, window: int = 20) -> pd.Series:
    """Garman-Klass Volatility Estimator"""
    try:
        log_hl = np.log(high / low) ** 2
        log_co = np.log(close / open_) ** 2
        
        gk = np.sqrt(
            0.5 * log_hl.rolling(window).mean() - 
            (2 * np.log(2) - 1) * log_co.rolling(window).mean()
        ) * np.sqrt(252)
        
        return gk
    except Exception as e:
        logger.warning(f"Garman-Klass calculation failed: {e}")
        return pd.Series(index=close.index, dtype=float)

def calculate_rogers_satchell(open_: pd.Series, high: pd.Series, low: pd.Series, 
                             close: pd.Series, window: int = 20) -> pd.Series:
    """Rogers-Satchell Volatility Estimator"""
    try:
        log_hc = np.log(high / close)
        log_ho = np.log(high / open_)
        log_lc = np.log(low / close)
        log_lo = np.log(low / open_)
        
        rs = np.sqrt(
            (log_hc * log_ho + log_lc * log_lo).rolling(window).mean() * 252
        )
        
        return rs
    except Exception as e:
        logger.warning(f"Rogers-Satchell calculation failed: {e}")
        return pd.Series(index=close.index, dtype=float)

def calculate_yang_zhang(open_: pd.Series, high: pd.Series, low: pd.Series, 
                        close: pd.Series, window: int = 20) -> pd.Series:
    """Yang-Zhang Volatility Estimator"""
    try:
        # Overnight volatility
        log_oc = np.log(open_ / close.shift(1))
        overnight_var = log_oc.rolling(window).var()
        
        # Open-to-close volatility
        log_co = np.log(close / open_)
        openclose_var = log_co.rolling(window).var()
        
        # Rogers-Satchell volatility
        rs = calculate_rogers_satchell(open_, high, low, close, window)
        rs_var = (rs ** 2) / 252
        
        # Yang-Zhang estimator
        k = 0.34 / (1.34 + (window + 1) / (window - 1))
        yz = np.sqrt(overnight_var + k * openclose_var + (1 - k) * rs_var) * np.sqrt(252)
        
        return yz
    except Exception as e:
        logger.warning(f"Yang-Zhang calculation failed: {e}")
        return pd.Series(index=close.index, dtype=float)

# =============================================================================
# MICROSTRUCTURE INDICATORS
# =============================================================================

def calculate_vpt(close: pd.Series, volume: pd.Series) -> pd.Series:
    """Calculate Volume Price Trend"""
    try:
        price_change = close.pct_change()
        vpt = (volume * price_change).cumsum()
        return vpt
    except Exception as e:
        logger.warning(f"VPT calculation failed: {e}")
        return pd.Series(index=close.index, dtype=float)

def calculate_vwap(high: pd.Series, low: pd.Series, close: pd.Series, 
                   volume: pd.Series, window: int = 20) -> pd.Series:
    """Calculate Volume Weighted Average Price"""
    try:
        typical_price = (high + low + close) / 3
        vwap = (typical_price * volume).rolling(window).sum() / volume.rolling(window).sum()
        return vwap
    except Exception as e:
        logger.warning(f"VWAP calculation failed: {e}")
        return pd.Series(index=close.index, dtype=float)

def calculate_cvd(close: pd.Series, volume: pd.Series, window: int = 20) -> pd.Series:
    """Calculate Cumulative Volume Delta"""
    try:
        price_direction = np.sign(close.diff())
        volume_delta = volume * price_direction
        cvd = volume_delta.rolling(window).sum()
        return cvd
    except Exception as e:
        logger.warning(f"CVD calculation failed: {e}")
        return pd.Series(index=close.index, dtype=float)

# =============================================================================
# TREND ANALYSIS INDICATORS
# =============================================================================

def calculate_trend_consistency(close: pd.Series, window: int = 20) -> pd.Series:
    """Calculate trend consistency (% of positive price changes)"""
    try:
        price_changes = close.diff()
        positive_changes = (price_changes > 0).rolling(window).sum()
        consistency = positive_changes / window
        return consistency
    except Exception as e:
        logger.warning(f"Trend Consistency calculation failed: {e}")
        return pd.Series(index=close.index, dtype=float)

def calculate_ma_alignment(close: pd.Series, windows: list = [5, 10, 20, 50]) -> pd.Series:
    """Calculate Moving Average Alignment Score"""
    try:
        mas = {}
        for w in windows:
            mas[w] = close.rolling(w).mean()
        
        alignment_scores = pd.Series(index=close.index, dtype=float)
        
        for i in range(len(close)):
            ma_values = []
            for w in windows:
                if i >= w - 1:  # Ensure we have enough data
                    ma_values.append(mas[w].iloc[i])
            
            if len(ma_values) < 2:
                alignment_scores.iloc[i] = 0
                continue
            
            # Check if MAs are in ascending or descending order
            ascending = all(ma_values[j] <= ma_values[j+1] for j in range(len(ma_values)-1))
            descending = all(ma_values[j] >= ma_values[j+1] for j in range(len(ma_values)-1))
            
            if ascending:
                alignment_scores.iloc[i] = 1  # Bullish alignment
            elif descending:
                alignment_scores.iloc[i] = -1  # Bearish alignment
            else:
                alignment_scores.iloc[i] = 0  # Mixed alignment
        
        return alignment_scores
    except Exception as e:
        logger.warning(f"MA Alignment calculation failed: {e}")
        return pd.Series(index=close.index, dtype=float)

# =============================================================================
# MAIN INDICATOR CALCULATION FUNCTION - FULL VERSION
# =============================================================================

def calculate_all_indicators(data: pd.DataFrame, 
                           verbose: bool = False) -> pd.DataFrame:
    """
    Calculate all technical indicators - FULL VERSION with 85+ indicators
    
    Args:
        data: DataFrame with OHLCV data
        verbose: Whether to print progress
        
    Returns:
        DataFrame with all indicators added
    """
    logger.info("Starting FULL indicator calculations (85+ indicators)...")
    
    # Create a copy to avoid modifying original
    df = data.copy()
    
    # Ensure we have required columns
    required = ['open', 'high', 'low', 'close']
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    # Check for volume
    has_volume = 'volume' in df.columns
    if not has_volume:
        logger.warning("Volume column not found - volume indicators will be limited")
        df['volume'] = 1.0  # Dummy volume
    
    # Extract price series
    open_ = df['open']
    high = df['high']
    low = df['low']
    close = df['close']
    volume = df['volume']
    
    try:
        # ======================
        # DIRECTION INDICATORS (20+)
        # ======================
        if verbose: print("Calculating Direction indicators (20+)...")
        
        # Simple Moving Averages
        for period in [5, 10, 20, 50, 100, 200]:
            df[f'SMA_{period}'] = SMAIndicator(close=close, window=period).sma_indicator()
        
        # Exponential Moving Averages
        for period in [9, 12, 20, 26, 50]:
            df[f'EMA_{period}'] = EMAIndicator(close=close, window=period).ema_indicator()
        
        # MACD
        macd = MACD(close=close)
        df['MACD'] = macd.macd()
        df['MACD_signal'] = macd.macd_signal()
        df['MACD_histogram'] = macd.macd_diff()
        
        # ADX
        # Fill NaN in high/low/close before ADX
        high = df['high'].fillna(method='ffill').fillna(0)  # or dropna if preferred
        low = df['low'].fillna(method='ffill').fillna(0)
        close = df['close'].fillna(method='ffill').fillna(0)

        adx = ADXIndicator(high=high, low=low, close=close)
        df['ADX'] = adx.adx()
        df['DI_plus'] = adx.adx_pos()
        df['DI_minus'] = adx.adx_neg()
        
        # Aroon
        aroon = AroonIndicator(high=high, low=low, window=25)
        df['Aroon_up'] = aroon.aroon_up()
        df['Aroon_down'] = aroon.aroon_down()
        df['Aroon_indicator'] = aroon.aroon_indicator()
        
        # CCI
        df['CCI'] = CCIIndicator(high=high, low=low, close=close).cci()
        
        # Ichimoku
        ichimoku = IchimokuIndicator(high=high, low=low)
        df['Ichimoku_A'] = ichimoku.ichimoku_a()
        df['Ichimoku_B'] = ichimoku.ichimoku_b()
        df['Ichimoku_base'] = ichimoku.ichimoku_base_line()
        df['Ichimoku_conversion'] = ichimoku.ichimoku_conversion_line()
        
        # PSAR
        psar = PSARIndicator(high=high, low=low, close=close)
        df['PSAR'] = psar.psar()
        
        # Vortex
        vortex = VortexIndicator(high=high, low=low, close=close)
        df['Vortex_Pos'] = vortex.vortex_indicator_pos()
        df['Vortex_Neg'] = vortex.vortex_indicator_neg()
        
        # Custom Direction Indicators
        supertrend_result = calculate_supertrend(high, low, close)
        df['SuperTrend'] = supertrend_result['SuperTrend']
        df['SuperTrend_Direction'] = supertrend_result['SuperTrend_Direction']
        
        df['DPO'] = calculate_dpo(close)
        df['KST'] = calculate_kst(close)

        # ======================
        # VOLATILITY INDICATORS (13+)
        # ======================
        if verbose: print("Calculating Volatility indicators (13+)...")
        
        # ATR
        atr = AverageTrueRange(high=high, low=low, close=close)
        df['ATR'] = atr.average_true_range()
        # df['NATR'] = (df['ATR'] / close) * 100 # Removed due to high correlation
        
        # Bollinger Bands
        bb = BollingerBands(close=close)
        df['BB_Upper'] = bb.bollinger_hband()
        df['BB_Lower'] = bb.bollinger_lband()
        # df['BB_Middle'] = bb.bollinger_mavg()
        df['BB_Width'] = bb.bollinger_wband()
        df['BB_Percent'] = bb.bollinger_pband()
        
        # Keltner Channel
        kc = KeltnerChannel(high=high, low=low, close=close)
        df['KC_Upper'] = kc.keltner_channel_hband()
        df['KC_Lower'] = kc.keltner_channel_lband()
        # df['KC_Middle'] = kc.keltner_channel_mband()
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
        df['Historical_Vol'] = close.pct_change().rolling(20).std() * np.sqrt(252)
        df['Parkinson'] = calculate_parkinson_estimator(high, low)
        df['GarmanKlass'] = calculate_garman_klass(open_, high, low, close)
        df['RogersSatchell'] = calculate_rogers_satchell(open_, high, low, close)
        df['YangZhang'] = calculate_yang_zhang(open_, high, low, close)

        # ======================
        # TREND STRENGTH INDICATORS (15+)
        # ======================
        if verbose: print("Calculating Trend Strength indicators (15+)...")
        
        # Linear Regression
        df['LinearReg_Slope'] = calculate_linear_regression_slope(close)
        df['Correlation'] = calculate_correlation_coefficient(close)
        
        # TSI
        df['TSI'] = TSIIndicator(close=close).tsi()
        
        # Trend Consistency
        df['Trend_Consistency'] = calculate_trend_consistency(close)
        
        # MA Alignment
        df['MA_Alignment'] = calculate_ma_alignment(close)
        
        # Choppiness Index (simplified version)
        df['Choppiness'] = 100 * np.log10(df['ATR'].rolling(14).sum() / (df['high'].rolling(14).max() - df['low'].rolling(14).min())) / np.log10(14)
        
        # ======================
        # VELOCITY INDICATORS (12+)
        # ======================
        if verbose: print("Calculating Velocity indicators (12+)...")
        
        # ROC
        for period in [5, 10, 20]:
            df[f'ROC_{period}'] = ROCIndicator(close=close, window=period).roc()
        
        # RSI
        for period in [14, 21]:
            df[f'RSI_{period}'] = RSIIndicator(close=close, window=period).rsi()
        
        # Momentum
        df['Momentum_10'] = close.diff(10)
        df['Momentum_20'] = close.diff(20)
        
        # Acceleration and Jerk
        df['Acceleration'] = calculate_acceleration(close)
        df['Jerk'] = calculate_jerk(close)
        
        # ======================
        # MICROSTRUCTURE INDICATORS (12+)
        # ======================
        if verbose: print("Calculating Microstructure indicators (12+)...")
        
        if has_volume:
            # Volume-based indicators
            df['OBV'] = OnBalanceVolumeIndicator(close=close, volume=volume).on_balance_volume()
            df['CMF'] = ChaikinMoneyFlowIndicator(high=high, low=low, close=close, volume=volume).chaikin_money_flow()
            df['MFI'] = MFIIndicator(high=high, low=low, close=close, volume=volume).money_flow_index()
            df['ADI'] = AccDistIndexIndicator(high=high, low=low, close=close, volume=volume).acc_dist_index()
            df['EOM'] = EaseOfMovementIndicator(high=high, low=low, volume=volume).ease_of_movement()
            df['FI'] = ForceIndexIndicator(close=close, volume=volume).force_index()
            
            # Custom Volume Indicators
            df['VPT'] = calculate_vpt(close, volume)
            df['VWAP'] = calculate_vwap(high, low, close, volume)
            df['CVD'] = calculate_cvd(close, volume)
            df['Delta'] = volume * np.sign(close.diff())
            
            # Volume ratios
            df['Volume_SMA'] = volume.rolling(20).mean()
            df['Volume_Ratio'] = volume / df['Volume_SMA']
        
        # ======================
        # MOMENTUM INDICATORS (Additional)
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
        # df['Log_Returns'] = np.log(close / close.shift(1)) # Removed due to high correlation
        df['Cumulative_Returns'] = (1 + df['Returns']).cumprod() - 1
        
        # Price features
        df['High_Low_Ratio'] = high / low
        # df['Close_Open_Ratio'] = close / open_ # Removed due to high correlation
        df['Typical_Price'] = (high + low + close) / 3
        df['Weighted_Close'] = (high + low + close * 2) / 4
        
        # Fill any NaN values with forward fill then backward fill
        df = df.fillna(method='ffill').fillna(method='bfill')
        
        # Final indicator count
        indicator_count = len(df.columns) - len(data.columns)
        logger.info(f"Calculated {indicator_count} indicators successfully")
        
        # Log indicator summary by dimension
        if verbose:
            print(f"\nIndicator Summary:")
            print(f"  Direction: 20+ indicators")
            print(f"  Trend Strength: 15+ indicators")
            print(f"  Velocity: 12+ indicators")
            print(f"  Volatility: 13+ indicators")
            print(f"  Microstructure: 12+ indicators")
            print(f"  Total: {indicator_count} indicators")
        
    except Exception as e:
        logger.error(f"Error in indicator calculation: {e}")
        raise
    
    return df

