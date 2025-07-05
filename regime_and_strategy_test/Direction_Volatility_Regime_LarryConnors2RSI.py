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
from tvDatafeed import TvDatafeed, Interval
import pandas as pd
import numpy as np
import talib

# Initialize TVDatafeed (use your TradingView credentials if needed; here we use the free version)
tv = TvDatafeed()

# Function to fetch data from TradingView
def fetch_data(symbol, exchange, n_bars=500):
    """
    Fetch historical daily data for a given symbol and exchange.
    
    Args:
        symbol (str): Instrument symbol (e.g., 'AAPL')
        exchange (str): Exchange name (e.g., 'NASDAQ')
        n_bars (int): Number of bars to fetch (default: 5000)
    
    Returns:
        pd.DataFrame: Historical data
    """
    try:
        data = tv.get_hist(symbol=symbol, exchange=exchange, interval=Interval.in_daily, n_bars=n_bars)
        if data is None or data.empty:
            print(f"No data retrieved for {symbol} on {exchange}")
            return None
        return data
    except Exception as e:
        print(f"Error fetching data for {symbol}: {e}")
        return None

# Function to calculate direction indicators and classify direction
def calculate_direction(df):
    """
    Calculate nine direction indicators and determine market direction via voting.
    
    Args:
        df (pd.DataFrame): DataFrame with OHLC data
    
    Returns:
        pd.DataFrame: DataFrame with direction classification
    """
    # Calculate indicators
    df['sma10'] = talib.SMA(df['close'], timeperiod=10)
    df['sma30'] = talib.SMA(df['close'], timeperiod=30)
    df['ema10'] = talib.EMA(df['close'], timeperiod=10)
    df['ema30'] = talib.EMA(df['close'], timeperiod=30)
    macd, signal, _ = talib.MACD(df['close'], fastperiod=12, slowperiod=26, signalperiod=9)
    df['macd'] = macd
    df['macd_signal'] = signal
    df['rsi'] = talib.RSI(df['close'], timeperiod=14)
    df['adx'] = talib.ADX(df['high'], df['low'], df['close'], timeperiod=14)
    df['plus_di'] = talib.PLUS_DI(df['high'], df['low'], df['close'], timeperiod=14)
    df['minus_di'] = talib.MINUS_DI(df['high'], df['low'], df['close'], timeperiod=14)
    df['sar'] = talib.SAR(df['high'], df['low'])
    df['stoch_k'], df['stoch_d'] = talib.STOCH(df['high'], df['low'], df['close'], fastk_period=14, slowk_period=3, slowd_period=3)
    df['linreg_slope'] = talib.LINEARREG_SLOPE(df['close'], timeperiod=20)

    # Define direction votes for each indicator
    df['dir_sma'] = np.where(df['sma10'] > df['sma30'] * 1.005, 'up', 
                            np.where(df['sma10'] < df['sma30'] * 0.995, 'down', 'sideways'))
    df['dir_ema'] = np.where(df['ema10'] > df['ema30'] * 1.005, 'up', 
                            np.where(df['ema10'] < df['ema30'] * 0.995, 'down', 'sideways'))
    df['dir_macd'] = np.where(df['macd'] > df['macd_signal'], 'up', 
                             np.where(df['macd'] < df['macd_signal'], 'down', 'sideways'))
    df['dir_rsi'] = pd.cut(df['rsi'], bins=[0, 40, 60, 100], labels=['down', 'sideways', 'up'], right=False)
    df['dir_adx'] = np.where((df['adx'] > 25) & (df['plus_di'] > df['minus_di']), 'up',
                            np.where((df['adx'] > 25) & (df['plus_di'] < df['minus_di']), 'down',
                                    np.where(df['adx'] < 20, 'sideways', 'nan')))
    df['dir_sar'] = np.where(df['sar'] < df['close'], 'up', 'down')
    df['dir_stoch'] = np.where((df['stoch_k'] > df['stoch_d']) & (df['stoch_k'] > 50), 'up',
                              np.where((df['stoch_k'] < df['stoch_d']) & (df['stoch_k'] < 50), 'down', 'sideways'))
    df['dir_linreg'] = np.where(df['linreg_slope'] > 0, 'up', 
                               np.where(df['linreg_slope'] < 0, 'down', 'sideways'))

    # List of direction vote columns
    dir_cols = ['dir_sma', 'dir_ema', 'dir_macd', 'dir_rsi', 'dir_adx', 'dir_sar', 'dir_stoch', 'dir_linreg']

    # Determine final direction by plurality vote
    df['direction'] = df[dir_cols].mode(axis=1)[0]
    return df

# Function to calculate volatility indicators and classify volatility
def calculate_volatility(df):
    """
    Calculate nine volatility indicators and determine market volatility via voting.
    
    Args:
        df (pd.DataFrame): DataFrame with OHLC data
    
    Returns:
        pd.DataFrame: DataFrame with volatility classification
    """
    # Calculate volatility indicators
    df['atr14'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
    df['atr50'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=50)
    df['std_close20'] = df['close'].rolling(20).std()
    df['std_close50'] = df['close'].rolling(50).std()
    upper, middle, lower = talib.BBANDS(df['close'], timeperiod=20, nbdevup=2, nbdevdn=2)
    df['bb_width'] = (upper - lower) / middle
    df['chaikin_vol'] = talib.ROC(talib.EMA(df['high'] - df['low'], timeperiod=10), timeperiod=10)
    df['hist_vol20'] = df['close'].pct_change().rolling(20).std() * np.sqrt(252)
    df['hist_vol50'] = df['close'].pct_change().rolling(50).std() * np.sqrt(252)
    df['avg_range20'] = (df['high'] - df['low']).rolling(20).mean()

    # List of volatility indicators (fixed line)
    vol_indicators = ['atr14', 'atr50', 'std_close20', 'std_close50', 'bb_width', 'chaikin_vol', 'hist_vol20', 'hist_vol50', 'avg_range20']

    # Classify each indicator based on historical percentiles (30th and 70th)
    for ind in vol_indicators:
        df[f'{ind}_p30'] = df[ind].rolling(252).quantile(0.3)
        df[f'{ind}_p70'] = df[ind].rolling(252).quantile(0.7)
        df[f'vol_{ind}'] = np.where(df[ind] > df[f'{ind}_p70'], 'high',
                                   np.where(df[ind] < df[f'{ind}_p30'], 'low', 'average'))

    # List of volatility vote columns
    vol_cols = [f'vol_{ind}' for ind in vol_indicators]

    # Determine final volatility by plurality vote
    df['volatility'] = df[vol_cols].mode(axis=1)[0]
    return df

# Function to determine market regime
def determine_regime(df):
    """
    Combine direction and volatility into a market regime.
    
    Args:
        df (pd.DataFrame): DataFrame with direction and volatility
    
    Returns:
        pd.DataFrame: DataFrame with regime column
    """
    df['regime'] = df['direction'] + '-' + df['volatility']
    return df

# Function to calculate 2-Period RSI signals
def calculate_rsi_signals(df):
    """
    Calculate 2-Period RSI signals based on Larry Connors' strategy.
    
    Args:
        df (pd.DataFrame): DataFrame with OHLC data
    
    Returns:
        pd.DataFrame: DataFrame with RSI signals
    """
    # Calculate 200-day SMA for trend filter
    df['sma200'] = talib.SMA(df['close'], timeperiod=200)
    # Calculate 5-day SMA for exit
    df['sma5'] = talib.SMA(df['close'], timeperiod=5)
    # Calculate 2-period RSI
    df['rsi2'] = talib.RSI(df['close'], timeperiod=2)

    # Generate buy/sell signals
    # Buy: Price > 200-day SMA and RSI2 < 10
    # Sell: Price < 200-day SMA and RSI2 > 90
    df['signal'] = np.where((df['close'] > df['sma200']) & (df['rsi2'] < 30), 1,  # Buy
                           np.where((df['close'] < df['sma200']) & (df['rsi2'] > 70), -1, 0))  # Sell

    return df

# Function to simulate trades based on 2-Period RSI strategy
def simulate_trades(df):
    """
    Implement Larry Connors' 2-Period RSI strategy and simulate trades, tracking regimes.
    
    Args:
        df (pd.DataFrame): DataFrame with OHLC, signal, and regime data
    
    Returns:
        pd.DataFrame: DataFrame of trades with regime and return
    """
    trades = []
    position = 0  # 0: no position, 1: long, -1: short
    entry_price = 0
    entry_regime = ''
    entry_date = None

    for i, row in df.iterrows():
        # Entry: Buy (1) or Sell (-1)
        if row['signal'] == 1 and position == 0:  # Buy signal
            position = 1
            entry_price = row['close']
            entry_regime = row['regime']
            entry_date = i
        elif row['signal'] == -1 and position == 0:  # Sell signal
            position = -1
            entry_price = row['close']
            entry_regime = row['regime']
            entry_date = i

        # Exit: Long exit when close > 5-day SMA, Short exit when close < 5-day SMA
        if position == 1 and row['close'] > row['sma5']:  # Exit long
            trade_return = (row['close'] - entry_price) / entry_price
            trades.append({'entry_date': entry_date, 'exit_date': i, 'regime': entry_regime, 'return': trade_return})
            position = 0
        elif position == -1 and row['close'] < row['sma5']:  # Exit short
            trade_return = (entry_price - row['close']) / entry_price  # Short return
            trades.append({'entry_date': entry_date, 'exit_date': i, 'regime': entry_regime, 'return': trade_return})
            position = 0

    return pd.DataFrame(trades)

# Function to calculate performance metrics
def calculate_metrics(trades_df):
    """
    Calculate performance metrics including win rate and average return.
    
    Args:
        trades_df (pd.DataFrame): DataFrame of trades
    
    Returns:
        dict: Metrics including overall and per-regime win rates and average returns
    """
    if trades_df.empty:
        return {
            'overall_win_rate': np.nan,
            'overall_avg_return': np.nan,
            'regime_win_rates': {},
            'regime_avg_returns': {},
            'regime_trade_counts': {}
        }

    # Overall metrics
    total_trades = len(trades_df)
    overall_win_rate = (trades_df['return'] > 0).mean()
    overall_avg_return = trades_df['return'].mean()

    # Per-regime metrics
    regime_win_rates = trades_df.groupby('regime').apply(lambda x: (x['return'] > 0).mean() if len(x) > 0 else np.nan).to_dict()
    regime_avg_returns = trades_df.groupby('regime')['return'].mean().to_dict()
    regime_trade_counts = trades_df['regime'].value_counts().to_dict()

    return {
        'overall_win_rate': overall_win_rate,
        'overall_avg_return': overall_avg_return,
        'total_trades': total_trades,
        'regime_win_rates': regime_win_rates,
        'regime_avg_returns': regime_avg_returns,
        'regime_trade_counts': regime_trade_counts
    }

# Main function to analyze an instrument
def analyze_instrument(symbol, exchange):
    """
    Analyze a single instrument: fetch data, define regimes, run 2-Period RSI strategy, and calculate metrics.
    
    Args:
        symbol (str): Instrument symbol
        exchange (str): Exchange name
    """
    # Fetch data
    df = fetch_data(symbol, exchange, n_bars=1000)
    if df is None:
        return

    print(f"Instrument: {symbol} ({exchange})")
    print(f"Initial data rows fetched: {len(df)} (approx. {len(df)/252:.2f} years)")

    # Calculate direction, volatility, and regimes
    df = calculate_direction(df)
    df = calculate_volatility(df)
    df = determine_regime(df)

    # Calculate 2-Period RSI signals
    df = calculate_rsi_signals(df)

    # Drop rows with insufficient data (e.g., NaNs from indicators or regimes)
    df = df.dropna(subset=['regime', 'sma200', 'sma5', 'rsi2', 'signal'])
    print(f"Data rows after dropping NaNs: {len(df)} (approx. {len(df)/252:.2f} years)")

    # Simulate trades
    trades_df = simulate_trades(df)
    print(f"Number of trades: {len(trades_df)}")

    # Calculate performance metrics
    metrics = calculate_metrics(trades_df)

    # Print results
    print(f"Overall Win Rate: {metrics['overall_win_rate']:.2%}")
    print(f"Overall Average Trade Return: {metrics['overall_avg_return']:.2%}")
    print(f"Total Trades: {metrics['total_trades']}")
    print("Regime Performance:")
    for regime in sorted(metrics['regime_win_rates'].keys()):
        win_rate = metrics['regime_win_rates'].get(regime, np.nan)
        avg_return = metrics['regime_avg_returns'].get(regime, np.nan)
        count = metrics['regime_trade_counts'].get(regime, 0)
        print(f"  {regime}: Win Rate = {win_rate:.2%}, Avg Return = {avg_return:.2%} ({count} trades)")
    print("\n")

# Example usage with multiple instruments
if __name__ == "__main__":
    # Define a list of instruments to analyze
    instruments = [
        {'symbol': 'EURUSD', 'exchange': 'OANDA'},
        {'symbol': 'NAS100USD', 'exchange': 'OANDA'},
        {'symbol': 'XAUUSD', 'exchange': 'OANDA'},
        {'symbol': 'ES1!', 'exchange': 'CME_MINI'},
        {'symbol': 'CORNUSD', 'exchange': 'OANDA'},
        {'symbol': 'WTICOUSD', 'exchange': 'OANDA'},
        # Add more instruments as needed, e.g., {'symbol': 'ES1!', 'exchange': 'CME'} for S&P 500 futures
    ]

    # Run analysis for each instrument
    for inst in instruments:
        analyze_instrument(inst['symbol'], inst['exchange'])

# %%
