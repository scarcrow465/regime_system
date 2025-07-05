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
from tvDatafeed import TvDatafeed, Interval
import os
import matplotlib.pyplot as plt
import numpy as np

def backtest_strategy(data, direction='both', spx_sma=None, spx_close=None, use_spx_filter=True):
    """
    Backtest the trading strategy with an optional SPX500USD filter.
    Returns a dictionary with performance metrics and trades for equity curve plotting.
    """
    assert direction in ['long', 'short', 'both'], f"Invalid direction: {direction}"
    trades = []
    
    # Iterate through data, starting from day 1 and ensuring room for 10-day exits
    for n in range(1, len(data) - 11):
        prev_day = data.index[n-1]
        # Apply SPX filter if enabled and data is available
        if use_spx_filter and spx_sma is not None and spx_close is not None:
            if prev_day not in spx_sma.index or prev_day not in spx_close.index:
                continue
            if not (spx_close.loc[prev_day] > spx_sma.loc[prev_day]):
                continue
        
        if direction in ['long', 'both']:
            # Long entry: close[n-1] < low[n-2] and high[n] > high[n-1]
            if (data['close'].iloc[n-1] < data['low'].iloc[n-2] and 
                data['high'].iloc[n] > data['high'].iloc[n-1]):
                entry_day = n + 1
                entry_price = data['open'].iloc[entry_day]
                # Find exit: first profitable close or after 10 days
                for k in range(entry_day, min(entry_day + 10, len(data))):
                    if data['close'].iloc[k] > entry_price:
                        exit_day = k
                        break
                else:
                    exit_day = min(entry_day + 9, len(data) - 1)
                exit_price = data['close'].iloc[exit_day]
                profit = (exit_price - entry_price) / entry_price
                win = 1 if profit > 0 else 0
                trades.append({
                    'type': 'long',
                    'entry_day': data.index[entry_day],
                    'entry_price': entry_price,
                    'exit_day': data.index[exit_day],
                    'exit_price': exit_price,
                    'profit': profit,
                    'win': win
                })
        
        if direction in ['short', 'both']:
            # Short entry: close[n-1] > high[n-2] and low[n] < low[n-1]
            if (data['close'].iloc[n-1] > data['high'].iloc[n-2] and 
                data['low'].iloc[n] < data['low'].iloc[n-1]):
                entry_day = n + 1
                entry_price = data['open'].iloc[entry_day]
                # Find exit: first profitable close or after 10 days
                for k in range(entry_day, min(entry_day + 10, len(data))):
                    if data['close'].iloc[k] < entry_price:
                        exit_day = k
                        break
                else:
                    exit_day = min(entry_day + 9, len(data) - 1)
                exit_price = data['close'].iloc[exit_day]
                profit = (entry_price - exit_price) / entry_price
                win = 1 if profit > 0 else 0
                trades.append({
                    'type': 'short',
                    'entry_day': data.index[entry_day],
                    'entry_price': entry_price,
                    'exit_day': data.index[exit_day],
                    'exit_price': exit_price,
                    'profit': profit,
                    'win': win
                })
    
    # Calculate metrics
    total_trades = len(trades)
    if total_trades > 0:
        overall_win_rate = sum(t['win'] for t in trades) / total_trades
        sum_positive = sum(t['profit'] for t in trades if t['profit'] > 0)
        sum_negative = sum(abs(t['profit']) for t in trades if t['profit'] < 0)
        profit_factor = sum_positive / sum_negative if sum_negative > 0 else float('inf')
    else:
        overall_win_rate = None
        profit_factor = None
    
    long_trades = [t for t in trades if t['type'] == 'long']
    if long_trades:
        long_win_rate = sum(t['win'] for t in long_trades) / len(long_trades)
        long_positive = sum(t['profit'] for t in long_trades if t['profit'] > 0)
        long_negative = sum(abs(t['profit']) for t in long_trades if t['profit'] < 0)
        long_profit_factor = long_positive / long_negative if long_negative > 0 else float('inf')
    else:
        long_win_rate = None
        long_profit_factor = None
    
    short_trades = [t for t in trades if t['type'] == 'short']
    if short_trades:
        short_win_rate = sum(t['win'] for t in short_trades) / len(short_trades)
        short_positive = sum(t['profit'] for t in short_trades if t['profit'] > 0)
        short_negative = sum(abs(t['profit']) for t in short_trades if t['profit'] < 0)
        short_profit_factor = short_positive / short_negative if short_negative > 0 else float('inf')
    else:
        short_win_rate = None
        short_profit_factor = None
    
    metrics = {
        'overall_win_rate': overall_win_rate,
        'long_win_rate': long_win_rate,
        'short_win_rate': short_win_rate,
        'total_trades': total_trades,
        'long_trades': len(long_trades),
        'short_trades': len(short_trades),
        'profit_factor': profit_factor,
        'long_profit_factor': long_profit_factor,
        'short_profit_factor': short_profit_factor
    }
    
    return {'metrics': metrics, 'trades': trades}

def plot_equity_curve(data, trades, instrument, initial_capital=10000, trade_size=1000):
    """
    Plot the equity curve for the given instrument based on the trades.
    Saves the plot as a PNG file.
    """
    equity_curve = []
    realized_profit = 0
    for day in data.index:
        # Sum realized profits from closed trades
        closed_trades = [t for t in trades if t['exit_day'] <= day]
        realized_profit = sum(t['profit'] * trade_size for t in closed_trades)
        
        # Sum unrealized profits from open trades
        open_trades = [t for t in trades if t['entry_day'] <= day < t['exit_day']]
        unrealized_profit = 0
        for t in open_trades:
            if t['type'] == 'long':
                up = (data['close'].loc[day] - t['entry_price']) / t['entry_price'] * trade_size
            elif t['type'] == 'short':
                up = (t['entry_price'] - data['close'].loc[day]) / t['entry_price'] * trade_size
            unrealized_profit += up
        
        equity = initial_capital + realized_profit + unrealized_profit
        equity_curve.append(equity)
    
    # Plot the equity curve
    plt.figure(figsize=(10, 6))
    plt.plot(data.index, equity_curve, label='Equity Curve')
    plt.title(f'Equity Curve for {instrument}')
    plt.xlabel('Date')
    plt.ylabel('Equity ($)')
    plt.grid(True)
    plt.legend()
    plt.savefig(f'{instrument}_equity_curve.png')
    plt.close()

# Initialize TvDatafeed
username = os.getenv('TV_USERNAME')
password = os.getenv('TV_PASSWORD')
if username and password:
    tv = TvDatafeed(username, password)
    print("Logged into TvDatafeed.")
else:
    tv = TvDatafeed()
    print("Using TvDatafeed without login, data may be limited.")

# Fetch SPX500USD data and calculate 50-day SMA
print("Fetching SPX500USD data for filter...")
spx_data = tv.get_hist(symbol='SPX500USD', exchange='OANDA', interval=Interval.in_daily, n_bars=5000)
if spx_data is None or len(spx_data) < 100:
    print("Insufficient SPX500USD data.")
    exit()
spx_data['SMA50'] = spx_data['close'].rolling(window=50).mean()

# Define instruments to analyze with direction and SPX filter toggle
instruments = [
    {'symbol': 'NAS100USD', 'exchange': 'OANDA', 'direction': 'both', 'use_spx_filter': False},
    {'symbol': 'EURUSD', 'exchange': 'OANDA', 'direction': 'long', 'use_spx_filter': True},
    {'symbol': 'XAUUSD', 'exchange': 'OANDA', 'direction': 'long', 'use_spx_filter': False},
    {'symbol': 'WTICOUSD', 'exchange': 'OANDA', 'direction': 'long', 'use_spx_filter': False},
    {'symbol': 'CORNUSD', 'exchange': 'OANDA', 'direction': 'long', 'use_spx_filter': True},
    {'symbol': 'USB10YUSD', 'exchange': 'OANDA', 'direction': 'both', 'use_spx_filter': False},
]

# Analyze each instrument
results = []
for instrument in instruments:
    symbol = instrument['symbol']
    exchange = instrument['exchange']
    direction = instrument['direction']
    use_spx_filter = instrument['use_spx_filter']
    print(f"Processing {symbol} with direction: {direction}, SPX filter: {use_spx_filter}...")
    try:
        # Fetch instrument data
        data = tv.get_hist(symbol=symbol, exchange=exchange, interval=Interval.in_daily, n_bars=5000)
        if data is None or len(data) < 100:
            print(f"Insufficient data for {symbol}")
            continue
        
        # Merge with SPX data using an inner join
        merged_data = data.merge(spx_data[['close', 'SMA50']], how='inner', left_index=True, right_index=True, suffixes=('', '_spx'))
        
        # Run backtest with SPX filter
        result = backtest_strategy(
            merged_data, 
            direction=direction, 
            spx_sma=merged_data['SMA50'], 
            spx_close=merged_data['close_spx'], 
            use_spx_filter=use_spx_filter
        )
        metrics = result['metrics']
        trades = result['trades']
        metrics['instrument'] = symbol
        results.append(metrics)
        print(f"Completed {symbol}: {metrics['total_trades']} trades")
        # Plot equity curve
        plot_equity_curve(merged_data, trades, symbol)
    except Exception as e:
        print(f"Error processing {symbol}: {e}")

# Display results
if results:
    results_df = pd.DataFrame(results)
    columns = ['instrument', 'overall_win_rate', 'long_win_rate', 'short_win_rate', 
               'total_trades', 'long_trades', 'short_trades', 'profit_factor', 
               'long_profit_factor', 'short_profit_factor']
    results_df = results_df[columns]
    
    # Create a display DataFrame with formatted values
    display_df = results_df.copy()
    for col in ['overall_win_rate', 'long_win_rate', 'short_win_rate', 'profit_factor', 'long_profit_factor', 'short_profit_factor']:
        display_df[col] = display_df[col].apply(lambda x: 'N/A' if pd.isna(x) else ('inf' if x == float('inf') else round(x, 2)))
    
    print("\nResults:")
    print(display_df.to_string(index=False))
else:
    print("No results to display.")

# %%
