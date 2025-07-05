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

# Step 1: Initialize tvDatafeed (use nologin for simplicity)
tv = TvDatafeed()

# Step 2: Define instruments to scan (symbol, exchange pairs)
instruments = [
    {'symbol': 'EURUSD', 'exchange': 'OANDA'},
    {'symbol': 'NAS100USD', 'exchange': 'OANDA'},
    {'symbol': 'WTICOUSD', 'exchange': 'OANDA'},
    {'symbol': 'CORNUSD', 'exchange': 'OANDA'},
    {'symbol': 'USDCAD', 'exchange': 'OANDA'},
    {'symbol': 'XAUUSD', 'exchange': 'OANDA'},
]

# Step 3: Fetch data for multiple instruments
def get_data(symbol, exchange, interval=Interval.in_1_hour, n_bars=1000):
    try:
        df = tv.get_hist(symbol=symbol, exchange=exchange, interval=interval, n_bars=n_bars)
        if df is None or df.empty:
            print(f"No data for {symbol}:{exchange}")
            return None
        # Exclude current day
        df = df.iloc[:-1]
        return df
    except Exception as e:
        print(f"Error fetching data for {symbol}:{exchange}: {e}")
        return None

# Step 4: Identify swing points (default 3-bar pattern)
def find_swings(df, window=3):
    df['swing_low'] = df['low'].rolling(window, center=True).min() == df['low']
    df['swing_high'] = df['high'].rolling(window, center=True).max() == df['high']
    return df

# Step 5: Classify trends based on swing points
def classify_trend(df, min_swings=2):
    df['regime'] = 'consolidation'
    if df is None or df.empty:
        return df, 'consolidation'

    # Get swing points
    swing_lows = df[df['swing_low']][['low']].reset_index().rename(columns={'low': 'price', 'datetime': 'timestamp'})
    swing_lows['type'] = 'low'
    swing_highs = df[df['swing_high']][['high']].reset_index().rename(columns={'high': 'price', 'datetime': 'timestamp'})
    swing_highs['type'] = 'high'

    # Type 2 trends (independent swing highs/lows)
    type2_up_count = 0
    type2_down_count = 0
    last_low_price = float('-inf')
    last_high_price = float('inf')

    # Process Type 2 Uptrend (higher swing lows)
    for _, row in swing_lows.iterrows():
        curr_price = row['price']
        curr_ts = row['timestamp']
        if curr_price > last_low_price:
            type2_up_count += 1
            if type2_up_count >= min_swings:
                df.loc[curr_ts, 'regime'] = 'type2_uptrend'
        else:
            type2_up_count = 0
        last_low_price = curr_price

    # Process Type 2 Downtrend (lower swing highs)
    for _, row in swing_highs.iterrows():
        curr_price = row['price']
        curr_ts = row['timestamp']
        if curr_price < last_high_price:
            type2_down_count += 1
            if type2_down_count >= min_swings:
                df.loc[curr_ts, 'regime'] = 'type2_downtrend'
        else:
            type2_down_count = 0
        last_high_price = curr_price

    # Combine and sort swings for Type 1 trends and retracement checks
    swings = pd.concat([swing_lows, swing_highs]).sort_values('timestamp').reset_index(drop=True)
    type1_up_count = 0
    type1_down_count = 0
    swing_sequence = []
    last_high_price = float('-inf')
    last_low_price = float('inf')
    last_swing_type = None
    latest_swing_type = None
    last_two_swings = []  # Track last two swings for Type 1 retracement

    # Process Type 1 trends and track swing sequence
    for i in range(len(swings)):
        curr_ts, curr_type, curr_price = swings['timestamp'][i], swings['type'][i], swings['price'][i]
        swing_sequence.append((curr_ts, curr_type, curr_price))
        last_swing_type = latest_swing_type
        latest_swing_type = curr_type
        last_two_swings.append((curr_ts, curr_type, curr_price))
        if len(last_two_swings) > 2:
            last_two_swings.pop(0)

        # Check for Type 1 trend violations
        if curr_type == 'high':
            if type1_up_count >= min_swings and curr_price <= last_high_price:
                type1_up_count = 0  # Violation: lower or equal swing high
            if type1_down_count >= min_swings and curr_price >= last_high_price:
                type1_down_count = 0  # Violation: higher or equal swing high
            last_high_price = curr_price
        elif curr_type == 'low':
            if type1_up_count >= min_swings and curr_price <= last_low_price:
                type1_up_count = 0  # Violation: lower or equal swing low
            if type1_down_count >= min_swings and curr_price >= last_low_price:
                type1_down_count = 0  # Violation: higher or equal swing low
            last_low_price = curr_price

        # Check Type 1 trends (need at least 4 swings for a cycle)
        if len(swing_sequence) >= 4:
            # Type 1 Uptrend
            if (swing_sequence[-4][1] == 'high' and
                swing_sequence[-3][1] == 'low' and
                swing_sequence[-2][1] == 'high' and
                swing_sequence[-1][1] == 'low'):
                if (swing_sequence[-2][2] > swing_sequence[-4][2] and  # Higher high
                    swing_sequence[-1][2] > swing_sequence[-3][2]):   # Higher low
                    type1_up_count += 1
                    if type1_up_count >= min_swings:
                        df.loc[swing_sequence[-4][0]:swing_sequence[-1][0], 'regime'] = 'type1_uptrend'
                else:
                    type1_up_count = 0

            # Type 1 Downtrend
            if (swing_sequence[-4][1] == 'high' and
                swing_sequence[-3][1] == 'low' and
                swing_sequence[-2][1] == 'high' and
                swing_sequence[-1][1] == 'low'):
                if (swing_sequence[-2][2] < swing_sequence[-4][2] and  # Lower high
                    swing_sequence[-1][2] < swing_sequence[-3][2]):   # Lower low
                    type1_down_count += 1
                    if type1_down_count >= min_swings:
                        df.loc[swing_sequence[-4][0]:swing_sequence[-1][0], 'regime'] = 'type1_downtrend'
                else:
                    type1_down_count = 0

    # Determine current regime for summary
    latest_regime = 'consolidation'
    if type1_up_count >= min_swings:
        # Check for Type 1 uptrend retracement (latest swing is high, previous is low, both higher)
        if (len(last_two_swings) == 2 and
            last_two_swings[-1][1] == 'high' and
            last_two_swings[-2][1] == 'low' and
            last_two_swings[-1][2] > last_high_price and
            last_two_swings[-2][2] > last_low_price):
            latest_regime = 'type1_uptrend_retracement'
        else:
            latest_regime = 'type1_uptrend'
    elif type1_down_count >= min_swings:
        # Check for Type 1 downtrend retracement (latest swing is low, previous is high, both lower)
        if (len(last_two_swings) == 2 and
            last_two_swings[-1][1] == 'low' and
            last_two_swings[-2][1] == 'high' and
            last_two_swings[-1][2] < last_low_price and
            last_two_swings[-2][2] < last_high_price):
            latest_regime = 'type1_downtrend_retracement'
        else:
            latest_regime = 'type1_downtrend'
    elif type2_up_count >= min_swings and type2_down_count >= min_swings:
        latest_regime = 'type2_uptrend_and_downtrend'
    elif type2_up_count >= min_swings:
        # Check for Type 2 uptrend retracement (latest swing is high, previous is low)
        if latest_swing_type == 'high' and last_swing_type == 'low':
            latest_regime = 'type2_uptrend_retracement'
        else:
            latest_regime = 'type2_uptrend'
    elif type2_down_count >= min_swings:
        # Check for Type 2 downtrend retracement (latest swing is low, previous is high)
        if latest_swing_type == 'low' and last_swing_type == 'high':
            latest_regime = 'type2_downtrend_retracement'
        else:
            latest_regime = 'type2_downtrend'

    return df, latest_regime

# Step 6: Scan all instruments
def scan_instruments(instruments, interval=Interval.in_weekly, n_bars=100, swing_window=3):
    results = {}
    for instr in instruments:
        symbol = instr['symbol']
        exchange = instr['exchange']
        print(f"Processing {symbol}:{exchange}...")
        
        # Fetch data
        df = get_data(symbol, exchange, interval, n_bars)
        if df is None:
            continue

        # Identify swings
        df = find_swings(df, window=swing_window)

        # Classify trends
        df, latest_regime = classify_trend(df)

        # Filter to show only swing points
        swing_points = df[df['swing_low'] | df['swing_high']].copy()
        swing_points['swing_price'] = swing_points.apply(
            lambda row: row['high'] if row['swing_high'] else row['low'] if row['swing_low'] else None, axis=1
        )

        # Store results
        results[f"{symbol}:{exchange}"] = latest_regime

        # Print swing points only
        print(f"\n{symbol}:{exchange} Swing Points:")
        print(swing_points[['swing_low', 'swing_high', 'swing_price', 'regime']].reset_index().to_string(index=False))

    return results

# Step 7: Run the scan
swing_window = 3  # Default 3-bar pattern for swing detection
results = scan_instruments(instruments, swing_window=swing_window)

# Step 8: Print summary
print("\nMarket Regime Summary:")
for instr, regime in results.items():
    print(f"{instr}: {regime}")

# %%

# %%
