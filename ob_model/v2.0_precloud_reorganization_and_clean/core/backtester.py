#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# core/backtester.py - Backtest Function for Strategy Evidence
"""
Backtester for Strategy Evidence
Runs simple backtests on strategies to prove "edge or not" with net $/% after costs.
Why: Gives real P&L proof (e.g., "RSI reversion long 3-day: Yes, +$150 net avg")—answers "does it make money?"
Use: Input df, strategy params (style, long_short, hold_days), output metrics dict (expectancy, win %, yearly %).
"""

import pandas as pd
import numpy as np
import talib  # For indicators like RSI, ADX (your TA-Lib 0.11.0)
from utils.logger import get_logger, log_execution_time, log_errors
from utils.debug_utils import check_data_sanity

logger = get_logger('backtester')

class Backtester:
    def __init__(self, df: pd.DataFrame, instrument: str = 'NQ', contracts: int = 1):
        """
        Setup backtester with data, instrument costs, fixed contracts.
        - df: Daily with 'close', 'high', 'low', 'vol' (add 'rsi', 'adx' if needed).
        - instrument: 'NQ' (tick_value=5, mult=20), 'ES' ($12.5, $50), etc.
        - contracts: Fixed 1 (scale $ results; % same).
        """
        self.df = check_data_sanity(df, logger, 'backtester')
        self.df['date'] = pd.to_datetime(self.df.index)
        self.df = self.df.dropna(subset=['close'])  # Ensure no NaN closes
        self.instrument = instrument
        self.contracts = contracts
        self.tick_value, self.mult = self.get_instrument_specs()
        self.costs = self.calculate_costs()  # Round-trip $ per contract
        
        # Add indicators if not in df
        if 'rsi' not in self.df:
            self.df['rsi'] = talib.RSI(self.df['close'], timeperiod=2)
        if 'adx' not in self.df:
            self.df['adx'] = talib.ADX(self.df['high'], self.df['low'], self.df['close'], timeperiod=14)
        if 'bb_upper' not in self.df or 'bb_lower' not in self.df:
            self.df['bb_upper'], self.df['bb_mid'], self.df['bb_lower'] = talib.BBANDS(self.df['close'], timeperiod=20)
        self.df['bb_width'] = (self.df['bb_upper'] - self.df['bb_lower']) / self.df['bb_mid']  # For chop
        # Drop rows with NaN in indicators
        self.df = self.df.dropna(subset=['rsi', 'adx', 'bb_width'])  # After indicators

    def get_instrument_specs(self) -> tuple:
        """Auto specs for futures (tick_value $, point mult)"""
        specs = {
            'NQ': (5, 20),  # 0.25 pt tick = $5, $20/pt
            'ES': (12.5, 50),
            'GC': (10, 100)
        }
        return specs.get(self.instrument, (5, 20))  # Default NQ

    def calculate_costs(self) -> float:
        """Real round-trip costs $ per contract (slippage 1.5 ticks/side, spread 1 tick, commission $3)"""
        slippage_ticks = 1.5 * 2  # Entry + exit
        spread_ticks = 1
        commission = 3
        cost_ticks = slippage_ticks + spread_ticks
        cost_dollar = cost_ticks * self.tick_value + commission
        return cost_dollar

    @log_execution_time
    @log_errors()
    def run(self, style: str, strategy_name: str, long_short: str, hold_days: int) -> dict:
        """
        Run backtest for strategy, long/short, hold_days.
        - style: 'behavioral' etc. to select proxy.
        - strategy_name: 'rsi_reversion' etc.
        - long_short: 'long' or 'short'.
        - hold_days: Fixed hold (e.g., 3).
        - Output: Metrics dict (net_expectancy, win_pct, avg_net_profit_pct, trades_count, yearly_profit_pct dict, etc.).
        """
        trades = []
        i = 0
        while i < len(self.df) - hold_days:
            row = self.df.iloc[i]
            
            # Entry condition based on style/strategy (long/short flip for short)
            entry = False
            if style == 'temporal':
                if strategy_name == 'monday_buy' and row.name.weekday() == 0:  # Monday
                    entry = True if long_short == 'long' else False
            elif style == 'directional':
                if strategy_name == 'ma_above' and row['close'] > row['close'].rolling(50).mean():
                    entry = True if long_short == 'long' else False
            elif style == 'behavioral':
                if strategy_name == 'rsi_reversion' and row['rsi'] < 30:
                    entry = True if long_short == 'long' else (row['rsi'] > 70 if long_short == 'short' else False)
            elif style == 'conditional':
                if strategy_name == 'low_vol_reversion' and row['vol'] < self.df['vol'].mean() and row['rsi'] < 30:
                    entry = True if long_short == 'long' else (row['rsi'] > 70 if long_short == 'short' else False)
            # Add more for other strategies/styles...
            
            if entry:
                entry_price = row['close']
                exit_row = self.df.iloc[i + hold_days]
                exit_price = exit_row['close']
                
                # Gross points (long: exit - entry; short: entry - exit)
                gross_points = (exit_price - entry_price) if long_short == 'long' else (entry_price - exit_price)
                
                # Gross/Net $
                gross_dollar = gross_points * self.mult * self.contracts
                net_dollar = gross_dollar - self.costs
                
                # % Return (net)
                entry_value = entry_price * self.mult * self.contracts
                net_pct = (net_dollar / entry_value) * 100
                
                trades.append({
                    'entry_date': row.name,
                    'exit_date': exit_row.name,
                    'net_dollar': net_dollar,
                    'net_pct': net_pct,
                    'year': row.name.year
                })
                
                i += hold_days  # No overlap
            else:
                i += 1
        
        if not trades:
            return {'edge': 'No', 'reason': 'No Trades', 'trades_count': 0}
        
        trades_df = pd.DataFrame(trades)
        win_pct = (trades_df['net_dollar'] > 0).mean() * 100
        avg_net_dollar = trades_df['net_dollar'].mean()
        avg_net_pct = trades_df['net_pct'].mean()
        expectancy = avg_net_dollar  # Simplified (full: win%*avg_win - loss%*avg_loss)
        trades_count = len(trades_df)
        yearly_pct = trades_df.groupby('year')['net_pct'].sum().to_dict()
        positive_years_pct = (trades_df.groupby('year')['net_dollar'].sum() > 0).mean() * 100
        sharpe = (trades_df['net_pct'].mean() / trades_df['net_pct'].std()) * np.sqrt(252 / hold_days) if trades_count > 1 else 0
        
        edge = 'Yes' if expectancy > 0 and win_pct > 50 and positive_years_pct > 70 and trades_count >= 1000 and sharpe > 0.5 else 'No'
        
        return {
            'edge': edge,
            'win_pct': win_pct,
            'avg_net_dollar': avg_net_dollar,
            'avg_net_pct': avg_net_pct,
            'expectancy': expectancy,
            'trades_count': trades_count,
            'sharpe': sharpe,
            'yearly_net_pct': yearly_pct,
            'positive_years_pct': positive_years_pct
        }

