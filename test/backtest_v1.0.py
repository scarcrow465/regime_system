import pandas as pd
import numpy as np
from typing import List, Optional, Tuple
from datetime import datetime
import logging
from dataclasses import dataclass
import backtrader as bt
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Instrument configuration
INSTRUMENT_CONFIGS = {
    'Micro_NQ': {
        'dollars_per_tick': 0.50,
        'commission_per_contract': 0.74,
        'spread_range': (0.25, 0.50),
        'ticks_per_point': 4.0,
    }
}

def parse_symbol(symbol_str):
    if not isinstance(symbol_str, str) or pd.isna(symbol_str):
        return None
    if len(symbol_str) < 3:
        return symbol_str
    year = symbol_str[-2:]
    if year.isdigit():
        month = symbol_str[-3]
        if month in {'F', 'G', 'H', 'J', 'K', 'M', 'N', 'Q', 'U', 'V', 'X', 'Z'}:
            return symbol_str[:-3]
    return symbol_str

def load_csv_data(csv_paths: List[str], symbols: List[str], start_date: Optional[pd.Timestamp] = None, end_date: Optional[pd.Timestamp] = None) -> pd.DataFrame:
    if isinstance(csv_paths, str):
        csv_paths = [csv_paths]
    all_dfs = []
    for csv_path in csv_paths:
        try:
            chunks = pd.read_csv(
                csv_path,
                parse_dates=['Date'],
                index_col='Date',
                chunksize=100000,
                dtype={'Symbol': str},
                low_memory=False
            )
            for chunk in chunks:
                if chunk.empty or chunk.index.hasnans:
                    continue
                chunk = chunk.copy()
                if chunk.index.tz is None:
                    chunk.index = chunk.index.tz_localize('America/New_York', ambiguous='raise', nonexistent='shift_forward')
                else:
                    chunk.index = chunk.index.tz_convert('America/New_York')
                symbol_cols = [col for col in chunk.columns if col.startswith('Symbol')]
                if not symbol_cols:
                    continue
                for sym_col in symbol_cols:
                    suffix = sym_col.replace('Symbol', '')
                    expected_cols = {
                        'symbol': sym_col,
                        'open': f'Open{suffix}',
                        'high': f'High{suffix}',
                        'low': f'Low{suffix}',
                        'close': f'Close{suffix}',
                        'volume': f'Volume{suffix}',
                        'openinterest': f'OpenInterest{suffix}'
                    }
                    if not all(col in chunk.columns for col in expected_cols.values()):
                        continue
                    sub_df = chunk[list(expected_cols.values())].copy()
                    sub_df.columns = ['symbol', 'open', 'high', 'low', 'close', 'volume', 'openinterest']
                    for col in ['open', 'high', 'low', 'close', 'volume', 'openinterest']:
                        sub_df[col] = pd.to_numeric(sub_df[col].astype(str).str.replace(',', '', regex=False), errors='coerce')
                    sub_df = sub_df.dropna(subset=['open', 'high', 'low', 'close'])
                    if sub_df.empty:
                        continue
                    symbol_value = sub_df['symbol'].dropna().iloc[0]
                    base_symbol = parse_symbol(symbol_value)
                    if base_symbol and base_symbol in symbols:
                        sub_df['symbol'] = symbol_value
                        sub_df['BaseSymbol'] = base_symbol
                        all_dfs.append(sub_df)
                        logger.info(f"Processed {sym_col} (base: {base_symbol}) with {len(sub_df)} rows")
        except Exception as e:
            logger.error(f"Error reading CSV {csv_path}: {e}")
            continue
    if not all_dfs:
        logger.error("No valid data loaded from any CSV files.")
        return pd.DataFrame()
    combined_df = pd.concat(all_dfs).sort_index()
    if start_date is not None:
        combined_df = combined_df[combined_df.index >= start_date]
    if end_date is not None:
        combined_df = combined_df[combined_df.index <= end_date]
    combined_df = combined_df.reset_index().groupby(['Date', 'BaseSymbol']).first().reset_index().set_index('Date')
    logger.info(f"Loaded {len(combined_df)} total rows for symbols: {combined_df['BaseSymbol'].unique()}")
    return combined_df

@dataclass
class BacktestConfig:
    def __init__(self, 
                 initial_capital: float = 50000.0,
                 risk_per_trade: float = 500.0,
                 reward_per_trade: float = 250.0,
                 allowed_hours: List[int] = None,
                 min_ob_size_ticks: float = 20.0,
                 max_ob_size_ticks: float = 500.0,
                 allowed_ob_types: List[str] = None,
                 use_time_stop: bool = True,
                 time_stop_candles: Optional[int] = 1,
                 instrument: str = "Micro_NQ",
                 dollars_per_tick: float = 0.50,
                 commission_per_contract: float = 0.74,
                 spread_range: Tuple[float, float] = (0.25, 0.50),
                 ticks_per_point: float = 4.0,
                 use_commissions: bool = True,
                 use_spreads: bool = True,
                 force_eod_exit: bool = True):
        self.initial_capital = initial_capital
        self.risk_per_trade = risk_per_trade
        self.reward_per_trade = reward_per_trade
        self.allowed_hours = allowed_hours or [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
        self.min_ob_size_ticks = min_ob_size_ticks
        self.max_ob_size_ticks = max_ob_size_ticks
        self.allowed_ob_types = allowed_ob_types or ['all']
        self.use_time_stop = use_time_stop
        self.time_stop_candles = time_stop_candles
        self.instrument = instrument
        self.dollars_per_tick = dollars_per_tick
        self.commission_per_contract = commission_per_contract
        self.spread_range = spread_range
        self.ticks_per_point = ticks_per_point
        self.use_commissions = use_commissions
        self.use_spreads = use_spreads
        self.force_eod_exit = force_eod_exit
        self.__post_init__()

    def __post_init__(self):
        if self.instrument not in INSTRUMENT_CONFIGS:
            raise ValueError(f"Instrument {self.instrument} not found in INSTRUMENT_CONFIGS")

class TimeStopStrategy(bt.Strategy):
    params = (
        ('ob_data', None),
        ('config', None),
    )

    def __init__(self):
        self.ob_dict = {}
        self.active_trades = []
        self.trades_log = []
        self.all_trades = []  # Store all completed trades
        if self.p.ob_data is not None:
            self.p.ob_data['entry_time'] = pd.to_datetime(self.p.ob_data['entry_time'], errors='coerce', utc=True).dt.tz_convert('America/New_York')
            filtered = self.p.ob_data[
                (self.p.ob_data['touched'] == True) &
                (self.p.ob_data['ob_size_ticks'] >= self.p.config.min_ob_size_ticks) &
                (self.p.ob_data['ob_size_ticks'] <= self.p.config.max_ob_size_ticks) &
                (self.p.ob_data['formation_time'].dt.hour.isin(self.p.config.allowed_hours)) &
                (self.p.ob_data['ignore_reason'].isna() | (self.p.ob_data['ignore_reason'] == ''))
            ]
            if 'all' not in self.p.config.allowed_ob_types:
                filtered = filtered[filtered['type'].isin(self.p.config.allowed_ob_types)]
            self.ob_dict = filtered.groupby('entry_time').apply(lambda x: x.to_dict('records')).to_dict()
        self.config = self.p.config
        self.current_candle_count = {}

    def next(self):
        dt = self.data.datetime.datetime(0)
        # Process new trade entries
        if dt in self.ob_dict:
            for ob in self.ob_dict[dt]:
                trade = self.simulate_entry(ob, dt)
                if trade:
                    self.active_trades.append(trade)
                    self.current_candle_count[trade['ob_id']] = 0
        # Update active trades
        for trade in self.active_trades[:]:
            self.current_candle_count[trade['ob_id']] += 1
            exit_reason, exit_price = self.check_exit(trade, dt)
            if exit_reason:
                pnl = self.calculate_pnl(trade, exit_price)
                self.trades_log.append({
                    'ob_id': trade['ob_id'],
                    'entry_time': trade['entry_time'],
                    'exit_time': dt,
                    'pnl': pnl,
                    'exit_reason': exit_reason,
                })
                self.all_trades.append({
                    'id': len(self.all_trades) + 1,
                    'direction': 'Buy' if trade['is_bullish'] else 'Sell',
                    'size': trade['contracts'],
                    'entry_time': trade['entry_time'].isoformat(),
                    'trade_price': trade['entry_price'],
                    'exit_info': f"{exit_price} {dt.isoformat()}",
                    'total_profit': pnl
                })
                self.active_trades.remove(trade)
                del self.current_candle_count[trade['ob_id']]

    def simulate_entry(self, ob, dt):
        entry_price = ob['entry_price']
        ob_size_ticks = ob['ob_size_ticks'] * self.config.ticks_per_point
        for contracts in range(1, 21):
            risk_ticks = self.config.risk_per_trade / (self.config.dollars_per_tick * contracts)
            if risk_ticks >= ob_size_ticks:
                break
        else:
            return None
        reward_risk_ratio = self.config.reward_per_trade / self.config.risk_per_trade
        target_ticks = risk_ticks * reward_risk_ratio
        spread = np.random.uniform(self.config.spread_range[0], self.config.spread_range[1]) if self.config.use_spreads else 0.0
        adjusted_entry_price = entry_price + spread if ob['type'].startswith('bullish') else entry_price - spread
        risk_points = risk_ticks / self.config.ticks_per_point
        target_points = target_ticks / self.config.ticks_per_point
        is_bullish = ob['type'].startswith('bullish')
        stop_loss = adjusted_entry_price - risk_points if is_bullish else adjusted_entry_price + risk_points
        target = adjusted_entry_price + target_points if is_bullish else adjusted_entry_price - target_points
        return {
            'ob_id': ob['ob_id'],
            'entry_time': dt,
            'entry_price': adjusted_entry_price,
            'is_bullish': is_bullish,
            'stop_loss': stop_loss,
            'target': target,
            'contracts': contracts,
        }

    def check_exit(self, trade, dt):
        high = self.data.high[0]
        low = self.data.low[0]
        close = self.data.close[0]
        eod_time = dt.replace(hour=16, minute=0, second=0, microsecond=0)
        if trade['is_bullish']:
            if low <= trade['stop_loss']:
                return 'stop_loss', trade['stop_loss']
            elif high >= trade['target']:
                return 'target', trade['target']
        else:
            if high >= trade['stop_loss']:
                return 'stop_loss', trade['stop_loss']
            elif low <= trade['target']:
                return 'target', trade['target']
        if self.config.use_time_stop and self.current_candle_count[trade['ob_id']] >= self.config.time_stop_candles:
            return 'time_stop', close
        if self.config.force_eod_exit and dt >= eod_time:
            return 'eod_exit', close
        return None, None

    def calculate_pnl(self, trade, exit_price):
        pnl_ticks = (exit_price - trade['entry_price']) * self.config.ticks_per_point if trade['is_bullish'] else (trade['entry_price'] - exit_price) * self.config.ticks_per_point
        gross_pnl = pnl_ticks * trade['contracts'] * self.config.dollars_per_tick
        commission = self.config.commission_per_contract * trade['contracts'] if self.config.use_commissions else 0.0
        return gross_pnl - commission

def run_time_stop_backtest(config: Optional[BacktestConfig] = None):
    if config is None:
        config = BacktestConfig()
    ob_details_csv = r"C:\Users\rs\OneDrive\Desktop\TV DB\correlation_analysis\20250702_175410_NQ\csv\order_blocks_20250702_175410.csv"
    ohlc_paths = [
        r"C:\Users\rs\OneDrive\Desktop\Excel\Data\New Data\8 Master 5m Data - Updated - Nearest Unadjusted - 2022_01_01 - 2025_04_01 .csv",
        r"C:\Users\rs\OneDrive\Desktop\Excel\Data\New Data\8.1 Master 5m Data - Updated - Nearest Unadjusted - 2019_01_01 - 2021_12_31 .csv",
        r"C:\Users\rs\OneDrive\Desktop\Excel\Data\New Data\8.2 Master 5m Data - Updated - Nearest Unadjusted - 2016_01_01 - 2018_12_31 .csv",
        r"C:\Users\rs\OneDrive\Desktop\Excel\Data\New Data\8.3 Master 5m Data - Updated - Nearest Unadjusted - 2013_01_01 - 2015_12_31 .csv"
    ]
    symbols = ["NQ"]
    ohlc_data = load_csv_data(ohlc_paths, symbols)
    ohlc_data = ohlc_data[ohlc_data['BaseSymbol'] == 'NQ'][['open', 'high', 'low', 'close']]
    ob_data = pd.read_csv(ob_details_csv)
    ob_data['formation_time'] = pd.to_datetime(ob_data['formation_time'], errors='coerce', utc=True).dt.tz_convert('America/New_York')
    cerebro = bt.Cerebro()
    data = bt.feeds.PandasData(dataname=ohlc_data)
    cerebro.adddata(data)
    cerebro.addstrategy(TimeStopStrategy, ob_data=ob_data, config=config)
    cerebro.broker.setcash(config.initial_capital)
    # Add analyzers
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trade_analyzer')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
    # Run backtest
    results = cerebro.run()
    strategy = results[0]
    final_value = cerebro.broker.getvalue()
    logger.info(f"Backtest completed. Initial Capital: ${config.initial_capital:.2f}, Final Value: ${final_value:.2f}")
    # Prepare output data
    price_data = [{'time': pd.Timestamp(t).isoformat(), 'open': o, 'high': h, 'low': l, 'close': c}
                  for t, o, h, l, c in zip(ohlc_data.index, ohlc_data['open'], ohlc_data['high'], ohlc_data['low'], ohlc_data['close'])]
    signals = [{'timestamp': t['entry_time'].isoformat(), 'type': 'buy', 'label': f"LONG {t['pnl']:.1f}%"} if t['pnl'] >= 0 else {'timestamp': t['exit_time'].isoformat(), 'type': 'sell', 'label': f"EXIT {t['pnl']:.1f}%"}
               for t in strategy.trades_log]
    trades = strategy.all_trades
    trade_analysis = strategy.analyzers.trade_analyzer.get_analysis()
    drawdown = strategy.analyzers.drawdown.get_analysis()
    sharpe = strategy.analyzers.sharpe.get_analysis()
    metrics = {
        'totalReturn': ((final_value - config.initial_capital) / config.initial_capital * 100) if config.initial_capital > 0 else 0.0,
        'winRate': (trade_analysis.won.total / trade_analysis.total.total * 100) if trade_analysis.total.total > 0 else 0.0,
        'maxDrawdown': drawdown.max.drawdown if drawdown.max.drawdown else 0.0,
        'numberOfTrades': trade_analysis.total.total,
        'winLossRatio': trade_analysis.won.total / trade_analysis.lost.total if trade_analysis.lost.total > 0 and trade_analysis.won.total > 0 else (trade_analysis.won.total / 1 if trade_analysis.won.total > 0 else 0.0),
        'sharpeRatio': sharpe.sharperatio if sharpe.sharperatio else 0.0
    }
    equity_curve = [{'time': pd.Timestamp(t).isoformat(), 'value': v} for t, v in zip(ohlc_data.index, cerebro.broker.getvalue())]  # Simplified; enhance with actual equity history
    output = {
        'priceData': price_data,
        'signals': signals,
        'trades': trades,
        'metrics': metrics,
        'equityCurve': equity_curve
    }
    with open('backtest.json', 'w') as f:
        json.dump(output, f, indent=2)
    return cerebro

if __name__ == "__main__":
    config = BacktestConfig(
        risk_per_trade=500.0,
        reward_per_trade=250.0,
        allowed_hours=[4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
        min_ob_size_ticks=20.0,  # Reduced to allow more trades
        max_ob_size_ticks=500.0,
        use_time_stop=False,
        time_stop_candles=None,
        use_commissions=True,
        use_spreads=True,
        force_eod_exit=True,
    )
    cerebro = run_time_stop_backtest(config)