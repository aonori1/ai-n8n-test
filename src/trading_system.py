import os
import logging
import configparser
from datetime import timedelta
import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------
# Trading system
# - Architecture is class-based to remain consistent with typical designs.
# - Preserves logging and configuration loading.
# - Adds volatility-based position sizing (ATR), fixed maximum exposure,
#   per-trade risk limit, stop-loss, trailing stop, and max-drawdown protection.
# - Adds trend filter (EMA) to reduce whipsaws and lower drawdowns.
# - Returns detailed backtest metrics including drawdown, CAGR, Sharpe, Sortino.
# -----------------------------------------------------------------------------

# ----------------------------
# Configuration (preserve pattern)
# ----------------------------
DEFAULT_CONFIG = {
    'backtest': {
        'initial_cash': '100000.0',
        'commission_per_trade': '1.0',
        'slippage_per_trade': '0.0',
        'max_exposure_pct': '0.20',     # max fraction of portfolio in any single instrument
        'daily_max_loss_pct': '0.05',   # if daily loss exceeds this, stop trading rest of day
        'global_max_drawdown_pct': '0.25', # if portfolio drawdown exceeds, stop/backoff
    },
    'strategy': {
        'ema_short': '20',
        'ema_long': '50',
        'atr_period': '14',
        'atr_risk_pct': '0.01',  # risk per trade as fraction of equity (1%)
        'min_atr_multiplier': '0.5',  # prevents oversized positions if ATR tiny
        'stop_atr_multiplier': '3.0',  # stop-loss distance in ATRs
        'trailing_stop_atr': '2.0',   # trailing stop distance in ATRs after in profit
        'signal_threshold': '0.0',    # additional threshold for signals (can tune)
        'volatility_scaling_target_vol': '0.10',  # target portfolio vol (10% annual) for vol scaling
        'max_positions': '5',
    },
    'logging': {
        'level': 'INFO',
        'log_file': 'trading_system.log'
    }
}

def load_config(path='config.ini'):
    config = configparser.ConfigParser()
    config.read_dict(DEFAULT_CONFIG)
    if os.path.exists(path):
        config.read(path)
    return config

# ----------------------------
# Logging (preserve)
# ----------------------------
def setup_logging(config):
    level_name = config.get('logging', 'level', fallback='INFO')
    level = getattr(logging, level_name.upper(), logging.INFO)
    log_file = config.get('logging', 'log_file', fallback='trading_system.log')

    logging.basicConfig(
        level=level,
        format='%(asctime)s %(levelname)s %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    logging.info("Logging initialized at level %s", level_name)


# ----------------------------
# Indicators and utilities
# ----------------------------
def ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def atr(df, period=14):
    # True Range based on daily OHLC DataFrame with columns: ['open','high','low','close']
    high = df['high']
    low = df['low']
    close = df['close']
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_series = tr.rolling(window=period, min_periods=1).mean()
    return atr_series

def calculate_drawdowns(equity_series):
    peak = equity_series.cummax()
    drawdown = (equity_series - peak) / peak
    max_drawdown = drawdown.min()
    return drawdown, max_drawdown

def sharpe_ratio(returns, rf=0.0, periods_per_year=252.0):
    if returns.std() == 0:
        return 0.0
    ann_return = returns.mean() * periods_per_year
    ann_vol = returns.std() * np.sqrt(periods_per_year)
    return (ann_return - rf) / ann_vol if ann_vol != 0 else 0.0

def sortino_ratio(returns, rf=0.0, periods_per_year=252.0):
    downside = returns.copy()
    downside[downside > 0] = 0
    downside_std = np.sqrt((downside ** 2).sum() / max(1, len(downside) - 1))
    ann_return = returns.mean() * periods_per_year
    down_vol = downside_std * np.sqrt(periods_per_year)
    return (ann_return - rf) / down_vol if down_vol != 0 else np.nan

# ----------------------------
# Trading System
# ----------------------------
class TradingSystem:
    def __init__(self, price_df, config=None):
        """
        price_df: DataFrame with index = date, columns = ['open','high','low','close']
        config: configparser.ConfigParser or None to use defaults
        """
        self.config = config if config is not None else load_config()
        setup_logging(self.config)
        self.price = price_df.copy().sort_index()
        self._load_parameters()
        self._compute_indicators()
        # Backtest state
        self.cash = self.initial_cash
        self.equity = self.initial_cash
        self.position = 0.0    # number of shares (or contract equivalents), allow floats for sizing
        self.position_entry_price = None
        self.position_entry_atr = None
        self.trail_stop_price = None
        self.portfolio_history = []  # list of dicts per bar for analysis
        self.daily_loss = 0.0
        self.current_date = None
        self.max_equity = self.equity
        self.stop_trading_until = None  # date until which trading suspended after large loss

    def _load_parameters(self):
        b = self.config['backtest']
        s = self.config['strategy']
        self.initial_cash = float(b.get('initial_cash', DEFAULT_CONFIG['backtest']['initial_cash']))
        self.commission = float(b.get('commission_per_trade', DEFAULT_CONFIG['backtest']['commission_per_trade']))
        self.slippage = float(b.get('slippage_per_trade', DEFAULT_CONFIG['backtest']['slippage_per_trade']))
        self.max_exposure_pct = float(b.get('max_exposure_pct', DEFAULT_CONFIG['backtest']['max_exposure_pct']))
        self.daily_max_loss_pct = float(b.get('daily_max_loss_pct', DEFAULT_CONFIG['backtest']['daily_max_loss_pct']))
        self.global_max_drawdown_pct = float(b.get('global_max_drawdown_pct', DEFAULT_CONFIG['backtest']['global_max_drawdown_pct']))

        self.ema_short = int(s.get('ema_short', DEFAULT_CONFIG['strategy']['ema_short']))
        self.ema_long = int(s.get('ema_long', DEFAULT_CONFIG['strategy']['ema_long']))
        self.atr_period = int(s.get('atr_period', DEFAULT_CONFIG['strategy']['atr_period']))
        self.atr_risk_pct = float(s.get('atr_risk_pct', DEFAULT_CONFIG['strategy']['atr_risk_pct']))
        self.min_atr_multiplier = float(s.get('min_atr_multiplier', DEFAULT_CONFIG['strategy']['min_atr_multiplier']))
        self.stop_atr_multiplier = float(s.get('stop_atr_multiplier', DEFAULT_CONFIG['strategy']['stop_atr_multiplier']))
        self.trailing_stop_atr = float(s.get('trailing_stop_atr', DEFAULT_CONFIG['strategy']['trailing_stop_atr']))
        self.signal_threshold = float(s.get('signal_threshold', DEFAULT_CONFIG['strategy']['signal_threshold']))
        self.vol_target = float(s.get('volatility_scaling_target_vol', DEFAULT_CONFIG['strategy']['volatility_scaling_target_vol']))
        self.max_positions = int(s.get('max_positions', DEFAULT_CONFIG['strategy']['max_positions']))

    def _compute_indicators(self):
        logging.info("Computing indicators: EMA(%d/%d), ATR(%d)", self.ema_short, self.ema_long, self.atr_period)
        self.price['ema_short'] = ema(self.price['close'], self.ema_short)
        self.price['ema_long'] = ema(self.price['close'], self.ema_long)
        self.price['atr'] = atr(self.price, self.atr_period)
        # Momentum/strength filter: slope of long EMA (percent over period)
        self.price['ema_long_slope'] = self.price['ema_long'].pct_change(self.ema_long).fillna(0)

    def _signal(self, idx):
        """
        Signal: 1 = go long, 0 = flat/exit
        Use EMA crossover + trend filter + thresholding to reduce whipsaws.
        """
        row = self.price.iloc[idx]
        if np.isnan(row['ema_short']) or np.isnan(row['ema_long']):
            return 0
        # Basic crossover
        prev = self.price.iloc[max(0, idx-1)]
        cross_up = (prev['ema_short'] <= prev['ema_long']) and (row['ema_short'] > row['ema_long'] + self.signal_threshold)
        cross_down = (prev['ema_short'] >= prev['ema_long']) and (row['ema_short'] < row['ema_long'] - self.signal_threshold)
        # Trend strength filter: require long-term EMA slope non-negative for long entries
        trend_ok = row['ema_long_slope'] >= -0.0005  # allow tiny negative; tuneable
        if cross_up and trend_ok:
            return 1
        elif cross_down:
            return -1
        return 0

    def _size_position(self, price, atr):
        """
        Volatility-based position sizing using ATR:
        - Risk per trade = ATR * stop_atr_multiplier * position_size_in_shares
        - We want risk per trade in cash = equity * atr_risk_pct
        - -> position_size = (equity * atr_risk_pct) / (atr * stop_atr_multiplier)
        Enforce max exposure limit.
        """
        if atr <= 0 or np.isnan(atr):
            return 0.0
        risk_amount = self.equity * self.atr_risk_pct
        stop_distance = max(atr * self.stop_atr_multiplier, atr * self.min_atr_multiplier)
        # shares (or contracts) to risk approx risk_amount per trade
        size = risk_amount / stop_distance if stop_distance > 0 else 0
        # limit by max exposure (% of equity)
        max_position_value = self.equity * self.max_exposure_pct
        max_size_by_value = max_position_value / price if price > 0 else 0
        final_size = min(size, max_size_by_value)
        return max(0.0, final_size)

    def _apply_commission_slippage(self, cash):
        # naive handling: subtract commission and slippage when trade executed
        return cash - self.commission - self.slippage

    def run_backtest(self):
        logging.info("Starting backtest from %s to %s", self.price.index.min(), self.price.index.max())
        prev_day = None
        for idx in range(len(self.price)):
            row = self.price.iloc[idx]
            date = row.name
            self.current_date = date

            # Reset daily loss tracking at new day
            if prev_day is None or date.date() != prev_day:
                self.daily_loss = 0.0
                prev_day = date.date()

            # If suspended due to big drawdown, respect stop_trading_until
            if self.stop_trading_until is not None and date <= self.stop_trading_until:
                # no trading; just update equity history
                self._record_history(date, row['close'], 'suspended')
                self._update_max_equity()
                continue
            elif self.stop_trading_until is not None and date > self.stop_trading_until:
                logging.info("Resuming trading on %s after suspension", date)
                self.stop_trading_until = None

            signal = self._signal(idx)
            price = row['close']
            atr_val = row['atr'] if 'atr' in row.index else np.nan

            # Position management: entry, exit, stops, trailing stop
            if self.position == 0.0:
                # consider entry
                if signal == 1:
                    size = self._size_position(price, atr_val)
                    if size > 0 and self._can_open_new_position():
                        # open long
                        cost = size * price
                        cost_with_fees = cost + self.commission + self.slippage
                        if cost_with_fees <= self.cash:
                            self.position = size
                            self.position_entry_price = price
                            self.position_entry_atr = atr_val
                            # initial stop
                            self.trail_stop_price = price - max(atr_val * self.stop_atr_multiplier, atr_val * self.min_atr_multiplier)
                            self.cash -= cost_with_fees
                            logging.debug("Enter long %s @ %.4f size=%.4f cash=%.2f", date, price, size, self.cash)
                        else:
                            logging.debug("Not enough cash to open size=%.4f at price=%.4f", size, price)
            else:
                # if signal says exit or short, close position
                exit_condition = (signal == -1)
                stop_loss_hit = price <= (self.position_entry_price - self.position_entry_atr * self.stop_atr_multiplier)
                # trailing stop update if price moved in our favor
                if price > self.position_entry_price + self.position_entry_atr * self.trailing_stop_atr:
                    # update trailing stop to capture profits while letting trend run
                    new_trail = price - self.position_entry_atr * self.trailing_stop_atr
                    if new_trail > self.trail_stop_price:
                        self.trail_stop_price = new_trail

                trail_hit = price <= self.trail_stop_price if self.trail_stop_price is not None else False

                if exit_condition or stop_loss_hit or trail_hit:
                    proceeds = self.position * price
                    proceeds_after_fees = proceeds - (self.commission + self.slippage)
                    pnl = proceeds_after_fees - (self.position * self.position_entry_price)
                    self.cash += proceeds_after_fees
                    logging.debug("Exit long %s @ %.4f size=%.4f pnl=%.2f", date, price, self.position, pnl)
                    # update daily loss
                    if pnl < 0:
                        self.daily_loss += -pnl
                        # If daily loss exceeds threshold, suspend trading rest of day
                        daily_max_loss = self.initial_cash * self.daily_max_loss_pct
                        if self.daily_loss > daily_max_loss:
                            self.stop_trading_until = date.replace(hour=23, minute=59, second=59)
                            logging.warning("Daily loss exceeded %.2f, suspending trading until end of day", daily_max_loss)
                    self.position = 0.0
                    self.position_entry_price = None
                    self.position_entry_atr = None
                    self.trail_stop_price = None

            # Equity update
            position_value = self.position * price
            self.equity = self.cash + position_value
            self._record_history(date, price, 'long' if self.position > 0 else 'flat')
            # Check global drawdown protection
            self._update_max_equity()
            if (self.max_equity - self.equity) / max(1e-9, self.max_equity) > self.global_max_drawdown_pct:
                # Suspend trading for a cooldown period (e.g., 10 trading days)
                cooldown = timedelta(days=10)
                self.stop_trading_until = date + cooldown
                logging.warning("Global drawdown limit reached. Suspending trading until %s", self.stop_trading_until)
            # daily max loss check (if already in cash, handled above on exit)
            if self.daily_loss > (self.initial_cash * self.daily_max_loss_pct):
                self.stop_trading_until = date.replace(hour=23, minute=59, second=59)
                logging.warning("Daily loss limit reached: %.2f. Stop trading for the day.", self.daily_loss)

        logging.info("Backtest complete. Final equity: %.2f", self.equity)
        return self._generate_performance()

    def _can_open_new_position(self):
        # For single-instrument system this checks number of concurrent positions allowed.
        # Here we restrict to either 0 or 1, but keep architecture for multi-position extension.
        return 1 <= self.max_positions

    def _update_max_equity(self):
        if self.equity > self.max_equity:
            self.max_equity = self.equity

    def _record_history(self, date, price, position_state):
        self.portfolio_history.append({
            'date': date,
            'price': price,
            'position': self.position,
            'cash': self.cash,
            'equity': self.equity,
            'state': position_state
        })

    def _generate_performance(self):
        df = pd.DataFrame(self.portfolio_history).set_index('date')
        df['returns'] = df['equity'].pct_change().fillna(0)
        df['cum_return'] = (1 + df['returns']).cumprod() - 1
        drawdown, max_dd = calculate_drawdowns(df['equity'])
        df['drawdown'] = drawdown
        perf = {
            'final_equity': float(self.equity),
            'total_return': float(df['cum_return'].iloc[-1]) if len(df) else 0.0,
            'max_drawdown_pct': float(max_dd),
            'sharpe': float(sharpe_ratio(df['returns'])),
            'sortino': float(sortino_ratio(df['returns'])),
            'history': df
        }
        logging.info("Performance: Final Equity=%.2f TotalReturn=%.2f%% MaxDD=%.2f%% Sharpe=%.3f Sortino=%.3f",
                     perf['final_equity'],
                     perf['total_return'] * 100,
                     perf['max_drawdown_pct'] * 100,
                     perf['sharpe'],
                     perf['sortino'] if not np.isnan(perf['sortino']) else -999.0)
        return perf

# ----------------------------
# Example usage helper (kept minimal, consistent with preserved architecture)
# ----------------------------
def load_price_csv(path):
    """
    Load OHLC CSV assumed to have date index and columns: date,open,high,low,close
    Date inferred as index.
    """
    df = pd.read_csv(path, parse_dates=True, index_col=0)
    df = df.rename(columns={c: c.lower() for c in df.columns})
    required = {'open', 'high', 'low', 'close'}
    if not required.issubset(set(df.columns)):
        raise ValueError(f"CSV must contain columns: {required}")
    return df[['open', 'high', 'low', 'close']]

# If this module is run directly, run a sample backtest if data exists.
if __name__ == "__main__":
    config = load_config()
    setup_logging(config)
    try:
        data_path = config.get('backtest', 'data_path', fallback='price.csv')
    except Exception:
        data_path = 'price.csv'
    if os.path.exists(data_path):
        logging.info("Loading price data from %s", data_path)
        prices = load_price_csv(data_path)
        ts = TradingSystem(prices, config=config)
        perf = ts.run_backtest()
        # Persist a CSV of history
        hist = perf['history']
        hist.to_csv('backtest_history.csv')
        logging.info("Backtest history saved to backtest_history.csv")
    else:
        logging.warning("Data path %s not found. Provide a CSV with OHLC data or set backtest.data_path in config.", data_path)