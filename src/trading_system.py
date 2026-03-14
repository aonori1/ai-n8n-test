import logging
import math
import json
from dataclasses import dataclass, field
from typing import Dict, Optional, List, Any
import numpy as np
import pandas as pd
from copy import deepcopy
import datetime as dt

# =========================
# Configuration (preserve / editable)
# =========================
DEFAULT_CONFIG = {
    "initial_capital": 100000.0,
    "risk_per_trade": 0.01,                # fraction of capital to risk per trade
    "max_drawdown_stop_pct": 0.20,         # stop trading if drawdown exceeds this fraction
    "drawdown_recovery_pct": 0.05,         # resume trading once drawdown recovers to this level
    "max_position_leverage": 1.0,          # max fraction of capital allocated to position (<=1 means no leverage)
    "atr_period": 14,
    "atr_multiplier_stop": 3.0,            # stop distance in ATRs
    "atr_multiplier_trail": 1.5,           # trailing stop distance in ATRs
    "ema_short": 20,
    "ema_long": 50,
    "adx_period": 14,
    "adx_threshold": 20,                   # require trend strength above this for trend trades
    "min_trade_holding_bars": 3,           # minimum bars to hold to avoid immediate whipsaws
    "max_trade_holding_bars": 200,         # time-based exit
    "position_size_floor": 0.0001,         # minimum position size in fraction of capital
    "logging": {
        "level": "INFO",
        "format": "%(asctime)s %(levelname)s %(message)s"
    }
}

# =========================
# Logging setup function (preserve)
# =========================
def setup_logging(config_logging: Dict[str, Any]):
    level = getattr(logging, config_logging.get("level", "INFO").upper(), logging.INFO)
    logging.basicConfig(level=level, format=config_logging.get("format"))
    return logging.getLogger("TradingSystem")


# =========================
# Indicators
# =========================
def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def true_range(df: pd.DataFrame) -> pd.Series:
    tr1 = df['high'] - df['low']
    tr2 = (df['high'] - df['close'].shift(1)).abs()
    tr3 = (df['low'] - df['close'].shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr


def atr(df: pd.DataFrame, period: int) -> pd.Series:
    tr = true_range(df)
    return tr.rolling(period, min_periods=1).mean()


def directional_indicators(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    # Simplified ADX calculation
    up_move = df['high'].diff()
    down_move = -df['low'].diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    tr = true_range(df)
    atr_series = tr.rolling(period, min_periods=1).mean()

    plus_di = 100 * (pd.Series(plus_dm).rolling(period, min_periods=1).sum() / atr_series.replace(0, np.nan))
    minus_di = 100 * (pd.Series(minus_dm).rolling(period, min_periods=1).sum() / atr_series.replace(0, np.nan))
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx = dx.rolling(period, min_periods=1).mean().fillna(0)
    out = pd.DataFrame({
        'plus_di': plus_di.fillna(0),
        'minus_di': minus_di.fillna(0),
        'adx': adx
    }, index=df.index)
    return out


# =========================
# Utilities / Metrics
# =========================
def compute_performance(equity_curve: pd.Series, risk_free_rate_annual=0.0) -> Dict[str, Any]:
    returns = equity_curve.pct_change().fillna(0)
    total_return = equity_curve.iloc[-1] / equity_curve.iloc[0] - 1.0
    days = (equity_curve.index[-1] - equity_curve.index[0]).days if isinstance(equity_curve.index[0], pd.Timestamp) else len(equity_curve)
    years = max(days / 365.25, 1/252)
    cagr = (equity_curve.iloc[-1] / equity_curve.iloc[0]) ** (1 / years) - 1
    ann_vol = returns.std() * np.sqrt(252)
    sharpe = (cagr - risk_free_rate_annual) / ann_vol if ann_vol != 0 else np.nan
    # Sortino
    neg_returns = returns[returns < 0]
    downside_vol = neg_returns.std() * np.sqrt(252)
    sortino = (cagr - risk_free_rate_annual) / downside_vol if downside_vol != 0 else np.nan
    peak = equity_curve.cummax()
    drawdown = (equity_curve - peak) / peak
    max_dd = drawdown.min()
    return {
        'total_return': total_return,
        'cagr': cagr,
        'ann_vol': ann_vol,
        'sharpe': sharpe,
        'sortino': sortino,
        'max_drawdown': max_dd,
        'equity_curve': equity_curve,
        'returns': returns
    }


# =========================
# Trading system core (preserve architecture)
# =========================
@dataclass
class Trade:
    entry_date: pd.Timestamp
    entry_price: float
    size: float                    # number of shares/contracts (positive for long, negative for short)
    direction: int                 # 1 long, -1 short
    stop: float
    trail_stop: Optional[float] = None
    exit_date: Optional[pd.Timestamp] = None
    exit_price: Optional[float] = None
    pnl: Optional[float] = None
    holding_bars: int = 0


@dataclass
class TradingSystem:
    config: Dict[str, Any] = field(default_factory=lambda: deepcopy(DEFAULT_CONFIG))
    logger: logging.Logger = field(init=False)

    def __post_init__(self):
        self.logger = setup_logging(self.config.get("logging", {}))
        self.logger.info("TradingSystem initialized with config: %s", json.dumps(self.config, default=str))
        # Internal runtime state
        self.equity = self.config["initial_capital"]
        self.max_equity = self.equity
        self.trades: List[Trade] = []
        self.current_trade: Optional[Trade] = None
        self.stopped_from_drawdown = False

    def reset(self):
        self.equity = self.config["initial_capital"]
        self.max_equity = self.equity
        self.trades = []
        self.current_trade = None
        self.stopped_from_drawdown = False
        self.logger.info("System reset")

    def compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df['ema_short'] = ema(df['close'], self.config['ema_short'])
        df['ema_long'] = ema(df['close'], self.config['ema_long'])
        df['atr'] = atr(df, self.config['atr_period'])
        di = directional_indicators(df, self.config['adx_period'])
        df = df.join(di)
        # Signal: ema crossover when ADX indicates a trend
        df['signal_raw'] = 0
        df.loc[df['ema_short'] > df['ema_long'], 'signal_raw'] = 1
        df.loc[df['ema_short'] < df['ema_long'], 'signal_raw'] = -1
        df['signal'] = 0
        # Only take signals when ADX shows trend strength
        df.loc[(df['signal_raw'] == 1) & (df['adx'] >= self.config['adx_threshold']), 'signal'] = 1
        df.loc[(df['signal_raw'] == -1) & (df['adx'] >= self.config['adx_threshold']), 'signal'] = -1
        # If ADX low, optionally no trade (reduce whipsaw)
        return df

    def _size_position(self, capital: float, price: float, atr: float, direction: int) -> float:
        """Size position using fixed fractional risk and ATR for stop distance."""
        risk_per_trade = self.config['risk_per_trade'] * capital
        stop_distance = max(atr * self.config['atr_multiplier_stop'], price * 0.001)  # tiny minimum
        if stop_distance <= 0:
            self.logger.warning("Stop distance computed as zero or negative. Defaulting to small positive.")
            stop_distance = price * 0.001
        # number of units to risk risk_per_trade given stop distance
        size = math.floor((risk_per_trade / stop_distance) / 1.0)  # assumes 1 contract/lot priced at 1 price move = 1 unit pnl
        # enforce leverage cap: position value = size * price
        max_position_value = self.config['max_position_leverage'] * capital
        max_size_by_value = math.floor(max_position_value / price) if price > 0 else 0
        final_size = int(max(0, min(size, max_size_by_value)))
        if final_size == 0:
            # fallback to minimal fractional position so small trades can still occur
            fallback_size = int(max(1, math.floor((self.config['position_size_floor'] * capital) / price)))
            final_size = fallback_size
        return final_size

    def _enter_trade(self, date: pd.Timestamp, price: float, direction: int, atr: float):
        size = self._size_position(self.equity, price, atr, direction)
        stop = price - direction * (atr * self.config['atr_multiplier_stop'])
        if direction == -1:
            # for short, stop is above entry
            stop = price + abs(direction) * (atr * self.config['atr_multiplier_stop'])
        trade = Trade(entry_date=date, entry_price=price, size=size, direction=direction, stop=stop, trail_stop=None)
        self.current_trade = trade
        self.logger.info("ENTER %s trade @ %s price=%.4f size=%d stop=%.4f equity=%.2f",
                         "LONG" if direction == 1 else "SHORT", date, price, size, stop, self.equity)

    def _exit_trade(self, date: pd.Timestamp, price: float, reason: str):
        t = self.current_trade
        if t is None:
            return
        t.exit_date = date
        t.exit_price = price
        # PnL calculation: direction * (exit - entry) * size
        t.pnl = t.direction * (t.exit_price - t.entry_price) * t.size
        self.equity += t.pnl
        self.max_equity = max(self.max_equity, self.equity)
        self.trades.append(t)
        self.logger.info("EXIT %s trade @ %s price=%.4f size=%d pnl=%.2f reason=%s equity=%.2f",
                         "LONG" if t.direction == 1 else "SHORT", date, price, t.size, t.pnl, reason, self.equity)
        self.current_trade = None

    def _update_trailing_and_check_stop(self, row: pd.Series) -> Optional[str]:
        """Update trailing stop and check if exit conditions met. Returns exit reason or None."""
        t = self.current_trade
        if t is None:
            return None
        price = row['close']
        atr_val = row['atr']
        # Update holding bars
        t.holding_bars += 1
        # Initialize trailing stop on first update
        trail_distance = atr_val * self.config['atr_multiplier_trail']
        if t.trail_stop is None:
            if t.direction == 1:
                t.trail_stop = t.entry_price - trail_distance
            else:
                t.trail_stop = t.entry_price + trail_distance
        else:
            # For long, trail_stop only moves up; for short, only moves down
            if t.direction == 1:
                new_trail = price - trail_distance
                if new_trail > t.trail_stop:
                    t.trail_stop = new_trail
            else:
                new_trail = price + trail_distance
                if new_trail < t.trail_stop:
                    t.trail_stop = new_trail

        # Hard stop check
        if t.direction == 1 and price <= t.stop:
            return "hard_stop"
        if t.direction == -1 and price >= t.stop:
            return "hard_stop"

        # Trailing stop check
        if t.direction == 1 and price <= t.trail_stop:
            return "trail_stop"
        if t.direction == -1 and price >= t.trail_stop:
            return "trail_stop"

        # Time-based exit
        if t.holding_bars >= self.config['max_trade_holding_bars']:
            return "time_exit"

        return None

    def _check_and_maybe_stop_trading_due_to_drawdown(self):
        # If drawdown too large, stop trading until recovery
        dd = (self.max_equity - self.equity) / max(1.0, self.max_equity)
        if not self.stopped_from_drawdown and dd >= self.config['max_drawdown_stop_pct']:
            self.stopped_from_drawdown = True
            self.logger.warning("Trading suspended due to drawdown %.2f%% >= threshold %.2f%%",
                                dd * 100, self.config['max_drawdown_stop_pct'] * 100)
        elif self.stopped_from_drawdown:
            # check recovery
            recovery = (self.equity - (self.max_equity * (1 - self.config['max_drawdown_stop_pct']))) / max(1.0, self.max_equity)
            if recovery >= self.config['drawdown_recovery_pct']:
                self.stopped_from_drawdown = False
                self.logger.info("Trading resumed after recovery: recovery metric %.4f", recovery)

    def run_backtest(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Main backtest loop. Dataframe must contain open/high/low/close and datetime index.
        Architecture preserved: indicators computed, signals generated, trades tracked, logging preserved.
        """
        df = df.copy()
        if not isinstance(df.index, pd.DatetimeIndex):
            try:
                df.index = pd.to_datetime(df.index)
            except Exception:
                # keep as-is if cannot parse
                pass
        df.sort_index(inplace=True)
        df = self.compute_indicators(df)
        equity_curve = []
        dates = []

        # iterate rows
        for idx, row in df.iterrows():
            price = row['close']
            date = idx

            # Update current trade trailing stops / check exits
            if self.current_trade is not None:
                exit_reason = self._update_trailing_and_check_stop(row)
                # Enforce minimum holding bars to avoid immediate flips
                if exit_reason is None and self.current_trade.holding_bars < self.config['min_trade_holding_bars']:
                    exit_reason = None  # don't exit even if signal flips immediately
                if exit_reason is not None:
                    # execute exit at current price
                    self._exit_trade(date, price, exit_reason)

            # Check and possibly suspend/resume trading due to drawdown
            self._check_and_maybe_stop_trading_due_to_drawdown()

            # Generate signals and enter if none open
            signal = int(row.get('signal', 0))
            if self.current_trade is None and not self.stopped_from_drawdown:
                # Only enter if signal exists and not in drawdown suspension
                if signal != 0:
                    # additional filter: don't flip on weak ADX or price too close to previous exit to avoid noise
                    # enter trade
                    self._enter_trade(date, price, signal, row['atr'])
            # If a trade is open and a new opposite signal appears after min holding, consider reversing
            elif self.current_trade is not None and signal != 0 and signal != self.current_trade.direction:
                if self.current_trade.holding_bars >= self.config['min_trade_holding_bars']:
                    # Exit current at market then enter new (reduces drawdown risk by forcing stops on reversals)
                    self._exit_trade(date, price, reason="signal_reversal")
                    # After exit, check drawdown suspension before re-enter
                    self._check_and_maybe_stop_trading_due_to_drawdown()
                    if not self.stopped_from_drawdown:
                        self._enter_trade(date, price, signal, row['atr'])

            # Record equity
            equity_curve.append(self.equity)
            dates.append(date)

        equity_series = pd.Series(data=equity_curve, index=dates)
        performance = compute_performance(equity_series)
        self.logger.info("Backtest completed. Total return: %.2f%% Max Drawdown: %.2f%% Sharpe: %.2f",
                         performance['total_return'] * 100,
                         performance['max_drawdown'] * 100,
                         performance['sharpe'] if not np.isnan(performance['sharpe']) else -999)
        return {
            'performance': performance,
            'trades': self.trades,
            'equity_curve': equity_series
        }


# =========================
# Example usage helper (keeps config/logging)
# =========================
def load_config(path: Optional[str] = None) -> Dict[str, Any]:
    config = deepcopy(DEFAULT_CONFIG)
    if path:
        try:
            with open(path, 'r') as f:
                user_config = json.load(f)
            # shallow update - preserve unknown keys in user_config
            config.update(user_config)
        except Exception as e:
            logging.getLogger("TradingSystem").warning("Failed to load config from %s, using defaults. Error: %s", path, e)
    return config


# =========================
# Module test / Backtest runner (can be used by existing architecture)
# =========================
if __name__ == "__main__":
    # This block is for quick local tests only. In production the system integrates into the existing architecture.
    import sys
    import argparse

    parser = argparse.ArgumentParser(description="Run trading system backtest (local test harness).")
    parser.add_argument("--data-csv", type=str, help="CSV file with columns: datetime,open,high,low,close,volume", required=False)
    parser.add_argument("--config", type=str, help="JSON config file to override defaults", required=False)
    args = parser.parse_args()

    cfg = load_config(args.config)
    logger = setup_logging(cfg.get("logging", {}))
    ts = TradingSystem(cfg)

    # If a CSV was supplied attempt to load it
    if args.data_csv:
        try:
            df = pd.read_csv(args.data_csv, parse_dates=['datetime'], index_col='datetime')
            # Ensure required columns exist
            for col in ['open', 'high', 'low', 'close']:
                if col not in df.columns:
                    raise ValueError(f"CSV missing required column: {col}")
            result = ts.run_backtest(df[['open', 'high', 'low', 'close']])
            perf = result['performance']
            logger.info("Finished. Total return %.2f%%, CAGR %.2f%%, MaxDD %.2f%%, Sharpe %.2f",
                        perf['total_return'] * 100, perf['cagr'] * 100, perf['max_drawdown'] * 100,
                        perf['sharpe'] if not np.isnan(perf['sharpe']) else -999)
            # Save trades log to csv for analysis
            trades_out = []
            for t in result['trades']:
                trades_out.append({
                    'entry_date': t.entry_date,
                    'exit_date': t.exit_date,
                    'entry_price': t.entry_price,
                    'exit_price': t.exit_price,
                    'size': t.size,
                    'direction': t.direction,
                    'pnl': t.pnl,
                    'holding_bars': t.holding_bars
                })
            if trades_out:
                trades_df = pd.DataFrame(trades_out)
                trades_df.to_csv("trades_log.csv", index=False)
                logger.info("Saved trades log to trades_log.csv")
        except Exception as e:
            logger.exception("Failed to run backtest: %s", e)
    else:
        logger.info("No data CSV provided. Example run skipped.")
        logger.info("To run backtest: python trading_system.py --data-csv YOUR_DATA.csv --config config.json")