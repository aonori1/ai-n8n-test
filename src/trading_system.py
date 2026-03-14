import argparse
import json
import logging
import math
import os
import sys
from collections import deque, namedtuple
from copy import deepcopy
from datetime import datetime

import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------
# Default configuration (kept as part of the architecture / preserved config)
# -----------------------------------------------------------------------------
DEFAULT_CONFIG = {
    "initial_capital": 100000.0,
    "fee_rate": 0.0005,  # proportional transaction cost
    "slippage": 0.0005,  # proportional slippage on entry/exit
    "target_annual_vol": 0.10,  # volatility targeting (10% annual)
    "vol_lookback_days": 21,
    "vol_annualization": 252,
    "max_leverage": 2.0,
    "max_position_pct": 0.5,  # max fraction of equity in single instrument
    "fast_sma": 20,
    "slow_sma": 100,
    "atr_period": 21,
    "atr_multiplier": 3.0,  # initial stop on entry = ATR * multiplier
    "trailing_atr_multiplier": 1.5,  # trailing stop = ATR * trailing_mult
    "signal_confirm_bars": 2,  # require signal to persist for N bars
    "min_trade_hold_bars": 2,
    "max_consecutive_losses": 4,
    "loss_pause_days": 10,  # pause trading after too many consecutive losses
    "max_drawdown_stop": 0.25,  # stop trading if equity drawdown exceeds 25%
    "reporting": {
        "log_level": "INFO",
        "log_file": None
    }
}

# -----------------------------------------------------------------------------
# Helper utilities
# -----------------------------------------------------------------------------
Trade = namedtuple("Trade", [
    "entry_dt", "exit_dt", "side", "entry_price", "exit_price",
    "quantity", "pnl", "fees", "slippage", "max_adverse_excursion", "max_favorable_excursion"
])


def setup_logger(cfg):
    level = getattr(logging, cfg["reporting"].get("log_level", "INFO").upper(), logging.INFO)
    logger = logging.getLogger("TradingSystem")
    logger.setLevel(level)
    if logger.handlers:
        # avoid duplicate handlers if called multiple times
        for h in logger.handlers[:]:
            logger.removeHandler(h)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    ch.setLevel(level)
    logger.addHandler(ch)
    if cfg["reporting"].get("log_file"):
        fh = logging.FileHandler(cfg["reporting"]["log_file"])
        fh.setFormatter(fmt)
        fh.setLevel(level)
        logger.addHandler(fh)
    return logger


def atr(high, low, close, period):
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(period, min_periods=1).mean()


def calculate_performance(equity_series, returns_series):
    # equity_series: pd.Series indexed by datetime
    # returns_series: pd.Series of periodic returns
    total_days = (equity_series.index[-1] - equity_series.index[0]).days
    years = max(total_days / 365.25, 1 / 365.25)
    cumulative_return = equity_series.iloc[-1] / equity_series.iloc[0] - 1.0
    cagr = (1.0 + cumulative_return) ** (1.0 / years) - 1.0
    ann_ret = (1 + returns_series.mean()) ** 252 - 1 if len(returns_series) >= 2 else 0.0
    ann_vol = returns_series.std() * math.sqrt(252) if len(returns_series) >= 2 else 0.0
    sharpe = (ann_ret / ann_vol) if ann_vol != 0 else 0.0
    # max drawdown
    rolling_max = equity_series.cummax()
    drawdown = (equity_series - rolling_max) / rolling_max
    max_dd = drawdown.min()
    # sortino
    neg_rets = returns_series[returns_series < 0]
    downside = neg_rets.std() * math.sqrt(252) if len(neg_rets) >= 1 else 0.0
    sortino = (ann_ret / downside) if downside != 0 else 0.0
    return {
        "cumulative_return": cumulative_return,
        "cagr": cagr,
        "annual_return": ann_ret,
        "annual_volatility": ann_vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_dd
    }


# -----------------------------------------------------------------------------
# Trading System Class (maintains architecture: data -> signals -> manage -> log)
# -----------------------------------------------------------------------------
class TradingSystem:
    def __init__(self, data, config=None, logger=None):
        self.cfg = deepcopy(DEFAULT_CONFIG)
        if config:
            # merge user config
            self._deep_update(self.cfg, config)
        self.logger = logger or setup_logger(self.cfg)
        self.data = data.copy()
        self._validate_data()
        self._prepare_indicators()
        self.equity = float(self.cfg["initial_capital"])
        self.cash = float(self.cfg["initial_capital"])
        self.position = 0.0  # positive for long, negative for short (this example only uses longs)
        self.position_entry_price = None
        self.position_entry_atr = None
        self.position_qty = 0.0
        self.position_entry_dt = None
        self.max_equity = self.equity
        self.last_trade_was_loss = False
        self.consecutive_losses = 0
        self.pause_until_index = -1
        self.trades = []
        self.equity_curve = pd.Series(index=self.data.index, dtype=float)
        self.daily_returns = pd.Series(index=self.data.index, dtype=float)
        self.pending_signals = deque(maxlen=self.cfg["signal_confirm_bars"])

    @staticmethod
    def _deep_update(base, new):
        for k, v in new.items():
            if isinstance(v, dict) and k in base and isinstance(base[k], dict):
                TradingSystem._deep_update(base[k], v)
            else:
                base[k] = v

    def _validate_data(self):
        required_cols = {"open", "high", "low", "close"}
        if not required_cols.issubset(set(self.data.columns)):
            raise ValueError(f"Data must contain columns: {required_cols}")

        # Ensure datetime index
        if not isinstance(self.data.index, pd.DatetimeIndex):
            if "date" in self.data.columns:
                self.data = self.data.set_index(pd.to_datetime(self.data["date"]))
            else:
                raise ValueError("Data must have a DatetimeIndex or a 'date' column.")
        self.data.sort_index(inplace=True)

    def _prepare_indicators(self):
        df = self.data
        cfg = self.cfg
        # SMAs
        df["sma_fast"] = df["close"].rolling(cfg["fast_sma"], min_periods=1).mean()
        df["sma_slow"] = df["close"].rolling(cfg["slow_sma"], min_periods=1).mean()
        # ATR
        df["atr"] = atr(df["high"], df["low"], df["close"], cfg["atr_period"])
        # daily returns and ewma volatility
        df["ret"] = df["close"].pct_change().fillna(0.0)
        df["vol_ewma"] = df["ret"].ewm(span=cfg["vol_lookback_days"], adjust=False).std() * math.sqrt(
            cfg["vol_annualization"])
        # momentum / mean reversion signals (basic: sma crossover)
        df["signal_raw"] = 0
        df.loc[df["sma_fast"] > df["sma_slow"], "signal_raw"] = 1
        df.loc[df["sma_fast"] < df["sma_slow"], "signal_raw"] = -1
        # Keep it simple: only take long side by default to reduce risk
        # However leave the signal_raw in place in case future expansion is needed
        self.data = df

    def _volatility_target_notional(self, idx):
        cfg = self.cfg
        vol = self.data["vol_ewma"].iat[idx]
        if vol <= 0 or np.isnan(vol):
            return 0.0
        target_vol = cfg["target_annual_vol"]
        equity = self.equity
        # Notional to allocate to instrument to target volatility
        notional = equity * (target_vol / vol)
        # Cap notional by max leverage and max position percent
        notional = min(notional, equity * cfg["max_leverage"])
        notional = min(notional, equity * cfg["max_position_pct"])
        return notional

    def _can_trade(self, idx):
        # respect pause and drawdown stop
        if idx <= self.pause_until_index:
            return False
        if self.max_equity > 0:
            drawdown = (self.max_equity - self.equity) / self.max_equity
            if drawdown >= self.cfg["max_drawdown_stop"]:
                self.logger.warning("Equity drawdown exceeded stop threshold. Halting trading.")
                return False
        return True

    def _compute_entry_price_with_slippage(self, price):
        # simulate slippage as a fraction of price (worse on entry)
        s = self.cfg["slippage"]
        return price * (1 + s)

    def _compute_exit_price_with_slippage(self, price):
        s = self.cfg["slippage"]
        return price * (1 - s)

    def _open_position(self, idx, entry_side=1):
        # Only allow long entries in this improved risk-focused system
        if entry_side <= 0:
            return False

        if not self._can_trade(idx):
            return False

        price = self.data["close"].iat[idx]
        entry_price = self._compute_entry_price_with_slippage(price)
        notional = self._volatility_target_notional(idx)
        if notional <= 0:
            return False
        qty = math.floor(notional / entry_price)
        if qty <= 0:
            return False
        fees = entry_price * qty * self.cfg["fee_rate"]
        cost = entry_price * qty + fees

        # Apply simple cash check - allow limited leverage via max_leverage notional above
        # We maintain cash as collateral; for sim we deduct fees now and treat position PnL in equity.
        self.cash -= fees
        self.position = entry_side
        self.position_qty = qty
        self.position_entry_price = entry_price
        self.position_entry_atr = self.data["atr"].iat[idx] if not np.isnan(self.data["atr"].iat[idx]) else 0.0
        self.position_entry_dt = self.data.index[idx]
        self.position_high_since_entry = entry_price
        self.position_low_since_entry = entry_price
        self.position_max_adverse = 0.0
        self.position_max_fav = 0.0
        if self.logger:
            self.logger.info(
                f"Open position qty={qty} entry_price={entry_price:.4f} dt={self.position_entry_dt} notional={qty * entry_price:.2f}")
        return True

    def _close_position(self, idx, reason="exit"):
        if self.position_qty == 0:
            return None

        exit_price = self.data["close"].iat[idx]
        exit_price = self._compute_exit_price_with_slippage(exit_price)
        qty = self.position_qty
        entry_price = self.position_entry_price
        pnl = (exit_price - entry_price) * qty
        fees = exit_price * qty * self.cfg["fee_rate"]
        self.cash += exit_price * qty - fees  # realize proceeds
        # Update equity
        realized_pnl = pnl - fees
        prev_equity = self.equity
        self.equity += realized_pnl
        # Track max equity for drawdown
        self.max_equity = max(self.max_equity, self.equity)
        # Reset position
        trade = Trade(
            entry_dt=self.position_entry_dt,
            exit_dt=self.data.index[idx],
            side=self.position,
            entry_price=entry_price,
            exit_price=exit_price,
            quantity=qty,
            pnl=realized_pnl,
            fees=fees,
            slippage=self.cfg["slippage"],
            max_adverse_excursion=self.position_max_adverse,
            max_favorable_excursion=self.position_max_fav
        )
        self.trades.append(trade)
        # logging
        if self.logger:
            self.logger.info(
                f"Close position qty={qty} exit_price={exit_price:.4f} dt={trade.exit_dt} pnl={realized_pnl:.2f} reason={reason}")
        # update loss counters
        if realized_pnl < 0:
            self.consecutive_losses += 1
            self.last_trade_was_loss = True
            if self.consecutive_losses >= self.cfg["max_consecutive_losses"]:
                self.pause_until_index = idx + self.cfg["loss_pause_days"]
                self.logger.warning(
                    f"Consecutive losses ({self.consecutive_losses}) >= max. Pausing trading until index {self.pause_until_index}.")
        else:
            self.consecutive_losses = 0
            self.last_trade_was_loss = False

        # Reset position trackers
        self.position = 0
        self.position_qty = 0
        self.position_entry_price = None
        self.position_entry_atr = None
        self.position_entry_dt = None
        self.position_high_since_entry = 0
        self.position_low_since_entry = 0
        self.position_max_adverse = 0.0
        self.position_max_fav = 0.0
        return trade

    def run_backtest(self):
        df = self.data
        n = len(df)
        cfg = self.cfg

        for idx in range(n):
            row_dt = df.index[idx]
            price = df["close"].iat[idx]

            # update daily equity for mark-to-market
            mtm_unrealized = 0.0
            if self.position_qty != 0:
                mtm_unrealized = (price - self.position_entry_price) * self.position_qty

            prev_equity = self.equity
            self.equity = self.cash + (price * self.position_qty if self.position_qty else 0) + mtm_unrealized
            self.max_equity = max(self.max_equity, self.equity)
            # store equity and returns
            self.equity_curve.iat[idx] = self.equity
            if idx == 0:
                self.daily_returns.iat[idx] = 0.0
            else:
                prev_equity_val = self.equity_curve.iat[idx - 1]
                self.daily_returns.iat[idx] = (self.equity_curve.iat[idx] / prev_equity_val - 1.0) if prev_equity_val > 0 else 0.0

            # compute signal and confirm persistence
            raw_sig = int(df["signal_raw"].iat[idx])
            self.pending_signals.append(raw_sig)
            # keep only positive signals for entry (reduce risk by ignoring shorts)
            confirmed_signal = 1 if len(self.pending_signals) == self.pending_signals.maxlen and all(s == 1 for s in self.pending_signals) else 0

            # Position management
            # Update MAE/MFE
            if self.position_qty != 0:
                # update highs/lows since entry
                self.position_high_since_entry = max(self.position_high_since_entry, row_dt and price or price)
                self.position_low_since_entry = min(self.position_low_since_entry, price)
                # compute excursions
                fav = (price - self.position_entry_price) * self.position_qty
                adv = (self.position_entry_price - price) * self.position_qty
                self.position_max_fav = max(self.position_max_fav, fav)
                self.position_max_adverse = max(self.position_max_adverse, adv)

            # Exit checks: ATR based stop-loss and trailing stop
            if self.position_qty != 0:
                current_atr = df["atr"].iat[idx]
                entry_atr = self.position_entry_atr if self.position_entry_atr else current_atr
                if entry_atr <= 0:
                    entry_atr = max(current_atr, 1e-6)
                # static stop-loss distance based on entry ATR
                static_stop = self.position_entry_price - cfg["atr_multiplier"] * entry_atr
                # trailing stop based on current ATR
                trailing_stop = price - cfg["trailing_atr_multiplier"] * current_atr
                # use the more conservative (higher) stop for long positions
                stop_price = max(static_stop, trailing_stop)
                # Also exit if trend reversed (fast sma below slow sma)
                trend_broken = df["sma_fast"].iat[idx] < df["sma_slow"].iat[idx]
                # Exit if price breaches stop or trend broken
                if price <= stop_price or trend_broken:
                    self._close_position(idx, reason="stop_or_trend_exit")
                    # after closing, continue to next index
                    continue

                # soft volatility spike exit: if vol jumps 2x and last trade was winner, tighten stop or exit
                vol = df["vol_ewma"].iat[idx]
                vol_prev = df["vol_ewma"].iat[max(0, idx - 1)]
                if vol_prev > 0 and vol / vol_prev >= 2.0:
                    # reduce position by half or exit if risk-averse
                    reduce_qty = max(1, math.floor(self.position_qty / 2))
                    # execute partial exit by selling reduce_qty at market
                    partial_exit_price = self._compute_exit_price_with_slippage(price)
                    pnl_partial = (partial_exit_price - self.position_entry_price) * reduce_qty
                    fees_partial = partial_exit_price * reduce_qty * cfg["fee_rate"]
                    self.cash += partial_exit_price * reduce_qty - fees_partial
                    self.position_qty -= reduce_qty
                    self.equity += pnl_partial - fees_partial
                    self.max_equity = max(self.max_equity, self.equity)
                    if self.logger:
                        self.logger.info(f"Vol spike: reduced position by {reduce_qty} at price {partial_exit_price:.4f}")

            # Entry checks
            if self.position_qty == 0 and confirmed_signal == 1:
                # avoid entry during pause or after large drawdown
                if self._can_trade(idx):
                    opened = self._open_position(idx, entry_side=1)
                    if opened:
                        # apply immediate stop placement information
                        if self.logger:
                            self.logger.debug(f"Position opened at index {idx} dt {row_dt}")
                else:
                    if self.logger:
                        self.logger.debug(f"Skipped opening because trading not allowed at idx {idx}")

            # Small risk control: if equity has dropped too much intraday (e.g., >10% from max), close
            if self.position_qty != 0:
                dd_intraday = (self.max_equity - self.equity) / self.max_equity if self.max_equity > 0 else 0
                if dd_intraday > 0.10:  # intraday severe drawdown threshold
                    self._close_position(idx, reason="intraday_equity_drawdown")

        # finalize: if position still open, close at last bar
        if self.position_qty != 0:
            self._close_position(len(df) - 1, reason="end_of_data")

        perf = calculate_performance(self.equity_curve.dropna(), self.daily_returns.dropna())
        self.logger.info("Backtest complete.")
        self.logger.info(f"Final equity: {self.equity:.2f}")
        self.logger.info(f"Trades: {len(self.trades)}")
        self.logger.info(f"Performance: {json.dumps(perf, default=lambda o: float(o))}")
        return {
            "equity_curve": self.equity_curve,
            "daily_returns": self.daily_returns,
            "trades": self.trades,
            "performance": perf
        }


# -----------------------------------------------------------------------------
# CLI and utility to load data/config and run system
# -----------------------------------------------------------------------------
def load_config(path):
    if not path:
        return {}
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    ext = os.path.splitext(path)[1].lower()
    with open(path, "r") as f:
        if ext in [".json"]:
            return json.load(f)
        else:
            # try JSON by default; YAML could be added if pyyaml present
            try:
                return json.load(f)
            except Exception:
                raise ValueError("Unsupported config file format; expected JSON.")


def load_data(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found: {path}")
    df = pd.read_csv(path, parse_dates=True, index_col=0)
    # Expect first column to be datetime index. Otherwise try to detect 'date' column.
    if not isinstance(df.index, pd.DatetimeIndex):
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            df.set_index("date", inplace=True)
        else:
            # try to parse index as date strings
            try:
                df.index = pd.to_datetime(df.index)
            except Exception:
                raise ValueError("Cannot determine datetime index for data. Provide a CSV with a date column or datetime index.")
    # Ensure required columns exist
    expected = {"open", "high", "low", "close"}
    cols = set([c.lower() for c in df.columns])
    # Normalize column names to lower-case
    df.columns = [c.lower() for c in df.columns]
    if not expected.issubset(set(df.columns)):
        raise ValueError(f"Data must contain columns: {expected}")
    return df


def main():
    parser = argparse.ArgumentParser(description="Improved Trading System Backtest")
    parser.add_argument("--data", required=True, help="CSV file with OHLC data (index or 'date' column)")
    parser.add_argument("--config", required=False, help="Optional JSON config file path")
    parser.add_argument("--logfile", required=False, help="Optional log file path (overrides config)")
    args = parser.parse_args()

    user_cfg = {}
    if args.config:
        user_cfg = load_config(args.config)
    if args.logfile:
        if "reporting" not in user_cfg:
            user_cfg["reporting"] = {}
        user_cfg["reporting"]["log_file"] = args.logfile

    logger = setup_logger(TradingSystem.load_cfg_if_exists(user_cfg) if False else DEFAULT_CONFIG)  # temporary default for logger init
    # Proper logger with merged config
    merged_cfg = deepcopy(DEFAULT_CONFIG)
    TradingSystem._deep_update(merged_cfg, user_cfg)
    logger = setup_logger(merged_cfg)
    # load data
    data = load_data(args.data)
    system = TradingSystem(data, config=user_cfg, logger=logger)
    results = system.run_backtest()

    # simple summary output to stdout
    perf = results["performance"]
    logger.info(f"Final equity: {system.equity:.2f}")
    logger.info(f"CAGR: {perf['cagr']:.2%}, Sharpe: {perf['sharpe']:.2f}, MaxDD: {perf['max_drawdown']:.2%}")
    # Save trades to CSV for downstream analysis
    trades_df = pd.DataFrame(results["trades"])
    try:
        trades_df.to_csv("trades.csv", index=False)
        logger.info("Trades saved to trades.csv")
    except Exception:
        logger.debug("Unable to write trades.csv")


# Add a static helper for potential external use (keeps architecture consistent)
setattr(TradingSystem, "load_cfg_if_exists", staticmethod(lambda cfg: cfg if cfg else DEFAULT_CONFIG))

if __name__ == "__main__":
    main()