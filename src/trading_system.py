import os
import logging
import json
import math
from collections import deque
from datetime import datetime
from typing import Optional, Dict, Any, List

import numpy as np
import pandas as pd
import yaml

# ------------------------------------------------------------------------------
# Configuration loading (preserves original pattern if config file exists)
# ------------------------------------------------------------------------------
DEFAULT_CONFIG = {
    "symbol": "EURUSD",
    "initial_capital": 100000.0,
    "risk_per_trade": 0.01,              # fraction of equity to risk per trade (1%)
    "target_volatility": 0.08,           # annualized target volatility (8%)
    "atr_period": 14,
    "ema_fast": 20,
    "ema_slow": 50,
    "stop_atr_multiplier": 3.0,
    "trail_atr_multiplier": 2.0,
    "max_position_leverage": 5.0,        # maximum gross leverage
    "commission_per_trade": 0.0,
    "slippage": 0.0,
    "max_drawdown_stop": 0.20,           # if drawdown exceeds 20%, stop trading
    "reduce_risk_after_dd": 0.10,        # reduce risk to min if drawdown > 10%
    "min_risk_per_trade": 0.002,         # minimum risk per trade after drawdown reduction
    "equity_update_frequency": 1,        # update equity per bar
    "logging": {
        "log_file": "trading_system.log",
        "level": "INFO",
    },
    "max_positions": 1,                  # single instrument system by default
    "warmup_bars": 200,
}


def load_config(path: str = "config.yaml") -> Dict[str, Any]:
    if os.path.exists(path):
        with open(path, "r") as f:
            try:
                cfg = yaml.safe_load(f)
                merged = DEFAULT_CONFIG.copy()
                merged.update(cfg or {})
                return merged
            except Exception:
                # if YAML invalid, fallback
                return DEFAULT_CONFIG.copy()
    else:
        return DEFAULT_CONFIG.copy()


# ------------------------------------------------------------------------------
# Logging setup (preserves logging behavior)
# ------------------------------------------------------------------------------
def setup_logger(cfg: Dict[str, Any]) -> logging.Logger:
    log_cfg = cfg.get("logging", {})
    level_name = log_cfg.get("level", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    logger = logging.getLogger("TradingSystem")
    logger.setLevel(level)
    if not logger.handlers:
        fmt = logging.Formatter(
            "%(asctime)s %(levelname)s %(name)s: %(message)s", "%Y-%m-%d %H:%M:%S"
        )
        # stream handler
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        logger.addHandler(sh)
        # file handler
        log_file = log_cfg.get("log_file", "trading_system.log")
        fh = logging.FileHandler(log_file)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


# ------------------------------------------------------------------------------
# Utilities: ATR calculation and simple indicators (keeps simple local implementations)
# ------------------------------------------------------------------------------
def compute_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> float:
    """
    Compute the most recent ATR value using Wilder's smoothing.
    Input arrays must be aligned and length >= period+1.
    Returns last ATR (float).
    """
    if len(close) < period + 1:
        return float(np.nan)
    high = np.asarray(high)
    low = np.asarray(low)
    close = np.asarray(close)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum.reduce([tr1, tr2, tr3])
    # Wilder's moving average
    atr = tr[:period].mean()
    for t in tr[period:]:
        atr = (atr * (period - 1) + t) / period
    return float(atr)


def ema(series: np.ndarray, period: int) -> np.ndarray:
    """
    Compute EMA; returns array of same length, first values may be NaN until enough data.
    """
    series = np.asarray(series, dtype=float)
    out = np.full_like(series, np.nan)
    if len(series) == 0:
        return out
    alpha = 2.0 / (period + 1.0)
    # seed with simple average of first period
    if len(series) >= period:
        seed = series[:period].mean()
        out[period - 1] = seed
        for i in range(period, len(series)):
            out[i] = alpha * series[i] + (1 - alpha) * out[i - 1]
    return out


# ------------------------------------------------------------------------------
# Order / Position dataclasses (simple dict-based to preserve lightweight architecture)
# ------------------------------------------------------------------------------
class Order:
    def __init__(self, symbol: str, quantity: float, price: float, side: str, timestamp: datetime):
        self.symbol = symbol
        self.quantity = quantity
        self.price = price
        self.side = side  # 'buy' or 'sell'
        self.timestamp = timestamp
        self.filled = True  # assume immediate fill for backtest agent
        self.fee = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "quantity": self.quantity,
            "price": self.price,
            "side": self.side,
            "timestamp": self.timestamp.isoformat(),
            "fee": self.fee,
            "filled": self.filled,
        }


class Position:
    def __init__(self, symbol: str, quantity: float, entry_price: float, atr_stop: float):
        self.symbol = symbol
        self.quantity = quantity
        self.entry_price = entry_price
        self.atr_stop = atr_stop  # ATR-based stop distance (absolute price)
        self.highest_price = entry_price
        self.lowest_price = entry_price

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "quantity": self.quantity,
            "entry_price": self.entry_price,
            "atr_stop": self.atr_stop,
            "highest_price": self.highest_price,
            "lowest_price": self.lowest_price,
        }


# ------------------------------------------------------------------------------
# Core Trading System (keeps architecture consistent with a single class handling bars)
# ------------------------------------------------------------------------------
class TradingSystem:
    def __init__(self, config_path: str = "config.yaml"):
        self.config = load_config(config_path)
        self.logger = setup_logger(self.config)
        self.logger.debug("Loaded configuration: %s", json.dumps(self.config))
        self.symbol = self.config["symbol"]
        self.initial_capital = float(self.config["initial_capital"])
        self.cash = float(self.initial_capital)
        self.equity = float(self.initial_capital)
        self.peak_equity = float(self.initial_capital)
        self.max_drawdown_stop = float(self.config["max_drawdown_stop"])
        self.reduce_risk_after_dd = float(self.config["reduce_risk_after_dd"])
        self.min_risk_per_trade = float(self.config["min_risk_per_trade"])
        self.risk_per_trade = float(self.config["risk_per_trade"])
        self.target_volatility = float(self.config["target_volatility"])
        self.atr_period = int(self.config["atr_period"])
        self.ema_fast_period = int(self.config["ema_fast"])
        self.ema_slow_period = int(self.config["ema_slow"])
        self.stop_atr_multiplier = float(self.config["stop_atr_multiplier"])
        self.trail_atr_multiplier = float(self.config["trail_atr_multiplier"])
        self.max_leverage = float(self.config["max_position_leverage"])
        self.commission = float(self.config.get("commission_per_trade", 0.0))
        self.slippage = float(self.config.get("slippage", 0.0))
        self.max_positions = int(self.config.get("max_positions", 1))
        self.warmup_bars = int(self.config.get("warmup_bars", 200))

        # Internal state
        self.history = deque(maxlen=2000)  # stores dicts of bars
        self.trades: List[Dict[str, Any]] = []
        self.positions: Dict[str, Position] = {}
        self.next_order_id = 1
        self.paused = False  # paused after severe drawdown
        self.bar_count = 0

        # performance metrics
        self.equity_curve: List[Dict[str, Any]] = []

        self.logger.info("Trading system initialized for symbol %s", self.symbol)

    # -------------------------
    # Interface that the backtester calls
    # -------------------------
    def on_bar(self, bar: Dict[str, Any]):
        """
        bar: dict-like with keys: 'timestamp' (datetime or iso str), 'open', 'high', 'low', 'close', 'volume'
        This method is the principal entry point for each new bar.
        """
        ts = bar.get("timestamp", None)
        if isinstance(ts, str):
            ts = datetime.fromisoformat(ts)
        self.bar_count += 1

        # Append to history
        self.history.append({
            "timestamp": ts,
            "open": float(bar["open"]),
            "high": float(bar["high"]),
            "low": float(bar["low"]),
            "close": float(bar["close"]),
            "volume": float(bar.get("volume", 0.0))
        })

        # Must have enough bars to compute indicators
        if self.bar_count < self.warmup_bars:
            if self.bar_count % 50 == 0:
                self.logger.debug("Warming up: %d/%d", self.bar_count, self.warmup_bars)
            return

        # Update positions and metrics
        self.update_equity(bar)
        self.enforce_drawdown_controls()

        # Do not trade if paused for drawdown control
        if self.paused:
            self.logger.debug("Trading paused due to drawdown control.")
            self.manage_positions(bar)
            return

        # Generate signals
        signal = self.generate_signal()

        # Manage existing positions regardless of signal
        self.manage_positions(bar)

        # Entry logic
        if signal != 0:
            # Only allow new position if we have room
            if len(self.positions) < self.max_positions:
                self.attempt_entry(signal, bar)
            else:
                self.logger.debug("Max positions reached (%d). Skipping new entry.", self.max_positions)

    # -------------------------
    # Indicator and Signal Generation
    # -------------------------
    def get_series(self):
        df = pd.DataFrame(self.history)
        return df

    def generate_signal(self) -> int:
        """
        Simple EMA crossover with trend filter and volatility check.
        Returns: 1 for buy (long), -1 for sell (short), 0 for no trade.
        """
        df = self.get_series()
        close = df["close"].values
        if len(close) < max(self.ema_slow_period, self.atr_period) + 5:
            return 0

        ema_fast = ema(close, self.ema_fast_period)
        ema_slow = ema(close, self.ema_slow_period)
        # determine last valid indices
        idx = np.where(~np.isnan(ema_slow))[0]
        if len(idx) == 0:
            return 0
        last = idx[-1]
        # require a directional crossover: last bar fast crosses above slow
        prev = last - 1
        if prev < 0:
            return 0

        # Trend filter: only trade in direction of long-term ema slope
        long_term_slope = ema_slow[last] - ema_slow[max(0, last - 5)]
        # Avoid trading if slope is near flat
        slope_threshold = 1e-6  # small number to avoid division issues
        if abs(long_term_slope) < slope_threshold:
            return 0

        # Crossover detection
        cross_up = ema_fast[prev] <= ema_slow[prev] and ema_fast[last] > ema_slow[last]
        cross_down = ema_fast[prev] >= ema_slow[prev] and ema_fast[last] < ema_slow[last]

        # Additional volatility filter: ensure ATR not extremely small (avoid oversized positions)
        atr = compute_atr(df["high"].values, df["low"].values, df["close"].values, self.atr_period)
        if np.isnan(atr) or atr <= 0:
            return 0

        # If long term slope positive prefer only cross_up, else prefer cross_down
        if long_term_slope > 0 and cross_up:
            return 1
        if long_term_slope < 0 and cross_down:
            return -1
        return 0

    # -------------------------
    # Position sizing and entry
    # -------------------------
    def compute_position_size(self, price: float, atr: float) -> float:
        """
        Compute quantity to buy/sell using a volatility targeting + fixed fractional risk method:
        - Determine stop distance = stop_atr_multiplier * ATR
        - Risk per trade (dollar) = equity * risk_per_trade
        - Quantity = risk_per_trade_dollars / stop_distance (for underlying where risk per unit = stop_distance)
        - Also apply volatility targeting: desired_position_value = equity * target_volatility / asset_volatility
          where asset_volatility ~ (ATR / price) * sqrt(252)
        - Combine both: final_position_value = min(size_by_risk * price, size_by_vol_target)
        """
        if atr <= 0 or price <= 0:
            return 0.0

        # Adjust risk_per_trade if we already suffered drawdown beyond threshold
        rpt = self.risk_per_trade
        dd = (self.peak_equity - self.equity) / max(1e-9, self.peak_equity)
        if dd >= self.reduce_risk_after_dd:
            rpt = max(self.min_risk_per_trade, rpt * 0.5)
            self.logger.debug("Reducing risk per trade from %.4f to %.4f due to drawdown %.2f%%",
                              self.risk_per_trade, rpt, dd * 100.0)

        stop_distance = self.stop_atr_multiplier * atr  # absolute price distance
        risk_dollars = self.equity * rpt
        if stop_distance <= 0:
            return 0.0
        qty_by_risk = max(0.0, risk_dollars / stop_distance)

        # Volatility targeting (annualized)
        # approximate daily volatility = ATR/price; annualized = * sqrt(252)
        asset_vol = (atr / price) * math.sqrt(252.0)
        if asset_vol <= 0:
            target_position_value = self.equity * 0.0
        else:
            target_position_value = self.equity * (self.target_volatility / asset_vol)

        qty_by_vol = max(0.0, target_position_value / price)

        # Choose the more conservative size (min of the two)
        qty = min(qty_by_risk, qty_by_vol)

        # Apply leverage cap
        max_qty_by_leverage = (self.equity * self.max_leverage) / price
        qty = min(qty, max_qty_by_leverage)

        # Floor to integer lots/shares where applicable
        qty = math.floor(qty)
        return float(qty)

    def attempt_entry(self, signal: int, bar: Dict[str, Any]):
        """
        Attempt to enter a position given a signal (+1 long, -1 short).
        """
        df = self.get_series()
        price = float(bar["close"])
        atr = compute_atr(df["high"].values, df["low"].values, df["close"].values, self.atr_period)
        if np.isnan(atr) or atr <= 0:
            self.logger.debug("ATR not available or non-positive. Skipping entry.")
            return

        qty = self.compute_position_size(price, atr)
        if qty <= 0:
            self.logger.debug("Computed position size zero. Skipping entry.")
            return

        side = "buy" if signal > 0 else "sell"
        # For simplicity assume long-only system if config enforces single direction; but allow shorts if qty>0
        # Compute stop price
        stop_distance = self.stop_atr_multiplier * atr
        if side == "buy":
            stop_price = price - stop_distance
        else:
            stop_price = price + stop_distance

        # Create order and position
        order = Order(self.symbol, qty if side == "buy" else -qty, price, side, bar["timestamp"])
        order.fee = self.commission
        # Apply slippage and fees to cash
        trade_value = qty * price
        slippage_cost = trade_value * self.slippage
        total_cost = trade_value + slippage_cost + order.fee if side == "buy" else -trade_value + slippage_cost + order.fee

        # Update cash and positions
        if side == "buy":
            self.cash -= (trade_value + slippage_cost + order.fee)
        else:
            # For short we assume proceeds credited; keep cash unchanged but mark position negative
            self.cash += (trade_value - slippage_cost - order.fee)

        pos = Position(self.symbol, qty if side == "buy" else -qty, price, stop_distance)
        self.positions[self.symbol] = pos
        self.logger.info("ENTER %s %d @ %.5f | stop_distance=%.5f | equity=%.2f",
                         side.upper(), abs(qty), price, stop_distance, self.equity)
        self.trades.append({
            "timestamp": bar["timestamp"].isoformat() if isinstance(bar["timestamp"], datetime) else str(bar["timestamp"]),
            "action": "enter",
            "side": side,
            "quantity": qty,
            "price": price,
            "fee": order.fee,
            "slippage": slippage_cost,
            "equity": self.equity,
            "cash": self.cash
        })

    # -------------------------
    # Position management and exits
    # -------------------------
    def manage_positions(self, bar: Dict[str, Any]):
        """
        Manage open positions: update trailing stops, check stop-loss and take-profit conditions.
        """
        if not self.positions:
            return

        df = self.get_series()
        price = float(bar["close"])
        timestamp = bar["timestamp"]

        # Check each position (single-instrument case)
        to_close = []
        for sym, pos in list(self.positions.items()):
            # Update highest/lowest price for trailing
            if pos.quantity > 0:
                pos.highest_price = max(pos.highest_price, price)
            else:
                pos.lowest_price = min(pos.lowest_price, price)

            # Trailing stop logic: update logical stop price based on ATR * trail multiplier
            atr = compute_atr(df["high"].values, df["low"].values, df["close"].values, self.atr_period)
            if np.isnan(atr) or atr <= 0:
                continue

            if pos.quantity > 0:
                trailing_stop_price = pos.highest_price - self.trail_atr_multiplier * atr
                fixed_stop = pos.entry_price - pos.atr_stop
                effective_stop = max(fixed_stop, trailing_stop_price)
                # Check hit
                if bar["low"] <= effective_stop:
                    self.logger.info("TRAIL/STOP HIT LONG at %.5f (stop=%.5f)", effective_stop, effective_stop)
                    to_close.append((sym, effective_stop, "stop"))
            else:
                trailing_stop_price = pos.lowest_price + self.trail_atr_multiplier * atr
                fixed_stop = pos.entry_price + pos.atr_stop
                effective_stop = min(fixed_stop, trailing_stop_price)
                if bar["high"] >= effective_stop:
                    self.logger.info("TRAIL/STOP HIT SHORT at %.5f (stop=%.5f)", effective_stop, effective_stop)
                    to_close.append((sym, effective_stop, "stop"))

        for sym, exit_price, reason in to_close:
            self.exit_position(sym, exit_price, bar["timestamp"], reason)

    def exit_position(self, symbol: str, price: float, timestamp: datetime, reason: str = "exit"):
        """
        Exit a position at given price (assume immediate fill).
        """
        pos = self.positions.get(symbol)
        if pos is None:
            return
        qty = pos.quantity
        side = "sell" if qty > 0 else "buy"  # to close long we sell, to close short we buy
        trade_value = abs(qty) * price
        slippage_cost = trade_value * self.slippage
        fee = self.commission

        if qty > 0:
            # closing long: receive cash
            self.cash += (trade_value - slippage_cost - fee)
        else:
            # closing short: pay back proceeds
            self.cash -= (trade_value + slippage_cost + fee)

        pnl = (price - pos.entry_price) * qty  # qty sign reflects direction
        realized_pnl = pnl - fee - slippage_cost
        self.equity = self.cash  # equity is cash + value of other positions; here we assume single pos
        # Remove position
        del self.positions[symbol]
        self.trades.append({
            "timestamp": timestamp.isoformat() if isinstance(timestamp, datetime) else str(timestamp),
            "action": "exit",
            "side": side,
            "quantity": abs(qty),
            "price": price,
            "fee": fee,
            "slippage": slippage_cost,
            "pnl": realized_pnl,
            "reason": reason,
            "equity": self.equity,
            "cash": self.cash
        })
        self.logger.info("EXIT %s %d @ %.5f | PnL=%.2f | reason=%s | equity=%.2f", side.upper(), abs(qty),
                         price, realized_pnl, reason, self.equity)

    # -------------------------
    # Equity and risk controls
    # -------------------------
    def update_equity(self, bar: Dict[str, Any]):
        """
        Update equity based on mark-to-market of open positions.
        """
        price = float(bar["close"])
        # Start with cash
        total = self.cash
        for pos in self.positions.values():
            total += pos.quantity * price
        self.equity = total

        # Update peak equity for drawdown tracking
        if self.equity > self.peak_equity:
            self.peak_equity = self.equity

        # Record equity curve point
        self.equity_curve.append({
            "timestamp": bar["timestamp"].isoformat() if isinstance(bar["timestamp"], datetime) else str(bar["timestamp"]),
            "equity": self.equity,
            "cash": self.cash,
            "positions_value": sum(pos.quantity * price for pos in self.positions.values())
        })

        # Log minor updates occasionally
        if len(self.equity_curve) % max(1, int(self.config.get("equity_update_frequency", 1))) == 0:
            self.logger.debug("Equity updated: %.2f (cash=%.2f, peak=%.2f)", self.equity, self.cash, self.peak_equity)

    def enforce_drawdown_controls(self):
        """
        Enforce maximum drawdown stop and risk reduction after moderate drawdown.
        """
        if self.peak_equity <= 0:
            return
        dd = (self.peak_equity - self.equity) / max(1e-9, self.peak_equity)
        if dd >= self.max_drawdown_stop:
            # Hard stop: close all positions and pause trading
            self.logger.warning("Max drawdown exceeded: %.2f%% >= %.2f%%. Closing all positions and pausing trading.",
                                dd * 100.0, self.max_drawdown_stop * 100.0)
            # Close all positions at last known price (mark-to-market)
            if self.history:
                last_bar = self.history[-1]
                for sym in list(self.positions.keys()):
                    self.exit_position(sym, float(last_bar["close"]), last_bar["timestamp"], reason="max_drawdown")
            self.paused = True
        elif dd >= self.reduce_risk_after_dd:
            # Soft control: reduce risk per trade (handled in compute_position_size) and log once
            self.logger.info("Drawdown %.2f%% reached; system will reduce risk per trade.", dd * 100.0)

    # -------------------------
    # End of run reporting, helpers
    # -------------------------
    def on_end(self):
        """
        Called at the end of the backtest to finalize metrics and logs.
        """
        # Close remaining positions at last price
        if self.positions and self.history:
            last_bar = self.history[-1]
            for sym in list(self.positions.keys()):
                self.exit_position(sym, float(last_bar["close"]), last_bar["timestamp"], reason="end_of_run")

        # Simple performance metrics
        pnl = self.equity - self.initial_capital
        dd_series = [self.peak_equity - e["equity"] for e in self.equity_curve]
        max_dd = max(dd_series) if dd_series else 0.0
        self.logger.info("Backtest complete. Final equity: %.2f, PnL: %.2f, Max Drawdown: %.2f",
                         self.equity, pnl, max_dd)

        # Save trades and equity curve to files for downstream analysis
        try:
            with open("trades.json", "w") as f:
                json.dump(self.trades, f, default=str, indent=2)
            eq_df = pd.DataFrame(self.equity_curve)
            eq_df.to_csv("equity_curve.csv", index=False)
            self.logger.info("Saved trades.json and equity_curve.csv")
        except Exception as e:
            self.logger.exception("Error saving output files: %s", e)

    # -------------------------
    # Utilities for external callers
    # -------------------------
    def get_state(self) -> Dict[str, Any]:
        """
        Returns the current internal state. Useful for unit tests and integration.
        """
        return {
            "cash": self.cash,
            "equity": self.equity,
            "peak_equity": self.peak_equity,
            "positions": {k: v.to_dict() for k, v in self.positions.items()},
            "trades_count": len(self.trades),
            "paused": self.paused
        }


# ------------------------------------------------------------------------------
# If run as script, provide a minimal demo/backtest runner that preserves architecture
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    # Minimal demonstration: load OHLC CSV if exists, otherwise no-op.
    cfg = load_config("config.yaml")
    logger = setup_logger(cfg)
    ts = TradingSystem("config.yaml")

    demo_csv = cfg.get("demo_ohlc_csv", "demo_ohlc.csv")
    if os.path.exists(demo_csv):
        logger.info("Running demo backtest on %s", demo_csv)
        df = pd.read_csv(demo_csv, parse_dates=["timestamp"])
        # Ensure proper columns
        expected_cols = {"timestamp", "open", "high", "low", "close", "volume"}
        if not expected_cols.issubset(set(df.columns)):
            logger.error("CSV missing required columns. Expected at least: %s", expected_cols)
        else:
            for _, row in df.iterrows():
                bar = {
                    "timestamp": row["timestamp"],
                    "open": float(row["open"]),
                    "high": float(row["high"]),
                    "low": float(row["low"]),
                    "close": float(row["close"]),
                    "volume": float(row.get("volume", 0.0)),
                }
                ts.on_bar(bar)
            ts.on_end()
    else:
        logger.info("No demo OHLC file found (%s). Trading system initialized and ready.", demo_csv)