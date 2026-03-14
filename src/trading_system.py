import json
import logging
import math
import os
from copy import deepcopy
from datetime import datetime
from typing import Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd


DEFAULT_CONFIG = {
    "initial_capital": 100000.0,
    "per_trade_risk": 0.01,               # Risk per trade as fraction of equity
    "atr_period": 14,
    "atr_multiplier": 3.0,                # Stop distance multiplier of ATR
    "ema_fast": 20,
    "ema_slow": 50,
    "rsi_period": 14,
    "rsi_long_threshold": 50,             # Simple trend filter
    "target_portfolio_vol": 0.10,         # Annualized target volatility for dynamic sizing
    "vol_lookback": 20,                   # Lookback for realized vol estimate
    "max_position_pct": 0.25,             # Maximum fraction of equity in one position
    "max_drawdown_allowed": 0.25,         # If drawdown exceeds this, pause trading
    "drawdown_scale_start": 0.05,         # Start scaling risk when dd exceeds this fraction
    "logging": {
        "level": "INFO",
        "log_to_file": False,
        "log_file": "trading_system.log"
    },
    "backtest": {
        "freq_per_year": 252  # used for annualization
    }
}


def setup_logger(config: Dict[str, Any]) -> logging.Logger:
    log_cfg = config.get("logging", {})
    level = getattr(logging, log_cfg.get("level", "INFO").upper(), logging.INFO)
    logger = logging.getLogger("TradingSystem")
    logger.setLevel(level)
    if not logger.handlers:
        fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
        ch = logging.StreamHandler()
        ch.setFormatter(fmt)
        logger.addHandler(ch)
        if log_cfg.get("log_to_file", False):
            fh = logging.FileHandler(log_cfg.get("log_file", "trading_system.log"))
            fh.setFormatter(fmt)
            logger.addHandler(fh)
    return logger


def load_config(path: Optional[str] = None) -> Dict[str, Any]:
    cfg = deepcopy(DEFAULT_CONFIG)
    if path and os.path.exists(path):
        try:
            with open(path, "r") as f:
                external = json.load(f)
            # deep merge external into default
            def deep_update(d, u):
                for k, v in u.items():
                    if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                        deep_update(d[k], v)
                    else:
                        d[k] = v
            deep_update(cfg, external)
        except Exception:
            pass
    return cfg


def atr(df: pd.DataFrame, n: int) -> pd.Series:
    # True range
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(n, min_periods=1).mean()


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def rsi(series: pd.Series, n: int) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.rolling(n, min_periods=1).mean()
    ma_down = down.rolling(n, min_periods=1).mean()
    rs = ma_up / (ma_down.replace(0, 1e-10))
    return 100 - (100 / (1 + rs))


def calculate_performance_metrics(equity_curve: pd.Series, freq_per_year: int = 252) -> Dict[str, Any]:
    returns = equity_curve.pct_change().fillna(0)
    total_return = equity_curve.iloc[-1] / equity_curve.iloc[0] - 1
    annualized_return = (1 + total_return) ** (freq_per_year / len(returns)) - 1 if len(returns) > 0 else 0.0
    ann_vol = returns.std() * np.sqrt(freq_per_year) if len(returns) > 1 else 0.0
    sharpe = (annualized_return / ann_vol) if ann_vol > 0 else 0.0
    # downside deviation
    neg_returns = returns[returns < 0]
    downside_deviation = neg_returns.std() * np.sqrt(freq_per_year) if len(neg_returns) > 1 else 0.0
    sortino = (annualized_return / downside_deviation) if downside_deviation > 0 else 0.0
    rolling_max = equity_curve.cummax()
    drawdown = (equity_curve / rolling_max) - 1.0
    max_dd = drawdown.min()
    # CAGR
    days = len(returns)
    cagr = (equity_curve.iloc[-1] / equity_curve.iloc[0]) ** (freq_per_year / days) - 1 if days > 0 else 0.0
    return {
        "total_return": total_return,
        "annualized_return": annualized_return,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_dd,
        "cagr": cagr
    }


class TradingSystem:
    """
    Modular trading system class that:
    - computes indicators
    - generates trade signals with trend filter
    - sizes positions using ATR-based volatility targeting and dynamic drawdown scaling
    - applies stops (hard and trailing)
    - logs activity and returns performance metrics

    Expected data format: pandas DataFrame with columns ['Open','High','Low','Close','Volume'] indexed by datetime
    """

    def __init__(self, market_data: pd.DataFrame, config_path: Optional[str] = None):
        self.config = load_config(config_path)
        self.logger = setup_logger(self.config)
        self.df = market_data.copy()
        self._validate_data()
        self.params = self.config
        self.initial_capital = float(self.config.get("initial_capital", 100000.0))
        self.cash = self.initial_capital
        self.position = 0.0  # number of units (positive long, negative short not implemented here)
        self.entry_price = None
        self.equity_history = []
        self.positions_history = []
        self.trade_log = []
        self.stop_price = None
        self.trail_distance_atr = None
        self.current_max_price = None  # for trailing stop only for longs
        self._precompute_indicators()
        self.logger.info("TradingSystem initialized with initial capital: %.2f", self.initial_capital)

    def _validate_data(self):
        required = {"Open", "High", "Low", "Close", "Volume"}
        missing = required - set(self.df.columns)
        if missing:
            raise ValueError(f"Market data missing required columns: {missing}")

    def _precompute_indicators(self):
        self.logger.debug("Precomputing indicators...")
        p = self.config
        self.df["ATR"] = atr(self.df, p["atr_period"])
        self.df["EMA_fast"] = ema(self.df["Close"], p["ema_fast"])
        self.df["EMA_slow"] = ema(self.df["Close"], p["ema_slow"])
        self.df["RSI"] = rsi(self.df["Close"], p["rsi_period"])
        # Realized vol (daily) used for portfolio volatility estimate
        self.df["ret"] = self.df["Close"].pct_change().fillna(0)
        self.df["realized_vol"] = self.df["ret"].rolling(p["vol_lookback"], min_periods=1).std() * math.sqrt(p["backtest"]["freq_per_year"])
        self.df.fillna(method="bfill", inplace=True)

    def _compute_risk_scale(self, equity_curve: pd.Series) -> float:
        # scale down risk when drawdowns grow; returns [0,1]
        if len(equity_curve) < 2:
            return 1.0
        rolling_max = equity_curve.cummax()
        current_dd = 1.0 - (equity_curve.iloc[-1] / rolling_max.iloc[-1])
        dd_start = self.config["drawdown_scale_start"]
        max_dd_allowed = self.config["max_drawdown_allowed"]
        if current_dd <= dd_start:
            return 1.0
        if current_dd >= max_dd_allowed:
            return 0.0
        # linear scaling between start and max
        scale = max(0.0, 1.0 - (current_dd - dd_start) / (max_dd_allowed - dd_start))
        self.logger.debug("Drawdown %.4f -> risk scale %.4f", current_dd, scale)
        return scale

    def _size_position(self, price: float, atr_value: float, equity: float, realized_vol: float) -> Tuple[float, float]:
        """
        Returns (position_size_in_units, stop_price)
        Position sizing uses:
        - per_trade_risk: fraction of equity to risk per trade
        - ATR-based stop distance
        - cap by max_position_pct
        - dynamic scaling by realized portfolio volatility vs target_portfolio_vol
        - scale down further based on drawdown
        """
        cfg = self.config
        per_trade_risk = cfg["per_trade_risk"]
        atr_mult = cfg["atr_multiplier"]
        max_pos_pct = cfg["max_position_pct"]
        target_vol = cfg["target_portfolio_vol"]
        freq = cfg["backtest"]["freq_per_year"]

        # compute stop distance and immediate risk per unit (in price terms)
        stop_distance = max(atr_value * atr_mult, atr_value * 0.5)  # avoid too-small stops
        stop_price = price - stop_distance  # only long logic here

        # dynamic volatility scaling: scale risk proportionally to target vol / current realized vol
        vol_scale = 1.0
        if realized_vol > 0:
            vol_scale = min(3.0, target_vol / realized_vol)  # avoid extreme scaling
        # compute raw position based on per-trade risk and stop distance
        raw_risk_amount = equity * per_trade_risk * vol_scale
        units = raw_risk_amount / stop_distance if stop_distance > 0 else 0.0

        # cap position by max position percent of equity
        max_dollars = equity * max_pos_pct
        max_units = max_dollars / price if price > 0 else 0.0
        units = max(0.0, min(units, max_units))

        self.logger.debug("Sizing position: price=%.2f atr=%.4f stop_dist=%.4f raw_risk=%.2f units=%.4f max_units=%.4f vol_scale=%.4f",
                          price, atr_value, stop_distance, raw_risk_amount, units, max_units, vol_scale)

        return units, stop_price

    def _enter_position(self, timestamp: pd.Timestamp, price: float, size_units: float, stop_price: float):
        if size_units <= 0:
            self.logger.debug("Size zero, no entry at %s", timestamp)
            return
        cost = size_units * price
        self.cash -= cost
        self.position += size_units
        self.entry_price = price
        self.stop_price = stop_price
        self.current_max_price = price
        self.logger.info("ENTER %s: size=%.4f price=%.2f cost=%.2f stop=%.2f cash=%.2f",
                         timestamp, size_units, price, cost, stop_price, self.cash)
        self.trade_log.append({
            "timestamp": timestamp,
            "action": "ENTER",
            "size": size_units,
            "price": price,
            "cost": cost,
            "stop": stop_price
        })

    def _exit_position(self, timestamp: pd.Timestamp, price: float, reason: str = "EXIT"):
        if self.position == 0:
            return
        proceeds = self.position * price
        pnl = proceeds - (self.position * (self.entry_price if self.entry_price is not None else price))
        self.cash += proceeds
        self.logger.info("%s %s: size=%.4f price=%.2f proceeds=%.2f pnl=%.2f cash=%.2f",
                         reason, timestamp, self.position, price, proceeds, pnl, self.cash)
        self.trade_log.append({
            "timestamp": timestamp,
            "action": reason,
            "size": self.position,
            "price": price,
            "proceeds": proceeds,
            "pnl": pnl
        })
        self.position = 0.0
        self.entry_price = None
        self.stop_price = None
        self.current_max_price = None

    def run_backtest(self) -> Dict[str, Any]:
        """
        Main backtest loop. Iterates bars, generates signals, sizes and executes trades,
        applies stops, records equity curve and performance.
        """
        df = self.df
        equity = self.initial_capital
        equity_series = []
        timestamps = []
        for idx, row in df.iterrows():
            price = row["Close"]
            atr_val = row["ATR"]
            realized_vol = row["realized_vol"]
            timestamps.append(idx)

            # equity and drawdown check
            eq = self.cash + self.position * price
            equity_series.append(eq)
            # determine if trading is allowed given drawdown
            current_equity_series = pd.Series(equity_series)
            risk_scale = self._compute_risk_scale(current_equity_series)
            if risk_scale == 0:
                # forced pause trading
                if self.position != 0:
                    # exit immediately at current price
                    self._exit_position(idx, price, reason="FORCED_EXIT_DD_LIMIT")
                # do not take new trades
                self.logger.warning("Trading paused due to max drawdown at %s. Equity %.2f", idx, eq)
                continue

            # trend filter: simple EMA condition and RSI
            is_trend_ok = (row["Close"] > row["EMA_slow"]) and (row["EMA_fast"] > row["EMA_slow"]) and (row["RSI"] > self.config["rsi_long_threshold"])

            # ENTRY logic: only go long when not in position and trend ok
            if self.position == 0 and is_trend_ok:
                # compute size with drawdown scaling applied
                units, stop_price = self._size_position(price, atr_val, eq * risk_scale, realized_vol)
                # apply drawdown scaling to units directly
                units = units * risk_scale
                if units > 0:
                    self._enter_position(idx, price, units, stop_price)

            # Manage existing position: trailing stop update and hard stop
            if self.position > 0:
                # update current_max_price for trailing stop
                if price > self.current_max_price:
                    self.current_max_price = price
                    # reset trailing stop to entry + some fraction? We'll keep ATR-based trail
                    self.stop_price = max(self.stop_price, price - row["ATR"] * self.config["atr_multiplier"])
                    self.logger.debug("Updated trailing stop to %.2f at %s", self.stop_price, idx)

                # check stop (hard stop)
                if price <= self.stop_price:
                    self._exit_position(idx, price, reason="STOP_HIT")
                else:
                    # optional profit taking: if price has moved n ATRs in favorable direction, reduce size (scale out)
                    # simple scale-out rule: if price >= entry + 2*ATR, sell half
                    if self.entry_price and price >= self.entry_price + 2.0 * atr_val:
                        # scale out half position to lock profits
                        sell_units = self.position * 0.5
                        proceeds = sell_units * price
                        self.position -= sell_units
                        self.cash += proceeds
                        self.logger.info("PARTIAL_TAKE %s: sold %.4f units at %.2f proceeds=%.2f remaining=%.4f", idx, sell_units, price, proceeds, self.position)
                        self.trade_log.append({
                            "timestamp": idx,
                            "action": "PARTIAL_TAKE",
                            "size": sell_units,
                            "price": price,
                            "proceeds": proceeds
                        })
                        # after partial take, tighten stop to breakeven if beneficial
                        self.stop_price = max(self.stop_price, self.entry_price)
                        if self.position == 0:
                            self.entry_price = None

            # record current equity explicitly at each bar
            # (already appended eq above; update last value in case of intrabar trades)
            equity_series[-1] = self.cash + self.position * price

        equity_indexed = pd.Series(equity_series, index=timestamps)
        metrics = calculate_performance_metrics(equity_indexed, self.config["backtest"]["freq_per_year"])
        results = {
            "equity_curve": equity_indexed,
            "metrics": metrics,
            "trade_log": self.trade_log,
            "config": self.config
        }
        self.logger.info("Backtest complete. Metrics: %s", metrics)
        return results


if __name__ == "__main__":
    # Example usage: load CSV with Date,Open,High,Low,Close,Volume
    import argparse

    parser = argparse.ArgumentParser(description="Run TradingSystem backtest")
    parser.add_argument("--data", required=True, help="CSV file with market data (Date,Open,High,Low,Close,Volume)")
    parser.add_argument("--config", required=False, help="Optional config JSON file to override defaults")
    parser.add_argument("--out", required=False, help="Optional CSV for equity curve output", default=None)
    args = parser.parse_args()

    data_file = args.data
    cfg_file = args.config

    df = pd.read_csv(data_file, parse_dates=["Date"])
    df.set_index("Date", inplace=True)
    ts = TradingSystem(df, cfg_file)
    res = ts.run_backtest()
    equity_curve = res["equity_curve"]
    if args.out:
        equity_curve.to_csv(args.out, header=["Equity"])
    # Log a brief summary
    logger = setup_logger(ts.config)
    metrics = res["metrics"]
    logger.info("Final equity: %.2f", equity_curve.iloc[-1])
    logger.info("Metrics: %s", metrics)