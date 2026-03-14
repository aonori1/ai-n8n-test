#!/usr/bin/env python3
"""
Production-ready modular algorithmic trading backtesting engine.

Features:
- Modular structure in a single-file distribution:
    - Config handling
    - Logging setup
    - Data loading
    - Strategy (Moving Average Crossover example)
    - Risk management (fixed-fraction sizing, max exposure, max drawdown, stop-loss/take-profit)
    - Broker simulator (order execution, commission, slippage)
    - Backtester orchestration
    - Performance metrics and result export
- Ready for backtesting with CSV OHLCV data
- Logging to console and rotating file
- Configurable through JSON file (or default config)
- Minimal dependencies: pandas, numpy (matplotlib optional)
"""

from __future__ import annotations
import os
import json
import math
import logging
import logging.handlers
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime
import pandas as pd
import numpy as np
import traceback

# Optional plotting
try:
    import matplotlib.pyplot as plt  # type: ignore
    _HAS_MPL = True
except Exception:
    _HAS_MPL = False

# -----------------------------
# Configuration handling
# -----------------------------

DEFAULT_CONFIG = {
    "data": {
        "path": "data/ohlcv.csv",  # CSV with Date,Open,High,Low,Close,Volume
        "datetime_col": "Date",
        "datetime_format": None,
        "index_col": "Date",
        "parse_dates": True,
        "required_cols": ["Open", "High", "Low", "Close", "Volume"],
        "cycle": "1D"  # not used but reserved
    },
    "strategy": {
        "name": "MovingAverageCrossover",
        "short_window": 20,
        "long_window": 50,
        "signal_delay": 1,  # execute on next bar (1) by default
        "allow_short": False,
        "min_volume": 0
    },
    "risk": {
        "initial_capital": 100000.0,
        "risk_per_trade": 0.01,  # fraction of capital to risk per trade
        "max_drawdown": 0.2,  # max allowed peak-to-valley drawdown (stop/backtest)
        "max_position_size": 0.5,  # max fraction of capital in single instrument
        "commission_per_trade": 1.0,  # flat commission per trade
        "slippage": 0.0005,  # fraction of price for slippage per trade
        "stop_loss_pct": 0.02,  # default stop loss distance (2%)
        "take_profit_pct": 0.04  # default take profit (4%)
    },
    "execution": {
        "order_size_type": "units",  # "units" or "value". By default we'll size by risk (units)
        "tick_size": 0.01
    },
    "backtest": {
        "start_date": None,
        "end_date": None,
        "verbose": True,
        "save_results_to": "results",
        "plot_results": True
    },
    "logging": {
        "level": "INFO",
        "log_dir": "logs",
        "log_file": "backtest.log",
        "max_bytes": 5 * 1024 * 1024,
        "backup_count": 3
    }
}


class Config:
    """Simple JSON-backed configuration loader with defaults."""

    def __init__(self, path: Optional[str] = None):
        self._config = DEFAULT_CONFIG.copy()
        if path:
            self.load(path)

    def load(self, path: str) -> None:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Config file not found: {path}")
        with open(path, "r") as f:
            raw = json.load(f)
        self._merge(self._config, raw)

    def _merge(self, base: Dict[str, Any], override: Dict[str, Any]) -> None:
        for k, v in override.items():
            if k in base and isinstance(base[k], dict) and isinstance(v, dict):
                self._merge(base[k], v)
            else:
                base[k] = v

    def get(self, key_path: str, default: Any = None) -> Any:
        parts = key_path.split(".")
        obj = self._config
        for p in parts:
            if isinstance(obj, dict) and p in obj:
                obj = obj[p]
            else:
                return default
        return obj

    def as_dict(self) -> Dict[str, Any]:
        return self._config


# -----------------------------
# Logging setup
# -----------------------------
def setup_logging(cfg: Config) -> logging.Logger:
    log_cfg = cfg.get("logging", {})
    log_dir = log_cfg.get("log_dir", "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, log_cfg.get("log_file", "backtest.log"))
    level = getattr(logging, log_cfg.get("level", "INFO").upper(), logging.INFO)
    logger = logging.getLogger("backtester")
    logger.setLevel(level)
    logger.handlers = []  # reset

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch_formatter = logging.Formatter("%(asctime)s %(levelname)-8s %(message)s")
    ch.setFormatter(ch_formatter)
    logger.addHandler(ch)

    # Rotating file handler
    fh = logging.handlers.RotatingFileHandler(
        filename=log_file,
        maxBytes=log_cfg.get("max_bytes", 5 * 1024 * 1024),
        backupCount=log_cfg.get("backup_count", 3),
        encoding="utf-8"
    )
    fh.setLevel(level)
    fh_formatter = logging.Formatter("%(asctime)s %(levelname)-8s %(name)s: %(message)s")
    fh.setFormatter(fh_formatter)
    logger.addHandler(fh)
    logger.propagate = False
    return logger


# -----------------------------
# Data Loader
# -----------------------------
class DataLoader:
    """
    DataLoader loads historical OHLCV data for backtesting.
    Expected CSV columns: Date (or configured index), Open, High, Low, Close, Volume
    """

    def __init__(self, cfg: Config, logger: logging.Logger):
        self.cfg = cfg
        self.logger = logger

    def load_csv(self, path: str) -> pd.DataFrame:
        cfg_data = self.cfg.get("data", {})
        datetime_col = cfg_data.get("datetime_col", "Date")
        parse_dates = cfg_data.get("parse_dates", True)
        date_format = cfg_data.get("datetime_format", None)

        self.logger.info("Loading data from %s", path)
        df = pd.read_csv(path, parse_dates=[datetime_col] if parse_dates else None)
        if datetime_col in df.columns:
            df.set_index(datetime_col, inplace=True)
        else:
            # If index_col specified and present
            index_col = cfg_data.get("index_col", "Date")
            if index_col in df.columns:
                df.set_index(index_col, inplace=True)
        # Ensure required columns exist
        required = cfg_data.get("required_cols", ["Open", "High", "Low", "Close", "Volume"])
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns in data: {missing}")
        # Sort by index ascending
        df.sort_index(inplace=True)
        self.logger.info("Loaded %d rows", len(df))
        return df


# -----------------------------
# Strategy
# -----------------------------
class StrategyBase:
    """Base class for strategies."""

    def __init__(self, cfg: Config, logger: logging.Logger):
        self.cfg = cfg
        self.logger = logger

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Given OHLCV data, return a DataFrame with a 'signal' column:
        1 = long, -1 = short, 0 = flat.
        """
        raise NotImplementedError


class MovingAverageCrossoverStrategy(StrategyBase):
    """Simple moving average crossover strategy producing discrete signals."""

    def __init__(self, cfg: Config, logger: logging.Logger):
        super().__init__(cfg, logger)
        s = cfg.get("strategy", {})
        self.short = s.get("short_window", 20)
        self.long = s.get("long_window", 50)
        self.allow_short = s.get("allow_short", False)
        self.signal_delay = s.get("signal_delay", 1)

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        self.logger.info("Generating signals: short=%d long=%d", self.short, self.long)
        df["ma_short"] = df["Close"].rolling(self.short, min_periods=1).mean()
        df["ma_long"] = df["Close"].rolling(self.long, min_periods=1).mean()
        df["raw_signal"] = 0
        df.loc[df["ma_short"] > df["ma_long"], "raw_signal"] = 1
        if self.allow_short:
            df.loc[df["ma_short"] < df["ma_long"], "raw_signal"] = -1
        else:
            df.loc[df["ma_short"] < df["ma_long"], "raw_signal"] = 0
        # Convert to discrete changes: signal only when crossover occurs
        df["signal"] = df["raw_signal"].shift(self.signal_delay).fillna(0).astype(int)
        # Ensure signal values are -1, 0, 1
        df["signal"] = df["signal"].where(df["signal"].isin([-1, 0, 1]), 0)
        return df


# -----------------------------
# Risk management
# -----------------------------
@dataclass
class Position:
    entry_time: pd.Timestamp
    entry_price: float
    units: int
    side: int  # 1 for long, -1 for short
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    exit_time: Optional[pd.Timestamp] = None
    exit_price: Optional[float] = None
    profit: Optional[float] = None
    commission: float = 0.0


class RiskManager:
    """
    Calculates position size and enforces risk rules.
    """

    def __init__(self, cfg: Config, logger: logging.Logger):
        self.cfg = cfg
        self.logger = logger
        self.initial_capital = float(self.cfg.get("risk.initial_capital", 100000.0))
        self.risk_per_trade = float(self.cfg.get("risk.risk_per_trade", 0.01))
        self.max_position_size = float(self.cfg.get("risk.max_position_size", 0.5))
        self.max_drawdown = float(self.cfg.get("risk.max_drawdown", 0.2))
        self.stop_loss_pct = float(self.cfg.get("risk.stop_loss_pct", 0.02))
        self.take_profit_pct = float(self.cfg.get("risk.take_profit_pct", 0.04))
        self.current_peak = self.initial_capital
        self.allow_short = bool(self.cfg.get("strategy.allow_short", False))

    def compute_units(self,
                      capital: float,
                      price: float,
                      side: int,
                      stop_loss_price: Optional[float] = None) -> int:
        """
        Compute units to trade given capital, price, and stop-loss.
        Uses fixed fractional risk per trade.
        If stop_loss_price is None, uses default stop_loss_pct (symmetrical).
        """
        if price <= 0:
            return 0
        if stop_loss_price is None:
            if side == 1:
                stop_distance = price * self.stop_loss_pct
            else:
                stop_distance = price * self.stop_loss_pct
        else:
            stop_distance = abs(price - stop_loss_price)
            # Prevent zero or extremely small stop distances
            min_tick = float(self.cfg.get("execution.tick_size", 0.01))
            stop_distance = max(stop_distance, min_tick)
        risk_amount = capital * self.risk_per_trade
        units = int(math.floor(risk_amount / stop_distance))
        # Enforce max position size
        max_units_by_value = int(math.floor((capital * self.max_position_size) / price))
        if max_units_by_value <= 0:
            return 0
        units = min(units, max_units_by_value)
        self.logger.debug("Computed units: %d (risk_amount=%.2f stop_distance=%.5f max_units=%d)",
                          units, risk_amount, stop_distance, max_units_by_value)
        return max(units, 0)

    def update_peak(self, capital: float):
        if capital > self.current_peak:
            self.current_peak = capital

    def check_drawdown(self, capital: float) -> bool:
        """
        Returns True if drawdown exceeds allowed maximum (i.e., should stop/backtest).
        """
        self.update_peak(capital)
        drawdown = (self.current_peak - capital) / max(1e-9, self.current_peak)
        self.logger.debug("Checking drawdown: peak=%.2f current=%.2f drawdown=%.4f", self.current_peak, capital, drawdown)
        return drawdown > self.max_drawdown


# -----------------------------
# Broker Simulator
# -----------------------------
class BrokerSimulator:
    """
    Simulates order execution, positions, cash, slippage, and commission.
    """

    def __init__(self, cfg: Config, logger: logging.Logger):
        self.cfg = cfg
        self.logger = logger
        self.reset()

    def reset(self):
        self.capital = float(self.cfg.get("risk.initial_capital", 100000.0))
        self.cash = self.capital
        self.positions: List[Position] = []
        self.open_position: Optional[Position] = None
        self.trade_log: List[Dict[str, Any]] = []
        self.total_commissions = 0.0
        self.equity_history: List[Tuple[pd.Timestamp, float]] = []

    def apply_slippage(self, price: float, side: int) -> float:
        slippage_frac = float(self.cfg.get("risk.slippage", 0.0005))
        if side == 1:
            # buy: price + slippage
            return price * (1.0 + slippage_frac)
        else:
            # sell: price - slippage
            return price * (1.0 - slippage_frac)

    def execute_market_order(self,
                             time: pd.Timestamp,
                             price: float,
                             units: int,
                             side: int,
                             stop_loss: Optional[float],
                             take_profit: Optional[float]) -> Optional[Position]:
        """
        Execute a market order (simplified). Returns Position object if opened.
        """
        if units <= 0:
            return None
        executed_price = self.apply_slippage(price, side)
        commission = float(self.cfg.get("risk.commission_per_trade", 1.0))
        cost = executed_price * units * side  # side=1 => positive cost (we buy, cash reduces)
        # For buys, cost is positive cash outflow; for sells (shorting) we don't implement margin handling here
        # For simplification: This engine supports only long positions by default (short optional).
        if side == 1:
            total_outflow = executed_price * units + commission
            if total_outflow > self.cash + 1e-9:
                self.logger.warning("Insufficient cash: required=%.2f available=%.2f", total_outflow, self.cash)
                return None
            self.cash -= total_outflow
        else:
            # For short, we will not model margin; allow proceeds into cash
            proceeds = executed_price * units - commission
            self.cash += proceeds

        pos = Position(
            entry_time=time,
            entry_price=executed_price,
            units=units,
            side=side,
            stop_loss=stop_loss,
            take_profit=take_profit,
            commission=commission
        )
        self.open_position = pos
        self.positions.append(pos)
        self.total_commissions += commission
        self.logger.info("Opened position: time=%s side=%s units=%d price=%.4f commission=%.2f",
                         time, "LONG" if side == 1 else "SHORT", units, executed_price, commission)
        return pos

    def close_position(self, time: pd.Timestamp, price: float) -> Optional[Position]:
        """Close the currently open position at market (simulated)."""
        pos = self.open_position
        if pos is None:
            return None
        executed_price = self.apply_slippage(price, -pos.side)
        commission = float(self.cfg.get("risk.commission_per_trade", 1.0))
        if pos.side == 1:
            proceeds = executed_price * pos.units - commission
            self.cash += proceeds
            profit = (executed_price - pos.entry_price) * pos.units - (pos.commission + commission)
        else:
            # short close: buy to cover; simplified
            cost = executed_price * pos.units + commission
            self.cash -= cost
            profit = (pos.entry_price - executed_price) * pos.units - (pos.commission + commission)
        pos.exit_time = time
        pos.exit_price = executed_price
        pos.profit = profit
        pos.commission += commission
        self.total_commissions += commission
        self.trade_log.append({
            "entry_time": pos.entry_time,
            "exit_time": pos.exit_time,
            "entry_price": pos.entry_price,
            "exit_price": pos.exit_price,
            "units": pos.units,
            "side": pos.side,
            "profit": pos.profit,
            "commission": pos.commission
        })
        self.logger.info("Closed position: entry=%s exit=%s units=%d entry_price=%.4f exit_price=%.4f profit=%.2f",
                         pos.entry_time, pos.exit_time, pos.units, pos.entry_price, pos.exit_price, pos.profit)
        self.open_position = None
        return pos

    def mark_to_market(self, time: pd.Timestamp, price: float) -> float:
        """Compute equity (cash + unrealized pnl) at given price/time."""
        unrealized = 0.0
        if self.open_position is not None:
            pos = self.open_position
            if pos.side == 1:
                unrealized = (price - pos.entry_price) * pos.units
            else:
                unrealized = (pos.entry_price - price) * pos.units
        equity = self.cash + unrealized
        self.equity_history.append((time, equity))
        return equity


# -----------------------------
# Metrics and reporting
# -----------------------------
class Metrics:
    @staticmethod
    def equity_series(trade_engine: BrokerSimulator) -> pd.Series:
        if not trade_engine.equity_history:
            return pd.Series(dtype=float)
        times, eq = zip(*trade_engine.equity_history)
        return pd.Series(list(eq), index=pd.to_datetime(list(times)))

    @staticmethod
    def compute_performance(equity: pd.Series) -> Dict[str, Any]:
        px = equity.dropna()
        if px.empty:
            return {}
        returns = px.pct_change().fillna(0)
        total_return = px.iloc[-1] / px.iloc[0] - 1
        days = (px.index[-1] - px.index[0]).days
        years = max(days / 365.25, 1e-9)
        cagr = (px.iloc[-1] / px.iloc[0]) ** (1.0 / years) - 1 if years > 0 else float('nan')
        ann_vol = returns.std() * np.sqrt(252) if len(returns) > 1 else float('nan')
        sharpe = (returns.mean() * 252) / ann_vol if ann_vol and not math.isnan(ann_vol) else float('nan')
        drawdown = Metrics.max_drawdown(px)
        return {
            "total_return": float(total_return),
            "cagr": float(cagr),
            "ann_vol": float(ann_vol) if not math.isnan(ann_vol) else None,
            "sharpe": float(sharpe) if not math.isnan(sharpe) else None,
            "max_drawdown": float(drawdown)
        }

    @staticmethod
    def max_drawdown(series: pd.Series) -> float:
        roll_max = series.cummax()
        drawdown = (roll_max - series) / roll_max
        return float(drawdown.max()) if not drawdown.empty else 0.0


# -----------------------------
# Backtester orchestration
# -----------------------------
class Backtester:
    def __init__(self, cfg: Config, logger: logging.Logger):
        self.cfg = cfg
        self.logger = logger
        self.data_loader = DataLoader(cfg, logger)
        # instantiate strategy
        strat_name = cfg.get("strategy.name", "MovingAverageCrossover")
        if strat_name == "MovingAverageCrossover":
            self.strategy = MovingAverageCrossoverStrategy(cfg, logger)
        else:
            self.strategy = MovingAverageCrossoverStrategy(cfg, logger)  # fallback
        self.risk_manager = RiskManager(cfg, logger)
        self.broker = BrokerSimulator(cfg, logger)

    def run(self, data: pd.DataFrame) -> Dict[str, Any]:
        self.logger.info("Starting backtest")
        self.broker.reset()
        df_signals = self.strategy.generate_signals(data)
        # restrict by date range
        start_date = self.cfg.get("backtest.start_date", None)
        end_date = self.cfg.get("backtest.end_date", None)
        if start_date:
            df_signals = df_signals[df_signals.index >= pd.to_datetime(start_date)]
        if end_date:
            df_signals = df_signals[df_signals.index <= pd.to_datetime(end_date)]
        # iterate bars
        for idx, row in df_signals.iterrows():
            try:
                self._on_bar(idx, row)
                # mark to market with close price
                equity = self.broker.mark_to_market(idx, float(row["Close"]))
                # enforce max drawdown
                if self.risk_manager.check_drawdown(equity):
                    self.logger.warning("Maximum drawdown exceeded at %s equity=%.2f. Stopping backtest.", idx, equity)
                    break
            except Exception as e:
                self.logger.error("Error on bar %s: %s\n%s", idx, str(e), traceback.format_exc())
                # In production, may decide to stop or continue; here we continue
                continue
        # finalize: if position open, close at last close price
        if self.broker.open_position is not None:
            last_idx = df_signals.index[-1]
            last_price = float(df_signals.iloc[-1]["Close"])
            self.broker.close_position(last_idx, last_price)
            self.broker.mark_to_market(last_idx, last_price)
        equity_series = Metrics.equity_series(self.broker)
        perf = Metrics.compute_performance(equity_series)
        results = {
            "trade_log": pd.DataFrame(self.broker.trade_log),
            "equity_curve": equity_series,
            "performance": perf,
            "final_cash": self.broker.cash,
            "total_commissions": self.broker.total_commissions
        }
        self.logger.info("Backtest finished. Performance: %s", perf)
        return results

    def _on_bar(self, time: pd.Timestamp, row: pd.Series) -> None:
        # check if an open position hits stop-loss or take-profit using high/low
        pos = self.broker.open_position
        if pos is not None:
            # check stop-loss and tp intrabar
            high = float(row["High"])
            low = float(row["Low"])
            exited = False
            # For long
            if pos.side == 1:
                # stop loss: low <= stop_loss
                if pos.stop_loss is not None and low <= pos.stop_loss:
                    self.logger.debug("Stop-loss triggered for long at %s (stop=%.4f low=%.4f)", time, pos.stop_loss, low)
                    self.broker.close_position(time, pos.stop_loss)
                    exited = True
                # take profit
                elif pos.take_profit is not None and high >= pos.take_profit:
                    self.logger.debug("Take-profit triggered for long at %s (tp=%.4f high=%.4f)", time, pos.take_profit, high)
                    self.broker.close_position(time, pos.take_profit)
                    exited = True
            else:
                # short position
                if pos.stop_loss is not None and high >= pos.stop_loss:
                    self.logger.debug("Stop-loss triggered for short at %s (stop=%.4f high=%.4f)", time, pos.stop_loss, high)
                    self.broker.close_position(time, pos.stop_loss)
                    exited = True
                elif pos.take_profit is not None and low <= pos.take_profit:
                    self.logger.debug("Take-profit triggered for short at %s (tp=%.4f low=%.4f)", time, pos.take_profit, low)
                    self.broker.close_position(time, pos.take_profit)
                    exited = True
            if exited:
                return

        # Next, check for new signals
        signal = int(row.get("signal", 0))
        current_open_pos = self.broker.open_position
        if signal == 0:
            # If signal is flat, close any open position at today's open price
            if current_open_pos is not None:
                self.broker.close_position(time, float(row["Open"]))
            return

        # If there is an open position and signal matches side, do nothing (hold)
        if current_open_pos is not None and current_open_pos.side == signal:
            return

        # If there is a position opposite to the signal, close it first
        if current_open_pos is not None and current_open_pos.side != signal:
            self.broker.close_position(time, float(row["Open"]))

        # Evaluate entry: size based on risk manager
        price = float(row["Open"])  # entry at open price of bar
        rm = self.risk_manager
        # compute stop_loss and take_profit price
        if signal == 1:
            stop = price * (1.0 - rm.stop_loss_pct)
            tp = price * (1.0 + rm.take_profit_pct)
        else:
            stop = price * (1.0 + rm.stop_loss_pct)
            tp = price * (1.0 - rm.take_profit_pct)
        units = rm.compute_units(capital=self.broker.cash + (self.broker.open_position.entry_price * self.broker.open_position.units if self.broker.open_position else 0),
                                 price=price, side=signal, stop_loss_price=stop)
        # If computed units are zero, skip entry
        if units <= 0:
            self.logger.debug("Zero units computed for trade at %s price=%.4f. Skipping.", time, price)
            return
        # Execute market order
        self.broker.execute_market_order(time=time, price=price, units=units, side=signal, stop_loss=stop, take_profit=tp)


# -----------------------------
# Utilities: I/O and plotting
# -----------------------------
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def save_results(results: Dict[str, Any], cfg: Config, logger: logging.Logger):
    out_dir = cfg.get("backtest.save_results_to", "results")
    ensure_dir(out_dir)
    # Save trade log
    trade_log: pd.DataFrame = results.get("trade_log", pd.DataFrame())
    trade_log_path = os.path.join(out_dir, "trade_log.csv")
    trade_log.to_csv(trade_log_path, index=False)
    logger.info("Trade log saved to %s", trade_log_path)
    # Save equity curve
    equity: pd.Series = results.get("equity_curve", pd.Series(dtype=float))
    equity_path = os.path.join(out_dir, "equity_curve.csv")
    equity.to_csv(equity_path, header=["equity"])
    logger.info("Equity curve saved to %s", equity_path)
    # Save performance summary
    perf = results.get("performance", {})
    perf_path = os.path.join(out_dir, "performance.json")
    with open(perf_path, "w") as f:
        json.dump(perf, f, indent=2)
    logger.info("Performance summary saved to %s", perf_path)


def plot_results(results: Dict[str, Any], cfg: Config, logger: logging.Logger):
    if not _HAS_MPL:
        logger.warning("matplotlib not available. Skipping plots.")
        return
    equity = results.get("equity_curve", pd.Series(dtype=float))
    trades = results.get("trade_log", pd.DataFrame())
    if equity.empty:
        logger.warning("Empty equity curve. No plots generated.")
        return
    fig, ax = plt.subplots(figsize=(12, 6))
    equity.plot(ax=ax, title="Equity Curve")
    ax.set_ylabel("Equity")
    out_dir = cfg.get("backtest.save_results_to", "results")
    ensure_dir(out_dir)
    fig_path = os.path.join(out_dir, "equity_curve.png")
    fig.tight_layout()
    fig.savefig(fig_path)
    logger.info("Equity curve plot saved to %s", fig_path)
    plt.close(fig)


# -----------------------------
# Main entrypoint
# -----------------------------
def main(config_path: Optional[str] = None, data_path: Optional[str] = None):
    # Load config
    cfg = Config()
    if config_path:
        try:
            cfg.load(config_path)
        except Exception as e:
            print(f"Failed to load config at {config_path}: {e}")
            return
    logger = setup_logging(cfg)
    logger.info("Configuration: %s", json.dumps(cfg.as_dict(), indent=2, default=str))
    # Load data
    dl = DataLoader(cfg, logger)
    data_file = data_path or cfg.get("data.path", "data/ohlcv.csv")
    if not os.path.isfile(data_file):
        logger.error("Data file not found: %s", data_file)
        return
    try:
        df = dl.load_csv(data_file)
    except Exception as e:
        logger.error("Failed to load data: %s", e)
        return
    # Run backtest
    backtester = Backtester(cfg, logger)
    results = backtester.run(df)
    # Save and plot results
    save_results(results, cfg, logger)
    if cfg.get("backtest.plot_results", True):
        plot_results(results, cfg, logger)
    logger.info("Backtest complete. Final cash: %.2f Total commissions: %.2f", results.get("final_cash", 0.0),
                results.get("total_commissions", 0.0))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Backtest Moving Average Crossover Strategy.")
    parser.add_argument("--config", "-c", help="Path to JSON config file", default=None)
    parser.add_argument("--data", "-d", help="Path to CSV OHLCV data file", default=None)
    args = parser.parse_args()
    main(config_path=args.config, data_path=args.data)