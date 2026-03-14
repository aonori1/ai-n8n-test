#!/usr/bin/env python3
"""
Production-ready, modular algorithmic trading backtesting framework (single-file).
Features:
- Configuration loading (JSON/YAML) with defaults
- Rotating file logging and console logging
- CSV data feed for OHLCV data (pandas)
- Simple indicator utilities
- Strategy base class and example Moving Average Crossover strategy
- Risk manager with position sizing, max drawdown and daily loss limits
- Portfolio and Order/Fill dataclasses
- Execution handler simulating fills with slippage and commission
- Backtest engine producing performance metrics and trade logs
- Ready for backtesting and easily extensible for production/backtest pipelines

Dependencies:
- Python 3.8+
- pandas, numpy, pyyaml (optional for YAML config)
"""

from __future__ import annotations
import os
import sys
import json
import math
import logging
import logging.handlers
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Tuple, Iterable
from datetime import datetime, timedelta
from collections import deque
import pandas as pd
import numpy as np

try:
    import yaml
    _HAS_YAML = True
except Exception:
    _HAS_YAML = False

# -----------------
# Configuration
# -----------------
DEFAULT_CONFIG = {
    "backtest": {
        "start_cash": 100000.0,
        "start_date": None,
        "end_date": None,
        "symbol": "AAPL",
        "data_path": "data/AAPL.csv",
        "initial_margin": 0.0,
        "commission": 0.0,
        "slippage": 0.0,
        "timeframe": "1d"
    },
    "strategy": {
        "name": "MovingAverageCrossover",
        "params": {
            "fast_window": 20,
            "slow_window": 50,
            "size": None,
            "risk_per_trade": 0.01,
            "use_percent_of_equity": True
        }
    },
    "risk": {
        "max_drawdown": 0.2,
        "max_daily_loss": 0.05,
        "max_position_size": 0.2,
        "min_trade_size": 1
    },
    "logging": {
        "level": "INFO",
        "logfile": "backtest.log",
        "max_bytes": 10 * 1024 * 1024,
        "backup_count": 3
    },
    "report": {
        "save_equity_curve": True,
        "equity_curve_path": "equity_curve.csv",
        "trades_path": "trades.csv"
    }
}


class Config:
    def __init__(self, config_path: Optional[str] = None):
        self._config = DEFAULT_CONFIG.copy()
        if config_path:
            self.load(config_path)

    def load(self, path: str):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Config file not found: {path}")
        try:
            if path.lower().endswith((".yml", ".yaml")) and _HAS_YAML:
                with open(path, "r") as fh:
                    cfg = yaml.safe_load(fh)
            else:
                with open(path, "r") as fh:
                    cfg = json.load(fh)
            self._deep_update(self._config, cfg)
        except Exception as e:
            raise RuntimeError(f"Failed to load config {path}: {e}")

    @staticmethod
    def _deep_update(d: dict, u: dict):
        for k, v in u.items():
            if isinstance(v, dict):
                d[k] = Config._deep_update(d.get(k, {}), v)
            else:
                d[k] = v
        return d

    def get(self, *keys, default=None):
        node = self._config
        for key in keys:
            if isinstance(node, dict) and key in node:
                node = node[key]
            else:
                return default
        return node

    def as_dict(self):
        return self._config

# -----------------
# Logging
# -----------------
def setup_logging(config: Config) -> logging.Logger:
    log_cfg = config.get("logging", default={})
    level_name = log_cfg.get("level", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    logfile = log_cfg.get("logfile", "backtest.log")
    max_bytes = int(log_cfg.get("max_bytes", 10 * 1024 * 1024))
    backup_count = int(log_cfg.get("backup_count", 3))

    logger = logging.getLogger("backtester")
    logger.setLevel(level)
    if not logger.handlers:
        fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s - %(message)s")
        # Console handler
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(level)
        ch.setFormatter(fmt)
        logger.addHandler(ch)
        # Rotating file handler
        fh = logging.handlers.RotatingFileHandler(logfile, maxBytes=max_bytes, backupCount=backup_count)
        fh.setLevel(level)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger

# -----------------
# Data Feed
# -----------------
class DataFeed:
    """
    Simple CSV data feed. Expects a CSV with columns: datetime (or Date), open, high, low, close, volume.
    """
    def __init__(self, data_path: str, start_date: Optional[str] = None, end_date: Optional[str] = None, logger: Optional[logging.Logger] = None):
        self.data_path = data_path
        self.start_date = start_date
        self.end_date = end_date
        self.logger = logger or logging.getLogger("backtester")
        self.df = self._load()
        self._validate_df()

    def _load(self) -> pd.DataFrame:
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data path not found: {self.data_path}")
        df = pd.read_csv(self.data_path, parse_dates=True, infer_datetime_format=True)
        # Try common column names
        datetime_cols = [c for c in df.columns if c.lower() in ("datetime", "date", "timestamp")]
        if datetime_cols:
            df.rename(columns={datetime_cols[0]: "datetime"}, inplace=True)
        else:
            # if index is datetime
            if not np.issubdtype(df.index.dtype, np.datetime64):
                raise ValueError("CSV must contain a datetime column named Date/Datetime/Timestamp or have a datetime index")

            df.reset_index(inplace=True)
            df.rename(columns={df.columns[0]: "datetime"}, inplace=True)
        # Standardize column names
        col_map = {}
        for c in df.columns:
            lc = c.lower()
            if lc in ("open", "high", "low", "close", "volume", "datetime"):
                col_map[c] = lc
        df = df.rename(columns=col_map)
        if "datetime" not in df.columns or "close" not in df.columns:
            raise ValueError("CSV must include datetime and close columns")
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.sort_values("datetime").set_index("datetime")
        # slice by date range
        if self.start_date:
            df = df[df.index >= pd.to_datetime(self.start_date)]
        if self.end_date:
            df = df[df.index <= pd.to_datetime(self.end_date)]
        return df

    def _validate_df(self):
        required = ["open", "high", "low", "close"]
        missing = [c for c in required if c not in self.df.columns]
        if missing:
            self.logger.warning(f"Missing columns in data feed: {missing}. Some features may not be available.")
        # Fill missing volume
        if "volume" not in self.df.columns:
            self.df["volume"] = 0.0

    def get_series(self) -> pd.DataFrame:
        return self.df.copy()

    def iter_bars(self) -> Iterable[Tuple[pd.Timestamp, Dict[str, float]]]:
        for idx, row in self.df.iterrows():
            bar = {
                "open": float(row.get("open", np.nan)),
                "high": float(row.get("high", np.nan)),
                "low": float(row.get("low", np.nan)),
                "close": float(row.get("close", np.nan)),
                "volume": float(row.get("volume", 0.0))
            }
            yield idx, bar

# -----------------
# Utilities / Indicators
# -----------------
def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=1).mean()

def ema(series: pd.Series, window: int) -> pd.Series:
    return series.ewm(span=window, adjust=False).mean()

# -----------------
# Orders, Fills, Positions
# -----------------
@dataclass
class Order:
    symbol: str
    quantity: int
    side: str  # 'buy' or 'sell'
    price: Optional[float] = None
    order_type: str = "market"
    timestamp: Optional[pd.Timestamp] = None
    id: Optional[int] = None

@dataclass
class Fill:
    order_id: int
    symbol: str
    quantity: int
    side: str
    price: float
    commission: float
    timestamp: pd.Timestamp

@dataclass
class Position:
    symbol: str
    quantity: int = 0
    avg_price: float = 0.0

    def update(self, quantity: int, trade_price: float):
        if self.quantity == 0:
            self.quantity = quantity
            self.avg_price = trade_price
            return
        # if same direction
        if (self.quantity >= 0 and quantity >= 0) or (self.quantity <= 0 and quantity <= 0):
            new_qty = self.quantity + quantity
            if new_qty == 0:
                self.quantity = 0
                self.avg_price = 0.0
            else:
                # weighted average price
                self.avg_price = (self.avg_price * abs(self.quantity) + trade_price * abs(quantity)) / (abs(new_qty))
                self.quantity = new_qty
        else:
            # reducing or flipping position
            new_qty = self.quantity + quantity
            if (self.quantity > 0 and new_qty >= 0) or (self.quantity < 0 and new_qty <= 0):
                self.quantity = new_qty
            else:
                # flip across zero
                self.quantity = new_qty
                self.avg_price = trade_price

# -----------------
# Strategy
# -----------------
class Strategy:
    """
    Base class for strategies. Should be subclassed and implement on_bar.
    """
    def __init__(self, symbol: str, config: Config, logger: Optional[logging.Logger] = None):
        self.symbol = symbol
        self.config = config
        self.logger = logger or logging.getLogger("backtester.strategy")
        self.latest_bar: Optional[pd.Series] = None
        self.bars = pd.DataFrame()

    def on_bar(self, timestamp: pd.Timestamp, bar: Dict[str, float]) -> List[Order]:
        raise NotImplementedError()

    def ingest_bar(self, timestamp: pd.Timestamp, bar: Dict[str, float]):
        row = pd.DataFrame([bar], index=[timestamp])
        self.bars = pd.concat([self.bars, row])
        # keep memory bounded? For backtesting we assume full history is used for indicators.

# Example strategy: Moving Average Crossover
class MovingAverageCrossoverStrategy(Strategy):
    def __init__(self, symbol: str, config: Config, logger: Optional[logging.Logger] = None):
        super().__init__(symbol, config, logger)
        params = config.get("strategy", "params", default={})
        self.fast = int(params.get("fast_window", 20))
        self.slow = int(params.get("slow_window", 50))
        self.fixed_size = params.get("size", None)
        self.risk_per_trade = float(params.get("risk_per_trade", 0.01))
        self.use_percent_of_equity = bool(params.get("use_percent_of_equity", True))
        self.logger.debug(f"Initialized MAC strategy fast={self.fast} slow={self.slow} size={self.fixed_size}")

    def on_bar(self, timestamp: pd.Timestamp, bar: Dict[str, float]) -> List[Order]:
        self.ingest_bar(timestamp, bar)
        orders: List[Order] = []
        close = self.bars["close"]
        if len(close) < 2:
            return orders
        fast_ma = sma(close, self.fast).iloc[-1]
        slow_ma = sma(close, self.slow).iloc[-1]
        prev_fast = sma(close, self.fast).iloc[-2] if len(close) >= 2 else None
        prev_slow = sma(close, self.slow).iloc[-2] if len(close) >= 2 else None
        # Determine crossovers
        if prev_fast is None or prev_slow is None:
            return orders
        # Bullish crossover: fast crosses above slow -> go long
        if prev_fast <= prev_slow and fast_ma > slow_ma:
            qty = None
            if self.fixed_size is not None:
                qty = int(self.fixed_size)
            orders.append(Order(symbol=self.symbol, quantity=qty or 0, side="buy", order_type="market", timestamp=timestamp))
        # Bearish crossover: fast crosses below slow -> go short or close long
        elif prev_fast >= prev_slow and fast_ma < slow_ma:
            qty = None
            if self.fixed_size is not None:
                qty = int(self.fixed_size)
            orders.append(Order(symbol=self.symbol, quantity=qty or 0, side="sell", order_type="market", timestamp=timestamp))
        return orders

# -----------------
# Risk Management
# -----------------
class RiskManager:
    def __init__(self, config: Config, logger: Optional[logging.Logger] = None):
        self.config = config
        self.logger = logger or logging.getLogger("backtester.risk")
        self.max_drawdown = float(config.get("risk", "max_drawdown", default=DEFAULT_CONFIG["risk"]["max_drawdown"]))
        self.max_daily_loss = float(config.get("risk", "max_daily_loss", default=DEFAULT_CONFIG["risk"]["max_daily_loss"]))
        self.max_position_size = float(config.get("risk", "max_position_size", default=DEFAULT_CONFIG["risk"]["max_position_size"]))
        self.min_trade_size = int(config.get("risk", "min_trade_size", default=DEFAULT_CONFIG["risk"]["min_trade_size"]))

    def assess_order(self, order: Order, portfolio: "Portfolio", price: float) -> Optional[Order]:
        """
        Applies risk rules: position sizing and caps.
        Returns an adjusted Order or None if the order should be rejected.
        """

        # Determine intended sign
        sign = 1 if order.side == "buy" else -1
        equity = portfolio.equity
        # Compute max position size in USD
        max_pos_usd = equity * self.max_position_size
        # Determine desired absolute dollar exposure for the order
        # If strategy provided a fixed quantity in order.quantity > 0, use that as baseline
        if order.quantity and order.quantity > 0:
            desired_dollar = order.quantity * price
            desired_qty = int(order.quantity)
        else:
            # Use risk_per_trade if available in strategy params
            strat_params = self.config.get("strategy", "params", default={})
            risk_per_trade = float(strat_params.get("risk_per_trade", 0.01))
            # We don't know stop loss distance; use conservative approach: risk_per_trade * equity / price => qty
            desired_dollar = equity * risk_per_trade
            desired_qty = int(max(self.min_trade_size, desired_dollar // price)) if price > 0 else 0

        # Cap by max position size
        if desired_qty * price > max_pos_usd:
            desired_qty = int(max_pos_usd // price)

        if desired_qty < self.min_trade_size:
            self.logger.debug(f"Desired quantity {desired_qty} below min_trade_size {self.min_trade_size}; rejecting order.")
            return None

        # Check margin and purchasing power
        if sign > 0:
            # buying: ensure enough cash
            if portfolio.cash < desired_qty * price:
                # reduce size if possible
                affordable_qty = int(portfolio.cash // price)
                if affordable_qty < self.min_trade_size:
                    self.logger.debug("Not enough cash for minimum trade size; rejecting buy order.")
                    return None
                desired_qty = affordable_qty
        else:
            # selling: check position to sell (allow shorting if allowed? For simplicity, disallow shorting unless position exists)
            current_pos = portfolio.positions.get(order.symbol)
            if current_pos is None or current_pos.quantity <= 0:
                self.logger.debug("No long position to sell; rejecting sell order (shorting disabled).")
                return None
            if desired_qty > current_pos.quantity:
                desired_qty = current_pos.quantity

        # Set the order to adjusted quantity
        adjusted_order = Order(symbol=order.symbol, quantity=desired_qty, side=order.side, order_type=order.order_type, timestamp=order.timestamp, price=order.price, id=order.id)
        return adjusted_order

    def check_drawdown(self, portfolio: "Portfolio") -> bool:
        """
        Returns True if a drawdown breach occurs (i.e., should stop trading).
        """
        peak = portfolio.equity_highwater
        if peak <= 0:
            return False
        drawdown = (peak - portfolio.equity) / peak
        if drawdown >= self.max_drawdown:
            self.logger.warning(f"Max drawdown breached: {drawdown:.2%} >= {self.max_drawdown:.2%}")
            return True
        return False

    def check_daily_loss(self, portfolio: "Portfolio", daily_return: float) -> bool:
        """
        daily_return is negative for loss. Return True if daily loss > max_daily_loss
        """
        if daily_return < 0 and abs(daily_return) >= self.max_daily_loss:
            self.logger.warning(f"Daily loss limit breached: {daily_return:.2%} <= -{self.max_daily_loss:.2%}")
            return True
        return False

# -----------------
# Execution Handler
# -----------------
class ExecutionHandler:
    def __init__(self, commission: float = 0.0, slippage: float = 0.0, logger: Optional[logging.Logger] = None):
        self.commission = commission
        self.slippage = slippage
        self.logger = logger or logging.getLogger("backtester.execution")
        self._order_id_seq = 1

    def create_order_id(self) -> int:
        oid = self._order_id_seq
        self._order_id_seq += 1
        return oid

    def execute_order(self, order: Order, bar: Dict[str, float], timestamp: pd.Timestamp) -> Optional[Fill]:
        """
        Simulates market execution. For market orders, execute at bar['open'] if available, else bar['close'].
        Apply slippage and commission. Returns Fill.
        """
        if order.quantity <= 0:
            self.logger.debug(f"Order quantity <= 0, skipping execution: {order}")
            return None
        execution_price = bar.get("open", None) or bar.get("close", None)
        if execution_price is None or np.isnan(execution_price):
            self.logger.error("Execution price not available in bar")
            return None
        # Apply slippage as fraction of price (e.g., 0.001 = 0.1%)
        slippage_amount = execution_price * self.slippage
        if order.side == "buy":
            exec_price = execution_price + slippage_amount
        else:
            exec_price = execution_price - slippage_amount
        commission = self.commission * order.quantity  # simple per-share commission
        fill = Fill(order_id=order.id or self.create_order_id(),
                    symbol=order.symbol,
                    quantity=order.quantity,
                    side=order.side,
                    price=exec_price,
                    commission=commission,
                    timestamp=timestamp)
        self.logger.debug(f"Executed order {order} -> fill {fill}")
        return fill

# -----------------
# Portfolio
# -----------------
class Portfolio:
    def __init__(self, start_cash: float, symbol: str, commission: float = 0.0, logger: Optional[logging.Logger] = None):
        self.start_cash = float(start_cash)
        self.cash = float(start_cash)
        self.symbol = symbol
        self.positions: Dict[str, Position] = {}
        self.holdings: float = 0.0
        self.equity: float = float(start_cash)
        self.equity_highwater: float = float(start_cash)
        self.trade_history: List[Dict[str, Any]] = []
        self.equity_curve: List[Dict[str, Any]] = []
        self.commission = commission
        self.logger = logger or logging.getLogger("backtester.portfolio")
        self.last_date: Optional[pd.Timestamp] = None

    def update_with_fill(self, fill: Fill):
        # Adjust cash and positions
        qty = fill.quantity if fill.side == "buy" else -fill.quantity
        cost = qty * fill.price
        total_commission = fill.commission
        self.cash -= cost + total_commission
        pos = self.positions.get(fill.symbol)
        if pos is None:
            pos = Position(symbol=fill.symbol, quantity=0, avg_price=0.0)
            self.positions[fill.symbol] = pos
        pos.update(qty, fill.price)
        # record trade
        trade_record = {
            "timestamp": fill.timestamp,
            "symbol": fill.symbol,
            "quantity": fill.quantity if fill.side == "buy" else -fill.quantity,
            "price": fill.price,
            "commission": total_commission,
            "trade_value": fill.quantity * fill.price,
            "side": fill.side,
            "order_id": fill.order_id
        }
        self.trade_history.append(trade_record)
        self.logger.info(f"Recorded trade: {trade_record}")
        self.recompute_holdings(fill.timestamp)

    def recompute_holdings(self, timestamp: pd.Timestamp, market_price: Optional[float] = None):
        # For our single-symbol case, compute holdings based on position and last known price
        pos = self.positions.get(self.symbol)
        if pos and pos.quantity != 0:
            price = market_price if market_price is not None else pos.avg_price
            self.holdings = pos.quantity * price
        else:
            self.holdings = 0.0
        self.equity = self.cash + self.holdings
        if self.equity > self.equity_highwater:
            self.equity_highwater = self.equity
        # Append equity curve
        self.equity_curve.append({
            "timestamp": timestamp,
            "cash": self.cash,
            "holdings": self.holdings,
            "equity": self.equity
        })
        self.last_date = timestamp

    def get_position(self, symbol: str) -> int:
        pos = self.positions.get(symbol)
        return pos.quantity if pos else 0

    def compute_daily_return(self) -> float:
        if len(self.equity_curve) < 2:
            return 0.0
        last = self.equity_curve[-1]["equity"]
        prev = self.equity_curve[-2]["equity"]
        if prev == 0:
            return 0.0
        return (last - prev) / prev

# -----------------
# Performance Metrics
# -----------------
def compute_performance(equity_curve: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not equity_curve:
        return {}
    df = pd.DataFrame(equity_curve).set_index("timestamp")
    df.index = pd.to_datetime(df.index)
    eq = df["equity"].astype(float)
    returns = eq.pct_change().fillna(0.0)
    total_return = float(eq.iloc[-1] / eq.iloc[0] - 1.0)
    days = (eq.index[-1] - eq.index[0]).days or 1
    # Annualize returns (assuming 252 trading days)
    annual_factor = 252.0
    cumulative_return = total_return
    annual_return = ((1.0 + cumulative_return) ** (annual_factor / max(days, 1))) - 1.0
    vol = returns.std() * math.sqrt(annual_factor)
    sharpe = (returns.mean() * annual_factor) / vol if vol != 0 else np.nan
    # Drawdown
    roll_max = eq.cummax()
    drawdown = (roll_max - eq) / roll_max
    max_drawdown = drawdown.max()
    # Basic metrics
    metrics = {
        "total_return": total_return,
        "annual_return": annual_return,
        "volatility": vol,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "final_equity": float(eq.iloc[-1])
    }
    return metrics

# -----------------
# Backtest Engine
# -----------------
class BacktestEngine:
    def __init__(self, config: Config, logger: Optional[logging.Logger] = None):
        self.config = config
        self.logger = logger or logging.getLogger("backtester")
        bt_cfg = config.get("backtest", default={})
        self.data_path = bt_cfg.get("data_path")
        self.symbol = bt_cfg.get("symbol", DEFAULT_CONFIG["backtest"]["symbol"])
        self.start_cash = float(bt_cfg.get("start_cash", DEFAULT_CONFIG["backtest"]["start_cash"]))
        self.commission = float(bt_cfg.get("commission", DEFAULT_CONFIG["backtest"]["commission"]))
        self.slippage = float(bt_cfg.get("slippage", DEFAULT_CONFIG["backtest"]["slippage"]))
        self.start_date = bt_cfg.get("start_date", None)
        self.end_date = bt_cfg.get("end_date", None)

        self.data_feed = DataFeed(data_path=self.data_path, start_date=self.start_date, end_date=self.end_date, logger=self.logger)
        self.execution = ExecutionHandler(commission=self.commission, slippage=self.slippage, logger=self.logger)
        self.portfolio = Portfolio(start_cash=self.start_cash, symbol=self.symbol, commission=self.commission, logger=self.logger)
        self.risk = RiskManager(config, logger=self.logger)
        self.strategy = self._init_strategy()
        self._orders_queue: deque[Order] = deque()
        self.trading_halted = False

    def _init_strategy(self) -> Strategy:
        strat_name = self.config.get("strategy", "name", default="MovingAverageCrossover")
        strat_map = {
            "MovingAverageCrossover": MovingAverageCrossoverStrategy
            # Add other strategies here
        }
        strat_cls = strat_map.get(strat_name)
        if not strat_cls:
            raise ValueError(f"Unknown strategy: {strat_name}")
        strat = strat_cls(symbol=self.symbol, config=self.config, logger=logging.getLogger(f"strategy.{strat_name}"))
        return strat

    def run(self):
        self.logger.info("Starting backtest")
        df = self.data_feed.get_series()
        prev_date = None

        for timestamp, bar in self.data_feed.iter_bars():
            if self.trading_halted:
                self.logger.info("Trading halted due to risk limits.")
                break
            # 1. Strategy processes bar and produces orders
            try:
                orders = self.strategy.on_bar(timestamp, bar)
            except Exception as e:
                self.logger.exception(f"Strategy error at {timestamp}: {e}")
                orders = []
            # Assign order ids
            for o in orders:
                o.id = self.execution.create_order_id()
            # 2. Risk manager assesses orders and may modify or reject
            approved_orders: List[Order] = []
            for o in orders:
                # Determine a reference price: use bar['open'] or 'close'
                ref_price = bar.get("open") or bar.get("close")
                assessed = self.risk.assess_order(o, self.portfolio, ref_price)
                if assessed:
                    approved_orders.append(assessed)
                else:
                    self.logger.debug(f"Order rejected by risk manager: {o}")
            # 3. Execute approved orders
            for o in approved_orders:
                fill = self.execution.execute_order(o, bar, timestamp)
                if fill:
                    # update portfolio
                    self.portfolio.update_with_fill(fill)
            # 4. End of bar updates
            # update market price for holdings valuation
            last_price = bar.get("close")
            self.portfolio.recompute_holdings(timestamp, market_price=last_price)
            # Check daily loss
            daily_ret = self.portfolio.compute_daily_return()
            if self.risk.check_daily_loss(self.portfolio, daily_ret):
                self.trading_halted = True
                self.logger.info("Halting trading due to daily loss limit.")
                break
            # Check drawdown
            if self.risk.check_drawdown(self.portfolio):
                self.trading_halted = True
                self.logger.info("Halting trading due to drawdown.")
                break
            prev_date = timestamp
        # Done
        perf = compute_performance(self.portfolio.equity_curve)
        self.logger.info(f"Backtest completed. Performance: {perf}")
        # Optionally save reports
        rep_cfg = self.config.get("report", default={})
        if rep_cfg.get("save_equity_curve", True):
            path = rep_cfg.get("equity_curve_path", "equity_curve.csv")
            self._save_equity_curve(path)
        trades_path = rep_cfg.get("trades_path", "trades.csv")
        self._save_trades(trades_path)
        return {
            "performance": perf,
            "equity_curve": self.portfolio.equity_curve,
            "trades": self.portfolio.trade_history
        }

    def _save_equity_curve(self, path: str):
        try:
            df = pd.DataFrame(self.portfolio.equity_curve)
            df.to_csv(path, index=False)
            self.logger.info(f"Saved equity curve to {path}")
        except Exception as e:
            self.logger.exception(f"Failed to save equity curve: {e}")

    def _save_trades(self, path: str):
        try:
            df = pd.DataFrame(self.portfolio.trade_history)
            if df.empty:
                self.logger.info("No trades to save.")
                return
            df.to_csv(path, index=False)
            self.logger.info(f"Saved trades to {path}")
        except Exception as e:
            self.logger.exception(f"Failed to save trades: {e}")

# -----------------
# Example main / usage
# -----------------
def main(config_path: Optional[str] = None):
    try:
        cfg = Config(config_path) if config_path else Config()
    except Exception as e:
        print(f"Failed to load config: {e}", file=sys.stderr)
        cfg = Config()
    logger = setup_logging(cfg)
    logger.info("Configuration loaded.")
    # validate data path
    data_path = cfg.get("backtest", "data_path", default=DEFAULT_CONFIG["backtest"]["data_path"])
    if not os.path.exists(data_path):
        logger.error(f"Data file not found: {data_path}")
        # Optionally provide guidance
        raise FileNotFoundError(f"Data file not found: {data_path}")

    engine = BacktestEngine(cfg, logger=logger)
    results = engine.run()
    # Print summary
    perf = results.get("performance", {})
    logger.info("Summary:")
    for k, v in perf.items():
        logger.info(f"{k}: {v}")
    return results

if __name__ == "__main__":
    # Accept optional config path as first arg
    config_path = None
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    main(config_path)