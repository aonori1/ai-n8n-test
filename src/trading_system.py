# algo_trading_framework.py
"""
Production-ready algorithmic trading framework (single-file).
Features:
- Modular structure (Config, LoggerFactory, DataHandler, Strategy, RiskManager, ExecutionSimulator, Portfolio, Backtester, Metrics)
- Configuration via dict or YAML (if pyyaml installed)
- Logging (file + console, rotating file handler)
- Risk management (position sizing, max drawdown, per-trade risk, leverage, max exposure)
- Ready for backtesting on OHLCV pandas DataFrame
- Example moving-average crossover strategy included
- Minimal external dependencies: pandas, numpy
"""

from __future__ import annotations
import os
import sys
import math
import json
import logging
from logging.handlers import RotatingFileHandler
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, Tuple, List
from datetime import datetime
import numpy as np
import pandas as pd

# -----------------------
# Configuration handling
# -----------------------
DEFAULT_CONFIG: Dict[str, Any] = {
    "logging": {
        "level": "INFO",
        "file": "backtest.log",
        "max_bytes": 10_000_000,
        "backup_count": 3
    },
    "backtest": {
        "start_cash": 100000.0,
        "commission_per_trade": 1.0,      # flat commission per order
        "commission_pct": 0.0,           # percent commission of trade value
        "slippage_pct": 0.0005,          # slippage percent (0.05%)
        "tick_size": 0.0,                # minimum price increment
        "leverage": 1.0,
        "max_position_pct": 0.10,        # max percent of portfolio per position
        "per_trade_risk_pct": 0.01,      # percent of equity risked per trade (for stop-loss based sizing)
        "max_drawdown_pct": 0.30,
        "min_trade_size": 1,             # minimum number of units (shares)
    },
    "strategy": {
        "fast_ma": 10,
        "slow_ma": 50,
        "stop_loss_pct": 0.02,
        "take_profit_pct": 0.05,
    }
}


class Config:
    """
    Configuration loader. Can load from YAML file if PyYAML is available.
    Falls back to DEFAULT_CONFIG.
    """
    def __init__(self, config_path: Optional[str] = None):
        self._cfg = DEFAULT_CONFIG.copy()
        if config_path:
            self.load_from_file(config_path)

    def load_from_file(self, path: str):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Config file not found: {path}")
        try:
            import yaml  # type: ignore
            with open(path, "r") as f:
                parsed = yaml.safe_load(f)
                self._deep_update(self._cfg, parsed or {})
        except Exception:
            # Try JSON
            with open(path, "r") as f:
                parsed = json.load(f)
                self._deep_update(self._cfg, parsed or {})

    @staticmethod
    def _deep_update(d: Dict[str, Any], u: Dict[str, Any]):
        for k, v in u.items():
            if isinstance(v, dict):
                d[k] = Config._deep_update(d.get(k, {}), v)
            else:
                d[k] = v
        return d

    def get(self, key: str, default: Any = None) -> Any:
        # dot-separated key
        parts = key.split(".")
        cur = self._cfg
        for p in parts:
            if isinstance(cur, dict) and p in cur:
                cur = cur[p]
            else:
                return default
        return cur

    def as_dict(self) -> Dict[str, Any]:
        return self._cfg.copy()


# -----------------------
# Logging
# -----------------------
class LoggerFactory:
    @staticmethod
    def create_logger(name: str, cfg: Config) -> logging.Logger:
        log_cfg = cfg.get("logging", {})
        level_name = log_cfg.get("level", "INFO")
        level = getattr(logging, level_name.upper(), logging.INFO)
        logger = logging.getLogger(name)
        logger.setLevel(level)
        if not logger.handlers:
            # Console handler
            ch = logging.StreamHandler(sys.stdout)
            ch.setLevel(level)
            fmt = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            ch.setFormatter(fmt)
            logger.addHandler(ch)
            # Rotating file handler
            log_file = log_cfg.get("file", "backtest.log")
            max_bytes = log_cfg.get("max_bytes", 10_000_000)
            backup_count = log_cfg.get("backup_count", 3)
            fh = RotatingFileHandler(log_file, maxBytes=max_bytes, backupCount=backup_count)
            fh.setLevel(level)
            fh.setFormatter(fmt)
            logger.addHandler(fh)
        return logger


# -----------------------
# Data Handler
# -----------------------
class DataHandler:
    """
    Expects DataFrame with index as datetime and columns: open, high, low, close, volume
    """
    def __init__(self, df: pd.DataFrame):
        required = {"open", "high", "low", "close", "volume"}
        if not required.issubset(set(df.columns)):
            raise ValueError(f"DataFrame must contain columns: {required}")
        if not isinstance(df.index, pd.DatetimeIndex):
            df = df.copy()
            df.index = pd.to_datetime(df.index)
        self.df = df.sort_index()

    def get_ohlcv(self) -> pd.DataFrame:
        return self.df.copy()

    def slice(self, start: Optional[pd.Timestamp] = None, end: Optional[pd.Timestamp] = None) -> DataHandler:
        df = self.df.copy()
        if start:
            df = df[df.index >= start]
        if end:
            df = df[df.index <= end]
        return DataHandler(df)


# -----------------------
# Strategy Base & Example
# -----------------------
class Strategy:
    """
    Base class. Implement generate_signals method.
    Signals DataFrame: same index, column 'signal' with 1 for long entry, -1 for short entry (optional), 0 otherwise.
    """
    def __init__(self, cfg: Config, logger: logging.Logger):
        self.cfg = cfg
        self.logger = logger

    def prepare(self, historical: pd.DataFrame) -> pd.DataFrame:
        """
        Optional preprocessing. Return DataFrame with indicators.
        """
        return historical

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Given data (OHLCV + indicators), return Series of signals (1, 0, -1).
        """
        raise NotImplementedError


class MovingAverageCrossStrategy(Strategy):
    def __init__(self, cfg: Config, logger: logging.Logger):
        super().__init__(cfg, logger)
        self.fast = int(cfg.get("strategy.fast_ma", DEFAULT_CONFIG["strategy"]["fast_ma"]))
        self.slow = int(cfg.get("strategy.slow_ma", DEFAULT_CONFIG["strategy"]["slow_ma"]))
        if self.fast >= self.slow:
            raise ValueError("fast_ma should be less than slow_ma")

    def prepare(self, historical: pd.DataFrame) -> pd.DataFrame:
        df = historical.copy()
        df["ma_fast"] = df["close"].rolling(self.fast, min_periods=1).mean()
        df["ma_slow"] = df["close"].rolling(self.slow, min_periods=1).mean()
        return df

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        df = data
        sig = pd.Series(0, index=df.index)
        prev_cross = (df["ma_fast"] > df["ma_slow"]).shift(1).fillna(False)
        curr_cross = (df["ma_fast"] > df["ma_slow"])
        # Entry when cross from false->true (long). Exit when true->false.
        sig[(~prev_cross) & (curr_cross)] = 1
        sig[(prev_cross) & (~curr_cross)] = 0  # explicit exit signal represented by 0; backtester handles exits
        return sig


# -----------------------
# Risk Management
# -----------------------
@dataclass
class RiskManager:
    cfg: Config
    logger: logging.Logger

    def compute_position_size(self, equity: float, price: float, stop_loss_price: float) -> int:
        """
        Compute position size (number of units) given per_trade_risk_pct and stop loss price.
        Size will also be capped by max_position_pct of equity and min_trade_size.
        """
        per_trade_risk_pct = float(self.cfg.get("backtest.per_trade_risk_pct", DEFAULT_CONFIG["backtest"]["per_trade_risk_pct"]))
        max_pos_pct = float(self.cfg.get("backtest.max_position_pct", DEFAULT_CONFIG["backtest"]["max_position_pct"]))
        min_trade_size = int(self.cfg.get("backtest.min_trade_size", DEFAULT_CONFIG["backtest"]["min_trade_size"]))
        leverage = float(self.cfg.get("backtest.leverage", 1.0))

        # Risk per share
        risk_per_share = abs(price - stop_loss_price)
        if risk_per_share <= 0:
            self.logger.debug("Risk per share <= 0, cannot compute position size.")
            return 0
        dollar_risk = equity * per_trade_risk_pct
        raw_size = dollar_risk / risk_per_share
        # Cap by max position value
        max_position_value = equity * max_pos_pct * leverage
        cap_size = max_position_value / price if price > 0 else 0
        size = int(max(min(raw_size, cap_size), 0))
        if size < min_trade_size:
            return 0
        return size


# -----------------------
# Execution Simulator
# -----------------------
@dataclass
class ExecutionSimulator:
    cfg: Config
    logger: logging.Logger

    def execute_order(self, timestamp: pd.Timestamp, side: int, size: int, price: float) -> Dict[str, Any]:
        """
        Simulate execution, applying slippage and commission.
        side: +1 buy, -1 sell
        returns dict: executed_price, size, commission, slippage_cost, value
        """
        if size <= 0:
            return {"executed_price": 0.0, "size": 0, "commission": 0.0, "slippage_cost": 0.0, "value": 0.0}
        slippage_pct = float(self.cfg.get("backtest.slippage_pct", DEFAULT_CONFIG["backtest"]["slippage_pct"]))
        commission_flat = float(self.cfg.get("backtest.commission_per_trade", DEFAULT_CONFIG["backtest"]["commission_per_trade"]))
        commission_pct = float(self.cfg.get("backtest.commission_pct", DEFAULT_CONFIG["backtest"]["commission_pct"]))
        tick = float(self.cfg.get("backtest.tick_size", DEFAULT_CONFIG["backtest"]["tick_size"]))

        # Apply slippage: move price against order by slippage_pct
        slippage = price * slippage_pct * (1 if side > 0 else -1)
        executed_price = price + slippage
        # Round to tick size
        if tick > 0:
            executed_price = round(executed_price / tick) * tick
        trade_value = executed_price * size * abs(side)
        commission = commission_flat + commission_pct * trade_value
        slippage_cost = abs(slippage) * size
        return {
            "executed_price": float(executed_price),
            "size": int(size),
            "commission": float(commission),
            "slippage_cost": float(slippage_cost),
            "value": float(trade_value)
        }


# -----------------------
# Portfolio
# -----------------------
@dataclass
class Position:
    size: int = 0
    avg_price: float = 0.0
    side: int = 0  # 1 long, -1 short, 0 none

    def value(self, price: float) -> float:
        return self.size * price * self.side


@dataclass
class Portfolio:
    cfg: Config
    logger: logging.Logger
    cash: float = 0.0
    positions: Dict[str, Position] = field(default_factory=dict)  # symbol -> Position
    equity_history: List[Tuple[pd.Timestamp, float]] = field(default_factory=list)
    trades: List[Dict[str, Any]] = field(default_factory=list)
    initial_cash: float = 0.0

    def __post_init__(self):
        self.initial_cash = float(self.cfg.get("backtest.start_cash", DEFAULT_CONFIG["backtest"]["start_cash"]))
        if self.cash == 0.0:
            self.cash = self.initial_cash

    def update_market(self, timestamp: pd.Timestamp, price_map: Dict[str, float]):
        """
        Update equity history using provided current prices for symbols.
        """
        total_value = self.cash
        for sym, pos in self.positions.items():
            price = price_map.get(sym, 0.0)
            total_value += pos.value(price)
        self.equity_history.append((timestamp, total_value))
        self.logger.debug(f"Portfolio updated: {timestamp} equity={total_value:.2f}")

    def place_trade(self, timestamp: pd.Timestamp, symbol: str, side: int, size: int, executed_price: float, commission: float, slippage_cost: float):
        """
        Apply trade to portfolio.
        side: +1 buy, -1 sell
        """
        if size <= 0:
            return
        pos = self.positions.get(symbol, Position())
        trade_value = executed_price * size * side
        # Update average price for position
        if pos.side == 0 or pos.size == 0:
            # Opening new position
            pos.side = side
            pos.size = size
            pos.avg_price = executed_price
        elif pos.side == side:
            # Adding to position (weighted average)
            new_size = pos.size + size
            pos.avg_price = (pos.avg_price * pos.size + executed_price * size) / new_size
            pos.size = new_size
        else:
            # Reducing or flipping position
            if size < pos.size:
                # partial close
                pos.size = pos.size - size
                # avg_price unchanged
            elif size == pos.size:
                # fully closed
                pos.size = 0
                pos.side = 0
                pos.avg_price = 0.0
            else:
                # flip side: close existing and open new remainder
                remainder = size - pos.size
                pos.side = side
                pos.size = remainder
                pos.avg_price = executed_price
        self.positions[symbol] = pos
        # adjust cash
        self.cash -= trade_value  # buys reduce cash (side positive), sells increase cash
        self.cash -= commission
        self.cash -= slippage_cost
        # record trade
        self.trades.append({
            "timestamp": timestamp,
            "symbol": symbol,
            "side": side,
            "size": size,
            "executed_price": executed_price,
            "commission": commission,
            "slippage_cost": slippage_cost,
            "cash_after": self.cash
        })
        self.logger.info(f"Trade executed: {timestamp} {symbol} side={side} size={size} price={executed_price:.2f} cash={self.cash:.2f}")

    def current_equity(self, price_map: Dict[str, float]) -> float:
        total = self.cash
        for sym, pos in self.positions.items():
            total += pos.value(price_map.get(sym, 0.0))
        return total

    def get_unrealized_pnl(self, price_map: Dict[str, float]) -> float:
        pnl = 0.0
        for sym, pos in self.positions.items():
            price = price_map.get(sym, 0.0)
            pnl += (price - pos.avg_price) * pos.size * pos.side
        return pnl


# -----------------------
# Performance Metrics
# -----------------------
class Metrics:
    @staticmethod
    def equity_curve(equity_history: List[Tuple[pd.Timestamp, float]]) -> pd.Series:
        if not equity_history:
            return pd.Series(dtype=float)
        times, vals = zip(*equity_history)
        return pd.Series(list(vals), index=pd.to_datetime(list(times)))

    @staticmethod
    def compute_metrics(equity_series: pd.Series, trades: List[Dict[str, Any]]) -> Dict[str, Any]:
        metrics: Dict[str, Any] = {}
        if equity_series.empty:
            return metrics
        returns = equity_series.pct_change().fillna(0.0)
        total_return = equity_series.iloc[-1] / equity_series.iloc[0] - 1.0
        days = (equity_series.index[-1] - equity_series.index[0]).days or 1
        annual_factor = 252 / (1 if len(equity_series) == 1 else (len(equity_series) / (days / 252) if days else 252))
        cagr = (equity_series.iloc[-1] / equity_series.iloc[0]) ** (1.0 / (days / 365.0)) - 1.0 if days >= 1 else 0.0
        vol = returns.std() * math.sqrt(252) if len(returns) > 1 else 0.0
        sharpe = (returns.mean() * 252) / vol if vol > 0 else np.nan
        # drawdown
        cum = equity_series.cummax()
        drawdown = (equity_series - cum) / cum
        max_drawdown = drawdown.min()
        # trade stats
        wins = 0
        losses = 0
        pl = []
        for t in trades:
            # naive: compute trade pnl as sign * (close - entry) * size - costs; last trade record may be close or open
            pl_val = None
            if "executed_price" in t and "side" in t and "size" in t:
                # No simple exit price info here; this is approximate per trade record
                pl_val = -t["commission"] - t["slippage_cost"]
            if pl_val is not None:
                pl.append(pl_val)
        win_rate = (sum(1 for p in pl if p > 0) / len(pl)) if pl else np.nan
        metrics.update({
            "total_return": float(total_return),
            "cagr": float(cagr),
            "sharpe": float(sharpe) if not math.isnan(sharpe) else None,
            "volatility": float(vol),
            "max_drawdown": float(max_drawdown),
            "win_rate": float(win_rate) if not math.isnan(win_rate) else None,
            "num_trades": len(trades)
        })
        return metrics


# -----------------------
# Backtester
# -----------------------
class Backtester:
    """
    Basic event-driven backtester that runs strategy on OHLCV bars.
    Currently supports only single-symbol backtest per run.
    """
    def __init__(self, df: pd.DataFrame, strategy: Strategy, cfg: Config, logger: logging.Logger):
        self.data = DataHandler(df)
        self.strategy = strategy
        self.cfg = cfg
        self.logger = logger
        self.risk = RiskManager(cfg, logger)
        self.exec_sim = ExecutionSimulator(cfg, logger)
        self.portfolio = Portfolio(cfg, logger)
        self.symbol = "SYM"  # single-symbol placeholder; in multi-symbol system pass symbol per data

    def run(self) -> Dict[str, Any]:
        df = self.data.get_ohlcv()
        df = self.strategy.prepare(df)
        signals = self.strategy.generate_signals(df)
        stop_loss_pct = float(self.cfg.get("strategy.stop_loss_pct", DEFAULT_CONFIG["strategy"]["stop_loss_pct"]))
        take_profit_pct = float(self.cfg.get("strategy.take_profit_pct", DEFAULT_CONFIG["strategy"]["take_profit_pct"]))
        max_dd = float(self.cfg.get("backtest.max_drawdown_pct", DEFAULT_CONFIG["backtest"]["max_drawdown_pct"]))

        # Variables to track active trade stop levels
        active_stop_price: Optional[float] = None
        active_take_price: Optional[float] = None

        for timestamp, row in df.iterrows():
            price = float(row["close"])
            # Update market valuations
            self.portfolio.update_market(timestamp, {self.symbol: price})
            current_equity = self.portfolio.current_equity({self.symbol: price})
            # check equity drawdown limit
            if self._breached_max_drawdown(current_equity, max_dd):
                self.logger.warning("Max drawdown breached. Halting trading.")
                break

            sig = signals.loc[timestamp] if timestamp in signals.index else 0
            pos = self.portfolio.positions.get(self.symbol, Position())

            # Check if stop-loss or take-profit triggered intrabar (use low/high)
            triggered_exit = False
            if pos.size > 0 and pos.side != 0:
                # for long position
                if pos.side > 0:
                    if row["low"] <= active_stop_price if active_stop_price is not None else False:
                        # Stop hit: sell at stop price
                        self._execute_exit(timestamp, pos, active_stop_price)
                        triggered_exit = True
                    elif row["high"] >= active_take_price if active_take_price is not None else False:
                        self._execute_exit(timestamp, pos, active_take_price)
                        triggered_exit = True
                elif pos.side < 0:
                    # short position (not implemented in strategy example)
                    if row["high"] >= active_stop_price if active_stop_price is not None else False:
                        self._execute_exit(timestamp, pos, active_stop_price)
                        triggered_exit = True
                    elif row["low"] <= active_take_price if active_take_price is not None else False:
                        self._execute_exit(timestamp, pos, active_take_price)
                        triggered_exit = True
                if triggered_exit:
                    active_stop_price = None
                    active_take_price = None

            # Process entry signals
            if sig == 1 and (pos.size == 0):
                # compute stop-loss price and position size
                stop_price = price * (1.0 - stop_loss_pct)
                size = self.risk.compute_position_size(current_equity, price, stop_price)
                if size > 0:
                    # execute market buy at close price
                    exec_res = self.exec_sim.execute_order(timestamp, side=1, size=size, price=price)
                    self.portfolio.place_trade(timestamp, self.symbol, 1, exec_res["size"], exec_res["executed_price"],
                                               exec_res["commission"], exec_res["slippage_cost"])
                    # set stop and take
                    active_stop_price = stop_price
                    active_take_price = price * (1.0 + take_profit_pct)
                else:
                    self.logger.debug(f"Computed size 0 at {timestamp}; skipping trade.")
            elif sig == 0 and pos.size > 0:
                # explicit exit signal: close entire position at close price
                exec_res = self.exec_sim.execute_order(timestamp, side=-pos.side, size=pos.size, price=price)
                self.portfolio.place_trade(timestamp, self.symbol, -pos.side, exec_res["size"], exec_res["executed_price"],
                                           exec_res["commission"], exec_res["slippage_cost"])
                active_stop_price = None
                active_take_price = None

        # After loop, compute final equity history snapshot
        last_price = float(df["close"].iloc[-1])
        last_time = df.index[-1]
        self.portfolio.update_market(last_time, {self.symbol: last_price})
        equity_series = Metrics.equity_curve(self.portfolio.equity_history)
        metrics = Metrics.compute_metrics(equity_series, self.portfolio.trades)
        result = {
            "equity_series": equity_series,
            "metrics": metrics,
            "trades": self.portfolio.trades,
            "positions": {k: asdict(v) for k, v in self.portfolio.positions.items()}
        }
        return result

    def _execute_exit(self, timestamp: pd.Timestamp, pos: Position, exit_price: float):
        if pos.size <= 0:
            return
        side = -pos.side  # to close
        exec_res = self.exec_sim.execute_order(timestamp, side=side, size=pos.size, price=exit_price)
        self.portfolio.place_trade(timestamp, self.symbol, side, exec_res["size"], exec_res["executed_price"],
                                   exec_res["commission"], exec_res["slippage_cost"])

    def _breached_max_drawdown(self, equity: float, max_dd_pct: float) -> bool:
        # compute peak from equity_history
        if not self.portfolio.equity_history:
            return False
        _, vals = zip(*self.portfolio.equity_history)
        peak = max(vals)
        if peak <= 0:
            return False
        drawdown = (peak - equity) / peak
        return drawdown >= max_dd_pct


# -----------------------
# Example usage (if run as script)
# -----------------------
def _generate_sample_data(start: str = "2020-01-01", periods: int = 500, seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic OHLCV series for testing/backtesting.
    """
    np.random.seed(seed)
    dates = pd.bdate_range(start, periods=periods)
    price = 100.0 + np.cumsum(np.random.normal(0, 1, size=len(dates)))
    # Build OHLC around price with small variation
    o = price + np.random.normal(0, 0.2, size=len(dates))
    h = price + np.abs(np.random.normal(0, 0.5, size=len(dates)))
    l = price - np.abs(np.random.normal(0, 0.5, size=len(dates)))
    c = price + np.random.normal(0, 0.2, size=len(dates))
    v = np.random.randint(100, 10000, size=len(dates))
    df = pd.DataFrame({"open": o, "high": h, "low": l, "close": c, "volume": v}, index=dates)
    return df


def main():
    cfg = Config()
    logger = LoggerFactory.create_logger("Backtester", cfg)
    logger.info("Starting backtest framework example run.")
    data = _generate_sample_data()
    strategy = MovingAverageCrossStrategy(cfg, logger)
    bt = Backtester(data, strategy, cfg, logger)
    results = bt.run()
    logger.info("Backtest completed.")
    logger.info(f"Metrics: {results['metrics']}")
    logger.info(f"Num trades: {len(results['trades'])}")
    # save equity curve to CSV
    eq = results["equity_series"]
    if not eq.empty:
        eq.to_csv("equity_curve.csv")
        logger.info("Equity curve saved to equity_curve.csv")


if __name__ == "__main__":
    main()