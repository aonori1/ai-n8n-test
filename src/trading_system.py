import argparse
import json
import logging
import logging.handlers
import os
import sys
import traceback
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# -------------------------
# Configuration Management
# -------------------------
class ConfigError(Exception):
    pass


class ConfigLoader:
    DEFAULT_CONFIG = {
        "logging": {
            "level": "INFO",
            "filename": "backtest.log",
            "max_bytes": 10 * 1024 * 1024,
            "backup_count": 5,
            "console": True
        },
        "data": {
            "symbol": "EURUSD",
            "csv_path": None,
            "time_column": "Date",
            "time_format": "%Y-%m-%d %H:%M:%S",
            "price_columns": {
                "open": "Open",
                "high": "High",
                "low": "Low",
                "close": "Close",
                "volume": "Volume"
            }
        },
        "strategy": {
            "name": "MovingAverageCross",
            "settings": {
                "fast_window": 20,
                "slow_window": 50
            }
        },
        "risk": {
            "max_position_size": 0.1,
            "max_notional_per_trade": None,
            "risk_per_trade": 0.01,
            "max_drawdown": 0.2,
            "initial_capital": 100000.0,
            "commission_per_share": 0.0,
            "slippage_bps": 0.0001,
            "use_atr_sizing": True,
            "atr_window": 14
        },
        "backtest": {
            "start_date": None,
            "end_date": None,
            "frequency": "1min",
            "result_dir": "results"
        }
    }

    @staticmethod
    def load(path: Optional[str]) -> dict:
        if path is None:
            return ConfigLoader.DEFAULT_CONFIG
        if not os.path.exists(path):
            raise ConfigError(f"Configuration file not found: {path}")
        try:
            with open(path, "r") as f:
                if path.lower().endswith(".json"):
                    cfg = json.load(f)
                else:
                    # attempt JSON parse for genericity
                    cfg = json.load(f)
        except Exception as e:
            raise ConfigError(f"Failed to load configuration: {e}")
        # Merge defaults with provided
        merged = ConfigLoader.DEFAULT_CONFIG.copy()
        ConfigLoader._deep_update(merged, cfg)
        return merged

    @staticmethod
    def _deep_update(orig: dict, new: dict):
        for k, v in new.items():
            if isinstance(v, dict) and k in orig and isinstance(orig[k], dict):
                ConfigLoader._deep_update(orig[k], v)
            else:
                orig[k] = v


# -------------------------
# Logging Setup
# -------------------------
def setup_logging(cfg: dict):
    log_cfg = cfg.get("logging", {})
    level_name = log_cfg.get("level", "INFO")
    level = getattr(logging, level_name.upper(), logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(level)

    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s %(message)s"
    )

    # File handler with rotation
    filename = log_cfg.get("filename", "backtest.log")
    max_bytes = log_cfg.get("max_bytes", 10 * 1024 * 1024)
    backup_count = log_cfg.get("backup_count", 5)
    file_handler = logging.handlers.RotatingFileHandler(
        filename, maxBytes=max_bytes, backupCount=backup_count
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Console handler
    if log_cfg.get("console", True):
        console = logging.StreamHandler(sys.stdout)
        console.setFormatter(formatter)
        logger.addHandler(console)

    logging.getLogger("matplotlib").setLevel(logging.WARNING)


# -------------------------
# Data Handler
# -------------------------
class DataHandler:
    """
    Loads historical data from CSV and provides indexed DataFrame for backtesting.
    Expects OHLCV data with a datetime column.
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.logger = logging.getLogger(self.__class__.__name__)
        self.df: Optional[pd.DataFrame] = None

    def load_csv(self, path: Optional[str]) -> pd.DataFrame:
        if path is None or not os.path.exists(path):
            raise FileNotFoundError(f"Data CSV not found: {path}")
        data_cfg = self.cfg["data"]
        time_col = data_cfg.get("time_column", "Date")
        time_fmt = data_cfg.get("time_format", None)
        price_cols = data_cfg.get("price_columns", {})
        try:
            parse_dates = [time_col]
            df = pd.read_csv(path, parse_dates=parse_dates)
            if time_fmt:
                df[time_col] = pd.to_datetime(df[time_col], format=time_fmt)
            else:
                df[time_col] = pd.to_datetime(df[time_col])
            df.set_index(time_col, inplace=True)
            # Ensure columns exist
            required = ["open", "high", "low", "close"]
            for k in required:
                colname = price_cols.get(k, k.capitalize())
                if colname not in df.columns:
                    raise ValueError(f"Required column '{colname}' not found in CSV")
                # normalize to lowercase columns
                df[k] = df[colname]
            vol_col = price_cols.get("volume", "Volume")
            if vol_col in df.columns:
                df["volume"] = df[vol_col]
            else:
                df["volume"] = np.nan
            self.df = df[["open", "high", "low", "close", "volume"]].copy()
            self.logger.info("Loaded data: %s rows", len(self.df))
            return self.df
        except Exception:
            self.logger.error("Failed to load CSV data:\n%s", traceback.format_exc())
            raise

    def get_slice(self, start: Optional[str], end: Optional[str]) -> pd.DataFrame:
        if self.df is None:
            raise ValueError("Data not loaded")
        df = self.df
        if start:
            df = df[df.index >= pd.to_datetime(start)]
        if end:
            df = df[df.index <= pd.to_datetime(end)]
        return df.copy()


# -------------------------
# Strategy Base and Example
# -------------------------
class Strategy:
    """
    Base class for strategies. Implement generate_signals to produce entries/exits.
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.logger = logging.getLogger(self.__class__.__name__)

    def prepare(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Precompute any indicators. Must return DataFrame with same index.
        """
        raise NotImplementedError

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Produce a DataFrame with a column 'signal' containing:
        1 => long, -1 => short, 0 => flat
        """
        raise NotImplementedError


class MovingAverageCrossStrategy(Strategy):
    def __init__(self, cfg: dict):
        super().__init__(cfg)
        settings = cfg.get("strategy", {}).get("settings", {})
        self.fast = int(settings.get("fast_window", 20))
        self.slow = int(settings.get("slow_window", 50))
        if self.fast >= self.slow:
            self.logger.warning("Fast window >= slow window; swapping for validity")
            self.fast, self.slow = min(self.fast, self.slow), max(self.fast, self.slow)

    def prepare(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        df["ma_fast"] = df["close"].rolling(window=self.fast, min_periods=1).mean()
        df["ma_slow"] = df["close"].rolling(window=self.slow, min_periods=1).mean()
        return df

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        df = self.prepare(data)
        df["signal"] = 0
        df.loc[df["ma_fast"] > df["ma_slow"], "signal"] = 1
        df.loc[df["ma_fast"] < df["ma_slow"], "signal"] = -1
        # compute signal changes (entry/exit)
        df["signal_change"] = df["signal"].diff().fillna(0)
        return df


# -------------------------
# Risk Management
# -------------------------
@dataclass
class Order:
    timestamp: pd.Timestamp
    symbol: str
    side: int  # 1=buy, -1=sell
    price: float
    quantity: float
    commission: float = 0.0
    slippage: float = 0.0


@dataclass
class Position:
    symbol: str
    quantity: float = 0.0
    avg_price: float = 0.0

    def value(self, price: float) -> float:
        return self.quantity * price


class RiskManager:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.logger = logging.getLogger(self.__class__.__name__)
        self.initial_capital = float(cfg["risk"].get("initial_capital", 100000.0))
        self.max_position_pct = float(cfg["risk"].get("max_position_size", 0.1))
        self.risk_per_trade = float(cfg["risk"].get("risk_per_trade", 0.01))
        self.max_drawdown = float(cfg["risk"].get("max_drawdown", 0.2))
        self.atr_window = int(cfg["risk"].get("atr_window", 14))
        self.use_atr = bool(cfg["risk"].get("use_atr_sizing", True))
        self.commission_per_share = float(cfg["risk"].get("commission_per_share", 0.0))
        self.slippage_bps = float(cfg["risk"].get("slippage_bps", 0.0001))

    def compute_notional_limit(self, current_equity: float) -> float:
        return current_equity * self.max_position_pct

    def position_size_from_risk(self, price: float, atr: Optional[float], current_equity: float) -> float:
        """
        Compute quantity such that the risk per trade is approximately risk_per_trade of equity.
        If ATR provided, set stop distance = ATR, else use a default percent.
        """
        risk_amount = current_equity * self.risk_per_trade
        if self.use_atr and atr and atr > 0:
            stop_dist = atr
        else:
            stop_dist = price * 0.01  # default 1%
        qty = risk_amount / max(stop_dist, 1e-8)
        max_notional = self.compute_notional_limit(current_equity)
        notional = qty * price
        if notional > max_notional:
            qty = max_notional / price
        return float(np.floor(qty))  # integer shares/units

    def compute_atr(self, df: pd.DataFrame) -> pd.Series:
        tr1 = df["high"] - df["low"]
        tr2 = (df["high"] - df["close"].shift(1)).abs()
        tr3 = (df["low"] - df["close"].shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=self.atr_window, min_periods=1).mean()
        return atr

    def apply_commission_slippage(self, price: float, side: int) -> Tuple[float, float]:
        # slippage in price units
        slippage = price * self.slippage_bps * (1 if side == 1 else -1)
        commission = self.commission_per_share
        executed_price = price + slippage
        return executed_price, commission


# -------------------------
# Execution Simulator
# -------------------------
class ExecutionHandler:
    """
    Simulates execution: using next bar open price by default.
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.logger = logging.getLogger(self.__class__.__name__)

    def execute_order(self, order: Order, bar: pd.Series) -> Order:
        """
        Simulate slippage/commission and adjust price using bar open or close.
        """
        # For simplicity, use bar['open'] as execution price if available, else order.price
        exec_price = order.price
        if "open" in bar and not np.isnan(bar["open"]):
            exec_price = bar["open"]
        # commission and slippage already included in order fields from RiskManager
        executed_order = Order(
            timestamp=order.timestamp,
            symbol=order.symbol,
            side=order.side,
            price=exec_price + order.slippage,
            quantity=order.quantity,
            commission=order.commission,
            slippage=order.slippage
        )
        self.logger.debug("Executed order: %s", asdict(executed_order))
        return executed_order


# -------------------------
# Portfolio and Backtest Engine
# -------------------------
class Portfolio:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.logger = logging.getLogger(self.__class__.__name__)
        self.cash = float(cfg["risk"].get("initial_capital", 100000.0))
        self.positions: Dict[str, Position] = {}
        self.trade_history: List[Dict] = []
        self.equity_history: List[Dict] = []

    def update_with_fill(self, order: Order):
        symbol = order.symbol
        qty = order.quantity if order.side == 1 else -order.quantity
        price = order.price
        commission = order.commission
        # update cash
        cost = qty * price + commission
        self.cash -= cost
        # update position
        pos = self.positions.get(symbol, Position(symbol, 0.0, 0.0))
        if pos.quantity + qty == 0:
            # closing position
            pos.avg_price = 0.0
            pos.quantity = 0.0
        elif pos.quantity == 0:
            pos.avg_price = price
            pos.quantity = qty
        else:
            # update weighted avg
            new_qty = pos.quantity + qty
            if new_qty == 0:
                pos.avg_price = 0.0
                pos.quantity = 0.0
            else:
                pos.avg_price = (pos.avg_price * pos.quantity + price * qty) / new_qty
                pos.quantity = new_qty
        self.positions[symbol] = pos
        # record trade
        self.trade_history.append({
            "timestamp": order.timestamp,
            "symbol": symbol,
            "side": "BUY" if order.side == 1 else "SELL",
            "price": price,
            "quantity": order.quantity,
            "commission": commission,
            "slippage": order.slippage,
            "cash": self.cash
        })
        self.logger.debug("Trade recorded. Cash: %s, Position: %s", self.cash, asdict(pos))

    def total_equity(self, market_prices: Dict[str, float]) -> float:
        equity = self.cash
        for sym, pos in self.positions.items():
            price = market_prices.get(sym, 0.0)
            equity += pos.quantity * price
        return equity

    def snapshot(self, timestamp: pd.Timestamp, market_prices: Dict[str, float]):
        equity = self.total_equity(market_prices)
        self.equity_history.append({"timestamp": timestamp, "equity": equity})
        self.logger.debug("Snapshot at %s: equity=%s", timestamp, equity)


class Backtester:
    def __init__(self, cfg: dict, data: pd.DataFrame, strategy: Strategy):
        self.cfg = cfg
        self.logger = logging.getLogger(self.__class__.__name__)
        self.data = data.copy()
        self.strategy = strategy
        self.risk_manager = RiskManager(cfg)
        self.executor = ExecutionHandler(cfg)
        self.portfolio = Portfolio(cfg)
        self.symbol = cfg["data"].get("symbol", "SYM")
        # compute ATR once
        self.atr = self.risk_manager.compute_atr(self.data)

    def run(self):
        df = self.strategy.generate_signals(self.data)
        # iterate rows
        last_signal = 0
        for idx, row in df.iterrows():
            try:
                signal = int(row.get("signal", 0))
            except Exception:
                signal = 0
            # Determine market price dictionary for portfolio valuation
            market_prices = {self.symbol: row["close"]}
            # snapshot pre-trade
            self.portfolio.snapshot(idx, market_prices)
            # Check for signal change
            if signal != last_signal:
                # Exit if going to flat or reversing
                if last_signal != 0 and signal != last_signal:
                    # create exit order for existing position
                    pos = self.portfolio.positions.get(self.symbol)
                    if pos and pos.quantity != 0:
                        side = -1 if pos.quantity > 0 else 1
                        qty = abs(pos.quantity)
                        price = float(row["open"] if "open" in row and not np.isnan(row["open"]) else row["close"])
                        # apply risk manager for commission/slippage
                        exec_price, commission = self.risk_manager.apply_commission_slippage(price, side)
                        order = Order(timestamp=idx, symbol=self.symbol, side=side, price=exec_price, quantity=qty,
                                      commission=commission, slippage=exec_price - price)
                        executed = self.executor.execute_order(order, row)
                        self.portfolio.update_with_fill(executed)
                        self.logger.info("%s EXIT @ %s qty=%s", idx, executed.price, executed.quantity)
                # Enter if new directional signal
                if signal != 0 and signal != last_signal:
                    price = float(row["open"] if "open" in row and not np.isnan(row["open"]) else row["close"])
                    atr_val = float(self.atr.loc[idx]) if idx in self.atr.index else None
                    qty = int(self.risk_manager.position_size_from_risk(price, atr_val, self.portfolio.total_equity({self.symbol: price})))
                    if qty <= 0:
                        self.logger.debug("Calculated position size 0, skipping entry")
                    else:
                        side = 1 if signal == 1 else -1
                        exec_price, commission = self.risk_manager.apply_commission_slippage(price, side)
                        slippage = exec_price - price
                        order = Order(timestamp=idx, symbol=self.symbol, side=side, price=exec_price, quantity=qty,
                                      commission=commission, slippage=slippage)
                        executed = self.executor.execute_order(order, row)
                        self.portfolio.update_with_fill(executed)
                        self.logger.info("%s ENTRY %s @ %s qty=%s", idx, "LONG" if side == 1 else "SHORT", executed.price, executed.quantity)
                # update last_signal
                last_signal = signal
            # After trades, check risk constraints (e.g., max drawdown)
            equity = self.portfolio.total_equity({self.symbol: row["close"]})
            self._check_risk_limits(equity, idx)
        # final snapshot
        last_price = float(self.data["close"].iloc[-1])
        self.portfolio.snapshot(self.data.index[-1], {self.symbol: last_price})
        self.logger.info("Backtest complete. Final equity: %s", self.portfolio.total_equity({self.symbol: last_price}))
        return self.portfolio

    def _check_risk_limits(self, equity: float, timestamp: pd.Timestamp):
        # check drawdown
        eq_hist = pd.DataFrame(self.portfolio.equity_history)
        if eq_hist.empty:
            return
        peak = eq_hist["equity"].cummax().iloc[-1]
        drawdown = (peak - equity) / peak if peak > 0 else 0.0
        if drawdown >= self.risk_manager.max_drawdown:
            # trigger risk action: close all positions
            self.logger.warning("Max drawdown exceeded at %s: drawdown=%.2f%%. Closing positions.", timestamp, drawdown * 100)
            # force close
            for sym, pos in list(self.portfolio.positions.items()):
                if pos.quantity != 0:
                    side = -1 if pos.quantity > 0 else 1
                    qty = abs(pos.quantity)
                    price = float(self.data.loc[timestamp]["close"]) if timestamp in self.data.index else float(self.data["close"].iloc[-1])
                    exec_price, commission = self.risk_manager.apply_commission_slippage(price, side)
                    order = Order(timestamp=timestamp, symbol=sym, side=side, price=exec_price, quantity=qty,
                                  commission=commission, slippage=exec_price - price)
                    executed = self.executor.execute_order(order, self.data.loc[timestamp])
                    self.portfolio.update_with_fill(executed)


# -------------------------
# Performance Analyzer
# -------------------------
class PerformanceAnalyzer:
    def __init__(self, portfolio: Portfolio):
        self.portfolio = portfolio
        self.logger = logging.getLogger(self.__class__.__name__)

    def get_equity_series(self) -> pd.Series:
        df = pd.DataFrame(self.portfolio.equity_history)
        if df.empty:
            return pd.Series(dtype=float)
        df.set_index("timestamp", inplace=True)
        return df["equity"]

    def calculate_metrics(self) -> Dict[str, float]:
        eq = self.get_equity_series()
        if eq.empty:
            return {}
        returns = eq.pct_change().fillna(0)
        cumulative_return = (eq.iloc[-1] / eq.iloc[0]) - 1 if eq.iloc[0] != 0 else np.nan
        annual_factor = 252 * 24 * 60  # assuming minute data; placeholder, user should adjust
        # compute sharpe with daily assumption protection
        mean_ret = returns.mean()
        std_ret = returns.std() if returns.std() != 0 else np.nan
        sharpe = (mean_ret / std_ret) * np.sqrt(annual_factor) if std_ret and not np.isnan(std_ret) else np.nan
        # max drawdown
        cummax = eq.cummax()
        drawdown = (cummax - eq) / cummax
        max_dd = drawdown.max()
        metrics = {
            "initial_equity": float(eq.iloc[0]),
            "final_equity": float(eq.iloc[-1]),
            "cumulative_return": float(cumulative_return),
            "sharpe": float(sharpe) if not np.isnan(sharpe) else np.nan,
            "max_drawdown": float(max_dd)
        }
        return metrics

    def save_trade_history(self, path: str):
        df = pd.DataFrame(self.portfolio.trade_history)
        if df.empty:
            self.logger.info("No trades to save.")
            return
        df.to_csv(path, index=False)
        self.logger.info("Saved trade history to %s", path)


# -------------------------
# Utilities / CLI
# -------------------------
def ensure_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def main(config_path: Optional[str], data_path: Optional[str]):
    try:
        cfg = ConfigLoader.load(config_path)
    except Exception as e:
        print(f"Failed to load config: {e}")
        sys.exit(1)

    setup_logging(cfg)
    logger = logging.getLogger("Main")
    logger.info("Starting backtest")

    # Load data
    data_cfg = cfg.get("data", {})
    csv_path = data_path if data_path else data_cfg.get("csv_path")
    if csv_path is None:
        logger.error("No data CSV specified in config or CLI.")
        sys.exit(1)

    data_handler = DataHandler(cfg)
    try:
        df = data_handler.load_csv(csv_path)
    except Exception as e:
        logger.error("Failed loading data: %s", e)
        sys.exit(1)

    # Trim date range
    bt_cfg = cfg.get("backtest", {})
    start = bt_cfg.get("start_date")
    end = bt_cfg.get("end_date")
    df = data_handler.get_slice(start, end)

    # Select and instantiate strategy
    strat_name = cfg.get("strategy", {}).get("name", "MovingAverageCross")
    if strat_name == "MovingAverageCross":
        strategy = MovingAverageCrossStrategy(cfg)
    else:
        logger.error("Unknown strategy: %s", strat_name)
        sys.exit(1)

    # Run backtest
    backtester = Backtester(cfg, df, strategy)
    portfolio = backtester.run()

    # Analyze performance
    analyzer = PerformanceAnalyzer(portfolio)
    metrics = analyzer.calculate_metrics()
    result_dir = bt_cfg.get("result_dir", "results")
    ensure_dir(result_dir)
    metrics_path = os.path.join(result_dir, "metrics.json")
    trades_path = os.path.join(result_dir, "trades.csv")
    try:
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        analyzer.save_trade_history(trades_path)
    except Exception:
        logger.error("Failed to write results: %s", traceback.format_exc())
    logger.info("Metrics: %s", metrics)
    logger.info("Backtest finished. Results saved to %s", result_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backtest engine")
    parser.add_argument("--config", "-c", help="Path to config JSON file", default=None)
    parser.add_argument("--data", "-d", help="Path to CSV data file (overrides config)", default=None)
    args = parser.parse_args()
    main(args.config, args.data)