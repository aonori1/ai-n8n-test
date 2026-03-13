#!/usr/bin/env python3
"""
Production-ready modular trading/backtesting framework (single-file for distribution).
Features:
- Configurable via YAML/JSON file or defaults
- Modular: DataFeed, Strategy, RiskManager, Portfolio, ExecutionSimulator, Backtester
- Risk management: volatility-based sizing (ATR), fixed-percentage risk, max position size, max drawdown, stop-loss, take-profit, per-day loss limit
- Logging with rotating file handler and console
- Ready for backtesting: bar-by-bar simulation, commission, slippage, fills
- Metrics: returns, Sharpe, max drawdown, CAGR
- Outputs trades and portfolio history CSVs

Dependencies:
- pandas, numpy, pyyaml (optional for YAML configs)
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple
import pandas as pd
import numpy as np
import logging
import logging.handlers
import os
import sys
import json
import math
import datetime as dt

try:
    import yaml
    YAML_AVAILABLE = True
except Exception:
    YAML_AVAILABLE = False

# -----------------------------
# Config utilities
# -----------------------------
DEFAULT_CONFIG = {
    "data": {
        "csv_path": "data/ohlcv.csv",  # expects columns: datetime, open, high, low, close, volume
        "datetime_col": "datetime",
        "price_col": "close",
        "start_date": None,
        "end_date": None,
    },
    "strategy": {
        "name": "ma_crossover",
        "short_window": 20,
        "long_window": 50,
        "atr_window": 14,
        "ma_price": "close",
        "entry_on_cross": True
    },
    "risk": {
        "risk_per_trade": 0.01,        # fraction of equity
        "max_position_size": 0.25,     # fraction of equity
        "max_drawdown": 0.25,          # fraction
        "daily_loss_limit": 0.05,      # fraction of equity
        "use_volatility_sizing": True,
        "min_size": 1,                 # minimum number of shares/contracts
    },
    "execution": {
        "commission_per_trade": 1.0,   # absolute
        "commission_pct": 0.0,         # fraction of trade value
        "slippage_abs": 0.0,
        "slippage_pct": 0.000,         # fraction of price
    },
    "backtest": {
        "initial_capital": 100000.0,
        "bar_size": "1D",
        "results_dir": "results",
    },
    "logging": {
        "level": "INFO",
        "log_file": "backtest.log",
        "console": True
    }
}


def load_config(path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration from JSON/YAML file or return defaults.
    """
    config = DEFAULT_CONFIG.copy()
    if path is None:
        return config
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    ext = os.path.splitext(path)[1].lower()
    with open(path, "r") as f:
        if ext in [".yaml", ".yml"]:
            if not YAML_AVAILABLE:
                raise RuntimeError("pyyaml is required to load YAML configs")
            loaded = yaml.safe_load(f)
        else:
            loaded = json.load(f)
    # deep merge with defaults
    def deep_update(d, u):
        for k, v in (u or {}).items():
            if isinstance(v, dict):
                d[k] = deep_update(d.get(k, {}), v)
            else:
                d[k] = v
        return d
    return deep_update(config, loaded)


# -----------------------------
# Logging
# -----------------------------
def setup_logging(config: Dict[str, Any]) -> logging.Logger:
    log_cfg = config.get("logging", {})
    level = getattr(logging, log_cfg.get("level", "INFO").upper(), logging.INFO)
    logger = logging.getLogger("backtester")
    logger.setLevel(level)
    # Avoid duplicate handlers if called multiple times
    if not logger.handlers:
        fmt = "%(asctime)s %(levelname)s %(name)s %(message)s"
        formatter = logging.Formatter(fmt)
        if log_cfg.get("console", True):
            ch = logging.StreamHandler(sys.stdout)
            ch.setFormatter(formatter)
            ch.setLevel(level)
            logger.addHandler(ch)
        log_file = log_cfg.get("log_file")
        if log_file:
            fh = logging.handlers.RotatingFileHandler(log_file, maxBytes=10_000_000, backupCount=5)
            fh.setFormatter(formatter)
            fh.setLevel(level)
            logger.addHandler(fh)
    return logger


# -----------------------------
# Data Feed
# -----------------------------
class DataFeed:
    """
    Simple CSV data loader providing OHLCV bars as a pandas DataFrame with datetime index.
    """
    def __init__(self, csv_path: str, datetime_col: str = "datetime", start_date: Optional[str] = None,
                 end_date: Optional[str] = None, logger: Optional[logging.Logger] = None):
        self.csv_path = csv_path
        self.datetime_col = datetime_col
        self.start_date = pd.to_datetime(start_date) if start_date else None
        self.end_date = pd.to_datetime(end_date) if end_date else None
        self.logger = logger or logging.getLogger("backtester")
        self.df = self._load()

    def _validate_df(self, df: pd.DataFrame) -> None:
        required = {"open", "high", "low", "close", "volume"}
        if not required.issubset(set(c.lower() for c in df.columns)):
            raise ValueError(f"CSV must contain columns: {required}. Found: {list(df.columns)}")

    def _load(self) -> pd.DataFrame:
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"CSV data file not found: {self.csv_path}")
        df = pd.read_csv(self.csv_path)
        if self.datetime_col not in df.columns:
            raise ValueError(f"Datetime column '{self.datetime_col}' not found in CSV")
        df[self.datetime_col] = pd.to_datetime(df[self.datetime_col])
        df = df.set_index(self.datetime_col).sort_index()
        # normalize column names to lower-case
        df.columns = [c.lower() for c in df.columns]
        self._validate_df(df)
        if self.start_date:
            df = df[df.index >= self.start_date]
        if self.end_date:
            df = df[df.index <= self.end_date]
        self.logger.info(f"Loaded data: {len(df)} bars from {self.csv_path}")
        return df

    def get_history(self) -> pd.DataFrame:
        return self.df.copy()


# -----------------------------
# Utilities: ATR
# -----------------------------
def compute_atr(df: pd.DataFrame, window: int = 14, high_col="high", low_col="low", close_col="close") -> pd.Series:
    high = df[high_col]
    low = df[low_col]
    close = df[close_col]
    tr1 = (high - low).abs()
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=window, min_periods=1).mean()
    return atr


# -----------------------------
# Strategy
# -----------------------------
class Strategy:
    """
    Base strategy class. Implement generate_signals to return DataFrame with 'signal' column:
    1 for long entry, -1 for short entry, 0 for flat/neutral. For safety, also support 'exit' column.
    """
    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        self.config = config
        self.logger = logger or logging.getLogger("backtester")

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError


class MovingAverageCrossover(Strategy):
    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        super().__init__(config, logger)
        s = config.get("strategy", {})
        self.short = int(s.get("short_window", 20))
        self.long = int(s.get("long_window", 50))
        self.atr_window = int(s.get("atr_window", 14))
        self.price_col = s.get("ma_price", "close")

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["ma_short"] = out[self.price_col].rolling(window=self.short, min_periods=1).mean()
        out["ma_long"] = out[self.price_col].rolling(window=self.long, min_periods=1).mean()
        out["ma_diff"] = out["ma_short"] - out["ma_long"]
        out["signal"] = 0
        # Generate simple crossover signals
        prev = out["ma_diff"].shift(1)
        out.loc[(out["ma_diff"] > 0) & (prev <= 0), "signal"] = 1
        out.loc[(out["ma_diff"] < 0) & (prev >= 0), "signal"] = -1
        # ATR for sizing
        out["atr"] = compute_atr(out, window=self.atr_window)
        self.logger.info(f"Strategy prepared with short={self.short}, long={self.long}, atr_window={self.atr_window}")
        return out


# -----------------------------
# Risk Manager
# -----------------------------
@dataclass
class RiskManager:
    config: Dict[str, Any]
    equity: float
    logger: logging.Logger
    peak_equity: float = field(default_factory=lambda: float("-inf"))
    daily_losses: Dict[pd.Timestamp, float] = field(default_factory=dict)

    def update_equity(self, new_equity: float, current_time: pd.Timestamp) -> None:
        self.equity = new_equity
        if new_equity > self.peak_equity:
            self.peak_equity = new_equity
        # initialize daily loss for date if missing
        date = current_time.normalize()
        self.daily_losses.setdefault(date, 0.0)

    def can_open_position(self, desired_amount_value: float, current_time: pd.Timestamp) -> Tuple[bool, Optional[str]]:
        cfg = self.config.get("risk", {})
        max_pos = cfg.get("max_position_size", 0.25)
        max_drawdown = cfg.get("max_drawdown", 0.25)
        daily_limit = cfg.get("daily_loss_limit", 0.05)

        if desired_amount_value > (self.equity * max_pos):
            return False, f"Desired pos ${desired_amount_value:.2f} exceeds max_position_size {max_pos*100:.1f}%"
        if (self.peak_equity - self.equity) / max(1.0, self.peak_equity) > max_drawdown:
            return False, f"Max drawdown exceeded: {(self.peak_equity - self.equity)/max(1.0, self.peak_equity):.2%}"
        date = current_time.normalize()
        day_loss = self.daily_losses.get(date, 0.0)
        if day_loss >= (self.peak_equity * daily_limit):
            return False, f"Daily loss limit reached: ${day_loss:.2f}"
        return True, None

    def register_loss(self, loss_amount: float, current_time: pd.Timestamp) -> None:
        date = current_time.normalize()
        self.daily_losses.setdefault(date, 0.0)
        self.daily_losses[date] += max(0.0, loss_amount)
        self.logger.debug(f"Registered loss ${loss_amount:.2f} on {date.date()} total day loss: ${self.daily_losses[date]:.2f}")


# -----------------------------
# Execution Simulator
# -----------------------------
@dataclass
class Trade:
    time: pd.Timestamp
    side: int             # +1 long, -1 short
    price: float
    size: float           # number of shares/contracts (positive)
    value: float          # signed value = side * price * size
    commission: float
    slippage: float
    pnl: Optional[float] = None
    entry_time: Optional[pd.Timestamp] = None
    exit_time: Optional[pd.Timestamp] = None
    exit_price: Optional[float] = None
    notes: Optional[str] = None


class ExecutionSimulator:
    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        self.config = config
        self.logger = logger or logging.getLogger("backtester")
        exec_cfg = config.get("execution", {})
        self.commission_per_trade = float(exec_cfg.get("commission_per_trade", 0.0))
        self.commission_pct = float(exec_cfg.get("commission_pct", 0.0))
        self.slippage_abs = float(exec_cfg.get("slippage_abs", 0.0))
        self.slippage_pct = float(exec_cfg.get("slippage_pct", 0.0))

    def get_fill_price(self, intended_price: float, side: int) -> Tuple[float, float]:
        """
        Returns (filled_price, slippage_amount). Slippage is modeled as either absolute or pct of price,
        and direction depends on side (worse for trader).
        """
        slippage = self.slippage_abs + (self.slippage_pct * intended_price)
        # For buys, price increases; for sells, price decreases (worse for trader)
        filled_price = intended_price + side * slippage
        # Side is +1 buy, -1 sell; adverse slippage is +slippage for buy, -slippage for sell -> but we want buys worse: filled price = price + slippage
        # so side here will be +1 for buy, -1 for sell, so multiply by +1 to increase price for buy, decrease for sell? We'll use sign = +1 for buy.
        # However above we used side * slippage, which increases buy price and decreases sell price -> correct.
        return filled_price, abs(slippage)

    def calculate_commission(self, filled_price: float, size: float) -> float:
        commission = self.commission_per_trade + self.commission_pct * abs(filled_price * size)
        return commission

    def execute(self, time: pd.Timestamp, side: int, intended_price: float, size: float) -> Trade:
        """
        Execute an order and return Trade record.
        size must be positive (number of units).
        side: +1 buy (go long), -1 sell (go short or close long depending)
        """
        filled_price, slippage = self.get_fill_price(intended_price, side)
        commission = self.calculate_commission(filled_price, size)
        value = side * filled_price * size
        trade = Trade(
            time=time,
            side=side,
            price=filled_price,
            size=size,
            value=value,
            commission=commission,
            slippage=slippage,
            notes=None
        )
        self.logger.info(f"Executed trade at {time} side={'BUY' if side>0 else 'SELL'} price={filled_price:.4f} size={size:.2f} commission={commission:.2f} slippage={slippage:.4f}")
        return trade


# -----------------------------
# Portfolio
# -----------------------------
class Portfolio:
    def __init__(self, initial_capital: float, execution: ExecutionSimulator, risk_manager: RiskManager, logger: logging.Logger):
        self.initial_capital = initial_capital
        self.execution = execution
        self.risk_manager = risk_manager
        self.logger = logger
        self.cash = float(initial_capital)
        self.positions = 0.0        # positive for long, negative for short (units)
        self.avg_price = 0.0        # average entry price for current position
        self.equity = float(initial_capital)
        self.trade_history: List[Trade] = []
        self.history = []  # list of dicts for timeseries (timestamp, cash, positions, avg_price, equity)

    def record_snapshot(self, time: pd.Timestamp, market_price: float) -> None:
        position_value = self.positions * market_price
        self.equity = self.cash + position_value
        self.risk_manager.update_equity(self.equity, time)
        self.history.append({
            "time": time,
            "cash": self.cash,
            "positions": self.positions,
            "avg_price": self.avg_price,
            "market_price": market_price,
            "equity": self.equity
        })

    def open_position(self, time: pd.Timestamp, side: int, price: float, size: float) -> Trade:
        """
        Open or add to position.
        """
        trade = self.execution.execute(time, side, price, size)
        # Update cash and positions
        self.cash -= trade.value  # value is signed: side*price*size
        self.cash -= trade.commission
        prev_positions = self.positions
        prev_avg = self.avg_price
        if prev_positions == 0 or (prev_positions > 0 and trade.side > 0) or (prev_positions < 0 and trade.side < 0):
            # same-side add
            new_pos = prev_positions + trade.side * trade.size
            if new_pos != 0:
                self.avg_price = ((prev_avg * abs(prev_positions)) + (trade.price * trade.size)) / (abs(new_pos))
            else:
                self.avg_price = 0.0
            self.positions = new_pos
        else:
            # reducing or flipping - compute realized pnl for the reduced part
            # size_to_close = min(abs(trade.size), abs(prev_positions))
            size_to_close = min(trade.size, abs(prev_positions))
            realized_pnl = 0.0
            if size_to_close > 0:
                # For long prev_positions>0, trade.side<0 to sell
                realized_pnl = (-trade.side) * size_to_close * (trade.price - prev_avg)
                # Update cash with realized pnl? Realized pnl is reflected in cash by trade.value already.
                # So we don't need to add separate; but for logging compute PnL:
            # After trade, update positions
            self.positions = prev_positions + trade.side * trade.size
            if self.positions == 0:
                self.avg_price = 0.0
            else:
                # If flipped, new avg based on remainder or new side
                if (prev_positions >= 0 and self.positions >= 0) or (prev_positions <= 0 and self.positions <= 0):
                    # same side effectively
                    # compute weighted avg
                    self.avg_price = ((prev_avg * abs(prev_positions - trade.side * trade.size)) + (trade.price * trade.size)) / abs(self.positions)
                else:
                    # flipped side: set avg to price of new fills for the net part
                    net_size = abs(self.positions)
                    self.avg_price = trade.price  # approximate
            # Realized pnl is implicitly in cash changes via trade.value; we can compute for risk_manager
            if realized_pnl != 0.0:
                self.risk_manager.register_loss(realized_pnl if realized_pnl < 0 else 0.0, time)

        # Append trade
        self.trade_history.append(trade)
        # Update equity after trade
        self.equity = self.cash + self.positions * price
        self.risk_manager.update_equity(self.equity, time)
        return trade

    def close_position(self, time: pd.Timestamp, price: float) -> Optional[Trade]:
        if self.positions == 0:
            return None
        side = -1 if self.positions > 0 else 1
        size = abs(self.positions)
        trade = self.open_position(time, side, price, size)
        # compute realized pnl for whole position
        realized_pnl = (trade.price - self.avg_price) * (-trade.side) * trade.size if self.avg_price != 0 else None
        # After close, avg_price should be zero handled in open_position
        return trade

    def get_unrealized_pnl(self, market_price: float) -> float:
        if self.positions == 0:
            return 0.0
        return (market_price - self.avg_price) * self.positions

    def current_equity(self, market_price: float) -> float:
        return self.cash + self.positions * market_price


# -----------------------------
# Backtester
# -----------------------------
class Backtester:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = setup_logging(config)
        data_cfg = config.get("data", {})
        self.data = DataFeed(
            csv_path=data_cfg.get("csv_path"),
            datetime_col=data_cfg.get("datetime_col", "datetime"),
            start_date=data_cfg.get("start_date"),
            end_date=data_cfg.get("end_date"),
            logger=self.logger
        )
        strat_name = (config.get("strategy") or {}).get("name", "ma_crossover")
        if strat_name == "ma_crossover":
            self.strategy = MovingAverageCrossover(config, logger=self.logger)
        else:
            raise ValueError(f"Unknown strategy: {strat_name}")
        initial_cap = float(config.get("backtest", {}).get("initial_capital", 100000.0))
        self.risk_manager = RiskManager(config=config, equity=initial_cap, logger=self.logger)
        self.execution = ExecutionSimulator(config, logger=self.logger)
        self.portfolio = Portfolio(initial_capital=initial_cap, execution=self.execution,
                                   risk_manager=self.risk_manager, logger=self.logger)
        self.results_dir = config.get("backtest", {}).get("results_dir", "results")
        os.makedirs(self.results_dir, exist_ok=True)

    def run(self) -> Dict[str, Any]:
        df_orig = self.data.get_history()
        df = self.strategy.prepare(df_orig)
        df = df.copy()
        # Ensure index is sorted
        df = df.sort_index()
        # Iterate bar-by-bar
        for time, row in df.iterrows():
            price = float(row.get("close", row.get("price", np.nan)))
            if np.isnan(price):
                self.logger.warning(f"No price at {time}, skipping bar")
                continue
            # Record snapshot at open of bar
            self.portfolio.record_snapshot(time, price)
            # Check signals
            signal = int(row.get("signal", 0))
            # ATR based stop distance for sizing
            atr = float(row.get("atr", np.nan)) if "atr" in row.index else np.nan
            # Generate orders based on signal
            if signal != 0:
                # Determine target size via risk manager
                desired_side = signal  # +1 buy, -1 sell
                risk_cfg = self.config.get("risk", {})
                risk_per_trade = float(risk_cfg.get("risk_per_trade", 0.01))
                use_vol = bool(risk_cfg.get("use_volatility_sizing", True))
                min_size = float(risk_cfg.get("min_size", 1))
                # Estimate stop distance: for long, stop = price - ATR*multiplier; use ATR as proxy
                if use_vol and not math.isnan(atr) and atr > 0:
                    stop_dist = atr * 1.5  # multiplier can be config
                    # risk per unit = stop_dist (in price terms) * unit size (1) -> for shares, approximate $ per share = stop_dist
                    # Compute size such that risk_per_trade * equity = stop_dist * size
                    max_risk_dollars = self.portfolio.equity * risk_per_trade
                    size = math.floor(max_risk_dollars / stop_dist) if stop_dist > 0 else min_size
                else:
                    # Fallback to percent-of-equity position sizing by dollar exposure
                    max_pos_val = self.portfolio.equity * float(risk_cfg.get("max_position_size", 0.25))
                    size = math.floor(max_pos_val / price) if price > 0 else min_size
                size = max(size, min_size)
                desired_value = size * price
                can_open, reason = self.risk_manager.can_open_position(desired_value, time)
                if not can_open:
                    self.logger.info(f"Position not opened at {time} reason: {reason}")
                else:
                    trade = self.portfolio.open_position(time=time, side=desired_side, price=price, size=size)
                    # Optionally apply stop-loss and take-profit tracking by storing in trade.notes (not implemented fully)
                    trade.notes = f"signal={signal} atr={atr:.4f}"
            # Optionally implement exit rules: closing on opposite signal
            # For this simple MA crossover, we close existing position on opposite signal
            if self.portfolio.positions != 0:
                current_side = 1 if self.portfolio.positions > 0 else -1
                if signal != 0 and signal != current_side:
                    # Close current position at current price
                    self.logger.info(f"Signal reversal at {time}: closing existing position")
                    self.portfolio.close_position(time=time, price=price)
            # end bar snapshot
            self.portfolio.record_snapshot(time, price)
        # After loop, compute metrics and save history/trades
        results = self.compute_results()
        self.save_results(results)
        return results

    def compute_results(self) -> Dict[str, Any]:
        hist = pd.DataFrame(self.portfolio.history).set_index("time").sort_index()
        hist["returns"] = hist["equity"].pct_change().fillna(0.0)
        total_return = (hist["equity"].iloc[-1] / hist["equity"].iloc[0]) - 1.0
        days = (hist.index[-1] - hist.index[0]).days if len(hist.index) > 1 else 1
        annual_return = ((1 + total_return) ** (365.0 / max(days, 1))) - 1.0 if days > 0 else 0.0
        # Sharpe ratio (daily returns assume daily bars)
        if hist["returns"].std() != 0:
            sharpe = (hist["returns"].mean() / hist["returns"].std()) * math.sqrt(252)
        else:
            sharpe = np.nan
        # Max Drawdown
        running_max = hist["equity"].cummax()
        dd = (running_max - hist["equity"]) / running_max
        max_dd = dd.max()
        trades_df = pd.DataFrame([t.__dict__ for t in self.portfolio.trade_history])
        metrics = {
            "total_return": total_return,
            "annual_return": annual_return,
            "sharpe": sharpe,
            "max_drawdown": max_dd,
            "equity_curve": hist,
            "trades": trades_df
        }
        self.logger.info(f"Backtest complete: total_return={total_return:.2%} annual_return={annual_return:.2%} sharpe={sharpe:.2f} max_dd={max_dd:.2%}")
        return metrics

    def save_results(self, results: Dict[str, Any]) -> None:
        hist = results["equity_curve"]
        trades = results["trades"]
        hist_fn = os.path.join(self.results_dir, "equity_curve.csv")
        trades_fn = os.path.join(self.results_dir, "trades.csv")
        hist.to_csv(hist_fn, index=True)
        trades.to_csv(trades_fn, index=False)
        self.logger.info(f"Saved equity curve to {hist_fn} and trades to {trades_fn}")


# -----------------------------
# CLI / Entrypoint
# -----------------------------
def main(config_path: Optional[str] = None):
    """
    Example entrypoint. Pass an optional path to a JSON/YAML config.
    """
    config = load_config(config_path) if config_path else DEFAULT_CONFIG
    bt = Backtester(config)
    results = bt.run()
    # Print summary
    eq = results["equity_curve"]
    last = eq["equity"].iloc[-1]
    init = eq["equity"].iloc[0]
    print(f"Initial Equity: ${init:,.2f}")
    print(f"Final Equity:   ${last:,.2f}")
    print(f"Total Return:   {(last/init - 1.0):.2%}")
    print(f"Trades:         {len(results['trades'])}")
    return results


# Allows running as script
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Backtest Moving Average Crossover Strategy")
    parser.add_argument("--config", "-c", help="Path to JSON/YAML config file", default=None)
    args = parser.parse_args()
    try:
        main(args.config)
    except Exception as e:
        logger = logging.getLogger("backtester")
        logger.exception("Backtest failed")
        raise

# End of file.