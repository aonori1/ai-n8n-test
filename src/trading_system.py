import logging
import logging.handlers
import json
import math
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, Optional, List, Any, Tuple

import numpy as np
import pandas as pd


# ----------------------------
# Configuration handling
# ----------------------------
class Config:
    """
    Configuration manager. Loads config from a dictionary or JSON/YAML file path (JSON supported natively).
    Provides default configuration for backtesting and risk management.
    """

    DEFAULT = {
        "logging": {
            "level": "INFO",
            "file": "backtest.log",
            "max_bytes": 10 * 1024 * 1024,
            "backup_count": 3
        },
        "backtest": {
            "start_cash": 100000.0,
            "commission_per_share": 0.0,
            "slippage_pct": 0.0,
            "data_frequency": "1D"
        },
        "risk": {
            "risk_per_trade": 0.01,   # fraction of equity to risk per trade
            "max_drawdown": 0.2,      # maximum allowed drawdown fraction before halting trading
            "max_exposure": 1.0,      # maximum portfolio exposure (fraction of equity)
            "max_position_size": 0.5, # maximum single position size (fraction of equity)
            "daily_loss_limit": 0.02  # fraction of equity daily loss limit
        },
        "strategy": {
            "fast_window": 20,
            "slow_window": 50,
            "use_atr": True,
            "atr_window": 14,
            "fixed_size": None       # override dynamic sizing, number of shares if set
        }
    }

    def __init__(self, config_source: Optional[str] = None, overrides: Optional[Dict[str, Any]] = None):
        """
        config_source: Optional path to JSON config file.
        overrides: Optional dict to merge on top of defaults.
        """
        self.config = dict(Config.DEFAULT)
        if config_source:
            self._load_from_file(config_source)
        if overrides:
            self._deep_update(self.config, overrides)

    def _load_from_file(self, path: str):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Config file not found: {path}")
        with open(path, 'r') as f:
            # Support JSON only for now (YAML could be added)
            cfg = json.load(f)
            self._deep_update(self.config, cfg)

    @staticmethod
    def _deep_update(dest: Dict[str, Any], upd: Dict[str, Any]):
        for k, v in upd.items():
            if isinstance(v, dict) and k in dest and isinstance(dest[k], dict):
                Config._deep_update(dest[k], v)
            else:
                dest[k] = v

    def get(self, path: str, default: Any = None) -> Any:
        """
        Get nested config value by dot-separated path, e.g., 'risk.max_drawdown'
        """
        parts = path.split('.')
        cur = self.config
        for p in parts:
            if not isinstance(cur, dict) or p not in cur:
                return default
            cur = cur[p]
        return cur

    def as_dict(self):
        return self.config


# ----------------------------
# Logging
# ----------------------------
def setup_logging(cfg: Config):
    cfg_dict = cfg.as_dict()
    log_cfg = cfg_dict.get('logging', {})
    level = getattr(logging, log_cfg.get('level', 'INFO').upper(), logging.INFO)
    log_file = log_cfg.get('file', 'backtest.log')
    max_bytes = log_cfg.get('max_bytes', 10 * 1024 * 1024)
    backup_count = log_cfg.get('backup_count', 3)

    logger = logging.getLogger()
    logger.setLevel(level)

    # Clear existing handlers
    if logger.handlers:
        for h in list(logger.handlers):
            logger.removeHandler(h)

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
    logger.addHandler(ch)

    # Rotating file handler
    fh = logging.handlers.RotatingFileHandler(log_file, maxBytes=max_bytes, backupCount=backup_count)
    fh.setLevel(level)
    fh.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
    logger.addHandler(fh)

    logging.debug("Logging initialized (level=%s file=%s)", logging.getLevelName(level), log_file)
    return logger


# ----------------------------
# Data handling
# ----------------------------
class DataHandler:
    """
    Simple data handler that holds DataFrame of OHLCV data for a single symbol.
    Expects index to be datetime-like and columns to include: Open, High, Low, Close, Volume
    """

    REQUIRED_COLS = ["Open", "High", "Low", "Close", "Volume"]

    def __init__(self, df: pd.DataFrame):
        if not isinstance(df, pd.DataFrame):
            raise ValueError("df must be a pandas DataFrame")
        for c in DataHandler.REQUIRED_COLS:
            if c not in df.columns:
                raise ValueError(f"Missing required column: {c}")
        df = df.copy()
        if not pd.api.types.is_datetime64_any_dtype(df.index):
            raise ValueError("DataFrame index must be datetime")
        df.sort_index(inplace=True)
        self.df = df
        self._n = len(df)
        self._pos = 0

    def reset(self):
        self._pos = 0

    def __len__(self):
        return self._n

    def get_latest_bar(self) -> Tuple[pd.Timestamp, pd.Series]:
        if self._pos >= self._n:
            raise IndexError("End of data")
        ts = self.df.index[self._pos]
        row = self.df.iloc[self._pos]
        return ts, row

    def step(self):
        if self._pos < self._n - 1:
            self._pos += 1
            return True
        self._pos = self._n
        return False

    def get_slice_up_to(self, pos: Optional[int] = None) -> pd.DataFrame:
        if pos is None:
            pos = self._pos
        return self.df.iloc[:pos + 1]


# ----------------------------
# Orders and Trades
# ----------------------------
@dataclass
class Order:
    symbol: str
    quantity: int
    side: str  # 'BUY' or 'SELL'
    price: float
    timestamp: pd.Timestamp
    order_type: str = "MARKET"


@dataclass
class Trade:
    symbol: str
    quantity: int
    side: str
    price: float
    timestamp: pd.Timestamp
    commission: float


# ----------------------------
# Strategy
# ----------------------------
class Strategy:
    """
    Base strategy interface.
    """

    def on_bar(self, bar_ts: pd.Timestamp, bar: pd.Series, history: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """
        Called every bar. Should return:
            None or {} -> do nothing
            {'signal': 1/-1, 'stop_loss_pct': 0.02, 'take_profit_pct': 0.04}
        """
        raise NotImplementedError()


class MovingAverageCrossStrategy(Strategy):
    """
    Simple moving average crossover:
      - Go long when fast MA crosses above slow MA
      - Exit when fast MA crosses below slow MA
    """

    def __init__(self, cfg: Config):
        self.fast_window = cfg.get("strategy.fast_window")
        self.slow_window = cfg.get("strategy.slow_window")
        self.atr_window = cfg.get("strategy.atr_window", 14)
        self.use_atr = cfg.get("strategy.use_atr", True)
        self.fixed_size = cfg.get("strategy.fixed_size", None)
        # state
        self._position = 0

    def _compute_indicators(self, history: pd.DataFrame) -> pd.DataFrame:
        df = history.copy()
        df['fast_ma'] = df['Close'].rolling(self.fast_window, min_periods=1).mean()
        df['slow_ma'] = df['Close'].rolling(self.slow_window, min_periods=1).mean()
        # ATR approx
        high_low = df['High'] - df['Low']
        high_close = (df['High'] - df['Close'].shift(1)).abs()
        low_close = (df['Low'] - df['Close'].shift(1)).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['atr'] = tr.rolling(self.atr_window, min_periods=1).mean()
        return df

    def on_bar(self, bar_ts: pd.Timestamp, bar: pd.Series, history: pd.DataFrame) -> Optional[Dict[str, Any]]:
        df = self._compute_indicators(history)
        last = df.iloc[-1]
        prev = df.iloc[-2] if len(df) >= 2 else None

        signal = 0
        # Entry condition
        if prev is not None:
            if prev['fast_ma'] <= prev['slow_ma'] and last['fast_ma'] > last['slow_ma']:
                signal = 1
            elif prev['fast_ma'] >= prev['slow_ma'] and last['fast_ma'] < last['slow_ma']:
                signal = -1

        if signal == 1:
            # Enter long
            self._position = 1
            # Suggest stop loss based on ATR or percent
            if self.use_atr and last['atr'] > 0 and not np.isnan(last['atr']):
                stop_loss_pct = (last['atr'] * 2) / last['Close']  # 2 ATR stop
            else:
                stop_loss_pct = 0.02
            return {'signal': 1, 'stop_loss_pct': float(stop_loss_pct), 'take_profit_pct': float(stop_loss_pct * 2)}
        elif signal == -1:
            # Exit or go short; for simplicity treat as exit (close long)
            self._position = 0
            return {'signal': -1}
        else:
            return None


# ----------------------------
# Risk Management
# ----------------------------
class RiskManager:
    """
    Risk manager used to validate and size orders.
    """

    def __init__(self, cfg: Config):
        self.risk_per_trade = cfg.get("risk.risk_per_trade")
        self.max_drawdown = cfg.get("risk.max_drawdown")
        self.max_exposure = cfg.get("risk.max_exposure")
        self.max_position_size = cfg.get("risk.max_position_size")
        self.daily_loss_limit = cfg.get("risk.daily_loss_limit")
        self.start_cash = cfg.get("backtest.start_cash")

    def validate_and_size(self,
                          portfolio: 'Portfolio',
                          signal: int,
                          price: float,
                          stop_loss_pct: Optional[float] = None,
                          fixed_size: Optional[int] = None) -> Optional[int]:
        """
        Return integer number of shares to buy/sell. Return None if order should be rejected.
        For long entries (signal == 1) compute position sizing dynamically:
           size = floor( (equity * risk_per_trade) / (stop_loss_amount) )
        For exits (signal == -1), size will be current position size (to close).
        Apply exposure and max position size limits.

        portfolio: current portfolio
        price: current price to size against
        stop_loss_pct: fractional stop loss (e.g., 0.02)
        fixed_size: optional override number of shares
        """
        current_equity = portfolio.equity
        if current_equity <= 0:
            logging.warning("Portfolio equity non-positive: %s", current_equity)
            return None

        # Check drawdown stop
        if portfolio.max_drawdown_exceeded(self.max_drawdown):
            logging.warning("Max drawdown exceeded: %.2f > %.2f", portfolio.max_drawdown(), self.max_drawdown)
            return None

        # Check daily loss limit
        if portfolio.daily_loss_exceeded(self.daily_loss_limit):
            logging.warning("Daily loss limit exceeded: %.4f > %.4f", portfolio.daily_loss(), self.daily_loss_limit)
            return None

        if signal == -1:
            # closing position: return negative of current position to flatten
            pos = portfolio.get_position_quantity()
            if pos == 0:
                return None
            return -pos

        if signal == 1:
            # Entry sizing
            if fixed_size is not None:
                size = fixed_size
            else:
                # Determine stop distance
                if stop_loss_pct is None or stop_loss_pct <= 0:
                    # fallback percent
                    stop_loss_pct = 0.02
                stop_amount = price * stop_loss_pct
                if stop_amount <= 0:
                    logging.warning("Stop amount computed 0 or negative, rejecting order.")
                    return None
                risk_amount = current_equity * self.risk_per_trade
                raw_size = math.floor(risk_amount / stop_amount)
                size = max(0, raw_size)

            if size <= 0:
                logging.info("Calculated position size is zero; skipping trade.")
                return None

            # Apply position size limits (in currency)
            proposed_position_value = size * price
            if proposed_position_value > current_equity * self.max_position_size:
                size = math.floor((current_equity * self.max_position_size) / price)
                logging.debug("Capped size by max_position_size to %s", size)

            # Apply overall exposure limit
            total_exposure = (portfolio.current_exposure() + proposed_position_value) / current_equity
            if total_exposure > self.max_exposure:
                # reduce to fit exposure
                available_exposure_value = self.max_exposure * current_equity - portfolio.current_exposure()
                if available_exposure_value <= 0:
                    logging.info("No exposure available; skipping trade.")
                    return None
                size = math.floor(available_exposure_value / price)
                logging.debug("Capped size by max_exposure to %s", size)

            if size <= 0:
                logging.info("Final computed size is zero after applying limits.")
                return None

            return int(size)

        logging.debug("Signal neither 1 nor -1: %s", signal)
        return None


# ----------------------------
# Execution Handler (Simulator)
# ----------------------------
class ExecutionHandler:
    """
    Simulates immediate market fills at provided fill_price with commission and slippage.
    """

    def __init__(self, cfg: Config):
        self.commission_per_share = cfg.get("backtest.commission_per_share")
        self.slippage_pct = cfg.get("backtest.slippage_pct")

    def execute_order(self, order: Order, bar: pd.Series) -> Trade:
        """
        Simulate execution. For market order we'll assume fill at bar['Open'] if available or bar['Close'].
        Apply slippage and commission.
        """
        # Determine fill price: use next bar Open (provided by caller as bar)
        base_price = float(bar.get('Open', order.price) if 'Open' in bar else order.price)
        # slippage in price terms
        slippage = base_price * self.slippage_pct * (1 if order.side == 'BUY' else -1)
        fill_price = base_price + slippage
        commission = abs(order.quantity) * self.commission_per_share
        logging.info("Executed %s %d @ %.4f (slippage=%.4f commission=%.4f)", order.side, order.quantity, fill_price, slippage, commission)
        return Trade(symbol=order.symbol, quantity=order.quantity, side=order.side, price=fill_price, timestamp=order.timestamp, commission=commission)


# ----------------------------
# Portfolio and Accounting
# ----------------------------
class Portfolio:
    """
    Very lightweight portfolio for single-symbol backtest.
    Tracks cash, positions, PnL, equity, and trade log.
    """

    def __init__(self, start_cash: float):
        self.start_cash = float(start_cash)
        self.cash = float(start_cash)
        self.position = 0  # signed integer share count (positive = long)
        self.position_entry_price = 0.0  # average entry price for current position
        self._trade_history: List[Trade] = []
        self._equity_history: List[Tuple[pd.Timestamp, float]] = []
        self._daily_pnl: Dict[pd.Timestamp, float] = {}  # date -> pnl for that date
        self._peak_equity = start_cash
        self._lowest_equity_since_peak = start_cash

    def current_exposure(self) -> float:
        return abs(self.position) * (self.position_entry_price if self.position_entry_price > 0 else 0.0)

    def get_position_quantity(self) -> int:
        return int(self.position)

    def record_trade(self, trade: Trade):
        # update cash and position
        qty = int(trade.quantity)
        notional = qty * trade.price
        if trade.side == 'BUY':
            self.cash -= (notional + trade.commission)
            # update average entry price
            if self.position + qty == 0:
                self.position_entry_price = 0.0
            elif self.position == 0:
                self.position_entry_price = trade.price
            else:
                # Weighted average for adding to existing position
                existing_notional = self.position * self.position_entry_price
                new_notional = existing_notional + notional
                new_qty = self.position + qty
                if new_qty != 0:
                    self.position_entry_price = new_notional / new_qty
            self.position += qty
        elif trade.side == 'SELL':
            self.cash += (notional - trade.commission)
            # if closing or reducing position adjust entry price if crossing to net short is allowed
            if self.position - qty == 0:
                self.position_entry_price = 0.0
            self.position -= qty
        else:
            raise ValueError("Unknown trade side: %s" % trade.side)

        self._trade_history.append(trade)
        logging.debug("Trade recorded. Position=%d EntryPrice=%.4f Cash=%.2f", self.position, self.position_entry_price, self.cash)

    def update_equity(self, timestamp: pd.Timestamp, mark_price: float):
        """
        Compute current equity and record history.
        """
        position_value = self.position * mark_price
        equity = self.cash + position_value
        self._equity_history.append((timestamp, equity))
        # Update peak and drawdown trackers
        if equity > self._peak_equity:
            self._peak_equity = equity
            self._lowest_equity_since_peak = equity
        else:
            if equity < self._lowest_equity_since_peak:
                self._lowest_equity_since_peak = equity
        # Update daily pnl
        date = timestamp.normalize()
        prev_equity = self._equity_history[-2][1] if len(self._equity_history) >= 2 else self.start_cash
        pnl = equity - prev_equity
        self._daily_pnl[date] = self._daily_pnl.get(date, 0.0) + pnl
        logging.debug("Equity updated: %s -> %.2f (position_value=%.2f)", timestamp, equity, position_value)

    def equity(self) -> float:
        if not self._equity_history:
            return self.start_cash
        return self._equity_history[-1][1]

    def max_drawdown(self) -> float:
        # Simple max drawdown from equity history
        if not self._equity_history:
            return 0.0
        eqs = np.array([e for _, e in self._equity_history])
        peaks = np.maximum.accumulate(eqs)
        drawdowns = (peaks - eqs) / peaks
        return float(np.max(drawdowns)) if len(drawdowns) > 0 else 0.0

    def max_drawdown_exceeded(self, allowed: float) -> bool:
        return self.max_drawdown() > allowed

    def daily_loss(self) -> float:
        # compute latest day's loss as fraction of equity start of day
        if not self._equity_history:
            return 0.0
        # find today's date
        last_ts, last_eq = self._equity_history[-1]
        date = last_ts.normalize()
        day_start_eq = None
        # find equity at first timestamp of that date
        for ts, eq in self._equity_history:
            if ts.normalize() == date:
                if day_start_eq is None:
                    day_start_eq = eq
        if day_start_eq is None:
            day_start_eq = last_eq
        if day_start_eq <= 0:
            return 0.0
        loss = max(0.0, day_start_eq - last_eq) / day_start_eq
        return float(loss)

    def daily_loss_exceeded(self, limit: float) -> bool:
        return self.daily_loss() > limit

    def trades(self) -> List[Trade]:
        return self._trade_history

    def equity_series(self) -> pd.Series:
        if not self._equity_history:
            return pd.Series(dtype=float)
        idx = pd.DatetimeIndex([ts for ts, _ in self._equity_history])
        vals = [v for _, v in self._equity_history]
        return pd.Series(data=vals, index=idx)


# ----------------------------
# Performance Metrics
# ----------------------------
def performance_metrics(equity_series: pd.Series, start_cash: float) -> Dict[str, Any]:
    """
    Compute simple performance metrics: total return, CAGR, sharpe (daily), max drawdown.
    """
    metrics: Dict[str, Any] = {}
    if equity_series.empty:
        metrics.update({
            "total_return": 0.0,
            "cagr": 0.0,
            "sharpe": 0.0,
            "max_drawdown": 0.0
        })
        return metrics

    returns = equity_series.pct_change().fillna(0.0)
    total_return = (equity_series.iloc[-1] / start_cash) - 1.0
    days = (equity_series.index[-1] - equity_series.index[0]).days or 1
    cagr = (equity_series.iloc[-1] / start_cash) ** (365.0 / days) - 1.0
    # Sharpe ratio using daily returns and zero risk-free
    if returns.std() != 0:
        sharpe = (returns.mean() / returns.std()) * math.sqrt(252)
    else:
        sharpe = 0.0
    # Max drawdown
    peaks = equity_series.cummax()
    drawdown = ((peaks - equity_series) / peaks).max()
    metrics.update({
        "total_return": float(total_return),
        "cagr": float(cagr),
        "sharpe": float(sharpe),
        "max_drawdown": float(drawdown)
    })
    return metrics


# ----------------------------
# Backtester orchestrator
# ----------------------------
class Backtester:
    """
    Coordinates data, strategy, risk manager, execution handler and portfolio for backtesting.
    Single symbol event-driven simulation.
    """

    def __init__(self, df: pd.DataFrame, cfg: Optional[Config] = None):
        self.cfg = cfg or Config()
        setup_logging(self.cfg)
        self.data = DataHandler(df)
        self.strategy = MovingAverageCrossStrategy(self.cfg)
        self.risk_manager = RiskManager(self.cfg)
        self.exec_handler = ExecutionHandler(self.cfg)
        self.portfolio = Portfolio(start_cash=self.cfg.get("backtest.start_cash"))
        self.symbol = "SYM"
        self.logger = logging.getLogger(__name__)

    def run(self):
        self.logger.info("Starting backtest with %d bars", len(self.data))
        self.data.reset()
        # Initialize with first bar mark
        pos = 0
        while True:
            try:
                ts, bar = self.data.get_latest_bar()
            except IndexError:
                break
            history = self.data.get_slice_up_to(self.data._pos)
            # Ask strategy
            try:
                decision = self.strategy.on_bar(ts, bar, history)
            except Exception as e:
                logging.exception("Strategy error at %s: %s", ts, e)
                decision = None

            if decision:
                signal = int(decision.get('signal', 0))
                stop_loss_pct = decision.get('stop_loss_pct', None)
                fixed_size = self.cfg.get("strategy.fixed_size")
                size = self.risk_manager.validate_and_size(self.portfolio, signal, float(bar['Close']), stop_loss_pct, fixed_size)
                if size is not None and size != 0:
                    # Create order
                    side = 'BUY' if size > 0 else 'SELL'
                    order = Order(symbol=self.symbol, quantity=abs(size), side=side, price=float(bar['Close']), timestamp=ts)
                    # Execute
                    try:
                        trade = self.exec_handler.execute_order(order, bar)
                        self.portfolio.record_trade(trade)
                    except Exception as e:
                        logging.exception("Execution failed: %s", e)
                else:
                    logging.debug("Order not created (size=%s)", size)

            # Update portfolio mark-to-market with bar close
            self.portfolio.update_equity(ts, float(bar['Close']))
            # Advance data
            if not self.data.step():
                break

        # Final metrics
        eq_series = self.portfolio.equity_series()
        metrics = performance_metrics(eq_series, self.portfolio.start_cash)
        self.logger.info("Backtest finished. Metrics: %s", metrics)
        return {
            "metrics": metrics,
            "equity_curve": eq_series,
            "trades": self.portfolio.trades()
        }


# ----------------------------
# Utilities: sample data generator
# ----------------------------
def generate_synthetic_data(start: str = "2020-01-01", periods: int = 500, seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic OHLCV data as a geometric Brownian motion for testing/backtesting.
    """
    rng = np.random.RandomState(seed)
    dates = pd.date_range(start=start, periods=periods, freq='D')
    price = 100.0
    mu = 0.0002
    sigma = 0.02
    prices = []
    for _ in range(periods):
        shock = rng.normal(loc=mu, scale=sigma)
        price = price * math.exp(shock)
        prices.append(price)
    close = np.array(prices)
    open_ = close * (1 + rng.normal(0, 0.001, size=periods))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.002, size=periods)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.002, size=periods)))
    volume = rng.randint(100, 1000, size=periods)
    df = pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume}, index=dates)
    return df


# ----------------------------
# Example run for manual testing
# ----------------------------
if __name__ == "__main__":
    # Create config with sensible parameters for demonstration
    overrides = {
        "logging": {
            "level": "INFO",
            "file": "demo_backtest.log"
        },
        "backtest": {
            "start_cash": 100000.0,
            "commission_per_share": 0.01,
            "slippage_pct": 0.0005
        },
        "risk": {
            "risk_per_trade": 0.005,
            "max_drawdown": 0.25,
            "max_exposure": 1.0,
            "max_position_size": 0.5,
            "daily_loss_limit": 0.05
        },
        "strategy": {
            "fast_window": 20,
            "slow_window": 50,
            "use_atr": True,
            "atr_window": 14,
            "fixed_size": None
        }
    }
    cfg = Config(overrides=overrides)
    setup_logging(cfg)
    logger = logging.getLogger(__name__)

    logger.info("Generating synthetic data...")
    data = generate_synthetic_data(start="2020-01-01", periods=800)
    logger.info("Data generated with %d bars", len(data))

    backtester = Backtester(data, cfg=cfg)
    results = backtester.run()

    # Print summary
    metrics = results["metrics"]
    logger.info("Final Results:")
    logger.info("Total Return: %.2f%%", metrics["total_return"] * 100)
    logger.info("CAGR: %.2f%%", metrics["cagr"] * 100)
    logger.info("Sharpe: %.3f", metrics["sharpe"])
    logger.info("Max Drawdown: %.2f%%", metrics["max_drawdown"] * 100)

    trades = results["trades"]
    logger.info("Total Trades: %d", len(trades))
    for t in trades[:50]:
        logger.info("%s %s %d @ %.4f (commission=%.2f) at %s", t.side, t.symbol, t.quantity, t.price, t.commission, t.timestamp)

    # Optionally, you could save equity curve and trades to CSV for further analysis
    try:
        eq = results["equity_curve"]
        eq.to_csv("equity_curve.csv")
        trades_df = pd.DataFrame([t.__dict__ for t in trades])
        trades_df.to_csv("trades.csv", index=False)
        logger.info("Saved equity_curve.csv and trades.csv")
    except Exception:
        logger.exception("Error saving output files")