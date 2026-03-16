import os
import json
import math
import logging
import datetime as dt
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Tuple

import numpy as np
import pandas as pd


# =========================
# Configuration and Logger
# =========================

@dataclass
class Config:
    data_path: str = "data.csv"
    start_cash: float = 100000.0
    risk_per_trade: float = 0.01                 # fraction of equity to risk per trade
    target_annual_vol: float = 0.10              # portfolio target annual volatility for vol targeting
    vol_lookback: int = 20                       # lookback to estimate realised vol (days)
    atr_lookback: int = 14                       # ATR lookback for stop placement
    atr_stop_multiplier: float = 3.0             # initial stop distance in ATRs
    trailing_atr_multiplier: float = 2.0         # trailing stop distance in ATRs
    max_leverage: float = 2.0                     # cap on gross leverage
    max_position_size: float = 0.3               # maximum fraction of portfolio in any single position
    max_portfolio_drawdown: float = 0.20         # maximum drawdown before halting trading (fraction)
    drawdown_cooldown_days: int = 10             # days to remain halted after a drawdown stop
    slippage: float = 0.0005                      # simple per-trade proportional slippage
    commission: float = 0.0                       # per-trade commission (proportional)
    log_file: str = "trading_system.log"
    verbose: bool = True
    random_seed: int = 42                         # reproducibility if random decisions added

    # Derived / placeholder for loading external config:
    extras: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls, path: str = "config.json") -> "Config":
        if os.path.exists(path):
            with open(path, "r") as f:
                cfg = json.load(f)
            base = cls()
            for k, v in cfg.items():
                if hasattr(base, k):
                    setattr(base, k, v)
                else:
                    base.extras[k] = v
            return base
        else:
            return cls()


def setup_logger(cfg: Config) -> logging.Logger:
    logger = logging.getLogger("TradingSystem")
    logger.setLevel(logging.DEBUG if cfg.verbose else logging.INFO)

    # Avoid adding multiple handlers if called multiple times
    if not logger.handlers:
        fh = logging.FileHandler(cfg.log_file)
        fh.setLevel(logging.DEBUG)
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO if cfg.verbose else logging.WARNING)
        fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        fh.setFormatter(fmt)
        ch.setFormatter(fmt)
        logger.addHandler(fh)
        logger.addHandler(ch)
    return logger


# =========================
# Utilities: Indicators
# =========================

def compute_atr(df: pd.DataFrame, lookback: int = 14) -> pd.Series:
    """
    True Range and ATR calculation; expects df to have 'High','Low','Close'
    """
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(lookback, min_periods=1).mean()
    return atr.fillna(method="bfill")


def realized_volatility(series: pd.Series, lookback: int = 20) -> pd.Series:
    # daily returns volatility (annualized)
    returns = series.pct_change().fillna(0)
    rolling_std = returns.rolling(lookback, min_periods=1).std()
    ann = rolling_std * np.sqrt(252)
    return ann.fillna(method="bfill")


# =========================
# Strategy (Signal Generation)
# =========================

class Strategy:
    """
    Keeps a clear API: generate_signals(data, t) => dict(symbol->signal)
    Signal values: 1 (long), -1 (short), 0 (flat)
    The current implementation is a simple mean-reversion / momentum hybrid example
    with volatility-adjusted entry thresholds. This can be replaced while maintaining API.
    """
    def __init__(self, cfg: Config, logger: logging.Logger):
        self.cfg = cfg
        self.logger = logger
        np.random.seed(self.cfg.random_seed)

    def generate_signals(self, data: pd.DataFrame, current_idx: int) -> Dict[str, int]:
        """
        data: DataFrame with index as dates and columns for 'Close' (or per-symbol if multi)
        current_idx: integer index (position in DataFrame) to make today's decision
        Returns a dictionary with a single symbol "ASSET" in this simplified architecture
        """
        # For architecture consistency, we treat single-asset (ASSET) data
        if current_idx <= 0:
            return {"ASSET": 0}

        # Use lookback windows for momentum and mean reversion
        short = 10
        long = 50
        closes = data["Close"].iloc[: current_idx + 1]
        if len(closes) < long:
            return {"ASSET": 0}

        sma_short = closes.rolling(short).mean().iloc[-1]
        sma_long = closes.rolling(long).mean().iloc[-1]
        vol = realized_volatility(closes, self.cfg.vol_lookback).iloc[-1]

        # Basic hybrid rule:
        # If short SMA > long SMA by a small margin -> momentum (go long)
        # If short SMA < long SMA by a small margin -> momentum (go short)
        # If price deviates significantly from short-term mean (z-score) -> mean reversion
        price = closes.iloc[-1]
        ma_short_series = closes.rolling(short).mean()
        ma_short = ma_short_series.iloc[-1]
        ma_short_std = closes[-short:].std() if len(closes) >= short else closes.std()

        if ma_short_std == 0:
            z = 0.0
        else:
            z = (price - ma_short) / ma_short_std

        signal = 0
        # Momentum bias based on SMA cross
        if sma_short > sma_long and abs(z) < 1.5:
            signal = 1
        elif sma_short < sma_long and abs(z) < 1.5:
            signal = -1
        # Mean reversion override: if price is extreme, take opposite
        if z > 2.0:
            signal = -1
        elif z < -2.0:
            signal = 1

        self.logger.debug(f"Signal gen @ idx {current_idx}: price={price:.2f}, sma_short={sma_short:.2f}, "
                          f"sma_long={sma_long:.2f}, z={z:.2f}, vol={vol:.4f}, signal={signal}")
        return {"ASSET": signal}


# =========================
# Risk Manager and Sizer
# =========================

class RiskManager:
    def __init__(self, cfg: Config, logger: logging.Logger):
        self.cfg = cfg
        self.logger = logger

    def position_size(self,
                      equity: float,
                      price: float,
                      atr: float,
                      signal: int,
                      asset_vol: float) -> Tuple[int, float]:
        """
        Determine number of units to trade and estimated dollar exposure.
        Uses volatility targeting + fixed fraction risk per trade with ATR-based stop.
        Returns (signed_units, exposure_dollars)
        """
        if signal == 0:
            return 0, 0.0

        # Protect against zero atr or vol
        atr = max(atr, 1e-8)
        asset_vol = max(asset_vol, 1e-8)

        # Target portfolio volatility in dollar terms (annualized)
        # For a single asset, desired exposure = equity * (target_annual_vol / asset_vol)
        vol_target_exposure = equity * min(self.cfg.max_leverage,
                                          self.cfg.target_annual_vol / asset_vol)

        # Determine risk per trade in dollars
        risk_dollars = equity * self.cfg.risk_per_trade

        # Stop distance in dollars per share based on ATR
        stop_distance = atr * self.cfg.atr_stop_multiplier

        # If stop_distance would be 0 (degenerate), fallback to small percent
        if stop_distance <= 0:
            stop_distance = price * 0.01

        # Units by risk sizing (how many shares would risk approximately risk_dollars)
        units_by_risk = math.floor(risk_dollars / stop_distance) if stop_distance > 0 else 0

        # Units by volatility targeting (cap exposure)
        units_by_vol = math.floor(vol_target_exposure / price) if price > 0 else 0

        # Choose the more conservative sizing
        units = min(units_by_risk, units_by_vol)

        # Enforce a maximum position size in portfolio terms
        max_position_dollars = equity * self.cfg.max_position_size
        max_units_by_position = math.floor(max_position_dollars / price) if price > 0 else 0
        units = min(units, max_units_by_position)

        # Ensure at least zero
        units = max(0, units)
        signed_units = units * np.sign(signal)

        exposure = signed_units * price
        self.logger.debug(f"Position sizing: equity={equity:.2f}, price={price:.2f}, atr={atr:.4f}, "
                          f"asset_vol={asset_vol:.4f}, vol_target_exposure={vol_target_exposure:.2f}, "
                          f"units_by_risk={units_by_risk}, units_by_vol={units_by_vol}, "
                          f"max_units_by_position={max_units_by_position}, final_units={signed_units}")
        return int(signed_units), exposure


# =========================
# Simple Backtester / Executor
# =========================

class TradingSystem:
    def __init__(self, cfg: Config, strategy: Strategy, logger: logging.Logger):
        self.cfg = cfg
        self.strategy = strategy
        self.logger = logger
        self.risk_manager = RiskManager(cfg, logger)

        # State
        self.cash = cfg.start_cash
        self.equity = cfg.start_cash
        self.position = 0                             # signed units
        self.avg_entry_price = 0.0
        self.halted_until: Optional[pd.Timestamp] = None

        # Performance records
        self.trades: List[Dict[str, Any]] = []
        self.daily_equity: List[Dict[str, Any]] = []
        self.max_equity = self.equity
        self.peak_date: Optional[pd.Timestamp] = None

    def run_backtest(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Main loop over data. data is expected to have DateTime index and columns: Close, High, Low
        Returns daily equity curve as DataFrame
        """
        data = data.copy().sort_index()
        data.index = pd.to_datetime(data.index)

        # Precompute indicators
        atr_series = compute_atr(data, lookback=self.cfg.atr_lookback)
        vol_series = realized_volatility(data["Close"], self.cfg.vol_lookback)

        n = len(data)
        dates = data.index

        # Reset state
        self.cash = self.cfg.start_cash
        self.equity = self.cfg.start_cash
        self.position = 0
        self.avg_entry_price = 0.0
        self.max_equity = self.equity
        self.trades.clear()
        self.daily_equity.clear()
        self.halted_until = None
        self.peak_date = dates[0] if n > 0 else None

        for i in range(n):
            today = dates[i]
            price = data["Close"].iloc[i]
            high = data["High"].iloc[i]
            low = data["Low"].iloc[i]
            atr = atr_series.iloc[i]
            asset_vol = vol_series.iloc[i]

            # Check if halted by drawdown stop
            if self.halted_until is not None and today < self.halted_until:
                self.logger.info(f"Trading halted until {self.halted_until.date()} due to drawdown. Days remaining: {(self.halted_until - today).days}")
                signal = 0
            else:
                signals = self.strategy.generate_signals(data, i)
                signal = signals.get("ASSET", 0)

            # If no signal and we have a position, consider trailing stop based on ATR
            exit_for_stop = False
            if self.position != 0:
                # trailing stop logic
                if self.position > 0:
                    trailing_stop = self.avg_entry_price - atr * self.cfg.trailing_atr_multiplier
                    if low <= trailing_stop:
                        exit_for_stop = True
                        self.logger.debug(f"Trailing stop hit (long). low={low:.2f} <= trailing_stop={trailing_stop:.2f}")
                else:
                    trailing_stop = self.avg_entry_price + atr * self.cfg.trailing_atr_multiplier
                    if high >= trailing_stop:
                        exit_for_stop = True
                        self.logger.debug(f"Trailing stop hit (short). high={high:.2f} >= trailing_stop={trailing_stop:.2f}")

            # If signal is zero or opposing, we may reduce/close position
            desired_signal = signal
            if exit_for_stop:
                desired_signal = 0

            # Position sizing
            signed_units, exposure = self.risk_manager.position_size(self.equity, price, atr, desired_signal, asset_vol)

            # Determine order to execute: convert from current position to desired
            order_units = signed_units - self.position

            # Execute order at today's close price (simple executor), apply slippage/commission
            if order_units != 0:
                # Slippage model is proportional to trade value (cfg.slippage) times side
                trade_value = abs(order_units) * price
                slippage_cost = trade_value * self.cfg.slippage
                commission_cost = trade_value * self.cfg.commission

                # Update cash and position
                # For buy: cash decreases; for sell: cash increases
                cash_change = -order_units * price - slippage_cost - commission_cost
                prev_cash = self.cash
                self.cash += cash_change

                # Update position average entry price if increasing position
                if np.sign(order_units) == np.sign(self.position) and self.position != 0:
                    # increasing same-direction position: update average entry
                    new_units = self.position + order_units
                    if new_units != 0:
                        self.avg_entry_price = (self.avg_entry_price * abs(self.position) + price * abs(order_units)) / abs(new_units)
                elif self.position == 0 and order_units != 0:
                    # opening new position
                    self.avg_entry_price = price
                else:
                    # reducing or reversing a position
                    if np.sign(order_units) != np.sign(self.position):
                        # Closing entirely and possibly reversing
                        if abs(order_units) >= abs(self.position):
                            # realize P&L on closed portion
                            closed_units = -self.position
                            pnl = closed_units * (price - self.avg_entry_price) * -1  # closed from previous position
                            # We already updated cash via price * order_units; PnL is reflected in cash implicitly
                            self.avg_entry_price = price if order_units != 0 else 0.0
                        else:
                            # partial close; average remains
                            pass

                prev_position = self.position
                self.position += int(order_units)

                trade = {
                    "date": today,
                    "units": int(order_units),
                    "price": price,
                    "trade_value": order_units * price,
                    "slippage": slippage_cost,
                    "commission": commission_cost,
                    "cash_before": prev_cash,
                    "cash_after": self.cash,
                    "position_before": prev_position,
                    "position_after": self.position,
                    "avg_entry_price": self.avg_entry_price
                }
                self.trades.append(trade)
                self.logger.info(f"Executed trade on {today.date()}: units={order_units}, price={price:.2f}, "
                                 f"cash_change={cash_change:.2f}, position={self.position}")

            # Mark-to-market equity
            mtm = self.position * price
            self.equity = self.cash + mtm
            self.daily_equity.append({"date": today, "equity": self.equity, "cash": self.cash, "position": self.position, "price": price})

            # Update peak equity for drawdown calculations
            if self.equity > self.max_equity:
                self.max_equity = self.equity
                self.peak_date = today

            drawdown = (self.max_equity - self.equity) / max(1.0, self.max_equity)
            if drawdown >= self.cfg.max_portfolio_drawdown:
                # Halt trading for a cooldown period
                self.halted_until = today + pd.Timedelta(days=self.cfg.drawdown_cooldown_days)
                self.logger.warning(f"Max drawdown exceeded ({drawdown:.2%}) on {today.date()}. Halting trading until {self.halted_until.date()}")
                # Close positions if any to prevent further deterioration
                if self.position != 0:
                    # close fully at next available close (we model immediate liquidation today)
                    close_units = -self.position
                    trade_value = abs(close_units) * price
                    slippage_cost = trade_value * self.cfg.slippage
                    commission_cost = trade_value * self.cfg.commission
                    self.cash += -close_units * price - slippage_cost - commission_cost
                    self.logger.info(f"Emergency close on {today.date()}: units={close_units}, price={price:.2f}")
                    self.trades.append({
                        "date": today,
                        "units": int(close_units),
                        "price": price,
                        "trade_value": close_units * price,
                        "slippage": slippage_cost,
                        "commission": commission_cost,
                        "cash_before": None,
                        "cash_after": self.cash,
                        "position_before": self.position,
                        "position_after": 0,
                        "avg_entry_price": 0.0
                    })
                    self.position = 0
                    self.avg_entry_price = 0.0
                    mtm = 0.0
                    self.equity = self.cash + mtm
                    self.daily_equity[-1]["equity"] = self.equity
                    self.daily_equity[-1]["position"] = 0

        equity_df = pd.DataFrame(self.daily_equity).set_index("date")
        equity_df.index = pd.to_datetime(equity_df.index)
        return equity_df

    # Evaluation metrics
    @staticmethod
    def compute_performance(equity_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Compute performance metrics: total return, CAGR, annualized vol, Sharpe, max drawdown, Sortino
        """
        if equity_df.empty:
            return {}

        eq = equity_df["equity"]
        returns = eq.pct_change().fillna(0)
        total_return = eq.iloc[-1] / eq.iloc[0] - 1.0
        days = (eq.index[-1] - eq.index[0]).days or 1
        years = days / 365.25
        cagr = (1 + total_return) ** (1 / years) - 1 if years > 0 else np.nan
        ann_vol = returns.std() * np.sqrt(252)
        sharpe = (returns.mean() / returns.std() * np.sqrt(252)) if returns.std() != 0 else np.nan

        # Sortino ratio
        neg_returns = returns[returns < 0]
        downside_std = neg_returns.std() * np.sqrt(252) if not neg_returns.empty else 0.0
        sortino = (returns.mean() * np.sqrt(252) / downside_std) if downside_std != 0 else np.nan

        # Max drawdown
        running_max = eq.cummax()
        drawdown = (running_max - eq) / running_max
        max_drawdown = drawdown.max()

        return {
            "total_return": total_return,
            "cagr": cagr,
            "annual_vol": ann_vol,
            "sharpe": sharpe,
            "sortino": sortino,
            "max_drawdown": max_drawdown,
            "start_equity": eq.iloc[0],
            "end_equity": eq.iloc[-1],
            "trading_days": len(eq)
        }


# =========================
# Helper to load data (preserve architecture)
# =========================

def load_data(path: str) -> pd.DataFrame:
    """
    Loads CSV with Date, Open, High, Low, Close, Volume (at minimum must have High, Low, Close).
    Ensures the index is datetime and sorted.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found at path: {path}")
    df = pd.read_csv(path, parse_dates=True, index_col=0)
    required = {"High", "Low", "Close"}
    if not required.issubset(df.columns):
        raise ValueError(f"Data must contain columns: {required}")
    df = df.sort_index()
    return df


# =========================
# Main Entrypoint (preserve logging & config)
# =========================

def main(config_path: str = "config.json"):
    cfg = Config.load(config_path)
    logger = setup_logger(cfg)
    logger.info("Starting trading system backtest")
    logger.info(f"Config: {cfg}")

    try:
        data = load_data(cfg.data_path)
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        raise

    strategy = Strategy(cfg, logger)
    system = TradingSystem(cfg, strategy, logger)
    equity_df = system.run_backtest(data)
    perf = TradingSystem.compute_performance(equity_df)

    logger.info("Backtest complete. Performance summary:")
    for k, v in perf.items():
        logger.info(f"  {k}: {v}")

    # Preserve logs of trades and equity curve
    trades_df = pd.DataFrame(system.trades)
    if not trades_df.empty:
        trades_file = "trades.csv"
        trades_df.to_csv(trades_file, index=False)
        logger.info(f"Saved trades to {trades_file}")

    equity_file = "equity_curve.csv"
    equity_df.to_csv(equity_file)
    logger.info(f"Saved equity curve to {equity_file}")

    return perf, equity_df, trades_df


if __name__ == "__main__":
    # Run with default config path. Keeps architecture consistent and logging intact.
    perf, equity_df, trades_df = main()