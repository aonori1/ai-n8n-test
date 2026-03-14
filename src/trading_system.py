import logging
import math
import os
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import List, Dict, Optional

import numpy as np
import pandas as pd


# -------------------------
# Configuration (preserved & extendable)
# -------------------------
CONFIG = {
    "initial_capital": 100000.0,
    "max_leverage": 1.0,  # maximum gross exposure (e.g., 1.0 = 100% of capital)
    "risk_per_trade": 0.01,  # fraction of equity to risk per trade (1%)
    "target_volatility_annual": 0.10,  # target annual volatility (10%) for volatility scaling
    "volatility_lookback": 20,  # lookback for realized volatility / ATR
    "atr_period": 14,  # ATR for stop sizing
    "stop_atr_multiplier": 3.0,  # initial stop distance in ATR multiples
    "trailing_stop_atr_multiplier": 2.0,  # trailing stop distance in ATR multiples
    "sma_fast": 20,
    "sma_slow": 100,
    "ema_trend": 200,
    "momentum_lookback": 5,
    "min_signal_confirmation": 2,  # require signal to persist for N bars
    "max_concurrent_positions": 1,  # single instrument system for simplicity
    "max_drawdown_limit": 0.20,  # if drawdown exceeds 20%, stop further trading
    "commission_per_trade": 1.0,  # flat commission or per trade fee
    "slippage": 0.0005,  # proportion slippage
    "verbose_logging": True,
    "log_file": "trading_system.log",
}


# -------------------------
# Logging (preserved)
# -------------------------
logger = logging.getLogger("TradingSystem")
logger.setLevel(logging.DEBUG if CONFIG["verbose_logging"] else logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

# Console handler
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG if CONFIG["verbose_logging"] else logging.INFO)
ch.setFormatter(formatter)
if not logger.handlers:
    logger.addHandler(ch)

# File handler
fh = logging.FileHandler(CONFIG["log_file"])
fh.setLevel(logging.DEBUG)
fh.setFormatter(formatter)
logger.addHandler(fh)


# -------------------------
# Data classes
# -------------------------
@dataclass
class Position:
    entry_date: pd.Timestamp
    side: int  # +1 long, -1 short
    entry_price: float
    size: float  # number of shares / contracts
    stop_price: float
    trailing_stop: Optional[float] = None
    exit_date: Optional[pd.Timestamp] = None
    exit_price: Optional[float] = None
    pnl: Optional[float] = None


# -------------------------
# Utility functions
# -------------------------
def calculate_atr(df: pd.DataFrame, period: int) -> pd.Series:
    high_low = df["High"] - df["Low"]
    high_close_prev = (df["High"] - df["Close"].shift(1)).abs()
    low_close_prev = (df["Low"] - df["Close"].shift(1)).abs()
    tr = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
    atr = tr.rolling(period, min_periods=1).mean()
    return atr


def annualize_volatility(returns: pd.Series, periods_per_year: int = 252) -> float:
    if len(returns) < 2:
        return 0.0
    return returns.std() * math.sqrt(periods_per_year)


def max_drawdown(series: pd.Series) -> float:
    cummax = series.cummax()
    drawdown = (series - cummax) / cummax
    return drawdown.min()


# -------------------------
# Trading System class
# -------------------------
class TradingSystem:
    def __init__(self, data: pd.DataFrame, config: Dict, logger: logging.Logger):
        self.data = data.copy().reset_index(drop=False)
        if "Date" not in self.data.columns:
            # try to infer index as date
            self.data.rename(columns={self.data.columns[0]: "Date"}, inplace=True)
        self.data["Date"] = pd.to_datetime(self.data["Date"])
        self.config = config
        self.logger = logger

        # State
        self.equity = config["initial_capital"]
        self.cash = config["initial_capital"]
        self.positions: List[Position] = []
        self.closed_positions: List[Position] = []
        self.equity_curve: List[Dict] = []  # record date, equity, cash, position_value
        self.current_position: Optional[Position] = None
        self.signal_persistence = 0

        # Precompute indicators
        self._prepare_indicators()

    def _prepare_indicators(self):
        df = self.data
        df["SMA_fast"] = df["Close"].rolling(self.config["sma_fast"], min_periods=1).mean()
        df["SMA_slow"] = df["Close"].rolling(self.config["sma_slow"], min_periods=1).mean()
        df["EMA_trend"] = df["Close"].ewm(span=self.config["ema_trend"], adjust=False).mean()
        df["Momentum"] = df["Close"].pct_change(self.config["momentum_lookback"])
        df["ATR"] = calculate_atr(df, self.config["atr_period"])
        # daily returns for realized volatility
        df["Return"] = df["Close"].pct_change().fillna(0)
        df["RealizedVol"] = df["Return"].rolling(self.config["volatility_lookback"], min_periods=2).std()
        self.data = df

    def generate_signal(self, idx: int) -> int:
        """
        Generate a discrete signal: +1 (long), 0 (flat), -1 (short)
        Improve entries by requiring:
          - trend filter using EMA_trend
          - SMA crossover confirmation
          - momentum sign
        """
        row = self.data.loc[idx]
        # trend filter
        price = row["Close"]
        trend = 1 if price > row["EMA_trend"] else -1

        sma_fast = row["SMA_fast"]
        sma_slow = row["SMA_slow"]
        momentum = row["Momentum"]

        # Basic SMA crossover signal
        if sma_fast > sma_slow and momentum > 0 and trend > 0:
            return 1
        elif sma_fast < sma_slow and momentum < 0 and trend < 0:
            return -1
        else:
            return 0

    def position_size(self, price: float, atr: float, side: int) -> float:
        """
        Position sizing uses volatility targeting with a cap and risk-per-trade.
        Size is computed so that expected ATR loss to stop * price * size <= risk_per_trade * equity
        and also scales to target annual volatility.
        """
        equity = self.equity
        risk_per_trade = self.config["risk_per_trade"]
        # If ATR is zero (flat), fallback to tiny size
        if atr <= 0 or price <= 0:
            return 0.0

        # 1) Stop-based sizing: stop distance = stop_atr_multiplier * ATR
        stop_distance = self.config["stop_atr_multiplier"] * atr
        risk_amount = equity * risk_per_trade
        size_stop_based = risk_amount / (stop_distance * price)

        # 2) Volatility target sizing: position volatility = ATR/price (approx), scale to target
        target_vol = self.config["target_volatility_annual"]
        # approximate current daily vol
        daily_vol = atr / price if price > 0 else 0.0
        if daily_vol > 0:
            # scaling factor to reach target annual vol (sqrt(252))
            scale = target_vol / (daily_vol * math.sqrt(252))
            size_vol_based = (equity * self.config["max_leverage"] * scale) / price
        else:
            size_vol_based = size_stop_based

        # Choose conservative min of methods, apply max leverage and ensure integer number of shares/contracts
        size = min(abs(size_stop_based), abs(size_vol_based))
        max_size = (equity * self.config["max_leverage"]) / price
        size = max(min(size, max_size), 0.0)

        # enforce at least zero
        return math.floor(size)  # integer shares/contracts

    def enter_position(self, idx: int, side: int):
        if self.current_position is not None:
            # we only allow one position in this simple system
            return

        row = self.data.loc[idx]
        price = row["Close"] * (1 + self.config["slippage"] * side)  # slippage applied to side
        atr = row["ATR"]
        size = self.position_size(price, atr, side)
        if size <= 0:
            self.logger.debug(f"Skipping entry at {row['Date']} due to zero size (atr={atr}, price={price}).")
            return

        stop_distance = self.config["stop_atr_multiplier"] * atr
        if side == 1:
            stop_price = price - stop_distance
            trailing_stop = price - self.config["trailing_stop_atr_multiplier"] * atr
        else:
            stop_price = price + stop_distance
            trailing_stop = price + self.config["trailing_stop_atr_multiplier"] * atr

        position = Position(
            entry_date=row["Date"],
            side=side,
            entry_price=price,
            size=size,
            stop_price=stop_price,
            trailing_stop=trailing_stop,
        )
        # Allocate cash (we assume full margin; flat commission)
        position_value = price * size
        if position_value > self.cash + 1e-8:  # simple cash check without leverage facilities
            # If not enough cash, scale down size
            affordable_size = math.floor(self.cash / price)
            if affordable_size <= 0:
                self.logger.debug(f"Not enough cash to enter position at {row['Date']} (cash={self.cash}, price={price}).")
                return
            position.size = affordable_size
            position_value = price * affordable_size

        # Deduct cash for position (conservative)
        self.cash -= position_value + self.config["commission_per_trade"]
        self.current_position = position
        self.positions.append(position)
        self.logger.info(f"ENTER {row['Date']} side={side} price={price:.2f} size={position.size} stop={position.stop_price:.2f}")

    def exit_position(self, idx: int, reason: str):
        if self.current_position is None:
            return
        row = self.data.loc[idx]
        price = row["Close"] * (1 - self.config["slippage"] * self.current_position.side)  # slippage on exit
        # Close
        pnl = (price - self.current_position.entry_price) * self.current_position.size * self.current_position.side
        self.cash += price * self.current_position.size - self.config["commission_per_trade"]
        self.current_position.exit_date = row["Date"]
        self.current_position.exit_price = price
        self.current_position.pnl = pnl
        self.closed_positions.append(self.current_position)
        self.logger.info(f"EXIT {row['Date']} side={self.current_position.side} price={price:.2f} size={self.current_position.size} pnl={pnl:.2f} reason={reason}")
        self.current_position = None

    def update_trailing_stop(self, idx: int):
        if self.current_position is None:
            return
        row = self.data.loc[idx]
        price = row["Close"]
        atr = row["ATR"]
        if atr <= 0:
            return
        # For long positions, trailing stop moves up with price; for short, moves down
        if self.current_position.side == 1:
            new_trail = price - self.config["trailing_stop_atr_multiplier"] * atr
            if new_trail > (self.current_position.trailing_stop or -np.inf):
                self.current_position.trailing_stop = new_trail
        else:
            new_trail = price + self.config["trailing_stop_atr_multiplier"] * atr
            if new_trail < (self.current_position.trailing_stop or np.inf):
                self.current_position.trailing_stop = new_trail

    def check_stop_conditions(self, idx: int):
        """
        Check both initial stop and trailing stop. Exit if any is hit.
        """
        if self.current_position is None:
            return
        row = self.data.loc[idx]
        low = row["Low"]
        high = row["High"]

        # For long
        if self.current_position.side == 1:
            # initial stop
            if low <= self.current_position.stop_price:
                self.exit_position(idx, reason="stop_hit_initial")
                return
            # trailing stop
            if self.current_position.trailing_stop is not None and low <= self.current_position.trailing_stop:
                self.exit_position(idx, reason="stop_hit_trailing")
                return
        else:
            # short
            if high >= self.current_position.stop_price:
                self.exit_position(idx, reason="stop_hit_initial")
                return
            if self.current_position.trailing_stop is not None and high >= self.current_position.trailing_stop:
                self.exit_position(idx, reason="stop_hit_trailing")
                return

    def record_equity(self, idx: int):
        row = self.data.loc[idx]
        pos_value = 0.0
        if self.current_position is not None:
            # mark to market
            pos_value = self.current_position.entry_price * self.current_position.size * self.current_position.side
            # better: use current price
            pos_value = self.current_position.size * row["Close"] * self.current_position.side
        equity = self.cash + pos_value
        self.equity = equity
        self.equity_curve.append({"Date": row["Date"], "Equity": equity, "Cash": self.cash, "PositionValue": pos_value})

    def should_halt_trading(self) -> bool:
        # Stop trading if drawdown crossed threshold
        if not self.equity_curve:
            return False
        equity_series = pd.Series([x["Equity"] for x in self.equity_curve])
        dd = -max_drawdown(equity_series)
        if dd >= self.config["max_drawdown_limit"]:
            self.logger.warning(f"Halting trading: drawdown {dd:.2%} >= limit {self.config['max_drawdown_limit']:.2%}")
            return True
        return False

    def run_backtest(self):
        df = self.data
        n = len(df)
        self.logger.info("Starting backtest")
        for idx in range(n):
            date = df.loc[idx, "Date"]

            # Update trailing stop based on latest price
            self.update_trailing_stop(idx)

            # Check stop hits
            self.check_stop_conditions(idx)

            # Record equity before new entries/exits at this bar's close
            self.record_equity(idx)

            # Risk check
            if self.should_halt_trading():
                break

            # Generate signal
            signal = self.generate_signal(idx)

            # Require persistence for entries (reduce whipsaw)
            if signal != 0:
                # If same direction as previous signal, increment persistence, else reset
                last_sign = getattr(self, "_last_signal", 0)
                if signal == last_sign:
                    self.signal_persistence += 1
                else:
                    self.signal_persistence = 1
                self._last_signal = signal
            else:
                self.signal_persistence = 0
                self._last_signal = 0

            # Entry logic
            if self.current_position is None and abs(signal) > 0 and self.signal_persistence >= self.config["min_signal_confirmation"]:
                # enter with signal direction
                self.enter_position(idx, signal)

            # Exit logic based on reversed confirmed signal
            if self.current_position is not None:
                # if reversed signal is persistent, exit
                if signal == -self.current_position.side and self.signal_persistence >= self.config["min_signal_confirmation"]:
                    self.exit_position(idx, reason="signal_reversal")

        # At end of backtest, close any open positions at last close
        if self.current_position is not None:
            idx = len(self.data) - 1
            self.exit_position(idx, reason="end_of_backtest")
            self.record_equity(idx)

        self.logger.info("Backtest complete")
        return self.results()

    def results(self) -> Dict:
        df_equity = pd.DataFrame(self.equity_curve).set_index("Date")
        df_equity["Returns"] = df_equity["Equity"].pct_change().fillna(0)
        total_return = (df_equity["Equity"].iloc[-1] / df_equity["Equity"].iloc[0]) - 1 if len(df_equity) >= 2 else 0.0
        annual_factor = 252 / len(df_equity) * len(df_equity) if len(df_equity) > 0 else 1
        # Compute annualized return and volatility robustly
        ann_return = (1 + total_return) ** (252 / len(df_equity)) - 1 if len(df_equity) > 1 else 0.0
        ann_vol = annualize_volatility(df_equity["Returns"], periods_per_year=252)
        sharpe = (ann_return / ann_vol) if ann_vol > 0 else 0.0
        # Sortino: downside deviation
        neg_returns = df_equity["Returns"].loc[df_equity["Returns"] < 0]
        downside_dev = neg_returns.std() * math.sqrt(252) if len(neg_returns) > 0 else 0.0
        sortino = (ann_return / downside_dev) if downside_dev > 0 else 0.0
        dd = -max_drawdown(df_equity["Equity"])

        trades = pd.DataFrame([asdict(p) for p in self.closed_positions])
        num_trades = len(trades)
        wins = trades[trades["pnl"] > 0] if not trades.empty else trades
        win_rate = len(wins) / num_trades if num_trades > 0 else 0.0
        avg_win = wins["pnl"].mean() if not wins.empty else 0.0
        avg_loss = trades[trades["pnl"] <= 0]["pnl"].mean() if not trades.empty else 0.0

        results = {
            "total_return": total_return,
            "annual_return": ann_return,
            "annual_volatility": ann_vol,
            "sharpe": sharpe,
            "sortino": sortino,
            "max_drawdown": dd,
            "num_trades": num_trades,
            "win_rate": win_rate,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "equity_curve": df_equity,
            "trades": trades,
        }

        self.logger.info(f"Results: Total Return: {total_return:.2%}, Annual Return: {ann_return:.2%}, "
                         f"Vol: {ann_vol:.2%}, Sharpe: {sharpe:.2f}, Sortino: {sortino:.2f}, Drawdown: {dd:.2%}, "
                         f"Trades: {num_trades}, WinRate: {win_rate:.2%}")

        return results


# -------------------------
# Entry point for running/backtesting
# -------------------------
def load_data_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV file not found: {path}")
    df = pd.read_csv(path)
    # Expect columns: Date, Open, High, Low, Close, Volume (at least)
    expected = {"Date", "Open", "High", "Low", "Close"}
    if not expected.issubset(set(df.columns)):
        raise ValueError(f"CSV missing required columns. Found: {df.columns}")
    df = df.loc[:, ["Date", "Open", "High", "Low", "Close"]]
    return df


if __name__ == "__main__":
    # Simple example usage; replace 'data.csv' with your historical data file
    DATA_FILE = "data.csv"
    try:
        data = load_data_csv(DATA_FILE)
    except Exception as e:
        logger.error(f"Unable to load data file {DATA_FILE}: {e}")
        # Create a minimal synthetic dataset to allow code to run (useful for testing)
        dates = pd.date_range(end=datetime.today(), periods=300, freq="B")
        price = np.cumprod(1 + np.random.normal(0, 0.001, size=len(dates))) * 100
        data = pd.DataFrame({
            "Date": dates,
            "Open": price * (1 + np.random.normal(0, 0.001, size=len(dates))),
            "High": price * (1 + np.abs(np.random.normal(0, 0.002, size=len(dates)))),
            "Low": price * (1 - np.abs(np.random.normal(0, 0.002, size=len(dates)))),
            "Close": price,
        })
        logger.info("Using synthetic data for demonstration.")

    ts = TradingSystem(data, CONFIG, logger)
    results = ts.run_backtest()

    # Example: save results to CSVs while preserving logging/config
    out_dir = "backtest_results"
    os.makedirs(out_dir, exist_ok=True)
    results["equity_curve"].to_csv(os.path.join(out_dir, "equity_curve.csv"))
    if not results["trades"].empty:
        results["trades"].to_csv(os.path.join(out_dir, "trades.csv"), index=False)
    # Save config snapshot
    pd.Series(CONFIG).to_csv(os.path.join(out_dir, "config_snapshot.csv"))
    logger.info(f"Backtest artifacts written to {out_dir}")