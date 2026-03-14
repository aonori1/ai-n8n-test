import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# -------------------------------------------------------------------
# Configuration (preserved / extended)
# -------------------------------------------------------------------
CONFIG = {
    "symbol": "EURUSD",
    "timeframe": "1H",
    "initial_capital": 100000.0,
    "risk_per_trade": 0.01,           # fraction of equity to risk per trade
    "max_drawdown": 0.20,             # stop trading if drawdown exceeds this fraction
    "max_positions": 1,               # max concurrent positions (keeps things conservative)
    "leverage": 2.0,                  # max leverage applied to position not per trade
    "ema_short": 20,
    "ema_long": 50,
    "rsi_period": 14,
    "rsi_oversold": 30,
    "rsi_overbought": 70,
    "atr_period": 14,
    "stop_atr_mult": 1.5,             # initial stop distance in ATR multiples
    "trail_atr_mult": 1.0,            # trailing stop distance in ATR multiples
    "take_profit_atr_mult": 3.0,      # take-profit distance in ATR multiples
    "min_position_usd": 1000,         # avoid creating tiny positions
    "cooldown_bars_after_loss": 10,   # skip trading after a losing trade for this many bars
    "smoothing_ema": 5,               # smoothing for signals to reduce whipsaw
    "verbose_logging": True,
}

# -------------------------------------------------------------------
# Logging (preserved)
# -------------------------------------------------------------------
logger = logging.getLogger("TradingSystem")
logger.setLevel(logging.DEBUG if CONFIG["verbose_logging"] else logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG if CONFIG["verbose_logging"] else logging.INFO)
formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
ch.setFormatter(formatter)
if not logger.handlers:
    logger.addHandler(ch)


# -------------------------------------------------------------------
# Utility indicators implemented without external dependencies
# -------------------------------------------------------------------
def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(period, min_periods=1).mean()


def rsi(series: pd.Series, period: int) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0.0)
    down = -1.0 * delta.clip(upper=0.0)
    ma_up = up.ewm(alpha=1.0 / period, adjust=False).mean()
    ma_down = down.ewm(alpha=1.0 / period, adjust=False).mean()
    rs = ma_up / (ma_down + 1e-12)
    return 100 - (100 / (1 + rs))


def atr(df: pd.DataFrame, period: int) -> pd.Series:
    # df must contain high, low, close
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


# -------------------------------------------------------------------
# Position data structure
# -------------------------------------------------------------------
@dataclass
class Position:
    entry_price: float
    size: float                 # positive = long dollars (or units) depending on implementation
    stop_price: float
    take_profit: Optional[float] = None
    entry_index: int = 0
    direction: int = 1          # +1 long, -1 short
    peak_unrealized: float = field(default_factory=lambda: -np.inf)


# -------------------------------------------------------------------
# Trading System class (architecture preserved)
# -------------------------------------------------------------------
class TradingSystem:
    def __init__(self, config: Dict):
        self.config = config.copy()
        self.capital = config["initial_capital"]
        self.equity_curve = []  # track equity over time
        self.positions: List[Position] = []
        self.open_orders = []
        self.cooldown = 0
        self.max_equity = self.capital
        self.last_trade_profit = 0.0

        # preserve logging
        logger.info("TradingSystem initialized with config: %s", self.config)

    def compute_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute indicators and smoothed signals. Returns df with added columns.
        """
        df = df.copy()
        df["EMA_short"] = ema(df["Close"], self.config["ema_short"])
        df["EMA_long"] = ema(df["Close"], self.config["ema_long"])
        df["Signal_raw"] = 0
        df["Signal_raw"] = np.where(df["EMA_short"] > df["EMA_long"], 1, np.where(df["EMA_short"] < df["EMA_long"], -1, 0))
        # Smooth the signal to reduce noise (small moving average on signal)
        df["Signal"] = ema(df["Signal_raw"].astype(float), self.config["smoothing_ema"])
        # RSI filter
        df["RSI"] = rsi(df["Close"], self.config["rsi_period"])
        # ATR for volatility and stops
        df["ATR"] = atr(df, self.config["atr_period"])
        logger.debug("Indicators computed (EMA, RSI, ATR).")
        return df

    def can_open_new_position(self) -> bool:
        if len(self.positions) >= self.config["max_positions"]:
            logger.debug("Max positions reached: %d", len(self.positions))
            return False
        if self.cooldown > 0:
            logger.debug("In cooldown for %d more bars", self.cooldown)
            return False
        # risk check: global drawdown
        drawdown = 1.0 - (self.capital / self.max_equity) if self.max_equity > 0 else 0.0
        if drawdown >= self.config["max_drawdown"]:
            logger.warning("Max drawdown exceeded (%.2f%%). Halting new positions.", drawdown * 100)
            return False
        return True

    def position_size_by_atr(self, atr_value: float, entry_price: float, direction: int) -> float:
        """
        Returns number of units (e.g., dollars to allocate to position) given ATR and risk settings.
        This implementation assumes trading in notional (USD) terms; adapt if trading lots.
        """
        risk_per_trade = self.capital * self.config["risk_per_trade"]
        stop_distance = self.config["stop_atr_mult"] * atr_value
        if stop_distance <= 0:
            logger.debug("Stop distance non-positive; skipping position sizing.")
            return 0.0
        # For fixed fractional money risk, position_notional = risk_per_trade / stop_distance_in_price
        # But risk per trade is in account currency; for FX, size calculation differs — keep generic approach:
        pos_notional = risk_per_trade / stop_distance
        # Respect leverage: ensure notional doesn't exceed capital * leverage
        max_allowed = self.capital * self.config["leverage"]
        pos_notional = min(pos_notional, max_allowed)
        # Enforce minimum position size
        if pos_notional * entry_price < self.config["min_position_usd"]:
            logger.debug("Computed position below minimum notional. pos_notional*price=%.2f", pos_notional * entry_price)
            return 0.0
        return pos_notional

    def open_position(self, index: int, price: float, direction: int, atr_value: float):
        size = self.position_size_by_atr(atr_value, price, direction)
        if size <= 0 or not self.can_open_new_position():
            return
        stop_price = price - direction * self.config["stop_atr_mult"] * atr_value
        take_profit = price + direction * self.config["take_profit_atr_mult"] * atr_value if self.config["take_profit_atr_mult"] else None
        pos = Position(entry_price=price, size=size, stop_price=stop_price, take_profit=take_profit, entry_index=index, direction=direction)
        self.positions.append(pos)
        logger.info("Opened %s position: entry=%.5f size=%.2f stop=%.5f tp=%s", 
                    "LONG" if direction == 1 else "SHORT", price, size, f"{take_profit:.5f}" if take_profit else "None")

    def close_position(self, pos: Position, exit_price: float, index: int):
        # Realized P&L in notional terms
        direction = pos.direction
        profit = (exit_price - pos.entry_price) * direction * pos.size
        self.capital += profit  # update capital by realized profit
        self.last_trade_profit = profit
        self.max_equity = max(self.max_equity, self.capital)
        logger.info("Closed %s position opened at %d: entry=%.5f exit=%.5f size=%.2f pnl=%.2f capital=%.2f",
                    "LONG" if direction == 1 else "SHORT", pos.entry_index, pos.entry_price, exit_price, pos.size, profit, self.capital)
        # Cooldown after loss
        if profit < 0:
            self.cooldown = max(self.cooldown, self.config["cooldown_bars_after_loss"])
            logger.debug("Loss incurred. Entering cooldown for %d bars.", self.cooldown)
        # remove position
        try:
            self.positions.remove(pos)
        except ValueError:
            logger.exception("Tried to remove position that was not in list.")
        # Update equity curve point after order execution
        self.equity_curve.append(self.capital)

    def update_trailing_stops_and_check_exit(self, row: pd.Series, index: int):
        """
        Update peak_unrealized and trailing stops. Exit if stop or take-profit hit.
        """
        current_price = row["Close"]
        for pos in self.positions.copy():
            # update peak unrealized PnL for trailing logic
            unrealized = (current_price - pos.entry_price) * pos.direction * pos.size
            pos.peak_unrealized = max(pos.peak_unrealized, unrealized)

            # Trailing stop based on peak price (conservative): set stop at peak - trail_atr_mult * ATR for long
            atr_val = row["ATR"] if "ATR" in row and not pd.isna(row["ATR"]) else np.nan
            if not math.isnan(atr_val) and atr_val > 0:
                if pos.direction == 1:
                    # For long, trailing stop moves up as price moves to lock in gains
                    new_stop = max(pos.stop_price, current_price - self.config["trail_atr_mult"] * atr_val)
                else:
                    new_stop = min(pos.stop_price, current_price + self.config["trail_atr_mult"] * atr_val)
                if new_stop != pos.stop_price:
                    logger.debug("Adjusting trailing stop for pos entered at %d: from %.5f to %.5f", pos.entry_index, pos.stop_price, new_stop)
                    pos.stop_price = new_stop

            # Check stop loss
            stop_hit = (current_price <= pos.stop_price) if pos.direction == 1 else (current_price >= pos.stop_price)
            tp_hit = False
            if pos.take_profit is not None:
                tp_hit = (current_price >= pos.take_profit) if pos.direction == 1 else (current_price <= pos.take_profit)

            if stop_hit or tp_hit:
                logger.debug("Exit condition met (stop=%s / tp=%s): current=%.5f stop=%.5f tp=%s",
                             stop_hit, tp_hit, current_price, pos.stop_price, pos.take_profit)
                self.close_position(pos, current_price, index)

    def generate_and_execute(self, df: pd.DataFrame):
        """
        Main loop: iterate bars and evaluate signals, risk management and execute (simulate) trades.
        """
        for index, row in df.iterrows():
            # update max equity
            self.max_equity = max(self.max_equity, self.capital)
            # decay cooldown
            if self.cooldown > 0:
                self.cooldown -= 1

            # First, update existing positions (trailing stops, exit rules)
            self.update_trailing_stops_and_check_exit(row, index)

            # Generate signals for potential new positions
            signal_value = row.get("Signal", 0.0)
            rsi_value = row.get("RSI", np.nan)
            atr_value = row.get("ATR", np.nan)
            price = row["Close"]

            # Basic confirmation: require direction, and RSI not overbought/oversold against signal
            direction = 0
            if signal_value > 0.5:
                direction = 1
            elif signal_value < -0.5:
                direction = -1

            # Apply RSI filter: don't open long if overbought; don't open short if oversold
            rsi_ok = True
            if not math.isnan(rsi_value):
                if direction == 1 and rsi_value >= self.config["rsi_overbought"]:
                    rsi_ok = False
                if direction == -1 and rsi_value <= self.config["rsi_oversold"]:
                    rsi_ok = False

            if direction != 0 and rsi_ok and self.can_open_new_position():
                # Additional check: require ATR > 0 to avoid tiny stops
                if not math.isnan(atr_value) and atr_value > 0:
                    # Avoid immediate entries on crossover whipsaw: require signal persistence (signal smoothing helps)
                    # Position sizing and opening
                    self.open_position(index, price, direction, atr_value)

            # Track equity (including unrealized P&L)
            unrealized = sum(((row["Close"] - p.entry_price) * p.direction * p.size) for p in self.positions)
            self.equity_curve.append(self.capital + unrealized)

        logger.info("Finished generate_and_execute loop. Final capital: %.2f", self.capital)

    def backtest(self, df: pd.DataFrame) -> Dict:
        """
        Run the end-to-end backtest on historical OHLC dataframe.
        Returns summary metrics including equity curve.
        """
        logger.info("Starting backtest for %s with %d bars", self.config["symbol"], len(df))
        df = df.copy()
        df = self.compute_signals(df)
        self.generate_and_execute(df)

        # derive metrics
        equity = pd.Series(self.equity_curve)
        ret = equity.pct_change().fillna(0)
        total_return = (equity.iloc[-1] / equity.iloc[0] - 1) if len(equity) > 0 else 0.0
        drawdown = (equity.cummax() - equity) / equity.cummax()
        max_drawdown = drawdown.max() if not drawdown.empty else 0.0
        sharpe = (ret.mean() / (ret.std() + 1e-12)) * math.sqrt(252 * (24 if "H" in self.config["timeframe"] else 1)) if ret.std() > 0 else 0.0

        summary = {
            "final_capital": float(self.capital),
            "total_return": float(total_return),
            "max_drawdown": float(max_drawdown),
            "sharpe_like": float(sharpe),
            "equity_curve": equity.reset_index(drop=True),
        }
        logger.info("Backtest summary: final_capital=%.2f total_return=%.2f%% max_drawdown=%.2f%% sharpe_like=%.2f",
                    summary["final_capital"], summary["total_return"] * 100, summary["max_drawdown"] * 100, summary["sharpe_like"])
        return summary


# -------------------------------------------------------------------
# Example usage (preserve architecture: this is a placeholder entrypoint)
# -------------------------------------------------------------------
if __name__ == "__main__":
    # WARNING: Replace 'ohlc.csv' with your OHLC data file loaded into a DataFrame.
    # The dataframe must have columns: ['Open', 'High', 'Low', 'Close'] and an index or implicit integer index.
    try:
        df = pd.read_csv("ohlc.csv", parse_dates=True)
        # Ensure required columns exist
        for col in ["Open", "High", "Low", "Close"]:
            if col not in df.columns:
                raise ValueError(f"Input CSV must contain column: {col}")
        ts = TradingSystem(CONFIG)
        results = ts.backtest(df)
        # Minimal reporting while preserving logging
        print("Final capital:", results["final_capital"])
        print("Total return:", results["total_return"])
        print("Max drawdown:", results["max_drawdown"])
        print("Sharpe-like:", results["sharpe_like"])
    except FileNotFoundError:
        logger.error("ohlc.csv not found. To run backtest, provide a CSV file named 'ohlc.csv' with Open/High/Low/Close columns.")
    except Exception as e:
        logger.exception("Error during backtest run: %s", e)