import logging
import math
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, Optional, Any, List

# ---------------------------------------------------------------------------
# Configuration - preserve and expose for easy adjustments / persistence
# ---------------------------------------------------------------------------
config = {
    "initial_capital": 100000.0,
    "risk_per_trade_pct": 0.01,          # fraction of equity to risk per trade
    "max_drawdown_pct": 0.20,            # stop trading if equity drawdown exceeds this fraction
    "atr_length": 14,                    # ATR length for stop placement and volatility estimate
    "sma_length": 200,                   # trend filter (only take long if price > SMA)
    "vol_target_annual": 0.10,           # target annualized volatility for portfolio sizing
    "bars_per_year": 252,                # for annualization (if daily data)
    "commission_per_trade": 1.0,         # flat commission per execution
    "slippage_pct": 0.0005,              # slippage as fraction of price
    "max_positions": 1,                  # keep architecture for single-instrument strategy
    "min_trade_size": 1,                 # minimum tradable lot (shares)
    "trailing_atr_mul": 3.0,             # ATR multiple for trailing stop
    "stop_atr_mul": 2.0,                 # ATR multiple for initial stop
    "vol_smoothing": 10,                 # EWMA length for volatility smoothing
    "logging_level": "INFO",
    "debug": False,
}

# ---------------------------------------------------------------------------
# Logging - preserve logging pattern
# ---------------------------------------------------------------------------
logger = logging.getLogger("TradingSystem")
log_level = getattr(logging, config.get("logging_level", "INFO").upper(), logging.INFO)
logger.setLevel(logging.DEBUG if config.get("debug") else log_level)
handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
handler.setFormatter(formatter)
if not logger.handlers:
    logger.addHandler(handler)


# ---------------------------------------------------------------------------
# Utilities and indicators
# ---------------------------------------------------------------------------
def compute_atr(high: pd.Series, low: pd.Series, close: pd.Series, length: int) -> pd.Series:
    high_low = high - low
    high_prev_close = (high - close.shift(1)).abs()
    low_prev_close = (low - close.shift(1)).abs()
    tr = pd.concat([high_low, high_prev_close, low_prev_close], axis=1).max(axis=1)
    atr = tr.rolling(length, min_periods=1).mean()
    return atr


def ewma_volatility(returns: pd.Series, span: int, bars_per_year: int) -> float:
    # returns expected annualized volatility (scalar)
    ewma_var = returns.ewm(span=span, adjust=False).var().iloc[-1]
    if np.isnan(ewma_var) or ewma_var <= 0:
        return 0.0
    return math.sqrt(ewma_var * bars_per_year)


# ---------------------------------------------------------------------------
# Data container - keeps structure consistent
# ---------------------------------------------------------------------------
@dataclass
class MarketData:
    df: pd.DataFrame  # expects DataFrame with columns: ['open','high','low','close','volume'] and datetime index

    def ensure_columns(self):
        required = {"open", "high", "low", "close"}
        missing = required - set(self.df.columns)
        if missing:
            raise ValueError(f"MarketData missing columns: {missing}")


# ---------------------------------------------------------------------------
# Order and Position models (simple, but explicit)
# ---------------------------------------------------------------------------
@dataclass
class Position:
    size: int = 0
    entry_price: Optional[float] = None
    stop_price: Optional[float] = None
    trailing_stop: Optional[float] = None
    entry_time: Optional[pd.Timestamp] = None


@dataclass
class Trade:
    time: pd.Timestamp
    side: str
    size: int
    price: float
    commission: float
    pnl: float = 0.0


# ---------------------------------------------------------------------------
# Strategy architecture - keep similar structure but enhance risk management
# ---------------------------------------------------------------------------
class TradingStrategy:
    def __init__(self, market_data: MarketData, cfg: Dict[str, Any], logger: logging.Logger):
        self.market = market_data
        self.cfg = cfg
        self.logger = logger
        self.position = Position()
        self.equity = cfg["initial_capital"]
        self.max_equity = self.equity
        self.trades: List[Trade] = []
        self.history = []  # daily snapshots
        self._prepare_indicators()

    def _prepare_indicators(self):
        df = self.market.df.copy()
        df["atr"] = compute_atr(df["high"], df["low"], df["close"], self.cfg["atr_length"])
        df["sma"] = df["close"].rolling(self.cfg["sma_length"], min_periods=1).mean()
        df["returns"] = df["close"].pct_change().fillna(0.0)
        # EWMA volatility of returns (for volatility targeting)
        df["ewma_vol"] = df["returns"].ewm(span=self.cfg["vol_smoothing"], adjust=False).std() * math.sqrt(self.cfg["bars_per_year"])
        self.market.df = df

    def run_backtest(self):
        df = self.market.df
        for t, row in df.iterrows():
            self.on_bar(t, row)
            # snapshot
            snapshot = dict(
                time=t,
                equity=self.equity,
                position=self.position.size,
                entry_price=self.position.entry_price,
                stop_price=self.position.stop_price,
            )
            self.history.append(snapshot)
            # risk control: stop trading if drawdown threshold breached
            dd = (self.max_equity - self.equity) / self.max_equity
            if dd >= self.cfg["max_drawdown_pct"]:
                self.logger.warning(f"Max drawdown {dd:.2%} breached at {t}. Halting trading.")
                break
        self.logger.info("Backtest complete.")
        return self._summarize()

    def on_bar(self, t: pd.Timestamp, row: pd.Series):
        price = row["close"]
        atr = row["atr"]
        sma = row["sma"]
        ewma_vol = row["ewma_vol"]

        # Update trailing stop if in position
        if self.position.size > 0:
            # trailing stop moves up for long when price increases
            new_trailing = max(self.position.trailing_stop or -np.inf, price - self.cfg["trailing_atr_mul"] * atr)
            if new_trailing != self.position.trailing_stop:
                self.logger.debug(f"{t} - Updating trailing stop from {self.position.trailing_stop} to {new_trailing}")
                self.position.trailing_stop = new_trailing

            # check stops: initial stop or trailing stop
            stop_triggered = False
            # use the tighter (higher) stop for long positions
            if self.position.stop_price is not None and price <= self.position.stop_price:
                self.logger.info(f"{t} - Initial stop hit: price {price} <= stop {self.position.stop_price}")
                stop_triggered = True
            if self.position.trailing_stop is not None and price <= self.position.trailing_stop:
                self.logger.info(f"{t} - Trailing stop hit: price {price} <= trailing {self.position.trailing_stop}")
                stop_triggered = True

            if stop_triggered:
                self._exit_position(t, price, reason="stop")
                return

            # optional profit-taking: reduce risk if volatility rises sharply
            if ewma_vol > max(0.0001, self.cfg["vol_target_annual"] * 1.5):  # if realized vol > 150% of target
                self.logger.info(f"{t} - High realized vol {ewma_vol:.2%} detected; reducing exposure.")
                self._reduce_position(t, price)
                return

            # maintain position otherwise
            return

        # If no position: generate signal
        signal = self._generate_signal(t, row)
        if signal == "long":
            # position sizing: volatility-targeted risk sizing with ATR stop
            if atr <= 0 or np.isnan(atr):
                self.logger.debug(f"{t} - ATR invalid ({atr}); skipping entry.")
                return

            # Determine per-trade risk amount in currency
            risk_amount = self.equity * self.cfg["risk_per_trade_pct"]

            # Calculate stop distance
            stop_distance = self.cfg["stop_atr_mul"] * atr
            # Conservative check: ensure stop_distance non-zero
            if stop_distance <= 0:
                self.logger.debug(f"{t} - stop_distance <= 0; skipping entry.")
                return

            # position size (units) = risk_amount / stop_distance rounded to integer lot
            raw_size = int(math.floor(risk_amount / stop_distance))
            size = max(self.cfg["min_trade_size"], raw_size)
            if size <= 0:
                self.logger.debug(f"{t} - Computed size {size} <= 0; skipping.")
                return

            # Additional volatility targeting: scale size if realized vol deviates from target
            vol_target = max(1e-6, self.cfg["vol_target_annual"])
            if ewma_vol > 0:
                vol_scale = vol_target / ewma_vol
                # clamp scaling to [0.5, 1.5] to avoid extreme sizing
                vol_scale = float(np.clip(vol_scale, 0.5, 1.5))
                size = max(self.cfg["min_trade_size"], int(math.floor(size * vol_scale)))

            if size <= 0:
                self.logger.debug(f"{t} - size after vol scaling {size} <= 0; skipping.")
                return

            # place entry
            entry_price = price * (1 + self.cfg["slippage_pct"])  # simulate slippage for entry
            stop_price = entry_price - stop_distance
            trailing_stop = entry_price - self.cfg["trailing_atr_mul"] * atr

            self._enter_position(t, size, entry_price, stop_price, trailing_stop)
            return

        # neutral or short signals are ignored (single-direction system)
        return

    def _generate_signal(self, t: pd.Timestamp, row: pd.Series) -> Optional[str]:
        # Simple trend-following: buy only if price > SMA (long-only)
        price = row["close"]
        sma = row["sma"]
        # require that price is above SMA and recent momentum positive
        recent_returns = self.market.df.loc[:t]["returns"].iloc[-5:] if t in self.market.df.index else pd.Series(dtype=float)
        momentum = recent_returns.mean() if not recent_returns.empty else 0.0
        if price > sma and momentum > 0:
            self.logger.debug(f"{t} - Signal LONG generated (price {price} > sma {sma}, momentum {momentum:.5f})")
            return "long"
        return None

    # Execution functions: maintain logging & trades history
    def _enter_position(self, t: pd.Timestamp, size: int, price: float, stop_price: float, trailing_stop: float):
        commission = self.cfg["commission_per_trade"]
        self.position = Position(
            size=size,
            entry_price=price,
            stop_price=stop_price,
            trailing_stop=trailing_stop,
            entry_time=t,
        )
        self.equity -= commission  # charge commission on entry
        self.trades.append(Trade(time=t, side="BUY", size=size, price=price, commission=commission))
        self.max_equity = max(self.max_equity, self.equity)
        self.logger.info(f"{t} - Entered LONG size={size} price={price:.2f} stop={stop_price:.2f} trailing={trailing_stop:.2f}")

    def _exit_position(self, t: pd.Timestamp, price: float, reason: str = "exit"):
        if self.position.size == 0:
            return
        commission = self.cfg["commission_per_trade"]
        # simulate slippage on exit
        exec_price = price * (1 - self.cfg["slippage_pct"])
        pnl = (exec_price - self.position.entry_price) * self.position.size
        self.equity += pnl
        self.equity -= commission
        trade = Trade(time=t, side="SELL", size=self.position.size, price=exec_price, commission=commission, pnl=pnl)
        self.trades.append(trade)
        self.logger.info(f"{t} - Exited position size={self.position.size} price={exec_price:.2f} pnl={pnl:.2f} reason={reason}")
        # reset position
        self.position = Position()

        # track max equity for drawdown calculations
        self.max_equity = max(self.max_equity, self.equity)

    def _reduce_position(self, t: pd.Timestamp, price: float):
        # reduce position by half (conservative) when realized volatility rises sharply
        if self.position.size <= self.cfg["min_trade_size"]:
            self.logger.debug(f"{t} - Position too small to reduce.")
            return
        reduce_size = max(self.cfg["min_trade_size"], int(math.floor(self.position.size / 2)))
        exec_price = price * (1 - self.cfg["slippage_pct"])
        pnl = (exec_price - self.position.entry_price) * reduce_size
        self.equity += pnl
        self.equity -= self.cfg["commission_per_trade"]
        self.trades.append(Trade(time=t, side="REDUCE", size=reduce_size, price=exec_price, commission=self.cfg["commission_per_trade"], pnl=pnl))
        self.position.size -= reduce_size
        self.logger.info(f"{t} - Reduced position by {reduce_size} at {exec_price:.2f}, pnl={pnl:.2f}. Remaining size={self.position.size}")
        # tighten stop to protect remaining position
        if self.position.size > 0:
            self.position.stop_price = max(self.position.stop_price, price - self.cfg["stop_atr_mul"] * row_safe_atr(self.market.df, t, self.cfg))

    def _summarize(self) -> Dict[str, Any]:
        # compute P&L summary and risk metrics
        df = self.market.df.copy()
        res = {}
        res["ending_equity"] = self.equity
        res["trades"] = len(self.trades)
        # compute equity curve from trade history
        equity_curve = [self.cfg["initial_capital"]]
        eq = self.cfg["initial_capital"]
        for tr in self.trades:
            eq -= tr.commission
            if tr.side in ("SELL", "REDUCE"):
                eq += tr.pnl
            equity_curve.append(eq)
        res["equity_curve"] = equity_curve
        # performance metrics: returns series
        # approximate daily returns from equity curve
        returns = pd.Series(equity_curve).pct_change().fillna(0)
        res["annual_return"] = ((1 + returns.mean()) ** self.cfg["bars_per_year"]) - 1 if len(returns) > 1 else 0.0
        res["annual_volatility"] = returns.std() * math.sqrt(self.cfg["bars_per_year"]) if len(returns) > 1 else 0.0
        res["sharpe"] = (res["annual_return"] / res["annual_volatility"]) if res["annual_volatility"] > 0 else np.nan
        # max drawdown calculation
        eq_array = np.array(equity_curve)
        peak = np.maximum.accumulate(eq_array)
        drawdowns = (peak - eq_array) / peak
        res["max_drawdown"] = float(np.nanmax(drawdowns)) if drawdowns.size else 0.0
        self.logger.info(f"Ending equity: {self.equity:.2f}, trades: {len(self.trades)}, max_drawdown: {res['max_drawdown']:.2%}, sharpe: {res['sharpe']}")
        return res


# ---------------------------------------------------------------------------
# Helper to safely fetch ATR for a specific timestamp
# ---------------------------------------------------------------------------
def row_safe_atr(df: pd.DataFrame, t: pd.Timestamp, cfg: Dict[str, Any]) -> float:
    try:
        return float(df.at[t, "atr"])
    except Exception:
        # fallback to last available ATR
        last_atr = df["atr"].ffill().iloc[-1]
        return float(last_atr if not np.isnan(last_atr) else cfg["atr_length"])


# ---------------------------------------------------------------------------
# Example usage (if run as a script). Data ingestion preserved external to core
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # This block assumes you replace data loading with your actual data source.
    # The architecture and configuration are preserved; only risk management and execution rules enhanced.
    import sys

    # Minimal synthetic example for illustration. Replace with CSV loader in production.
    dates = pd.date_range(start="2020-01-01", periods=500, freq="B")
    np.random.seed(42)
    price = pd.Series(100 + np.cumsum(np.random.normal(0, 1, len(dates))), index=dates)
    high = price + np.random.uniform(0.1, 1.0, len(dates))
    low = price - np.random.uniform(0.1, 1.0, len(dates))
    openp = price + np.random.uniform(-0.5, 0.5, len(dates))
    close = price
    volume = pd.Series(np.random.randint(100, 1000, len(dates)), index=dates)
    df = pd.DataFrame({"open": openp, "high": high, "low": low, "close": close, "volume": volume}, index=dates)

    market = MarketData(df=df)
    market.ensure_columns()
    strat = TradingStrategy(market, config, logger)
    results = strat.run_backtest()

    logger.info("Summary:")
    for k, v in results.items():
        if k != "equity_curve":
            logger.info(f"{k}: {v}")