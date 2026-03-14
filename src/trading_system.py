import logging
import math
import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any

# ---------------------------
# Configuration (preserved and extendable)
# ---------------------------
DEFAULT_CONFIG = {
    "initial_capital": 100000.0,
    "risk_per_trade": 0.01,                # fraction of equity to risk per trade
    "max_drawdown_allowed": 0.2,           # if portfolio drawdown exceeds this, reduce sizing / stop trading
    "drawdown_scaling_start": 0.05,        # begin scaling position sizes once drawdown exceeds this fraction
    "max_leverage": 3.0,                   # maximum portfolio leverage
    "atr_period": 14,
    "atr_multiplier": 3.0,                 # stop-loss distance in ATRs
    "trail_atr_multiplier": 2.0,           # trailing stop distance in ATRs
    "ma_fast": 20,
    "ma_slow": 50,
    "rsi_period": 14,
    "rsi_oversold": 30,
    "rsi_overbought": 70,
    "vol_target_annual": 0.10,             # target annualized portfolio vol (volatility targeting)
    "trading_cost_per_trade": 1.0,         # flat cost per trade (for simplicity)
    "slippage": 0.0,                       # slippage per share / contract
    "min_position_size": 0.0,              # minimum position size in units
    "max_position_size_fraction": 0.25,    # max fraction of capital allocated to a single position
    "logging_level": "INFO",               # logging level
    "verbose_logging": True
}

# ---------------------------
# Logger (preserve logging)
# ---------------------------
logger = logging.getLogger("TradingSystem")
log_handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
log_handler.setFormatter(formatter)
if not logger.handlers:
    logger.addHandler(log_handler)
logger.setLevel(getattr(logging, DEFAULT_CONFIG["logging_level"].upper(), logging.INFO))


# ---------------------------
# Utility functions
# ---------------------------
def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    """Compute ATR using Wilder's smoothing (approximately)."""
    high_low = high - low
    high_close = (high - close.shift(1)).abs()
    low_close = (low - close.shift(1)).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr_series = tr.ewm(alpha=1 / period, adjust=False).mean()
    return atr_series


def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.ewm(alpha=1 / period, adjust=False).mean()
    ma_down = down.ewm(alpha=1 / period, adjust=False).mean()
    rs = ma_up / ma_down.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50)
    return rsi


def annualized_vol(returns: pd.Series, periods_per_year: int = 252) -> float:
    return returns.std(ddof=0) * np.sqrt(periods_per_year)


def sharpe_ratio(returns: pd.Series, risk_free: float = 0.0, periods_per_year: int = 252) -> float:
    ann_ret = returns.mean() * periods_per_year
    ann_vol = annualized_vol(returns, periods_per_year)
    if ann_vol == 0:
        return 0.0
    return (ann_ret - risk_free) / ann_vol


def max_drawdown(equity_curve: pd.Series) -> float:
    running_max = equity_curve.cummax()
    drawdown = (equity_curve - running_max) / running_max
    return drawdown.min()


# ---------------------------
# Trading System Class
# ---------------------------
@dataclass
class BacktestResult:
    equity_curve: pd.Series
    trades: pd.DataFrame
    metrics: Dict[str, Any]


class TradingSystem:
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = DEFAULT_CONFIG.copy()
        if config:
            self.config.update(config)
        logger.setLevel(getattr(logging, self.config.get("logging_level", "INFO").upper(), logging.INFO))
        self._log_config()

    def _log_config(self):
        if self.config.get("verbose_logging", True):
            logger.info("Initialized TradingSystem with config:")
            for k, v in sorted(self.config.items()):
                logger.info("  %s: %s", k, v)

    def compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["ma_fast"] = df["close"].rolling(self.config["ma_fast"], min_periods=1).mean()
        df["ma_slow"] = df["close"].rolling(self.config["ma_slow"], min_periods=1).mean()
        df["atr"] = atr(df["high"], df["low"], df["close"], self.config["atr_period"])
        df["rsi"] = compute_rsi(df["close"], self.config["rsi_period"])
        df["returns"] = df["close"].pct_change().fillna(0)
        # add signal column placeholder
        df["signal_raw"] = 0
        return df

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Signal logic:
         - Basic trend filter: fast MA crossover slow MA
         - RSI to avoid entering on extreme overbought/oversold signals
         - Require ATR to be non-zero and recent
         - Smoothing: require signal to persist for N bars (reduces whipsaw)
        """
        df = df.copy()
        df["signal_raw"] = 0
        ma_cross_long = (df["ma_fast"] > df["ma_slow"]) & (df["ma_fast"].shift(1) <= df["ma_slow"].shift(1))
        ma_cross_short = (df["ma_fast"] < df["ma_slow"]) & (df["ma_fast"].shift(1) >= df["ma_slow"].shift(1))
        # Basic signals
        df.loc[ma_cross_long, "signal_raw"] = 1
        df.loc[ma_cross_short, "signal_raw"] = -1

        # Filter signals with RSI band and ATR presence
        df.loc[df["rsi"] > self.config["rsi_overbought"], "signal_raw"] = df.loc[df["rsi"] > self.config["rsi_overbought"], "signal_raw"].apply(lambda x: x if x < 0 else 0)
        df.loc[df["rsi"] < self.config["rsi_oversold"], "signal_raw"] = df.loc[df["rsi"] < self.config["rsi_oversold"], "signal_raw"].apply(lambda x: x if x > 0 else 0)
        df.loc[df["atr"].isna() | (df["atr"] <= 0), "signal_raw"] = 0

        # Exponential smoothing of the signal to avoid whipsaw: require consecutive confirmations
        persistence = 2
        df["signal"] = 0
        for i in range(len(df)):
            if i == 0:
                df.iat[i, df.columns.get_loc("signal")] = 0
                continue
            # if last N raw signals are same, accept
            window = df["signal_raw"].iloc[max(0, i - persistence + 1): i + 1]
            if len(window) == persistence and (window == window.iloc[0]).all() and window.iloc[0] != 0:
                df.iat[i, df.columns.get_loc("signal")] = int(window.iloc[0])
            else:
                # maintain previous position until exit signal
                df.iat[i, df.columns.get_loc("signal")] = df.iat[i - 1, df.columns.get_loc("signal")]
        return df

    def position_size(self, equity: float, atr: float, price: float, current_drawdown: float) -> float:
        """
        Volatility-based position sizing:
         - risk_per_trade fraction of equity is risked (e.g., 1%)
         - stop_distance = atr * atr_multiplier
         - size = (risk_per_trade * equity) / (stop_distance * price)
         - enforce max_position_size_fraction and min_position_size
         - scale down position when drawdown increases (risk control)
        """
        cfg = self.config
        if atr <= 0 or price <= 0:
            return 0.0

        # scale based on current drawdown (linearly reduce sizing if drawdown between start and max)
        dd = max(0.0, current_drawdown)
        if dd >= cfg["max_drawdown_allowed"]:
            # do not open new positions if drawdown exceeded allowed level
            logger.warning("Drawdown %.2f exceeds max allowed %.2f: skipping new positions", dd, cfg["max_drawdown_allowed"])
            return 0.0
        elif dd > cfg["drawdown_scaling_start"]:
            scale = max(0.0, 1.0 - (dd - cfg["drawdown_scaling_start"]) / (cfg["max_drawdown_allowed"] - cfg["drawdown_scaling_start"]))
        else:
            scale = 1.0

        stop_distance = atr * cfg["atr_multiplier"]
        risk_amount = equity * cfg["risk_per_trade"] * scale
        # shares/contracts = risk_amount / (stop_distance * price)
        raw_size = risk_amount / (stop_distance * price)
        # limit by maximum position fraction
        max_units = (equity * cfg["max_position_size_fraction"]) / price
        size = min(raw_size, max_units)
        if size * price < cfg["min_position_size"]:
            return 0.0
        # ensure size is non-negative and integer-ish if needed (round down to whole units)
        size = max(0.0, size)
        return float(size)

    def run_backtest(self, df: pd.DataFrame) -> BacktestResult:
        """
        Core backtest loop. Single-asset assumption.
        Preserves logging and returns trade ledger and equity curve.
        """
        df = df.copy().reset_index(drop=True)
        df = self.compute_indicators(df)
        df = self.generate_signals(df)

        cfg = self.config
        initial_capital = cfg["initial_capital"]
        equity = initial_capital
        equity_curve = []
        positions = 0.0
        entry_price = 0.0
        entry_atr = 0.0
        stop_price = None
        trail_price = None

        trades = []  # records of trades

        peak_equity = equity
        for i, row in df.iterrows():
            price = row["close"]
            signal = int(row.get("signal", 0))
            atr_value = row.get("atr", np.nan)
            ret = row.get("returns", 0.0)

            # update portfolio equity with mark-to-market P&L
            if positions != 0:
                # unrealized pnl = positions * (price - last_price)
                # We use returns to update equity proportionally:
                equity = equity * (1 + ret * (positions * price) / max(1e-12, (equity if equity != 0 else 1)))
            # record peak equity for drawdown computations
            if equity > peak_equity:
                peak_equity = equity

            current_drawdown = (peak_equity - equity) / peak_equity if peak_equity > 0 else 0.0

            # Risk management: if drawdown beyond limit, close positions
            if current_drawdown >= cfg["max_drawdown_allowed"] and positions != 0:
                logger.info("Drawdown %.2f >= max allowed %.2f: closing positions", current_drawdown, cfg["max_drawdown_allowed"])
                # close position at current price
                proceeds = positions * price
                cost = cfg["trading_cost_per_trade"]
                pnl = proceeds - positions * entry_price - cost
                equity += pnl  # realized P&L already included in mark-to-market; adjust for rounding/cost
                trades.append({
                    "index": i,
                    "action": "close_for_drawdown",
                    "size": positions,
                    "price": price,
                    "pnl": pnl,
                    "equity": equity
                })
                positions = 0.0
                entry_price = 0.0
                entry_atr = 0.0
                stop_price = None
                trail_price = None
                # after closing, continue without opening new positions
                equity_curve.append(equity)
                continue

            # If currently flat and signal requests entry
            if positions == 0 and signal != 0:
                # Determine position size
                size = self.position_size(equity, atr_value, price, current_drawdown)
                # Volatility targeting: scale position to get closer to target annual vol
                # compute historical vol from recent returns
                hist_window = min(len(df.iloc[:i + 1]), 21)
                if hist_window >= 2:
                    hist_vol = annualized_vol(df["returns"].iloc[max(0, i - hist_window + 1): i + 1])
                else:
                    hist_vol = 0.0
                vol_target = cfg["vol_target_annual"]
                if hist_vol > 0 and vol_target > 0:
                    vol_scale = vol_target / hist_vol
                    # limit exposures by max_leverage
                    vol_scale = min(vol_scale, cfg["max_leverage"])
                else:
                    vol_scale = 1.0
                size = size * vol_scale
                if size <= 0:
                    if cfg["verbose_logging"]:
                        logger.debug("Calculated size 0 at index %d, skipping entry", i)
                else:
                    # place trade
                    cost = cfg["trading_cost_per_trade"]
                    slippage = cfg["slippage"] * size
                    cash_change = -size * price - cost - slippage
                    equity += cash_change  # reduce cash/equity by position cost (simplified)
                    entry_price = price
                    entry_atr = atr_value
                    positions = size if signal > 0 else -size
                    # set stop and trailing stop depending on direction
                    stop_price = price - np.sign(positions) * (entry_atr * cfg["atr_multiplier"])
                    trail_price = price - np.sign(positions) * (entry_atr * cfg["trail_atr_multiplier"])
                    trades.append({
                        "index": i,
                        "action": "entry",
                        "size": positions,
                        "price": price,
                        "pnl": 0.0,
                        "equity": equity
                    })
                    if cfg["verbose_logging"]:
                        logger.info("Entered position %s size=%.4f price=%.2f equity=%.2f", "LONG" if positions > 0 else "SHORT", abs(positions), price, equity)

            # If in a position, check stop-loss / trailing stop / exit on reverse signal
            elif positions != 0:
                exit_reason = None
                # Update trailing stop based on new ATR if it increases (only move stop in direction of protecting profit)
                if atr_value is not None and not np.isnan(atr_value):
                    new_trail = price - np.sign(positions) * (atr_value * cfg["trail_atr_multiplier"])
                    # Only move trail towards current price (i.e., tightening)
                    if np.sign(positions) > 0:
                        trail_price = max(trail_price, new_trail) if trail_price is not None else new_trail
                    else:
                        trail_price = min(trail_price, new_trail) if trail_price is not None else new_trail

                # check stop price hit
                if stop_price is not None:
                    if (positions > 0 and price <= stop_price) or (positions < 0 and price >= stop_price):
                        exit_reason = "stop"
                # check trailing stop hit
                if exit_reason is None and trail_price is not None:
                    if (positions > 0 and price <= trail_price) or (positions < 0 and price >= trail_price):
                        exit_reason = "trail"
                # exit on reverse signal (crossover)
                if exit_reason is None and signal != 0 and np.sign(signal) != np.sign(positions):
                    exit_reason = "signal_reverse"

                if exit_reason is not None:
                    # close position
                    proceeds = abs(positions) * price
                    cost = cfg["trading_cost_per_trade"]
                    slippage = cfg["slippage"] * abs(positions)
                    realized_pnl = 0.0
                    # compute pnl relative to entry_price for direction
                    realized_pnl = (price - entry_price) * positions - cost - slippage
                    # update equity
                    equity += realized_pnl
                    trades.append({
                        "index": i,
                        "action": f"exit_{exit_reason}",
                        "size": positions,
                        "price": price,
                        "pnl": realized_pnl,
                        "equity": equity
                    })
                    if cfg["verbose_logging"]:
                        logger.info("Exited position reason=%s size=%.4f price=%.2f pnl=%.2f equity=%.2f", exit_reason, positions, price, realized_pnl, equity)
                    positions = 0.0
                    entry_price = 0.0
                    entry_atr = 0.0
                    stop_price = None
                    trail_price = None

            # compute and store equity (mark to market)
            equity_curve.append(equity)

        # Construct outputs
        equity_index = df.index
        equity_series = pd.Series(equity_curve, index=equity_index, name="equity")
        trades_df = pd.DataFrame(trades)

        # Post-process metrics
        returns = equity_series.pct_change().fillna(0)
        metrics = {}
        metrics["final_equity"] = float(equity_series.iloc[-1]) if len(equity_series) > 0 else float(equity)
        metrics["total_return"] = (equity_series.iloc[-1] / initial_capital - 1.0) if len(equity_series) > 0 else 0.0
        metrics["sharpe"] = float(sharpe_ratio(returns))
        metrics["annual_vol"] = float(annualized_vol(returns))
        metrics["max_drawdown"] = float(abs(max_drawdown(equity_series)))
        metrics["trades"] = len(trades_df)
        # Sortino ratio optional
        negative_returns = returns[returns < 0]
        downside_vol = negative_returns.std(ddof=0) * np.sqrt(252) if len(negative_returns) > 0 else 0.0
        if downside_vol == 0:
            metrics["sortino"] = np.nan
        else:
            ann_ret = returns.mean() * 252
            metrics["sortino"] = float(ann_ret / downside_vol)

        if cfg["verbose_logging"]:
            logger.info("Backtest complete. Metrics: %s", metrics)

        return BacktestResult(equity_curve=equity_series, trades=trades_df, metrics=metrics)


# ---------------------------
# Example usage
# ---------------------------
if __name__ == "__main__":
    # This example demonstrates how to run the trading system on CSV price data.
    # The CSV is expected to have columns: date, open, high, low, close, volume
    import argparse
    parser = argparse.ArgumentParser(description="Run TradingSystem backtest on single-asset CSV")
    parser.add_argument("--data", type=str, default=None, help="Path to CSV file with price data")
    parser.add_argument("--initial_capital", type=float, default=None, help="Initial capital override")
    args = parser.parse_args()

    # generate simple synthetic data if none provided
    if args.data is None:
        logger.info("No data file provided. Generating synthetic price series for demo.")
        dates = pd.date_range(start="2020-01-01", periods=500, freq="B")
        np.random.seed(42)
        price = 100 + np.cumsum(np.random.normal(0, 0.5, size=len(dates)))
        high = price + np.random.uniform(0.0, 0.5, size=len(dates))
        low = price - np.random.uniform(0.0, 0.5, size=len(dates))
        openp = price + np.random.uniform(-0.2, 0.2, size=len(dates))
        close = price
        volume = np.random.randint(100, 1000, size=len(dates))
        data = pd.DataFrame({"date": dates, "open": openp, "high": high, "low": low, "close": close, "volume": volume})
        data.set_index("date", inplace=True)
    else:
        data = pd.read_csv(args.data, parse_dates=True, index_col=0)
        # ensure required columns exist
        for col in ["open", "high", "low", "close", "volume"]:
            if col not in data.columns:
                raise ValueError(f"Missing required column in CSV: {col}")

    cfg = DEFAULT_CONFIG.copy()
    if args.initial_capital is not None:
        cfg["initial_capital"] = args.initial_capital
    # Set verbose logging in example
    cfg["verbose_logging"] = True

    ts = TradingSystem(config=cfg)
    result = ts.run_backtest(data.reset_index(drop=True))
    # Print summary
    logger.info("Final metrics: %s", result.metrics)
    logger.info("Number of trades: %d", len(result.trades))
    # Optionally save results
    try:
        result.trades.to_csv("trades.csv", index=False)
        result.equity_curve.to_csv("equity_curve.csv", index=True, header=True)
        logger.info("Saved trades.csv and equity_curve.csv")
    except Exception as e:
        logger.warning("Could not save outputs: %s", str(e))