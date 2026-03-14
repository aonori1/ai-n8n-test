import logging
import logging.handlers
import os
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from math import floor

# -------------------------------------------------------------------
# Configuration (preserved and extendable)
# -------------------------------------------------------------------
CONFIG: Dict[str, Any] = {
    "initial_capital": 1_000_000.0,
    "risk_per_trade": 0.005,               # fraction of equity to risk per trade (0.5%)
    "max_portfolio_risk": 0.02,            # maximum fraction of equity at risk across all open trades
    "target_annual_vol": 0.10,             # volatility targeting (10% annual)
    "max_leverage": 3.0,                   # maximum portfolio leverage from vol targeting
    "atr_period": 21,
    "atr_multiplier": 3.0,                 # stop distance in ATRs
    "trailing_atr_multiplier": 2.0,        # trailing stop distance in ATRs
    "volatility_filter_period": 63,        # lookback for realized volatility filter
    "volatility_filter_multiplier": 1.25,  # only take trades if current vol < multiplier * long_term_vol
    "ma_fast": 20,
    "ma_slow": 100,
    "max_concurrent_positions": 5,
    "max_drawdown_tol": 0.25,              # if equity drawdown exceeds 25% stop trading
    "cooldown_days_after_dd": 10,          # days to pause after hitting drawdown tolerance
    "commission_per_trade": 1.0,           # flat commission per trade
    "slippage_pct": 0.0005,                # slippage per trade as pct of trade value
    "log_dir": "logs",
    "log_file": "trading_system.log",
    "verbose": True
}

# -------------------------------------------------------------------
# Logging (preserved)
# -------------------------------------------------------------------
os.makedirs(CONFIG["log_dir"], exist_ok=True)
logger = logging.getLogger("TradingSystem")
logger.setLevel(logging.DEBUG if CONFIG["verbose"] else logging.INFO)
formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
file_handler = logging.handlers.RotatingFileHandler(
    os.path.join(CONFIG["log_dir"], CONFIG["log_file"]),
    maxBytes=5_000_000,
    backupCount=3
)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)


# -------------------------------------------------------------------
# Utilities
# -------------------------------------------------------------------
def atr(series_high: pd.Series, series_low: pd.Series, series_close: pd.Series, period: int) -> pd.Series:
    """Compute ATR using Wilder's smoothing."""
    high_low = series_high - series_low
    high_close = (series_high - series_close.shift(1)).abs()
    low_close = (series_low - series_close.shift(1)).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr_val = tr.ewm(alpha=1 / period, adjust=False).mean()
    return atr_val


def annualized_vol(returns: pd.Series, trading_days: int = 252) -> float:
    return float(returns.std() * np.sqrt(trading_days))


# -------------------------------------------------------------------
# Trading System
# -------------------------------------------------------------------
class TradingSystem:
    def __init__(self, config: Dict[str, Any]):
        self.config = config.copy()
        self.equity = config["initial_capital"]
        self.cash = config["initial_capital"]
        self.positions: Dict[str, Dict[str, Any]] = {}  # symbol -> {size, entry_price, stop, trail}
        self.trade_log = []
        self.daily_equity = []
        self.in_cooldown_until = None
        logger.info("Trading system initialized with capital: %.2f", self.equity)

    def generate_signals(self, price_df: pd.DataFrame) -> pd.DataFrame:
        """
        price_df expected columns: ['open','high','low','close','volume'] indexed by datetime
        Single-symbol system for clarity; extendable to multi-asset.
        Returns a DataFrame with a 'signal' column: 1 buy, -1 sell/short (we use only long here), 0 flat
        """
        df = price_df.copy()
        df["ma_fast"] = df["close"].rolling(self.config["ma_fast"]).mean()
        df["ma_slow"] = df["close"].rolling(self.config["ma_slow"]).mean()
        df["atr"] = atr(df["high"], df["low"], df["close"], self.config["atr_period"])
        df["ret"] = df["close"].pct_change()
        df["realized_vol"] = df["ret"].rolling(self.config["volatility_filter_period"]).std() * np.sqrt(252)
        # Basic moving average crossover
        df["ma_signal"] = 0
        df.loc[df["ma_fast"] > df["ma_slow"], "ma_signal"] = 1
        df.loc[df["ma_fast"] < df["ma_slow"], "ma_signal"] = 0
        # Signal only when trend and volatility filter satisfied
        df["vol_filter_pass"] = df["realized_vol"] < (self.config["volatility_filter_multiplier"] * df["realized_vol"].rolling(252, min_periods=1).mean())
        df["signal"] = 0
        df.loc[(df["ma_signal"] == 1) & (df["vol_filter_pass"]), "signal"] = 1
        # De-noise: require ma_fast to be sufficiently above ma_slow (relative threshold)
        rel_thresh = 0.005  # 0.5% above
        df.loc[(df["ma_signal"] == 1) & ((df["ma_fast"] - df["ma_slow"]) / df["ma_slow"] < rel_thresh), "signal"] = 0
        logger.debug("Signals generated (head):\n%s", df[["close", "ma_fast", "ma_slow", "atr", "realized_vol", "signal"]].head(10))
        return df

    def size_position(self, price: float, stop_distance: float, equity: float) -> int:
        """
        Determine number of shares/contracts to buy given risk per trade and stop distance.
        stop_distance: absolute price distance to stop (dollars)
        """
        if stop_distance <= 0 or price <= 0:
            return 0
        risk_amount = equity * self.config["risk_per_trade"]
        raw_size = risk_amount / (stop_distance)
        size = max(0, floor(raw_size))
        logger.debug("Sizing position: price=%.4f stop_dist=%.4f equity=%.2f risk_amt=%.2f raw_size=%.2f -> size=%d",
                     price, stop_distance, equity, risk_amount, raw_size, size)
        return size

    def apply_vol_targeting(self, allocation_notional: float, returns: pd.Series) -> float:
        """
        Scale allocation notional according to volatility targeting.
        Returns scaled notional (absolute dollars).
        """
        current_vol = annualized_vol(returns.dropna()) if returns.dropna().shape[0] >= 2 else None
        if current_vol is None or current_vol == 0:
            logger.debug("Vol targeting: insufficient vol data; skipping scaling.")
            return allocation_notional
        scale = self.config["target_annual_vol"] / current_vol
        scale = min(scale, self.config["max_leverage"])
        scaled = allocation_notional * scale
        logger.debug("Vol targeting: current_vol=%.4f target=%.4f scale=%.4f -> scaled_notional=%.2f",
                     current_vol, self.config["target_annual_vol"], scale, scaled)
        return scaled

    def _enter_trade(self, symbol: str, price: float, size: int, atr_val: float, date: pd.Timestamp):
        stop_distance = atr_val * self.config["atr_multiplier"]
        stop_price = price - stop_distance
        trail_distance = atr_val * self.config["trailing_atr_multiplier"]
        trail_stop = price - trail_distance
        self.positions[symbol] = {
            "size": size,
            "entry_price": price,
            "stop_price": stop_price,
            "trail_stop": trail_stop,
            "atr": atr_val,
            "entry_date": date
        }
        cost = price * size
        total_commission = self.config["commission_per_trade"]
        slippage = abs(cost) * self.config["slippage_pct"]
        self.cash -= (cost + total_commission + slippage)
        logger.info("ENTER %s %s shares at %.4f on %s | stop=%.4f trail=%.4f cost=%.2f",
                    symbol, size, price, date.strftime("%Y-%m-%d"), stop_price, trail_stop, cost)

    def _exit_trade(self, symbol: str, price: float, date: pd.Timestamp, reason: str):
        pos = self.positions.pop(symbol, None)
        if pos is None:
            return
        size = pos["size"]
        proceeds = price * size
        commission = self.config["commission_per_trade"]
        slippage = abs(proceeds) * self.config["slippage_pct"]
        self.cash += (proceeds - commission - slippage)
        pnl = (price - pos["entry_price"]) * size - commission - slippage
        self.trade_log.append({
            "symbol": symbol,
            "entry_date": pos["entry_date"],
            "exit_date": date,
            "entry_price": pos["entry_price"],
            "exit_price": price,
            "size": size,
            "pnl": pnl,
            "reason": reason
        })
        logger.info("EXIT %s %d shares at %.4f on %s | pnl=%.2f reason=%s",
                    symbol, size, price, date.strftime("%Y-%m-%d"), pnl, reason)

    def backtest(self, price_df: pd.DataFrame, symbol: str = "SYM") -> Dict[str, Any]:
        """
        Single-symbol backtest to improve risk-adjusted performance:
        - ATR stop
        - Volatility targeting
        - Max concurrent positions
        - Drawdown/cooldown rules
        """
        df = self.generate_signals(price_df)
        df = df.dropna(subset=["close"])
        equity_series = []
        peak_equity = self.equity
        cooldown_days_remaining = 0

        # iterate through rows
        for date, row in df.iterrows():
            price = float(row["close"])
            atr_val = float(row["atr"]) if not np.isnan(row["atr"]) else 0.0
            signal = int(row["signal"])

            # enforce cooldown from large drawdown
            if self.in_cooldown_until is not None and date <= self.in_cooldown_until:
                logger.debug("In cooldown until %s; skipping new entries on %s", self.in_cooldown_until, date)
                allow_new_entries = False
            else:
                allow_new_entries = True

            # Update trailing stops for open positions
            for sym, pos in list(self.positions.items()):
                # update trailing stop: only tighten
                new_trail = price - pos["atr"] * self.config["trailing_atr_multiplier"]
                if new_trail > pos["trail_stop"]:
                    logger.debug("Tightening trailing stop for %s from %.4f to %.4f", sym, pos["trail_stop"], new_trail)
                    pos["trail_stop"] = new_trail
                # Check stops: stop_price or trail_stop hit -> exit
                if price <= pos["stop_price"]:
                    self._exit_trade(sym, price, date, reason="stop_loss")
                elif price <= pos["trail_stop"]:
                    self._exit_trade(sym, price, date, reason="trailing_stop")

            # Check if we should open a new trade
            if signal == 1 and allow_new_entries:
                # Respect max concurrent positions
                if len(self.positions) < self.config["max_concurrent_positions"]:
                    # compute stop distance from ATR
                    stop_distance = atr_val * self.config["atr_multiplier"] if atr_val > 0 else max(0.01 * price, 1.0)
                    # compute nominal allocation before vol targeting
                    raw_notional = self.equity / max(1, self.config["max_concurrent_positions"])
                    # Apply volatility targeting to notional to avoid oversized positions in calm markets
                    scaled_notional = self.apply_vol_targeting(raw_notional, df["ret"].loc[:date])
                    # initial size from risk per trade and stop distance (but also cap by scaled_notional)
                    size_by_risk = self.size_position(price, stop_distance, self.equity)
                    size_by_notional = max(0, floor(scaled_notional / price))
                    size = int(min(size_by_risk, size_by_notional))
                    # Ensure we don't violate portfolio-level max risk
                    estimated_risk = size * stop_distance
                    max_port_risk_amt = self.equity * self.config["max_portfolio_risk"]
                    total_current_risk = sum([p["size"] * (p["atr"] * self.config["atr_multiplier"]) for p in self.positions.values()])
                    if (total_current_risk + estimated_risk) > max_port_risk_amt:
                        logger.debug("Skipping entry due to portfolio risk cap. total_current_risk=%.2f estimated_risk=%.2f max_port_risk_amt=%.2f",
                                     total_current_risk, estimated_risk, max_port_risk_amt)
                        size = 0
                    if size > 0:
                        # Apply cash/leverage check
                        notional = size * price
                        # determine available buying power (allow some leverage up to max_leverage * equity)
                        max_notional = self.equity * self.config["max_leverage"]
                        current_notional = sum([p["size"] * p["entry_price"] for p in self.positions.values()])
                        if (current_notional + notional) <= max_notional:
                            self._enter_trade(symbol, price, size, atr_val, date)
                        else:
                            logger.debug("Skipping entry due to leverage cap. current_notional=%.2f notional=%.2f max_notional=%.2f",
                                         current_notional, notional, max_notional)
                else:
                    logger.debug("Max concurrent positions reached (%d). Skipping new entry.", self.config["max_concurrent_positions"])

            # Daily P&L update: mark-to-market positions
            mtm_pnl = sum([(price - p["entry_price"]) * p["size"] for p in self.positions.values()])
            unrealized = mtm_pnl
            total_equity = self.cash + sum([p["size"] * price for p in self.positions.values()])
            self.equity = total_equity
            equity_series.append((date, total_equity))
            # track peak equity and drawdown
            if total_equity > peak_equity:
                peak_equity = total_equity
            drawdown = (peak_equity - total_equity) / peak_equity if peak_equity > 0 else 0.0
            if drawdown >= self.config["max_drawdown_tol"]:
                # initiate cooldown
                self.in_cooldown_until = date + pd.Timedelta(days=self.config["cooldown_days_after_dd"])
                logger.warning("Drawdown %.2f exceeded tolerance %.2f. Entering cooldown until %s.", drawdown, self.config["max_drawdown_tol"], self.in_cooldown_until)
            # Store daily metrics
            self.daily_equity.append({
                "date": date,
                "equity": total_equity,
                "cash": self.cash,
                "positions_notional": sum([p["size"] * price for p in self.positions.values()]),
                "mtm_unrealized": unrealized,
                "drawdown": drawdown
            })

        # On backtest end: close all positions at last price
        last_date, last_row = df.index[-1], df.iloc[-1]
        last_price = float(last_row["close"])
        for sym in list(self.positions.keys()):
            self._exit_trade(sym, last_price, last_date, reason="end_of_backtest")

        result = {
            "initial_capital": self.config["initial_capital"],
            "final_equity": self.equity,
            "trade_log": pd.DataFrame(self.trade_log),
            "daily_equity": pd.DataFrame(self.daily_equity).set_index("date")
        }
        logger.info("Backtest complete. Initial capital=%.2f Final equity=%.2f", result["initial_capital"], result["final_equity"])
        return result


# -------------------------------------------------------------------
# Example Runner (keeps configuration and logging)
# -------------------------------------------------------------------
def load_price_csv(path: str) -> pd.DataFrame:
    """
    Load CSV with columns: date, open, high, low, close, volume
    Date parsed as index. This helper preserves architecture for file-based inputs.
    """
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.set_index("date").sort_index()
    required = {"open", "high", "low", "close", "volume"}
    if not required.issubset(df.columns):
        raise ValueError(f"CSV missing required columns. Required: {required}")
    return df


if __name__ == "__main__":
    # Example usage: python trading_system.py data.csv
    import sys
    if len(sys.argv) < 2:
        logger.error("Please provide a CSV data file path as the first argument.")
        sys.exit(1)
    data_path = sys.argv[1]
    prices = load_price_csv(data_path)
    ts = TradingSystem(CONFIG)
    results = ts.backtest(prices, symbol="SYM")
    # Save outputs
    out_dir = "results"
    os.makedirs(out_dir, exist_ok=True)
    results["trade_log"].to_csv(os.path.join(out_dir, "trade_log.csv"), index=False)
    results["daily_equity"].to_csv(os.path.join(out_dir, "daily_equity.csv"))
    logger.info("Results written to %s", out_dir)