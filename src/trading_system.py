import copy
import logging
import math
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd

# -------------------------
# Configuration (preserve)
# -------------------------
CONFIG: Dict = {
    "initial_capital": 1_000_000,
    "risk_per_trade": 0.005,  # fraction of equity to risk per trade (0.5%)
    "atr_period": 14,
    "stop_atr_multiplier": 3.0,  # initial stop distance in ATRs
    "trail_atr_multiplier": 2.0,  # trailing stop distance in ATRs
    "vol_target_annual": 0.10,  # target portfolio volatility (10% annual)
    "vol_lookback_days": 21,
    "max_drawdown": 0.20,  # stop trading / de-risk if drawdown exceeds 20%
    "drawdown_scale_start": 0.05,  # begin scaling risk after 5% drawdown
    "max_leverage": 3.0,
    "trend_fast": 50,
    "trend_slow": 200,
    "min_holding_days": 2,
    "trade_cost": 0.0005,  # fraction (0.05%)
    "slippage": 0.0005,
    "verbose_logging": True,
    "seed": 42,
}

# -------------------------
# Logging (preserve)
# -------------------------
logger = logging.getLogger("trading_system")
logger.setLevel(logging.DEBUG if CONFIG["verbose_logging"] else logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG if CONFIG["verbose_logging"] else logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)


# -------------------------
# Utilities
# -------------------------
def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=period, min_periods=1).mean()


def rolling_volatility(returns: pd.Series, lookback: int) -> pd.Series:
    # annualized volatility
    return returns.rolling(window=lookback, min_periods=1).std() * np.sqrt(252)


def max_drawdown(equity: pd.Series) -> float:
    roll_max = equity.cummax()
    drawdown = (equity - roll_max) / roll_max
    return drawdown.min()


def annualized_return(returns: pd.Series) -> float:
    total_return = (1 + returns).prod() - 1
    days = returns.shape[0]
    if days <= 0:
        return 0.0
    return (1 + total_return) ** (252.0 / days) - 1


def annualized_vol(returns: pd.Series) -> float:
    return returns.std() * np.sqrt(252)


def sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    vol = annualized_vol(returns)
    if vol == 0:
        return 0.0
    return (annualized_return(returns) - risk_free_rate) / vol


# -------------------------
# Strategy
# -------------------------
@dataclass
class Position:
    size: float = 0.0  # positive for long
    entry_price: float = 0.0
    stop_price: float = 0.0
    entry_date: pd.Timestamp = None


class TradingSystem:
    def __init__(self, config: Dict, logger: logging.Logger):
        self.config = copy.deepcopy(config)
        self.logger = logger
        self.reset()

    def reset(self):
        self.equity = self.config["initial_capital"]
        self.initial_equity = self.equity
        self.position = Position()
        self.history = []
        self.peak_equity = self.equity
        self.max_drawdown_seen = 0.0
        self.last_entry_index: Optional[int] = None
        self.trading_disabled = False

    def prepare_data(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy().sort_index().copy()
        # compute returns
        df["close_prev"] = df["Close"].shift(1)
        df["ret"] = df["Close"].pct_change().fillna(0.0)
        # ATR
        df["atr"] = atr(df["High"], df["Low"], df["Close"], self.config["atr_period"]).fillna(method="bfill")
        # trend filters
        df["sma_fast"] = df["Close"].rolling(window=self.config["trend_fast"], min_periods=1).mean()
        df["sma_slow"] = df["Close"].rolling(window=self.config["trend_slow"], min_periods=1).mean()
        # rolling vol
        df["vol"] = rolling_volatility(df["ret"], self.config["vol_lookback_days"]).replace(0, 1e-9)
        # momentum (5-day return)
        df["mom5"] = df["Close"].pct_change(5)
        return df

    def compute_dynamic_risk_scale(self, current_equity: float) -> float:
        drawdown = max(0.0, (self.peak_equity - current_equity) / max(self.peak_equity, 1e-9))
        self.max_drawdown_seen = max(self.max_drawdown_seen, drawdown)
        if drawdown <= self.config["drawdown_scale_start"]:
            return 1.0
        # linear scaling from 1 -> 0.1 as drawdown goes from drawdown_scale_start -> max_drawdown
        start = self.config["drawdown_scale_start"]
        end = self.config["max_drawdown"]
        if drawdown >= end:
            return 0.0
        scale = 1.0 - (drawdown - start) / max(end - start, 1e-9)
        return max(0.1, scale)

    def compute_position_size(self, equity: float, price: float, atr: float, vol: float, dynamic_scale: float) -> float:
        # Method combines volatility targeting and per-trade risk
        # compute notional to target vol: notional = equity * vol_target / current_vol
        target_vol = self.config["vol_target_annual"]
        notional_by_vol = equity * (target_vol / max(vol, 1e-9))
        # compute risk-per-trade based on ATR stop distance
        stop_distance = atr * self.config["stop_atr_multiplier"]
        if stop_distance <= 0:
            return 0.0
        risk_amount = equity * self.config["risk_per_trade"] * dynamic_scale
        size_by_risk = risk_amount / (stop_distance)
        # approximate size by shares = min(notional, size_by_risk_in_shares)
        size_by_notional = notional_by_vol / price
        chosen_size = min(size_by_notional, size_by_risk)
        # cap by leverage
        max_shares_by_leverage = (equity * self.config["max_leverage"]) / price
        chosen_size = max(0.0, min(chosen_size, max_shares_by_leverage))
        return math.floor(chosen_size)

    def run_backtest(self, raw_data: pd.DataFrame) -> Dict:
        df = self.prepare_data(raw_data)
        # iterate over DataFrame rows
        equity_curve = []
        returns = []
        position_value = 0.0

        for i, row in enumerate(df.itertuples()):
            date = getattr(row, IndexName(df))
            close = row.Close
            high = row.High
            low = row.Low
            atr_val = row.atr
            vol = row.vol
            sma_fast = row.sma_fast
            sma_slow = row.sma_slow
            mom5 = row.mom5

            # Update peak equity and drawdown
            equity_curve.append(self.equity)
            self.peak_equity = max(self.peak_equity, self.equity)
            current_drawdown = (self.peak_equity - self.equity) / max(self.peak_equity, 1e-9)
            if current_drawdown >= self.config["max_drawdown"]:
                if not self.trading_disabled:
                    self.trading_disabled = True
                    self.logger.warning(
                        f"{date} - Max drawdown exceeded ({current_drawdown:.2%}). Disabling new entries and scaling down risk."
                    )

            # Check exit conditions first (stop loss, trend reversal, time)
            pnl = 0.0
            if self.position.size > 0:
                # update trailing stop
                trailing_stop = max(self.position.stop_price, close - atr_val * self.config["trail_atr_multiplier"])
                if trailing_stop > self.position.stop_price:
                    self.logger.debug(f"{date} - Updating trailing stop from {self.position.stop_price:.4f} to {trailing_stop:.4f}")
                    self.position.stop_price = trailing_stop

                # Check stop hit (assume executed at close if close <= stop_price)
                if close <= self.position.stop_price:
                    exit_price = close * (1 - self.config["slippage"]) - self.config["trade_cost"] * close
                    proceeds = self.position.size * exit_price
                    entry_cost = self.position.size * self.position.entry_price
                    pnl = proceeds - entry_cost
                    # update equity
                    self.equity += pnl
                    self.logger.info(
                        f"{date} - STOP EXIT: size={self.position.size}, entry={self.position.entry_price:.4f}, exit={exit_price:.4f}, pnl={pnl:.2f}, equity={self.equity:.2f}"
                    )
                    # record trade
                    self.history.append(
                        {
                            "date": date,
                            "action": "exit_stop",
                            "size": self.position.size,
                            "entry_price": self.position.entry_price,
                            "exit_price": exit_price,
                            "pnl": pnl,
                            "equity": self.equity,
                        }
                    )
                    self.position = Position()
                    self.last_entry_index = None

                # trend reversal exit: if price drops below sma_fast or sma_fast below sma_slow
                elif close < sma_fast or sma_fast < sma_slow:
                    # require minimum holding period
                    holding_days = (date - (self.position.entry_date or date)).days if self.position.entry_date else 9999
                    if holding_days >= self.config["min_holding_days"]:
                        exit_price = close * (1 - self.config["slippage"]) - self.config["trade_cost"] * close
                        proceeds = self.position.size * exit_price
                        entry_cost = self.position.size * self.position.entry_price
                        pnl = proceeds - entry_cost
                        self.equity += pnl
                        self.logger.info(
                            f"{date} - TREND EXIT: size={self.position.size}, entry={self.position.entry_price:.4f}, exit={exit_price:.4f}, pnl={pnl:.2f}, equity={self.equity:.2f}"
                        )
                        self.history.append(
                            {
                                "date": date,
                                "action": "exit_trend",
                                "size": self.position.size,
                                "entry_price": self.position.entry_price,
                                "exit_price": exit_price,
                                "pnl": pnl,
                                "equity": self.equity,
                            }
                        )
                        self.position = Position()
                        self.last_entry_index = None

            # Entry logic
            enter_long = False
            if self.position.size == 0 and not self.trading_disabled:
                # Require: price > fast SMA and trend upward (fast > slow) and short momentum positive
                if close > sma_fast and sma_fast > sma_slow and (mom5 is not None and mom5 > 0):
                    enter_long = True

            if enter_long:
                dynamic_scale = self.compute_dynamic_risk_scale(self.equity)
                if dynamic_scale <= 0:
                    self.logger.warning(f"{date} - Dynamic scale is zero, skipping entry.")
                else:
                    size = self.compute_position_size(self.equity, close, atr_val, vol, dynamic_scale)
                    if size >= 1:
                        entry_price = close * (1 + self.config["slippage"]) + self.config["trade_cost"] * close
                        stop_price = entry_price - atr_val * self.config["stop_atr_multiplier"]
                        # Allocate
                        cost = size * entry_price
                        # Do not let position cost exceed max_leverage*equity
                        if cost > self.equity * self.config["max_leverage"]:
                            size = math.floor((self.equity * self.config["max_leverage"]) / entry_price)
                        if size >= 1:
                            self.position = Position(size, entry_price, stop_price, date)
                            self.last_entry_index = i
                            self.logger.info(
                                f"{date} - ENTRY: size={self.position.size}, entry_price={self.position.entry_price:.4f}, stop={self.position.stop_price:.4f}, equity={self.equity:.2f}, dynamic_scale={dynamic_scale:.3f}"
                            )
                            self.history.append(
                                {
                                    "date": date,
                                    "action": "entry",
                                    "size": self.position.size,
                                    "entry_price": self.position.entry_price,
                                    "stop_price": self.position.stop_price,
                                    "equity": self.equity,
                                    "dynamic_scale": dynamic_scale,
                                }
                            )

            # mark-to-market P&L for open position (not realized)
            position_value = 0.0
            if self.position.size > 0:
                position_value = self.position.size * close
            net_equity = self.equity  # Equity already includes realized P&L; unrealized is separate if needed

            # record daily returns based on change in total capital if we mark-to-market
            # For performance metrics we compute returns based on mark-to-market: unrealized included
            total_capital = net_equity + position_value
            if len(equity_curve) >= 2:
                prev_total = equity_curve[-2] + (self.position.size * getattr(df.iloc[i - 1], "Close") if i - 1 >= 0 and self.position.size > 0 else 0.0)
                # compute daily return in a safe way
                if prev_total <= 0:
                    daily_ret = 0.0
                else:
                    daily_ret = (total_capital - prev_total) / max(prev_total, 1e-9)
            else:
                daily_ret = 0.0
            returns.append(daily_ret)

            # Update logs and equity tracking
            # Update peak equity using mark-to-market total_capital
            self.peak_equity = max(self.peak_equity, total_capital)
            # if trading disabled and drawdown recovers, re-enable gradually
            if self.trading_disabled and (self.peak_equity - total_capital) / max(self.peak_equity, 1e-9) < self.config["drawdown_scale_start"]:
                self.trading_disabled = False
                self.logger.info(f"{date} - Drawdown recovered below threshold. Re-enabling entries.")

        # Build performance DataFrame
        perf = pd.DataFrame({"equity": equity_curve}, index=df.index)
        perf["returns"] = returns
        perf["equity_mark_to_market"] = perf["equity"] + (df["Close"] * self.position.size if hasattr(self.position, "size") else 0.0)

        # Basic stats
        total_return = perf["equity"].iloc[-1] / self.initial_equity - 1.0
        ann_ret = annualized_return(pd.Series(returns))
        ann_vol = annualized_vol(pd.Series(returns))
        sr = sharpe_ratio(pd.Series(returns))
        mdd = max_drawdown(pd.Series(perf["equity"]))
        win_trades = [h for h in self.history if "pnl" in h and h["pnl"] > 0]
        loss_trades = [h for h in self.history if "pnl" in h and h["pnl"] <= 0]
        win_rate = len(win_trades) / max(1, (len(win_trades) + len(loss_trades)))
        stats = {
            "total_return": total_return,
            "annual_return": ann_ret,
            "annual_vol": ann_vol,
            "sharpe": sr,
            "max_drawdown": mdd,
            "win_rate": win_rate,
            "n_trades": len([h for h in self.history if h["action"].startswith("exit")]),
        }

        self.logger.info(f"Backtest complete. Total return: {total_return:.2%}, Annualized return: {ann_ret:.2%}, Sharpe: {sr:.2f}, MaxDD: {mdd:.2%}, Trades: {stats['n_trades']}")
        return {"perf": perf, "stats": stats, "history": self.history}


# Helper to get index name attribute for namedtuple rows
def IndexName(df: pd.DataFrame) -> str:
    # pandas naming for namedtuple: Index is the index name; when index has no name, attribute is Index
    # We can use df.index.name if set, else 'Index'
    name = df.index.name
    if name is None:
        return "Index"
    # If name contains spaces or characters, pandas uses it directly
    return name


# -------------------------
# Example usage (keeps architecture but does not run on import)
# -------------------------
if __name__ == "__main__":
    # Minimal sanity test: synthetic data if real data not provided
    import sys

    def generate_synthetic_data(n_days=500, seed=CONFIG["seed"]):
        np.random.seed(seed)
        dates = pd.bdate_range(end=pd.Timestamp.today(), periods=n_days)
        # simulate log returns with slight drift
        mu = 0.0002
        sigma = 0.01
        rets = np.random.normal(mu, sigma, size=n_days)
        price = 100 * np.exp(np.cumsum(rets))
        high = price * (1 + np.random.uniform(0.0, 0.01, size=n_days))
        low = price * (1 - np.random.uniform(0.0, 0.01, size=n_days))
        open_ = price * (1 + np.random.uniform(-0.005, 0.005, size=n_days))
        df = pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": price}, index=dates)
        return df

    data = generate_synthetic_data()
    ts = TradingSystem(CONFIG, logger)
    result = ts.run_backtest(data)
    stats = result["stats"]
    logger.info(f"Stats: {stats}")