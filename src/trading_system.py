import logging
import math
import sys
from dataclasses import dataclass, field
from typing import Dict, Optional, List

import numpy as np
import pandas as pd

# =========================
# Configuration (preserved)
# =========================
CONFIG = {
    "data": {
        "csv_path": "price_data.csv",  # expects columns: datetime, open, high, low, close, volume
        "datetime_col": "datetime",
        "price_col": "close",
    },
    "strategy": {
        "fast_ema": 20,
        "slow_ema": 50,
        "atr_period": 14,
        "atr_multiplier_initial_stop": 3.0,
        "atr_trail_multiplier": 2.0,
        "min_atr": 1e-6,
        "trade_cooldown_bars": 1,  # minimum bars between trades
        "trend_filter": True,
        "trend_ema": 200,
        "allow_short": False,  # keep long-only to reduce drawdown
    },
    "risk": {
        "portfolio_risk_target_pct": 0.01,  # target risk per trade as fraction of equity (1%)
        "max_position_pct_of_equity": 0.20,  # maximum position size as fraction of equity (20%)
        "max_drawdown_pct": 0.20,  # portfolio-level max drawdown cap: reduce risk when reached
        "drawdown_recovery_pct": 0.10,  # reduce risk until equity recovers this amount
        "volatility_target_annualized": 0.10,  # used to scale exposures if desired
        "max_leverage": 3.0,
    },
    "execution": {
        "commission_per_trade": 1.0,  # flat commission per trade
        "slippage_pct": 0.0005,  # 0.05% slippage per fill
    },
    "backtest": {
        "initial_capital": 100_000.0,
        "min_bars": 300,
        "verbose": True,
    },
    "logging": {
        "level": "INFO",
    },
}

# =========================
# Logging (preserved)
# =========================
logger = logging.getLogger("trading_system")
log_level = getattr(logging, CONFIG["logging"].get("level", "INFO").upper(), logging.INFO)
logger.setLevel(log_level)
if not logger.handlers:
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(log_level)
    fmt = logging.Formatter("%(asctime)s %(levelname)-8s %(name)s - %(message)s")
    ch.setFormatter(fmt)
    logger.addHandler(ch)


# =========================
# Utilities / Indicators
# =========================
def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def atr(df: pd.DataFrame, period: int, high_col="high", low_col="low", close_col="close") -> pd.Series:
    high = df[high_col]
    low = df[low_col]
    close = df[close_col]
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_series = tr.rolling(window=period, min_periods=1).mean()
    return atr_series


def sharpe_ratio(returns: pd.Series, annualization=252) -> float:
    if returns.empty:
        return 0.0
    mu = returns.mean() * annualization
    sigma = returns.std() * math.sqrt(annualization)
    return (mu / sigma) if sigma != 0 else 0.0


def max_drawdown(equity: pd.Series) -> float:
    running_max = equity.cummax()
    drawdown = (equity - running_max) / running_max
    return drawdown.min()


def sortino_ratio(returns: pd.Series, annualization=252) -> float:
    if returns.empty:
        return 0.0
    mu = returns.mean() * annualization
    negative_returns = returns[returns < 0]
    downside_std = negative_returns.std() * math.sqrt(annualization) if not negative_returns.empty else 0.0
    return (mu / downside_std) if downside_std != 0 else 0.0


# =========================
# Data Handler
# =========================
class DataHandler:
    def __init__(self, df: pd.DataFrame, config: Dict):
        self.df = df.copy().reset_index(drop=True)
        self.datetime_col = config["data"]["datetime_col"]
        self.price_col = config["data"]["price_col"]
        # ensure required cols
        required = {"open", "high", "low", "close"}
        if not required.issubset(set(self.df.columns)):
            raise ValueError(f"Data must contain columns: {required}")
        # ensure datetime
        if self.datetime_col in self.df.columns:
            self.df[self.datetime_col] = pd.to_datetime(self.df[self.datetime_col])
            self.df.set_index(self.datetime_col, inplace=True)

    def get_dataframe(self) -> pd.DataFrame:
        return self.df


# =========================
# Strategy
# - preserves logging and config usage
# =========================
class Strategy:
    def __init__(self, df: pd.DataFrame, cfg: Dict):
        self.df = df.copy()
        self.cfg = cfg
        self._prepare_indicators()
        self.last_trade_idx = -9999

    def _prepare_indicators(self):
        s = self.df["close"]
        self.df["ema_fast"] = ema(s, self.cfg["strategy"]["fast_ema"])
        self.df["ema_slow"] = ema(s, self.cfg["strategy"]["slow_ema"])
        self.df["trend_ema"] = ema(s, self.cfg["strategy"]["trend_ema"])
        self.df["atr"] = atr(self.df, self.cfg["strategy"]["atr_period"])
        # fill small ATRs
        self.df["atr"] = self.df["atr"].clip(lower=self.cfg["strategy"]["min_atr"])
        logger.info("Indicators computed: EMA(%d), EMA(%d), ATR(%d)",
                    self.cfg["strategy"]["fast_ema"], self.cfg["strategy"]["slow_ema"],
                    self.cfg["strategy"]["atr_period"])

    def generate_signals(self) -> pd.DataFrame:
        """
        Generates signals:
        - 1 => long entry signal
        - 0 => no position / flat
        For risk reduction, we keep strategy long-only by default and enforce trend EMA filter.
        """
        df = self.df.copy()
        df["signal"] = 0
        fast = df["ema_fast"]
        slow = df["ema_slow"]
        trend = df["trend_ema"]

        # Basic entry: fast crosses above slow
        cross_up = (fast > slow) & (fast.shift(1) <= slow.shift(1))
        cross_down = (fast < slow) & (fast.shift(1) >= slow.shift(1))

        for idx in df.index:
            i = df.index.get_loc(idx)
            if not cross_up.iloc[i] and not cross_down.iloc[i]:
                continue

            # Enforce cooldown to reduce overtrading
            if (i - self.last_trade_idx) < self.cfg["strategy"]["trade_cooldown_bars"]:
                logger.debug("Cooldown enforced at bar %s", idx)
                continue

            if cross_up.iloc[i]:
                # trend filter: only take longs in uptrend if enabled
                if self.cfg["strategy"]["trend_filter"] and df["close"].iloc[i] < trend.iloc[i]:
                    logger.debug("Trend filter blocked long at %s", idx)
                    continue
                df.at[idx, "signal"] = 1
                self.last_trade_idx = i
                logger.debug("Long signal at %s, price %.2f", idx, df["close"].iloc[i])
            elif cross_down.iloc[i] and self.cfg["strategy"].get("allow_short", False):
                df.at[idx, "signal"] = -1
                self.last_trade_idx = i
                logger.debug("Short signal at %s, price %.2f", idx, df["close"].iloc[i])

        return df[["close", "high", "low", "atr", "signal", "ema_fast", "ema_slow", "trend_ema"]]


# =========================
# Risk Manager
# - volatility targeting, ATR-based stops, drawdown caps
# =========================
@dataclass
class RiskManager:
    config: Dict
    capital: float
    peak_equity: float = field(default=0.0)
    max_drawdown_reached: bool = field(default=False)
    reduced_risk_multiplier: float = field(default=1.0)

    def update_equity(self, equity: float):
        if equity > self.peak_equity:
            self.peak_equity = equity
        dd = (self.peak_equity - equity) / max(1.0, self.peak_equity)
        if dd >= self.config["risk"]["max_drawdown_pct"]:
            if not self.max_drawdown_reached:
                logger.warning("Max drawdown threshold reached: %.2f%%, reducing risk", dd * 100)
            self.max_drawdown_reached = True
            self.reduced_risk_multiplier = 0.5  # reduce risk by half on severe drawdown
        elif self.max_drawdown_reached and dd <= self.config["risk"]["drawdown_recovery_pct"]:
            logger.info("Drawdown recovered to %.2f%%, restoring risk profile", dd * 100)
            self.max_drawdown_reached = False
            self.reduced_risk_multiplier = 1.0

    def position_size(self, price: float, atr: float) -> int:
        """
        Determine position size in units/contracts using volatility targeting:
        - risk per trade = target_pct * equity * reduced_risk_multiplier
        - stop distance = atr * atr_multiplier_initial_stop
        - position_size = risk_per_trade / (stop_distance * price)  (for cash per share model)
        Caps by max_position_pct_of_equity and max_leverage.
        """
        equity = self.capital
        cfg = self.config
        risk_target = cfg["risk"]["portfolio_risk_target_pct"] * equity * self.reduced_risk_multiplier
        stop_distance = max(cfg["strategy"]["atr_multiplier_initial_stop"] * atr, cfg["strategy"]["min_atr"])
        # dollar risk per share = stop_distance * price
        dollar_risk_per_unit = stop_distance * price
        if dollar_risk_per_unit <= 0:
            return 0
        raw_units = risk_target / dollar_risk_per_unit
        # cap by max position size (dollars)
        max_position_dollars = cfg["risk"]["max_position_pct_of_equity"] * equity
        max_units_by_cap = max_position_dollars / price
        units = int(max(0, min(raw_units, max_units_by_cap)))
        # enforce leverage cap (approx)
        approx_position_dollars = units * price
        leverage = approx_position_dollars / max(1.0, equity)
        if leverage > cfg["risk"]["max_leverage"]:
            units = int((cfg["risk"]["max_leverage"] * equity) / price)
        logger.debug("Position sizing: price=%.2f atr=%.4f risk_target=%.2f raw_units=%.2f final_units=%d",
                     price, atr, risk_target, raw_units, units)
        return max(0, units)


# =========================
# Portfolio & Execution
# =========================
class Portfolio:
    def __init__(self, initial_capital: float, config: Dict):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.position = 0  # number of units (positive for long)
        self.entry_price = 0.0
        self.equity_curve = []
        self.config = config
        self.trades: List[Dict] = []
        self.current_equity = initial_capital

    def apply_fill(self, side: int, units: int, fill_price: float, atr: float, dt):
        """
        side: 1 for buy, -1 for sell/exit
        units: number of units to buy/sell (positive)
        """
        commission = self.config["execution"]["commission_per_trade"]
        slippage = fill_price * self.config["execution"]["slippage_pct"]
        fill_price_with_slippage = fill_price + slippage if side == 1 else fill_price - slippage

        if side == 1:
            cost = units * fill_price_with_slippage + commission
            if cost > self.cash:
                # partial fill if not enough cash
                units = int(self.cash / (fill_price_with_slippage + 1e-12))
                cost = units * fill_price_with_slippage + commission
            if units <= 0:
                logger.debug("No units purchased due to cash constraints")
                return
            self.position += units
            self.cash -= cost
            # new average entry price
            if self.entry_price == 0:
                self.entry_price = fill_price_with_slippage
            else:
                self.entry_price = ((self.entry_price * (self.position - units)) + (fill_price_with_slippage * units)) / self.position
            self.trades.append({
                "datetime": dt,
                "side": "BUY",
                "units": units,
                "price": fill_price_with_slippage,
                "commission": commission,
                "atr": atr,
            })
            logger.info("BUY %d units at %.4f (slippage %.4f) on %s", units, fill_price_with_slippage, slippage, dt)
        elif side == -1:
            # exit or reduce
            units_to_sell = min(units, self.position)
            if units_to_sell <= 0:
                logger.debug("No units to sell")
                return
            proceeds = units_to_sell * fill_price_with_slippage - commission
            self.position -= units_to_sell
            self.cash += proceeds
            realized_pnl = units_to_sell * (fill_price_with_slippage - self.entry_price)
            if self.position == 0:
                self.entry_price = 0.0
            self.trades.append({
                "datetime": dt,
                "side": "SELL",
                "units": units_to_sell,
                "price": fill_price_with_slippage,
                "commission": commission,
                "realized_pnl": realized_pnl,
                "atr": atr,
            })
            logger.info("SELL %d units at %.4f (slippage %.4f) on %s; realized pnl %.2f", units_to_sell, fill_price_with_slippage, slippage, dt, realized_pnl)

    def mark_to_market(self, price: float, dt):
        market_value = self.position * price
        equity = self.cash + market_value
        self.current_equity = equity
        self.equity_curve.append({"datetime": dt, "equity": equity})
        logger.debug("Mark to market at %s price %.4f position %d cash %.2f equity %.2f",
                     dt, price, self.position, self.cash, equity)
        return equity

    def unrealized_pnl(self, price: float) -> float:
        if self.position == 0:
            return 0.0
        return (price - self.entry_price) * self.position

    def get_equity_series(self) -> pd.Series:
        if not self.equity_curve:
            return pd.Series(dtype=float)
        df = pd.DataFrame(self.equity_curve).set_index("datetime")
        return df["equity"]


# =========================
# Backtester
# =========================
class Backtester:
    def __init__(self, df: pd.DataFrame, config: Dict):
        self.df = df.copy()
        self.cfg = config
        self.initial_capital = config["backtest"]["initial_capital"]
        self.portfolio = Portfolio(self.initial_capital, config)
        self.risk_manager = RiskManager(config, self.portfolio.current_equity)
        self.strategy = Strategy(self.df, self.cfg)
        self.signals = self.strategy.generate_signals()

    def run(self):
        df = self.signals.copy()
        last_signal = 0
        last_trade_bar = -9999
        for i, idx in enumerate(df.index):
            row = df.loc[idx]
            price = float(row["close"])
            atr_val = float(row["atr"])
            signal = int(row["signal"])

            # Update equity and risk manager at each bar
            equity = self.portfolio.mark_to_market(price, idx)
            self.risk_manager.capital = equity
            self.risk_manager.update_equity(equity)

            # Manage existing position: apply ATR-based trailing stops
            if self.portfolio.position > 0:
                # compute stop: trailing at entry or ATR-based
                trail_stop = price - self.cfg["strategy"]["atr_trail_multiplier"] * atr_val
                # simple strategy: if price falls below trail_stop, exit
                if price <= trail_stop:
                    logger.info("Trailing ATR stop triggered at %s price %.4f trail_stop %.4f", idx, price, trail_stop)
                    self.portfolio.apply_fill(side=-1, units=self.portfolio.position, fill_price=price, atr=atr_val, dt=idx)
                    last_trade_bar = i

            # Entry logic: only enter when a signal appears and no existing position
            if signal == 1 and self.portfolio.position == 0 and (i - last_trade_bar) >= self.cfg["strategy"]["trade_cooldown_bars"]:
                # compute position size
                units = self.risk_manager.position_size(price=price, atr=atr_val)
                if units > 0:
                    self.portfolio.apply_fill(side=1, units=units, fill_price=price, atr=atr_val, dt=idx)
                    last_trade_bar = i
                else:
                    logger.debug("No units allocated at %s due to sizing=0", idx)

            # Exit logic on cross down if allowed or on trend break
            if signal == 0 and self.portfolio.position > 0:
                # If fast EMAs cross down or trend filter fails, exit to reduce drawdown
                fast = row["ema_fast"]
                slow = row["ema_slow"]
                trend = row["trend_ema"]
                if fast < slow or (self.cfg["strategy"]["trend_filter"] and price < trend):
                    self.portfolio.apply_fill(side=-1, units=self.portfolio.position, fill_price=price, atr=atr_val, dt=idx)
                    last_trade_bar = i

            # Update risk manager capital at each bar
            self.risk_manager.capital = self.portfolio.current_equity

        # Final mark-to-market
        final_price = float(df["close"].iloc[-1])
        final_equity = self.portfolio.mark_to_market(final_price, df.index[-1])
        self.risk_manager.update_equity(final_equity)

        logger.info("Backtest completed. Final equity: %.2f", final_equity)
        return self.results()

    def results(self) -> Dict:
        equity_series = self.portfolio.get_equity_series()
        if equity_series.empty or len(equity_series) < 2:
            return {}
        equity_series = equity_series.sort_index()
        returns = equity_series.pct_change().dropna()
        total_return = equity_series.iloc[-1] / equity_series.iloc[0] - 1.0
        ann_return = (1.0 + total_return) ** (252.0 / len(equity_series)) - 1.0 if len(equity_series) > 0 else 0.0
        sr = sharpe_ratio(returns)
        sortino = sortino_ratio(returns)
        mdd = max_drawdown(equity_series)
        trades = pd.DataFrame(self.portfolio.trades)
        winning_trades = trades[trades.get("realized_pnl", 0) > 0].shape[0] if not trades.empty else 0
        losing_trades = trades[trades.get("realized_pnl", 0) <= 0].shape[0] if not trades.empty else 0
        win_rate = winning_trades / max(1, (winning_trades + losing_trades))
        results = {
            "initial_capital": self.initial_capital,
            "final_equity": float(equity_series.iloc[-1]),
            "total_return": float(total_return),
            "annual_return_est": float(ann_return),
            "sharpe": float(sr),
            "sortino": float(sortino),
            "max_drawdown": float(mdd),
            "trades": trades.to_dict("records"),
            "num_trades": len(trades),
            "win_rate": float(win_rate),
            "equity_curve": equity_series,
        }
        logger.info("Results: Total Return %.2f%% Annual Return %.2f%% Sharpe %.2f MaxDD %.2f%% Trades %d WinRate %.2f",
                    results["total_return"] * 100, results["annual_return_est"] * 100, results["sharpe"], results["max_drawdown"] * 100,
                    results["num_trades"], results["win_rate"])
        return results


# =========================
# Entrypoint
# =========================
def load_csv(path: str, datetime_col: str = "datetime") -> pd.DataFrame:
    df = pd.read_csv(path)
    if datetime_col in df.columns:
        df[datetime_col] = pd.to_datetime(df[datetime_col])
    # ensure required columns present
    required = {"open", "high", "low", "close"}
    if not required.issubset(set(df.columns)):
        raise ValueError(f"CSV must contain columns: {required}")
    return df


def main():
    cfg = CONFIG
    try:
        df = load_csv(cfg["data"]["csv_path"], cfg["data"]["datetime_col"])
    except Exception as e:
        logger.error("Failed to load data: %s", e)
        return

    if len(df) < cfg["backtest"]["min_bars"]:
        logger.warning("Insufficient bars (%d) for meaningful backtest; need at least %d", len(df), cfg["backtest"]["min_bars"])

    data_handler = DataHandler(df, cfg)
    market_df = data_handler.get_dataframe()
    bt = Backtester(market_df, cfg)
    results = bt.run()

    # Logging results summary (preserved logging)
    logger.info("Backtest summary:")
    logger.info("Initial capital: %.2f", results.get("initial_capital", 0.0))
    logger.info("Final equity: %.2f", results.get("final_equity", 0.0))
    logger.info("Total return: %.2f%%", results.get("total_return", 0.0) * 100)
    logger.info("Annual return (est): %.2f%%", results.get("annual_return_est", 0.0) * 100)
    logger.info("Sharpe ratio: %.2f", results.get("sharpe", 0.0))
    logger.info("Sortino ratio: %.2f", results.get("sortino", 0.0))
    logger.info("Max drawdown: %.2f%%", results.get("max_drawdown", 0.0) * 100)
    logger.info("Number of trades: %d", results.get("num_trades", 0))
    logger.info("Win rate: %.2f", results.get("win_rate", 0.0))

    # Optionally return results object for programmatic consumption
    return results


if __name__ == "__main__":
    main()