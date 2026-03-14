import os
import json
import logging
import logging.handlers
from datetime import datetime
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd


# =========================
# Configuration & Logging
# =========================

DEFAULT_CONFIG: Dict[str, Any] = {
    "data_path": "data.csv",
    "start_cash": 1_000_000,
    "commission_per_share": 0.0005,  # proportion (0.05%)
    "fixed_commission": 0.0,  # fixed per-trade cost
    "slippage_proportion": 0.0005,  # slippage as proportion of price
    "vol_target": 0.08,  # annualized volatility target for portfolio sizing (8%)
    "vol_lookback": 21,  # lookback days for daily vol estimate
    "atr_period": 14,
    "atr_multiplier": 3.0,  # stop-loss = entry_price - ATR*multiplier for longs
    "ema_short": 12,
    "ema_long": 26,
    "signal_smooth": 5,  # smooth signals with EMA
    "max_position_size": 0.25,  # max fraction of equity in any single position
    "max_drawdown_limit": 0.20,  # stop trading / reduce to cash after hitting this DD (20%)
    "trailing_atr_multiplier": 2.0,
    "min_signal_threshold": 0.02,  # minimum signal to enter (price return threshold)
    "rebalance_freq": "B",  # business day frequency for rebalancing/signals
    "log_file": "trading_system.log",
    "log_level": "INFO",
    "results_path": "backtest_results.csv",
    "seed": 42,
}


def load_config(path: str = "config.json") -> Dict[str, Any]:
    cfg = DEFAULT_CONFIG.copy()
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                user_cfg = json.load(f)
            cfg.update(user_cfg)
        except Exception:
            pass
    return cfg


def setup_logger(cfg: Dict[str, Any]) -> logging.Logger:
    logger = logging.getLogger("TradingSystem")
    logger.setLevel(getattr(logging, cfg.get("log_level", "INFO").upper(), logging.INFO))
    if not logger.handlers:
        fh = logging.handlers.RotatingFileHandler(cfg.get("log_file", "trading_system.log"),
                                                  maxBytes=5_000_000, backupCount=5)
        fh.setLevel(logger.level)
        ch = logging.StreamHandler()
        ch.setLevel(logger.level)
        fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        fh.setFormatter(fmt)
        ch.setFormatter(fmt)
        logger.addHandler(fh)
        logger.addHandler(ch)
    return logger


# =========================
# Utilities & Indicators
# =========================

def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift(1)).abs()
    low_close = (df["Low"] - df["Close"].shift(1)).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(period, min_periods=1).mean()
    return atr


def ewma(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def annualize_vol(daily_returns: pd.Series, trading_days: int = 252) -> float:
    return daily_returns.std() * np.sqrt(trading_days)


def compute_vol_target_size(equity: float, price: float, recent_daily_ret: pd.Series,
                            vol_target: float, max_position_frac: float) -> int:
    # estimate daily vol from recent returns (use robust method)
    if recent_daily_ret.dropna().empty:
        return 0
    vol_est = annualize_vol(recent_daily_ret)
    if vol_est <= 0:
        return 0
    # target dollar volatility per position
    target_portfolio_vol = vol_target * equity
    # proportion of portfolio allocated to this instrument based on vol
    allocation = target_portfolio_vol / (vol_est * equity)
    # allocation fraction clipped to max position size
    allocation_frac = np.clip(allocation, -max_position_frac, max_position_frac)
    dollars = allocation_frac * equity
    if price <= 0 or np.isnan(price):
        return 0
    size = int(np.floor(dollars / price))
    return size


# =========================
# Strategy Class
# =========================

class Strategy:
    def __init__(self, df: pd.DataFrame, cfg: Dict[str, Any], logger: logging.Logger):
        self.df = df.copy().sort_index()
        self.cfg = cfg
        self.log = logger
        self.prepare_data()

    def prepare_data(self):
        df = self.df
        # Basic checks
        required_cols = {"Open", "High", "Low", "Close", "Volume"}
        if not required_cols.issubset(df.columns):
            self.log.error("Input data missing required OHLCV columns.")
            raise ValueError("Data must contain Open, High, Low, Close, Volume")

        # compute returns
        df["Return"] = df["Close"].pct_change().fillna(0)

        # EMA based momentum
        df["EMA_short"] = ewma(df["Close"], span=self.cfg["ema_short"])
        df["EMA_long"] = ewma(df["Close"], span=self.cfg["ema_long"])
        df["EMA_diff"] = (df["EMA_short"] - df["EMA_long"]) / df["EMA_long"]

        # Smooth signal
        df["Signal_raw"] = df["EMA_diff"]
        df["Signal"] = ewma(df["Signal_raw"], span=self.cfg["signal_smooth"])

        # ATR
        df["ATR"] = compute_atr(df, period=self.cfg["atr_period"])

        # rolling vol of returns
        df["DailyVol"] = df["Return"].rolling(self.cfg["vol_lookback"], min_periods=1).std()

        # shifting signals to avoid look-ahead
        df["Signal"] = df["Signal"].shift(1)
        df["EMA_short"] = df["EMA_short"].shift(1)
        df["EMA_long"] = df["EMA_long"].shift(1)
        df["ATR"] = df["ATR"].shift(1)
        df["DailyVol"] = df["DailyVol"].shift(1)

        self.df = df
        self.log.debug("Data prepared with indicators: EMA, ATR, Vol.")

    def generate_signals(self) -> pd.DataFrame:
        df = self.df
        cfg = self.cfg

        # Signal rules:
        # - Go long when EMA_short > EMA_long and smoothed signal > min threshold
        # - Go flat / exit when EMA_short < EMA_long or stop/trailing hit
        df["PositionSignal"] = 0
        long_condition = (df["EMA_short"] > df["EMA_long"]) & (df["Signal"].abs() > cfg["min_signal_threshold"])
        df.loc[long_condition, "PositionSignal"] = 1
        # allow shorting? For safety default to long-only; could extend: short when EMA_short < EMA_long
        short_condition = (df["EMA_short"] < df["EMA_long"]) & (df["Signal"].abs() > cfg["min_signal_threshold"])
        df.loc[short_condition, "PositionSignal"] = -1

        # position smoothing to avoid frequent churn: only change when sustained for n days
        # use rolling mode over 3 days
        df["PositionSignal"] = df["PositionSignal"].rolling(3, min_periods=1).apply(
            lambda x: pd.Series(x).mode().iat[0] if len(pd.Series(x).mode()) > 0 else x[-1]
        )

        self.df = df
        self.log.info("Signals generated.")
        return df[["Signal", "PositionSignal", "ATR", "DailyVol", "Close"]]

    def backtest(self) -> pd.DataFrame:
        cfg = self.cfg
        df = self.df.copy()
        self.log.info("Starting backtest.")
        # Setup backtest columns
        df["Position"] = 0  # number of shares
        df["Holdings"] = 0.0  # value of holdings
        df["Cash"] = 0.0
        df["Equity"] = 0.0
        df["Trade"] = 0  # shares traded
        df["TradeCost"] = 0.0
        df["Pnl"] = 0.0
        df["StopPrice"] = np.nan
        df["TrailingStop"] = np.nan

        cash = cfg["start_cash"]
        position = 0
        entry_price = np.nan
        last_equity_peak = cash
        max_drawdown = 0.0

        np.random.seed(cfg.get("seed", 42))

        for i, (date, row) in enumerate(df.iterrows()):
            close = row["Close"]
            price = close  # assume execution at close for simplicity
            signal = int(row["PositionSignal"])
            atr = row["ATR"]
            daily_vol = row["DailyVol"]

            # compute equity before trading (mark-to-market)
            holdings = position * price
            equity = cash + holdings

            # update peak & drawdown
            last_equity_peak = max(last_equity_peak, equity)
            drawdown = (last_equity_peak - equity) / last_equity_peak if last_equity_peak > 0 else 0.0
            max_drawdown = max(max_drawdown, drawdown)

            # hard drawdown stop: if exceeded stop trading and cut positions
            if drawdown >= cfg["max_drawdown_limit"]:
                if position != 0:
                    # liquidate position immediately
                    trade_shares = -position
                    trade_cost = self._apply_trade_cost(abs(trade_shares), price)
                    cash += trade_shares * price - trade_cost
                    self.log.warning(f"{date.date()}: Max drawdown exceeded ({drawdown:.2%}). Liquidating position of {position} shares.")
                    position = 0
                    entry_price = np.nan
                df.at[date, "Position"] = position
                df.at[date, "Holdings"] = position * price
                df.at[date, "Cash"] = cash
                df.at[date, "Equity"] = cash + position * price
                df.at[date, "Pnl"] = 0.0
                df.at[date, "StopPrice"] = np.nan
                df.at[date, "TrailingStop"] = np.nan
                continue

            # Determine target size via volatility targeting
            target_shares = 0
            if not np.isnan(daily_vol) and daily_vol > 0:
                recent_returns = df["Return"].loc[:date].tail(cfg["vol_lookback"])
                target_shares = compute_vol_target_size(
                    equity=equity,
                    price=price,
                    recent_daily_ret=recent_returns,
                    vol_target=cfg["vol_target"],
                    max_position_frac=cfg["max_position_size"]
                )
                # apply signal orientation
                target_shares = int(np.sign(signal) * abs(target_shares))

            # Stop-loss and trailing stop enforcement
            stop_price = df.at[date, "StopPrice"]
            trailing_stop = df.at[date, "TrailingStop"]
            # If we have a position, check stops
            exit_due_to_stop = False
            if position != 0:
                if stop_price is not np.nan and not pd.isna(stop_price):
                    if (position > 0 and price <= stop_price) or (position < 0 and price >= stop_price):
                        exit_due_to_stop = True
                        self.log.info(f"{date.date()}: Stop-loss hit. Price {price:.2f} vs stop {stop_price:.2f}.")
                if trailing_stop is not np.nan and not pd.isna(trailing_stop):
                    if (position > 0 and price <= trailing_stop) or (position < 0 and price >= trailing_stop):
                        exit_due_to_stop = True
                        self.log.info(f"{date.date()}: Trailing stop hit. Price {price:.2f} vs trailing {trailing_stop:.2f}.")

            # Execution logic with some hysteresis (only change if meaningful)
            change = target_shares - position
            # Only trade if change is sufficiently large to justify costs
            if abs(change) * price > 0.001 * equity:  # threshold: trades smaller than 0.1% of equity ignored
                # If exit due to stop, go to zero
                if exit_due_to_stop:
                    trade_shares = -position
                else:
                    trade_shares = target_shares - position

                # apply trade cost model and slippage
                trade_cost = self._apply_trade_cost(abs(trade_shares), price)
                slippage = price * cfg["slippage_proportion"] * np.sign(trade_shares)
                executed_price = price + slippage

                # Update cash and position
                cash -= executed_price * trade_shares + trade_cost
                position += trade_shares
                # record entry price if we opened a new position
                if trade_shares != 0 and entry_price is np.nan:
                    entry_price = executed_price
                if trade_shares != 0 and position != 0:
                    # set stop-loss and trailing stop after entering
                    if atr > 0:
                        stop_price = executed_price - np.sign(position) * cfg["atr_multiplier"] * atr
                        trailing_stop = executed_price - np.sign(position) * cfg["trailing_atr_multiplier"] * atr
                    else:
                        stop_price = np.nan
                        trailing_stop = np.nan
                # If fully liquidated, clear entry & stops
                if position == 0:
                    entry_price = np.nan
                    stop_price = np.nan
                    trailing_stop = np.nan

                df.at[date, "Trade"] = trade_shares
                df.at[date, "TradeCost"] = trade_cost + abs(slippage * trade_shares)  # approximate
                self.log.debug(f"{date.date()}: Executed trade {trade_shares} @ {executed_price:.2f}, cost {trade_cost:.2f}, slippage {slippage * trade_shares:.2f}")

            # Update trailing stop if position moves favorably
            if position != 0 and not pd.isna(price) and not pd.isna(trailing_stop):
                if position > 0:
                    new_trail = price - cfg["trailing_atr_multiplier"] * atr if atr > 0 else trailing_stop
                    trailing_stop = max(trailing_stop, new_trail)
                else:
                    new_trail = price + cfg["trailing_atr_multiplier"] * atr if atr > 0 else trailing_stop
                    trailing_stop = min(trailing_stop, new_trail)

            # Update row state
            df.at[date, "Position"] = position
            df.at[date, "Holdings"] = position * price
            df.at[date, "Cash"] = cash
            df.at[date, "Equity"] = cash + position * price
            df.at[date, "StopPrice"] = stop_price
            df.at[date, "TrailingStop"] = trailing_stop
            df.at[date, "Pnl"] = (df.at[date, "Equity"] - equity)  # change in equity since before trades (realized+unrealized)

            # record metrics
            if i % 50 == 0:
                self.log.debug(f"{date.date()}: Equity={df.at[date, 'Equity']:.2f}, Position={position}, Cash={cash:.2f}, MaxDD={max_drawdown:.2%}")

        self.log.info(f"Backtest complete. Max drawdown experienced: {max_drawdown:.2%}")
        # Write results
        try:
            df.to_csv(cfg.get("results_path", "backtest_results.csv"))
            self.log.info(f"Backtest results written to {cfg.get('results_path', 'backtest_results.csv')}")
        except Exception as e:
            self.log.error(f"Failed to write results: {e}")

        return df

    def _apply_trade_cost(self, shares: float, price: float) -> float:
        cfg = self.cfg
        if shares <= 0:
            return 0.0
        # cost proportional to trade size plus fixed
        cost = shares * price * cfg["commission_per_share"] + cfg.get("fixed_commission", 0.0)
        return cost


# =========================
# Orchestration
# =========================

def load_data(path: str, logger: logging.Logger) -> pd.DataFrame:
    if not os.path.exists(path):
        logger.error(f"Data file not found: {path}")
        raise FileNotFoundError(path)
    df = pd.read_csv(path, parse_dates=True, index_col=0)
    # ensure proper columns
    df_columns = {c.lower(): c for c in df.columns}
    # normalize column names if common lowercase
    normalize_map = {}
    for want in ["Open", "High", "Low", "Close", "Volume"]:
        if want not in df.columns and want.lower() in df_columns:
            normalize_map[df_columns[want.lower()]] = want
    if normalize_map:
        df = df.rename(columns=normalize_map)
    return df


def run(config_path: str = "config.json"):
    cfg = load_config(config_path)
    logger = setup_logger(cfg)
    logger.info("Trading system started.")
    logger.debug(f"Config: {json.dumps(cfg, default=str)}")

    try:
        df = load_data(cfg["data_path"], logger)
    except Exception as e:
        logger.exception("Failed to load data.")
        raise

    strat = Strategy(df, cfg, logger)
    strat.generate_signals()
    results = strat.backtest()

    # compute summary stats
    eq = results["Equity"].ffill().fillna(cfg["start_cash"])
    returns = eq.pct_change().fillna(0)
    ann_return = (eq.iloc[-1] / eq.iloc[0]) ** (252.0 / len(results)) - 1 if len(results) > 0 else 0.0
    ann_vol = annualize_vol(returns)
    sharpe = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0.0
    max_dd = ((eq.cummax() - eq) / eq.cummax()).max()

    logger.info(f"Performance: Annual Return {ann_return:.2%}, Annual Vol {ann_vol:.2%}, Sharpe {sharpe:.2f}, MaxDD {max_dd:.2%}")
    print("Performance summary:")
    print(f"  Annual Return: {ann_return:.2%}")
    print(f"  Annual Volatility: {ann_vol:.2%}")
    print(f"  Sharpe Ratio: {sharpe:.2f}")
    print(f"  Max Drawdown: {max_dd:.2%}")

    return results


if __name__ == "__main__":
    # default run
    run()