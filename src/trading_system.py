import logging
import json
import os
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd

# ---------------------------
# Configuration (can be overridden by config.json)
# ---------------------------
DEFAULT_CONFIG: Dict[str, Any] = {
    "risk": {
        "risk_per_trade": 0.01,           # fraction of equity to risk per trade
        "max_position_fraction": 0.25,    # max fraction of equity allocated to one position
        "max_total_alloc": 0.8,           # max fraction of equity invested in all positions
        "max_drawdown": 0.20,             # if reached, trading will pause for cooldown_days
        "cooldown_days": 10
    },
    "strategy": {
        "fast_ema": 50,
        "slow_ema": 200,
        "atr_period": 14,
        "atr_entry_multiplier": 1.0,      # use ATR to scale entry aggressiveness
        "stop_atr_multiplier": 3.0,       # stoploss distance in ATRs
        "trailing_atr_multiplier": 2.0,   # trailing stop in ATRs
        "takeprofit_atr_multiplier": 6.0, # initial takeprofit target in ATRs
        "min_volatility_atr": 0.005,      # avoid entries if ATR/P > threshold (too volatile)
        "min_gap_to_ema": 0.0,            # require price to be above/below EMA by fraction to consider
        "require_trend": True             # require fast_ema > slow_ema for longs and reverse for shorts
    },
    "execution": {
        "slippage": 0.0,
        "commission_per_trade": 0.0,
        "tick_size": 0.0
    },
    "logging": {
        "logfile": "trading_system.log",
        "level": "INFO"
    },
    "backtest": {
        "initial_cash": 100000.0
    }
}


# ---------------------------
# Logging and config loader
# ---------------------------
def load_config(path: Optional[str] = "config.json") -> Dict[str, Any]:
    config = DEFAULT_CONFIG.copy()
    if path and os.path.exists(path):
        try:
            with open(path, "r") as f:
                user_conf = json.load(f)
            # deep update
            def deep_update(d, u):
                for k, v in u.items():
                    if isinstance(v, dict):
                        d[k] = deep_update(d.get(k, {}), v)
                    else:
                        d[k] = v
                return d
            config = deep_update(config, user_conf)
        except Exception as e:
            logging.getLogger().warning(f"Failed to load config {path}: {e}")
    return config


def setup_logging(config: Dict[str, Any]):
    log_conf = config.get("logging", {})
    level = getattr(logging, log_conf.get("level", "INFO").upper(), logging.INFO)
    logfile = log_conf.get("logfile", "trading_system.log")

    logger = logging.getLogger()
    logger.setLevel(level)
    # Remove existing handlers
    for h in list(logger.handlers):
        logger.removeHandler(h)

    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    ch.setLevel(level)
    logger.addHandler(ch)

    fh = logging.FileHandler(logfile)
    fh.setFormatter(fmt)
    fh.setLevel(level)
    logger.addHandler(fh)


# ---------------------------
# Utility indicators
# ---------------------------
def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(period, min_periods=1).mean()


# ---------------------------
# Strategy and Backtester
# ---------------------------
class Strategy:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.risk_cfg = config["risk"]
        self.strat_cfg = config["strategy"]
        self.exec_cfg = config["execution"]
        self.state = {
            "cooldown_until": None
        }
        # Logging preserved as required
        logging.info("Strategy initialized with config: %s", json.dumps(self.config, indent=2))

    def prepare_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["EMA_fast"] = ema(df["Close"], self.strat_cfg["fast_ema"])
        df["EMA_slow"] = ema(df["Close"], self.strat_cfg["slow_ema"])
        df["ATR"] = atr(df, self.strat_cfg["atr_period"])
        # normalized ATR as fraction of price
        df["ATR_pct"] = df["ATR"] / df["Close"]
        # trend
        df["trend_long"] = df["EMA_fast"] > df["EMA_slow"]
        df["trend_short"] = df["EMA_fast"] < df["EMA_slow"]
        # signal placeholder
        df["signal"] = 0
        return df

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        cfg = self.strat_cfg
        df = df.copy()
        # Simple crossover filter with volatility and ATR-based sizing constraints
        # Long signal: price > EMA_fast and EMA_fast > EMA_slow (trend), price gap requirement optional
        price = df["Close"]
        gap = cfg["min_gap_to_ema"]
        long_condition = (price > df["EMA_fast"] * (1.0 + gap))
        short_condition = (price < df["EMA_fast"] * (1.0 - gap))

        if cfg.get("require_trend", True):
            long_condition &= df["trend_long"]
            short_condition &= df["trend_short"]

        # Avoid extreme volatility days: ATR pct too high -> avoid new entries
        vol_ok = df["ATR_pct"] <= cfg["min_volatility_atr"]

        entries_long = long_condition & vol_ok
        entries_short = short_condition & vol_ok

        # Signal values: 1 = long entry, -1 = short entry
        df.loc[entries_long & ~entries_long.shift(1).fillna(False), "signal"] = 1
        df.loc[entries_short & ~entries_short.shift(1).fillna(False), "signal"] = -1
        return df

    def position_size(self, price: float, atr: float, equity: float) -> int:
        """
        Compute number of shares/contracts to buy given risk per trade and ATR stop placement.
        Uses conservative fixed-fraction risk per trade divided by ATR*stop_multiplier to get quantity.
        """
        rp = self.risk_cfg["risk_per_trade"]
        stop_atr = max(0.0001, self.strat_cfg["stop_atr_multiplier"] * atr)  # price units
        # Dollars at risk per unit = stop_atr
        dollars_risk_per_unit = stop_atr
        if dollars_risk_per_unit <= 0:
            return 0
        max_dollars_risk = equity * rp
        qty = int(np.floor(max_dollars_risk / dollars_risk_per_unit))
        # cap by max position fraction
        max_alloc = equity * self.risk_cfg["max_position_fraction"]
        max_qty_by_alloc = int(np.floor(max_alloc / price)) if price > 0 else qty
        qty = min(qty, max_qty_by_alloc)
        return max(0, qty)

    def run_backtest(self, df: pd.DataFrame, initial_cash: float = 100000.0) -> Dict[str, Any]:
        """
        Simple event-driven backtester that processes bar by bar.
        Preserves logging of trades, stops, and configuration.
        Returns performance summary and trade list.
        """
        data = df.copy().reset_index(drop=True)
        data = self.prepare_indicators(data)
        data = self.generate_signals(data)

        cash = initial_cash
        equity = initial_cash
        position = 0  # positive for long, negative for short
        entry_price = 0.0
        entry_atr = 0.0
        trailing_stop = None
        trade_list: List[Dict[str, Any]] = []
        peak_equity = equity
        max_dd = 0.0
        last_trade_idx = None

        for i, row in data.iterrows():
            date = row["Date"] if "Date" in row else row.get("Datetime", None)
            price = row["Close"]
            atr_val = row["ATR"]
            signal = int(row["signal"])
            # enforce cooldown logic
            if self.state.get("cooldown_until") and date is not None:
                if isinstance(self.state["cooldown_until"], str):
                    cooldown_until_dt = datetime.fromisoformat(self.state["cooldown_until"])
                else:
                    cooldown_until_dt = self.state["cooldown_until"]
                if date < cooldown_until_dt:
                    allowed_to_trade = False
                else:
                    allowed_to_trade = True
                    self.state["cooldown_until"] = None
            else:
                allowed_to_trade = True

            # Update current equity (mark-to-market)
            mtm = position * price
            equity = cash + mtm

            # Update drawdown
            if equity > peak_equity:
                peak_equity = equity
            dd = (peak_equity - equity) / peak_equity if peak_equity > 0 else 0.0
            if dd > max_dd:
                max_dd = dd
            # Check for max_drawdown trigger
            if max_dd >= self.risk_cfg["max_drawdown"] and self.state.get("cooldown_until") is None:
                # set cooldown
                cooldown_days = self.risk_cfg.get("cooldown_days", 10)
                if isinstance(date, pd.Timestamp):
                    cooldown_until = date + pd.Timedelta(days=cooldown_days)
                elif isinstance(date, datetime):
                    cooldown_until = date + timedelta(days=cooldown_days)
                else:
                    cooldown_until = None
                self.state["cooldown_until"] = cooldown_until
                logging.warning("Max drawdown reached (%.2f%%). Pausing trading until %s", max_dd * 100, cooldown_until)

            # Manage existing position: stops and trailing stop
            if position != 0:
                # stop loss and trailing stop logic
                stop_distance = self.strat_cfg["stop_atr_multiplier"] * entry_atr
                trailing_dist = self.strat_cfg["trailing_atr_multiplier"] * entry_atr
                # For longs
                if position > 0:
                    stop_price = entry_price - stop_distance
                    # update trailing stop to max of previous and current (for long)
                    new_trailing = price - trailing_dist
                    if trailing_stop is None or new_trailing > trailing_stop:
                        trailing_stop = new_trailing
                    exit_by_stop = price <= stop_price
                    exit_by_trail = trailing_stop is not None and price <= trailing_stop
                    takeprofit_price = entry_price + self.strat_cfg["takeprofit_atr_multiplier"] * entry_atr
                    exit_by_tp = price >= takeprofit_price
                    if exit_by_stop or exit_by_trail or exit_by_tp:
                        # exit at close price with slippage & commission
                        exit_price = price - self.exec_cfg.get("slippage", 0.0)
                        proceeds = position * exit_price
                        commission = abs(position) * self.exec_cfg.get("commission_per_trade", 0.0)
                        cash += proceeds - commission
                        pnl = (exit_price - entry_price) * position - commission
                        logging.info("Exit LONG at %s price %.4f qty %d pnl %.2f (stop:%.4f trail:%.4f tp:%.4f)",
                                     date, exit_price, position, pnl, stop_price, trailing_stop, takeprofit_price)
                        trade_list.append({
                            "entry_date": last_trade_idx,
                            "exit_date": date,
                            "side": "LONG",
                            "entry_price": entry_price,
                            "exit_price": exit_price,
                            "qty": position,
                            "pnl": pnl
                        })
                        position = 0
                        entry_price = 0.0
                        entry_atr = 0.0
                        trailing_stop = None
                        last_trade_idx = None
                else:
                    # Shorts
                    stop_price = entry_price + stop_distance
                    new_trailing = price + trailing_dist
                    if trailing_stop is None or new_trailing < trailing_stop:
                        trailing_stop = new_trailing
                    exit_by_stop = price >= stop_price
                    exit_by_trail = trailing_stop is not None and price >= trailing_stop
                    takeprofit_price = entry_price - self.strat_cfg["takeprofit_atr_multiplier"] * entry_atr
                    exit_by_tp = price <= takeprofit_price
                    if exit_by_stop or exit_by_trail or exit_by_tp:
                        exit_price = price + self.exec_cfg.get("slippage", 0.0)
                        proceeds = position * exit_price
                        commission = abs(position) * self.exec_cfg.get("commission_per_trade", 0.0)
                        cash += proceeds - commission
                        pnl = (entry_price - exit_price) * (-position) - commission
                        logging.info("Exit SHORT at %s price %.4f qty %d pnl %.2f (stop:%.4f trail:%.4f tp:%.4f)",
                                     date, exit_price, position, pnl, stop_price, trailing_stop, takeprofit_price)
                        trade_list.append({
                            "entry_date": last_trade_idx,
                            "exit_date": date,
                            "side": "SHORT",
                            "entry_price": entry_price,
                            "exit_price": exit_price,
                            "qty": -position,
                            "pnl": pnl
                        })
                        position = 0
                        entry_price = 0.0
                        entry_atr = 0.0
                        trailing_stop = None
                        last_trade_idx = None

            # Evaluate new signals only if allowed and not holding a position (simple strategy: 1 position at a time)
            if allowed_to_trade and position == 0 and signal != 0:
                qty = self.position_size(price=price, atr=atr_val, equity=equity)
                if qty <= 0:
                    logging.debug("Computed qty 0 at %s price %.4f atr %.4f equity %.2f", date, price, atr_val, equity)
                else:
                    # ensure not exceeding total allocation
                    if (qty * price) > equity * self.risk_cfg["max_total_alloc"]:
                        qty = int(np.floor((equity * self.risk_cfg["max_total_alloc"]) / price))
                    if qty <= 0:
                        logging.debug("Qty after max_total_alloc constraint <=0, skipping trade")
                    else:
                        # Execute entry
                        if signal == 1:
                            entry_price = price + self.exec_cfg.get("slippage", 0.0)
                            position = qty
                            cash -= position * entry_price
                            entry_atr = atr_val
                            trailing_stop = entry_price - self.strat_cfg["trailing_atr_multiplier"] * entry_atr
                            last_trade_idx = date
                            logging.info("Enter LONG at %s price %.4f qty %d equity %.2f", date, entry_price, position, equity)
                        elif signal == -1:
                            entry_price = price - self.exec_cfg.get("slippage", 0.0)
                            position = -qty
                            cash -= position * entry_price  # subtracting a negative gives + proceeds in some models; here we keep consistent accounting (note this yields correct mtm via position*price)
                            entry_atr = atr_val
                            trailing_stop = entry_price + self.strat_cfg["trailing_atr_multiplier"] * entry_atr
                            last_trade_idx = date
                            logging.info("Enter SHORT at %s price %.4f qty %d equity %.2f", date, entry_price, -position, equity)

        # At the end of backtest, close any open position at final price
        final_price = data.iloc[-1]["Close"]
        final_date = data.iloc[-1].get("Date", data.index[-1])
        if position != 0:
            exit_price = final_price - self.exec_cfg.get("slippage", 0.0) if position > 0 else final_price + self.exec_cfg.get("slippage", 0.0)
            proceeds = position * exit_price
            commission = abs(position) * self.exec_cfg.get("commission_per_trade", 0.0)
            cash += proceeds - commission
            pnl = (exit_price - entry_price) * position - commission if position > 0 else (entry_price - exit_price) * (-position) - commission
            side = "LONG" if position > 0 else "SHORT"
            logging.info("Final exit %s at %s price %.4f qty %d pnl %.2f", side, final_date, exit_price, abs(position), pnl)
            trade_list.append({
                "entry_date": last_trade_idx,
                "exit_date": final_date,
                "side": side,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "qty": abs(position),
                "pnl": pnl
            })
            position = 0

        final_equity = cash
        returns = final_equity / initial_cash - 1.0

        # Performance metrics
        trade_df = pd.DataFrame(trade_list)
        total_trades = len(trade_df)
        wins = trade_df[trade_df["pnl"] > 0] if not trade_df.empty else pd.DataFrame()
        losses = trade_df[trade_df["pnl"] <= 0] if not trade_df.empty else pd.DataFrame()
        win_rate = len(wins) / total_trades if total_trades > 0 else np.nan
        avg_win = wins["pnl"].mean() if not wins.empty else 0.0
        avg_loss = losses["pnl"].mean() if not losses.empty else 0.0
        gross_profit = wins["pnl"].sum() if not wins.empty else 0.0
        gross_loss = losses["pnl"].sum() if not losses.empty else 0.0

        # Approximate Sharpe: use daily returns series from equity curve if we produce it; otherwise approximate from trade p/l
        # We build simple equity curve by applying trade pnl at exit dates; for simplicity create daily returns based on equity at each bar
        equity_curve = []
        cash_temp = initial_cash
        pos_temp = 0
        entry_temp = 0.0
        trade_iter = iter(trade_list)
        next_trade = next(trade_iter, None)
        # reconstruct equity curve using daily close price
        for i, row in data.iterrows():
            date = row.get("Date", None)
            price = row["Close"]
            # apply trade exits that match date
            applied = False
            while next_trade is not None and next_trade["exit_date"] == date:
                # apply pnl already included in cash in our backtest simulation; here we append equity as recorded
                applied = True
                next_trade = next(trade_iter, None)
            # derive mtm: if not holding a position then equity = cash (we don't know exact cash trace here)
            # approximate equity by assuming the final cash at the time, but to create a returns series we can use close-to-close returns
            equity_curve.append(cash + pos_temp * price)
        equity_series = pd.Series(equity_curve).ffill().fillna(initial_cash)
        daily_returns = equity_series.pct_change().fillna(0)
        if daily_returns.std() > 0:
            sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)
        else:
            sharpe = np.nan

        summary = {
            "initial_cash": initial_cash,
            "final_equity": final_equity,
            "total_return": returns,
            "max_drawdown": max_dd,
            "total_trades": total_trades,
            "win_rate": win_rate,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "gross_profit": gross_profit,
            "gross_loss": gross_loss,
            "sharpe": sharpe,
            "trades": trade_list
        }

        logging.info("Backtest complete. Final equity: %.2f Total return: %.2f%% Sharpe: %.2f MaxDD: %.2f%% Trades: %d",
                     final_equity, returns * 100, sharpe if not np.isnan(sharpe) else 0.0, max_dd * 100, total_trades)
        return summary


# ---------------------------
# Example usage function (keeps architecture consistent)
# ---------------------------
def run_trading_system(data: pd.DataFrame, config_path: Optional[str] = "config.json") -> Dict[str, Any]:
    """
    Main entry point to execute the strategy backtest.
    Expects data to be a pandas DataFrame with columns: Date (optional), Open, High, Low, Close, Volume
    """
    config = load_config(config_path)
    setup_logging(config)
    strat = Strategy(config)
    result = strat.run_backtest(data, initial_cash=config["backtest"].get("initial_cash", 100000.0))
    return result


# ---------------------------
# If run as a script, demonstrate example flow (without any real data)
# ---------------------------
if __name__ == "__main__":
    # For users: replace this block by loading real historical OHLCV data into `df`
    setup_logging(DEFAULT_CONFIG)
    logging.info("Running example with synthetic data (replace with real OHLCV data)")

    # Synthetic daily data generation (for demonstration only)
    np.random.seed(42)
    days = 500
    date_index = pd.date_range(end=datetime.today(), periods=days, freq="B")
    price = np.cumprod(1 + np.random.normal(0, 0.0015, size=days)) * 100.0
    high = price * (1 + np.abs(np.random.normal(0, 0.002, size=days)))
    low = price * (1 - np.abs(np.random.normal(0, 0.002, size=days)))
    open_p = price * (1 + np.random.normal(0, 0.001, size=days))
    volume = np.random.randint(100, 1000, size=days)

    df = pd.DataFrame({
        "Date": date_index,
        "Open": open_p,
        "High": high,
        "Low": low,
        "Close": price,
        "Volume": volume
    })

    result = run_trading_system(df, config_path=None)
    logging.info("Result summary: %s", json.dumps({
        "initial_cash": result["initial_cash"],
        "final_equity": result["final_equity"],
        "total_return": result["total_return"],
        "max_drawdown": result["max_drawdown"],
        "total_trades": result["total_trades"],
        "win_rate": result["win_rate"],
        "sharpe": result["sharpe"]
    }, indent=2))