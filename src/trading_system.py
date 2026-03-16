import logging
import math
import os
from dataclasses import dataclass, asdict
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd

# ----------------------------
# Configuration (preserved & extendable)
# ----------------------------
@dataclass
class Config:
    data_path: Optional[str] = None  # path to CSV with Open, High, Low, Close, Volume
    initial_capital: float = 100000.0
    risk_per_trade: float = 0.005  # fraction of capital risked per trade (0.5%)
    max_position_pct: float = 0.2  # max percent of capital in one position
    max_leverage: float = 2.0
    stop_atr_multiplier: float = 3.0
    trailing_atr_multiplier: float = 2.0
    fast_ma: int = 20
    slow_ma: int = 50
    adx_period: int = 14
    adx_threshold: float = 20.0
    rsi_period: int = 14
    rsi_oversold: float = 30.0
    rsi_overbought: float = 70.0
    atr_period: int = 14
    vol_target_annual: float = 0.12  # target annualized volatility for vol-targeting (12%)
    trading_cooldown_days: int = 5  # cooldown after an excessive drawdown
    max_drawdown_limit: float = 0.25  # stop trading if equity drawdown exceeds 25%
    max_consecutive_losses: int = 5  # reduce risk if exceeded
    reduce_risk_factor_after_losses: float = 0.5  # reduce risk by this factor after streak
    slippage_pct: float = 0.0005  # slippage per trade
    commission_per_trade: float = 1.0  # fixed commission per trade
    verbose: bool = True


# ----------------------------
# Logging (preserved)
# ----------------------------
logger = logging.getLogger("TradingSystem")
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)


# ----------------------------
# Utility indicator functions
# ----------------------------
def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    high_low = high - low
    high_close = (high - close.shift()).abs()
    low_close = (low - close.shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(window=period, min_periods=1).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.ewm(alpha=1 / period, adjust=False).mean()
    ma_down = down.ewm(alpha=1 / period, adjust=False).mean()
    rs = ma_up / ma_down
    rsi_series = 100 - (100 / (1 + rs))
    return rsi_series


def adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    # Implementation of ADX
    # Based on Wilder's smoothing
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = ((up_move > down_move) & (up_move > 0)) * up_move
    minus_dm = ((down_move > up_move) & (down_move > 0)) * down_move
    tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    atr_ = tr.rolling(window=period, min_periods=1).mean()
    plus_di = 100 * (plus_dm.rolling(window=period, min_periods=1).sum() / atr_)
    minus_di = 100 * (minus_dm.rolling(window=period, min_periods=1).sum() / atr_)
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)).replace([np.inf, -np.inf], 0).fillna(0) * 100
    adx_series = dx.rolling(window=period, min_periods=1).mean()
    return adx_series


def annualized_volatility(returns: pd.Series, periods_per_year: int = 252) -> float:
    return returns.std() * math.sqrt(periods_per_year)


# ----------------------------
# Trading system core (architecture kept)
# ----------------------------
class TradingSystem:
    def __init__(self, config: Config):
        self.config = config
        self.logger = logger
        if self.config.verbose:
            self.logger.setLevel(logging.INFO)
        else:
            self.logger.setLevel(logging.WARNING)
        self.reset()

    def reset(self):
        self.positions = []  # list of dicts with trade details
        self.equity_curve = None
        self.trades = []
        self.last_entry_index = -999
        self.loss_streak = 0
        self.cooldown_until = None

    def load_data(self, df: pd.DataFrame) -> pd.DataFrame:
        # Expect df with columns: Open, High, Low, Close, Volume (Volume optional)
        df = df.copy()
        required_cols = {"Open", "High", "Low", "Close"}
        if not required_cols.issubset(df.columns):
            raise ValueError(f"DataFrame must contain columns {required_cols}")
        df.sort_index(inplace=True)
        df["Close"] = df["Close"].astype(float)
        # Calculate indicators
        df["ema_fast"] = ema(df["Close"], self.config.fast_ma)
        df["ema_slow"] = ema(df["Close"], self.config.slow_ma)
        df["atr"] = atr(df["High"], df["Low"], df["Close"], self.config.atr_period)
        df["rsi"] = rsi(df["Close"], self.config.rsi_period)
        df["adx"] = adx(df["High"], df["Low"], df["Close"], self.config.adx_period)
        df["returns"] = df["Close"].pct_change().fillna(0)
        # rolling vol for vol-targeting (daily returns)
        df["rolling_vol"] = df["returns"].rolling(window=20, min_periods=5).std() * math.sqrt(252)
        return df

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        # MA crossover with ADX and RSI filter and a simple cooling mechanism
        signal = pd.Series(0, index=df.index)
        fast = df["ema_fast"]
        slow = df["ema_slow"]
        crossover_up = (fast > slow) & (fast.shift() <= slow.shift())
        crossover_down = (fast < slow) & (fast.shift() >= slow.shift())

        for idx in df.index:
            if self.cooldown_until is not None and idx <= self.cooldown_until:
                # in cooldown, no trading
                signal.at[idx] = 0
                continue

            if crossover_up.at[idx]:
                # apply ADX and RSI filters
                if df.at[idx, "adx"] >= self.config.adx_threshold and df.at[idx, "rsi"] < self.config.rsi_overbought:
                    signal.at[idx] = 1
                else:
                    signal.at[idx] = 0
            elif crossover_down.at[idx]:
                if df.at[idx, "adx"] >= self.config.adx_threshold and df.at[idx, "rsi"] > self.config.rsi_oversold:
                    signal.at[idx] = -1
                else:
                    signal.at[idx] = 0
            else:
                signal.at[idx] = 0
        return signal

    def position_sizing(self, price: float, atr: float, equity: float, signal: int, recent_vol: float) -> int:
        """
        Calculate number of shares/contracts to trade based on ATR volatility and configurable risk per trade.
        Returns integer number of units (positive for long, negative for short).
        """
        cfg = self.config
        if signal == 0:
            return 0

        # adaptive risk scaling after loss streak
        risk_per_trade = cfg.risk_per_trade
        if self.loss_streak >= cfg.max_consecutive_losses:
            risk_per_trade *= cfg.reduce_risk_factor_after_losses

        # account-level risk dollar amount
        risk_dollars = risk_per_trade * equity

        # use ATR-based stop distance
        stop_distance = cfg.stop_atr_multiplier * max(atr, 1e-8)  # price units
        if stop_distance <= 0 or np.isnan(stop_distance):
            return 0

        # position units before leverage limit
        units = risk_dollars / stop_distance / price

        # vol-targeting: scale by target vol / realized vol (protect when vol is low/high)
        if recent_vol and recent_vol > 0:
            vol_scale = cfg.vol_target_annual / recent_vol
            units *= vol_scale

        # cap by max_position_pct of equity and by max_leverage
        max_position_value = cfg.max_position_pct * equity
        max_units_by_pct = max_position_value / price
        units = min(units, max_units_by_pct)

        # apply leverage cap: units * price / equity <= max_leverage
        max_units_by_lev = (cfg.max_leverage * equity) / price
        units = min(units, max_units_by_lev)

        # ensure integer units (for stocks) and respect direction
        units = math.floor(abs(units))
        if units <= 0:
            return 0
        return units if signal > 0 else -units

    def run_backtest(self, raw_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        cfg = self.config
        df = self.load_data(raw_df)
        signals = self.generate_signals(df)

        # initialize equity series
        equity = cfg.initial_capital
        equity_curve = pd.Series(index=df.index, dtype=float)
        position = 0
        entry_price = 0.0
        entry_atr = 0.0
        stop_price = None
        trailing_stop = None
        entry_index = None

        trades = []
        peak_equity = equity
        drawdown = 0.0

        for i, idx in enumerate(df.index):
            row = df.loc[idx]
            price = row["Close"]
            signal = signals.at[idx]
            current_atr = row["atr"] if not np.isnan(row["atr"]) else 0.0
            recent_vol = row["rolling_vol"] if not np.isnan(row["rolling_vol"]) else None

            # if drawdown limit hit, enter cooldown
            if (peak_equity - equity) / peak_equity >= cfg.max_drawdown_limit and self.cooldown_until is None:
                # set cooldown
                self.cooldown_until = idx + pd.Timedelta(days=cfg.trading_cooldown_days)
                self.logger.warning(f"Max drawdown limit hit. Entering cooldown until {self.cooldown_until}")

            # manage existing position: update trailing stop using ATR
            if position != 0:
                # adjust trailing stop only in favorable direction
                if position > 0:
                    new_trailing = price - cfg.trailing_atr_multiplier * current_atr
                    if new_trailing > trailing_stop:
                        trailing_stop = new_trailing
                    # stop or exit on close below trailing stop or below initial stop
                    if price <= trailing_stop or price <= stop_price:
                        # exit
                        exit_price = price * (1 - cfg.slippage_pct)
                        pnl = (exit_price - entry_price) * position - cfg.commission_per_trade
                        equity += pnl
                        trades.append({
                            "entry_index": entry_index, "exit_index": idx,
                            "entry_price": entry_price, "exit_price": exit_price,
                            "units": position, "pnl": pnl
                        })
                        self.logger.info(f"Exit LONG at {idx} price {exit_price:.2f} pnl {pnl:.2f}")
                        # update loss streak
                        if pnl < 0:
                            self.loss_streak += 1
                        else:
                            self.loss_streak = 0
                        position = 0
                        stop_price = None
                        trailing_stop = None
                        entry_price = 0.0
                        entry_atr = 0.0
                        entry_index = None
                else:
                    # short
                    new_trailing = price + cfg.trailing_atr_multiplier * current_atr
                    if trailing_stop is None or new_trailing < trailing_stop:
                        trailing_stop = new_trailing
                    if price >= trailing_stop or price >= stop_price:
                        exit_price = price * (1 + cfg.slippage_pct)
                        pnl = (entry_price - exit_price) * (-position) - cfg.commission_per_trade
                        equity += pnl
                        trades.append({
                            "entry_index": entry_index, "exit_index": idx,
                            "entry_price": entry_price, "exit_price": exit_price,
                            "units": position, "pnl": pnl
                        })
                        self.logger.info(f"Exit SHORT at {idx} price {exit_price:.2f} pnl {pnl:.2f}")
                        if pnl < 0:
                            self.loss_streak += 1
                        else:
                            self.loss_streak = 0
                        position = 0
                        stop_price = None
                        trailing_stop = None
                        entry_price = 0.0
                        entry_atr = 0.0
                        entry_index = None

            # ENTRY logic: only enter if no position and signal present and not in cooldown
            if position == 0 and signal != 0 and (self.cooldown_until is None or idx > self.cooldown_until):
                # compute units
                units = self.position_sizing(price, current_atr, equity, signal, recent_vol)
                if units != 0:
                    # enter at next bar close price adjusted for slippage
                    if units > 0:
                        entry_price = price * (1 + cfg.slippage_pct)
                        stop_price = entry_price - cfg.stop_atr_multiplier * current_atr
                    else:
                        entry_price = price * (1 - cfg.slippage_pct)
                        stop_price = entry_price + cfg.stop_atr_multiplier * current_atr
                    position = units
                    entry_atr = current_atr
                    trailing_stop = entry_price - cfg.trailing_atr_multiplier * entry_atr if units > 0 else entry_price + cfg.trailing_atr_multiplier * entry_atr
                    entry_index = idx
                    # reduce equity by required margin if leveraged (simple approximation)
                    used_cap = min(abs(units) * price, cfg.max_position_pct * equity)
                    # no immediate equity reduction for fully cash-backed system, but commissions are applied at exit
                    self.logger.info(f"Enter {'LONG' if units>0 else 'SHORT'} at {idx} price {entry_price:.2f} units {units}")
                    # Note: do not adjust equity on entry to keep P&L realized-based

            # record equity (mark-to-market)
            mtm = 0.0
            if position != 0:
                mtm = (price - entry_price) * position if position > 0 else (entry_price - price) * (-position)
            equity_curve.at[idx] = equity + mtm

            # update peak equity and drawdown
            if equity_curve.at[idx] > peak_equity:
                peak_equity = equity_curve.at[idx]
            drawdown = min(drawdown, (equity_curve.at[idx] - peak_equity) / peak_equity)

        # at the end, force close any open position at final price
        if position != 0:
            final_price = df["Close"].iloc[-1]
            if position > 0:
                exit_price = final_price * (1 - cfg.slippage_pct)
                pnl = (exit_price - entry_price) * position - cfg.commission_per_trade
                self.logger.info(f"Forced exit LONG at end price {exit_price:.2f} pnl {pnl:.2f}")
            else:
                exit_price = final_price * (1 + cfg.slippage_pct)
                pnl = (entry_price - exit_price) * (-position) - cfg.commission_per_trade
                self.logger.info(f"Forced exit SHORT at end price {exit_price:.2f} pnl {pnl:.2f}")
            equity += pnl
            trades.append({
                "entry_index": entry_index, "exit_index": df.index[-1],
                "entry_price": entry_price, "exit_price": exit_price,
                "units": position, "pnl": pnl
            })
            position = 0
            equity_curve.iloc[-1] = equity

        # finalize equity curve forward-filling any NaN
        equity_series = equity_curve.fillna(method="ffill").fillna(cfg.initial_capital)
        trades_df = pd.DataFrame(trades)

        # compute metrics
        perf = self.analyze_performance(equity_series)
        self.logger.info(f"Backtest completed. Final equity: {equity_series.iloc[-1]:.2f}")
        for k, v in perf.items():
            self.logger.info(f"{k}: {v}")

        return equity_series.to_frame(name="equity"), trades_df

    def analyze_performance(self, equity_series: pd.Series) -> dict:
        returns = equity_series["equity"].pct_change().fillna(0)
        cum_return = equity_series["equity"].iloc[-1] / equity_series["equity"].iloc[0] - 1
        ann_return = (1 + cum_return) ** (252 / len(equity_series)) - 1 if len(equity_series) > 0 else 0.0
        ann_vol = annualized_volatility(returns)
        sharpe = ann_return / ann_vol if ann_vol > 0 else np.nan
        downside_returns = returns.copy()
        downside_returns[downside_returns > 0] = 0
        downside_vol = downside_returns.std() * math.sqrt(252)
        sortino = ann_return / downside_vol if downside_vol > 0 else np.nan
        rolling_max = equity_series["equity"].cummax()
        drawdown = (equity_series["equity"] - rolling_max) / rolling_max
        max_dd = drawdown.min()
        dd_duration = (drawdown < 0).astype(int).groupby((drawdown >= 0).astype(int).cumsum()).sum().max() if len(drawdown) > 0 else 0
        return {
            "Cumulative Return": cum_return,
            "Annualized Return": ann_return,
            "Annualized Volatility": ann_vol,
            "Sharpe Ratio": sharpe,
            "Sortino Ratio": sortino,
            "Max Drawdown": max_dd,
            "Max Drawdown Duration (days)": dd_duration
        }


# ----------------------------
# Example usage / entrypoint (keeps configuration and logging)
# ----------------------------
def load_csv_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=True, index_col=0)
    # Ensure required columns exist
    if not {"Open", "High", "Low", "Close"}.issubset(df.columns):
        raise ValueError("CSV must contain Open,High,Low,Close columns")
    return df


def generate_sample_data(n_days: int = 500, start_price: float = 100.0, seed: Optional[int] = 42) -> pd.DataFrame:
    # Generate a simple synthetic price series (GBM-like) for testing/backtest
    np.random.seed(seed)
    dt = 1 / 252
    mu = 0.05
    sigma = 0.2
    returns = np.random.normal(loc=(mu - 0.5 * sigma ** 2) * dt, scale=sigma * math.sqrt(dt), size=n_days)
    price = start_price * np.exp(np.cumsum(returns))
    dates = pd.bdate_range(end=pd.Timestamp.today(), periods=n_days)
    df = pd.DataFrame(index=dates)
    df["Close"] = price
    df["Open"] = df["Close"].shift(1).fillna(df["Close"])
    df["High"] = np.maximum(df["Open"], df["Close"]) * (1 + np.random.rand(n_days) * 0.01)
    df["Low"] = np.minimum(df["Open"], df["Close"]) * (1 - np.random.rand(n_days) * 0.01)
    df["Volume"] = np.random.randint(100, 1000, size=n_days)
    return df


if __name__ == "__main__":
    cfg = Config()
    # allow overriding the data path via environment variable for convenience
    if os.getenv("DATA_PATH"):
        cfg.data_path = os.getenv("DATA_PATH")

    ts = TradingSystem(cfg)

    if cfg.data_path and os.path.exists(cfg.data_path):
        data = load_csv_data(cfg.data_path)
        logger.info(f"Loaded data from {cfg.data_path}")
    else:
        logger.info("No data path provided or file not found. Generating sample data for backtest.")
        data = generate_sample_data(n_days=630)

    equity_curve, trades_df = ts.run_backtest(data)

    logger.info(f"Number of trades: {len(trades_df)}")
    if not trades_df.empty:
        logger.info(f"Sample trades:\n{trades_df.head().to_string(index=False)}")

    # Save outputs if needed (preser ving architecture: leave hooks for persistence)
    # For safety, do not write files automatically unless explicitly desired by user.
    # End of script.