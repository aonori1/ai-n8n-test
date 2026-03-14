import logging
import os
import copy
from dataclasses import dataclass
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd

# -------------------------
# Configuration (preserved & extensible)
# -------------------------
CONFIG = {
    "symbol": "TEST",
    "initial_capital": 100000.0,
    "commission_per_trade": 1.0,         # flat commission per trade
    "slippage_pct": 0.0005,              # slippage as fraction of price
    "target_volatility_annual": 0.08,    # portfolio target volatility (annualized)
    "volatility_lookback": 21,           # days to estimate realized volatility
    "atr_lookback": 14,                  # ATR lookback for stop placement
    "ema_fast": 20,
    "ema_slow": 50,
    "adx_lookback": 14,
    "adx_threshold": 20,                 # only trade when ADX > threshold (trend present)
    "rsi_lookback": 14,
    "rsi_oversold": 30,
    "rsi_overbought": 70,
    "risk_per_trade": 0.01,              # fraction of capital to risk per trade (ATR-based)
    "max_position_size": 0.25,           # max fraction of capital in single position
    "max_portfolio_drawdown": 0.25,      # temporary stop trading if drawdown exceeds this
    "drawdown_cooloff_days": 21,         # cooldown after hitting max drawdown
    "trailing_atr_multiplier": 2.0,      # trailing stop in ATRs
    "stop_atr_multiplier": 3.0,          # initial stop in ATRs
    "min_trade_interval_days": 1,        # minimal holding/cooldown between trades on same symbol
    "verbose_logging": True,
    "seed": 42,
}

# -------------------------
# Logging (preserved)
# -------------------------
logger = logging.getLogger("trading_system")
logger.setLevel(logging.DEBUG if CONFIG["verbose_logging"] else logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG if CONFIG["verbose_logging"] else logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s: %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

# -------------------------
# Utility indicators and helpers
# -------------------------
def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def rsi(series: pd.Series, period:int=14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.rolling(period, min_periods=1).mean()
    ma_down = down.rolling(period, min_periods=1).mean()
    rs = ma_up / (ma_down + 1e-12)
    return 100 - (100 / (1 + rs))

def atr(df: pd.DataFrame, period:int=14) -> pd.Series:
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(period, min_periods=1).mean()

def adx(df: pd.DataFrame, period:int=14) -> pd.Series:
    # Wilder's ADX implementation
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    upmove = high.diff()
    downmove = -low.diff()
    plus_dm = np.where((upmove > downmove) & (upmove > 0), upmove, 0.0)
    minus_dm = np.where((downmove > upmove) & (downmove > 0), downmove, 0.0)
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_val = tr.rolling(period, min_periods=1).mean()
    plus_di = 100 * pd.Series(plus_dm).rolling(period, min_periods=1).sum() / (atr_val + 1e-12)
    minus_di = 100 * pd.Series(minus_dm).rolling(period, min_periods=1).sum() / (atr_val + 1e-12)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-12)
    adx_series = dx.rolling(period, min_periods=1).mean()
    adx_series.index = df.index
    return adx_series

def annualize_vol(daily_returns: pd.Series, trading_days: int = 252) -> float:
    return np.sqrt(trading_days) * daily_returns.std()

# -------------------------
# Trading System Class (architecture preserved)
# -------------------------
@dataclass
class TradingSystem:
    config: Dict[str, Any]

    def __post_init__(self):
        self.strategy_config = copy.deepcopy(self.config)
        self.capital = self.strategy_config["initial_capital"]
        self.initial_capital = self.capital
        self.position = 0.0  # number of shares (positive long, negative short - we'll use only long for simplicity)
        self.position_value = 0.0
        self.last_trade_index = None
        self.portfolio_values = []
        self.equity_curve = None
        self.trades = []
        self.drawdown_flag = False
        self.drawdown_start_idx = None
        self.cooldown_counter = 0

    def compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["EMA_fast"] = ema(df["Close"], self.strategy_config["ema_fast"])
        df["EMA_slow"] = ema(df["Close"], self.strategy_config["ema_slow"])
        df["RSI"] = rsi(df["Close"], self.strategy_config["rsi_lookback"])
        df["ATR"] = atr(df, self.strategy_config["atr_lookback"])
        df["ADX"] = adx(df, self.strategy_config["adx_lookback"])
        # daily returns for realized volatility
        df["Daily_Return"] = df["Close"].pct_change().fillna(0.0)
        df["RealizedVol"] = df["Daily_Return"].rolling(self.strategy_config["volatility_lookback"], min_periods=1).std() * np.sqrt(252)
        # signals
        # Trend-following: EMA cross
        df["Trend_Signal"] = 0
        df.loc[df["EMA_fast"] > df["EMA_slow"], "Trend_Signal"] = 1
        df.loc[df["EMA_fast"] < df["EMA_slow"], "Trend_Signal"] = -1
        # We only take long trades for simplicity and to reduce tail risk; can extend to shorts carefully
        df["Signal"] = 0
        df.loc[(df["Trend_Signal"] == 1) &
               (df["ADX"] >= self.strategy_config["adx_threshold"]) &
               (df["RSI"] < self.strategy_config["rsi_overbought"]), "Signal"] = 1
        # Filter: do not open if too volatile relative to target or if in drawdown cooldown
        df["AllowedByVol"] = df["RealizedVol"] <= (self.strategy_config["target_volatility_annual"] * 1.5)
        return df

    def position_size(self, price: float, atr_value: float, idx: pd.Timestamp) -> Tuple[float, float]:
        """
        Determines number of shares to buy given ATR-based stop risk and volatility targeting.
        Returns: (shares, position_notional)
        """
        # Basic ATR-based stop
        stop_distance = max(1e-6, atr_value * self.strategy_config["stop_atr_multiplier"])
        capital = self.capital
        # Risk amount per trade limited to risk_per_trade * capital
        risk_amount = capital * self.strategy_config["risk_per_trade"]
        # position size such that stop_distance * shares = risk_amount
        shares = int(risk_amount / stop_distance) if stop_distance > 0 else 0
        # enforce max position size
        max_notional = capital * self.strategy_config["max_position_size"]
        if shares * price > max_notional:
            shares = int(max_notional / price)
        return shares, shares * price

    def apply_slippage_and_commission(self, price: float, shares: int, is_entry: bool) -> float:
        # simple slippage model: percent of price
        direction = 1 if shares > 0 else -1
        slippage = price * self.strategy_config["slippage_pct"]
        price_with_slippage = price + direction * slippage
        commission = self.strategy_config["commission_per_trade"]
        total_cost = price_with_slippage * shares + commission
        return total_cost

    def backtest(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df = self.compute_indicators(df)
        # Prepare bookkeeping columns
        df["Position"] = 0.0
        df["Cash"] = 0.0
        df["Holdings"] = 0.0
        df["Total"] = 0.0
        df["Trade"] = ""
        cash = self.capital
        holdings = 0.0
        position_shares = 0
        entry_price = None
        trailing_stop = None
        max_portfolio_value = self.capital
        last_trade_day = None

        # Iterate rows - vectorization for order logic is complex; using loop keeps architecture clear
        for idx, row in df.iterrows():
            price_close = row["Close"]
            price_open = row.get("Open", price_close)  # fallback to close if Open not present
            signal = row["Signal"]
            atr_val = row["ATR"]
            allowed_by_vol = row["AllowedByVol"]
            adx_val = row["ADX"]

            # Update portfolio value mark-to-market
            holdings = position_shares * price_close
            total = cash + holdings
            df.at[idx, "Holdings"] = holdings
            df.at[idx, "Cash"] = cash
            df.at[idx, "Total"] = total
            df.at[idx, "Position"] = position_shares

            # Track equity curve and drawdown
            self.portfolio_values.append(total)
            if len(self.portfolio_values) > 0:
                max_portfolio_value = max(max_portfolio_value, total)
                drawdown = 1.0 - (total / max_portfolio_value if max_portfolio_value > 0 else 1.0)
                if drawdown >= self.strategy_config["max_portfolio_drawdown"]:
                    if not self.drawdown_flag:
                        logger.warning(f"Max drawdown reached at {idx.date()}: drawdown={drawdown:.2%}. Entering cooldown for {self.strategy_config['drawdown_cooloff_days']} days.")
                        self.drawdown_flag = True
                        self.drawdown_start_idx = idx
                        self.cooldown_counter = self.strategy_config["drawdown_cooloff_days"]
                if self.drawdown_flag:
                    # cooldown mechanism
                    if self.cooldown_counter > 0:
                        self.cooldown_counter -= 1
                        # During cooldown do not open new trades; optionally close existing positions to protect capital
                        if position_shares != 0:
                            # Close position at open (or close) price
                            cost = self.apply_slippage_and_commission(price_open, -position_shares, is_entry=False)
                            # Update cash after closing
                            cash += -cost
                            logger.info(f"Cooldown: closing position of {position_shares} shares at {price_open:.2f} on {idx.date()}. Cash now {cash:.2f}")
                            df.at[idx, "Trade"] = f"CloseCooldown {position_shares}"
                            position_shares = 0
                            entry_price = None
                            trailing_stop = None
                    else:
                        # end cooldown
                        logger.info(f"Cooldown ended at {idx.date()}. Resuming trading.")
                        self.drawdown_flag = False

            # If we're in cooldown due to drawdown, skip trade entries
            if self.drawdown_flag and self.cooldown_counter > 0:
                df.at[idx, "Position"] = position_shares
                df.at[idx, "Cash"] = cash
                df.at[idx, "Holdings"] = position_shares * price_close
                df.at[idx, "Total"] = cash + position_shares * price_close
                continue

            # Exit conditions: stop loss via ATR trailing stop or signal flip
            exit_trade = False
            if position_shares > 0:
                # Update trailing stop: move up only
                new_trailing = price_close - atr_val * self.strategy_config["trailing_atr_multiplier"]
                if trailing_stop is None or new_trailing > trailing_stop:
                    trailing_stop = new_trailing
                # If price breaches trailing stop close at open/close
                if price_close <= trailing_stop:
                    exit_trade = True
                    logger.info(f"Trailing stop hit at {idx.date()}: price={price_close:.2f}, trailing_stop={trailing_stop:.2f}")
                    df.at[idx, "Trade"] = f"Exit_Trail {position_shares}"
                # If signal flips (EMA slow > EMA fast) exit
                if row["Trend_Signal"] == -1:
                    exit_trade = True
                    df.at[idx, "Trade"] = f"Exit_Flip {position_shares}"
            # Execute exit
            if exit_trade and position_shares != 0:
                # Close at open to simulate realistic immediate exit
                exit_price = price_open
                cost = self.apply_slippage_and_commission(exit_price, -position_shares, is_entry=False)
                cash += -cost
                logger.debug(f"Exit trade on {idx.date()}: shares {position_shares} at {exit_price:.2f}, cash {cash:.2f}")
                # Record trade
                self.trades.append({
                    "date": idx,
                    "type": "exit",
                    "shares": position_shares,
                    "price": exit_price,
                    "cash_after": cash
                })
                position_shares = 0
                entry_price = None
                trailing_stop = None
                last_trade_day = idx

            # Entry conditions: only open if not currently in a position, signal==1, allowed_by_vol True, ADX > threshold
            can_enter_signal = (signal == 1) and allowed_by_vol and (adx_val >= self.strategy_config["adx_threshold"])
            if position_shares == 0 and can_enter_signal:
                # Ensure we obey min trade interval
                if last_trade_day is not None and (idx - last_trade_day).days < self.strategy_config["min_trade_interval_days"]:
                    logger.debug(f"Skipping entry at {idx.date()} due to min trade interval.")
                else:
                    shares, notional = self.position_size(price_open, atr_val, idx)
                    if shares > 0 and notional > 0 and cash >= notional * (1 + self.strategy_config["slippage_pct"]) + self.strategy_config["commission_per_trade"]:
                        # Execute at open price
                        cost = self.apply_slippage_and_commission(price_open, shares, is_entry=True)
                        cash -= cost
                        position_shares = shares
                        entry_price = price_open
                        trailing_stop = price_open - atr_val * self.strategy_config["trailing_atr_multiplier"]
                        logger.info(f"Entry trade on {idx.date()}: shares {shares} at {price_open:.2f}, cash {cash:.2f}")
                        self.trades.append({
                            "date": idx,
                            "type": "entry",
                            "shares": shares,
                            "price": price_open,
                            "cash_after": cash
                        })
                        last_trade_day = idx
                        df.at[idx, "Trade"] = f"Enter {shares}"
                    else:
                        logger.debug(f"Not enough cash to enter or zero position size at {idx.date()}. shares={shares}, cost_est={notional:.2f}")

            # Update bookkeeping after potential trade
            holdings = position_shares * price_close
            total = cash + holdings
            df.at[idx, "Holdings"] = holdings
            df.at[idx, "Cash"] = cash
            df.at[idx, "Total"] = total
            df.at[idx, "Position"] = position_shares

        # Final equity curve
        df["Equity"] = df["Total"]
        self.equity_curve = df["Equity"]
        logger.info(f"Backtest complete. Final capital: {df['Equity'].iloc[-1]:.2f}")
        return df

    def performance_summary(self, equity: pd.Series) -> Dict[str, Any]:
        returns = equity.pct_change().fillna(0.0)
        cumulative_return = equity.iloc[-1] / equity.iloc[0] - 1.0
        annualized_return = (1 + cumulative_return) ** (252 / len(equity)) - 1 if len(equity) > 0 else 0.0
        annualized_volatility = annualize_vol(returns)
        sharpe = (annualized_return / (annualized_volatility + 1e-12)) if annualized_volatility > 0 else np.nan
        peak = equity.cummax()
        drawdown = (equity - peak) / peak
        max_drawdown = drawdown.min()
        summary = {
            "cumulative_return": cumulative_return,
            "annualized_return": annualized_return,
            "annualized_volatility": annualized_volatility,
            "sharpe": sharpe,
            "max_drawdown": max_drawdown
        }
        return summary

# -------------------------
# Example usage function: kept simple to match architecture
# -------------------------
def run_backtest(prices: pd.DataFrame, config: Dict[str, Any] = CONFIG) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    prices: DataFrame with index of timestamps and columns: Open, High, Low, Close, Volume (Volume optional)
    Returns: (results_df, performance_summary)
    """
    ts = TradingSystem(config)
    results = ts.backtest(prices)
    perf = ts.performance_summary(results["Equity"])
    logger.info("Performance Summary:")
    logger.info(f"Cumulative Return: {perf['cumulative_return']:.2%}")
    logger.info(f"Annualized Return: {perf['annualized_return']:.2%}")
    logger.info(f"Annualized Volatility: {perf['annualized_volatility']:.2%}")
    logger.info(f"Sharpe Ratio: {perf['sharpe']:.2f}")
    logger.info(f"Max Drawdown: {perf['max_drawdown']:.2%}")
    return results, perf

# -------------------------
# If module run directly: minimal demonstration runner (preserves architecture, but does not execute I/O)
# -------------------------
if __name__ == "__main__":
    # Minimal demo: synthetic price series if no external feed provided
    dates = pd.bdate_range(start="2020-01-01", end="2021-12-31")
    np.random.seed(CONFIG["seed"])
    # Generate synthetic walk with drift
    price = 100 + np.cumsum(np.random.normal(loc=0.0005, scale=0.02, size=len(dates)))
    df_demo = pd.DataFrame(index=dates)
    df_demo["Close"] = price
    df_demo["Open"] = df_demo["Close"].shift(1).fillna(df_demo["Close"])
    df_demo["High"] = np.maximum(df_demo["Open"], df_demo["Close"]) + np.abs(np.random.normal(0, 0.2, size=len(dates)))
    df_demo["Low"] = np.minimum(df_demo["Open"], df_demo["Close"]) - np.abs(np.random.normal(0, 0.2, size=len(dates)))
    df_demo["Volume"] = 1000

    results_df, perf = run_backtest(df_demo, CONFIG)
    # Note: in real usage, external code will handle results and plotting; logging has been preserved.