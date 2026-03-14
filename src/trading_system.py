import logging
import math
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List

# -------------------------
# Configuration (preserved / extended)
# -------------------------
CONFIG = {
    "initial_capital": 100000.0,
    "risk_per_trade": 0.01,            # fraction of equity risked per trade
    "max_position_size_pct": 0.2,      # maximum fraction of equity in a single position
    "atr_length": 14,
    "atr_multiplier_stop": 3.0,        # initial stop = entry - atr_multiplier_stop * ATR
    "atr_multiplier_trail": 2.0,       # trailing stop distance in ATRs
    "ema_short": 20,
    "ema_long": 50,
    "ema_trend": 200,                  # long-term trend filter
    "volatility_filter_max_atr_pct": 0.05,  # if ATR/price > this, avoid new trades
    "min_trade_days_between": 1,       # minimum days between consecutive entries
    "max_drawdown_stop": 0.25,         # if equity drawdown exceeds this, halt new entries
    "slippage_per_share": 0.0,
    "commission_per_trade": 0.0,
    "contract_size": 1,                # number of units per "share" (for futures)
    "use_trailing_stop": True,
    "partial_take_profit_pct": 0.5,    # take this fraction of position at first profit target
    "take_profit_atr_multiplier": 4.0, # take-profit level in ATRs for first partial
    "logging_level": logging.INFO,
}

# -------------------------
# Logging (preserved)
# -------------------------
logger = logging.getLogger("TradingSystem")
logger.setLevel(CONFIG["logging_level"])
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setLevel(CONFIG["logging_level"])
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)


# -------------------------
# Helper indicators & metrics
# -------------------------
def atr(df: pd.DataFrame, length: int) -> pd.Series:
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return tr.rolling(length, min_periods=1).mean()


def ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False).mean()


def sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    if returns.std() == 0:
        return 0.0
    ann_factor = 252 ** 0.5
    return ((returns.mean() - risk_free_rate / 252) / returns.std()) * ann_factor


def max_drawdown(equity_curve: pd.Series) -> float:
    peak = equity_curve.cummax()
    drawdown = (equity_curve - peak) / peak
    return -drawdown.min()


# -------------------------
# Strategy (architecture preserved, improved signal & risk)
# -------------------------
@dataclass
class StrategyConfig:
    initial_capital: float = CONFIG["initial_capital"]
    risk_per_trade: float = CONFIG["risk_per_trade"]
    max_position_size_pct: float = CONFIG["max_position_size_pct"]
    atr_length: int = CONFIG["atr_length"]
    atr_multiplier_stop: float = CONFIG["atr_multiplier_stop"]
    atr_multiplier_trail: float = CONFIG["atr_multiplier_trail"]
    ema_short: int = CONFIG["ema_short"]
    ema_long: int = CONFIG["ema_long"]
    ema_trend: int = CONFIG["ema_trend"]
    volatility_filter_max_atr_pct: float = CONFIG["volatility_filter_max_atr_pct"]
    min_trade_days_between: int = CONFIG["min_trade_days_between"]
    max_drawdown_stop: float = CONFIG["max_drawdown_stop"]
    slippage_per_share: float = CONFIG["slippage_per_share"]
    commission_per_trade: float = CONFIG["commission_per_trade"]
    contract_size: int = CONFIG["contract_size"]
    use_trailing_stop: bool = CONFIG["use_trailing_stop"]
    partial_take_profit_pct: float = CONFIG["partial_take_profit_pct"]
    take_profit_atr_multiplier: float = CONFIG["take_profit_atr_multiplier"]


@dataclass
class Position:
    entry_price: float
    size: int
    stop_price: float
    entry_index: int
    highest_price: float = field(default=0.0)
    partial_taken: bool = field(default=False)


class Strategy:
    def __init__(self, config: StrategyConfig):
        self.cfg = config

    def prepare_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["ATR"] = atr(df, self.cfg.atr_length)
        df["EMA_short"] = ema(df["Close"], self.cfg.ema_short)
        df["EMA_long"] = ema(df["Close"], self.cfg.ema_long)
        df["EMA_trend"] = ema(df["Close"], self.cfg.ema_trend)
        # Smoother signal: trend must be bullish and short EMA must cross long EMA
        df["EMA_diff"] = df["EMA_short"] - df["EMA_long"]
        df["EMA_diff_prev"] = df["EMA_diff"].shift(1)
        df["bullish_cross"] = (df["EMA_diff"] > 0) & (df["EMA_diff_prev"] <= 0)
        # Volatility as fraction of price
        df["ATR_pct"] = df["ATR"] / df["Close"]
        return df

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """
        Returns '1' for long entry signals, '0' for no entry. This strategy only goes long.
        """
        df = self.prepare_indicators(df)
        signals = pd.Series(0, index=df.index)
        # Entry when EMA_short crosses EMA_long, overall trend (price above EMA_trend), and volatility acceptable
        entry_mask = (
            df["bullish_cross"]
            & (df["Close"] > df["EMA_trend"])
            & (df["ATR_pct"] <= self.cfg.volatility_filter_max_atr_pct)
        )
        signals[entry_mask] = 1
        return signals, df


# -------------------------
# Trading engine / Backtester (preserved architecture, improved risk management)
# -------------------------
class Backtester:
    def __init__(self, df: pd.DataFrame, strategy: Strategy, config: StrategyConfig):
        self.df = df.copy().reset_index(drop=True)
        self.strategy = strategy
        self.cfg = config
        self.cash = config.initial_capital
        self.equity = config.initial_capital
        self.position: Optional[Position] = None
        self.equity_curve = []
        self.position_history: List[Dict[str, Any]] = []
        self.last_entry_idx: Optional[int] = None
        self.trade_count = 0
        self.halted_due_to_drawdown = False

    def _can_enter(self, idx: int) -> bool:
        if self.halted_due_to_drawdown:
            logger.debug("New entries halted due to drawdown stop.")
            return False
        if self.last_entry_idx is not None:
            if idx - self.last_entry_idx < self.cfg.min_trade_days_between:
                logger.debug("Skipping entry: minimum days between trades not met.")
                return False
        return True

    def _position_size_from_risk(self, equity: float, entry_price: float, stop_price: float) -> int:
        """
        Calculate position size based on the configured risk per trade and the distance to stop.
        Caps position by max_position_size_pct of equity.
        """
        risk_amount = equity * self.cfg.risk_per_trade
        per_unit_risk = abs(entry_price - stop_price) * self.cfg.contract_size
        if per_unit_risk <= 0:
            return 0
        raw_size = math.floor(risk_amount / per_unit_risk)
        # cap by maximum position size (in $)
        max_notional = equity * self.cfg.max_position_size_pct
        max_size_by_notional = math.floor(max_notional / (entry_price * self.cfg.contract_size))
        size = max(0, min(raw_size, max_size_by_notional))
        return size

    def _enter_long(self, idx: int, price: float, atr: float):
        # Determine stop and position sizing
        stop_price = price - self.cfg.atr_multiplier_stop * atr
        if stop_price <= 0:
            logger.debug("Computed stop <= 0, aborting entry")
            return False
        size = self._position_size_from_risk(self.equity, price + self.cfg.slippage_per_share, stop_price)
        if size <= 0:
            logger.debug("Position size computed as 0, skipping entry.")
            return False
        # check sufficient cash (allow margin-like behavior up to equity limits)
        cost = size * price * self.cfg.contract_size + self.cfg.commission_per_trade
        if cost > (self.cash + self.equity):  # conservative check; in simple backtest we allow using equity
            logger.debug("Insufficient funds for entry: cost=%s cash=%s equity=%s", cost, self.cash, self.equity)
            return False
        self.position = Position(entry_price=price, size=size, stop_price=stop_price, entry_index=idx, highest_price=price, partial_taken=False)
        self.last_entry_idx = idx
        self.trade_count += 1
        logger.info("Entered long at idx=%d price=%.2f size=%d stop=%.2f", idx, price, size, stop_price)
        # subtract commission
        self.cash -= self.cfg.commission_per_trade
        return True

    def _exit_position(self, idx: int, price: float, reason: str):
        pos = self.position
        if pos is None:
            return
        pnl = (price - pos.entry_price) * pos.size * self.cfg.contract_size
        # account for slippage and commission
        pnl -= (self.cfg.slippage_per_share * pos.size * self.cfg.contract_size)
        pnl -= self.cfg.commission_per_trade
        self.cash += pos.size * price * self.cfg.contract_size + pnl - pos.size * pos.entry_price * self.cfg.contract_size
        # For clarity: apply PnL directly to cash (we use equity tracking separately)
        logger.info("Exited position at idx=%d price=%.2f size=%d pnl=%.2f reason=%s", idx, price, pos.size, pnl, reason)
        self.position_history.append({
            "entry_index": pos.entry_index,
            "exit_index": idx,
            "entry_price": pos.entry_price,
            "exit_price": price,
            "size": pos.size,
            "pnl": pnl,
            "reason": reason,
        })
        self.position = None

    def _apply_trailing_stop(self, current_price: float, atr: float):
        pos = self.position
        if pos is None:
            return
        # update highest price
        if current_price > pos.highest_price:
            pos.highest_price = current_price
        # trailing stop based on highest price
        trail_stop = pos.highest_price - self.cfg.atr_multiplier_trail * atr
        if trail_stop > pos.stop_price:
            logger.debug("Trailing stop moved from %.2f to %.2f", pos.stop_price, trail_stop)
            pos.stop_price = trail_stop

    def run_backtest(self) -> Dict[str, Any]:
        signals, df_ind = self.strategy.generate_signals(self.df)
        df = df_ind  # includes indicators
        n = len(df)
        equity = self.equity
        self.equity_curve = []
        daily_returns = []

        for idx in range(n):
            row = df.iloc[idx]
            price = row["Close"]
            atr_val = row["ATR"]
            date_info = df.index[idx] if hasattr(df.index, "values") else idx

            # Check for new entry
            if signals.iloc[idx] == 1 and self.position is None and self._can_enter(idx):
                # safety: skip if recent big drawdown exceeded
                if max_drawdown(pd.Series(self.equity_curve)) if len(self.equity_curve) > 0 else 0.0 > self.cfg.max_drawdown_stop:
                    logger.warning("Max drawdown threshold exceeded; halting new entries.")
                    self.halted_due_to_drawdown = True
                if not self.halted_due_to_drawdown:
                    entered = self._enter_long(idx, price + self.cfg.slippage_per_share, atr_val)
                    if entered:
                        # upon entry we don't immediately debit cash by notional in this simple backtest;
                        # P&L will be settled upon exit. Commission already debited.
                        pass

            # Manage existing position
            if self.position is not None:
                pos = self.position
                # trailing stop adjustment
                if self.cfg.use_trailing_stop:
                    self._apply_trailing_stop(price, atr_val)

                # Check stop-loss hit
                if price <= pos.stop_price:
                    self._exit_position(idx, pos.stop_price, reason="stop_hit")
                else:
                    # Partial profit taking: if price reaches take-profit threshold
                    if (not pos.partial_taken) and (price >= pos.entry_price + self.cfg.take_profit_atr_multiplier * atr_val):
                        take_amount = math.floor(pos.size * self.cfg.partial_take_profit_pct)
                        if take_amount > 0:
                            # simulate taking profit on part of the position
                            pnl_partial = (price - pos.entry_price) * take_amount * self.cfg.contract_size
                            pnl_partial -= (self.cfg.slippage_per_share * take_amount * self.cfg.contract_size)
                            pnl_partial -= self.cfg.commission_per_trade
                            # reduce position size and credit cash
                            pos.size -= take_amount
                            pos.partial_taken = True
                            self.cash += pnl_partial + take_amount * price * self.cfg.contract_size - take_amount * pos.entry_price * self.cfg.contract_size
                            logger.info("Partial take-profit at idx=%d price=%.2f taken=%d pnl=%.2f", idx, price, take_amount, pnl_partial)
                            # if nothing left, close entirely
                            if pos.size <= 0:
                                self._exit_position(idx, price, reason="partial_exit_all")
                                continue

            # Update equity: mark-to-market current position
            mtm = 0.0
            if self.position is not None:
                pos = self.position
                mtm = (price - pos.entry_price) * pos.size * self.cfg.contract_size
                # do not double-count commissions; they were applied at entry/partial/exit
            equity = self.cash + (pos.size * price * self.cfg.contract_size if self.position else 0)
            self.equity_curve.append(equity)
            # compute daily return compared to previous equity if available
            if len(self.equity_curve) > 1:
                ret = (self.equity_curve[-1] - self.equity_curve[-2]) / max(self.equity_curve[-2], 1e-6)
                daily_returns.append(ret)

            # Enforce absolute capital-based stop: if drawdown beyond absolute threshold, halt entries
            if len(self.equity_curve) > 1:
                dd = max_drawdown(pd.Series(self.equity_curve))
                if dd >= self.cfg.max_drawdown_stop:
                    if not self.halted_due_to_drawdown:
                        logger.warning("Drawdown %.2f exceeded threshold %.2f: halting new entries.", dd, self.cfg.max_drawdown_stop)
                    self.halted_due_to_drawdown = True

        # End of loop: if position still open, close at last close price
        if self.position is not None:
            final_price = df.iloc[-1]["Close"]
            self._exit_position(n - 1, final_price, reason="end_of_data")
            equity = self.cash
            self.equity_curve[-1] = equity

        equity_series = pd.Series(self.equity_curve)
        stats = {
            "final_equity": equity_series.iloc[-1] if len(equity_series) > 0 else self.equity,
            "returns_series": pd.Series(daily_returns) if daily_returns else pd.Series(dtype=float),
            "sharpe": sharpe_ratio(pd.Series(daily_returns)) if daily_returns else 0.0,
            "max_drawdown": max_drawdown(equity_series) if len(equity_series) > 0 else 0.0,
            "trade_count": self.trade_count,
            "position_history": self.position_history,
            "equity_curve": equity_series,
        }
        logger.info("Backtest complete: final_equity=%.2f trades=%d sharpe=%.2f max_dd=%.2f", stats["final_equity"], stats["trade_count"], stats["sharpe"], stats["max_drawdown"])
        return stats


# -------------------------
# Main execution helper (preserved)
# -------------------------
def run_trading_system(price_df: pd.DataFrame, cfg_dict: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    price_df must be a DataFrame with columns: ['Open','High','Low','Close','Volume']
    """
    if cfg_dict is None:
        cfg_dict = CONFIG
    cfg = StrategyConfig(**cfg_dict)
    strategy = Strategy(cfg)
    backtester = Backtester(price_df, strategy, cfg)
    results = backtester.run_backtest()
    return results


# -------------------------
# Example usage (kept as reference, commented to preserve module importability)
# -------------------------
if __name__ == "__main__":
    # Example: generate synthetic data or load from CSV for demonstration
    import datetime as dt

    dates = pd.date_range(start=dt.date(2018, 1, 1), periods=500, freq="B")
    np.random.seed(42)
    price = np.cumprod(1 + np.random.normal(0.0005, 0.01, len(dates))) * 100
    high = price * (1 + np.random.uniform(0.0, 0.01, len(dates)))
    low = price * (1 - np.random.uniform(0.0, 0.01, len(dates)))
    openp = price * (1 + np.random.uniform(-0.005, 0.005, len(dates)))
    volume = np.random.randint(100, 1000, len(dates))

    df = pd.DataFrame({
        "Open": openp,
        "High": high,
        "Low": low,
        "Close": price,
        "Volume": volume,
    }, index=dates)

    results = run_trading_system(df)
    logger.info("Example run: final equity %.2f, trades %d, max drawdown %.2f, sharpe %.2f",
                results["final_equity"], results["trade_count"], results["max_drawdown"], results["sharpe"])