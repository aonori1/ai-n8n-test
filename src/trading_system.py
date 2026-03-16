import logging
import math
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional, Dict, Any

# Configuration (preserve and expose for easy tuning)
CONFIG: Dict[str, Any] = {
    "initial_capital": 100000.0,
    "target_annual_vol": 0.10,           # target portfolio volatility (10% annual)
    "vol_lookback_days": 21,             # window to estimate volatility (in trading days)
    "atr_lookback": 14,                  # ATR lookback for stop placement
    "ema_short": 21,
    "ema_long": 200,
    "rsi_period": 14,
    "rsi_oversold": 30,
    "rsi_overbought": 70,
    "adx_period": 14,
    "adx_trend_threshold": 20,           # only trade in presence of trend
    "max_position_size_pct": 0.20,       # max fraction of portfolio in single trade
    "max_leverage": 1.0,
    "commission_per_trade": 0.0,
    "slippage_pct": 0.0005,              # 5 bps
    "stop_atr_multiplier": 3.0,          # initial stop: 3 * ATR
    "trailing_stop_atr_multiplier": 2.0, # trailing stop uses 2 * ATR
    "min_trade_interval_bars": 5,        # cooldown between trades
    "max_consecutive_losses": 5,         # safety stop if too many losses
    "max_drawdown_stop_pct": 0.25,       # stop trading if drawdown exceeds 25%
    "logging_level": logging.INFO,
    "allow_short": False,                # keep simple long-only unless configured
}

# Logger setup (preserve logging)
logger = logging.getLogger("TradingSystem")
if not logger.handlers:
    ch = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s", "%Y-%m-%d %H:%M:%S"
    )
    ch.setFormatter(formatter)
    logger.addHandler(ch)
logger.setLevel(CONFIG.get("logging_level", logging.INFO))


@dataclass
class Position:
    entry_price: float
    size: float
    entry_idx: int
    stop_price: float
    trailing_stop_price: Optional[float] = None
    direction: int = 1  # 1 for long, -1 for short (if enabled)
    realized_pnl: float = 0.0
    closed: bool = False
    exit_idx: Optional[int] = None
    exit_price: Optional[float] = None


class TradingSystem:
    """
    TradingSystem - preserves previous architecture: indicator generation, signal logic, position
    management, backtest loop, logging and configuration.
    Improvements applied:
      - Volatility-targeted position sizing (improves risk-adjusted returns)
      - ATR-based stop-loss and trailing stop (reduces drawdown)
      - Trend filter via EMA and ADX (reduces noise & avoids counter-trend trades)
      - RSI to avoid entering into overbought conditions
      - Cooldown between trades, max position cap, consecutive loss safety stop
      - Max drawdown stop to halt trading if catastrophic drawdown occurs
    """

    def __init__(self, price_df: pd.DataFrame, config: Dict[str, Any], logger: logging.Logger):
        """
        price_df must include columns: ['open', 'high', 'low', 'close', 'volume'] indexed by timestamp
        """
        self.df = price_df.copy().reset_index(drop=True)
        self.config = config
        self.logger = logger
        self.positions: list[Position] = []
        self.trade_history: list[Dict[str, Any]] = []
        self.equity_curve = pd.Series(dtype=float)
        self._init_state()

    def _init_state(self):
        self.cash = float(self.config["initial_capital"])
        self.portfolio_value = float(self.config["initial_capital"])
        self.last_trade_idx = -9999
        self.consecutive_losses = 0
        self.peak_equity = self.cash
        self.current_position: Optional[Position] = None
        # Preallocate arrays for performance
        n = len(self.df)
        self.df["ema_short"] = np.nan
        self.df["ema_long"] = np.nan
        self.df["atr"] = np.nan
        self.df["volatility"] = np.nan
        self.df["rsi"] = np.nan
        self.df["adx"] = np.nan
        self.df["signal"] = 0  # 1 buy signal, -1 sell signal
        self.equity_series = np.full(n, np.nan)

    # Indicator computations
    @staticmethod
    def _ema(series: pd.Series, span: int) -> pd.Series:
        return series.ewm(span=span, adjust=False).mean()

    @staticmethod
    def _true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
        prev_close = close.shift(1)
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr

    def _compute_atr(self):
        tr = self._true_range(self.df["high"], self.df["low"], self.df["close"])
        atr = tr.ewm(alpha=1 / self.config["atr_lookback"], adjust=False).mean()
        self.df["atr"] = atr

    def _compute_volatility(self):
        # Compute daily returns volatility annualized
        returns = self.df["close"].pct_change()
        vol = returns.rolling(window=self.config["vol_lookback_days"], min_periods=2).std()
        annual_factor = math.sqrt(252)
        self.df["volatility"] = vol * annual_factor

    def _compute_rsi(self):
        delta = self.df["close"].diff()
        up = delta.clip(lower=0)
        down = -1 * delta.clip(upper=0)
        ema_up = up.ewm(alpha=1 / self.config["rsi_period"], adjust=False).mean()
        ema_down = down.ewm(alpha=1 / self.config["rsi_period"], adjust=False).mean()
        rs = ema_up / (ema_down + 1e-12)
        rsi = 100 - (100 / (1 + rs))
        self.df["rsi"] = rsi

    def _compute_adx(self):
        # Simplified ADX calculation
        up_move = self.df["high"].diff()
        down_move = -self.df["low"].diff()
        plus_dm = ((up_move > down_move) & (up_move > 0)) * up_move
        minus_dm = ((down_move > up_move) & (down_move > 0)) * down_move
        tr = self._true_range(self.df["high"], self.df["low"], self.df["close"])
        atr = tr.rolling(self.config["adx_period"], min_periods=1).mean()
        plus_di = 100 * (plus_dm.ewm(alpha=1 / self.config["adx_period"], adjust=False).mean() / (atr + 1e-12))
        minus_di = 100 * (minus_dm.ewm(alpha=1 / self.config["adx_period"], adjust=False).mean() / (atr + 1e-12))
        dx = (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-12) * 100
        adx = dx.ewm(alpha=1 / self.config["adx_period"], adjust=False).mean()
        self.df["adx"] = adx.fillna(0)

    def compute_indicators(self):
        self.df["ema_short"] = self._ema(self.df["close"], self.config["ema_short"])
        self.df["ema_long"] = self._ema(self.df["close"], self.config["ema_long"])
        self._compute_atr()
        self._compute_volatility()
        self._compute_rsi()
        self._compute_adx()

    def generate_signals(self):
        """
        Signal rules:
          - Long entry when ema_short crosses above ema_long, ADX indicates trending market,
            RSI not overbought, and price above ema_long (trend filter).
          - Exit when ema_short crosses below ema_long OR stop is hit OR RSI overbought.
        The signal is 1 for potential entry and -1 for exit. Actual position management uses stops.
        """
        ema_s = self.df["ema_short"]
        ema_l = self.df["ema_long"]

        # Use crossover detection with shift
        prev_diff = (ema_s - ema_l).shift(1)
        curr_diff = (ema_s - ema_l)

        cross_up = (prev_diff <= 0) & (curr_diff > 0)
        cross_down = (prev_diff >= 0) & (curr_diff < 0)

        # Trend filter via ADX
        strong_trend = self.df["adx"] >= self.config["adx_trend_threshold"]

        # RSI filter
        rsi_ok = self.df["rsi"] < self.config["rsi_overbought"]

        # Price above long-term EMA for long-only orientation
        price_above_ema_long = self.df["close"] > ema_l

        # Compile signals
        entries = cross_up & strong_trend & rsi_ok & price_above_ema_long
        exits = cross_down | (self.df["rsi"] > self.config["rsi_overbought"])

        # Soften signals: require some smoothing - avoid whipsaws by requiring persistent signal for 1 bar
        entries = entries & entries.shift(1).fillna(True)
        exits = exits & exits.shift(1).fillna(True)

        self.df.loc[entries, "signal"] = 1
        self.df.loc[exits, "signal"] = -1
        # else 0

    def _compute_position_size(self, idx: int) -> float:
        """
        Volatility-targeted position sizing:
        size = (target_vol / asset_vol) * portfolio_value
        Cap by max_position_size_pct and max_leverage
        """
        vol = self.df.loc[idx, "volatility"]
        if pd.isna(vol) or vol <= 0:
            # fallback to conservative small size
            proposed_pct = 0.01
        else:
            daily_target_vol = self.config["target_annual_vol"]
            # Position fraction of portfolio in asset. This is approximate as single-asset system.
            proposed_pct = float(daily_target_vol / vol)
        # constrain
        proposed_pct = max(0.0, min(proposed_pct, self.config["max_position_size_pct"] * self.config["max_leverage"]))
        # size in USD
        usd_size = proposed_pct * self.portfolio_value
        # convert to units
        price = self.df.loc[idx, "close"]
        units = usd_size / (price * (1 + self.config["slippage_pct"])) if price > 0 else 0.0
        return float(units)

    def _apply_slippage_and_commission(self, price: float) -> float:
        slippage = price * self.config["slippage_pct"]
        effective_price = price + slippage
        return effective_price

    def _enter_position(self, idx: int):
        if self.current_position is not None:
            return  # already in position

        # Safety checks
        if (idx - self.last_trade_idx) < self.config["min_trade_interval_bars"]:
            self.logger.debug(f"Trade cooldown active. idx {idx} last_trade_idx {self.last_trade_idx}")
            return

        # Stop trading if catastrophic drawdown
        if (self.peak_equity - self.portfolio_value) / max(1e-9, self.peak_equity) > self.config["max_drawdown_stop_pct"]:
            self.logger.warning("Max drawdown exceeded. Halting new trades.")
            return

        # Compute intended size
        units = self._compute_position_size(idx)
        if units <= 0:
            return

        entry_price = self._apply_slippage_and_commission(self.df.loc[idx, "close"])
        usd_required = units * entry_price
        if usd_required > self.cash:
            # If not enough cash, adjust units
            units = self.cash / entry_price
            usd_required = units * entry_price
            # If still negligible, abort
            if units < 1e-8:
                return

        # ATR-based initial stop
        atr = self.df.loc[idx, "atr"]
        stop_price = entry_price - self.config["stop_atr_multiplier"] * atr if not pd.isna(atr) else entry_price * 0.98

        pos = Position(
            entry_price=entry_price,
            size=units,
            entry_idx=idx,
            stop_price=stop_price,
            trailing_stop_price=stop_price,
            direction=1,
        )
        self.current_position = pos
        self.cash -= usd_required + self.config["commission_per_trade"]
        self.last_trade_idx = idx
        self.logger.info(
            f"ENTER idx={idx} price={entry_price:.4f} units={units:.4f} usd_used={usd_required:.2f} stop={stop_price:.4f}"
        )

    def _exit_position(self, idx: int, reason: str = "signal"):
        pos = self.current_position
        if pos is None:
            return
        exit_price = self._apply_slippage_and_commission(self.df.loc[idx, "close"])
        usd_received = pos.size * exit_price
        pnl = usd_received - (pos.size * pos.entry_price) - self.config["commission_per_trade"]
        self.cash += usd_received - self.config["commission_per_trade"]
        pos.exit_price = exit_price
        pos.exit_idx = idx
        pos.realized_pnl = pnl
        pos.closed = True
        self.positions.append(pos)
        self.trade_history.append(
            {
                "entry_idx": pos.entry_idx,
                "exit_idx": pos.exit_idx,
                "entry_price": pos.entry_price,
                "exit_price": pos.exit_price,
                "size": pos.size,
                "pnl": pos.realized_pnl,
                "reason": reason,
            }
        )
        self.logger.info(
            f"EXIT idx={idx} exit_price={exit_price:.4f} size={pos.size:.4f} pnl={pnl:.2f} reason={reason}"
        )
        # Update consecutive losses
        if pnl < 0:
            self.consecutive_losses += 1
            self.logger.debug(f"Consecutive losses incremented to {self.consecutive_losses}")
        else:
            self.consecutive_losses = 0
        self.current_position = None

    def _update_trailing_stop(self, idx: int):
        pos = self.current_position
        if pos is None:
            return
        atr = self.df.loc[idx, "atr"]
        if pd.isna(atr):
            return
        # For long positions, raise trailing stop to max(previous trailing, close - multiplier*atr)
        candidate = self.df.loc[idx, "close"] - self.config["trailing_stop_atr_multiplier"] * atr
        if candidate > pos.trailing_stop_price:
            old = pos.trailing_stop_price
            pos.trailing_stop_price = candidate
            self.logger.debug(f"Trailing stop updated from {old:.4f} to {pos.trailing_stop_price:.4f} at idx {idx}")

    def _check_stops(self, idx: int):
        pos = self.current_position
        if pos is None:
            return
        low = self.df.loc[idx, "low"]
        # Stop hit if low <= stop_price or low <= trailing_stop_price
        if low <= pos.stop_price:
            self.logger.info(f"Initial stop hit at idx {idx} low={low:.4f} stop={pos.stop_price:.4f}")
            self._exit_position(idx, reason="stop_initial")
        elif pos.trailing_stop_price is not None and low <= pos.trailing_stop_price:
            self.logger.info(f"Trailing stop hit at idx {idx} low={low:.4f} trailing_stop={pos.trailing_stop_price:.4f}")
            self._exit_position(idx, reason="stop_trailing")

    def run_backtest(self):
        self.compute_indicators()
        self.generate_signals()

        n = len(self.df)
        for idx in range(n):
            # Update portfolio value: cash + mark-to-market of any open position
            if self.current_position is not None:
                mtm = self.current_position.size * self.df.loc[idx, "close"]
                self.portfolio_value = self.cash + mtm
            else:
                self.portfolio_value = self.cash

            # Track peak equity and drawdown
            if self.portfolio_value > self.peak_equity:
                self.peak_equity = self.portfolio_value

            drawdown = (self.peak_equity - self.portfolio_value) / max(1e-9, self.peak_equity)
            if drawdown > self.config["max_drawdown_stop_pct"]:
                self.logger.warning(f"Drawdown {drawdown:.2%} exceeded limit. Halting any new entries.")
                # We still allow exits to close existing positions
                allow_new_entries = False
            else:
                allow_new_entries = True

            # If in position, update trailing stops and check stops
            if self.current_position is not None:
                self._update_trailing_stop(idx)
                self._check_stops(idx)

            # Enforce consecutive losses safety stop: if exceeded, don't enter new trades
            if self.consecutive_losses >= self.config["max_consecutive_losses"]:
                allow_new_entries = False
                self.logger.warning("Consecutive loss limit hit. Pausing new entries until reset by a win.")

            # Handle signals
            sig = int(self.df.loc[idx, "signal"])

            if sig == 1 and self.current_position is None and allow_new_entries:
                # Entry condition
                self._enter_position(idx)

            elif sig == -1 and self.current_position is not None:
                # Exit condition
                self._exit_position(idx, reason="signal_exit")

            # Update equity series
            if self.current_position is not None:
                mtm = self.current_position.size * self.df.loc[idx, "close"]
                self.equity_series[idx] = self.cash + mtm
            else:
                self.equity_series[idx] = self.cash

        # Close any open positions at the end
        if self.current_position is not None:
            self._exit_position(len(self.df) - 1, reason="end_of_backtest")

        # Store equity curve
        self.df["equity"] = self.equity_series
        self.equity_curve = pd.Series(self.equity_series, index=self.df.index).ffill().fillna(self.config["initial_capital"])

        self.logger.info("Backtest complete.")
        return self._generate_results()

    def _generate_results(self):
        # Compute returns and statistics
        eq = self.equity_curve.fillna(method="ffill").fillna(self.config["initial_capital"])
        returns = eq.pct_change().fillna(0)
        cum_return = eq.iloc[-1] / eq.iloc[0] - 1
        annual_return = (1 + cum_return) ** (252 / len(eq)) - 1 if len(eq) > 0 else 0.0
        annual_vol = returns.std() * math.sqrt(252)
        sharpe = (returns.mean() * 252) / (returns.std() * math.sqrt(252) + 1e-12)
        drawdown_series = (eq.cummax() - eq) / eq.cummax()
        max_drawdown = drawdown_series.max()

        trades_df = pd.DataFrame(self.trade_history)
        win_rate = None
        avg_win = None
        avg_loss = None
        if not trades_df.empty:
            wins = trades_df[trades_df["pnl"] > 0]
            losses = trades_df[trades_df["pnl"] <= 0]
            win_rate = len(wins) / len(trades_df)
            avg_win = wins["pnl"].mean() if not wins.empty else 0.0
            avg_loss = losses["pnl"].mean() if not losses.empty else 0.0

        results = {
            "equity_curve": eq,
            "total_return": float(cum_return),
            "annual_return": float(annual_return),
            "annual_vol": float(annual_vol),
            "sharpe": float(sharpe),
            "max_drawdown": float(max_drawdown),
            "n_trades": len(trades_df),
            "win_rate": float(win_rate) if win_rate is not None else None,
            "avg_win": float(avg_win) if avg_win is not None else None,
            "avg_loss": float(avg_loss) if avg_loss is not None else None,
            "trade_log": trades_df,
        }
        self.logger.info(
            f"Results: total_return={cum_return:.2%} annual_return={annual_return:.2%} "
            f"sharpe={sharpe:.2f} max_dd={max_drawdown:.2%} n_trades={len(trades_df)}"
        )
        return results


# Example usage notes (preserve configuration pattern; actual calling code should supply price_df):
# - Input DataFrame 'price_df' must contain 'open','high','low','close','volume' columns.
# - To run a backtest:
#       ts = TradingSystem(price_df, CONFIG, logger)
#       results = ts.run_backtest()
#       equity_curve = results['equity_curve']
# The example usage is intentionally commented out to keep this module import-safe.

# If run as a script for a quick smoke test, generate a synthetic price series (optional).
if __name__ == "__main__":
    # Minimal smoke test with synthetic data if no real data provided.
    np.random.seed(42)
    days = 500
    dt_index = pd.date_range("2020-01-01", periods=days, freq="B")
    # Generate a synthetic random walk price series with modest drift
    returns = np.random.normal(loc=0.0003, scale=0.01, size=days)
    price = 100 * (1 + pd.Series(returns)).cumprod()
    price_df = pd.DataFrame({
        "open": price * (1 + np.random.normal(0, 0.001, size=days)),
        "high": price * (1 + np.random.normal(0.002, 0.0025, size=days)).clip(lower=1.0),
        "low": price * (1 - np.random.normal(0.002, 0.0025, size=days)).clip(lower=0.0),
        "close": price,
        "volume": 1000 + np.random.randint(-100, 100, size=days),
    }, index=dt_index)

    ts = TradingSystem(price_df, CONFIG, logger)
    results = ts.run_backtest()
    # Log summary
    logger.info(f"Final equity: {results['equity_curve'].iloc[-1]:.2f}")
    logger.info(f"Sharpe: {results['sharpe']:.2f}, Max Drawdown: {results['max_drawdown']:.2%}, Trades: {results['n_trades']}")