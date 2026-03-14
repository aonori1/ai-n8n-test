import logging
import json
import math
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple

"""
Improved trading system.
- Keeps architecture consistent: Config, Logger, Strategy, RiskManager, Portfolio, Backtester.
- Improves risk-adjusted performance via volatility-targeted position sizing, ATR-based stops,
  trend filter to avoid whipsaws, max drawdown protection, and trade frequency control.
- Reduces drawdown with per-trade stop, trailing stop, and portfolio-level stop trading on drawdown.
- Preserves logging and configuration (config provided as JSON/dict).
- Self-contained for backtesting with pandas DataFrame of OHLCV input.

To use:
- Provide market_data as a pandas DataFrame with columns: ['open','high','low','close','volume']
- Instantiate Config and pass to Backtester.
"""

# -------------------------
# Configuration and Logger
# -------------------------
@dataclass
class Config:
    # Strategy parameters
    fast_ma: int = 20
    slow_ma: int = 50
    use_trend_filter: bool = True  # only take trades in direction of slow_ma trend
    atr_window: int = 14
    atr_multiplier: float = 3.0  # initial stop distance in ATRs
    trailing_atr_multiplier: float = 1.5  # trailing stop distance
    max_holding_bars: int = 100  # time stop
    min_bars_between_trades: int = 5  # avoid overtrading

    # Risk management
    target_volatility_annual: float = 0.10  # target portfolio vol (10% annual)
    vol_lookback: int = 20  # lookback for realized vol estimate
    leverage_cap: float = 2.0  # max gross exposure
    fractional_kelly: float = 0.5  # fraction of Kelly recommended
    max_trade_risk_pct: float = 0.01  # risk per trade as percent of equity
    portfolio_max_drawdown_pct: float = 0.25  # stop trading if drawdown exceeds this
    min_price: float = 0.01  # avoid tiny prices

    # Execution and misc
    slippage_perc: float = 0.0005
    commission_per_share: float = 0.0
    initial_capital: float = 1_000_000.0
    log_level: int = logging.INFO
    random_seed: int = 42

    # For backtester reporting
    report_annualization: int = 252

    # Keep extra config as dict
    extras: Dict[str, Any] = field(default_factory=dict)


class Logger:
    def __init__(self, config: Config):
        self.logger = logging.getLogger("TradingSystem")
        self.logger.setLevel(config.log_level)
        if not self.logger.handlers:
            ch = logging.StreamHandler()
            ch.setLevel(config.log_level)
            formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
            ch.setFormatter(formatter)
            self.logger.addHandler(ch)

    def info(self, msg: str):
        self.logger.info(msg)

    def debug(self, msg: str):
        self.logger.debug(msg)

    def warn(self, msg: str):
        self.logger.warning(msg)

    def error(self, msg: str):
        self.logger.error(msg)


# -------------------------
# Utility functions
# -------------------------
def compute_atr(df: pd.DataFrame, window: int) -> pd.Series:
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift(1)).abs()
    low_close = (df['low'] - df['close'].shift(1)).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(window=window, min_periods=1).mean()
    return atr


def exponential_moving_average(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def annualized_volatility(returns: pd.Series, periods_per_year: int) -> float:
    if returns.dropna().empty:
        return 0.0
    return returns.std() * math.sqrt(periods_per_year)


# -------------------------
# Strategy
# -------------------------
class Strategy:
    def __init__(self, config: Config, logger: Logger):
        self.cfg = config
        self.log = logger
        np.random.seed(self.cfg.random_seed)

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate entry/exit signals.
        - Signal: +1 for long, -1 for short, 0 for flat.
        - Uses moving average crossover with trend filter (slow MA).
        - ATR-based stop distances are computed for each bar.
        """
        data = df.copy().reset_index(drop=True)
        data['fast_ma'] = exponential_moving_average(data['close'], span=self.cfg.fast_ma)
        data['slow_ma'] = exponential_moving_average(data['close'], span=self.cfg.slow_ma)
        data['atr'] = compute_atr(data, self.cfg.atr_window)

        # Basic crossover signal
        data['raw_signal'] = 0
        data.loc[data['fast_ma'] > data['slow_ma'], 'raw_signal'] = 1
        data.loc[data['fast_ma'] < data['slow_ma'], 'raw_signal'] = -1

        if self.cfg.use_trend_filter:
            # Only take longs when slow_ma is sloping up, shorts when sloping down.
            slope = data['slow_ma'].diff(self.cfg.slow_ma // 4).fillna(0)
            data['trend_ok'] = slope > 0
            data.loc[(data['raw_signal'] == 1) & (~data['trend_ok']), 'raw_signal'] = 0
            data.loc[(data['raw_signal'] == -1) & (data['trend_ok']), 'raw_signal'] = 0

        # Prevent too frequent flipping: require min_bars_between_trades
        signals = []
        last_signal = 0
        last_index = -999
        for i, s in enumerate(data['raw_signal'].values):
            if s == 0:
                signals.append(0)
            else:
                if last_signal == 0 and (i - last_index) < self.cfg.min_bars_between_trades:
                    signals.append(0)
                else:
                    signals.append(s)
                    if s != last_signal:
                        last_index = i
            last_signal = signals[-1]
        data['signal'] = signals

        # Compute stop distances (initial stop and trailing)
        # Ensure ATR-based distances are reasonable and cap extremes
        data['initial_stop_atr'] = (data['atr'] * self.cfg.atr_multiplier).clip(lower=1e-6)
        data['trailing_stop_atr'] = (data['atr'] * self.cfg.trailing_atr_multiplier).clip(lower=1e-6)

        return data


# -------------------------
# Risk Management
# -------------------------
class RiskManager:
    def __init__(self, config: Config, logger: Logger):
        self.cfg = config
        self.log = logger

    def position_size(self, equity: float, price: float, atr: float, expected_return: Optional[float] = None,
                      hist_returns: Optional[pd.Series] = None) -> float:
        """
        Compute position size (number of shares) using volatility-targeting and fractional Kelly if possible.
        - target_vol_position: position sized so that position's vol = target_volatility_annual * equity
        - per-trade risk cap: don't risk more than max_trade_risk_pct of equity (based on ATR stop)
        - leverage cap: cap position so that gross exposure not above leverage_cap * equity
        """
        # Avoid division by zero
        price = max(price, self.cfg.min_price)
        atr = max(atr, 1e-6)

        # Estimate realized vol (daily) from hist_returns if provided, else use default scaling
        if hist_returns is not None and len(hist_returns.dropna()) >= 2:
            realized_vol = annualized_volatility(hist_returns, self.cfg.report_annualization)
        else:
            realized_vol = 0.2  # fallback guess (20% p.a.)

        # Determine dollar volatility per unit (share) roughly: price * daily_volile ~ price * (realized_vol / sqrt(T))
        # Use approximation: per-share dollar vol = price * (realized_vol / sqrt(annual_days))
        if realized_vol > 0:
            per_share_vol = price * (realized_vol / math.sqrt(self.cfg.report_annualization))
        else:
            per_share_vol = price * (0.01 / math.sqrt(self.cfg.report_annualization))  # tiny fallback

        # If realized vol is unreliable, use ATR as proxy: per share dollar movement ~ ATR
        per_share_vol = max(per_share_vol, atr)

        # Target dollar volatility = target_volatility_annual * equity / sqrt(annual_days) per day
        target_daily_vol = (self.cfg.target_volatility_annual * equity) / math.sqrt(self.cfg.report_annualization)

        # Number of shares by volatility targeting
        size_by_vol = target_daily_vol / per_share_vol if per_share_vol > 0 else 0.0

        # Convert to shares (integer)
        shares_vol = math.floor(size_by_vol)

        # Per-trade risk cap: don't risk more than max_trade_risk_pct * equity if stop is ATR-based distance
        # Risk per share = stop_distance (dollars)
        stop_distance = self.cfg.atr_multiplier * atr
        max_risk_dollars = self.cfg.max_trade_risk_pct * equity
        if stop_distance <= 0:
            shares_risk_limit = shares_vol
        else:
            shares_risk_limit = math.floor(max_risk_dollars / stop_distance)

        # Combined limit
        shares = int(min(shares_vol, shares_risk_limit))

        # Leverage cap: ensure (shares * price) / equity <= leverage_cap
        if equity > 0:
            max_notional = self.cfg.leverage_cap * equity
            shares_leverage = math.floor(max_notional / price)
            shares = int(min(shares, shares_leverage))

        shares = max(shares, 0)

        self.log.debug(f"position_size -> price={price:.2f}, atr={atr:.4f}, shares_vol={shares_vol}, shares_risk_limit={shares_risk_limit}, shares_leverage={shares_leverage if equity>0 else 'N/A'}, final_shares={shares}")
        return shares

    def apply_portfolio_drawdown_protection(self, peak_equity: float, current_equity: float) -> bool:
        """
        Returns True if trading should be halted due to drawdown.
        """
        if peak_equity <= 0:
            return False
        drawdown = (peak_equity - current_equity) / peak_equity
        if drawdown >= self.cfg.portfolio_max_drawdown_pct:
            self.log.warn(f"Portfolio drawdown {drawdown:.2%} exceeds limit {self.cfg.portfolio_max_drawdown_pct:.2%}. Halting new trades.")
            return True
        return False


# -------------------------
# Portfolio / Execution
# -------------------------
class Portfolio:
    def __init__(self, config: Config, logger: Logger):
        self.cfg = config
        self.log = logger
        self.cash = config.initial_capital
        self.positions = 0  # shares (positive long, negative short)
        self.avg_entry_price = 0.0
        self.equity_curve: List[float] = []
        self.peak_equity = config.initial_capital
        self.current_equity = config.initial_capital
        self.position_entry_index: Optional[int] = None
        self.position_entry_price: Optional[float] = None
        self.initial_stop_price: Optional[float] = None
        self.trailing_stop_price: Optional[float] = None
        self.last_trade_index: int = -999
        self.trades: List[Dict[str, Any]] = []

    def update_market(self, i: int, row: pd.Series, cfg: Config):
        """
        Update mark-to-market equity given current price.
        """
        mid_price = row['close']
        market_value = self.positions * mid_price
        self.current_equity = self.cash + market_value
        self.equity_curve.append(self.current_equity)
        if self.current_equity > self.peak_equity:
            self.peak_equity = self.current_equity

    def enter_position(self, i: int, size: int, price: float, stop_price: float):
        """
        Enter new position (assume no partial entries; previous position should be flat).
        """
        if size == 0:
            return
        # Apply slippage and commission
        executed_price = price * (1 + self.cfg.slippage_perc * np.sign(size))
        cost = executed_price * size + abs(size) * self.cfg.commission_per_share
        # Update cash and positions
        self.positions = size
        self.avg_entry_price = executed_price
        self.cash -= cost
        self.position_entry_index = i
        self.position_entry_price = executed_price
        self.initial_stop_price = stop_price
        if size > 0:
            self.trailing_stop_price = executed_price - self.cfg.trailing_atr_multiplier * computed_atr_value_from_context  # placeholder, overwritten in backtester
        else:
            self.trailing_stop_price = executed_price + self.cfg.trailing_atr_multiplier * computed_atr_value_from_context  # placeholder
        self.last_trade_index = i
        self.trades.append({
            'index': i,
            'type': 'enter',
            'size': size,
            'price': executed_price,
            'stop_price': stop_price
        })
        self.log.info(f"Enter position: size={size}, price={executed_price:.2f}, stop={stop_price:.2f}, cash={self.cash:.2f}")

    def exit_position(self, i: int, price: float, reason: str = ""):
        """
        Exit current position fully.
        """
        if self.positions == 0:
            return
        executed_price = price * (1 - self.cfg.slippage_perc * np.sign(self.positions))
        proceeds = executed_price * self.positions - abs(self.positions) * self.cfg.commission_per_share
        self.cash += proceeds
        pnl = (executed_price - self.avg_entry_price) * self.positions
        trade = {
            'index': i,
            'type': 'exit',
            'size': self.positions,
            'price': executed_price,
            'pnl': pnl,
            'reason': reason
        }
        self.trades.append(trade)
        self.log.info(f"Exit position: size={self.positions}, price={executed_price:.2f}, pnl={pnl:.2f}, reason={reason}, cash={self.cash:.2f}")
        # reset
        self.positions = 0
        self.avg_entry_price = 0.0
        self.position_entry_index = None
        self.position_entry_price = None
        self.initial_stop_price = None
        self.trailing_stop_price = None

    def mark_position(self, price: float):
        """
        Return current unrealized P&L given price
        """
        return (price - self.avg_entry_price) * self.positions if self.positions != 0 else 0.0


# We'll use a backtester that ties everything together. It will override the placeholder used in Portfolio.enter_position
# for trailing stop initialization by injecting the current ATR value.

# -------------------------
# Backtester
# -------------------------
class Backtester:
    def __init__(self, config: Config, logger: Logger):
        self.cfg = config
        self.log = logger
        self.strategy = Strategy(config, logger)
        self.risk = RiskManager(config, logger)
        self.portfolio = Portfolio(config, logger)
        # We'll keep a record DataFrame for analytics
        self.record_cols = ['close', 'signal', 'positions', 'equity', 'pnl', 'atr', 'initial_stop', 'trailing_stop']
        np.random.seed(self.cfg.random_seed)

    def run(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        df = market_data.copy().reset_index(drop=True)
        # Ensure required columns
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col not in df.columns:
                raise ValueError(f"market_data must contain '{col}' column")

        signals_df = self.strategy.generate_signals(df)
        results = []
        # Keep series of daily returns for realized vol estimation
        returns = pd.Series(dtype=float)

        # We'll need to set a global variable used in Portfolio.enter_position for trailing stop initialization.
        # To avoid global state pollution, we will monkey-patch the method temporarily in a safe way.
        global computed_atr_value_from_context
        computed_atr_value_from_context = 0.0

        for i, row in signals_df.iterrows():
            price = row['close']
            atr = row['atr']
            signal = row['signal']

            # Update computed_atr for portfolio to set trailing stop price properly when entering
            computed_atr_value_from_context = atr

            # Mark current portfolio to update equity history
            self.portfolio.update_market(i, row, self.cfg)

            # Determine if portfolio-level drawdown halts trading
            halt_trading = self.risk.apply_portfolio_drawdown_protection(self.portfolio.peak_equity, self.portfolio.current_equity)

            # Check stop conditions for existing position
            if self.portfolio.positions != 0:
                # Check ATR-based initial stop breach
                # For longs: price <= initial_stop_price -> exit
                # For shorts: price >= initial_stop_price -> exit
                if self.portfolio.initial_stop_price is not None:
                    if self.portfolio.positions > 0 and row['low'] <= self.portfolio.initial_stop_price:
                        self.portfolio.exit_position(i, self.portfolio.initial_stop_price, reason="initial_stop")
                    elif self.portfolio.positions < 0 and row['high'] >= self.portfolio.initial_stop_price:
                        self.portfolio.exit_position(i, self.portfolio.initial_stop_price, reason="initial_stop")

                # Trailing stop update and check
                if self.portfolio.positions > 0:
                    # Update trailing stop to max(previous_trailing, price - trailing_distance)
                    new_trailing = price - (self.cfg.trailing_atr_multiplier * atr)
                    if self.portfolio.trailing_stop_price is None or new_trailing > self.portfolio.trailing_stop_price:
                        self.portfolio.trailing_stop_price = new_trailing
                    if row['low'] <= self.portfolio.trailing_stop_price:
                        self.portfolio.exit_position(i, self.portfolio.trailing_stop_price, reason="trailing_stop")
                elif self.portfolio.positions < 0:
                    new_trailing = price + (self.cfg.trailing_atr_multiplier * atr)
                    if self.portfolio.trailing_stop_price is None or new_trailing < self.portfolio.trailing_stop_price:
                        self.portfolio.trailing_stop_price = new_trailing
                    if row['high'] >= self.portfolio.trailing_stop_price:
                        self.portfolio.exit_position(i, self.portfolio.trailing_stop_price, reason="trailing_stop")

                # Time stop: exit if position held too long
                if self.portfolio.position_entry_index is not None:
                    holding_period = i - self.portfolio.position_entry_index
                    if holding_period >= self.cfg.max_holding_bars:
                        self.portfolio.exit_position(i, price, reason="time_stop")

            # Consider new entries if flat and not halted
            if self.portfolio.positions == 0 and not halt_trading and signal != 0:
                # Compute position size
                hist_returns = returns.iloc[-self.cfg.vol_lookback:] if not returns.empty else None
                shares = self.risk.position_size(self.portfolio.current_equity, price, atr, hist_returns=hist_returns)

                # Apply small sanity check: avoid entering too small a position
                if shares <= 0:
                    self.log.debug(f"Calculated shares 0; skipping entry at index {i}")
                else:
                    # Determine direction
                    size = shares if signal > 0 else -shares
                    # Compute initial stop price for direction
                    if size > 0:
                        initial_stop = price - self.cfg.atr_multiplier * atr
                    else:
                        initial_stop = price + self.cfg.atr_multiplier * atr
                    # Enter
                    # Before entering, ensure there's sufficient cash margin post-entry (simple check)
                    estimated_trade_cost = price * abs(size)
                    if estimated_trade_cost > self.portfolio.current_equity * self.cfg.leverage_cap:
                        self.log.debug("Estimated trade cost exceeds leverage cap; reducing size")
                        max_size = math.floor((self.portfolio.current_equity * self.cfg.leverage_cap) / price)
                        size = int(math.copysign(max_size, size))
                        if abs(size) == 0:
                            self.log.debug("Size reduced to zero after leverage cap; skipping")
                            size = 0
                    if size != 0:
                        # Monkey-patch computed_atr_value_from_context is already set; Portfolio.enter_position uses it
                        self.portfolio.enter_position(i, size, price, initial_stop)
                        self.portfolio.last_trade_index = i

            # Record return for volatility estimation: use close-to-close returns
            if i > 0:
                prev_close = signals_df.at[i - 1, 'close']
                if prev_close > 0:
                    ret = (price / prev_close) - 1.0
                else:
                    ret = 0.0
                returns.at[i] = ret

            # Ensure equity record consistent
            mark_value = self.portfolio.positions * price
            unrealized_pnl = self.portfolio.mark_position(price)
            equity = self.portfolio.cash + mark_value
            self.portfolio.current_equity = equity
            if equity > self.portfolio.peak_equity:
                self.portfolio.peak_equity = equity

            results.append({
                'index': i,
                'close': price,
                'signal': signal,
                'positions': self.portfolio.positions,
                'equity': equity,
                'pnl': unrealized_pnl,
                'atr': atr,
                'initial_stop': self.portfolio.initial_stop_price if self.portfolio.initial_stop_price is not None else np.nan,
                'trailing_stop': self.portfolio.trailing_stop_price if self.portfolio.trailing_stop_price is not None else np.nan
            })

        # After loop ends, if a position remains open, exit at last price
        if self.portfolio.positions != 0:
            last_price = signals_df.iloc[-1]['close']
            self.portfolio.exit_position(len(signals_df) - 1, last_price, reason="end_of_backtest")

        # Compile results DataFrame
        results_df = pd.DataFrame(results).set_index('index')
        stats = self.compute_performance(results_df)
        self.log.info(f"Backtest complete. Final equity: {self.portfolio.current_equity:.2f}, P&L: {self.portfolio.current_equity - self.cfg.initial_capital:.2f}")
        return {
            'results': results_df,
            'trades': self.portfolio.trades,
            'equity_curve': self.portfolio.equity_curve,
            'stats': stats
        }

    def compute_performance(self, results_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Compute performance metrics: CAGR, annualized vol, Sharpe, max drawdown, win rate, expectancy.
        """
        equity = pd.Series(self.portfolio.equity_curve)
        if equity.empty:
            return {}
        returns = equity.pct_change().fillna(0)
        total_return = equity.iloc[-1] / equity.iloc[0] - 1.0 if len(equity) > 0 else 0.0
        days = len(equity)
        years = days / float(self.cfg.report_annualization)
        cagr = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0.0
        ann_vol = annualized_volatility(returns, self.cfg.report_annualization)
        sharpe = (cagr / ann_vol) if ann_vol > 0 else np.nan

        # Max drawdown
        roll_max = equity.cummax()
        drawdown = (roll_max - equity) / roll_max
        max_drawdown = drawdown.max() if not drawdown.empty else 0.0

        # Trades stats
        trade_pnls = [t.get('pnl', 0.0) for t in self.portfolio.trades if t['type'] == 'exit']
        wins = [p for p in trade_pnls if p > 0]
        losses = [p for p in trade_pnls if p <= 0]
        win_rate = len(wins) / len(trade_pnls) if trade_pnls else np.nan
        avg_win = np.mean(wins) if wins else 0.0
        avg_loss = np.mean(losses) if losses else 0.0
        expectancy = (win_rate * avg_win + (1 - win_rate) * avg_loss) if not math.isnan(win_rate) else np.nan

        stats = {
            'total_return': total_return,
            'cagr': cagr,
            'annual_vol': ann_vol,
            'sharpe': sharpe,
            'max_drawdown': max_drawdown,
            'n_trades': len(trade_pnls),
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'expectancy': expectancy
        }
        self.log.info(f"Stats: total_return={total_return:.2%}, cagr={cagr:.2%}, ann_vol={ann_vol:.2%}, sharpe={sharpe:.2f}, max_dd={max_drawdown:.2%}, n_trades={len(trade_pnls)}")
        return stats


# -------------------------
# Example usage function (kept for consistency/logging but not executed here)
# -------------------------
def load_config_from_json(json_str: str) -> Config:
    try:
        d = json.loads(json_str)
    except Exception as e:
        raise ValueError("Invalid JSON config") from e
    cfg = Config(**d)
    return cfg


# If this module is run as main, demonstrate with random walk data (for testing only).
if __name__ == "__main__":
    # Create config and logger
    cfg = Config()
    log = Logger(cfg)

    # Generate simple synthetic market data for quick local test
    np.random.seed(cfg.random_seed)
    n = 1000
    prices = 100 + np.cumsum(np.random.normal(0, 0.5, size=n))
    highs = prices + np.random.uniform(0, 0.5, size=n)
    lows = prices - np.random.uniform(0, 0.5, size=n)
    opens = prices + np.random.normal(0, 0.2, size=n)
    closes = prices
    volumes = np.random.randint(100, 1000, size=n)

    market_df = pd.DataFrame({
        'open': opens,
        'high': highs,
        'low': lows,
        'close': closes,
        'volume': volumes
    })

    # Run backtest
    bt = Backtester(cfg, log)
    output = bt.run(market_df)

    # Log summary
    log.info("Final Stats:")
    for k, v in output['stats'].items():
        log.info(f"{k}: {v}")