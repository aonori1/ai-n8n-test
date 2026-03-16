import logging
import logging.handlers
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from collections import deque
import math
import json
import datetime

# -------------------------
# Configuration (preserved and extensible)
# -------------------------
DEFAULT_CONFIG: Dict = {
    "initial_capital": 1_000_000.0,
    "risk_per_trade": 0.01,                   # fraction of equity risked per trade
    "max_positions": 5,                       # max concurrent positions
    "atr_period": 21,
    "atr_multiplier_stop": 3.0,               # for initial stop (wider gives fewer stopouts)
    "trailing_stop_atr": 2.0,                 # ATR multiplier for trailing stop
    "ema_short": 20,
    "ema_long": 100,
    "signal_persistence": 2,                  # number of consecutive bars with valid signal required
    "max_drawdown_limit": 0.25,               # absolute equity drawdown fraction to stop trading
    "reduce_risk_after_drawdown": {           # adaptive risk scaling after drawdown
        "trigger": 0.10,                      # drawdown fraction to trigger risk reduction
        "scale_factor": 0.5                   # reduce risk_per_trade by this factor
    },
    "commission_per_trade": 1.0,              # fixed commission per trade (can be extended)
    "slippage": 0.0005,                       # slippage fraction of price
    "max_position_size_fraction": 0.25,       # no single position bigger than this fraction of equity
    "pyramiding": 1,                          # max additions to a position
    "min_price_for_trade": 0.01,              # avoid silly trades
    "logging": {
        "logfile": "trading_system.log",
        "level": "INFO"
    },
    "performance": {
        "equity_curve_smoothing_window": 5
    }
}

# -------------------------
# Logging setup (preserved)
# -------------------------
def setup_logger(log_config: Dict) -> logging.Logger:
    logger = logging.getLogger("TradingSystem")
    if logger.handlers:
        return logger  # already configured
    level = getattr(logging, log_config.get("level", "INFO").upper(), logging.INFO)
    logger.setLevel(level)
    formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')

    fh = logging.handlers.RotatingFileHandler(log_config.get("logfile", "trading_system.log"),
                                              maxBytes=10 * 1024 * 1024, backupCount=3)
    fh.setLevel(level)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger


# -------------------------
# Utility indicators
# -------------------------
def compute_ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def compute_atr(df: pd.DataFrame, period: int) -> pd.Series:
    high = df['high']
    low = df['low']
    close = df['close']
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period, min_periods=1).mean()  # simple moving avg of TR for stability
    return atr


# -------------------------
# Data classes
# -------------------------
@dataclass
class Position:
    symbol: str
    entry_price: float
    size: int
    entry_index: int
    stop_price: float
    trailing_atr: float
    additions: int = 0  # number of pyramiding additions made


@dataclass
class Trade:
    symbol: str
    entry_time: pd.Timestamp
    exit_time: Optional[pd.Timestamp]
    entry_price: float
    exit_price: Optional[float]
    size: int
    pnl: float
    fees: float


# -------------------------
# Trading system class (architecture preserved)
# -------------------------
class TradingSystem:
    def __init__(self, config: Dict = None, logger: Optional[logging.Logger] = None):
        self.config = dict(DEFAULT_CONFIG)
        if config:
            # deep merge simple keys
            self.config.update(config)
            # merge nested dicts (logging, reduce_risk_after_drawdown, performance)
            for k in ('logging', 'reduce_risk_after_drawdown', 'performance'):
                if k in (config or {}):
                    self.config[k].update(config[k])
        self.logger = logger or setup_logger(self.config['logging'])
        self.logger.debug("System initialized with config: %s", json.dumps(self.config, default=str))

        # State
        self.equity = self.config['initial_capital']
        self.cash = self.equity
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.equity_curve = []
        self.max_equity = self.equity
        self.drawdown = 0.0
        self.current_risk_per_trade = self.config['risk_per_trade']

    def analyze_backtest(self, equity_curve: pd.Series) -> Dict:
        # Basic performance metrics
        returns = equity_curve.pct_change().fillna(0)
        ann_return = (equity_curve.iloc[-1] / equity_curve.iloc[0]) ** (252 / len(equity_curve)) - 1 if len(equity_curve) > 1 else 0.0
        vol = returns.std() * np.sqrt(252)
        sharpe = (returns.mean() * 252) / vol if vol > 0 else 0.0
        dd_curve = (equity_curve.cummax() - equity_curve) / equity_curve.cummax()
        max_dd = dd_curve.max()
        return {
            "ann_return": ann_return,
            "volatility": vol,
            "sharpe": sharpe,
            "max_drawdown": max_dd
        }

    # Signal generation with smoothing and market filter
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        # Returns 1 for long, 0 for flat
        ema_short = compute_ema(df['close'], self.config['ema_short'])
        ema_long = compute_ema(df['close'], self.config['ema_long'])
        trend_ok = ema_short > ema_long

        ma_cross = ema_short - ema_long
        raw_signal = (ma_cross > 0).astype(int)

        # Apply persistence: require the last N bars to be True
        persistence = self.config['signal_persistence']
        if persistence > 1:
            rolling = raw_signal.rolling(window=persistence, min_periods=1).sum()
            signal = (rolling >= persistence).astype(int)
        else:
            signal = raw_signal

        # Market filter: only trade when close above long EMA (avoids mean-reversion zones)
        signal = signal.where(df['close'] > ema_long, other=0)

        return signal

    # position sizing using ATR and risk-per-trade
    def calculate_position_size(self, price: float, atr: float, equity: float) -> Tuple[int, float]:
        if np.isnan(atr) or atr <= 0:
            self.logger.debug("ATR invalid, defaulting to small size")
            return 0, 0.0
        risk_amount = equity * self.current_risk_per_trade
        # initial stop distance
        stop_distance = atr * self.config['atr_multiplier_stop']
        # per-share risk approximated by stop_distance
        per_unit_risk = max(stop_distance, self.config['min_price_for_trade'])  # guard vs zero price
        raw_size = math.floor(risk_amount / per_unit_risk) if per_unit_risk > 0 else 0

        # Limit position to fraction of equity to avoid concentration
        max_position_dollars = equity * self.config['max_position_size_fraction']
        max_size_by_cap = math.floor(max_position_dollars / price) if price > 0 else 0

        size = int(min(raw_size, max_size_by_cap))
        # Ensure at least 1 share if size positive and price not too small
        if size < 1 or price < self.config['min_price_for_trade']:
            return 0, 0.0

        estimated_stop = price - stop_distance
        return size, estimated_stop

    def apply_slippage_and_commission(self, price: float, side: str = "buy") -> float:
        # simple slippage model: fraction of price
        slippage = self.config['slippage'] * price
        price_adj = price + slippage if side == "buy" else price - slippage
        return price_adj

    def update_trailing_stop(self, pos: Position, current_price: float, atr: float) -> float:
        if atr <= 0:
            return pos.stop_price
        trailing_distance = atr * self.config['trailing_stop_atr']
        new_stop = max(pos.stop_price, current_price - trailing_distance)
        return new_stop

    # Manage risk during backtest loop
    def risk_manager_before_entry(self, df: pd.DataFrame, i: int, symbol: str):
        # check overall drawdown limit
        if (self.max_equity - self.equity) / max(1.0, self.max_equity) >= self.config['max_drawdown_limit']:
            self.logger.warning("Max drawdown limit reached. Halting new entries.")
            return False

        # adapt risk per trade if previous drawdown exceeded trigger
        current_dd = (self.max_equity - self.equity) / max(1.0, self.max_equity)
        trigger = self.config['reduce_risk_after_drawdown']['trigger']
        if current_dd >= trigger:
            old = self.current_risk_per_trade
            self.current_risk_per_trade = max(0.001, old * self.config['reduce_risk_after_drawdown']['scale_factor'])
            self.logger.info("Reducing risk per trade from %.4f to %.4f due to drawdown %.4f",
                             old, self.current_risk_per_trade, current_dd)
        return True

    def enter_position(self, df: pd.DataFrame, i: int, size: int, entry_price: float, stop_price: float):
        # execute buy
        executed_price = self.apply_slippage_and_commission(entry_price, side="buy")
        cost = executed_price * size + self.config['commission_per_trade']
        if cost > self.cash:
            self.logger.debug("Insufficient cash to enter position: cost=%.2f cash=%.2f", cost, self.cash)
            return None
        self.cash -= cost
        pos = Position(symbol="SYNTH", entry_price=executed_price, size=size, entry_index=i,
                       stop_price=stop_price, trailing_atr=self.config['trailing_stop_atr'])
        self.positions[pos.symbol] = pos
        trade = Trade(symbol=pos.symbol, entry_time=df.index[i], exit_time=None, entry_price=executed_price,
                      exit_price=None, size=size, pnl=0.0, fees=self.config['commission_per_trade'])
        self.trades.append(trade)
        self.logger.info("Entered position: price=%.4f size=%d stop=%.4f cash=%.2f", executed_price, size, stop_price, self.cash)
        return pos

    def exit_position(self, df: pd.DataFrame, i: int, pos: Position, reason: str = "exit"):
        exit_price_raw = df['close'].iat[i]
        executed_price = self.apply_slippage_and_commission(exit_price_raw, side="sell")
        proceeds = executed_price * pos.size - self.config['commission_per_trade']
        pnl = proceeds - (pos.entry_price * pos.size) - pos.entry_price * 0  # entry fees already accounted
        self.cash += proceeds
        # Close trade record
        # find last trade for this position with exit_time None
        for t in reversed(self.trades):
            if t.symbol == pos.symbol and t.exit_time is None and t.entry_price == pos.entry_price and t.size == pos.size:
                t.exit_time = df.index[i]
                t.exit_price = executed_price
                t.pnl = pnl
                t.fees += self.config['commission_per_trade']
                break
        self.logger.info("Exited position at %.4f size=%d pnl=%.2f reason=%s cash=%.2f", executed_price, pos.size, pnl, reason, self.cash)
        del self.positions[pos.symbol]

    def run_backtest(self, df: pd.DataFrame) -> Dict:
        # Expect df with columns: open, high, low, close, volume
        df = df.copy()
        df = df.dropna(subset=['open', 'high', 'low', 'close']).reset_index(drop=False)
        # ensure index is datetime in index column 0
        if not isinstance(df.columns[0], str) or df.columns[0].lower() not in ['index', 'date', 'timestamp']:
            # move original index to 'index' column to track times
            df.rename(columns={df.columns[0]: 'index'}, inplace=True)
        df['index'] = pd.to_datetime(df['index'])
        df.set_index('index', inplace=True)

        # compute indicators
        df['ema_short'] = compute_ema(df['close'], self.config['ema_short'])
        df['ema_long'] = compute_ema(df['close'], self.config['ema_long'])
        df['atr'] = compute_atr(df, self.config['atr_period'])
        signals = self.generate_signals(df)

        # iterate bars
        n = len(df)
        for i in range(n):
            current_time = df.index[i]
            close = df['close'].iat[i]
            atr = df['atr'].iat[i]
            signal = signals.iat[i]

            # update trailing stops for existing positions first
            for pos in list(self.positions.values()):
                old_stop = pos.stop_price
                pos.stop_price = self.update_trailing_stop(pos, close, atr)
                if pos.stop_price > old_stop:
                    self.logger.debug("Trailing stop updated from %.4f to %.4f", old_stop, pos.stop_price)

            # check existing positions for stop hit
            for pos in list(self.positions.values()):
                # exit if price <= stop
                if df['low'].iat[i] <= pos.stop_price:
                    # execute exit at stop price (simulate market fill)
                    executed_price = pos.stop_price
                    executed_price = self.apply_slippage_and_commission(executed_price, side="sell")
                    proceeds = executed_price * pos.size - self.config['commission_per_trade']
                    pnl = proceeds - (pos.entry_price * pos.size)
                    self.cash += proceeds
                    # update trade record
                    for t in reversed(self.trades):
                        if t.symbol == pos.symbol and t.exit_time is None and t.entry_price == pos.entry_price and t.size == pos.size:
                            t.exit_time = df.index[i]
                            t.exit_price = executed_price
                            t.pnl = pnl
                            t.fees += self.config['commission_per_trade']
                            break
                    self.logger.info("Stop hit: Exited at %.4f size=%d pnl=%.2f", executed_price, pos.size, pnl)
                    del self.positions[pos.symbol]

            # decide new entries only if we have capacity
            if signal == 1 and len(self.positions) < self.config['max_positions']:
                # risk manager check
                if self.risk_manager_before_entry(df, i, "SYNTH"):
                    size, estimated_stop = self.calculate_position_size(close, atr, self.cash + sum([p.entry_price * p.size for p in self.positions.values()]))
                    if size > 0:
                        # Improved entry rule: confirm trend (EMA short > EMA long)
                        if df['ema_short'].iat[i] > df['ema_long'].iat[i]:
                            # apply limit to positions per symbol/pyramiding if already present
                            if "SYNTH" not in self.positions:
                                self.enter_position(df, i, size, close, estimated_stop)
                            else:
                                pos = self.positions["SYNTH"]
                                if pos.additions < self.config['pyramiding']:
                                    # allow pyramiding if price moved favorably by ATR
                                    if close <= pos.entry_price - (atr * 0.5):  # buy more when price pullback ~0.5 ATR
                                        add_size, add_stop = self.calculate_position_size(close, atr, self.cash + pos.entry_price * pos.size)
                                        add_size = min(add_size, size)
                                        if add_size > 0:
                                            executed_price = self.apply_slippage_and_commission(close, side="buy")
                                            cost = executed_price * add_size + self.config['commission_per_trade']
                                            if cost <= self.cash:
                                                self.cash -= cost
                                                pos.size += add_size
                                                pos.additions += 1
                                                # widen stop conservatively
                                                pos.stop_price = min(pos.stop_price, add_stop)
                                                self.trades.append(Trade(symbol=pos.symbol, entry_time=df.index[i], exit_time=None,
                                                                         entry_price=executed_price, exit_price=None, size=add_size, pnl=0.0, fees=self.config['commission_per_trade']))
                                                self.logger.info("Pyramided position: added size=%d at price=%.4f new size=%d stop=%.4f cash=%.2f",
                                                                 add_size, executed_price, pos.size, pos.stop_price, self.cash)

            # small risk management: if equity dips, reduce exposure by exiting half of positions proactively
            total_position_value = sum([p.entry_price * p.size for p in self.positions.values()])
            total_net = self.cash + total_position_value
            if total_net > self.max_equity:
                self.max_equity = total_net
            current_drawdown = (self.max_equity - total_net) / max(1.0, self.max_equity)
            if current_drawdown >= self.config['reduce_risk_after_drawdown']['trigger'] * 1.5:
                # exit half of positions to protect equity
                for pos in list(self.positions.values()):
                    half_size = pos.size // 2
                    if half_size >= 1:
                        executed_price = self.apply_slippage_and_commission(close, side="sell")
                        proceeds = executed_price * half_size - self.config['commission_per_trade']
                        pnl = proceeds - (pos.entry_price * half_size)
                        self.cash += proceeds
                        pos.size -= half_size
                        # record trade exit for portion
                        self.trades.append(Trade(symbol=pos.symbol, entry_time=pos.entry_index, exit_time=df.index[i],
                                                 entry_price=pos.entry_price, exit_price=executed_price, size=half_size, pnl=pnl, fees=self.config['commission_per_trade']))
                        self.logger.info("Proactive haircut: sold %d shares of %s at %.4f to reduce drawdown", half_size, pos.symbol, executed_price)
                        if pos.size == 0:
                            del self.positions[pos.symbol]

            # compute mark-to-market equity
            position_value = sum([p.entry_price * p.size for p in self.positions.values()])  # conservative: use entry price
            equity = self.cash + position_value
            self.equity_curve.append({'time': df.index[i], 'equity': equity})
            self.equity = equity
            if equity > self.max_equity:
                self.max_equity = equity
            self.drawdown = (self.max_equity - equity) / max(1.0, self.max_equity)

            # log periodic stats
            if i % 50 == 0 or i == n - 1:
                self.logger.info("Bar %d/%d time=%s equity=%.2f positions=%d drawdown=%.4f", i, n, df.index[i], equity, len(self.positions), self.drawdown)

            # stop trading entirely if catastrophic drawdown
            if self.drawdown >= self.config['max_drawdown_limit']:
                self.logger.warning("Stopping trading loop due to drawdown >= %.2f", self.config['max_drawdown_limit'])
                break

        # finalize - close remaining positions at last close
        if len(self.positions) > 0:
            last_idx = n - 1
            for pos in list(self.positions.values()):
                self.exit_position(df, last_idx, pos, reason="end_of_backtest")

        # Build equity curve series
        eq_df = pd.DataFrame(self.equity_curve).set_index('time')
        eq_df['equity'] = eq_df['equity'].astype(float)
        # smoothing optional
        window = self.config['performance'].get('equity_curve_smoothing_window', 1)
        if window > 1:
            eq_df['equity_smooth'] = eq_df['equity'].rolling(window=window, min_periods=1).mean()
        else:
            eq_df['equity_smooth'] = eq_df['equity']
        perf = self.analyze_backtest(eq_df['equity_smooth'])
        self.logger.info("Backtest complete. Performance: %s", perf)
        return {
            "equity_curve": eq_df,
            "trades": self.trades,
            "performance": perf,
            "final_cash": self.cash,
            "final_equity": self.equity
        }


# -------------------------
# Example usage function (kept for completeness, not executed on import)
# -------------------------
def example_run():
    # Example synthesised data if user wants to test
    dates = pd.date_range(start="2020-01-01", periods=500, freq='B')
    np.random.seed(42)
    price = np.cumprod(1 + np.random.normal(0.0002, 0.01, size=len(dates))) * 100.0
    df = pd.DataFrame(index=dates)
    df['open'] = price * (1 + np.random.normal(0, 0.001, size=len(price)))
    df['high'] = np.maximum(df['open'], price * (1 + np.abs(np.random.normal(0, 0.005, size=len(price)))))
    df['low'] = np.minimum(df['open'], price * (1 - np.abs(np.random.normal(0, 0.005, size=len(price)))))
    df['close'] = price
    df['volume'] = np.random.randint(100, 1000, size=len(price))

    user_config = {
        "initial_capital": 500_000,
        "risk_per_trade": 0.005,
        "logging": {"logfile": "example_trading.log", "level": "INFO"},
        "ema_short": 15,
        "ema_long": 60,
        "signal_persistence": 3
    }

    logger = setup_logger(user_config['logging'])
    ts = TradingSystem(config=user_config, logger=logger)
    results = ts.run_backtest(df)
    eq = results['equity_curve']
    print("Final equity:", results['final_equity'])
    print("Performance summary:", results['performance'])
    # Optionally save results
    # eq.to_csv("equity_curve.csv")
    return results


if __name__ == "__main__":
    # Run example if executed directly
    example_run()