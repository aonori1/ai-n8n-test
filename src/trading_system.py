import os
import json
import logging
from logging.handlers import RotatingFileHandler
from dataclasses import dataclass, field
from typing import Dict, Optional
import pandas as pd
import numpy as np
import math
import datetime as dt

# -------------------------
# Configuration handling
# -------------------------
DEFAULT_CONFIG = {
    "initial_capital": 100000.0,
    "risk_per_trade": 0.01,                # fraction of capital risked per trade
    "max_portfolio_risk": 0.1,             # max fraction of capital at risk across all trades
    "max_position_percent": 0.25,          # max fraction of capital in any single position
    "commission_per_trade": 1.0,           # flat commission per trade
    "slippage_percent": 0.0005,            # slippage as fraction of price
    "atr_period": 14,
    "ema_fast": 12,
    "ema_slow": 26,
    "rsi_period": 14,
    "rsi_oversold": 30,
    "rsi_overbought": 70,
    "volatility_scale_target": 0.01,       # target volatility for position sizing (e.g., 1% ATR)
    "min_trade_dollars": 100.0,
    "max_concurrent_positions": 5,
    "max_drawdown_stop": 0.25,             # if drawdown from peak exceeds this, stop trading
    "trailing_atr_multiplier": 2.0,        # trailing stop distance in ATR multiples
    "stop_loss_atr_multiplier": 2.5,       # initial stop-loss distance in ATR multiples
    "risk_scaling_volatility_window": 20,  # window to compute realized volatility for scaling risk
    "logging": {
        "level": "INFO",
        "file": "trading_system.log",
        "max_bytes": 10_000_000,
        "backup_count": 3
    }
}

def load_config(path: Optional[str] = None) -> Dict:
    if path and os.path.exists(path):
        with open(path, 'r') as f:
            cfg = json.load(f)
        # merge defaults for missing keys
        merged = DEFAULT_CONFIG.copy()
        merged.update(cfg)
        return merged
    return DEFAULT_CONFIG.copy()

# -------------------------
# Logging (preserve)
# -------------------------
def setup_logger(config: Dict) -> logging.Logger:
    log_cfg = config.get("logging", {})
    logger = logging.getLogger("TradingSystem")
    if logger.handlers:
        return logger  # already configured
    level = getattr(logging, log_cfg.get("level", "INFO").upper(), logging.INFO)
    logger.setLevel(level)
    fh = RotatingFileHandler(log_cfg.get("file", "trading_system.log"),
                             maxBytes=log_cfg.get("max_bytes", 10_000_000),
                             backupCount=log_cfg.get("backup_count", 3))
    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    return logger

# -------------------------
# Utilities / Indicators
# -------------------------
def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def rsi(series: pd.Series, period: int) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.ewm(com=period - 1, adjust=False).mean()
    ma_down = down.ewm(com=period - 1, adjust=False).mean()
    rs = ma_up / (ma_down + 1e-12)
    return 100 - (100 / (1 + rs))

def atr(df: pd.DataFrame, period: int) -> pd.Series:
    # df must have columns: High, Low, Close
    high_low = df['High'] - df['Low']
    high_close = (df['High'] - df['Close'].shift()).abs()
    low_close = (df['Low'] - df['Close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(window=period, min_periods=1).mean()

def realized_volatility(series: pd.Series, window: int) -> float:
    # returns annualized volatility estimate based on log returns
    logret = np.log(series / series.shift(1)).dropna()
    if len(logret) < 2:
        return np.nan
    return logret.rolling(window=window).std().iloc[-1] * math.sqrt(252)

# -------------------------
# Data structures
# -------------------------
@dataclass
class Position:
    symbol: str
    entry_price: float
    size: int  # number of shares/contracts
    direction: int  # 1 for long, -1 for short (we will focus on long-only by default)
    entry_time: pd.Timestamp
    stop_price: float
    trailing_stop: Optional[float] = None
    atr_at_entry: Optional[float] = None

    def market_value(self, price: float) -> float:
        return self.size * price * self.direction

@dataclass
class Portfolio:
    cash: float
    positions: Dict[str, Position] = field(default_factory=dict)
    equity_curve: list = field(default_factory=list)
    realized_pnl: float = 0.0
    peak_equity: float = 0.0
    stopped_out_permanently: bool = False  # stop trading if drawdown breached
    logger: Optional[logging.Logger] = None

    def total_equity(self, price_map: Dict[str, float]) -> float:
        total = self.cash
        for sym, pos in self.positions.items():
            price = price_map.get(sym, pos.entry_price)
            total += pos.market_value(price)
        return total

    def update_peak(self, equity: float):
        if equity > self.peak_equity:
            self.peak_equity = equity

    def current_drawdown(self, equity: float) -> float:
        if self.peak_equity <= 0:
            return 0.0
        return max(0.0, (self.peak_equity - equity) / self.peak_equity)

# -------------------------
# Trading system (architecture preserved)
# -------------------------
class TradingSystem:
    def __init__(self, config_path: Optional[str] = None, logger: Optional[logging.Logger] = None):
        self.config = load_config(config_path)
        self.logger = logger or setup_logger(self.config)
        self.initial_capital = float(self.config["initial_capital"])
        self.portfolio = Portfolio(self.initial_capital, logger=self.logger)
        self.max_concurrent_positions = int(self.config["max_concurrent_positions"])
        self._metrics = {"trades": 0, "wins": 0, "losses": 0}
        self.logger.info("Initialized TradingSystem with capital: %.2f", self.initial_capital)

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        # Expect DataFrame with Index (datetime) and columns: Open, High, Low, Close, Volume
        df = df.copy()
        df['EMA_fast'] = ema(df['Close'], self.config['ema_fast'])
        df['EMA_slow'] = ema(df['Close'], self.config['ema_slow'])
        df['RSI'] = rsi(df['Close'], self.config['rsi_period'])
        df['ATR'] = atr(df[['High', 'Low', 'Close']], self.config['atr_period'])
        # signal: EMA fast crosses above EMA slow and RSI not overbought
        df['signal_raw'] = 0
        df.loc[(df['EMA_fast'] > df['EMA_slow']) & (df['EMA_fast'].shift() <= df['EMA_slow'].shift()), 'signal_raw'] = 1
        # filter with RSI (avoid entries near overbought)
        df['signal'] = df['signal_raw'] & (df['RSI'] < self.config['rsi_overbought'])
        # also support exit signals when EMA fast crosses below EMA slow or RSI overbought
        df['exit_signal'] = 0
        df.loc[(df['EMA_fast'] < df['EMA_slow']) & (df['EMA_fast'].shift() >= df['EMA_slow'].shift()), 'exit_signal'] = 1
        df.loc[df['RSI'] > self.config['rsi_overbought'], 'exit_signal'] = 1
        # Add realized volatility for scaling
        df['realized_vol'] = df['Close'].rolling(self.config['risk_scaling_volatility_window']).apply(
            lambda x: realized_volatility(x, min(len(x), self.config['risk_scaling_volatility_window'])))
        return df

    def position_size(self, capital: float, price: float, atr: float) -> int:
        """
        Compute size using volatility-targeted position sizing:
        size = (capital * risk_per_trade) / (stop_distance_in_dollars)
        where stop_distance_in_dollars = atr * stop_multiplier
        Also limit by max_position_percent and portfolio risk.
        """
        risk_per_trade = float(self.config['risk_per_trade'])
        stop_multiplier = float(self.config['stop_loss_atr_multiplier'])
        commission = float(self.config['commission_per_trade'])
        if atr <= 0 or price <= 0:
            return 0
        stop_distance = atr * stop_multiplier
        # dollars risked per share = stop_distance
        dollars_risked = capital * risk_per_trade
        raw_size = (dollars_risked / stop_distance) if stop_distance > 0 else 0
        # apply min trade dollars
        min_trade_dollars = float(self.config['min_trade_dollars'])
        max_position_value = capital * float(self.config['max_position_percent'])
        max_size_by_value = max(0, math.floor(max_position_value / price))
        size = math.floor(raw_size)
        size = max(0, size)
        size = min(size, max_size_by_value)
        # ensure trade is economically viable after commission
        if size * price < min_trade_dollars:
            return 0
        return int(size)

    def execute_entry(self, timestamp: pd.Timestamp, symbol: str, price: float, atr: float, df_row: pd.Series):
        if self.portfolio.stopped_out_permanently:
            self.logger.info("Trading stopped permanently due to drawdown. No new entries.")
            return
        if len(self.portfolio.positions) >= self.max_concurrent_positions:
            self.logger.debug("Max concurrent positions reached; skipping entry.")
            return
        equity = self.portfolio.total_equity({symbol: price})
        size = self.position_size(self.portfolio.cash, price, atr)
        if size <= 0:
            self.logger.debug("Calculated position size is 0; skipping entry at %s", timestamp)
            return
        # respect overall portfolio risk: estimate new total at-risk after entry
        estimated_risk_added = size * atr * self.config['stop_loss_atr_multiplier']
        total_risk_allowed = self.initial_capital * float(self.config['max_portfolio_risk'])
        existing_risk = sum([abs(p.size) * (p.atr_at_entry or atr) * self.config['stop_loss_atr_multiplier'] for p in self.portfolio.positions.values()])
        if existing_risk + estimated_risk_added > total_risk_allowed:
            self.logger.debug("Would exceed portfolio risk budget; skipping entry.")
            return
        # Calculate cash impact including slippage and commission
        slippage = price * float(self.config['slippage_percent'])
        exec_price = price + slippage  # assume long-only for now
        cost = exec_price * size + float(self.config['commission_per_trade'])
        if cost > self.portfolio.cash:
            # Possibly allow margin? For now, skip if insufficient cash
            self.logger.debug("Insufficient cash for entry: need %.2f have %.2f", cost, self.portfolio.cash)
            return
        # Stop price initial
        stop_price = exec_price - atr * float(self.config['stop_loss_atr_multiplier'])
        trailing_stop = exec_price - atr * float(self.config['trailing_atr_multiplier'])
        pos = Position(symbol=symbol, entry_price=exec_price, size=size, direction=1,
                       entry_time=timestamp, stop_price=stop_price, trailing_stop=trailing_stop, atr_at_entry=atr)
        self.portfolio.positions[symbol] = pos
        self.portfolio.cash -= cost
        self._metrics["trades"] += 1
        self.logger.info("ENTRY %s @ %.4f size=%d cost=%.2f stop=%.4f", symbol, exec_price, size, cost, stop_price)

    def execute_exit(self, timestamp: pd.Timestamp, symbol: str, price: float):
        pos = self.portfolio.positions.get(symbol)
        if pos is None:
            return
        slippage = price * float(self.config['slippage_percent'])
        exec_price = price - slippage  # assume exiting long
        proceeds = exec_price * pos.size - float(self.config['commission_per_trade'])
        pnl = proceeds - (pos.entry_price * pos.size)
        self.portfolio.cash += proceeds
        self.portfolio.realized_pnl += pnl
        del self.portfolio.positions[symbol]
        # update metrics
        if pnl >= 0:
            self._metrics["wins"] += 1
        else:
            self._metrics["losses"] += 1
        self.logger.info("EXIT  %s @ %.4f size=%d pnl=%.2f", symbol, exec_price, pos.size, pnl)

    def check_trailing_and_stop(self, timestamp: pd.Timestamp, symbol: str, price: float, atr: float):
        pos = self.portfolio.positions.get(symbol)
        if pos is None:
            return
        # update trailing stop if price moved favorably
        new_trail = price - atr * float(self.config['trailing_atr_multiplier'])
        if new_trail > (pos.trailing_stop or -np.inf):
            pos.trailing_stop = new_trail
            self.logger.debug("Updated trailing stop for %s to %.4f", symbol, pos.trailing_stop)
        # Evaluate stop-loss
        if price <= pos.stop_price or price <= pos.trailing_stop:
            self.logger.info("Stop triggered for %s at price %.4f (stop_price=%.4f trailing_stop=%.4f)", symbol, price, pos.stop_price, pos.trailing_stop)
            self.execute_exit(timestamp, symbol, price)

    def run_backtest(self, df: pd.DataFrame, symbol: str = "SYMBOL"):
        df_signals = self.generate_signals(df)
        self.logger.info("Starting backtest for %s with %d bars", symbol, len(df_signals))
        equity_history = []
        last_equity = self.initial_capital
        for idx, row in df_signals.iterrows():
            price = float(row['Close'])
            atr_val = float(row['ATR']) if not np.isnan(row['ATR']) else 0.0
            # Exit on signal first (reduces drawdown)
            if row.get('exit_signal', 0) and symbol in self.portfolio.positions:
                self.execute_exit(idx, symbol, price)
            # Check trailing stops & ordinary stops every bar
            if symbol in self.portfolio.positions:
                self.check_trailing_and_stop(idx, symbol, price, atr_val)
            # Entry signal
            if row.get('signal', 0):
                self.execute_entry(idx, symbol, price, atr_val, row)
            # Update equity
            price_map = {symbol: price}
            equity = self.portfolio.total_equity(price_map)
            self.portfolio.update_peak(equity)
            drawdown = self.portfolio.current_drawdown(equity)
            equity_history.append({"time": idx, "equity": equity, "drawdown": drawdown})
            # stop trading if drawdown exceeds configured threshold
            if drawdown >= float(self.config['max_drawdown_stop']):
                self.portfolio.stopped_out_permanently = True
                self.logger.warning("Max drawdown threshold reached (%.2f%%). Stopping trading.", drawdown * 100)
            last_equity = equity
        # finalize
        self.portfolio.equity_curve = equity_history
        self.logger.info("Backtest completed. Final equity: %.2f  Realized PnL: %.2f  Trades: %d Wins:%d Losses:%d",
                         last_equity, self.portfolio.realized_pnl, self._metrics["trades"], self._metrics["wins"], self._metrics["losses"])
        return pd.DataFrame(equity_history)

# -------------------------
# Example CLI / usage (keeps architecture)
# -------------------------
def load_price_data(path: str) -> pd.DataFrame:
    # Expect CSV with columns: Date, Open, High, Low, Close, Volume
    df = pd.read_csv(path, parse_dates=['Date'])
    df = df.set_index('Date').sort_index()
    # ensure required columns exist
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        if col not in df.columns:
            raise ValueError(f"Price data missing required column: {col}")
    return df

def main(data_path: str, config_path: Optional[str] = None):
    config = load_config(config_path)
    logger = setup_logger(config)
    ts = TradingSystem(config_path=config_path, logger=logger)
    df = load_price_data(data_path)
    symbol = os.path.splitext(os.path.basename(data_path))[0]
    equity_df = ts.run_backtest(df, symbol=symbol)
    # save equity curve for analysis
    out_file = f"{symbol}_equity_curve.csv"
    equity_df.to_csv(out_file, index=False)
    logger.info("Equity curve saved to %s", out_file)
    return equity_df

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run backtest for TradingSystem")
    parser.add_argument("--data", required=True, help="CSV file with OHLCV data, Date column required")
    parser.add_argument("--config", required=False, help="Optional JSON config file path")
    args = parser.parse_args()
    main(args.data, args.config)