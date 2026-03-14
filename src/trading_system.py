import logging
import math
import json
from copy import deepcopy
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

# Configuration (preserved and extendable)
CONFIG = {
    "target_volatility": 0.10,          # annual target volatility for the portfolio (10%)
    "vol_lookback_days": 20,            # lookback to estimate volatility
    "atr_lookback": 14,                 # ATR lookback for stop calculations
    "risk_per_trade": 0.01,             # fraction of equity risked per trade
    "max_leverage": 3.0,                # maximum gross leverage
    "commission_per_trade": 0.0005,     # commission as fraction of traded notional
    "slippage_per_trade": 0.0005,       # slippage fraction
    "signal_smoothing_span": 5,         # EMA span for smoothing raw signals
    "entry_zscore": 1.0,                # z-score threshold to enter
    "exit_zscore": 0.5,                 # z-score threshold to exit / reduce
    "min_trade_value": 100.0,           # minimum notional to open a position
    "portfolio_max_drawdown": 0.20,     # stop trading if portfolio drawdown > 20%
    "cooldown_days_after_dd": 5,        # cooldown after large portfolio drawdown
    "max_position_concentration": 0.30, # max fraction of capital in single position
    "risk_scaling_by_volatility": True,  # scale position sizes by individual vol
    "use_trailing_stop": True,
    "trailing_stop_atr_mult": 3.0,       # ATR multiple for trailing stop
    "fixed_stop_atr_mult": 4.0,         # ATR multiple for initial stop
    "logging_config": {
        "level": logging.INFO,
        "fmt": "%(asctime)s %(levelname)s %(message)s"
    }
}

# Set up logger (preserved)
logging.basicConfig(level=CONFIG["logging_config"]["level"], format=CONFIG["logging_config"]["fmt"])
logger = logging.getLogger("TradingSystem")


class TradingSystem:
    """
    TradingSystem: signal generation + risk management + backtesting logic.
    Architecture:
      - Input: dictionary of pandas DataFrames keyed by asset symbol. Each DataFrame must have columns:
               ['open', 'high', 'low', 'close', 'volume'] and a DateTime index.
      - Methods:
         generate_raw_signals: user-defined signal logic (kept simple for architecture consistency).
         smooth_signals: reduce turnover and false signals.
         compute_volatility_and_atr: helper to compute vol and ATR.
         size_positions: volatility-targeted sizing with risk per trade and concentration limits.
         apply_stops_and_risk: attach stop levels to positions.
         backtest: simulates the portfolio over time, logs trades, computes metrics.
    """

    def __init__(self, price_data: dict, config: dict = None):
        self.config = deepcopy(CONFIG)
        if config:
            self.config.update(config)
        self.price_data = price_data  # dict of DataFrames
        # Check data
        for sym, df in self.price_data.items():
            if not isinstance(df, pd.DataFrame):
                raise ValueError(f"Price data for {sym} must be a pandas DataFrame")
            required = {"open", "high", "low", "close"}
            if not required.issubset(set(df.columns)):
                raise ValueError(f"Price data for {sym} must contain columns {required}")
        # Internal state
        self.signals = {}       # raw signals per asset (DataFrame)
        self.smoothed = {}      # smoothed signals per asset (Series)
        self.volatility = {}    # estimated vol per asset (Series)
        self.atr = {}           # ATR per asset (Series)
        # Logs and results
        self.trade_log = []     # list of trade dicts
        self.daily_log = []     # daily portfolio snapshots

        logger.info("Trading system initialized with %d assets", len(self.price_data))

    # ---------- Indicator computations ----------
    @staticmethod
    def compute_atr(df: pd.DataFrame, lookback: int) -> pd.Series:
        high = df["high"]
        low = df["low"]
        close = df["close"]
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(lookback, min_periods=1).mean()
        return atr

    @staticmethod
    def compute_rolling_vol(df: pd.DataFrame, lookback: int) -> pd.Series:
        # Use log returns to estimate volatility
        logret = np.log(df["close"] / df["close"].shift(1)).fillna(0.0)
        # Annualize assuming 252 trading days
        vol = logret.rolling(lookback, min_periods=1).std() * math.sqrt(252)
        return vol

    # ---------- Signal generation & smoothing ----------
    def generate_raw_signals(self):
        """
        Place-holder signal generator. For each asset we compute a momentum z-score over vol_lookback_days.
        This keeps architecture consistent (user can replace this), and provides a reasonable baseline.
        """
        lookback = self.config["vol_lookback_days"]
        for sym, df in self.price_data.items():
            close = df["close"]
            returns = close.pct_change().fillna(0.0)
            # Momentum: cumulative return over lookback
            cumret = (1 + returns).rolling(lookback, min_periods=1).apply(lambda x: np.prod(x) - 1.0)
            # Standardize the momentum to produce a z-score
            mu = cumret.rolling(lookback, min_periods=1).mean()
            sigma = cumret.rolling(lookback, min_periods=1).std().replace(0, 1e-9)
            z = (cumret - mu) / sigma
            self.signals[sym] = z.rename("raw_signal")
            logger.debug("Generated raw signals for %s", sym)

    def smooth_signals(self):
        """
        Smooth raw signals with EMA to reduce turnover and noise.
        """
        span = self.config["signal_smoothing_span"]
        for sym, raw in self.signals.items():
            sm = raw.ewm(span=span, adjust=False).mean()
            self.smoothed[sym] = sm.rename("smoothed_signal")
            logger.debug("Smoothed signals for %s with span %s", sym, span)

    # ---------- Position sizing and risk management ----------
    def compute_volatility_and_atr(self):
        """
        Compute per-asset volatility and ATR series for use in sizing & stops.
        """
        vol_lookback = self.config["vol_lookback_days"]
        atr_lookback = self.config["atr_lookback"]
        for sym, df in self.price_data.items():
            self.volatility[sym] = self.compute_rolling_vol(df, vol_lookback).rename("vol")
            self.atr[sym] = self.compute_atr(df, atr_lookback).rename("atr")
            logger.debug("Computed vol and atr for %s", sym)

    def size_positions(self, date, portfolio_value, prices):
        """
        Determine position sizes at a given date given smoothed signals and volatility targeting.

        - Uses risk_per_trade to limit loss per trade.
        - Scales position size by volatility: larger positions in lower-vol assets if risk_scaling_by_volatility enabled.
        - Caps per-position exposure by max_position_concentration and portfolio leverage by max_leverage.
        - Returns dict: symbol -> target_shares (signed).
        """
        target_vol = self.config["target_volatility"]
        risk_per_trade = self.config["risk_per_trade"]
        max_conc = self.config["max_position_concentration"]
        min_trade_value = self.config["min_trade_value"]
        max_leverage = self.config["max_leverage"]

        # Collect candidate signals and vols at date
        assets = []
        for sym, sm in self.smoothed.items():
            if date not in sm.index:
                continue
            signal = sm.at[date]
            vol_series = self.volatility.get(sym)
            if vol_series is None or date not in vol_series.index:
                continue
            vol = vol_series.at[date]
            price = prices.get(sym)
            if price is None or price <= 0 or not np.isfinite(price):
                continue
            assets.append((sym, float(signal), float(vol), float(price)))

        # Convert into DataFrame
        if not assets:
            return {}

        df = pd.DataFrame(assets, columns=["sym", "signal", "vol", "price"]).set_index("sym")
        # Only consider signals above entry threshold magnitude
        entry_z = self.config["entry_zscore"]
        df["dir"] = np.sign(df["signal"])
        df["strength"] = df["signal"].abs()
        df = df[df["strength"] >= entry_z]
        if df.empty:
            return {}

        # Volatility-adjusted notional per asset: target_vol * portfolio_value * (signal_strength / sum_strength)
        sum_strength = df["strength"].sum()
        if sum_strength <= 0:
            return {}

        # Notional allocation before risk-per-trade & concentration cap
        notional_alloc = {}
        for sym, row in df.iterrows():
            weight = row["strength"] / sum_strength
            notional = portfolio_value * target_vol * weight / max(row["vol"], 1e-6)
            notional_alloc[sym] = notional

        # Convert notional to shares using price and apply risk per trade and concentration limits
        target_shares = {}
        for sym in df.index:
            price = df.at[sym, "price"]
            vol = df.at[sym, "vol"]
            dir_sign = int(df.at[sym, "dir"])
            # Risk per trade logic: compute stop distance in price units conservatively using atr*fixed_stop_atr_mult
            atr_series = self.atr.get(sym)
            stop_dist = None
            if atr_series is not None and date in atr_series.index:
                stop_dist = atr_series.at[date] * self.config["fixed_stop_atr_mult"]
            # fallback to percent (e.g., 2% of price)
            if not stop_dist or stop_dist <= 0:
                stop_dist = price * 0.02
            # Max notional based on risk_per_trade: notional <= (risk_per_trade * portfolio_value) / (stop_dist / price)
            notional_based_on_risk = (risk_per_trade * portfolio_value) / (stop_dist / price)
            notional = min(notional_alloc[sym], notional_based_on_risk, max_conc * portfolio_value)
            # Ensure minimum trade size
            if notional < min_trade_value:
                continue
            shares = math.floor(notional / price)
            if shares == 0:
                continue
            target_shares[sym] = dir_sign * int(shares)

        # Cap total leverage: ensure sum(abs(notional))/portfolio_value <= max_leverage
        total_notional = sum(abs(shares * self.price_data[sym].loc[date, "close"]) for sym, shares in target_shares.items())
        if total_notional / portfolio_value > max_leverage:
            scale = (max_leverage * portfolio_value) / total_notional
            for sym in list(target_shares.keys()):
                adj = int(math.floor(abs(target_shares[sym]) * scale))
                target_shares[sym] = np.sign(target_shares[sym]) * adj
                if target_shares[sym] == 0:
                    del target_shares[sym]

        logger.debug("Sized positions for %s: %s", date, target_shares)
        return target_shares

    # ---------- Backtest / execution ----------
    def backtest(self, start_date=None, end_date=None, initial_capital=1_000_000):
        """
        Run a simple backtest over the intersection of asset dates.
        Execution model:
          - Daily bar simulation (open->close)
          - Orders executed at next day's open (slippage+commission applied)
          - Stop-losses executed intraday conservatively at worse of (stop price, low) simplified to close price adjustment
        """
        # Prepare combined date index: intersection of all assets' indices
        common_index = None
        for df in self.price_data.values():
            if common_index is None:
                common_index = df.index
            else:
                common_index = common_index.intersection(df.index)
        if common_index is None or common_index.empty:
            raise ValueError("No common dates across assets")

        if start_date:
            common_index = common_index[common_index >= pd.to_datetime(start_date)]
        if end_date:
            common_index = common_index[common_index <= pd.to_datetime(end_date)]
        if common_index.empty:
            raise ValueError("No dates in the requested range after applying start/end")

        # Pre-compute indicators
        self.generate_raw_signals()
        self.smooth_signals()
        self.compute_volatility_and_atr()

        # Backtest state
        cash = float(initial_capital)
        positions = {}  # sym -> shares (signed)
        portfolio_value = float(initial_capital)
        peak_value = portfolio_value
        last_dd_date = None
        cooldown_until = None

        # Run through each date
        for current_date in common_index:
            # Skip during cooldown
            if cooldown_until is not None and current_date <= cooldown_until:
                self.daily_log.append({
                    "date": current_date,
                    "cash": cash,
                    "positions": deepcopy(positions),
                    "portfolio_value": portfolio_value
                })
                continue

            # Build price lookup at current_date (use close price for marking)
            prices = {}
            for sym, df in self.price_data.items():
                if current_date in df.index:
                    prices[sym] = float(df.loc[current_date, "close"])
            # Compute mark-to-market
            current_notional = sum(positions.get(sym, 0) * prices.get(sym, 0.0) for sym in prices)
            portfolio_value = cash + current_notional

            # Check portfolio drawdown stop
            peak_value = max(peak_value, portfolio_value)
            drawdown = 1.0 - (portfolio_value / peak_value) if peak_value > 0 else 0.0
            if drawdown >= self.config["portfolio_max_drawdown"]:
                # Enter cooldown period
                cooldown_days = self.config["cooldown_days_after_dd"]
                cooldown_until = current_date + timedelta(days=cooldown_days)
                logger.warning("Portfolio drawdown %.2f >= max_drawdown %.2f on %s. Cooling until %s",
                               drawdown, self.config["portfolio_max_drawdown"], current_date, cooldown_until)
                self.daily_log.append({
                    "date": current_date,
                    "cash": cash,
                    "positions": deepcopy(positions),
                    "portfolio_value": portfolio_value,
                    "drawdown": drawdown
                })
                continue

            # Determine target positions (sized on current prices)
            target_shares = self.size_positions(current_date, portfolio_value, prices)

            # Generate trades: close, reduce or open positions at next open (we simulate immediate execution at close with slippage)
            # For fairness, we execute trades at current close price with slippage and commission applied.
            executed_trades = []
            for sym in set(list(positions.keys()) + list(target_shares.keys())):
                current_shares = positions.get(sym, 0)
                target = target_shares.get(sym, 0)
                if current_shares == target:
                    continue
                # Execute
                price = prices.get(sym)
                if price is None:
                    # cannot execute if price not available
                    continue
                qty = target - current_shares
                # Slippage & commissions reduce cash
                trade_notional = qty * price
                slippage = abs(trade_notional) * self.config["slippage_per_trade"]
                commission = abs(trade_notional) * self.config["commission_per_trade"]
                cash -= trade_notional  # pay for buys, receive for sells
                cash -= slippage
                cash -= commission
                positions[sym] = target  # set new position
                executed_trades.append({
                    "date": current_date,
                    "symbol": sym,
                    "shares": int(qty),
                    "price": price,
                    "notional": trade_notional,
                    "slippage": slippage,
                    "commission": commission
                })
                # Log trade
                self.trade_log.append({
                    "date": current_date,
                    "symbol": sym,
                    "shares": int(qty),
                    "price": price,
                    "notional": trade_notional,
                    "slippage": slippage,
                    "commission": commission,
                    "reason": "rebalance"
                })

            # Update marks after execution
            current_notional = sum(positions.get(sym, 0) * prices.get(sym, 0.0) for sym in prices)
            portfolio_value = cash + current_notional
            peak_value = max(peak_value, portfolio_value)
            drawdown = 1.0 - (portfolio_value / peak_value) if peak_value > 0 else 0.0

            # Apply stop-loss/trailing stops conservatively at close price:
            # If a position has moved against us beyond the fixed stop (atr-based), we close it.
            stops_triggered = []
            for sym, shares in list(positions.items()):
                if shares == 0:
                    continue
                df = self.price_data[sym]
                # we require that we have today's high/low to check intraday movement
                if current_date not in df.index:
                    continue
                row = df.loc[current_date]
                close = float(row["close"])
                low = float(row.get("low", close))
                high = float(row.get("high", close))
                # Directional stop: initial stop based on ATR from that asset
                atr_series = self.atr.get(sym)
                if atr_series is not None and current_date in atr_series.index:
                    atr_val = atr_series.at[current_date]
                else:
                    atr_val = close * 0.02
                # For long positions, stop = entry_price - fixed_stop_atr_mult*atr
                # Since we do not track entry_price per position in this simplified simulator,
                # use current close as proxy for entry for new positions; this is conservative.
                # A better simulator would track per-lot entry prices.
                if shares > 0:
                    stop_price = close - self.config["fixed_stop_atr_mult"] * atr_val
                    trailing = close - self.config["trailing_stop_atr_mult"] * atr_val if self.config["use_trailing_stop"] else None
                    stop_triggered = low <= stop_price
                else:
                    stop_price = close + self.config["fixed_stop_atr_mult"] * atr_val
                    trailing = close + self.config["trailing_stop_atr_mult"] * atr_val if self.config["use_trailing_stop"] else None
                    stop_triggered = high >= stop_price
                if stop_triggered:
                    # Close the position fully at close (conservative)
                    qty = -positions[sym]
                    notional = qty * close
                    slippage = abs(notional) * self.config["slippage_per_trade"]
                    commission = abs(notional) * self.config["commission_per_trade"]
                    cash -= notional
                    cash -= slippage
                    cash -= commission
                    self.trade_log.append({
                        "date": current_date,
                        "symbol": sym,
                        "shares": int(qty),
                        "price": close,
                        "notional": notional,
                        "slippage": slippage,
                        "commission": commission,
                        "reason": "stop_loss"
                    })
                    positions[sym] = 0
                    stops_triggered.append(sym)

            # Remove zero positions to keep dict small
            positions = {s: q for s, q in positions.items() if q != 0}

            # Mark portfolio and record daily
            current_notional = sum(positions.get(sym, 0) * prices.get(sym, 0.0) for sym in prices)
            portfolio_value = cash + current_notional
            self.daily_log.append({
                "date": current_date,
                "cash": cash,
                "positions": deepcopy(positions),
                "portfolio_value": portfolio_value,
                "drawdown": drawdown
            })

        # End of backtest - compute metrics
        results = self.calculate_metrics(initial_capital)
        logger.info("Backtest complete. Final portfolio value: %.2f, Sharpe: %.3f, MaxDrawdown: %.3f",
                    results["final_value"], results["sharpe"], results["max_drawdown"])
        return results

    # ---------- Performance metrics ----------
    def calculate_metrics(self, initial_capital):
        """
        Compute daily returns, Sharpe, max drawdown, CAGR.
        """
        df_daily = pd.DataFrame(self.daily_log)
        if df_daily.empty:
            return {
                "final_value": initial_capital,
                "returns": pd.Series(dtype=float),
                "sharpe": 0.0,
                "max_drawdown": 0.0,
                "cagr": 0.0
            }

        df_daily = df_daily.set_index("date").sort_index()
        pv = df_daily["portfolio_value"].astype(float)
        returns = pv.pct_change().fillna(0.0)
        # Annualized Sharpe (assume 252 trading days)
        avg = returns.mean() * 252
        vol = returns.std() * math.sqrt(252)
        sharpe = (avg / vol) if vol > 0 else 0.0
        # Max drawdown
        running_max = pv.cummax()
        drawdowns = 1.0 - (pv / running_max)
        max_dd = drawdowns.max()
        # CAGR
        total_days = (pv.index[-1] - pv.index[0]).days or 1
        years = total_days / 365.25
        cagr = (pv.iloc[-1] / pv.iloc[0]) ** (1 / years) - 1 if years > 0 else 0.0

        results = {
            "final_value": float(pv.iloc[-1]),
            "returns": returns,
            "sharpe": float(sharpe),
            "max_drawdown": float(max_dd),
            "cagr": float(cagr),
            "daily": df_daily,
            "trade_log": pd.DataFrame(self.trade_log)
        }
        return results

    # ---------- Utility: save logs and configuration ----------
    def save_state(self, trade_log_path="trade_log.csv", daily_log_path="daily_log.csv", config_path="config.json"):
        try:
            pd.DataFrame(self.trade_log).to_csv(trade_log_path, index=False)
            pd.DataFrame(self.daily_log).to_csv(daily_log_path, index=False)
            with open(config_path, "w") as f:
                json.dump(self.config, f, indent=2, default=str)
            logger.info("Saved trade log, daily log, and configuration.")
        except Exception as e:
            logger.exception("Failed to save state: %s", e)


# ---------- Example usage (kept for completeness, remove or adapt for production) ----------
if __name__ == "__main__":
    # Minimal example to demonstrate API (users will replace with real data loader)
    symbols = ["AAA", "BBB"]
    dates = pd.date_range("2020-01-01", "2021-12-31", freq="B")
    np.random.seed(42)

    price_data = {}
    for sym in symbols:
        # Simulate random walk price series
        p = 100 + np.cumsum(np.random.normal(0, 1, len(dates)))
        high = p + np.random.uniform(0, 1.0, len(p))
        low = p - np.random.uniform(0, 1.0, len(p))
        openp = p + np.random.uniform(-0.5, 0.5, len(p))
        volume = np.random.randint(100, 1000, len(p))
        df = pd.DataFrame({
            "open": openp,
            "high": high,
            "low": low,
            "close": p,
            "volume": volume
        }, index=dates)
        price_data[sym] = df

    # Create system and run backtest
    ts = TradingSystem(price_data)
    results = ts.backtest(initial_capital=1_000_000)
    # Save logs and config
    ts.save_state()