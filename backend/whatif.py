"""What-if backtest (long-only) for OTP signals.

This module simulates a simple, explainable strategy:
  - BUY  : enter long with (almost) all available cash
  - SELL : exit to cash
  - HOLD : do nothing

It produces a daily equity curve + drawdown and some summary metrics.

Assumptions (kept explicit on purpose):
  - Execution price is the same day's *close* (no look-ahead inside the day).
    If you prefer "next day open" execution, we can switch it in one place.
  - Long-only, no leverage, no short.
  - Fees are applied on notional (buy and sell).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class WhatIfParams:
    initial_capital: float = 1_000_000.0
    fee_rate: float = 0.001  # 0.1% per trade side
    slippage: float = 0.0    # optional, e.g. 0.0005
    buy_threshold: float = 0.55
    sell_threshold: float = 0.45


def signals_from_probability(
    df: pd.DataFrame,
    buy_threshold: float,
    sell_threshold: float,
    prob_col: str = "p_up_next_5d",
) -> pd.Series:
    """Convert probability to BUY/SELL/HOLD.

    - BUY  if p >= buy_threshold
    - SELL if p <= sell_threshold
    - HOLD otherwise
    """
    p = df[prob_col].astype(float)
    out = np.where(p >= buy_threshold, "BUY", np.where(p <= sell_threshold, "SELL", "HOLD"))
    return pd.Series(out, index=df.index, name="signal")


def _compute_drawdown(equity: pd.Series) -> pd.Series:
    peak = equity.cummax()
    dd = equity / peak - 1.0
    return dd


def simulate_long_only(
    df: pd.DataFrame,
    params: WhatIfParams,
    date_col: str = "date",
    price_col: str = "close",
    signal_col: str = "signal",
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Run the backtest and return (daily_df, summary_metrics)."""

    if df.empty:
        raise ValueError("Input dataframe is empty")

    work = df.copy()
    work[date_col] = pd.to_datetime(work[date_col])
    work.sort_values(date_col, inplace=True)
    work.reset_index(drop=True, inplace=True)

    price = work[price_col].astype(float)
    signal = work[signal_col].fillna("HOLD").astype(str)

    cash = float(params.initial_capital)
    shares = 0.0

    # Track trade-level returns for win_rate etc.
    in_trade = False
    entry_equity = None  # equity at entry after fees
    trade_returns: List[float] = []
    trades = 0

    rows = []

    for i in range(len(work)):
        dt = work.at[i, date_col]
        px = float(price.iat[i])
        sig = signal.iat[i].upper()

        # Execute at close with optional slippage
        buy_px = px * (1.0 + params.slippage)
        sell_px = px * (1.0 - params.slippage)

        if sig == "BUY" and shares == 0.0:
            # Invest all cash (minus fee)
            notional = cash
            fee = notional * params.fee_rate
            cash_after_fee = cash - fee
            if cash_after_fee > 0:
                shares = cash_after_fee / buy_px
                cash = 0.0
                trades += 1
                in_trade = True
                # equity right after entry (should equal shares*px, but keep explicit)
                entry_equity = shares * px

        elif sig == "SELL" and shares > 0.0:
            notional = shares * sell_px
            fee = notional * params.fee_rate
            cash = notional - fee
            shares = 0.0
            if in_trade and entry_equity is not None:
                exit_equity = cash
                trade_returns.append(exit_equity / entry_equity - 1.0)
            in_trade = False
            entry_equity = None

        equity = cash + shares * px

        rows.append(
            {
                "date": dt.strftime("%Y-%m-%d"),
                "price": px,
                "signal": sig,
                "cash": cash,
                "shares": shares,
                "equity": equity,
            }
        )

    daily = pd.DataFrame(rows)
    daily["drawdown"] = _compute_drawdown(daily["equity"]).astype(float)

    # Buy & Hold benchmark (invest all on first day close, hold through end; no fees by default)
    bh_equity = params.initial_capital * (daily["price"] / float(daily["price"].iloc[0]))
    daily["bh_equity"] = bh_equity

    # Summary metrics
    start = float(params.initial_capital)
    end = float(daily["equity"].iloc[-1])
    bh_end = float(daily["bh_equity"].iloc[-1])

    total_return_pct = (end / start - 1.0) * 100.0
    bh_return_pct = (bh_end / start - 1.0) * 100.0
    max_drawdown_pct = float(daily["drawdown"].min() * 100.0)

    if trade_returns:
        win_rate = float(np.mean(np.array(trade_returns) > 0) * 100.0)
        avg_trade_return_pct = float(np.mean(trade_returns) * 100.0)
        median_trade_return_pct = float(np.median(trade_returns) * 100.0)
    else:
        win_rate = 0.0
        avg_trade_return_pct = 0.0
        median_trade_return_pct = 0.0

    summary = {
        "start_capital": start,
        "end_capital": end,
        "total_return_pct": total_return_pct,
        "bh_return_pct": bh_return_pct,
        "max_drawdown_pct": max_drawdown_pct,
        "trades": float(trades),
        "round_trips": float(len(trade_returns)),
        "win_rate_pct": win_rate,
        "avg_trade_return_pct": avg_trade_return_pct,
        "median_trade_return_pct": median_trade_return_pct,
    }

    return daily, summary


def load_otp_for_whatif(
    price_csv: str,
    signals_csv: str,
    buy_threshold: float,
    sell_threshold: float,
) -> pd.DataFrame:
    """Load price + signal probability and return merged df with BUY/SELL/HOLD."""
    def _to_hu_trading_date(series: pd.Series) -> pd.Series:
        """Parse timestamps and align to HU calendar dates.

        The historical price file contains timezone-aware timestamps (e.g. +01:00).
        Converting those to UTC shifts the calendar date (often to 23:00 previous day),
        which breaks merges with signal data that uses plain dates.

        We therefore convert to Europe/Budapest first (if tz-aware), then normalize to
        midnight, and finally drop tz.
        """
        # Always parse into UTC to avoid "mixed time zones" object dtype.
        dt = pd.to_datetime(series, utc=True, errors="coerce")
        # Convert back to HU time, then drop timezone.
        dt = dt.dt.tz_convert("Europe/Budapest").dt.normalize().dt.tz_localize(None)
        return dt

    price_df = pd.read_csv(price_csv)
    price_df = price_df.rename(columns={"Date": "date", "Close": "close"})
    price_df["date"] = _to_hu_trading_date(price_df["date"])

    sig_df = pd.read_csv(signals_csv)
    sig_df = sig_df.rename(columns={"Date": "date"})
    sig_df["date"] = _to_hu_trading_date(sig_df["date"])

    merged = price_df[["date", "close"]].merge(
        sig_df[["date", "p_up_next_5d"]], on="date", how="left"
    )
    merged["p_up_next_5d"] = merged["p_up_next_5d"].astype(float)
    merged["signal"] = signals_from_probability(
        merged, buy_threshold=buy_threshold, sell_threshold=sell_threshold
    )
    return merged


if __name__ == "__main__":
    # Manual run: generate a CSV to inspect.
    df = load_otp_for_whatif(
        price_csv="backend/data/otp_ohlcv_history.csv",
        signals_csv="backend/model/latest_signals.csv",
        buy_threshold=0.55,
        sell_threshold=0.45,
    )
    daily, summary = simulate_long_only(df, WhatIfParams())

    out_path = "backend/model/whatif_equity.csv"
    daily.to_csv(out_path, index=False)
    print("✅ what-if daily exported:", out_path)
    print("Summary:")
    for k, v in summary.items():
        print(f"  {k}: {v}")
