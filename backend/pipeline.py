"""
Backend pipeline for OTP decision-support signals.

- load price history from CSV
- engineer features
- train a conservative model (RandomForest)
- produce daily signals with explanations
- export a stable CSV contract used by the API + UI

Design goals:
- Deterministic output
- One code-path for cron + manual recalc
- No trading / no portfolio logic
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import roc_auc_score



from config import INSTRUMENTS


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "model"


FORECAST_HORIZON_DAYS = 5

def load_price_history(symbol: str) -> pd.DataFrame:
    cfg = INSTRUMENTS[symbol]
    path = cfg["csv"]

    df = pd.read_csv(path)

    # --- dátum ---
    df["Date"] = (
        pd.to_datetime(df["Date"], errors="coerce")
        .dt.tz_localize(None)
        .dt.normalize()
    )

    # --- numerikus oszlopok KÉNYSZERÍTÉSE ---
    num_cols = ["Open", "High", "Low", "Close", "Volume"]
    for col in num_cols:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.replace(",", "", regex=False)
                .replace("None", None)
            )
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # --- takarítás ---
    df = df.dropna(subset=["Date", "Close"])
    df = df.sort_values("Date").reset_index(drop=True)

    return df


@dataclass(frozen=True)
class SignalConfig:
    buy_threshold: float = 0.60
    min_confidence: float = 0.55
    horizon_days: int = FORECAST_HORIZON_DAYS

    @property
    def sell_threshold(self) -> float:
        return 1.0 - self.buy_threshold


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))



def make_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["ret_1"] = out["Close"].pct_change(fill_method=None)
    out["ret_3"] = out["Close"].pct_change(3, fill_method=None)
    out["ret_5"] = out["Close"].pct_change(5, fill_method=None)

    out["sma_10"] = out["Close"].rolling(10).mean()
    out["sma_20"] = out["Close"].rolling(20).mean()
    out["price_vs_sma20"] = out["Close"] / out["sma_20"] - 1

    out["vol_10"] = out["ret_1"].rolling(10).std()
    out["vol_20"] = out["ret_1"].rolling(20).std()

    out["range_pct"] = (out["High"] - out["Low"]) / out["Close"]
    out["body_pct"] = (out["Close"] - out["Open"]) / out["Close"]

    out["log_vol"] = np.log1p(out["Volume"])
    out["vol_sma_20"] = out["log_vol"].rolling(20).mean()
    out["rel_vol_20"] = out["log_vol"] - out["vol_sma_20"]

    out["rsi_14"] = _rsi(out["Close"], 14)

    return out


def make_target(df: pd.DataFrame, horizon_days: int) -> pd.DataFrame:
    out = df.copy()
    out["future_close"] = out["Close"].shift(-horizon_days)
    out["target_up"] = (out["future_close"] > out["Close"]).astype(int)
    out["future_return"] = (out["future_close"] / out["Close"]) - 1
    return out


FEATURES: List[str] = [
    "ret_1",
    "ret_3",
    "ret_5",
    "price_vs_sma20",
    "vol_10",
    "vol_20",
    "range_pct",
    "body_pct",
    "rel_vol_20",
    "rsi_14",
]


def make_model() -> RandomForestClassifier:
    return RandomForestClassifier(
        # daily run-hoz elég gyors
        n_estimators=400,
        max_depth=6,
        min_samples_leaf=30,
        random_state=42,
        class_weight="balanced_subsample",
        n_jobs=-1,
    )


def _risk_level(vol_20: float, vol_20_quantiles: Tuple[float, float]) -> str:
    q33, q66 = vol_20_quantiles
    if np.isnan(vol_20):
        return "medium"
    if vol_20 <= q33:
        return "low"
    if vol_20 >= q66:
        return "high"
    return "medium"


def _explain(latest_row: pd.Series) -> List[str]:
    reasons: List[str] = []

    rsi = float(latest_row.get("rsi_14", np.nan))
    if not np.isnan(rsi) and rsi < 30:
        reasons.append("RSI alulértékelt tartományban")
    elif not np.isnan(rsi) and rsi > 70:
        reasons.append("RSI túlvett tartományban")

    ret_5 = float(latest_row.get("ret_5", np.nan))
    if not np.isnan(ret_5) and ret_5 > 0:
        reasons.append("Rövid távú momentum pozitív (5 nap)")
    elif not np.isnan(ret_5) and ret_5 < 0:
        reasons.append("Rövid távú momentum negatív (5 nap)")

    v10 = float(latest_row.get("vol_10", np.nan))
    v20 = float(latest_row.get("vol_20", np.nan))
    if not np.isnan(v10) and not np.isnan(v20):
        if v10 < v20:
            reasons.append("Volatilitás csökken (10 nap < 20 nap)")
        elif v10 > v20:
            reasons.append("Volatilitás nő (10 nap > 20 nap)")

    p_vs = float(latest_row.get("price_vs_sma20", np.nan))
    if not np.isnan(p_vs) and p_vs > 0:
        reasons.append("Árfolyam a 20 napos trend felett")
    elif not np.isnan(p_vs) and p_vs < 0:
        reasons.append("Árfolyam a 20 napos trend alatt")

    return reasons[:5]


def _expected_return_from_history(history: pd.DataFrame, proba: float, k: int = 200) -> float:
    h = history.dropna(subset=["p_up_next_5d", "future_return"]).copy()
    if len(h) < 50:
        return float("nan")

    h["dist"] = (h["p_up_next_5d"] - proba).abs()
    h = h.sort_values("dist").head(k)
    return float(h["future_return"].mean())


def generate_signals(symbol: str, config: SignalConfig = SignalConfig()) -> pd.DataFrame:
    raw = load_price_history(symbol)
    feat = make_features(raw)
    ds = make_target(feat, config.horizon_days)

    ds = ds.replace([np.inf, -np.inf], np.nan)
    ds_model = ds.dropna(subset=FEATURES + ["target_up"]).copy()

    # model = make_model()
    # model.fit(ds_model[FEATURES], ds_model["target_up"])
    # ds_model["p_up_next_5d"] = model.predict_proba(ds_model[FEATURES])[:, 1]

    # save_model_meta(
    #     symbol=symbol,
    #     model=model,
    #     cv_accuracy=cv_acc,
    #     roc_auc=roc_auc,
    #     feature_names=FEATURES,
    #     train_rows=len(X)
    # )

    model = make_model()

    X = ds_model[FEATURES]
    y = ds_model["target_up"]

    tscv = TimeSeriesSplit(n_splits=5)
    cv_acc = cross_val_score(model, X, y, cv=tscv, scoring="accuracy").mean()

    model.fit(X, y)

    y_proba = model.predict_proba(X)[:, 1]
    roc_auc = roc_auc_score(y, y_proba)

    ds_model["p_up_next_5d"] = y_proba

    save_model_meta(
        symbol=symbol,
        model=model,
        cv_accuracy=cv_acc,
        roc_auc=roc_auc,
        feature_names=FEATURES,
        train_rows=len(X)
    )

    vol_20_q = (
        float(ds_model["vol_20"].quantile(0.33)),
        float(ds_model["vol_20"].quantile(0.66)),
    )

    sell_threshold = config.sell_threshold

    def decide(row: pd.Series):
        p = float(row["p_up_next_5d"])
        buy_conf = p
        sell_conf = 1.0 - p
        best_conf = max(buy_conf, sell_conf)

        if best_conf < config.min_confidence:
            rec = "NO SIGNAL"
            conf = best_conf
        elif p >= config.buy_threshold:
            rec = "BUY"
            conf = buy_conf
        elif p <= sell_threshold:
            rec = "SELL"
            conf = sell_conf
        else:
            rec = "HOLD"
            conf = best_conf

        exp_ret = _expected_return_from_history(ds_model, p)
        risk = _risk_level(float(row.get("vol_20", np.nan)), vol_20_q)
        reasons = _explain(row)
        return rec, conf, exp_ret, risk, reasons

    decided = ds_model.apply(decide, axis=1, result_type="expand")
    decided.columns = ["recommendation", "confidence", "expected_return_5d", "risk_level", "explanation"]

    out = pd.concat([ds_model[["Date", "Close", "p_up_next_5d"]], decided], axis=1)
    out["explanation"] = out["explanation"].apply(lambda xs: json.dumps(xs, ensure_ascii=False))
    out = out.sort_values("Date").reset_index(drop=True)
    return out


def export_signals(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False)

def run_pipeline(
    symbol: str,
    buy_threshold: float = 0.60,
    min_confidence: float = 0.55
) -> Path:

    cfg = SignalConfig(buy_threshold=buy_threshold, min_confidence=min_confidence)
    signals = generate_signals(symbol, cfg)

    out_dir = INSTRUMENTS[symbol]["model_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)


    out_path = out_dir / "latest_signals.csv"
    signals["generated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    signals.to_csv(out_path, index=False)

    return out_path



def mini_backtest_90d(signals_df: pd.DataFrame,price_df: pd.DataFrame,horizon_days: int = 5):
    # --- DEFENSIVE DEFAULTS ---
    bh_return = 0.0
    total_return = 0.0
    wins = 0
    trades = []

    # --- BASIC VALIDATION ---
    if signals_df is None or price_df is None:
        return {
            "window_days": 90,
            "model_return": 0.0,
            "buy_hold_return": 0.0,
            "win_rate": 0.0,
            "trades": 0,
            "note": "Hiányzó bemeneti adatok."
        }

    if "Date" not in signals_df.columns or "Close" not in price_df.columns:
        return {
            "window_days": 90,
            "model_return": 0.0,
            "buy_hold_return": 0.0,
            "win_rate": 0.0,
            "trades": 0,
            "note": "Hiányzó Date / Close oszlop."
        }

    # --- NORMALIZE ---
    sig = signals_df.copy()
    prices = price_df.copy()

    sig["Date"] = (
        pd.to_datetime(sig["Date"], utc=True, errors="coerce")
        .dt.tz_convert(None)
        .dt.normalize()
    )

    prices["Date"] = (
        pd.to_datetime(prices["Date"], utc=True, errors="coerce")
        .dt.tz_convert(None)
        .dt.normalize()
    )


    sig = sig.dropna(subset=["Date"])
    prices = prices.dropna(subset=["Date"])

    if "recommendation" not in sig.columns:
        return {
            "window_days": 90,
            "model_return": 0.0,
            "buy_hold_return": 0.0,
            "win_rate": 0.0,
            "trades": 0,
            "note": "Hiányzó recommendation oszlop."
        }

    # --- LAST 90 DAYS ---
    cutoff = sig["Date"].max() - pd.Timedelta(days=90)
    sig = sig[sig["Date"] >= cutoff]

    if sig.empty or prices.empty:
        return {
            "window_days": 90,
            "model_return": 0.0,
            "buy_hold_return": 0.0,
            "win_rate": 0.0,
            "trades": 0,
            "note": "Nincs elegendő adat a 90 napos ablakban."
        }

    price_map = prices.set_index("Date")["Close"]

    # --- TRADES ---
    for _, row in sig.iterrows():
        rec = str(row.get("recommendation", "")).upper()
        if rec != "BUY":
            continue

        entry_date = row["Date"]
        exit_date = entry_date + pd.Timedelta(days=horizon_days)

        if entry_date not in price_map.index or exit_date not in price_map.index:
            continue

        entry = float(price_map.loc[entry_date])
        exit_ = float(price_map.loc[exit_date])

        if entry <= 0 or exit_ <= 0:
            continue

        r = (exit_ / entry) - 1
        trades.append(r)
        total_return += r

        if r > 0:
            wins += 1

    # --- BUY & HOLD (ALWAYS COMPUTED) ---
    bh_prices = prices[prices["Date"] >= cutoff]
    bh_prices = bh_prices.copy()

    bh_prices["Close"] = (
        pd.to_numeric(bh_prices["Close"], errors="coerce")
    )

    bh_prices = bh_prices.dropna(subset=["Close"])

    if len(bh_prices) < 2:
        return {
            "model_return": 0.0,
            "buy_hold_return": 0.0,
            "trades": 0,
            "winrate": 0.0,
        }

    bh_return = (bh_prices["Close"].iloc[-1] / bh_prices["Close"].iloc[0]) - 1


    trade_count = len(trades)

    # --- NO TRADES CASE ---
    if trade_count == 0:
        return {
            "window_days": 90,
            "model_return": 0.0,
            "buy_hold_return": round(bh_return * 100, 2),
            "win_rate": 0.0,
            "trades": 0,
            "note": "Nincs lezárt BUY trade az elmúlt 90 napban (5 napos horizon)."
        }

    # --- NORMAL RESULT ---
    win_rate = wins / trade_count

    return {
        "window_days": 90,
        "model_return": round(total_return * 100, 2),
        "buy_hold_return": round(bh_return * 100, 2),
        "win_rate": round(win_rate * 100, 1),
        "trades": trade_count
    }


def save_model_meta(symbol, model, cv_accuracy, roc_auc, feature_names, train_rows):
    model_dir = Path("backend/model") / symbol.split(".")[0].lower()
    model_dir.mkdir(parents=True, exist_ok=True)

    meta = {
        "symbol": symbol,
        "model": type(model).__name__,
        "cv_accuracy": round(float(cv_accuracy), 4),
        "roc_auc": round(float(roc_auc), 4),
        "cv_type": "TimeSeriesSplit",
        "train_rows": int(train_rows),
        "features": feature_names,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M")
    }

    meta_path = model_dir / "model_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")


if __name__ == "__main__":
    for symbol in INSTRUMENTS.keys():
        out_path = run_pipeline(symbol)
        print(f"✅ {symbol} latest_signals.csv updated: {out_path}")
