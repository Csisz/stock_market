import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import os


SYMBOL = os.getenv("SYMBOL", "otp").lower()

print(f"🚀 Modell tanítása: {SYMBOL.upper()}")

DATA_PATH = Path(f"backend/data/{SYMBOL}_ohlcv_history.csv")
MODEL_OUT = Path(f"backend/model/{SYMBOL}_latest_signals.csv")


FORECAST_HORIZON = 5
SPLIT_DATE = pd.Timestamp("2025-01-01")

# Küszöbök – stratégiához hasznos
THRESHOLDS = [0.50, 0.55, 0.60, 0.65]


# =========================
# 1) Betöltés (robosztus dátum)
# =========================

# df = pd.read_csv(OTP_PATH)
df = pd.read_csv(DATA_PATH)
df["Date"] = pd.to_datetime(df["Date"], utc=True, errors="coerce").dt.tz_convert(None).dt.normalize()
df = df.set_index("Date").sort_index()

# =========================
# 2) Feature engineering (OTP-only, clean)
# =========================

# Momentum
df["ret_1"] = df["Close"].pct_change(fill_method=None)
df["ret_3"] = df["Close"].pct_change(3, fill_method=None)
df["ret_5"] = df["Close"].pct_change(5, fill_method=None)

# Trend / mean reversion
df["sma_10"] = df["Close"].rolling(10).mean()
df["sma_20"] = df["Close"].rolling(20).mean()
df["price_vs_sma20"] = df["Close"] / df["sma_20"] - 1

# Volatility
df["vol_10"] = df["ret_1"].rolling(10).std()
df["vol_20"] = df["ret_1"].rolling(20).std()

# Range + candle body (price action)
df["range_pct"] = (df["High"] - df["Low"]) / df["Close"]
df["body_pct"] = (df["Close"] - df["Open"]) / df["Close"]

# Volume – log + relative volume (stabilabb, mint pct_change)
df["log_vol"] = np.log1p(df["Volume"])
df["vol_sma_20"] = df["log_vol"].rolling(20).mean()
df["rel_vol_20"] = df["log_vol"] - df["vol_sma_20"]  # mennyire tér el a szokásostól

# =========================
# 3) Target (irány 5 napra)
# =========================
df["target_up"] = (df["Close"].shift(-FORECAST_HORIZON) > df["Close"]).astype(int)

# =========================
# 4) Dataset összeállítás
# =========================
FEATURES = [
    "ret_1", "ret_3", "ret_5",
    "price_vs_sma20",
    "vol_10", "vol_20",
    "range_pct", "body_pct",
    "rel_vol_20",
]

df.replace([np.inf, -np.inf], np.nan, inplace=True)
data = df[FEATURES + ["target_up", "Close"]].dropna()

X = data[FEATURES]
y = data["target_up"]

print("Használható minták:", len(data))

# =========================
# 5) Modell (konzervatív RF)
# =========================
def make_model():
    return RandomForestClassifier(
        n_estimators=800,
        max_depth=6,
        min_samples_leaf=30,
        random_state=42,
        class_weight="balanced_subsample",
        n_jobs=-1
    )

# =========================
# 6) Idősoros CV
# =========================
tscv = TimeSeriesSplit(n_splits=5)
accs, aucs = [], []

for i, (train_idx, test_idx) in enumerate(tscv.split(X), start=1):
    model = make_model()
    model.fit(X.iloc[train_idx], y.iloc[train_idx])
    proba = model.predict_proba(X.iloc[test_idx])[:, 1]
    pred = (proba >= 0.5).astype(int)

    accs.append(accuracy_score(y.iloc[test_idx], pred))
    aucs.append(roc_auc_score(y.iloc[test_idx], proba))
    print(f"Fold {i} | Acc: {accs[-1]:.3f} | ROC-AUC: {aucs[-1]:.3f}")

print("\n==== CV ÖSSZEGZÉS ====")
print(f"Átlag Acc: {np.mean(accs):.3f}")
print(f"Átlag ROC-AUC: {np.mean(aucs):.3f}")

# =========================
# 7) Out-of-sample (2025 után)
# =========================
train = data.loc[data.index < SPLIT_DATE]
test = data.loc[data.index >= SPLIT_DATE]

model = make_model()
model.fit(train[FEATURES], train["target_up"])

test_proba = model.predict_proba(test[FEATURES])[:, 1]
test_results = pd.DataFrame({
    "Close": test["Close"],
    "p_up_next_5d": test_proba,
    "actual": test["target_up"]
}, index=test.index)

print("\n==== OUT-OF-SAMPLE (2025 után) ====")

for thr in THRESHOLDS:
    pred = (test_results["p_up_next_5d"] >= thr).astype(int)
    print(f"\n--- threshold={thr:.2f} ---")
    print(classification_report(test_results["actual"], pred, digits=4))

# =========================
# 8) Diagram: Close + p_up + jelzések (thr=0.60)
# =========================
# Jelzés létrehozása vizualizációhoz
PLOT_THRESHOLD = 0.60
test_results["signal"] = (test_results["p_up_next_5d"] >= PLOT_THRESHOLD).astype(int)


plt.figure(figsize=(16, 9), facecolor="#f4f6f8")

# Valós záróár – sötétkék
plt.plot(
    test_results.index,
    test_results["Close"],
    label="Valós záróár",
    color="#1f3b4d",
    linewidth=2.2,
    alpha=0.95
)

# UP jelzések – zöld/arany háromszög
sig_up = test_results[test_results["signal"] == 1]
plt.scatter(
    sig_up.index,
    sig_up["Close"],
    marker="^",
    s=110,
    color="#2ecc71",
    edgecolor="#145a32",
    linewidth=1.2,
    label="UP jel (p ≥ 0.60)",
    zorder=5
)

# Esztétika
plt.title(
    "OTP – Valós záróár és modell által jelzett belépési pontok (2025 után)",
    fontsize=18,
    fontweight="bold",
    pad=15
)

plt.xlabel("Dátum", fontsize=13)
plt.ylabel("Ár (HUF)", fontsize=13)

plt.grid(
    which="major",
    linestyle="--",
    linewidth=0.6,
    alpha=0.35
)

plt.legend(
    fontsize=12,
    frameon=True,
    fancybox=True,
    framealpha=0.9
)

plt.tight_layout()
plt.show()


# =========================
# 9) 3 random esettanulmány (2025 után, thr=0.60)
# =========================
random.seed(42)

cases = test_results.copy()
# valós 5 nap múlva záróár
cases["future_close"] = df["Close"].shift(-FORECAST_HORIZON).reindex(cases.index)
cases = cases.dropna()

print("\n===== RANDOM ESETTANULMÁNYOK (thr=0.60) =====")
for date in random.sample(list(cases.index), 3):
    row = cases.loc[date]
    decision = "UP" if row["p_up_next_5d"] >= thr else "NO-TRADE/DOWN"

    print("\n----------------------------")
    print(f"Dátum: {date.date()}")
    print(f"Akkori záróár: {row['Close']:.0f}")
    print(f"p_up_next_5d: {row['p_up_next_5d']:.3f} -> döntés: {decision}")

    future = df.loc[date:date + pd.Timedelta(days=7), "Close"].iloc[1:FORECAST_HORIZON + 1]
    print("Valós következő 5 nap záróár:")
    for d, v in future.items():
        print(f"  {d.date()} → {v:.0f}")

    real = "UP" if future.iloc[-1] > row["Close"] else "DOWN"
    print("Valós kimenet 5 nap múlva:", real)



# =========================
# 10) MINI BACKTEST (5 napos tartás)
# =========================

BACKTEST_THRESHOLD = 0.55   # ezt érdemes később variálni
HOLD_DAYS = 5

bt = test_results.copy()

# jövőbeli záróár (kilépés)
bt["exit_price"] = df["Close"].shift(-HOLD_DAYS).reindex(bt.index)

# csak ahol van elég jövőbeli adat
bt = bt.dropna(subset=["exit_price"])

# jelzés
bt["trade"] = bt["p_up_next_5d"] >= BACKTEST_THRESHOLD

# hozam számítás
bt["return"] = np.where(
    bt["trade"],
    (bt["exit_price"] / bt["Close"]) - 1,
    0.0
)

# equity görbe (1 egység indulótőke)
bt["equity"] = (1 + bt["return"]).cumprod()

# drawdown
bt["cum_max"] = bt["equity"].cummax()
bt["drawdown"] = bt["equity"] / bt["cum_max"] - 1


# =========================
# Backtest összegzés
# =========================
trades = bt[bt["trade"]]

print("\n===== MINI BACKTEST (5 napos tartás) =====")
print(f"Threshold: p_up ≥ {BACKTEST_THRESHOLD}")
print(f"Trade-ek száma: {len(trades)}")

if len(trades) > 0:
    print(f"Találati arány: {(trades['return'] > 0).mean():.2%}")
    print(f"Átlag hozam / trade: {trades['return'].mean():.2%}")
    print(f"Medián hozam / trade: {trades['return'].median():.2%}")
    print(f"Összes hozam: {bt['equity'].iloc[-1] - 1:.2%}")
    print(f"Max drawdown: {bt['drawdown'].min():.2%}")
else:
    print("Nem volt egyetlen trade sem ezen a thresholdon.")


# =========================
# Equity curve plot
# =========================
plt.figure(figsize=(14, 7))
plt.plot(bt.index, bt["equity"], label="Equity görbe", linewidth=2)
plt.title(f"Mini backtest – 5 napos tartás (threshold={BACKTEST_THRESHOLD})")
plt.xlabel("Dátum")
plt.ylabel("Tőke (relatív)")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()


# =========================
# 11) API-hoz export
# =========================
EXPORT_PATH = MODEL_OUT

api_df = test_results.copy()
api_df["signal"] = api_df["p_up_next_5d"].apply(
    lambda x: "BUY" if x >= BACKTEST_THRESHOLD else "HOLD"
)

api_df = api_df.reset_index()[[
    "Date", "Close", "p_up_next_5d", "signal"
]]

EXPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
api_df.to_csv(EXPORT_PATH, index=False)

print(f"✅ API export kész: {EXPORT_PATH}")
