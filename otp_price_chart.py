import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

# ========= Konstansok =========
TICKER = "OTP.BD"
DATA_DIR = Path("data")
CSV_PATH = DATA_DIR / "otp_ohlcv_history.csv"

# ========= Mappa =========
DATA_DIR.mkdir(exist_ok=True)

print("📥 OTP OHLCV adatok letöltése...")

# ========= Letöltés =========
otp = yf.Ticker(TICKER)
df = otp.history(period="max")

if df.empty:
    raise RuntimeError("❌ Nem sikerült adatot letölteni Yahoo Finance-ről.")

# ========= Szükséges oszlopok =========
df = df[["Open", "High", "Low", "Close", "Volume"]]

# Index neve legyen Date (CSV-hez fontos)
df.index.name = "Date"

print(f"✅ Letöltött sorok száma: {len(df)}")
print(df.head())

# ========= CSV mentés =========
df.to_csv(CSV_PATH, encoding="utf-8")
print(f"💾 CSV elmentve ide: {CSV_PATH.resolve()}")

# ========= Diagram =========
plt.figure(figsize=(14, 7))

plt.plot(df.index, df["Close"], label="Záró ár", linewidth=1.4)
plt.plot(df.index, df["Open"], label="Nyitó ár", linewidth=0.8, alpha=0.7)

plt.title("OTP Bank – Nyitó és záró árfolyam (teljes idősor)")
plt.xlabel("Dátum")
plt.ylabel("Ár (HUF)")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.show()
