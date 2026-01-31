import yfinance as yf
import pandas as pd
from pandas_datareader import data as pdr
from pathlib import Path
from datetime import datetime

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

TODAY = datetime.today().strftime("%Y-%m-%d")

# =========================
# OTP – Yahoo
# =========================
print("📥 OTP letöltés: OTP.BD")
otp = yf.Ticker("OTP.BD").history(period="max")
otp = otp[["Open", "High", "Low", "Close", "Volume"]]
otp.index.name = "Date"
otp.to_csv(DATA_DIR / "otp_ohlcv_history.csv")
print("💾 Mentve: data/otp_ohlcv_history.csv")

# =========================
# EUR/HUF – Yahoo
# =========================
print("📥 EUR/HUF letöltés: EURHUF=X")
fx = yf.Ticker("EURHUF=X").history(period="max")
fx = fx[["Open", "High", "Low", "Close", "Volume"]]
fx.index.name = "Date"
fx.to_csv(DATA_DIR / "eurhuf_history.csv")
print("💾 Mentve: data/eurhuf_history.csv")

# =========================
# BUX – STOOQ
# =========================
print("📥 BUX letöltés: ^BUX (Stooq)")

# Stooq ticker: ^BUX → BUX
# bux = pdr.DataReader("BUX", "stooq")
bux = pdr.DataReader("^BUX", "stooq")


if bux.empty:
    raise RuntimeError("BUX adat nem érkezett Stooq-ról")

# Stooq fordított időrendben adja
bux = bux.sort_index()

bux = bux.rename(columns={
    "Open": "Open",
    "High": "High",
    "Low": "Low",
    "Close": "Close",
    "Volume": "Volume"
})

bux.index.name = "Date"
bux.to_csv(DATA_DIR / "bux_history.csv")
print("💾 Mentve: data/bux_history.csv")
