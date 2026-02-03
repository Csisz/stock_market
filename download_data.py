import yfinance as yf
from pathlib import Path
from datetime import datetime

DATA_DIR = Path("backend/data")
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
# MOL – Yahoo
# =========================
print("📥 MOL letöltés: MOL.BD")
mol = yf.Ticker("MOL.BD").history(period="max")
mol = mol[["Open", "High", "Low", "Close", "Volume"]]
mol.index.name = "Date"
mol.to_csv(DATA_DIR / "mol_ohlcv_history.csv")
print("💾 Mentve: data/mol_ohlcv_history.csv")
