import yfinance as yf
import pandas as pd
from pathlib import Path
from datetime import datetime

DATA_DIR = Path(__file__).parent

SYMBOLS = {
    "OTP": "OTP.BD",
    "MOL": "MOL.BD"
}

def update_symbol(symbol, ticker):
    print(f"📥 Updating {symbol} ({ticker})")

    df = yf.download(
        ticker,
        period="max",
        interval="1d",
        auto_adjust=True,
        progress=False
    )

    if df.empty:
        print(f"⚠️ No data for {symbol}")
        return

    df.reset_index(inplace=True)
    

    if "Date" not in df.columns:
        if "Datetime" in df.columns:
            df.rename(columns={"Datetime": "Date"}, inplace=True)
        elif "index" in df.columns:
            df.rename(columns={"index": "Date"}, inplace=True)
        elif "date" in df.columns:
            df.rename(columns={"date": "Date"}, inplace=True)
 
    df.rename(columns={
        "date": "Date",
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "volume": "Volume",
    }, inplace=True)


    out_file = DATA_DIR / f"{symbol.lower()}_ohlcv_history.csv"
    df.to_csv(out_file, index=False)

    print(f"✅ Saved {out_file} ({len(df)} rows)")

def update_prices(ticker: str):
    """
    Frissíti egyetlen instrumentum árfolyamát ticker alapján
    pl: "OTP.BD"
    """
    for symbol, t in SYMBOLS.items():
        if t == ticker:
            update_symbol(symbol, t)
            return

    raise ValueError(f"Ismeretlen ticker: {ticker}")

def run():
    for symbol, ticker in SYMBOLS.items():
        update_symbol(symbol, ticker)

if __name__ == "__main__":
    run()

