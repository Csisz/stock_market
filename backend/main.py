from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from pathlib import Path

app = FastAPI(title="Stock Predictor API")

# --- CORS (frontend miatt) ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # demo
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "model"

OTP_CSV = DATA_DIR / "otp_ohlcv_history.csv"
SIGNALS_CSV = MODEL_DIR / "latest_signals.csv"


@app.get("/api/price")
def get_price(symbol: str = Query("OTP.BD")):
    if symbol != "OTP.BD":
        return {"error": "Csak OTP.BD támogatott egyelőre"}

    df = pd.read_csv(OTP_CSV)
    df["Date"] = pd.to_datetime(df["Date"])

    return {
        "symbol": symbol,
        "data": [
            {
                "date": row["Date"].strftime("%Y-%m-%d"),
                "close": float(row["Close"])
            }
            for _, row in df.iterrows()
        ]
    }


@app.get("/api/signals")
def get_signals(threshold: float = Query(0.55)):
    df = pd.read_csv(SIGNALS_CSV)
    df["Date"] = pd.to_datetime(df["Date"])

    df["signal"] = df["p_up_next_5d"].apply(
        lambda x: "BUY" if x >= threshold else "HOLD"
    )

    return {
        "threshold": threshold,
        "data": [
            {
                "date": row["Date"].strftime("%Y-%m-%d"),
                "close": float(row["Close"]),
                "p_up": float(row["p_up_next_5d"]),
                "signal": row["signal"]
            }
            for _, row in df.iterrows()
        ]
    }
