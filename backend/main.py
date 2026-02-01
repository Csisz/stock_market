from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from pathlib import Path

from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path


app = FastAPI(title="Stock Predictor API")

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"

app.mount(
    "/static",
    StaticFiles(directory=FRONTEND_DIR),
    name="static"
)


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

@app.get("/")
def serve_frontend():
    index_path = FRONTEND_DIR / "index.html"
    return FileResponse(index_path)

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

@app.get("/api/equity")
def get_equity():
    path = MODEL_DIR / "equity_curve.csv"

    if not path.exists():
        return {
            "error": "equity_curve.csv nem található",
            "hint": "Futtasd le előbb: python backend/equity.py"
        }

    df = pd.read_csv(path)

    return {
        "data": [
            {
                "date": row["date"],
                "bh_equity": float(row["bh_equity"]),
                "model_equity": float(row["model_equity"])
            }
            for _, row in df.iterrows()
        ]
    }

