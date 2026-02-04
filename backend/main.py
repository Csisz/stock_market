from fastapi import Body, FastAPI, Query, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from apscheduler.schedulers.background import BackgroundScheduler
from pathlib import Path
from datetime import datetime, timedelta

import subprocess
import pandas as pd
import json
import yfinance as yf
import requests

from .pipeline import run_pipeline, mini_backtest_90d
from .equity import generate_equity_curve
from .config import INSTRUMENTS


app = FastAPI(title="Stock Predictor API")

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

_YAHOO_FIXING_CACHE = {}

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # demo
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).parent
MODEL_DIR = BASE_DIR / "model"

def get_yahoo_previous_close_cached(symbol: str):
    now = datetime.utcnow()

    cached = _YAHOO_FIXING_CACHE.get(symbol)
    if cached and now - cached["ts"] < timedelta(hours=6):
        return cached["value"]

    try:
        value = get_yahoo_previous_close(symbol)
    except Exception:
        if cached:
            return cached["value"]
        return None

    _YAHOO_FIXING_CACHE[symbol] = {
        "value": value,
        "ts": now
    }
    return value

def get_intraday_data(symbol: str):
    intraday = get_intraday_price(symbol)

    fixing = get_yahoo_previous_close_cached(symbol)

    if not intraday:
        return {
            "price": None,
            "price_time": None,
            "previous_close": fixing
        }

    return {
        "price": intraday["price"],
        "price_time": intraday["price_time"],
        "previous_close": fixing
    }

    
def get_yahoo_previous_close(symbol: str):
    url = (
        "https://query1.finance.yahoo.com/v10/finance/quoteSummary/"
        f"{symbol}?modules=price"
    )

    r = requests.get(url, timeout=5)
    r.raise_for_status()
    data = r.json()

    price = data["quoteSummary"]["result"][0]["price"]
    return price["regularMarketPreviousClose"]["raw"]

    
def _require_symbol(symbol: str) -> str:
    if symbol not in INSTRUMENTS:
        raise ValueError("Ismeretlen instrument")
    return symbol


def _signals_csv(symbol: str) -> Path:
    # per-instrument signals file
    return INSTRUMENTS[symbol]["model_dir"] / "latest_signals.csv"


def _equity_csv(symbol: str) -> Path:
    # per-instrument equity curve file
    return INSTRUMENTS[symbol]["model_dir"] / "equity_curve.csv"


def _load_price_df(symbol: str) -> pd.DataFrame:
    csv_path = INSTRUMENTS[symbol]["csv"]
    df = pd.read_csv(csv_path)
    df["Date"] = (
        pd.to_datetime(df["Date"], utc=True, errors="coerce")
        .dt.tz_convert(None)
        .dt.normalize()
    )

    df = df.dropna(subset=["Date"]).sort_values("Date")
    return df


@app.get("/")
def serve_frontend():
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/api/price")
def get_price(symbol: str = Query("OTP.BD")):
    try:
        symbol = _require_symbol(symbol)
    except ValueError:
        return {"error": "Ismeretlen instrument"}

    df = _load_price_df(symbol)

    return {
        "symbol": symbol,
        "data": [
            {"date": row["Date"].strftime("%Y-%m-%d"), "close": float(row["Close"])}
            for _, row in df.iterrows()
        ],
    }


@app.get("/api/signals")
def get_signals(
    symbol: str = Query("OTP.BD"),
    buy_threshold: float = Query(0.60, ge=0.5, le=0.9),
    min_confidence: float = Query(0.55, ge=0.5, le=0.9),
):
    try:
        symbol = _require_symbol(symbol)
    except ValueError:
        return {"error": "Ismeretlen instrument"}

    signals_csv = _signals_csv(symbol)

    if not signals_csv.exists():
        run_pipeline(symbol, buy_threshold, min_confidence)

    df = pd.read_csv(signals_csv)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date")

    sell_threshold = 1.0 - buy_threshold

    def recalc_rec(row):
        p = float(row["p_up_next_5d"])
        buy_conf = p
        sell_conf = 1.0 - p
        best = max(buy_conf, sell_conf)
        if best < min_confidence:
            return "NO SIGNAL"
        if p >= buy_threshold:
            return "BUY"
        if p <= sell_threshold:
            return "SELL"
        return "HOLD"

    df["recommendation"] = df.apply(recalc_rec, axis=1)

    def parse_expl(x):
        try:
            return json.loads(x) if isinstance(x, str) and x.strip() else []
        except Exception:
            return []

    return {
        "symbol": symbol,
        "buy_threshold": buy_threshold,
        "min_confidence": min_confidence,
        "data": [
            {
                "date": row["Date"].strftime("%Y-%m-%d"),
                "close": float(row["Close"]),
                "p_up": float(row["p_up_next_5d"]),
                "recommendation": row["recommendation"],
                "confidence": float(row.get("confidence", 0.0)),
                "expected_return_5d": float(row.get("expected_return_5d", 0.0))
                if str(row.get("expected_return_5d", "")).lower() != "nan"
                else None,
                "risk_level": str(row.get("risk_level", "medium")),
                "explanation": parse_expl(row.get("explanation", "")),
            }
            for _, row in df.iterrows()
        ],
    }


@app.get("/api/latest")
def get_latest(
    symbol: str = Query("OTP.BD"),
    buy_threshold: float = Query(0.60, ge=0.5, le=0.9),
    min_confidence: float = Query(0.55, ge=0.5, le=0.9),
):
    try:
        symbol = _require_symbol(symbol)
    except ValueError:
        return {"error": "Ismeretlen instrument"}

    signals_csv = _signals_csv(symbol)
    if not signals_csv.exists():
        run_pipeline(symbol, buy_threshold, min_confidence)

    df = pd.read_csv(signals_csv)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date")
    latest = df.iloc[-1]

    p = float(latest["p_up_next_5d"])
    sell_threshold = 1.0 - buy_threshold
    buy_conf = p
    sell_conf = 1.0 - p
    best = max(buy_conf, sell_conf)

    if best < min_confidence:
        rec = "NO SIGNAL"
        conf = best
    elif p >= buy_threshold:
        rec = "BUY"
        conf = buy_conf
    elif p <= sell_threshold:
        rec = "SELL"
        conf = sell_conf
    else:
        rec = "HOLD"
        conf = best

    try:
        expl = json.loads(latest.get("explanation", "[]"))
    except Exception:
        expl = []

    return JSONResponse({
        "symbol": symbol,
        "date": latest["Date"].strftime("%Y-%m-%d"),
        "ticker": symbol,
        "close": float(latest["Close"]),
        "recommendation": rec,
        "p_up_next_5d": p,
        "confidence": float(conf),
        "expected_return_5d": float(latest.get("expected_return_5d", 0.0))
        if str(latest.get("expected_return_5d", "")).lower() != "nan"
        else None,
        "risk_level": str(latest.get("risk_level", "medium")),
        "explanation": expl,
        "buy_threshold": buy_threshold,
        "min_confidence": min_confidence,
        "updated_at": latest.get("generated_at"),
    })


@app.post("/api/recalculate")
def recalculate(
    symbol: str = Query("OTP.BD"),
    buy_threshold: float = Query(0.60, ge=0.5, le=0.9),
    min_confidence: float = Query(0.55, ge=0.5, le=0.9),
):
    try:
        symbol = _require_symbol(symbol)
    except ValueError:
        return {"error": "Ismeretlen instrument"}

    out_path = run_pipeline(symbol, buy_threshold, min_confidence)

    # equity újragenerálás (BUY/HOLD logikával) – per symbol
    price_df = _load_price_df(symbol).rename(columns={"Date": "date", "Close": "close"})
    price_df["date"] = pd.to_datetime(price_df["date"], utc=True, errors="coerce").dt.tz_convert(None).dt.normalize()

    sig_df = pd.read_csv(out_path)
    sig_df = sig_df.rename(columns={"Date": "date", "recommendation": "signal"})
    sig_df["date"] = pd.to_datetime(sig_df["date"], utc=True, errors="coerce").dt.tz_convert(None).dt.normalize()
    sig_df = sig_df.dropna(subset=["date"])

    # SELL/NO SIGNAL -> HOLD (equity csak BUY/HOLD)
    sig_df["signal"] = sig_df["signal"].replace({"SELL": "HOLD", "NO SIGNAL": "HOLD"})

    eq_out = generate_equity_curve(price_df, sig_df)

    # mentsük per-symbol mappába
    eq_path = _equity_csv(symbol)
    eq_path.parent.mkdir(parents=True, exist_ok=True)
    eq_out.to_csv(eq_path, index=False)

    return {
        "ok": True,
        "symbol": symbol,
        "signals_csv": str(Path(out_path).resolve()),
        "equity_csv": str(eq_path.resolve()),
    }


@app.get("/api/equity")
def get_equity(symbol: str = Query("OTP.BD")):
    try:
        symbol = _require_symbol(symbol)
    except ValueError:
        return {"error": "Ismeretlen instrument"}

    path = _equity_csv(symbol)
    if not path.exists():
        return {"error": "equity_curve.csv nem található", "hint": "Futtasd le előbb: POST /api/recalculate"}

    df = pd.read_csv(path)
    return {
        "symbol": symbol,
        "data": [
            {"date": row["date"], "bh_equity": float(row["bh_equity"]), "model_equity": float(row["model_equity"])}
            for _, row in df.iterrows()
        ]
    }


@app.get("/api/mini-backtest")
def get_mini_backtest(symbol: str = Query("OTP.BD")):
    try:
        symbol = _require_symbol(symbol)
    except ValueError:
        return {"error": "Ismeretlen instrument"}

    signals_csv = _signals_csv(symbol)
    if not signals_csv.exists():
        return {"error": "Nincs signal adat."}

    signals = pd.read_csv(signals_csv)
    prices = _load_price_df(symbol)

    stats = mini_backtest_90d(signals, prices)

    if not stats:
        return {"error": "Nincs elég adat a mini backtesthez."}

    stats["symbol"] = symbol
    return stats

@app.get("/api/backtest-summary")
def backtest_summary(symbol: str, buy_threshold: float):
    try:
        symbol = _require_symbol(symbol)
    except ValueError:
        return {"error": "Ismeretlen instrument"}

    # --- equity szükséges ---
    eq_path = _equity_csv(symbol)
    if not eq_path.exists():
        return {
            "error": "NO_BACKTEST",
            "hint": "Futtasd le előbb: POST /api/recalculate"
        }

    # equity betöltés
    eq = pd.read_csv(eq_path)

    # modell hozam
    total_return = eq["model_equity"].iloc[-1] - 1.0

    # max drawdown
    peak = eq["model_equity"].cummax()
    dd = (eq["model_equity"] - peak) / peak
    max_dd = dd.min()

    # mini backtest találati arány
    signals = pd.read_csv(_signals_csv(symbol))
    prices = _load_price_df(symbol)
    mini = mini_backtest_90d(signals, prices)

    hit_rate = mini.get("win_rate", 0) / 100.0 if mini else 0.0

    return {
        "hit_rate": hit_rate,
        "total_return": total_return,
        "max_drawdown": max_dd,
        "hold_days": 5
    }

scheduler = BackgroundScheduler()

def scheduled_daily_update():
    now = datetime.now()

    # hétvége
    if now.weekday() >= 5:
        return
    
    subprocess.run(["python", "backend/update_pipeline.py"])

scheduler.add_job(
    scheduled_daily_update,
    "cron",
    hour=18,
    minute=30
)


def get_intraday_price(symbol="OTP.BD"):
    df = yf.download(
        symbol,
        period="1d",
        interval="5m",
        progress=False
    )

    if df.empty:
        return None

    last = df.iloc[-1]

    return {
        "price": float(pd.to_numeric(last["Close"], errors="coerce").iloc[0]),
        "price_time": last.name.strftime("%Y-%m-%d %H:%M")
    }


@app.get("/api/intraday-price")
def intraday_price(symbol: str = "OTP.BD"):
    return get_intraday_data(symbol)


def start_scheduler():
    if not scheduler.running:
        scheduler.start()
        print("[SCHEDULER] BackgroundScheduler elindult")
    else:
        print("[SCHEDULER] már fut")


@app.on_event("startup")
def on_startup():
    # --- fixing cache warmup ---
    for symbol in ["OTP.BD"]:
        try:
            val = get_yahoo_previous_close(symbol)
            if val:
                _YAHOO_FIXING_CACHE[symbol] = {
                    "value": val,
                    "ts": datetime.utcnow()
                }
                print(f"[FIXING CACHE WARMED] {symbol} = {val}")
        except Exception as e:
            print("[FIXING CACHE FAILED]", e)

    # --- scheduler indítása ---
    start_scheduler()





@app.get("/api/status")
def status(symbol: str = "OTP.BD"):
    # --- intraday ár ---
    # price_data = intraday_price(symbol)  # ha a /api/intraday-price függvényed így hívható
    price_data = get_intraday_data(symbol)

    if isinstance(price_data, dict) and price_data.get("error"):
        price_data = {"price": None, "price_time": None}
        # price_data = get_intraday_data(symbol)

    # --- modell info: latest_signals.csv utolsó sora ---
    # igazítsd a path-ot a saját struktúrádhoz
    model_path = Path("backend/model") / symbol.split(".")[0].lower() / "latest_signals.csv"
    if not model_path.exists():
        raise HTTPException(status_code=404, detail=f"Missing model file: {model_path}")


    # CSV beolvasás
    df = pd.read_csv(model_path)
    last = df.iloc[-1]

    # 1️⃣ ha van explicit időbélyeg az adatban
    model_updated_at = (
        last.get("generated_at")
        or last.get("updated_at")
    )

    # 2️⃣ ha nincs → fájl módosítási ideje
    if not model_updated_at:
        ts = model_path.stat().st_mtime
        model_updated_at = datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M")

    return {
        "symbol": symbol,
        "price": price_data.get("price"),
        "price_time": price_data.get("price_time"),
        "model_updated_at": model_updated_at,
        "model_date": last.get("Date"),
        "recommendation": last.get("recommendation"),
        "confidence": float(last.get("p_up")) if last.get("p_up") is not None else None,
    }



SUBS_FILE = Path("backend/data/subscribers.json")
SUBS_FILE.parent.mkdir(parents=True, exist_ok=True)
if not SUBS_FILE.exists():
    SUBS_FILE.write_text("[]", encoding="utf-8")

@app.get("/api/subscribers")
def list_subscribers():
    return json.loads(SUBS_FILE.read_text(encoding="utf-8") or "[]")

@app.post("/api/subscribers")
def add_subscriber(email: str = Body(..., embed=True)):
    email = email.strip().lower()
    subs = json.loads(SUBS_FILE.read_text(encoding="utf-8") or "[]")
    if email not in subs:
        subs.append(email)
        SUBS_FILE.write_text(json.dumps(subs, indent=2), encoding="utf-8")
    return {"ok": True, "count": len(subs)}

@app.delete("/api/subscribers")
def remove_subscriber(email: str):
    email = email.strip().lower()
    subs = json.loads(SUBS_FILE.read_text(encoding="utf-8") or "[]")
    subs = [x for x in subs if x != email]
    SUBS_FILE.write_text(json.dumps(subs, indent=2), encoding="utf-8")
    return {"ok": True, "count": len(subs)}


from backend.notifications.emailer import send_email
from backend.notifications.alerts import _badge_color
import os
from datetime import datetime

@app.post("/api/test-signal-email")
def test_signal_email(
    symbol: str = "OTP.BD",
    rec: str = "BUY",
    p_up: float = 0.62,
):
    BASE_URL = os.getenv("BASE_URL", "http://127.0.0.1:8000")
    color = "#16a34a" if rec.upper() == "BUY" else "#dc2626"
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    subject = f"[StockTrade – TESZT] {symbol} jelzés: {rec}"

    plain_body = (
        f"TESZT EMAIL\n\n"
        f"Symbol: {symbol}\n"
        f"Ajánlás: {rec}\n"
        f"p_up (5 nap): {p_up}\n"
        f"Időpont: {now}\n\n"
        f"Dashboard: {BASE_URL}"
    )

    html_body = f"""
    <!doctype html>
    <html>
      <body style="margin:0;padding:0;background:#0b1220;font-family:Arial,Helvetica,sans-serif;">
        <div style="max-width:640px;margin:0 auto;padding:24px;">
          <div style="background:#0f1a2e;border:1px solid #1f2a44;border-radius:16px;padding:20px;color:#e5e7eb;">
            <div style="display:flex;align-items:center;justify-content:space-between;">
              <div style="font-size:18px;font-weight:700;">StockTrade – TESZT jelzés</div>
              <div style="background:{color};color:white;padding:6px 10px;border-radius:999px;font-size:12px;font-weight:700;">
                {rec}
              </div>
            </div>

            <div style="margin-top:14px;font-size:14px;color:#cbd5e1;">
              <div><b>Instrumentum:</b> {symbol}</div>
              <div><b>Ajánlás:</b> {rec}</div>
              <div><b>p_up (5 nap):</b> {p_up}</div>
              <div><b>Időpont:</b> {now}</div>
            </div>

            <div style="margin-top:18px;">
              <a href="{BASE_URL}"
                 style="display:inline-block;background:#2563eb;color:white;text-decoration:none;
                        padding:10px 14px;border-radius:12px;font-weight:700;">
                Megnyitás a dashboardon →
              </a>
            </div>

            <div style="margin-top:18px;font-size:12px;color:#94a3b8;">
              Ez egy TESZT email. Éles környezetben BUY/SELL esetén küldjük.
            </div>
          </div>
        </div>
      </body>
    </html>
    """

    send_email(
        to_email=os.getenv("SMTP_USER"),
        subject=subject,
        body=plain_body,
        html_body=html_body
    )

    return {"ok": True}
@app.get("/api/stats")
def get_stats(symbol: str = Query("OTP.BD")):
    try:
        symbol = _require_symbol(symbol)
    except ValueError:
        return {"error": "Ismeretlen instrument"}

    # --- model meta ---
    meta_path = INSTRUMENTS[symbol]["model_dir"] / "model_meta.json"
    if not meta_path.exists():
        return {"error": "MODEL_META_MISSING", "hint": "Futtasd le a pipeline-t"}

    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    # --- trade-ek száma ---
    sig_path = _signals_csv(symbol)
    signals = pd.read_csv(sig_path)
    trades = int((signals["recommendation"] == "BUY").sum())

    # --- mini backtest ---
    prices = _load_price_df(symbol)
    mini = mini_backtest_90d(signals, prices)
    avg_trade = float(mini.get("avg_trade", 0.0)) if mini else 0.0

    return {
        "symbol": symbol,
        "cv_accuracy": meta["cv_accuracy"],
        "roc_auc": meta["roc_auc"],
        "avg_trade": round(avg_trade, 2),
        "trades": trades,
        "model": meta["model"],
        "generated_at": meta["generated_at"]
    }
