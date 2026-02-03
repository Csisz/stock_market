import os
import json
from pathlib import Path

import pandas as pd

from backend.notifications.emailer import send_email

# =====================
# Konstansok / utak
# =====================

DATA_DIR = Path("backend/data")
SUBS_FILE = DATA_DIR / "subscribers.json"
STATE_FILE = DATA_DIR / "last_alert_state.json"

BASE_URL = os.getenv("BASE_URL", "http://127.0.0.1:8000")


# =====================
# Helper függvények
# =====================

def _badge_color(rec: str) -> str:
    return "#16a34a" if rec == "BUY" else "#dc2626"


def _load_subscribers():
    if not SUBS_FILE.exists():
        return []
    return json.loads(SUBS_FILE.read_text(encoding="utf-8") or "[]")


def _load_state():
    if not STATE_FILE.exists():
        return {}
    return json.loads(STATE_FILE.read_text(encoding="utf-8") or "{}")


def _save_state(state: dict):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=2), encoding="utf-8")


# =====================
# Fő logika
# =====================

def notify_if_new_signal(symbol: str, model_path: Path, min_conf: float = 0.0):
    """
    Email értesítés küldése, ha:
    - az utolsó jelzés BUY vagy SELL
    - új (date + rec kombináció alapján)
    - megfelel a min_conf küszöbnek
    """

    subs = _load_subscribers()
    if not subs:
        return

    if not model_path.exists():
        return

    df = pd.read_csv(model_path)
    if df.empty:
        return

    last = df.iloc[-1]

    rec = str(last.get("recommendation", "")).upper()
    p_up = last.get("p_up", None)
    date = str(last.get("Date", last.get("date", "")))
    updated_at = str(last.get("generated_at", last.get("updated_at", "")))

    # Csak BUY / SELL
    if rec not in ("BUY", "SELL"):
        return

    # Confidence küszöb
    if p_up is not None and min_conf > 0.0:
        try:
            if float(p_up) < float(min_conf):
                return
        except Exception:
            pass

    # Spam védelem (ugyanazt ne küldje újra)
    state = _load_state()
    key = symbol
    current = f"{date}|{rec}"
    if state.get(key) == current:
        return

    # =====================
    # Email tartalom
    # =====================

    subject = f"[StockTrade] {symbol} jelzés: {rec}"

    plain_body = (
        f"Új napi jelzés érkezett:\n\n"
        f"Instrumentum: {symbol}\n"
        f"Dátum: {date}\n"
        f"Ajánlás: {rec}\n"
        f"p_up (5 nap): {p_up}\n"
        f"Model frissítve: {updated_at}\n\n"
        f"Dashboard: {BASE_URL}\n\n"
        f"Ez egy automatikus értesítés."
    )

    color = _badge_color(rec)

    html_body = f"""
<!doctype html>
<html>
  <body style="margin:0;padding:0;background:#0b1220;font-family:Arial,Helvetica,sans-serif;">
    <div style="max-width:640px;margin:0 auto;padding:24px;">
      <div style="background:#0f1a2e;border:1px solid #1f2a44;border-radius:16px;padding:20px;color:#e5e7eb;">
        <div style="display:flex;align-items:center;justify-content:space-between;gap:12px;">
          <div style="font-size:18px;font-weight:700;">
            StockTrade – napi jelzés
          </div>
          <div style="background:{color};color:white;padding:6px 12px;border-radius:999px;
                      font-size:12px;font-weight:700;">
            {rec}
          </div>
        </div>

        <div style="margin-top:14px;font-size:14px;line-height:1.6;color:#cbd5e1;">
          <div><b>Instrumentum:</b> {symbol}</div>
          <div><b>Dátum:</b> {date}</div>
          <div><b>p_up (5 nap):</b> {p_up}</div>
          <div><b>Model frissítve:</b> {updated_at}</div>
        </div>

        <div style="margin-top:18px;">
          <a href="{BASE_URL}"
             style="display:inline-block;background:#2563eb;color:white;
                    text-decoration:none;padding:10px 16px;
                    border-radius:12px;font-weight:700;font-size:14px;">
            Megnyitás a dashboardon →
          </a>
        </div>

        <div style="margin-top:18px;font-size:12px;color:#94a3b8;">
          Ez egy automatikus értesítés. Leiratkozni a dashboardon tudsz.
        </div>
      </div>

      <div style="margin-top:12px;text-align:center;font-size:11px;color:#64748b;">
        © StockTrade
      </div>
    </div>
  </body>
</html>
"""

    # =====================
    # Küldés
    # =====================

    for email in subs:
        send_email(
            to_email=email,
            subject=subject,
            body=plain_body,
            html_body=html_body
        )

    # State mentése
    state[key] = current
    _save_state(state)
