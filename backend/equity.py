import pandas as pd

HOLD_DAYS = 5

def generate_equity_curve(price_df: pd.DataFrame, signal_df: pd.DataFrame):
    df = price_df.merge(
        signal_df[["date", "signal"]],
        on="date",
        how="left"
    )

    df["signal"].fillna("HOLD", inplace=True)
    df["ret"] = df["close"].pct_change().fillna(0)

    # BUY & HOLD
    df["bh_equity"] = (1 + df["ret"]).cumprod()

    # MODEL equity
    equity = 1.0
    position = 0
    hold = 0
    model_equity = []

    for _, row in df.iterrows():
        if position == 0 and row["signal"] == "BUY":
            position = 1
            hold = HOLD_DAYS

        if position == 1:
            equity *= (1 + row["ret"])
            hold -= 1
            if hold <= 0:
                position = 0
        else:
            equity *= 1.0 

        model_equity.append(equity)

    df["model_equity"] = model_equity

    out = df[["date", "bh_equity", "model_equity"]]
    out.to_csv("backend/model/equity_curve.csv", index=False)

    return out

if __name__ == "__main__":
    print("▶ Equity curve generálás indul...")

    price_df = pd.read_csv("backend/data/otp_ohlcv_history.csv")
    price_df.rename(columns={"Date": "date", "Close": "close"}, inplace=True)

    if "date" not in price_df.columns:
        price_df.reset_index(inplace=True)
        price_df.rename(columns={"index": "date"}, inplace=True)

    price_df["date"] = pd.to_datetime(
        price_df["date"],
        utc=True
    ).dt.tz_localize(None)

    signal_df = pd.read_csv("backend/model/latest_signals.csv")
    signal_df.rename(columns={"Date": "date"}, inplace=True)

    signal_df["date"] = pd.to_datetime(
        signal_df["date"],
        utc=True
    ).dt.tz_localize(None)

    generate_equity_curve(price_df, signal_df)

    print("✅ equity_curve.csv elkészült: backend/model/equity_curve.csv")

