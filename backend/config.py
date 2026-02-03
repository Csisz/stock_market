from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

INSTRUMENTS = {
    "OTP.BD": {
        "name": "OTP Bank Nyrt.",
        "csv": BASE_DIR / "data" / "otp_ohlcv_history.csv",
        "model_dir": BASE_DIR / "model" / "otp",
    },
    "MOL.BD": {
        "name": "MOL Nyrt.",
        "csv": BASE_DIR / "data" / "mol_ohlcv_history.csv",
        "model_dir": BASE_DIR / "model" / "mol",
    },
}
