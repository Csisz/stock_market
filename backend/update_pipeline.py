import subprocess
from pathlib import Path
from backend.notifications.alerts import notify_if_new_signal

ROOT = Path(__file__).parents[1]

def run():
    print("🚀 Starting update pipeline")

    subprocess.run(["python", "backend/data/update_prices.py"], cwd=ROOT)
    subprocess.run(["python", "backend/model/retrain_all.py"], cwd=ROOT)

    model_path = Path("backend/model") / symbol.split(".")[0].lower() / "latest_signals.csv"
    notify_if_new_signal(symbol, model_path, min_conf=0.55)

    print("✅ Pipeline finished")

if __name__ == "__main__":
    run()
