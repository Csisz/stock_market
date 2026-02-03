import subprocess
from pathlib import Path

ROOT = Path(__file__).parents[2]

def retrain():
    print("🧠 Retraining models...")
    subprocess.run(["python", "backend/pipeline.py"], cwd=ROOT)

if __name__ == "__main__":
    retrain()
