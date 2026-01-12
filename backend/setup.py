import os
import subprocess

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BACKEND_DIR = os.path.join(BASE_DIR, "backend")

MODEL_PATH = os.path.join(BACKEND_DIR, "model.pkl")
RUL_PATH = os.path.join(BACKEND_DIR, "rul_model.pkl")

print("Checking ML models...")

if not os.path.exists(MODEL_PATH):
    print("Training failure model...")
    subprocess.run(["python", os.path.join(BACKEND_DIR, "train_model.py")])

if not os.path.exists(RUL_PATH):
    print("Training RUL model...")
    subprocess.run(["python", os.path.join(BACKEND_DIR, "train_rul_model.py")])

print("Models ready.")
