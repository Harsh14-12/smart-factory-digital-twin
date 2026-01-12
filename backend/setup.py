import os
import subprocess

BASE_DIR = os.path.dirname(__file__)

MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
RUL_PATH = os.path.join(BASE_DIR, "rul_model.pkl")

if not os.path.exists(MODEL_PATH):
    subprocess.run(["python", os.path.join(BASE_DIR, "train_model.py")], check=True)

if not os.path.exists(RUL_PATH):
    subprocess.run(["python", os.path.join(BASE_DIR, "train_rul_model.py")], check=True)

print("Models ready.")
