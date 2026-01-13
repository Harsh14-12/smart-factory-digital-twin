import os
import subprocess
import sys

BASE_DIR = os.path.dirname(__file__)

MODEL = os.path.join(BASE_DIR, "model.pkl")
RUL = os.path.join(BASE_DIR, "rul_model.pkl")

python = sys.executable   # THIS IS CRITICAL

if not os.path.exists(MODEL):
    subprocess.run([python, os.path.join(BASE_DIR, "train_model.py")], check=True)

if not os.path.exists(RUL):
    subprocess.run([python, os.path.join(BASE_DIR, "train_rul_model.py")], check=True)

print("Models ready.")
