import os

print("Checking ML models...")

if not os.path.exists("backend/model.pkl"):
    print("Training failure model...")
    os.system("python backend/train_model.py")

if not os.path.exists("backend/rul_model.pkl"):
    print("Training RUL model...")
    os.system("python backend/train_rul_model.py")

print("Models ready.")

