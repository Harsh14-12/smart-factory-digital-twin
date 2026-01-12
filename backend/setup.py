import os
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier

BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "..", "data")

DATA_PATH = os.path.join(DATA_DIR, "train_FD001.txt")
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
RUL_MODEL_PATH = os.path.join(BASE_DIR, "rul_model.pkl")

print("Checking models...")

if os.path.exists(MODEL_PATH) and os.path.exists(RUL_MODEL_PATH):
    print("Models already exist.")
    exit(0)

print("Training models...")

# Column names
columns = ["engine_id", "cycle", "setting1", "setting2", "setting3"]
for i in range(1, 22):
    columns.append(f"sensor_{i}")

df = pd.read_csv(DATA_PATH, sep=r"\s+", header=None, names=columns)

# Compute RUL
max_cycles = df.groupby("engine_id")["cycle"].max()
df["RUL"] = df.apply(lambda x: max_cycles[x.engine_id] - x.cycle, axis=1)

# Failure label
df["fail"] = df["RUL"] <= 30

X = df[[f"sensor_{i}" for i in range(1, 22)]]
y = df["fail"]

# Failure classifier
clf = RandomForestClassifier(n_estimators=150, max_depth=10)
clf.fit(X, y)
joblib.dump(clf, MODEL_PATH)

# RUL regressor
rul = RandomForestClassifier(n_estimators=120)
rul.fit(X, df["RUL"])
joblib.dump(rul, RUL_MODEL_PATH)

print("Models trained and saved.")
