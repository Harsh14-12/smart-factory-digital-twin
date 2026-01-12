import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
import os

BASE_DIR = os.path.dirname(__file__)
DATA_PATH = os.path.abspath(os.path.join(BASE_DIR, "..", "data", "train_FD001.txt"))
MODEL_PATH = os.path.abspath(os.path.join(BASE_DIR, "rul_model.pkl"))

columns = ["engine_id", "cycle", "setting1", "setting2", "setting3"]
for i in range(1, 22):
    columns.append(f"sensor_{i}")

df = pd.read_csv(DATA_PATH, sep=r"\s+", header=None, names=columns)

max_cycles = df.groupby("engine_id")["cycle"].max()
df["RUL"] = df["engine_id"].map(max_cycles) - df["cycle"]

X = df[[f"sensor_{i}" for i in range(1, 22)]]
y = df["RUL"]

model = RandomForestRegressor(n_estimators=150, max_depth=10, random_state=42)
model.fit(X, y)

joblib.dump(model, MODEL_PATH)
print("RUL model saved")
