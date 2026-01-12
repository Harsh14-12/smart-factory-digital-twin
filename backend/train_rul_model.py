import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor

# Column names
columns = ["engine_id", "cycle", "setting1", "setting2", "setting3"]
for i in range(1,22):
    columns.append(f"sensor_{i}")

df = pd.read_csv(
    r"C:\Users\harsh\OneDrive\Desktop\Smart Factory\data\train_FD001.txt",
    sep=r"\s+",
    header=None,
    names=columns
)

# True RUL
max_cycles = df.groupby("engine_id")["cycle"].max()
df["RUL"] = df.apply(lambda x: max_cycles[x.engine_id] - x.cycle, axis=1)

X = df[[f"sensor_{i}" for i in range(1,22)]]
y = df["RUL"]

model = RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42)
model.fit(X, y)

joblib.dump(model, r"C:\Users\harsh\OneDrive\Desktop\Smart Factory\backend\rul_model.pkl")
print("RUL model saved")
