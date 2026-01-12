
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier

# Column names
columns = ["engine_id", "cycle", "setting1", "setting2", "setting3"]
for i in range(1, 22):
    columns.append(f"sensor_{i}")

# Load data
df = pd.read_csv(
    r"C:\Users\harsh\OneDrive\Desktop\Smart Factory\data\train_FD001.txt",
    sep=r"\s+",
    header=None,
    names=columns
)

# Compute RUL
max_cycles = df.groupby("engine_id")["cycle"].max()
df["RUL"] = df.apply(lambda x: max_cycles[x.engine_id] - x.cycle, axis=1)

# Failure label
df["fail"] = df["RUL"] <= 30

# Features and target
X = df[[f"sensor_{i}" for i in range(1, 22)]]
y = df["fail"]

# Train model
model = RandomForestClassifier(n_estimators=150, max_depth=10)
model.fit(X, y)

# Save model
joblib.dump(model, "model.pkl")
print("Model trained and saved.")
