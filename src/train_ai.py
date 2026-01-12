import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load data
columns = ["engine_id", "cycle", "setting1", "setting2", "setting3"]
for i in range(1, 22):
    columns.append(f"sensor_{i}")

df = pd.read_csv(
    r"C:\Users\harsh\OneDrive\Desktop\Smart Factory\data\train_FD001.txt",
    sep=r"\s+",
    header=None,
    names=columns
)

# Compute RUL
max_cycles = df.groupby("engine_id")["cycle"].max()
df["RUL"] = df.apply(lambda row: max_cycles[row["engine_id"]] - row["cycle"], axis=1)

# Create failure label
df["will_fail_soon"] = df["RUL"] <= 30

# Features = sensors
X = df[[f"sensor_{i}" for i in range(1, 22)]]

# Target = failure risk
y = df["will_fail_soon"]

# Train / test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# Accuracy
accuracy = model.score(X_test, y_test)
print("Smart Factory AI Accuracy:", accuracy)
