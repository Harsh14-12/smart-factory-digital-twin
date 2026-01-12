import pandas as pd
import matplotlib.pyplot as plt

# CMAPSS always has 26 columns
columns = [
    "engine_id", "cycle",
    "setting1", "setting2", "setting3"
]

for i in range(1, 22):
    columns.append(f"sensor_{i}")

# Load using whitespace separator
df = pd.read_csv(
    r"C:\Users\harsh\OneDrive\Desktop\Smart Factory\data\train_FD001.txt",
    sep=r"\s+",
    header=None,
    names=columns
)

# Pick engine 1
engine_data = df[df["engine_id"] == 1]

print(engine_data.head())

# Plot one sensor
plt.plot(engine_data["cycle"], engine_data["sensor_3"])
plt.xlabel("Time (Cycles)")
plt.ylabel("Sensor 3")
plt.title("Engine 1 Health Over Time")
plt.show()
