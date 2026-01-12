import pandas as pd

# Column names as defined by NASA
columns = [
    "engine_id", "cycle",
    "setting1", "setting2", "setting3"
]

# 21 sensor columns
for i in range(1, 22):
    columns.append(f"sensor_{i}")

# Load data
df = pd.read_csv(r"C:\Users\harsh\OneDrive\Desktop\Smart Factory\data\train_FD001.txt",
                 sep=" ", header=None, names=columns)


# Remove empty columns caused by spacing
df = df.dropna(axis=1)

print("Rows:", df.shape[0])
print("Columns:", df.shape[1])
print(df.head())
