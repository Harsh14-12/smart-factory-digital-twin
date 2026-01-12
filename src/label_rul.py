import pandas as pd

columns = ["engine_id", "cycle", "setting1", "setting2", "setting3"]
for i in range(1, 22):
    columns.append(f"sensor_{i}")

df = pd.read_csv(
    r"C:\Users\harsh\OneDrive\Desktop\Smart Factory\data\train_FD001.txt",
    sep=r"\s+",
    header=None,
    names=columns
)

# Find the last cycle for each engine
max_cycles = df.groupby("engine_id")["cycle"].max()

# Compute Remaining Useful Life
df["RUL"] = df.apply(lambda row: max_cycles[row["engine_id"]] - row["cycle"], axis=1)

print(df[["engine_id", "cycle", "RUL"]].head())
print("\nLast cycle of engine 1:")
print(df[df["engine_id"] == 1].tail(3)[["cycle","RUL"]])
