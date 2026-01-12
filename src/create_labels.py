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

max_cycles = df.groupby("engine_id")["cycle"].max()
df["RUL"] = df.apply(lambda row: max_cycles[row["engine_id"]] - row["cycle"], axis=1)

# Create failure label
df["will_fail_soon"] = df["RUL"] <= 30

print(df[["engine_id","cycle","RUL","will_fail_soon"]].head(40))
