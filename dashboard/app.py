import streamlit as st
import pandas as pd
import joblib

# ------------------------------
# Page Setup
# ------------------------------
st.set_page_config(layout="wide")
st.title("Smart Factory Digital Twin")

# ------------------------------
# Load Models
# ------------------------------
model = joblib.load(r"C:\Users\harsh\OneDrive\Desktop\Smart Factory\backend\model.pkl")
rul_model = joblib.load(r"C:\Users\harsh\OneDrive\Desktop\Smart Factory\backend\rul_model.pkl")

# ------------------------------
# Load Dataset with Proper Columns
# ------------------------------
columns = ["engine_id", "cycle", "setting1", "setting2", "setting3"]
for i in range(1, 22):
    columns.append(f"sensor_{i}")

data = pd.read_csv(
    r"C:\Users\harsh\OneDrive\Desktop\Smart Factory\data\train_FD001.txt",
    sep=r"\s+",
    header=None,
    names=columns
)

# ------------------------------
# Sidebar Controls
# ------------------------------
st.sidebar.header("Controls")

engine = st.sidebar.slider("Select Machine (Engine ID)", 1, int(data["engine_id"].max()), 1)

engine_data = data[data["engine_id"] == engine].reset_index(drop=True)

step = st.sidebar.slider("Simulation Step", 0, len(engine_data) - 1, 0)

# ------------------------------
# Extract Sensor Snapshot
# ------------------------------
sensor_cols = [f"sensor_{i}" for i in range(1, 22)]
sensors = engine_data.loc[[step], sensor_cols]

# ------------------------------
# Predictions
# ------------------------------
failure_prob = model.predict_proba(sensors)[0][1]
predicted_rul = rul_model.predict(sensors)[0]

# ------------------------------
# Layout
# ------------------------------
col1, col2 = st.columns(2)

# ------------------------------
# Health Panel
# ------------------------------
with col1:
    st.subheader(f"Machine {engine} Health")

    if failure_prob > 0.7:
        st.error(f"⚠ High Failure Risk: {failure_prob:.2f}")
    elif failure_prob > 0.4:
        st.warning(f"Machine Degrading: {failure_prob:.2f}")
    else:
        st.success(f"Machine Healthy: {1 - failure_prob:.2f}")

    st.metric("Failure Probability", f"{failure_prob * 100:.2f}%")
    st.metric("Estimated Remaining Life", f"{int(predicted_rul)} cycles")

# ------------------------------
# Live Sensor Plot (REAL degradation)
# ------------------------------
with col2:
    st.subheader("Live Sensor Stream (Sensor 3 – Degradation Indicator)")

    history = engine_data.iloc[: step + 1].copy()
    history["time"] = history.index

    st.line_chart(history.set_index("time")["sensor_3"])

# ------------------------------
# Current Snapshot
# ------------------------------
st.subheader("Current Sensor Snapshot")
st.dataframe(sensors)
st.download_button(
    label="Download Snapshot as CSV",
    data=sensors.to_csv(index=False).encode("utf-8"),
    file_name=f"engine_{engine}_step_{step}_snapshot.csv",
    mime="text/csv",
)