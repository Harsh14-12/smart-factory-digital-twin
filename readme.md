# Smart Factory Digital Twin

An AI-powered digital twin system for predictive maintenance of industrial machines using NASA’s CMAPSS dataset.

This system monitors machine sensor data in real time, predicts failure probability, and estimates remaining useful life (RUL) to enable proactive maintenance decisions.

---

## 🚀 Features
- Real-time machine health monitoring
- Failure probability prediction using machine learning
- Remaining Useful Life (RUL) forecasting
- Live sensor dashboards
- Industrial digital twin visualization

---

## 📊 Dataset
This project uses NASA’s CMAPSS turbofan engine degradation dataset, which simulates real industrial sensor data from aircraft engines.

---

## 🧠 Machine Learning Models
Two models are used:
- Failure Classification Model (Random Forest)
- RUL Regression Model (Random Forest)

These models learn degradation patterns from historical sensor data.

---

## 🖥️ Digital Twin Dashboard
Built using Streamlit. It allows users to:
- Select any machine
- Move forward in time
- View sensor degradation
- See failure risk and remaining life

---

## ⚙️ How to Run

### Install dependencies
```bash
pip install pandas numpy scikit-learn streamlit joblib
