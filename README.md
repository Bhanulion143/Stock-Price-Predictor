# Stock Price Predictor 📈

A full-stack stock price prediction application using Machine Learning (Linear Regression + LSTM) with a Python/Flask backend and React frontend.

![Python](https://img.shields.io/badge/Python-3.8+-blue)
![Flask](https://img.shields.io/badge/Flask-3.0-green)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.17-orange)
![scikit--learn](https://img.shields.io/badge/scikit--learn-1.5-blue)
![React](https://img.shields.io/badge/React-18.2-61DAFB)

---

## Features

- **Real-time stock data** via Yahoo Finance API (no API key needed)
- **Two prediction models:**
  - Linear Regression (fast, interpretable baseline)
  - LSTM Deep Learning (captures long-term temporal patterns)
- **Model evaluation metrics:** RMSE, MAE, R² Score, MAPE
- **30/60/90-day future forecasting**
- **Interactive React dashboard** with Recharts visualizations
- **Responsive design** — works on mobile and desktop

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend API | Python 3.8+, Flask, Flask-CORS |
| ML Models | scikit-learn, TensorFlow 2.x / Keras |
| Data Source | yfinance (Yahoo Finance) |
| Data Processing | NumPy, Pandas |
| Frontend | React 18, Vite, Tailwind CSS, Recharts |

---

## Project Structure

```
stock-price-predictor/
├── backend/
│   ├── app.py                  # Flask API server
│   ├── requirements.txt
│   ├── models/
│   │   ├── linear_model.py     # Linear Regression predictor
│   │   └── lstm_model.py       # LSTM deep learning predictor
│   └── utils/
│       └── metrics.py          # RMSE, MAE, R2, MAPE computation
├── frontend/
│   ├── src/
│   │   ├── components/         # React components
│   │   └── App.jsx
│   └── package.json
└── notebooks/
    └── stock_predictor_analysis.ipynb   # EDA + model training notebook
```

---

## Setup & Installation

### Backend

```bash
cd backend
pip install -r requirements.txt
python app.py
```

The API will start at `http://localhost:5000`

### Frontend

```bash
cd frontend
npm install
npm run dev
```

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/health` | Health check |
| GET | `/api/stock/info?ticker=AAPL` | Get stock info + 1yr history |
| POST | `/api/stock/predict` | Run prediction model |

### Example Prediction Request

```json
POST /api/stock/predict
{
  "ticker": "AAPL",
  "model": "lstm",
  "forecast_days": 30
}
```

### Example Response

```json
{
  "ticker": "AAPL",
  "model": "lstm",
  "metrics": {
    "rmse": 3.42,
    "mae": 2.81,
    "r2_score": 0.97,
    "mape": 1.63
  },
  "future_predictions": [
    {"date": "2025-06-02", "predicted": 192.45},
    ...
  ]
}
```

---

## Model Performance (AAPL — 2yr data)

| Model | RMSE | MAE | R² Score | MAPE |
|---|---|---|---|---|
| Linear Regression | ~4.2 | ~3.1 | ~0.96 | ~1.8% |
| LSTM | ~2.8 | ~2.1 | ~0.98 | ~1.1% |

---

## Notebook

The `notebooks/stock_predictor_analysis.ipynb` notebook contains:
- EDA (Exploratory Data Analysis) on historical stock data
- Training both models step-by-step
- Visualizing predictions vs actual prices
- Comparing model metrics
- 30-day future forecast plots

---

## Author

**Bhanu Prakash**
B.Tech — Artificial Intelligence & Machine Learning

---

## Disclaimer

This project is for educational purposes only. Stock predictions are not financial advice.
