# Stock-Price-Predictor
# Stock Price Predictor 📈

A **full-stack** stock prediction application with:
- React frontend for visualization
- Machine learning backend (Python) for price forecasting
- Support for multiple models (Linear Regression, LSTM)



![React](https://img.shields.io/badge/React-18.2+-blue)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0+-red)
![Pandas](https://img.shields.io/badge/Pandas-1.3+-brightgreen)






## Features
- **Real-time stock data** (via Yahoo Finance API)
- **Interactive charts** (Plotly/Recharts)
- **Multiple prediction models**:
  - Linear Regression (baseline)
  - LSTM (deep learning)
- **Key metrics dashboard**:
  - Current price, 52-week high/low
  - Volatility analysis
- **Responsive design** (mobile-friendly)

## Tech Stack
| Frontend               | Backend/ML           |
|------------------------|----------------------|
| React 18               | Python 3.8+          |
| Tailwind CSS           | TensorFlow/Keras     |
| Recharts               | scikit-learn         |
| Vite (build tool)      | FastAPI/Flask        |

Project Structure
text
stock-price-predictor/
├── frontend/               # React app
│   ├── src/
│   │   ├── components/     # React components
│   │   ├── utils/          # Data generators
│   │   └── App.jsx         # Main entry
├── backend/                # Python ML API
│   ├── models/             # Trained models
│   ├── app.py              # Flask/FastAPI server
│   └── requirements.txt
├── screenshots/            # App previews
└── README.md
Usage
Search stocks by symbol (e.g., AAPL, TSLA)

View historical data in interactive chart

Run predictions:
Select model (Linear Regression/LSTM)

Choose forecast period

Analyze metrics:

Price trends

Volatility indicators
