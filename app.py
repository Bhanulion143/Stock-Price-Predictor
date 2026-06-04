from flask import Flask, request, jsonify
from flask_cors import CORS
import yfinance as yf
import numpy as np
import pandas as pd
from models.linear_model import LinearRegressionModel
from models.lstm_model import LSTMModel
from utils.metrics import compute_metrics
import warnings
warnings.filterwarnings("ignore")

app = Flask(__name__)
CORS(app)


def fetch_stock_data(ticker: str, period: str = "2y") -> pd.DataFrame:
    stock = yf.Ticker(ticker)
    df = stock.history(period=period)
    if df.empty:
        raise ValueError(f"No data found for ticker '{ticker}'")
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    df.index = df.index.tz_localize(None)
    return df


@app.route("/api/stock/info", methods=["GET"])
def get_stock_info():
    ticker = request.args.get("ticker", "").upper()
    if not ticker:
        return jsonify({"error": "Ticker symbol is required"}), 400
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        df = fetch_stock_data(ticker, period="1y")
        close_prices = df["Close"]
        return jsonify({
            "ticker": ticker,
            "name": info.get("longName", ticker),
            "current_price": round(float(close_prices.iloc[-1]), 2),
            "52_week_high": round(float(close_prices.max()), 2),
            "52_week_low": round(float(close_prices.min()), 2),
            "volatility": round(float(close_prices.pct_change().std() * np.sqrt(252) * 100), 2),
            "sector": info.get("sector", "N/A"),
            "market_cap": info.get("marketCap", "N/A"),
            "history": [
                {"date": str(date.date()), "close": round(float(price), 2)}
                for date, price in zip(df.index, close_prices)
            ]
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/stock/predict", methods=["POST"])
def predict():
    data = request.get_json()
    ticker = data.get("ticker", "").upper()
    model_type = data.get("model", "linear")
    forecast_days = int(data.get("forecast_days", 30))

    if not ticker:
        return jsonify({"error": "Ticker symbol is required"}), 400
    if forecast_days < 1 or forecast_days > 90:
        return jsonify({"error": "forecast_days must be between 1 and 90"}), 400

    try:
        df = fetch_stock_data(ticker, period="2y")
        close_prices = df["Close"].values

        if model_type == "lstm":
            model = LSTMModel(lookback=60)
        else:
            model = LinearRegressionModel(lookback=30)

        model.train(close_prices)
        predictions, future_predictions = model.predict(close_prices, forecast_days)
        metrics = compute_metrics(close_prices[len(close_prices) - len(predictions):], predictions)

        last_date = df.index[-1]
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1),
                                     periods=forecast_days, freq="B")

        return jsonify({
            "ticker": ticker,
            "model": model_type,
            "metrics": metrics,
            "historical_predictions": [
                {"date": str(df.index[i + (len(close_prices) - len(predictions))].date()),
                 "actual": round(float(close_prices[i + (len(close_prices) - len(predictions))]), 2),
                 "predicted": round(float(predictions[i]), 2)}
                for i in range(len(predictions))
            ],
            "future_predictions": [
                {"date": str(d.date()), "predicted": round(float(p), 2)}
                for d, p in zip(future_dates, future_predictions)
            ]
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "message": "Stock Price Predictor API is running"})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
