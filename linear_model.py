import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler


class LinearRegressionModel:
    """
    Linear Regression model for stock price prediction.
    Uses a sliding window of past prices as features.
    """

    def __init__(self, lookback: int = 30):
        self.lookback = lookback
        self.model = LinearRegression()
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.is_trained = False

    def _create_sequences(self, data: np.ndarray):
        X, y = [], []
        for i in range(self.lookback, len(data)):
            X.append(data[i - self.lookback:i, 0])
            y.append(data[i, 0])
        return np.array(X), np.array(y)

    def train(self, prices: np.ndarray):
        scaled = self.scaler.fit_transform(prices.reshape(-1, 1))
        X, y = self._create_sequences(scaled)
        self.model.fit(X, y)
        self.is_trained = True

    def predict(self, prices: np.ndarray, forecast_days: int = 30):
        if not self.is_trained:
            raise RuntimeError("Model must be trained before predicting.")

        scaled = self.scaler.transform(prices.reshape(-1, 1))
        X, _ = self._create_sequences(scaled)

        # Historical predictions
        pred_scaled = self.model.predict(X).reshape(-1, 1)
        predictions = self.scaler.inverse_transform(pred_scaled).flatten()

        # Future predictions (recursive)
        window = list(scaled[-self.lookback:].flatten())
        future = []
        for _ in range(forecast_days):
            x_input = np.array(window[-self.lookback:]).reshape(1, -1)
            next_val = self.model.predict(x_input)[0]
            window.append(next_val)
            future.append(next_val)

        future_prices = self.scaler.inverse_transform(
            np.array(future).reshape(-1, 1)
        ).flatten()

        return predictions, future_prices
