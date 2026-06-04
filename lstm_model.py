import numpy as np
from sklearn.preprocessing import MinMaxScaler


class LSTMModel:
    """
    LSTM (Long Short-Term Memory) deep learning model for stock price prediction.
    Uses TensorFlow/Keras. Lazy-imports TF to avoid slow startup if not needed.
    """

    def __init__(self, lookback: int = 60, epochs: int = 20, batch_size: int = 32):
        self.lookback = lookback
        self.epochs = epochs
        self.batch_size = batch_size
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = None
        self.is_trained = False

    def _build_model(self):
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout

        model = Sequential([
            LSTM(units=50, return_sequences=True, input_shape=(self.lookback, 1)),
            Dropout(0.2),
            LSTM(units=50, return_sequences=False),
            Dropout(0.2),
            Dense(units=25),
            Dense(units=1)
        ])
        model.compile(optimizer="adam", loss="mean_squared_error")
        return model

    def _create_sequences(self, data: np.ndarray):
        X, y = [], []
        for i in range(self.lookback, len(data)):
            X.append(data[i - self.lookback:i, 0])
            y.append(data[i, 0])
        X = np.array(X)
        X = X.reshape((X.shape[0], X.shape[1], 1))
        return X, np.array(y)

    def train(self, prices: np.ndarray):
        scaled = self.scaler.fit_transform(prices.reshape(-1, 1))
        X, y = self._create_sequences(scaled)

        self.model = self._build_model()
        self.model.fit(
            X, y,
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=0,
            validation_split=0.1
        )
        self.is_trained = True

    def predict(self, prices: np.ndarray, forecast_days: int = 30):
        if not self.is_trained:
            raise RuntimeError("Model must be trained before predicting.")

        scaled = self.scaler.transform(prices.reshape(-1, 1))
        X, _ = self._create_sequences(scaled)

        # Historical predictions
        pred_scaled = self.model.predict(X, verbose=0)
        predictions = self.scaler.inverse_transform(pred_scaled).flatten()

        # Future predictions (recursive)
        window = list(scaled[-self.lookback:].flatten())
        future = []
        for _ in range(forecast_days):
            x_input = np.array(window[-self.lookback:]).reshape(1, self.lookback, 1)
            next_val = self.model.predict(x_input, verbose=0)[0][0]
            window.append(next_val)
            future.append(next_val)

        future_prices = self.scaler.inverse_transform(
            np.array(future).reshape(-1, 1)
        ).flatten()

        return predictions, future_prices
