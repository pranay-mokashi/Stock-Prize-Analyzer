import yfinance as yf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

def get_stock_data(symbol):
    data = yf.download(symbol, period="2y")

    if data.empty:
        return None

    return data


def train_lstm(data):
    close_data = data[['Close']].dropna().values   

    
    if len(close_data) < 60:
        return None, None, None

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(close_data)

    X, y = [], []

    for i in range(60, len(scaled_data)):
        X.append(scaled_data[i-60:i])
        y.append(scaled_data[i])

    if len(X) == 0:
        return None, None, None

    X, y = np.array(X), np.array(y)

    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(60, 1)))
    model.add(LSTM(50))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X, y, epochs=3, batch_size=32, verbose=0)

    return model, scaler, X


def predict_price(model, scaler, X):
    if model is None or scaler is None or X is None:
        return None

    last_60 = X[-1].reshape(1, 60, 1)
    pred = model.predict(last_60)

    return scaler.inverse_transform(pred)[0][0]