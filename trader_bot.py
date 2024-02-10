import pandas as pd
from binance.client import Client
import joblib

# Binance API credentials
API_KEY = 'your_api_key'
API_SECRET = 'your_api_secret'

# Initialize Binance client
client = Client(API_KEY, API_SECRET)

# Function to determine candle color
def get_candle_color(open_price, close_price):
    return 1 if close_price > open_price else 0

# Function to get historical candlestick data from Binance
def get_historical_data(symbol, interval, limit):
    klines = client.get_klines(symbol=symbol, interval=interval, limit=limit)
    df = pd.DataFrame(klines, columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close time', 'Quote asset volume', 'Number of trades', 'Taker buy base asset volume', 'Taker buy quote asset volume', 'Ignore'])
    df['Open'] = df['Open'].astype(float)
    df['Close'] = df['Close'].astype(float)
    return df

# Function to predict candle color
def predict_candle_color(model, X):
    return model.predict(X)

# Function to place buy order
def place_buy_order(symbol, quantity):
    order = client.create_order(
        symbol=symbol,
        side=Client.SIDE_BUY,
        type=Client.ORDER_TYPE_MARKET,
        quantity=quantity
    )
    return order

# Function to place sell order
def place_sell_order(symbol, quantity):
    order = client.create_order(
        symbol=symbol,
        side=Client.SIDE_SELL,
        type=Client.ORDER_TYPE_MARKET,
        quantity=quantity
    )
    return order

# Main function
def main():
    # Parameters
    symbol = 'BTCUSDT'
    interval = Client.KLINE_INTERVAL_1MINUTE
    limit = 51  # 50 previous candles + current candle

    # Fetch historical data
    data = get_historical_data(symbol, interval, limit)

    # Create features and target
    X = data[['Open', 'Close', 'High', 'Low']][-50:].values.reshape(1, -1)

    model = joblib.load('xgboost_trained_model.pkl')
    # Predict candle color
    predicted_color = predict_candle_color(model, X)

    # Perform trading based on predicted color
    if predicted_color == 1:  # Green candle, buy
        # Replace 'quantity' with the amount of cryptocurrency you want to buy
        quantity = 0.001  # Example quantity
        place_buy_order(symbol, quantity)
        print("Buy order placed.")
    else:  # Red candle, sell
        # Replace 'quantity' with the amount of cryptocurrency you want to sell
        quantity = 0.001  # Example quantity
        place_sell_order(symbol, quantity)
        print("Sell order placed.")
