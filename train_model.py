import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import yfinance as yf
import joblib

# Fetch BTCUSD data from Yahoo Finance
btc_data = yf.download('BTC-USD', start='2024-01-01', end='2024-01-02', interval='1m')

# Function to determine candle color
def get_candle_color(open_price, close_price):
    return 1 if close_price > open_price else 0

# Determine candle color
btc_data['candle_color'] = btc_data.apply(lambda x: get_candle_color(x['Open'], x['Close']), axis=1)

# Create features and target
X = []
y = []
for i in range(len(btc_data) - 51):
    X.append(btc_data[['Open', 'Close', 'High', 'Low']][i:i+50])
    y.append(btc_data['candle_color'][i+50])

X = np.array(X)
y = np.array(y)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Reshape X for compatibility with XGBoost
X_train = X_train.reshape(X_train.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)

# Train XGBoost model
model = XGBClassifier()
model.fit(X_train, y_train)

# Save the trained model
joblib.dump(model, 'xgboost_trained_model.pkl')

# Make predictions
y_pred = model.predict(X_test)

# Evaluate model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")