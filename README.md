# Crypto Trading Bot with XGBoost

This project implements a cryptocurrency trading bot that uses an XGBoost model to predict the next candle color (green or red) and executes buy or sell orders accordingly on the Binance exchange.

## Usage
- Clone this repository.
- Install the required dependencies using `pip install -r requirements.txt`.
- Train the XGBoost model using `train_model.py`.
- Replace `your_api_key` and `your_api_secret` in `trader_bot.py` with your Binance API credentials.
- Run `run.py` to start the trading bot.

## Requirements
- Python 3.x
- pandas
- numpy
- scikit-learn
- xgboost
- yfinance
- python-binance

## How to Train the Model
- Modify `train_model.py` to fetch historical data from the desired source.
- Run `train_model.py` to train the model and save it as `xgboost_trained_model.pkl`.

## How to Run the Trader Bot
- Ensure that you have set up your Binance API credentials in `trader_bot.py`.
- Run `run.py` to start the bot. It will fetch historical data, make predictions, and execute trades every minute.
