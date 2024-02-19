import warnings

import requests
import pandas as pd
import numpy as np
import joblib
import csv
from datetime import datetime


def request_data():
    # CoinGecko API endpoint for Bitcoin
    url = 'https://api.coingecko.com/api/v3/coins/bitcoin'

    response = requests.get(url)
    data = response.json()

    market_cap = data['market_data']['market_cap']['usd']
    high_price = data['market_data']['high_24h']['usd']
    low_price = data['market_data']['low_24h']['usd']
    return market_cap, high_price, low_price


def import_data(filename):
    df = pd.read_csv(filename)
    return df


def get_open_price():
    global open_price
    open_price = float(input("Enter Open Price: "))


def get_datetime():
    dt = datetime.now()
    return dt


def add_data():
    market_cap, high_price, low_price = request_data()
    get_open_price()
    df = import_data('new_input_df.csv')
    column_names = ['Market Cap', 'Low', 'High', 'Open*']
    df = df[column_names]
    formatting(df=df, column_names=column_names, date=False)
    new_data = {'Market Cap': market_cap, 'Low': low_price, 'High': high_price, 'Open*': open_price}
    df = df._append(new_data, ignore_index=True)

    # Calculating Moving Average
    window_size = 10
    df['Moving Average'] = df['Open*'].rolling(window=window_size).mean()

    # Calculating Bollinger Bands
    std_dev = 2
    # Calculate the rolling mean and standard deviation
    rolling_mean = df['Open*'].rolling(window_size).mean()
    rolling_std = df['Open*'].rolling(window_size).std()

    # Calculate the upper and lower Bollinger Bands
    upper_band = rolling_mean + (rolling_std * std_dev)

    df['Upper Band'] = upper_band
    # print(df)
    log_df = np.log(df)
    # print(log_df)
    return log_df


def extract_data():
    log_df = add_data()
    input_df = log_df.tail(1)
    input_array = np.array(input_df)
    print(input_array)
    return input_array


def calculate_ma(prices, window=10):
    """
    Calculate Moving Average. Specify Window of the days that you want to calculate moving average of.
    """
    ma = prices.rolling(window=window).mean()
    print('Moving Average: ', ma)
    return ma


def formatting(df, column_names, date=True):
    for i in column_names:
        df[i] = df[i].str.replace('$', '')
        df[i] = df[i].str.replace(',', '')
        df[i] = pd.to_numeric(df[i], errors='coerce')

    if date:
        df['Date'] = pd.to_datetime(df['Date'])
        df['Year'] = df['Date'].dt.year
        df['Day'] = df['Date'].dt.day
        df['Month'] = df['Date'].dt.month
        df = df.drop('Date', axis=1)

    df = df.iloc[::-1]
    df = df.reset_index(drop=True)
    return df


def make_predictions():
    # Loading the model
    model = joblib.load("model_rfr.joblib")

    data = extract_data()
    predictions = model.predict(data)
    closing = np.exp(predictions)
    print("Closing: ", closing)

    file_path = 'predicted_data.csv'
    market_cap, high_price, low_price = request_data()
    dt = get_datetime()
    with open(file_path, 'a', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow([dt, open_price, high_price, low_price, 'N/A', 'N/A', market_cap, closing])


if __name__ == '__main__':
    make_predictions()
