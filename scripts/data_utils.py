import os
import pandas as pd
import talib
import pynance as pn
import matplotlib.pyplot as plt

def load_data(filepath):
    """
    Load the dataset from the given file path.
    """
    data = pd.read_csv(filepath)
    print("Data loaded successfully.")
    return data

def process_stock_data(filepath):
    """
    Process the stock data by calculating technical indicators
    and performing financial analysis.
    """
    # Load the data
    df = load_data(filepath)

    # Perform data quality check
    data_quality_check(df)

    # Convert 'Date' to datetime and set it as the index
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)

    # Calculate technical indicators using TA-Lib
    df['SMA_20'] = talib.SMA(df['Close'], timeperiod=20)
    df['SMA_50'] = talib.SMA(df['Close'], timeperiod=50)
    df['EMA_20'] = talib.EMA(df['Close'], timeperiod=20)
    df['RSI_14'] = talib.RSI(df['Close'], timeperiod=14)
    df['MACD'], df['MACD_Signal'], df['MACD_Hist'] = talib.MACD(
        df['Close'], fastperiod=12, slowperiod=26, signalperiod=9
    )

    # Calculate financial metrics using Pandas
    df['Daily_Returns'] = df['Close'].pct_change()
    df['Volatility'] = df['Daily_Returns'].rolling(window=20).std()

    # Visualize the results
    stock_name = os.path.basename(filepath).replace('.csv', '')
    visualize_data(df, stock_name)

    return df

def visualize_data(df, stock_name):
    """
    Visualize the technical indicators and financial metrics.
    """
    plt.figure(figsize=(14, 10))

    # Plot Close Price with Moving Averages
    plt.subplot(3, 1, 1)
    plt.plot(df['Close'], label=f'{stock_name} Close Price', color='black')
    plt.plot(df['SMA_20'], label='20-day SMA', color='blue')
    plt.plot(df['SMA_50'], label='50-day SMA', color='red')
    plt.plot(df['EMA_20'], label='20-day EMA', color='green')
    plt.title(f'{stock_name} Stock Price with Moving Averages')
    plt.legend()

    # Plot RSI
    plt.subplot(3, 1, 2)
    plt.plot(df['RSI_14'], label='14-day RSI', color='purple')
    plt.axhline(70, color='red', linestyle='--', label='Overbought')
    plt.axhline(30, color='green', linestyle='--', label='Oversold')
    plt.title(f'{stock_name} RSI')
    plt.legend()

    # Plot MACD
    plt.subplot(3, 1, 3)
    plt.plot(df['MACD'], label='MACD', color='blue')
    plt.plot(df['MACD_Signal'], label='MACD Signal', color='red')
    plt.bar(df.index, df['MACD_Hist'], label='MACD Histogram', color='gray')
    plt.title(f'{stock_name} MACD')
    plt.legend()

    plt.tight_layout()
    plt.show()

def process_all_stocks(data_dir):
    """
    Process all stock data files in the given directory.
    """
    for filename in os.listdir(data_dir):
        if filename.endswith('.csv'):
            print(f'Processing {filename}...')
            filepath = os.path.join(data_dir, filename)
            df = process_stock_data(filepath)

            # # Save the processed data with indicators
            # output_filepath = os.path.join(data_dir, f'processed_{filename}')
            # df.to_csv(output_filepath)
            # print(f'Saved processed data to {output_filepath}.\n')