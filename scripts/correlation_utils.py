import pandas as pd
from textblob import TextBlob

def process_stock_data(stock_filepath, news_data):
    """
    Processes a single stock data file and calculates the correlation between 
    average daily sentiment and daily stock returns.

    Parameters:
    - stock_filepath: str, path to the stock data CSV file.
    - news_data: DataFrame, pre-loaded news data with a 'Date' column.

    Returns:
    - correlation: float, the Pearson correlation between average daily sentiment and daily stock returns.
    - merged_data: DataFrame, the merged data with sentiment and stock returns.
    """
    # Load the stock data
    stock_data = pd.read_csv(stock_filepath, parse_dates=['Date'])
    stock_data['Date'] = pd.to_datetime(stock_data['Date'], format='ISO8601')
    
    # Convert date columns to datetime and localize to UTC
    stock_data['Date'] = stock_data['Date'].dt.tz_localize('Etc/GMT').dt.tz_convert('UTC')
    
    # Normalize dates (e.g., round news timestamps to the nearest trading day)
    news_data['Date'] = news_data['Date'].dt.floor('D')
    
    # Merge the stock data with the news data
    merged_data = pd.merge(stock_data, news_data, on='Date', how='inner')

    # Sentiment analysis on headlines
    merged_data['polarity'] = merged_data['headline'].apply(lambda x: TextBlob(x).sentiment.polarity)
    merged_data['sentiment'] = merged_data['polarity'].apply(lambda p: 'positive' if p > 0 else 'negative' if p < 0 else 'neutral')

    # Calculate daily stock returns
    merged_data['Daily_Returns'] = merged_data['Close'].pct_change()

    # Aggregate daily sentiment scores
    daily_sentiment = merged_data.groupby('Date')['polarity'].mean().reset_index()
    daily_sentiment.columns = ['Date', 'Average_Sentiment']

    # Merge the average sentiment back to the merged_data DataFrame
    merged_data = pd.merge(merged_data, daily_sentiment, on='Date', how='inner')

    # Calculate the correlation between average sentiment and daily returns
    correlation = merged_data['Average_Sentiment'].corr(merged_data['Daily_Returns'])
    
    return correlation, merged_data

# import pandas as pd
# import spacy
# from spacytextblob.spacytextblob import SpacyTextBlob

# # Load SpaCy model and add SpacyTextBlob to the pipeline
# nlp = spacy.load("en_core_web_sm")
# nlp.add_pipe('spacytextblob')

# def process_stock_data(stock_filepath, news_data):
#     """
#     Processes a single stock data file and calculates the correlation between 
#     average daily sentiment and daily stock returns.

#     Parameters:
#     - stock_filepath: str, path to the stock data CSV file.
#     - news_data: DataFrame, pre-loaded news data with a 'Date' column.

#     Returns:
#     - correlation: float, the Pearson correlation between average daily sentiment and daily stock returns.
#     - merged_data: DataFrame, the merged data with sentiment and stock returns.
#     """
#     # Load the stock data
#     stock_data = pd.read_csv(stock_filepath, parse_dates=['Date'])
#     stock_data['Date'] = pd.to_datetime(stock_data['Date'], format='ISO8601')
    
#     # Convert date columns to datetime and localize to UTC
#     stock_data['Date'] = stock_data['Date'].dt.tz_localize('Etc/GMT').dt.tz_convert('UTC')
    
#     # Normalize dates (e.g., round news timestamps to the nearest trading day)
#     news_data['Date'] = news_data['Date'].dt.floor('D')
    
#     # Merge the stock data with the news data
#     merged_data = pd.merge(stock_data, news_data, on='Date', how='inner')

#     # Sentiment analysis on headlines using SpaCy with SpacyTextBlob
#     def get_sentiment(text):
#         doc = nlp(text)
#         return doc._.polarity
    
#     merged_data['polarity'] = merged_data['headline'].apply(get_sentiment)
#     merged_data['sentiment'] = merged_data['polarity'].apply(lambda p: 'positive' if p > 0 else 'negative' if p < 0 else 'neutral')

#     # Calculate daily stock returns
#     merged_data['Daily_Returns'] = merged_data['Close'].pct_change()

#     # Aggregate daily sentiment scores
#     daily_sentiment = merged_data.groupby('Date')['polarity'].mean().reset_index()
#     daily_sentiment.columns = ['Date', 'Average_Sentiment']

#     # Merge the average sentiment back to the merged_data DataFrame
#     merged_data = pd.merge(merged_data, daily_sentiment, on='Date', how='inner')

#     # Calculate the correlation between average sentiment and daily returns
#     correlation = merged_data['Average_Sentiment'].corr(merged_data['Daily_Returns'])
    
#     return correlation, merged_data
