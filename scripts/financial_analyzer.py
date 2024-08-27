import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from textblob import TextBlob
from sklearn.feature_extraction.text import CountVectorizer

class FinancialNewsAnalyzer:
    def __init__(self, filepath):
        """
        Initialize the FinancialNewsAnalyzer with the path to the dataset.
        """
        self.filepath = filepath
        self.data = None
        self.headlines = None
        self.publishers = None
        self.dates = None

    def load_data(self):
        """
        Load the dataset and initialize key components.
        """
        self.data = pd.read_csv(self.filepath)
        self.headlines = self.data['headline']
        self.publishers = self.data['publisher']
        self.dates = pd.to_datetime(self.data['date'],format='ISO8601')

        print("Data loaded successfully.")
        print(self.data.info())

    def perform_descriptive_stats(self):
        """
        Perform basic descriptive statistics on the dataset.
        """
        # Headline length analysis
        self.data['headline_length'] = self.headlines.apply(len)
        print("Headline Length Stats:\n", self.data['headline_length'].describe())

        # Publisher frequency
        publisher_counts = self.publishers.value_counts()
        print("Publisher Frequency:\n", publisher_counts)

    def perform_text_analysis(self):
        """
        Perform text analysis including sentiment analysis and keyword extraction.
        """
        # Sentiment analysis
        self.data['sentiment'] = self.headlines.apply(lambda x: TextBlob(x).sentiment.polarity)
        print("Sentiment Analysis Stats:\n", self.data['sentiment'].describe())

        # Keyword extraction
        vectorizer = CountVectorizer(max_features=10)
        word_counts = vectorizer.fit_transform(self.headlines)
        print("Top Keywords:\n", vectorizer.get_feature_names_out())

    def perform_time_series_analysis(self):
        """
        Perform time series analysis on the publication dates.
        """
        # Articles published per day
        daily_articles = self.dates.dt.date.value_counts().sort_index()
        print("Daily Articles Published:\n", daily_articles)

        # Plotting the publication frequency
        daily_articles.plot(kind='line', title='Daily Articles Published')
        plt.show()

    def perform_publisher_analysis(self):
        """
        Analyze the distribution of publishers.
        """
        # Top 10 publishers by article count
        top_publishers = self.publishers.value_counts().head(10)
        print("Top 10 Publishers:\n", top_publishers)

        # Plotting top publishers
        top_publishers.plot(kind='bar', title='Top 10 Publishers')
        plt.show()

    def visualize_results(self):
        """
        Visualize the results from the various analyses.
        """
        # Sentiment distribution plot
        plt.figure(figsize=(10, 6))
        plt.hist(self.data['sentiment'], bins=20, color='skyblue', edgecolor='black')
        plt.title('Sentiment Distribution')
        plt.xlabel('Sentiment')
        plt.ylabel('Frequency')
        plt.show()

    def run_eda(self):
        """
        Run the entire EDA process.
        """
        self.load_data()
        self.perform_descriptive_stats()
        self.perform_text_analysis()
        self.perform_time_series_analysis()
        self.perform_publisher_analysis()
        self.visualize_results()



