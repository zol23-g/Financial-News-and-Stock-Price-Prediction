import unittest
import pandas as pd
from io import StringIO
import sys
import os
# sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '../scripts')))
# from financial_analyzer import FinancialNewsAnalyzer
from scripts.financial_analyzer import FinancialNewsAnalyzer
class TestFinancialNewsAnalyzer(unittest.TestCase):

    def setUp(self):
        """
        Set up a small sample dataset for testing.
        """
        data = """headline,publisher,date
        "Company A hits new high","Publisher 1","2020-05-22 00:00:00"
        "Company B stock plunges","Publisher 2","2020-05-23 00:00:00"
        "Company C reports earnings","Publisher 1","2020-05-24 00:00:00"
        "Company A under investigation","Publisher 3","2020-05-25 00:00:00"
        "Company B rebounds","Publisher 2","2020-05-26 00:00:00"
        """

        self.sample_data = StringIO(data)
        self.analyzer = FinancialNewsAnalyzer(self.sample_data)

    def test_load_data(self):
        """
        Test that the data is loaded correctly.
        """
        self.analyzer.load_data()
        self.assertEqual(len(self.analyzer.data), 5)
        self.assertIn('headline', self.analyzer.data.columns)
        self.assertIn('publisher', self.analyzer.data.columns)
        self.assertIn('date', self.analyzer.data.columns)

    def test_perform_descriptive_stats(self):
        """
        Test the descriptive statistics function.
        """
        self.analyzer.load_data()
        self.analyzer.perform_descriptive_stats()
        self.assertIn('headline_length', self.analyzer.data.columns)
        self.assertEqual(self.analyzer.data['headline_length'].mean(), 33.8)  # Example assertion

    def test_perform_text_analysis(self):
        """
        Test the text analysis function.
        """
        self.analyzer.load_data()
        self.analyzer.perform_text_analysis()
        self.assertIn('sentiment', self.analyzer.data.columns)
        self.assertGreaterEqual(self.analyzer.data['sentiment'].min(), -1)
        self.assertLessEqual(self.analyzer.data['sentiment'].max(), 1)

    def test_perform_time_series_analysis(self):
        """
        Test the time series analysis function.
        """
        self.analyzer.load_data()
        self.analyzer.perform_time_series_analysis()
        # Since this method doesn't return values but plots, ensure that dates were processed.
        self.assertEqual(len(self.analyzer.dates.unique()), 5)

    def test_perform_publisher_analysis(self):
        """
        Test the publisher analysis function.
        """
        self.analyzer.load_data()
        self.analyzer.perform_publisher_analysis()
        top_publishers = self.analyzer.publishers.value_counts().head(3)
        self.assertEqual(len(top_publishers), 3)
        self.assertEqual(top_publishers.iloc[0], 2)  # The top publisher appears twice in the sample data

    def test_run_eda(self):
        """
        Test the full EDA process.
        """
        # Simply run to ensure no errors and all methods are called.
        self.analyzer.run_eda()

if __name__ == "__main__":
    unittest.main()
