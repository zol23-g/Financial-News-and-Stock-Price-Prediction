# data_utils.py

import pandas as pd

def load_data(filepath):
    """
    Load the dataset from the given file path.
    """
    data = pd.read_csv(filepath)
    print("Data loaded successfully.")
    return data

def data_quality_check(data):
    """
    Perform basic data quality checks on the dataset.
    """
    # Checking for missing values
    missing_values = data.isnull().sum()
    print("Missing values in each column:\n", missing_values)
    
    # Checking for duplicates
    duplicate_rows = data.duplicated().sum()
    print(f"Number of duplicate rows: {duplicate_rows}")
    
    # Checking data types
    print("Data types:\n", data.dtypes)
    
    return missing_values, duplicate_rows

