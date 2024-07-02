import pandas as pd

# Reload for Exploratory Data Analysis
data = pd.read_csv('boston.csv')

# Basic prints for understanding
print(data.head())
print(data.info())
print(data.describe())
print(data.isnull().sum())