import pandas as pd

# Reads the Boston house prices csv
data = pd.read_csv('boston.csv')

# This command drops the rows which have missing values
# data.fillna could also be used (it would fill said values)
data.dropna(inplace=True)

# This method creates a binary column for each categorical variable (1 if the value corresponds, 0 if else)
# The One-Hot aproach is used so that the machine learning algorithm can access numerical data
data = pd.get_dummies(data)