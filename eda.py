import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Reload for Exploratory Data Analysis
data = pd.read_csv('boston.csv')

# Basic prints for understanding
print(data.head())
print(data.info())
print(data.describe())
print(data.isnull().sum())

# Median Value [MEDV] distribution visualized
plt.figure(1, figsize=(6, 4))
plt.title("MEDV")
plt.xlabel("Median Value")
plt.ylabel("Frequency")
sns.histplot(data['MEDV'], kde=True)
plt.show()
