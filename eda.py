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

# Heatmap representing the correlation between values
plt.figure(2, figsize=(14, 6))
# Matrix of correlation
M_corr = data.corr()
plt.title("Correlation Heatmap")
sns.heatmap(M_corr, annot=True, cmap='coolwarm')
plt.show()

# Pairplot to show selected features respective to main variable
plt.figure(3)
plt.title("Features -> MEDV")
sns.pairplot(data[features])