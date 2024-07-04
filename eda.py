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
plt.figure(2, figsize=(12, 6))
# Matrix of correlation
M_corr = data.corr()
plt.title("Correlation Heatmap")
sns.heatmap(M_corr, annot=True, cmap='coolwarm')
plt.show()

# Pairplot to show selected features respective to main variable
print("Show pairplot? [y/n] (memory intense)")
choice = input()
if(choice.lower() == 'y'):
    features = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']
    sns.pairplot(data[features])
    plt.show()

# Boxplot/violinplot for crime rate Vs median value
plt.figure(3, figsize=(10,5))
plt.title("Crime Rate vs Median Value")
plt.xlabel("Crime Rate")
plt.ylabel("Median Value")
sns.violinplot(x='CRIM', y='MEDV', data=data)
plt.show()

# Count plot to visualize the RAD feature (index)
plt.figure(4, figsize=(8, 3))
plt.title("Index of accessibility to radial highways")
plt.xlabel("Count")
plt.ylabel("RAD")
sns.countplot(data=data, y='RAD', order=data['RAD'].value_counts().index)
plt.show()