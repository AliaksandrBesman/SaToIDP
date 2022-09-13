import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt


data = pd.read_csv('res/titanic.csv')
print(data.columns)
print(data.head(5))
print("4:7")
print((data[4:7])['age'])
print((data[4:7]).describe())
print("MEAN")
print((data[4:7])['age'].mean())
print((data[4:7])['age'].median())

print(data[['sex', 'survived']])

sns.histplot(data=data['sex'], bins =2)
plt.show()
sns.histplot(data=data['age'], bins =20)
plt.axvline(data['age'].mean(), color = 'r')
plt.axvline(data['age'].median(), color = 'g')

plt.show()
print("Age describe")
print(data['age'].describe())
sns.boxplot(data=data['age'])

print("Version 1")
print("Median: ", np.median(data['age']))
print(data['age'])
print(np.mean(data['age']))
upper_quartile = np.percentile(data['age'], 75)
print("upper_quartile: ", np.percentile(data['age'], 75))
lower_quartile = np.percentile(data['age'], 25)
print("lower_quartile: ", lower_quartile)

iqr = upper_quartile - lower_quartile
print("iqr: ", iqr)
upper_whisker = data['age'][data['age']<=upper_quartile+1.5*iqr].max()
print("upper_whisker: ", upper_whisker)
lower_whisker = data['age'][data['age']>=lower_quartile-1.5*iqr].min()
print("lower_whisker: ", lower_whisker)

# Versdion 2
print("22222222222")
print("Median: ", data['age'].median())
print(data['age'])
print("MEAN", data['age'].mean())
upper_quartile = data['age'].quantile(0.75)
print("upper_quartile: ", upper_quartile)
lower_quartile = data['age'].quantile(0.25)
print("lower_quartile: ", lower_quartile)

iqr = upper_quartile - lower_quartile
print("iqr: ", iqr)
upper_whisker = data['age'][data['age']<=upper_quartile+1.5*iqr].max()
print("upper_whisker: ", upper_whisker)
lower_whisker = data['age'][data['age']>=lower_quartile-1.5*iqr].min()
print("lower_whisker: ", lower_whisker)
plt.show()