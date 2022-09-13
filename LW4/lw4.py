import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt



data = pd.read_csv('res/titanic.csv')
print(data[['sex','age']].head())

print("data.columns[:10]: ")
cols = data.columns[:10]
print(cols)


colours = ['#eeeeee', '#ff0000']
print("data[cols].isnull()")
print(data[cols].isnull())

sns.heatmap(data[cols].isnull(),cmap = sns.color_palette(colours))
plt.show()




print("data.count()")
print(data.count())

print("data.count(axis='columns')")
print(data.count(axis='columns'))


print("(data.loc[data['age'].eq(30),'age']).eq(30)")
print((data.loc[data['age'].eq(30),'age']).eq(30))

print("pd.Series([100, 250], index=[\"cost\", \"revenue\"])")
print(data[1:20].eq(pd.Series([30,'male'], index=["age",'sex'])))

print("data.isnull()")
print(data.isnull())
print("end")
print(data[data.isnull().any(axis=1)])


print(data.info())

print("start")
print(data[data.eq(0)].count().sort_values())
print("____________________")
print(data[data.isnull()].count().sort_values())
print("____________________")
print(data[data.eq(0)].count().sort_values().index.values[12])

print("1111111")
print(data[data[data.eq(0)].count().sort_values().index.values[12]])
print("222222")
print(data[data[data.isnull()].count().sort_values().index.values[12]])

data.drop(data[data.isnull()].count().sort_values().index.values[12], axis=1, inplace=True)

# data.dropna(axis='columns')
# data.dropna(how='all')
# data.dropna(thresh=2)
# data.dropna(subset=['age'])

item_counts = data['age'].value_counts()
max_item = item_counts.max()
print("MAX: ", max_item)

data.fillna(data.mode())
data['age'] = data['age'].fillna(data['age'].mean())
data.fillna(value=None , method="bfill")
data.fillna(value=None , method="ffill")

data.hist()
plt.show()

sns.heatmap(data.isnull(),cmap = sns.color_palette(colours))

plt.show()

print("FFFFFFFFF")
print(data[data.isnull()].count())

data = pd.read_csv('res/titanic.csv')
print("LLLLLLLLLLLLLL")
print(data[data.isnull()].count())
plt.show()

