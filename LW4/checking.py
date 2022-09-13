import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt


data = pd.read_csv('res/titanic.csv')
print(data[['sex','age']].head(10))


print("Check eq")

print((data[['sex','age']]==29.00).head(10))


print("Check eq")
print(data[['sex','age']].eq(29.00).head(10))

print("Check eq")
print(data[['sex','age']].head(10) != pd.Series(["male", 25], index=['sex','age']))

print("Check isnull")
print(pd.Series([5, 6, np.NaN]).isnull())

print("Check isnull")
print(pd.Series([5, 6, np.NaN]).eq(0))

print("mean")
print(pd.Series([5, 6, np.NaN]).mean())

print("mean")
print(pd.Series([5, 6, np.NaN]).mean(skipna=False))

print("mode")
print(data[['sex','age']].head(10).mode())


cols = data.columns[:8]
print("mode")
print(data[cols].mode(dropna=False))

print("mode")
print(data["name"].mode(dropna=False))

data.hist()
plt.show()

print("dropna")
print(data.dropna())

print("dropna")
print(data.dropna(axis='columns'))

print("dropna")
print(data.dropna(how='all'))

print("dropna")
print(data.dropna(thresh=2))

print("dropna")
print(data.dropna(subset=['sex', 'name']))

print("count")
print(data.count())

print("count")
print(data.count(axis='columns'))

print("info")
print(data.info())



print("fillna")
print(data.fillna(0))

print("count")
print(data.fillna(0).count())



print("fillna")
print(data.fillna(method="ffill"))

print("count")
print(data.fillna(method="ffill").count())

print("count")
print(data["age"].fillna(data["age"].mean()).count())


print("Check isnull")

pf_s = pd.Series([5, 6, np.NaN])
print(pf_s)
print(pf_s.mean())
print(pf_s.mode())
print(pf_s.fillna(pf_s.mode()))
print(pf_s.fillna(pf_s.mean()))

pf_s = pd.Series([5, 6, np.NaN,5])
print("test 2")
print(pf_s)
print(pf_s.mean())
print(pf_s.mode())
print(pf_s.fillna(pf_s.mode()[0]))
print(pf_s.fillna(pf_s.mean()))


data.hist()
plt.show()
data["age"].hist()
plt.show()
data["age"].fillna(data["age"].mean()).hist()
plt.show()






