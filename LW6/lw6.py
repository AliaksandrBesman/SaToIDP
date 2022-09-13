import pandas as pd
import numpy as np
import time
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split

boston_dataset = load_boston()

data = pd.DataFrame(data = boston_dataset.data, columns=boston_dataset.feature_names)
data['PRICE'] = boston_dataset.target

print(data.head())
print(pd.isnull(data).any())

plt.figure(figsize=(10,6))
plt.hist(data['PRICE'], bins=50, ec='black', color='#2196f3')
plt.show()

# g = sns.PairGrid(data)
# g.map_diag(sns.histplot)
# g.map_offdiag(sns.scatterplot)
# plt.show()

hm = sns.heatmap(data.corr(),
                 cbar=True,
                 annot=True)
plt.show()

plt.figure(figsize=(10,6))

plt.scatter(x=data['RM'], y = data['PRICE'], alpha=0.6, s=80, color='skyblue')
sns.lmplot(x='RM', y='PRICE', data=data, size = 7)
plt.show()

prices = data['PRICE']
features = data.drop('PRICE', axis = 1)

X_train, X_test, y_train, y_test = train_test_split(features, prices, test_size=0.2, random_state=10)

regr = LinearRegression()
regr.fit(X_train, y_train)

train_predictions = regr.predict(X_test)
print('Trainig data r-squared:', regr.score(X_train,y_train))
print('Test data r-squared:', regr.score(X_test,y_test))

print('Intercept', regr.intercept_)

print(pd.DataFrame(data=regr.coef_, index=X_train.columns, columns=['coef']))


#LOG

print(data['PRICE'].skew())
print(data['PRICE'].min)

y_log = np.log(data['PRICE'])

print(y_log.tail())
print(y_log.skew())

sns.displot(y_log)
plt.show()

prices = np.log(data['PRICE'])
features = data.drop('PRICE', axis = 1)

X_train, X_test, y_train, y_test = train_test_split(features, prices, test_size=0.2, random_state=10)

regr = LinearRegression()
regr.fit(X_train, y_train)

print('Trainig data r-squared:', regr.score(X_train,y_train))
print('Test data r-squared:', regr.score(X_test,y_test))

print('Intercept', regr.intercept_)

print(pd.DataFrame(data=regr.coef_, index=X_train.columns, columns=['coef']))