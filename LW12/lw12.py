import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import methods as m
from itertools import product
import statsmodels.api as sm


data = pd.read_csv('res/DailyDelhiClimateTrain2.csv')

print(data.info())
print(data.head())

print("log", np.log(np.exp(1)))
print("log2", np.log(10))

print(data.meantemp.max())
print(data.meantemp.min())

print(data.meantemp.shape[0])
print(data.meantemp.shape[0] + 5)

plt.figure(figsize=(14, 7))
plt.plot(data.meantemp)
plt.title('Closing price of New Germany Fund Inc (GF)')
plt.ylabel('Closing price ($)')
plt.xlabel('Trading day')
plt.grid(False)
plt.show()

m.plot_moving_average(data['meantemp'],30)
m.plot_moving_average(data['meantemp'],30, True)
#
m.plot_exponential_smoothing(data['meantemp'], [0.05, 0.3])
#
m.plot_double_exponential_smoothing(data['meantemp'], alphas=[0.9, 0.02], betas=[0.9, 0.02])

m.tsplot(data['meantemp'], lags=30)

data_diff = data['meantemp'] - data['meantemp'].shift(1)

m.tsplot(data_diff[1:], lags=30)

data_diff = np.log(data['meantemp'])

m.tsplot(data_diff[1:], lags=30)

# # Установите начальные значения и некоторые границыs
# times = 3
# ps = range(0, times)
# d = 1
# qs = range(0, times)
# Ps = range(0, times)
# D = 1
# Qs = range(0, times)
# s = 5
#
# # Составьте список со всеми возможными комбинациями параметров
# parameters = product(ps, qs, Ps, Qs)
# parameters_list = list(parameters)
# print(len(parameters_list))
#
# result_table = m.optimize_SARIMA(parameters_list,data['meantemp'], d, D, s)
# print(result_table)
#
# # Set parameters that give the lowest AIC (Akaike Information Criteria)
#
# p, q, P, Q = result_table.parameters[0]
#
# best_model = sm.tsa.statespace.SARIMAX(data['meantemp'], order=(p, d, q),
#                                        seasonal_order=(P, D, Q, s)).fit(disp=-1)
#
# # печатаем краткие характеристики лучшей модели
# print(best_model.summary())
# # plot_SARIMA(data, best_model, 5)
# print(best_model.predict(start=data.meantemp.shape[0], end=data.meantemp.shape[0] + 5))
# print(m.mean_absolute_percentage_error(data.meantemp[s+d:], best_model.fittedvalues[s+d:]))


#log
# Установите начальные значения и некоторые границыs
data['meantemp'] = np.log(data['meantemp'])

times = 5
ps = range(0, times)
d = 1
qs = range(0, times)
Ps = range(0, times)
D = 1
Qs = range(0, times)
s = 5

# Составьте список со всеми возможными комбинациями параметров
parameters = product(ps, qs, Ps, Qs)
parameters_list = list(parameters)
print(len(parameters_list))

result_table = m.optimize_SARIMA(parameters_list,data['meantemp'], d, D, s)
print(result_table)

# Set parameters that give the lowest AIC (Akaike Information Criteria)

p, q, P, Q = result_table.parameters[0]

best_model = sm.tsa.statespace.SARIMAX(data['meantemp'], order=(p, d, q),
                                       seasonal_order=(P, D, Q, s)).fit(disp=-1)

# печатаем краткие характеристики лучшей модели
print(best_model.summary())
# plot_SARIMA(data, best_model, 5)
print(best_model.predict(start=data.meantemp.shape[0], end=data.meantemp.shape[0] + 5))
print(np.exp(best_model.predict(start=data.meantemp.shape[0], end=data.meantemp.shape[0] + 5)))
print("BEST: ", best_model.fittedvalues[s+d:])
print(m.mean_absolute_percentage_error(data.meantemp[s+d:], best_model.fittedvalues[s+d:]))

plt.figure(figsize=(14, 7))
plt.plot(data.meantemp[s+d:])
plt.plot(best_model.fittedvalues[s+d:])
plt.plot(best_model.predict(start=0, end=data.meantemp.shape[0]))
plt.title('Train')
plt.ylabel('Closing price ($)')
plt.xlabel('Trading day')
plt.grid(False)
plt.show()

# data_test = pd.read_csv('res/DailyDelhiClimateTest2.csv')
#
# print(data_test.meantemp)
# print(data_test.count())
# print(np.exp(best_model.predict(start=data.meantemp.shape[0], end=data.meantemp.shape[0] + data_test.meantemp.shape[0])))
# print((np.exp(best_model.predict(start=data.meantemp.shape[0], end=data.meantemp.shape[0] + data_test.meantemp.shape[0]))).count())
#
# plt.figure(figsize=(14, 7))
# plt.plot(np.exp(best_model.predict(start=data.meantemp.shape[0], end=data.meantemp.shape[0] + data_test.meantemp.shape[0])))
# plt.plot(data_test.meantemp)
# plt.title('Closing price of New Germany Fund Inc (GF)')
# plt.ylabel('Closing price ($)')
# plt.xlabel('Trading day')
# plt.grid(False)
# plt.show()