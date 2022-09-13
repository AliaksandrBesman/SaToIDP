import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt


data = pd.read_csv('res/titanic.csv')
print("Hello1")
print(data['age'])
print(data[data['age'] < 5].get('age'))
print("Hello2")
print(data['age'])
print("Hello3")
print(data['age'][data['age'] < 5])