import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

fig = plt.figure(figsize=(10, 6))

axes = fig.add_axes([0.1,0.1,0.8,0.8])
x = np.arange(0,1,0.1)
axes.plot(x, x**2, 'r')
axes.plot(x, x**3, 'b*--') # добавить вторую крикую на холст синего цвета с маркерами другого типа

axes.set_xlabel('x') # добавить название оси Х
axes.set_ylabel('y') # добавить название оси Y
axes.set_title('Hello proglib') # добавить название всего графика
axes.legend([r'$x^2$',r'$x^3$'], loc=2) # добавить легенду

plt.show() # вывести график на экран

# Второй график
fig = plt.figure()

axes1 = fig.add_axes([0.1, 0.1, 0.8, 0.8]) # основной график
axes2 = fig.add_axes([0.2, 0.5, 0.4, 0.3]) # внутренный график. Его размер меньше, цифры показывают долю от figsize

# Основной график
axes1.plot(x, x**2, 'r')
axes1.set_xlabel('x')
axes1.set_ylabel('y')
axes1.set_title('Я внешний график')

# Вложенный график
axes2.plot(x**2, x, 'b')
axes2.set_xlabel('y')
axes2.set_ylabel('x')
axes2.set_title('Я внутренний график')

plt.show()

# Три графика
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(16, 5))

for pow, ax in enumerate(axes):
    ax.plot(x, x**(pow + 1), 'b')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(f'$y = x^{pow + 1}$', fontsize=18)
fig.tight_layout() # автоматически вписывает все графики в размер холста

plt.show()

#Все три модуля

data = pd.read_csv('resources/titanic.csv')

fig = plt.figure()
axes = fig.add_axes([0.0, 0.0, 1.0, 1.0])
bins = 20 #  количество столбцов
index = np.arange(bins) # создаем список от 0 до bins - 1
axes.hist(data[data['sex'] == 'male']['age'].dropna(), bins=bins, alpha=0.6, label='Мужчины') # добавляем на холст гистограмму распределения возрастов среди мужчин
axes.hist(data[data['sex'] == 'female']['age'].dropna(), bins=bins, alpha=0.6, label='Женщины') # добавляем на холст гистограмму распределения возрастов среди женщин

axes.legend() # строим легенду
axes.set_xlabel('Возраст', fontsize=18)
axes.set_ylabel('Количество', fontsize=18)
axes.set_title('Распределение возрастов по полу человека', fontsize=18)

plt.show()