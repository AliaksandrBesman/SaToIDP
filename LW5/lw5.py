import pandas as pd
import numpy as np
import time
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn import tree

data = pd.read_csv('res/titanic.csv')

print(data.info())

# col = data.columns
# for col in data:
#   print(col + ":")
#   print(data[col].unique())
#   print()


# Keep important columns

data.drop("name", axis = 1, inplace=True)
data.drop("ticket", axis = 1, inplace=True)
data.drop("cabin", axis = 1, inplace=True)
data.drop("boat", axis = 1, inplace=True)
data.drop("body", axis = 1, inplace=True)
data.drop("fare", axis = 1, inplace=True)
data.drop("home.dest", axis = 1, inplace=True)
print(data.info())
print(data.count())


data.dropna(inplace=True)
print(data.count())

col = data.columns

print("Unique Values")
for col in data:
  print(col + ":")
  print(data[col].unique())
  print()



data["sex"]=data["sex"].map({"male": 0, "female": 1})
data["embarked"]=data["embarked"].map({"S": 0, "C": 1, "Q":2})

print("Unique Values")
for col in data:
  print(col + ":")
  print(data[col].unique())
  print()


y = data["survived"]
X = data.drop("survived" , axis = 1)

print("Size info")
print(X.shape," ; ", y.shape) # проверим размерность

X_train, X_valid, y_train, y_valid = train_test_split(   # по умолчанию 75% и 25%
    X, y, test_size=0.3, random_state=17)

first_tree = DecisionTreeClassifier(random_state=17)

cross_val_score(first_tree, X_train, y_train, cv=5) # оценка модели с помощью кросс-валидации

np.mean(cross_val_score(first_tree, X_train, y_train, cv=5)) #среднее по пяти оценкам

# соседи
first_knn = KNeighborsClassifier()
np.mean(cross_val_score(first_knn, X_train, y_train, cv=5))

# улучшение
tree_params = {"max_depth": np.arange(1, 11), "max_features": [0.5, 0.7, 1]}
tree_grid = GridSearchCV(first_tree, tree_params, cv=5, n_jobs=-1)

st=time.time()
print(tree_grid.fit(X_train, y_train))
print("----%.2f----"%(time.time()-st))
print(tree_grid.best_score_, tree_grid.best_params_)
# улучшаем метод KNN
knn_params = {"n_neighbors": range(5, 30, 5)}
knn_grid = GridSearchCV(first_knn, knn_params, cv=11)
st=time.time()
print(knn_grid.fit(X_train, y_train))
print("----%.2f----"%(time.time()-st))
print(knn_grid.best_score_, knn_grid.best_params_)
print("Estimator")
print(tree_grid.best_estimator_)




tree_valid_pred = tree_grid.predict(X_valid) # прогноз на отложенной выборке
print(tree_valid_pred[0:20]) # первые 20 спрогнозированых меток
print(accuracy_score(y_valid, tree_valid_pred))
print(confusion_matrix(y_valid, tree_valid_pred))

first_tree.fit(X_train,y_train)
tree.export_graphviz(first_tree, out_file='tree.dot', feature_names=X_train.columns)

error = []
for i in range(1, 40):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_valid)
    error.append(np.mean(pred_i != y_valid))
plt.figure(figsize=(12, 6))
plt.plot(range(1, 40), error, color='red', linestyle='dashed', marker='o',
         markerfacecolor='blue', markersize=10)
plt.title('Error Rate K Value')
plt.xlabel('K Value')
plt.ylabel('Mean Error')
plt.show()


cv_scores, holdout_scores = [], []
n_neighb = [1, 2, 3, 5] + list(range(50, 550, 50))

for k in n_neighb:

    knn = KNeighborsClassifier(n_neighbors=k)
    cv_scores.append(np.mean(cross_val_score(knn, X_train, y_train, cv=5)))
    knn.fit(X_train, y_train)
    holdout_scores.append(accuracy_score(y_valid, knn.predict(X_valid)))

plt.plot(n_neighb, cv_scores, label='CV')
plt.plot(n_neighb, holdout_scores, label='holdout')
plt.title('Easy task. kNN fails')
plt.legend();
plt.show()

