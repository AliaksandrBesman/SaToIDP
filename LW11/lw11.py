# https://www.youtube.com/watch?v=2Bkp4B8sJ2Y
# https://coderoad.ru/36382572/%D0%9A%D0%B0%D0%BA-%D0%BF%D1%80%D0%B8%D0%BC%D0%B5%D0%BD%D0%B8%D1%82%D1%8C-%D0%BC%D0%BE%D0%B4%D0%B5%D0%BB%D1%8C-%D0%BE%D0%B1%D1%83%D1%87%D0%B5%D0%BD%D0%BD%D1%83%D1%8E-%D1%81-%D0%BF%D0%BE%D0%BC%D0%BE%D1%89%D1%8C%D1%8E-PCA-%D0%B8-%D1%81%D0%BB%D1%83%D1%87%D0%B0%D0%B9%D0%BD%D0%BE%D0%B3%D0%BE-%D0%BB%D0%B5%D1%81%D0%B0-%D0%BA-%D1%82%D0%B5%D1%81%D1%82%D0%BE%D0%B2%D1%8B%D0%BC

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.model_selection import learning_curve
from sklearn.metrics import accuracy_score

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn import decomposition

data = pd.read_csv('res/titanic.csv')

print(data.info())

print(data["body"])
# Keep important columns
data.drop("name", axis = 1, inplace=True)
data.drop("ticket", axis = 1, inplace=True)
data.drop("cabin", axis = 1, inplace=True)
data.drop("boat", axis = 1, inplace=True)
data.drop("body", axis = 1, inplace=True)
# data.drop("fare", axis = 1, inplace=True)
data.drop("home.dest", axis = 1, inplace=True)
data.drop("embarked", axis = 1, inplace=True)

data["sex"]=data["sex"].map({"male": 0, "female": 1})
print(data.count())
print("Survived1: ",data["survived"].unique())
# data["survived"]=data["survived"].map({0: -1,1:1})
print("Survived2: ",data["survived"].unique())
print(data.count())

data.dropna(inplace=True)
print(data.count())

y = data["survived"]
X = data.drop("survived" , axis = 1)

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0)


classifiers = [
    KNeighborsClassifier(5),
    SVC(probability=True, kernel='rbf'),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    LogisticRegression(),
]

log_cols = ['Classifier', 'Accuracy']
log = pd.DataFrame(columns=log_cols)
acc_dict = {}

for clf in classifiers:
    name = clf.__class__.__name__
    clf.fit(X_train,y_train)
    train_predictions = clf.predict(X_test)
    acc = accuracy_score(y_test, train_predictions)
    acc_dict[name] = acc

for clf in acc_dict:
    log_entry = pd.DataFrame([[clf,acc_dict[clf]]], columns= log_cols)
    log = log.append(log_entry)

plt.xlabel('Accuracy')
plt.title('Classifier Accuracy')

sns.set_color_codes("muted")
sns.barplot(x='Accuracy', y = 'Classifier', data=log.sort_values(by='Accuracy'), color='b')
plt.show()


# Learning Curve
# Построим график процента отклонения от количества компонент
pca = decomposition.PCA().fit(X)
plt.figure(figsize=(10,7))
# pca.explained_variance_ratio_ Процент отклонения, объясняемый
# каждым из выбранных компонентов
plt.plot(np.cumsum(pca.explained_variance_ratio_), color='b', lw=2)
plt.xlabel('Number of components')
plt.ylabel('Total explained variance')
plt.xlim(0, 5)
plt.yticks(np.arange(0, 1.1, 0.1))
plt.axhline(0.9, c='r')
plt.show();


print("PCA X: ", X.count())
pca = decomposition.PCA(n_components=5)
X_centered = X - X.mean(axis=0)
pca.fit(X_centered)
X_pca = pca.transform(X_centered)

X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=.3,
                                                    stratify=y,
                                                    random_state=42)
log_cols = ['Classifier', 'Accuracy']
log = pd.DataFrame(columns=log_cols)
acc_dict = {}

for clf in classifiers:
    name = clf.__class__.__name__
    clf.fit(X_train,y_train)
    train_predictions = clf.predict(X_test)
    acc = accuracy_score(y_test, train_predictions)
    acc_dict[name] = acc

for clf in acc_dict:
    log_entry = pd.DataFrame([[clf,acc_dict[clf]]], columns= log_cols)
    log = log.append(log_entry)

plt.xlabel('Accuracy')
plt.title('Classifier Accuracy')

sns.set_color_codes("muted")
sns.barplot(x='Accuracy', y = 'Classifier', data=log.sort_values(by='Accuracy'), color='b')
plt.show()




