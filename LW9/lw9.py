# https://www.youtube.com/watch?v=2Bkp4B8sJ2Y

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

data = pd.read_csv('res/titanic.csv')

print(data.info())

# Keep important columns
data.drop("name", axis = 1, inplace=True)
data.drop("ticket", axis = 1, inplace=True)
data.drop("cabin", axis = 1, inplace=True)
data.drop("boat", axis = 1, inplace=True)
data.drop("body", axis = 1, inplace=True)
data.drop("fare", axis = 1, inplace=True)
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
for clf in classifiers:
    train_sizes, train_scores, test_scores = learning_curve(clf, X, y, cv=5, scoring='accuracy',
                                                            n_jobs=-1, train_sizes=np.linspace(0.01, 1, 50), verbose=1)

    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)

    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    plt.plot(train_sizes, train_mean, label='Training Score')
    plt.plot(train_sizes, test_mean, label='Cross-Validations Score')
    plt.title(clf.__class__.__name__)

    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, color='#DDDDDD')
    plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, color='#DDDDDD')
    plt.show()




