import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import svm

def result_info(clf,X_train,y_train,X_test,y_test,test_number = 1 ):
    print("Test Number: ", test_number)
    print("Score Train: ", clf.score(X_train, y_train))
    print("Score Test: ", clf.score(X_test, y_test))

    # calculate the predicted values
    y_pred = clf.predict(X_test)

    cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
    print("error matrix: \n", cnf_matrix)
    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
    print("Precision:", metrics.precision_score(y_test, y_pred))
    print("Recall:", metrics.recall_score(y_test, y_pred))

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
data["survived"]=data["survived"].map({0: -1,1:1})
print("Survived2: ",data["survived"].unique())
print(data.count())

data.dropna(inplace=True)
print(data.count())

y = data["survived"]
X = data.drop("survived" , axis = 1)

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0)

clf = svm.SVC()
clf.fit(X_train,y_train)
result_info(clf, X_train, y_train, X_test, y_test, 1)

clf = svm.SVC(C=10,gamma=0.2)
clf.fit(X_train,y_train)
result_info(clf, X_train, y_train, X_test, y_test, 2)

clf = svm.SVC(kernel='poly',degree=3)
clf.fit(X_train,y_train)
result_info(clf, X_train, y_train, X_test, y_test, 3)

clf = svm.SVC(C=10, kernel='poly',degree=3)
clf.fit(X_train,y_train)
result_info(clf, X_train, y_train, X_test, y_test, 4)

clf = svm.SVC(kernel='rbf')
clf.fit(X_train,y_train)
result_info(clf, X_train, y_train, X_test, y_test, 5)

clf = svm.SVC(C=5, gamma=0.4, kernel='rbf')
clf.fit(X_train,y_train)
result_info(clf, X_train, y_train, X_test, y_test, 6)




