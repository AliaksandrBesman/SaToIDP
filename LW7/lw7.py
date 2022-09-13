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
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
# import the metrics class
from sklearn import metrics

def lessXY(x):
    if x > 0.5:
        return 1
    else:
        return 0

def lessXY2(x):
    if x > 0.621:
        return 1
    else:
        return 0

sigm = lambda xs: np.exp(xs) / (1 + np.exp(xs))

# def lessXY(x):
#     return  1

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
data.drop("embarked", axis = 1, inplace=True)

data["sex"]=data["sex"].map({"male": 0, "female": 1})
print(data.info())
print(data.count())


data.dropna(inplace=True)
print(data.count())

y = data["survived"]
X = data.drop("survived" , axis = 1)


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0)

# instantiate the model (using the default parameters)
logreg = LogisticRegression(max_iter = 150)
linreg = LinearRegression()

# fit the model with data
logreg.fit(X_train,y_train)
linreg.fit(X_train,y_train)
print("logreg: ",logreg.predict(X_train[100:130]))
# print("linreg: ",linreg.predict(X_train[100:130]))
print("linreg2: ",sigm(linreg.predict(X_train[100:130])))
eee2 = lambda xs: lessXY2(xs)
u2 = [eee2(el) for el in sigm(linreg.predict(X_train[100:130]))]
print("linreg3: ",u2)
print("TRUE: ",  (logreg.predict(X_train[:120]) == [eee2(el) for el in sigm(linreg.predict(X_train[:120]))]).all())
print("TRUE: ",  (logreg.predict(X_train) == [eee2(el) for el in linreg.predict(X_train)]).all())



# show
plt.hist(logreg.predict(X_train))
plt.show()
plt.hist(linreg.predict(X_train),color='r')
plt.show()
eee = lambda xs: lessXY(xs)


u = linreg.predict(X_train)
plt.hist(u,color='b')
plt.show()
u = [eee(el) for el in u]

plt.hist(u,color='b')

plt.show()

print("Score: ",logreg.score(X, y))

# calculate the predicted values
y_pred=logreg.predict(X_test)
print("Y predict: ", y_pred)

cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
print("error matrix: ",cnf_matrix)


class_names=[0,1] # name  of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
# create heatmap
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.show()


print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred))
print("Recall:",metrics.recall_score(y_test, y_pred))


y_pred_proba = logreg.predict_proba(X_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
auc = metrics.roc_auc_score(y_test, y_pred_proba)
plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)
# plt.show()

# Improving
model = LogisticRegression( C=10.0, max_iter=150, random_state=0)
model.fit(X, y)
print("Score: ", model.score(X, y))


y_pred=model.predict(X_test)
print("Y predict: ", y_pred)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred))#чувс
print("Recall:",metrics.recall_score(y_test, y_pred))#спец

print("Intercept: ",model.intercept_)
print("Coef: ",model.coef_)

# AUC
y_pred_proba = model.predict_proba(X_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
auc = metrics.roc_auc_score(y_test, y_pred_proba)
plt.plot(fpr,tpr,label="data 1, auc="+str(auc),color = 'r')
plt.legend(loc=4)
plt.show()

# Improving
model = LogisticRegression( C=5.0, max_iter=150, random_state=0)
model.fit(X, y)
print("Score: ", model.score(X, y))


y_pred=model.predict(X_test)
print("Y predict: ", y_pred)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred))
print("Recall:",metrics.recall_score(y_test, y_pred))

print("Intercept: ",model.intercept_)
print("Coef: ",model.coef_)
