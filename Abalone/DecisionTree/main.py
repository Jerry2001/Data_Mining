import pandas as pd
import numpy as np
from subprocess import call
from sklearn import tree
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report 
from sklearn.metrics import confusion_matrix 
from sklearn.model_selection import train_test_split

data = pd.read_csv("../Dataset/abalone.csv", delimiter=",")

xTrain, xTest, yTrain, yTest = train_test_split(data.iloc[:, :-1], data.iloc[:, -1:], test_size=0.33, random_state= 0)
treeClassifier = tree.DecisionTreeClassifier()
treeClassifier.fit(xTrain, yTrain)
labelPredict = treeClassifier.predict(xTest)
print(accuracy_score(labelTest, labelPredict))