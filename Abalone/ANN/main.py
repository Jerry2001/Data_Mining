import pandas as pd
import numpy as np
from subprocess import call
from sklearn import tree
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report 
from sklearn.metrics import confusion_matrix 
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

data = pd.read_csv("../Dataset/onehot.csv", delimiter=",")

xTrain, xTest, yTrain, yTest = train_test_split(data.iloc[:, :-1], data.iloc[:, -1:], test_size=0.33, random_state= 0)
network = MLPClassifier()
network.fit(xTrain, yTrain)
yPredict = network.predict(xTest)
yTrainPredict = network.predict(xTrain)
print(accuracy_score(yTest, yPredict) * 100)
print(accuracy_score(yTrain, yTrainPredict) * 100)
