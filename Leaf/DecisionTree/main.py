import pandas as pd
import numpy as np
from subprocess import call
from sklearn import tree
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report 
from sklearn.metrics import confusion_matrix 
from sklearn.model_selection import train_test_split

data = pd.read_csv("../Dataset/leaf.csv", delimiter=",")

xTrain, xTest, yTrain, yTest = train_test_split(data.iloc[:, :-1], data.iloc[:, -1:], test_size=0.33, random_state= 0)
treeClassifier = tree.DecisionTreeClassifier(max_depth = 11, max_features = "auto", random_state=0)
treeClassifier.fit(xTrain, yTrain)
yPredict = treeClassifier.predict(xTest)
yTrainPredict = treeClassifier.predict(xTrain)
print(accuracy_score(yTest, yPredict) * 100)
print(accuracy_score(yTrain, yTrainPredict) * 100)
# file = "../Visualization/binary.dot"
# tree.export_graphviz(treeClassifier, out_file=file, feature_names = data.columns[:-1], class_names = True, filled=True, rounded=True, special_characters=True)
# call(['dot', '-Tpng', file, '-o', 'binary.png', '-Gdpi=600'])