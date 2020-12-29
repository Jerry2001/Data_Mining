import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import accuracy_score 
from sklearn.neural_network import MLPClassifier
import sys
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

sys.path.append("../ANN")
sys.path.append("../DecisionTree")
sys.path.append("../SupportVectorMachine")

import binaryANNs 
import binaryDTs
import binarySVMs

trainData = pd.read_csv("../Dataset/preprocesstrain.csv", delimiter=",")

def accuracyCalc(label, predict):
	match = 0
	confusion_label = [[[], []] for i in range(7)]
	for i in range(len(label)):
		curAccuracy = 0
		u = str(label[i])
		v = str(predict[i])
		u = (7 - len(u)) * "0" + u
		v = (7 - len(v)) * "0" + v
		for j in range(len(u)):
			confusion_label[j][0].append(u[j])
			confusion_label[j][1].append(v[j])
			if(u[j] == v[j]): 
				match += 1
				curAccuracy += 1
		#print(u, v, curAccuracy / 7.0)
	for i in range(7):
		print(("Green frog", "Brown frog", "Common toad", "Fire-bellied toad", "Tree frog", "Common newt", "Great crested newt")[i])
		print(confusion_matrix(confusion_label[i][0], confusion_label[i][1], labels = ["0", "1"]))
		print()
	return (match * 1.0 / (len(label) * 7))

attributeTrain, attributeTest, labelTrain, labelTest = train_test_split(trainData.iloc[:, :-1], trainData.iloc[:, -1:], test_size = 0.2, random_state=0) 
for label in attributeTrain.columns[-6:]:
	labelTrain[label] = attributeTrain[label]
	labelTest[label] = attributeTest[label]

attributeTrain.drop(columns=attributeTrain.columns[-6:], inplace = True)
attributeTest.drop(columns=attributeTest.columns[-6:], inplace = True)

labelTrain = labelTrain[trainData.columns[-7:]]
labelTest = labelTest[trainData.columns[-7:]]

attributeTrainMeta = binarySVMs.returnPredict(attributeTrain, labelTrain, attributeTest)
attributeTrainMeta = pd.concat([attributeTrainMeta, binaryDTs.returnPredict(attributeTrain, labelTrain, attributeTest)], axis=1)
attributeTrainMeta = pd.concat([attributeTrainMeta, binaryANNs.returnPredict(attributeTrain, labelTrain, attributeTest)], axis=1)
#forLabel = pd.read_csv("../Dataset/preprocesstrain.csv", delimiter=",")
labelTest.reset_index(inplace = True)

attributeTestMeta = binarySVMs.returnPredictMeta()
attributeTestMeta = attributeTestMeta.astype(int)
attributeTestMeta = pd.concat([attributeTestMeta, binaryDTs.returnPredictMeta()], axis=1)
attributeTestMeta = pd.concat([attributeTestMeta, binaryANNs.returnPredictMeta()], axis=1)

binLabel = []
for index, row in labelTest.iterrows():
	binLabel.append(str(row["Green frogs"]) + str(row["Brown frogs"]) + str(row["Common toad"]) + str(row["Fire-bellied toad"])
	 + str(row["Tree frog"]) + str(row["Common newt"]) + str(row["Great crested newt"]))

binTestLabel = pd.DataFrame()
binTestLabel["label"] = binLabel

bayesClassifier = GaussianNB()
bayesClassifier = bayesClassifier.fit(attributeTrainMeta, binLabel)
predictLabel = bayesClassifier.predict(attributeTestMeta)

testData = pd.read_csv("../Dataset/binarytest.csv", delimiter=",") 
labelTestFinal = [(row.label) for row in testData.itertuples()]

print(accuracyCalc(labelTestFinal, predictLabel))