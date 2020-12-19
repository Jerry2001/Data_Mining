import pandas as pd
import numpy as np

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix 

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


def validationAccuracy():
	trainData = pd.read_csv("../Dataset/binarytrain.csv", delimiter=",") 

	for column in (trainData.columns):
		if(column == "label"): break
		trainData[column] = trainData[column].apply(lambda x: (x*1.0 - trainData[column].min()) / 
			(trainData[column].max() - trainData[column].min()))

	attributeTrain = list(([row.SR, row.NR, row.TR, row.VR, row.SUR1, row.SUR2, row.SUR3, row.UR, row.FR, row.OR, row.RR, row.BR, row.MR, row.CR]) for row in trainData.itertuples())
	labelTrain = [(row.label) for row in trainData.itertuples()]

	testData = pd.read_csv("../Dataset/binarytest.csv", delimiter=",") 

	for column in (testData.columns):
		if(column == "label"): break
		testData[column] = testData[column].apply(lambda x: (x*1.0 - testData[column].min()) / 
			(testData[column].max() - testData[column].min()))

	attributeTest = list(([row.SR, row.NR, row.TR, row.VR, row.SUR1, row.SUR2, row.SUR3, row.UR, row.FR, row.OR, row.RR, row.BR, row.MR, row.CR]) for row in testData.itertuples())
	labelTest = [(row.label) for row in testData.itertuples()]

	
	network = MLPClassifier(hidden_layer_sizes = (21, 3), solver='lbfgs', alpha = 1e-5, random_state = 1, max_iter = 2000)
	network.fit(attributeTrain, labelTrain)
	labelPredict = network.predict(attributeTest)
	print("Accuracy", end = ": ")
	print(accuracyCalc(labelTest, labelPredict) * 100)

def kFoldAccuracy():
	data = pd.read_csv("../Dataset/binary.csv", delimiter=",")
	kf = KFold(n_splits = len(data), random_state = None, shuffle = False)
	totalAccuracy = 0.0
	for train_index, test_index in kf.split(data):
		trainData = data.iloc[train_index]
		testData = data.iloc[test_index]

		attributeTrain = list(([row.SR, row.NR, row.TR, row.VR, row.SUR1, row.SUR2, row.SUR3, row.UR, row.FR, row.OR, row.RR, row.BR, row.MR, row.CR]) for row in trainData.itertuples())
		labelTrain = [(row.label) for row in trainData.itertuples()]
		attributeTest = list(([row.SR, row.NR, row.TR, row.VR, row.SUR1, row.SUR2, row.SUR3, row.UR, row.FR, row.OR, row.RR, row.BR, row.MR, row.CR]) for row in testData.itertuples())
		labelTest = [(row.label) for row in testData.itertuples()]
		
		network = MLPClassifier(hidden_layer_sizes = (8, 6), solver='lbfgs', random_state = 1)
		network = network.fit(attributeTrain, labelTrain)
		labelPredict = network.predict(attributeTest)
		print("Accuracy", end = ": ")
		print(accuracyCalc(labelTest, labelPredict))
		#labelPredict = network.predict(attributeTrain)
		#print("*" + str(accuracyCalc(labelTrain, labelPredict)))
		totalAccuracy += accuracyCalc(labelTest, labelPredict)
	print(totalAccuracy)

validationAccuracy()