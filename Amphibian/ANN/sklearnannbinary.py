import pandas as pd
import numpy as np

from sklearn.neural_network import MLPClassifier

def accuracyCalc(label, predict):
	match = 0
	for i in range(len(label)):
		curAccuracy = 0
		u = str(label[i])
		v = str(predict[i])
		u = (7 - len(u)) * "0" + u
		v = (7 - len(v)) * "0" + v
		for j in range(len(u)):
			if(u[j] == v[j]): 
				match += 1
				curAccuracy += 1
		#print(u, v, curAccuracy / 7.0)
	return (match * 1.0 / (len(label) * 7))

trainData = pd.read_csv("../Dataset/binarytrain.csv", delimiter=",") 

attributeTrain = list(([row.SR, row.NR, row.TR, row.VR, row.SUR1, row.SUR2, row.SUR3, row.UR, row.FR, row.OR, row.RR, row.BR, row.MR, row.CR]) for row in trainData.itertuples())
labelTrain = [(row.label) for row in trainData.itertuples()]

testData = pd.read_csv("../Dataset/binarytest.csv", delimiter=",") 

attributeTest = list(([row.SR, row.NR, row.TR, row.VR, row.SUR1, row.SUR2, row.SUR3, row.UR, row.FR, row.OR, row.RR, row.BR, row.MR, row.CR]) for row in testData.itertuples())
labelTest = [(row.label) for row in testData.itertuples()]

network = MLPClassifier(hidden_layer_sizes = (20, 9), solver='lbfgs', random_state = 1)
network.fit(attributeTrain, labelTrain)

labelPredict = network.predict(attributeTest)
print("Accuracy", end = ": ")
print(accuracyCalc(labelTest, labelPredict))