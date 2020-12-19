import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import accuracy_score 
from sklearn.neural_network import MLPClassifier
attribute = ["Water Reservoir Surface", "Number of Reservoir", "Type of Reservoir", "Presence of Vegetation", "The Most Dominant Land Type"
, "The Second Most Dominant Land Type", "The Third Most Dominant Land Type", "Use of Water Reservoir", "Presence of Fishing", "Precentage Access to Undeveloped Area"
, "Minimum Distance to Road", "Minimum Distance to Building", "Maintenance Status of Reservoir", "Type of Shore"]

stdout = open('ANN.txt', 'w')
fileLabel =["gf", "bf", "ct", "ft", "tf", "cn", "gn"]
label = ("Green frog", "Brown frog", "Common toad", "Fire-bellied toad", "Tree frog", "Common newt", "Great crested newt")
trainData = pd.read_csv("../Dataset/preprocesstrain.csv", delimiter=",") 

# for column in (trainData.columns):
# 	if(column == "label"): break
# 	trainData[column] = trainData[column].apply(lambda x: (x*1.0 - trainData[column].min()) / 
# 		(trainData[column].max() - trainData[column].min()))

attributeTrain = list(([row.SR, row.NR, row.TR, row.VR, row.SUR1, row.SUR2, row.SUR3, row.UR, row.FR, row.OR, row.RR, row.BR, row.MR, row.CR]) for row in trainData.itertuples())
labelTrain = []
labelTrain.append(list((row._15) for row in trainData.itertuples()))
labelTrain.append(list((row._16) for row in trainData.itertuples()))
labelTrain.append(list((row._17) for row in trainData.itertuples()))
labelTrain.append(list((row._18) for row in trainData.itertuples()))
labelTrain.append(list((row._19) for row in trainData.itertuples()))
labelTrain.append(list((row._20) for row in trainData.itertuples()))
labelTrain.append(list((row._21) for row in trainData.itertuples()))

testData = pd.read_csv("../Dataset/preprocesstest.csv", delimiter=",") 

# for column in (testData.columns):
# 	if(column == "label"): break
# 	testData[column] = testData[column].apply(lambda x: (x*1.0 - testData[column].min()) / 
# 		(testData[column].max() - testData[column].min()))

attributeTest = list(([row.SR, row.NR, row.TR, row.VR, row.SUR1, row.SUR2, row.SUR3, row.UR, row.FR, row.OR, row.RR, row.BR, row.MR, row.CR]) for row in testData.itertuples())
labelTest = []
labelTest.append(list(row._15 for row in testData.itertuples()))
labelTest.append(list((row._16) for row in testData.itertuples()))
labelTest.append(list((row._17) for row in testData.itertuples()))
labelTest.append(list((row._18) for row in testData.itertuples()))
labelTest.append(list((row._19) for row in testData.itertuples()))
labelTest.append(list((row._20) for row in testData.itertuples()))
labelTest.append(list((row._21) for row in testData.itertuples()))

cherryPick = [[13, 7], [9, 4], [11, 11], [13, 7], [18, 17], [20, 18], [17, 12]]

totalAccuracy = 0.0

for index in range(0, 7):
	network = MLPClassifier(hidden_layer_sizes = cherryPick[index], alpha = 1e-5, solver='lbfgs', random_state = 1, max_iter = 2000)
	network = network.fit(attributeTrain, labelTrain[index])
	#print(labelTrain[index])
	labelPredict = network.predict(attributeTest)
	totalAccuracy += accuracy_score(labelTest[index], labelPredict) * 100
	print(label[index], end = ": ")
	print(accuracy_score(labelTest[index], labelPredict) * 100)
	print(confusion_matrix(labelTest[index], labelPredict, labels = [0, 1]))
	print(file = stdout)

totalAccuracy /= 7
print("Average accuracy: " + str(totalAccuracy))
