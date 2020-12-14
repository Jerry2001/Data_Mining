import pandas as pd
from subprocess import call
from sklearn import svm
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report 
from sklearn.metrics import confusion_matrix

attribute = ["Water Reservoir Surface", "Number of Reservoir", "Type of Reservoir", "Presence of Vegetation", "The Most Dominant Land Type"
, "The Second Most Dominant Land Type", "The Third Most Dominant Land Type", "Use of Water Reservoir", "Presence of Fishing", "Precentage Access to Undeveloped Area"
, "Minimum Distance to Road", "Minimum Distance to Building", "Maintenance Status of Reservoir", "Type of Shore"]

fileLabel =["gf", "bf", "ct", "ft", "tf", "cn", "gn"]
label = ("Green frog", "Brown frog", "Common toad", "Fire-bellied toad", "Tree frog", "Common newt", "Great crested newt")
trainData = pd.read_csv("../Dataset/preprocesstrain.csv", delimiter=",") 

for column in (trainData.columns):
	if(column == "label"): break
	trainData[column] = trainData[column].apply(lambda x: (x*1.0 - trainData[column].min()) / 
		(trainData[column].max() - trainData[column].min()))

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

for column in (testData.columns):
	if(column == "label"): break
	testData[column] = testData[column].apply(lambda x: (x*1.0 - testData[column].min()) / 
		(testData[column].max() - testData[column].min()))

attributeTest = list(([row.SR, row.NR, row.TR, row.VR, row.SUR1, row.SUR2, row.SUR3, row.UR, row.FR, row.OR, row.RR, row.BR, row.MR, row.CR]) for row in testData.itertuples())
labelTest = []
labelTest.append(list(row._15 for row in testData.itertuples()))
labelTest.append(list((row._16) for row in testData.itertuples()))
labelTest.append(list((row._17) for row in testData.itertuples()))
labelTest.append(list((row._18) for row in testData.itertuples()))
labelTest.append(list((row._19) for row in testData.itertuples()))
labelTest.append(list((row._20) for row in testData.itertuples()))
labelTest.append(list((row._21) for row in testData.itertuples()))
totalAccuracy = 0.0

for index in range(7):
	svmClassifier = svm.SVC()
	svmClassifier = svmClassifier.fit(attributeTrain, labelTrain[index])
	labelPredict = svmClassifier.predict(attributeTest)
	totalAccuracy += accuracy_score(labelTest[index], labelPredict) * 100
	print(label[index], end = ": ")
	print(accuracy_score(labelTest[index], labelPredict) * 100)
	labelTest[index] = list(map(lambda a: int(a), labelTest[index]))
	labelPredict = list(map(lambda a: int(a), labelPredict))
	#print(labelTest[index], labelPredict)
	print(confusion_matrix(labelTest[index], labelPredict, labels = [0, 1]))
totalAccuracy /= 7
print("\nAverage accuracy: " + str(totalAccuracy))