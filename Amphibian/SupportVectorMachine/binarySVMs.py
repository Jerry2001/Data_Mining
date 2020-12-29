#Construct 7 binary SVMs for 7 amphian species 

import pandas as pd
from subprocess import call
from sklearn import svm
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report 
from sklearn.metrics import confusion_matrix

toReturn = pd.DataFrame()

def returnPredict():
	validationAccuracy()
	return toReturn

def validationAccuracy():
	#Global variable store sites attributes, species label and abbreviation 
	attribute = ["Water Reservoir Surface", "Number of Reservoir", "Type of Reservoir", "Presence of Vegetation", "The Most Dominant Land Type"
	, "The Second Most Dominant Land Type", "The Third Most Dominant Land Type", "Use of Water Reservoir", "Presence of Fishing", "Precentage Access to Undeveloped Area"
	, "Minimum Distance to Road", "Minimum Distance to Building", "Maintenance Status of Reservoir", "Type of Shore"]
	fileLabel =["gf", "bf", "ct", "ft", "tf", "cn", "gn"]
	label = ("Green frog", "Brown frog", "Common toad", "Fire-bellied toad", "Tree frog", "Common newt", "Great crested newt")

	#Getting training data
	trainData = pd.read_csv("../Dataset/preprocesstrain.csv", delimiter=",")

	#Normalize training data 
	for column in (trainData.columns):
		if(column == "label"): break
		trainData[column] = trainData[column].apply(lambda x: (x*1.0 - trainData[column].min()) / 
			(trainData[column].max() - trainData[column].min()))

	stdout = open('SVM.txt', 'w')

	#Get the label of the species and the site attributes of each instance in the training set
	attributeTrain = list(([row.SR, row.NR, row.TR, row.VR, row.SUR1, row.SUR2, row.SUR3, row.UR, row.FR, row.OR, row.RR, row.BR, row.MR, row.CR]) for row in trainData.itertuples())
	labelTrain = []
	labelTrain.append(list((row._15) for row in trainData.itertuples()))
	labelTrain.append(list((row._16) for row in trainData.itertuples()))
	labelTrain.append(list((row._17) for row in trainData.itertuples()))
	labelTrain.append(list((row._18) for row in trainData.itertuples()))
	labelTrain.append(list((row._19) for row in trainData.itertuples()))
	labelTrain.append(list((row._20) for row in trainData.itertuples()))
	labelTrain.append(list((row._21) for row in trainData.itertuples()))

	#Getting testing data
	testData = pd.read_csv("../Dataset/preprocesstest.csv", delimiter=",") 

	#Normalize testing data 
	for column in (testData.columns):
		if(column == "label"): break
		testData[column] = testData[column].apply(lambda x: (x*1.0 - testData[column].min()) / 
			(testData[column].max() - testData[column].min()))

	#Get the label of the species and the site attributes of each instance in the testing set
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

	cherry = [['poly', 8, 1000, 'balanced'],['poly', 3, 9, None],['rbf', 3, 1, None],['poly', 8, 1, None],['poly', 3, 4, 'balanced'],['rbf', 3, 1, 'balanced'],['rbf', 3, 2, 'balanced']]

	#Loop through 7 species and create a SVM in each iteration
	for index in range(0,7):
		svmClassifier = svm.SVC(C = cherry[index][2], kernel = cherry[index][0], degree = cherry[index][1], class_weight = cherry[index][3])
		svmClassifier = svmClassifier.fit(attributeTrain, labelTrain[index])
		labelPredict = svmClassifier.predict(attributeTest)

		toReturn.insert(len(toReturn.columns), 'SVM' + fileLabel[index], labelPredict)

		#Add the accuracy to the total accuracy and print out the accuracy for the current SVM
		totalAccuracy += accuracy_score(labelTest[index], labelPredict) * 100
		print(label[index], end = ": ")
		print(accuracy_score(labelTest[index], labelPredict) * 100)

		#Cast the label to int to create the confusion matrix 
		labelTest[index] = list(map(lambda a: int(a), labelTest[index]))
		labelPredict = list(map(lambda a: int(a), labelPredict))
		print(confusion_matrix(labelTest[index], labelPredict, labels = [0, 1]))

		#Test the model with training data
		#print(labelTest[index], labelPredict)
		#print(confusion_matrix(labelTrain[index], list(map(lambda a: int(a), svmClassifier.predict(attributeTrain))), labels = [0, 1]))
		print()

	#Get the average accuracy for 7 SVMs
	totalAccuracy /= 7
	print("Average accuracy: " + str(totalAccuracy))
