import pandas as pd
from subprocess import call
from sklearn import svm
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report 
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix 
from sklearn.model_selection import KFold

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
	return (match * 1.0 / (len(label) * 7))

def validationAccuracy():
	trainData = pd.read_csv("../Dataset/binarytrain.csv", delimiter=",") 

	attributeTrain = list(([row.SR, row.NR, row.TR, row.VR, row.SUR1, row.SUR2, row.SUR3, row.UR, row.FR, row.OR, row.RR, row.BR, row.MR, row.CR]) for row in trainData.itertuples())
	labelTrain = [(row.label) for row in trainData.itertuples()]

	testData = pd.read_csv("../Dataset/binarytest.csv", delimiter=",") 

	attributeTest = list(([row.SR, row.NR, row.TR, row.VR, row.SUR1, row.SUR2, row.SUR3, row.UR, row.FR, row.OR, row.RR, row.BR, row.MR, row.CR]) for row in testData.itertuples())
	labelTest = [(row.label) for row in testData.itertuples()]
	totalAccuracy = 0.0

	svmClassifier = svm.SVC()
	svmClassifier = svmClassifier.fit(attributeTrain, list(labelTrain))
	#print(labelTrain[index])
	labelPredict = svmClassifier.predict(attributeTest)
	totalAccuracy = str(accuracyCalc(labelTest, labelPredict))
	print("Average accuracy: " + str(totalAccuracy))

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
		
		svmClassifier = svm.SVC()
		svmClassifier = svmClassifier.fit(attributeTrain, labelTrain)
		labelPredict = svmClassifier.predict(attributeTest)
		print("Accuracy", end = ": ")
		print(accuracyCalc(labelTest, labelPredict))
		#labelPredict = svmClassifier.predict(attributeTrain)
		#print("*" + str(accuracyCalc(labelTrain, labelPredict)))
		totalAccuracy += accuracyCalc(labelTest, labelPredict)
	totalAccuracy /= 189
	print(totalAccuracy)

validationAccuracy()