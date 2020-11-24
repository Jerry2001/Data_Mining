import pandas as pd
from subprocess import call
from sklearn import svm
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report 
from sklearn.model_selection import train_test_split

def accuracyCalc(label, predict):
	if(not isinstance(label, list)): label = list(label['label'])
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

trainData = pd.read_csv("../Dataset/binary.csv", delimiter=",")
attributeTrain, attributeTest, labelTrain, labelTest = train_test_split(trainData.iloc[:, :-1], trainData.iloc[:, -1:], test_size = 0.33) 


totalAccuracy = 0.0

svmClassifier = svm.SVC()
svmClassifier = svmClassifier.fit(attributeTrain, list(labelTrain['label']))
#print(labelTrain[index])
labelPredict = svmClassifier.predict(attributeTest)
totalAccuracy = str(accuracyCalc(labelTest, labelPredict))
print("Average accuracy: " + str(totalAccuracy))