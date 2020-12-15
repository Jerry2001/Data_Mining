import pandas as pd
from subprocess import call
from sklearn import svm
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report 
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

trainData = pd.read_csv("../Dataset/preprocess.csv", delimiter=",") 

for column in (trainData.columns):
	if(column == "label"): break
	trainData[column] = trainData[column].apply(lambda x: (x*1.0 - trainData[column].min()) / 
		(trainData[column].max() - trainData[column].min()))

attributeTrain, attributeTest, labelTrain, labelTest = train_test_split(trainData.iloc[:, :-1], trainData.iloc[:, -1:], test_size = 0.33) 

totalAccuracy = 0.0


for label in attributeTrain.columns[-6:]:
	labelTrain[label] = attributeTrain[label]
	labelTest[label] = attributeTest[label]

attributeTrain.drop(columns=attributeTrain.columns[-6:], inplace = True)
attributeTest.drop(columns=attributeTest.columns[-6:], inplace = True)

labelTrain = labelTrain[trainData.columns[-7:]]
labelTest = labelTest[trainData.columns[-7:]]

for label in labelTest.columns:
	svmClassifier = svm.SVC(C = 1, kernel='rbf', degree = 3)
	svmClassifier = svmClassifier.fit(attributeTrain, labelTrain[label])
	labelPredict = svmClassifier.predict(attributeTest)
	totalAccuracy += accuracy_score(labelTest[label], labelPredict) * 100
	print(label, end = ": ")
	print(accuracy_score(labelTest[label], labelPredict) * 100)
	labelTest[label] = list(map(lambda a: int(a), labelTest[label]))
	labelPredict = list(map(lambda a: int(a), labelPredict))
	#print(labelTest[label], labelPredict)
	print(confusion_matrix(labelTest[label], labelPredict, labels = [0, 1]))
	print()
totalAccuracy /= 7
print("Average accuracy: " + str(totalAccuracy))