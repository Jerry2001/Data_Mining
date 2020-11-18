import pandas as pd
from subprocess import call
from sklearn import tree
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report 
from sklearn.metrics import confusion_matrix 

attribute = ["Water Reservoir Surface", "Number of Reservoir", "Type of Reservoir", "Presence of Vegetation", "The Most Dominant Land Type"
, "The Second Most Dominant Land Type", "The Third Most Dominant Land Type", "Use of Water Reservoir", "Presence of Fishing", "Precentage Access to Undeveloped Area"
, "Minimum Distance to Road", "Minimum Distance to Building", "Maintenance Status of Reservoir", "Type of Shore"]

fileLabel =["gf", "bf", "ct", "ft", "tf", "cn", "gn"]
label = ("Green frog", "Brown frog", "Common toad", "Fire-bellied toad", "Tree frog", "Common newt", "Great crested newt")
trainData = pd.read_csv("../Dataset/preprocesstrain.csv", delimiter=",") 

attributeTrain = list(([row.SR, row.NR, row.TR, row.VR, row.SUR1, row.SUR2, row.SUR3, row.UR, row.FR, row.OR, row.RR, row.BR, row.MR, row.CR]) for row in trainData.itertuples())
labelTrain = []
labelTrain.append(list(([row._15]) for row in trainData.itertuples()))
labelTrain.append(list(([row._16]) for row in trainData.itertuples()))
labelTrain.append(list(([row._17]) for row in trainData.itertuples()))
labelTrain.append(list(([row._18]) for row in trainData.itertuples()))
labelTrain.append(list(([row._19]) for row in trainData.itertuples()))
labelTrain.append(list(([row._20]) for row in trainData.itertuples()))
labelTrain.append(list(([row._21]) for row in trainData.itertuples()))

testData = pd.read_csv("../Dataset/preprocesstest.csv", delimiter=",") 

attributeTest = list(([row.SR, row.NR, row.TR, row.VR, row.SUR1, row.SUR2, row.SUR3, row.UR, row.FR, row.OR, row.RR, row.BR, row.MR, row.CR]) for row in testData.itertuples())
labelTest = []
labelTest.append(list(([row._15]) for row in testData.itertuples()))
labelTest.append(list(([row._16]) for row in testData.itertuples()))
labelTest.append(list(([row._17]) for row in testData.itertuples()))
labelTest.append(list(([row._18]) for row in testData.itertuples()))
labelTest.append(list(([row._19]) for row in testData.itertuples()))
labelTest.append(list(([row._20]) for row in testData.itertuples()))
labelTest.append(list(([row._21]) for row in testData.itertuples()))
totalAccuracy = 0.0

for index in range(7):
	treeClassifier = tree.DecisionTreeClassifier(max_depth = 4)
	treeClassifier = treeClassifier.fit(attributeTrain, labelTrain[index])
	labelPredict = treeClassifier.predict(attributeTest)
	totalAccuracy += accuracy_score(labelTest[index], labelPredict) * 100
	print(label[index], end = ": ")
	print(accuracy_score(labelTest[index], labelPredict) * 100)
	diagramClass = ["Not " + label[index], label[index]]
	file = fileLabel[index] + ".dot"
	
	tree.export_graphviz(treeClassifier, out_file=file, feature_names = attribute, class_names = diagramClass, filled=True, rounded=True, special_characters=True)
	call(['dot', '-Tpng', file, '-o', label[index] + '.png', '-Gdpi=600'])
	
	#labelPredict = treeClassifier.predict(attributeTrain)
	#print("*" + str(accuracy_score(labelTrain[index], labelPredict) * 100))
totalAccuracy /= 7
print("\nAverage accuracy:" + str(totalAccuracy))