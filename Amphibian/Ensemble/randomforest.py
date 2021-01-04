import pandas as pd
import numpy as np
from subprocess import call
from sklearn import tree
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report 
from sklearn.metrics import confusion_matrix 
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier

labelPredict = []
attribute = ["Water Reservoir Surface", "Number of Reservoir", "Type of Reservoir", "Presence of Vegetation", "The Most Dominant Land Type"
, "The Second Most Dominant Land Type", "The Third Most Dominant Land Type", "Use of Water Reservoir", "Presence of Fishing", "Precentage Access to Undeveloped Area"
, "Minimum Distance to Road", "Minimum Distance to Building", "Maintenance Status of Reservoir", "Type of Shore"]

fileLabel =["gf", "bf", "ct", "ft", "tf", "cn", "gn"]
label = ("Green frog", "Brown frog", "Common toad", "Fire-bellied toad", "Tree frog", "Common newt", "Great crested newt")


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

def analTree(tree, nodePath):
	global labelPredict
	numNode = tree.tree_.node_count
	leftChild = tree.tree_.children_left
	rightChild = tree.tree_.children_right
	attribute = tree.tree_.feature
	threshold = tree.tree_.threshold
	nodeDepth = np.zeros(shape=numNode, dtype = np.int64)
	isLeaf = np.zeros(shape=numNode, dtype = bool)
	stack = [(0, -1)]
	while len(stack) > 0:
		nodeID, pDepth = stack.pop()
		nodeDepth[nodeID] = pDepth + 1
		if(leftChild[nodeID] != rightChild[nodeID]):
			stack.append((leftChild[nodeID], pDepth + 1))
			stack.append((rightChild[nodeID], pDepth + 1))
		else: isLeaf[nodeID] = True
	'''
	for i in range(numNode):
	    if isLeaf[i]:
	        print("%snode=%s leaf node." % (nodeDepth[i] * "\t", i))
	    else:
	        print("%snode=%s test node: go to node %s if X[:, %s] <= %s else to "
	              "node %s."
	              % (nodeDepth[i] * "\t",
	                 i,
	                 leftChild[i],
	                 attribute[i],
	                 threshold[i],
	                 rightChild[i],
	                 ))
	'''
	totalMisclassify = np.zeros(shape=sum(isLeaf))
	totalNode = np.zeros(shape=sum(isLeaf), dtype = np.int32)
	for testInstance in range(len(testData)):
		nodeIndex = nodePath.indices[nodePath.indptr[testInstance]: nodePath.indptr[testInstance + 1]]
		totalNode[sum(isLeaf[:nodeIndex[-1]])] += 1
		totalMisclassify[sum(isLeaf[:nodeIndex[-1]])] += accuracyCalc([labelPredict[testInstance]],  [labelTest[testInstance]])
		#print(sum(isLeaf[:nodeIndex[-1]]), accuracyCalc([labelPredict[testInstance]],  [list(labelTest['label'])[testInstance]]))
	for i in range(sum(isLeaf)):
		print("Leaf node %d classified %d tests" % 
			(i, totalNode[i]), end = ' ')
		if(totalNode[i] > 0): print("- Accuracy: %.3f" % ((totalMisclassify[i] * 1.0 / totalNode[i] * 100.0)))		
		else: print()

def validationAccuracy():
	trainData = pd.read_csv("../Dataset/binarytrain.csv", delimiter=",") 

	attributeTrain = list(([row.SR, row.NR, row.TR, row.VR, row.SUR1, row.SUR2, row.SUR3, row.UR, row.FR, row.OR, row.RR, row.BR, row.MR, row.CR]) for row in trainData.itertuples())
	labelTrain = [(row.label) for row in trainData.itertuples()]

	testData = pd.read_csv("../Dataset/binarytest.csv", delimiter=",") 

	attributeTest = list(([row.SR, row.NR, row.TR, row.VR, row.SUR1, row.SUR2, row.SUR3, row.UR, row.FR, row.OR, row.RR, row.BR, row.MR, row.CR]) for row in testData.itertuples())
	labelTest = [(row.label) for row in testData.itertuples()]

	totalAccuracy = 0.0

	treeClassifier = RandomForestClassifier(max_depth = None, random_state = 6)
	treeClassifier = treeClassifier.fit(attributeTrain, labelTrain)
	labelPredict = treeClassifier.predict(attributeTest)
	print("Accuracy", end = ": ")
	print(accuracyCalc(labelTest, labelPredict))
	#file = "binary.dot"

	#tree.export_graphviz(treeClassifier, out_file=file, feature_names = attribute, class_names = True, filled=True, rounded=True, special_characters=True)
	#call(['dot', '-Tpng', file, '-o', 'binary.png', '-Gdpi=600'])

	#labelPredict = treeClassifier.predict(attributeTrain)
	#print("*" + str(accuracyCalc(labelTrain, labelPredict)))

	#analTree(treeClassifier, treeClassifier.decision_path(attributeTest))
	print()

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

		treeClassifier = tree.DecisionTreeClassifier(max_depth = 4)
		treeClassifier = treeClassifier.fit(attributeTrain, labelTrain)
		labelPredict = treeClassifier.predict(attributeTest)
		print("Accuracy", end = ": ")
		print(accuracyCalc(labelTest, labelPredict))
		#labelPredict = treeClassifier.predict(attributeTrain)
		#print("*" + str(accuracyCalc(labelTrain, labelPredict)))
		totalAccuracy += accuracyCalc(labelTest, labelPredict)
	totalAccuracy /= 189
	print(totalAccuracy)

validationAccuracy()