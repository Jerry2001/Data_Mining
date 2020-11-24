import pandas as pd
import numpy as np
from subprocess import call
from sklearn import tree
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report 
from sklearn.metrics import confusion_matrix 
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

labelPredict = []
attribute = ["Water Reservoir Surface", "Number of Reservoir", "Type of Reservoir", "Presence of Vegetation", "The Most Dominant Land Type"
, "The Second Most Dominant Land Type", "The Third Most Dominant Land Type", "Use of Water Reservoir", "Presence of Fishing", "Precentage Access to Undeveloped Area"
, "Minimum Distance to Road", "Minimum Distance to Building", "Maintenance Status of Reservoir", "Type of Shore"]

trainData = pd.read_csv("../Dataset/binary.csv", delimiter=",")
attributeTrain, attributeTest, labelTrain, labelTest = train_test_split(trainData.iloc[:, :-1], trainData.iloc[:, -1:], test_size = 0.33) 

testData = attributeTest.copy()
testData["label"] = labelTest

totalAccuracy = 0.0

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
		totalMisclassify[sum(isLeaf[:nodeIndex[-1]])] += accuracyCalc([labelPredict[testInstance]],  [list(labelTest['label'])[testInstance]])
		#print(sum(isLeaf[:nodeIndex[-1]]), accuracyCalc([labelPredict[testInstance]],  [list(labelTest['label'])[testInstance]]))
	for i in range(sum(isLeaf)):
		print("Leaf node %d classified %d tests" % 
			(i, totalNode[i]), end = ' ')
		if(totalNode[i] > 0): print("- Accuracy: %.3f" % ((totalMisclassify[i] * 1.0 / totalNode[i] * 100.0)))		
		else: print()

treeClassifier = tree.DecisionTreeClassifier(max_depth = 8)
treeClassifier = treeClassifier.fit(attributeTrain, labelTrain)
labelPredict = treeClassifier.predict(attributeTest)
totalAccuracy = accuracyCalc(labelTest, labelPredict)
analTree(treeClassifier, treeClassifier.decision_path(attributeTest))


print(totalAccuracy)
#file = "binary.dot"

#tree.export_graphviz(treeClassifier, out_file=file, feature_names = attribute, class_names = True, filled=True, rounded=True, special_characters=True)
#call(['dot', '-Tpng', file, '-o', 'binary.png', '-Gdpi=600'])

labelPredict = treeClassifier.predict(attributeTrain)
print("*" + str(accuracyCalc(labelTrain, labelPredict)))

print()

print("Average accuracy: " + str(totalAccuracy))