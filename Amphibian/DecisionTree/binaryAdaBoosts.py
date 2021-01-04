import pandas as pd
import numpy as np
from subprocess import call
from sklearn import tree
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report 
from sklearn.metrics import confusion_matrix 
from sklearn.ensemble import AdaBoostClassifier

toReturnMeta = pd.DataFrame()
toReturn = pd.DataFrame()

def returnPredictMeta():
	global toReturnMeta
	toReturnMeta = pd.DataFrame()
	validationAccuracy()
	return toReturnMeta

def returnPredict(attributeTrain, labelTrain, attributeTest):
	global toReturn
	toReturn = pd.DataFrame()
	labelPredict = []
	fileLabel =["gf", "bf", "ct", "ft", "tf", "cn", "gn"]
	cherry = [[5, 'gini', 'auto', None],[5, 'entropy', 'sqrt', None],[5, 'entropy', 'log2', 'balanced'],[None, 'gini', 'sqrt', 'balanced'],[None, 'gini', 'sqrt', 'balanced'],[2, 'gini', None, 'balanced'],[4, 'entropy', None, 'balanced']]

	for index in range(0, 7):
		treeClassifier = tree.DecisionTreeClassifier(criterion = cherry[index][1],max_depth = cherry[index][0], max_features = cherry[index][2], class_weight = cherry[index][3], random_state=0)
		treeClassifier = treeClassifier.fit(attributeTrain, labelTrain.iloc[:, index])
		labelPredict = treeClassifier.predict(attributeTest)
		toReturn.insert(len(toReturn.columns), 'DT' + fileLabel[index], labelPredict)
	return toReturn
	
def validationAccuracy():
	labelPredict = []
	attribute = ["Water Reservoir Surface", "Number of Reservoir", "Type of Reservoir", "Presence of Vegetation", "The Most Dominant Land Type"
	, "The Second Most Dominant Land Type", "The Third Most Dominant Land Type", "Use of Water Reservoir", "Presence of Fishing", "Precentage Access to Undeveloped Area"
	, "Minimum Distance to Road", "Minimum Distance to Building", "Maintenance Status of Reservoir", "Type of Shore"]

	fileLabel =["gf", "bf", "ct", "ft", "tf", "cn", "gn"]
	label = ("Green frog", "Brown frog", "Common toad", "Fire-bellied toad", "Tree frog", "Common newt", "Great crested newt")
	trainData = pd.read_csv("../Dataset/preprocesstrain.csv", delimiter=",") 

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

	attributeTest = list(([row.SR, row.NR, row.TR, row.VR, row.SUR1, row.SUR2, row.SUR3, row.UR, row.FR, row.OR, row.RR, row.BR, row.MR, row.CR]) for row in testData.itertuples())
	labelTest = []
	labelTest.append(list((row._15) for row in testData.itertuples()))
	labelTest.append(list((row._16) for row in testData.itertuples()))
	labelTest.append(list((row._17) for row in testData.itertuples()))
	labelTest.append(list((row._18) for row in testData.itertuples()))
	labelTest.append(list((row._19) for row in testData.itertuples()))
	labelTest.append(list((row._20) for row in testData.itertuples()))
	labelTest.append(list((row._21) for row in testData.itertuples()))
	totalAccuracy = 0.0

	cherry = [[5, 'gini', 'auto', None],[5, 'entropy', 'sqrt', None],[5, 'entropy', 'log2', 'balanced'],[None, 'gini', 'sqrt', 'balanced'],[None, 'gini', 'sqrt', 'balanced'],[2, 'gini', None, 'balanced'],[4, 'entropy', None, 'balanced']]

	for index in range(0, 7):
		treeClassifier = tree.DecisionTreeClassifier(criterion = cherry[index][1],max_depth = cherry[index][0], max_features = cherry[index][2], class_weight = cherry[index][3], random_state=0)
		adaClassifier = AdaBoostClassifier(treeClassifier, learning_rate = 0.1, random_state = 0)
		adaClassifier = adaClassifier.fit(attributeTrain, labelTrain[index])
		labelPredict = adaClassifier.predict(attributeTest)
		toReturnMeta.insert(len(toReturnMeta.columns), 'DT' + fileLabel[index], labelPredict)
		totalAccuracy += accuracy_score(labelTest[index], labelPredict) * 100
		print(label[index], end = ": ")
		print(accuracy_score(labelTest[index], labelPredict) * 100)
		diagramClass = ["Not " + label[index], label[index]]
		file = fileLabel[index] + ".dot"
		print(confusion_matrix(labelTest[index], labelPredict, labels = [0, 1]))
		
		#tree.export_graphviz(treeClassifier, out_file=file, feature_names = attribute, class_names = diagramClass, filled=True, rounded=True, special_characters=True)
		#call(['dot', '-Tpng', file, '-o', label[index] + '.png', '-Gdpi=600'])
		
		#labelPredict = treeClassifier.predict(attributeTrain)
		##print("*" + str(accuracy_score(labelTrain[index], labelPredict) * 100))

		#analTree(treeClassifier, treeClassifier.decision_path(attributeTest), index)
		print()

	totalAccuracy /= 7
	print("Average accuracy: " + str(totalAccuracy))

def analTree(tree, nodePath, classIndex):
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
	        #print("%snode=%s leaf node." % (nodeDepth[i] * "\t", i))
	    else:
	        #print("%snode=%s test node: go to node %s if X[:, %s] <= %s else to "
	              "node %s."
	              % (nodeDepth[i] * "\t",
	                 i,
	                 leftChild[i],
	                 attribute[i],
	                 threshold[i],
	                 rightChild[i],
	                 ))
	'''
	totalMisclassify = np.zeros(shape=sum(isLeaf), dtype = np.int32)
	totalNode = np.zeros(shape=sum(isLeaf), dtype = np.int32)
	for testInstance in range(len(testData)):
		nodeIndex = nodePath.indices[nodePath.indptr[testInstance]: nodePath.indptr[testInstance + 1]]
		totalNode[sum(isLeaf[:nodeIndex[-1]])] += 1
		if (labelPredict[testInstance] != labelTest[index][testInstance]):
			totalMisclassify[sum(isLeaf[:nodeIndex[-1]])] += 1
	for i in range(sum(isLeaf)):
		print("Leaf node %d misclassified %d tests over %d tests" % 
			(i, totalMisclassify[i], totalNode[i]), end = ' ')
		if(totalNode[i] > 0): print("- Accuracy: %.3f" % (100.0 - (totalMisclassify[i] * 1.0 / totalNode[i] * 100.0)))		
		else: print()

validationAccuracy()