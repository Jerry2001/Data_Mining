from csv import reader
import pandas as pd

stdout = open('BinaryHashDecisionTree.txt', 'w')#open('ManyDecisionTree.txt', 'w')


def loadCSV(filename):
	file = open(filename, "rt")
	lines = reader(file)
	dataset = list(lines)
	return dataset

label = ["Water Reservoir Surface", "Number of Reservoir", "Type of Reservoir", "Presence of Vegetation", "The Most Dominant Land Type"
, "The Second Most Dominant Land Type", "The Third Most Dominant Land Type", "Use of Water Reservoir", "Presence of Fishing", "Precentage Access to Undeveloped Area"
, "Minimum Distance to Road", "Minimum Distance to Building", "Maintenance Status of Reservoir", "Type of Shore"]

group = ("Green frog", "Brown frog", "Common toad", "Fire-bellied toad", "Tree frog", "Common newt", "Great crested newt")

def toTerminal(group):
	outcomes = [row[-1] for row in group]
	return max(set(outcomes), key=outcomes.count)

def giniIndex(groups, classes):
	insNum = float(sum(len(group) for group in groups))
	gini = 0.0
	for group in groups:
		size = float(len(group))
		if size == 0:
			continue
		score = 0.0
		for classVal in classes:
			p = [row[-1] for row in group].count(classVal) / size
			score += p * p
		gini += (1.0 - score) * (size / insNum)
	return gini 

def testSplit(index, value, dataset):
	left, right = list(), list()
	for row in dataset:
		if row[index] < value:
			left.append(row)
		else: 
			right.append(row)
	return left, right

def split(node, maxDepth, minSize, depth):
	left, right = node['groups']
	del(node['groups'])

	if not left or not right:
		node['left'] = node['right'] = toTerminal(left + right)
		return 

	if depth >= maxDepth:
		node['left'], node['right'] = toTerminal(left), toTerminal(right)
		return 
	if len(left) <= minSize:
		node['left'] = toTerminal(left)
	else:
		node['left'] = getSplit(left)
		split(node['left'], maxDepth, minSize, depth + 1)
	if len(right) <= minSize:
		node['right'] = toTerminal(right)
	else:
		node['right'] = getSplit(right)
		split(node['right'], maxDepth, minSize, depth + 1)


def getSplit(dataset):
	classVal = list(set(row[-1] for row in dataset))
	bIndex, bValue, bScore, bGroups = 999, 999, 999, None
	for index in range(len(dataset[0]) - 1):
		for row in dataset:
			groups = testSplit(index, row[index], dataset)
			gini = giniIndex(groups, classVal)
			#print('X%d < %.3f Gini=%.3f' % ((index+1), row[index], gini))
			if gini < bScore:
				bIndex, bValue, bScore, bGroups = index, row[index], gini, groups
	instance = 0
	for group in bGroups:
		instance += len(group)
	return {'index': bIndex, 'value': bValue, 'groups':bGroups, 'gini' : bScore, 'instance': instance}

def buildTree(train, maxDepth, minSize):
	root = getSplit(train)
	split(root, maxDepth, minSize, 1)
	return root

def printTree(node, depth=0):
	global label
	if isinstance(node, dict):
		print('%s%s < %.3f | Gini:%.3f | Instance:%d' % (depth*'|'+'--', label[node['index']], node['value'], node['gini'], node['instance']), file = stdin)
		printTree(node['left'], depth + 1)
		printTree(node['right'], depth + 1)
	else: 
		print('%s[%s]' % (depth*'|'+'--', node), file = stdin)
 
def predict(tree, sample):
 	if(int(sample[tree['index']]) < tree['value']):
 		if(isinstance(tree['left'], dict)):
 			return predict(tree['left'], sample)
 		else: return tree['left']
 	else:
 		if(isinstance(tree['right'], dict)):
 			return predict(tree['right'], sample)
 		else: return tree['right']

def matchRate(predict, label):
	if(len(label) > 1): predict = (7 - len(str(predict)))*"0" + str(predict) 
	else: predict = str(predict)
	match = 0
	for i in range(min(len(predict), len(label))):
		if(predict[i] == label[i]): match += 1.0
	return (match / len(label))

def totalmatchRate(tree, testset):
	total = 0.0
	for sample in testset:
		predict_ = predict(tree, sample)
		label = sample[-1]
		total += matchRate(predict_, label)
	return total / len(testset)

def binaryHash():
	dataset = loadCSV("../Dataset/binarytrain.csv")
	dataset.pop(0)
	intDataset = [(list(map(int, group))) for group in dataset]

	testset = loadCSV("../Dataset/binarytest.csv")
	testset.pop(0)

	tree = buildTree(intDataset, 5, 0)
	printTree(tree)
	print('Final Accuracy %.3f' % totalmatchRate(tree, testset), file = stdin)

def multipleTree():
	dataset = loadCSV("../Dataset/preprocesstrain.csv")
	dataset.pop(0)
	intDataset = [(list(map(int, group))) for group in dataset]
	testset = loadCSV("../Dataset/preprocesstest.csv")
	testset.pop(0)
	sumMatchRate = 0
	for i in range(-7, 0):
		print('Decision Tree for %s' % group[i], file = stdin)
		trainSpeciesSet = [(group[0:14]) + [group[i]] for group in intDataset]
		testSpeciesSet = [(group[0:14]) + [group[i]] for group in testset]
		tree = buildTree(trainSpeciesSet, 5, 0)
		sumMatchRate += totalmatchRate(tree, testSpeciesSet)
		printTree(tree)
		print('Accuracy %.3f' % totalmatchRate(tree, testSpeciesSet), file = stdin)
		print(file = stdin)
	print('Final Accuracy %.3f' % (sumMatchRate / 7), file = stdin)
	print(file = stdin)

binaryHash()
multipleTree()