from csv import reader
import pandas as pd

def load_csv(filename):
	file = open(filename, "rt")
	lines = reader(file)
	dataset = list(lines)
	return dataset

def to_terminal(group):
	outcomes = [row[-1] for row in group]
	return max(set(outcomes), key=outcomes.count)

def gini_index(groups, classes):
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

def test_split(index, value, dataset):
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
		node['left'] = node['right'] = to_terminal(left + right)
		return 

	if depth >= maxDepth:
		node['left'], node['right'] = to_terminal(left), to_terminal(right)
		return 
	if len(left) <= minSize:
		node['left'] = to_terminal(left)
	else:
		node['left'] = get_split(left)
		split(node['left'], maxDepth, minSize, depth + 1)
	if len(right) <= minSize:
		node['right'] = to_terminal(right)
	else:
		node['right'] = get_split(right)
		split(node['right'], maxDepth, minSize, depth + 1)


def get_split(dataset):
	classVal = list(set(row[-1] for row in dataset))
	bIndex, bValue, bScore, bGroups = 999, 999, 999, None
	for index in range(len(dataset[0]) - 1):
		for row in dataset:
			groups = test_split(index, row[index], dataset)
			gini = gini_index(groups, classVal)
			#print('X%d < %.3f Gini=%.3f' % ((index+1), row[index], gini))
			if gini < bScore:
				bIndex, bValue, bScore, bGroups = index, row[index], gini, groups
	return {'index': bIndex, 'value': bValue, 'groups':bGroups}

def build_tree(train, maxDepth, minSize):
	root = get_split(train)
	split(root, maxDepth, minSize, 1)
	return root

def print_tree(node, depth=0):
	if isinstance(node, dict):
		print('%s[X%d < %.3f]' % (depth*' ', (node['index'] + 1), node['value']))
		print_tree(node['left'], depth + 1)
		print_tree(node['right'], depth + 1)
	else: 
		print('%s[%s]' % (depth * ' ', node))
 
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
	predict = (7 - len(str(predict)))*"0" + str(predict) 
	match = 0
	for i in range(7):
		if(predict[i] == label[i]): match += 1.0
	return (match / 7.0)

def totalmatchRate(tree):
	total = 0.0
	for sample in testset:
		predict_ = predict(tree, sample)
		label = sample[-1]
		total += matchRate(predict_, label)
	return total / len(testset)

dataset = load_csv("binarytrain.csv")
dataset.pop(0)
intDataset = [(list(map(int, group))) for group in dataset]

testset = load_csv("binarytest.csv")
testset.pop(0)

tree = build_tree(intDataset, 30, 0)
print(totalmatchRate(tree))