from csv import reader
import pandas as pd
import random

k = 10
ordinal = [1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0]

trainData = pd.read_csv("../Dataset/binarytrain.csv", delimiter=",")

def distance(train, test):
	d = 0
	for i in range(len(train) - 1):
		if(ordinal[i] == 1): 
			d += abs(train.iloc[i] - test.iloc[i])
		else:
			d += int(train.iloc[i] == test.iloc[i])
	return d

def analLabel():
	data = pd.read_csv("../Dataset/binary.csv", delimiter=",")
	count = [0, 0]
	for label in list(data.iloc[:, -1]):
		label = str(int(label))
		label = (7 - len(label)) * "0" + label
		for bin in label:
			count[int(bin)] += 1
	print(count)

def accuracyCalc(label, predict):
	match = 0
	label = str(int(label))
	label = (7 - len(label)) * "0" + label
	for i in range(len(label)):
		if(label[i] == predict[i]): match += 1
	return (match / 7.0)

def construct(key):
	ans = ""

	for i in range(len(key)):
		key[i] = str(int(key[i]))
		key[i] = (7 - len(key[i])) * "0" + key[i]
	#print(key)
	for species in range(len(key[0])):
		count = [0, 0]
		for i in range(len(key)):
			count[int(key[i][species])] += 1
		if(count[0] == count[1]):
			if(random.randrange(0, 10) < 4): ans += "1" 
			else: ans += "0"  
		if(count[0] < count[1]):
			ans += "1"
		else: ans += "0"
		#print(count)
	return ans
	
for column in (trainData.columns):
	if(column == "label"): break
	trainData[column] = trainData[column].apply(lambda x: (x*1.0 - trainData[column].min()) / 
		(trainData[column].max() - trainData[column].min()))

testData = pd.read_csv("../Dataset/binarytest.csv", delimiter=",") 

for column in (testData.columns):
	if(column == "label"): break
	testData[column] = testData[column].apply(lambda x: (x*1.0 - testData[column].min()) / 
		(testData[column].max() - testData[column].min()))

distance(trainData.iloc[0, :], testData.iloc[0, :])

accuracy = 0

for testIndex, test in testData.iterrows():
	pair = []
	for trainIndex, train in trainData.iterrows():
		pair.append([trainIndex, distance(test, train)])
	pair.sort(key= lambda x : x[-1])
	key = list(trainData.iloc[i, :].iloc[-1] for i in range(k))
	predict = construct(key)
	accuracy += accuracyCalc(test.iloc[-1], predict)
	#print(accuracyCalc(test.iloc[-1], predict))

print(accuracy * 100 / len(testData))