import pandas as pd 
from csv import reader
pd.options.mode.chained_assignment = None

testingIndex = []
binLabel = []

def selectFromSimilarLabel(labelList):
	global testingIndex
	inRow = []
	for i in range(round(len(labelList) * 2.0 / 3.0)):
		if(len(labelList) == 0): break
		if i == 0: 
			testingIndex.append(labelList[i][-2])
			labelList.pop(i)
		else:
			maxDiff = -1
			maxDiffRow = []
			maxID = -1
			count = -1
			for testRow in labelList:
				diff = 0
				count += 1
				for curInRow in inRow:
					for index in range(int(len(curInRow))):
						diff += abs(curInRow[i] - testRow[i])
				if(diff > maxDiff):
					maxDiff = diff
					maxDiffRow = testRow
					maxID = count
			inRow.append(maxDiffRow)
			labelList.pop(maxID)
	for row in inRow:
		testingIndex.append(row[-2])

mainDF = pd.read_csv("dataset.csv", delimiter=";", skiprows = 1) 
mainDF = mainDF.drop(columns = ['ID', 'Motorway'])

mainDF.to_csv('preprocess.csv', index = False)

binCSV  = mainDF.iloc[:, 0:14]
binRow = []
for index, row in mainDF.iterrows():
	curBin = str(row["Green frogs"]) + str(row["Brown frogs"]) + str(row["Common toad"]) + str(row["Fire-bellied toad"]) + str(row["Tree frog"]) + str(row["Common newt"]) + str(row["Great crested newt"])
	binRow.append(list(row))
	binLabel. append(curBin)
	binRow[len(binRow) - 1].append(index)
	binRow[len(binRow) - 1].append(curBin)

binRow.sort(key=lambda instance: instance[len(instance) - 1])

curBin = ""
curRow = []
curDifference = []
for row in binRow:
	if(row[len(row) - 1] != curBin):
		if(len(curDifference) != 0):
			selectFromSimilarLabel(curDifference)
			curDifference = []
		curRow = row
		curDifference.append(row)
		curBin = row[len(row) - 1]
	else:
		curDifference.append(row) 

#print(len(binRow), len(set(binRow)))
#binRow.sort()
#print(binRow)
#for label in set(binLabel):
#	print(label, binLabel.count(label)) 
#print(testingIndex, len(testingIndex))

mainDF = pd.read_csv("dataset.csv", delimiter=";", skiprows = 1) 
mainDF = mainDF.drop(columns = ['ID', 'Motorway'])

binCSV  = mainDF.iloc[:, 0:14]
binCSV["label"] = binLabel
mask = [(i not in testingIndex) for i in range(189)]

(mainDF.loc[mask]).to_csv('preprocesstest.csv', index = False)
(binCSV.loc[mask]).to_csv('binarytest.csv', index = False)

mask = [(i in testingIndex) for i in range(189)]

(mainDF.loc[mask]).to_csv('preprocesstrain.csv', index = False)
(binCSV.loc[mask]).to_csv('binarytrain.csv', index = False)

(binCSV).to_csv('binary.csv', index = False)