import pandas as pd 
from csv import reader
pd.options.mode.chained_assignment = None

mainDF = pd.read_csv("dataset.csv", delimiter=";", skiprows = 1) 
mainDF = mainDF.drop(columns = ['ID', 'Motorway'])

mainDF.to_csv('preprocess.csv', index = False)

binCSV  = mainDF.iloc[:, 0:14]
binLabel = []
for index, row in mainDF.iterrows():
	binLabel.append(str(row["Green frogs"]) + str(row["Brown frogs"]) + str(row["Common toad"]) + str(row["Fire-bellied toad"])
	 + str(row["Tree frog"]) + str(row["Common newt"]) + str(row["Great crested newt"]))

#print(len(binLabel), len(set(binLabel)))
#for label in set(binLabel):
#	print(label, binLabel.count(label)) 

testingIndex = []
testingSet = []

for index, label in enumerate(binLabel):
	if(label not in set(testingSet)):
		testingIndex.append(index)
		testingSet.append(label)

for index, label in enumerate(binLabel):
	if(len(testingIndex) == 63): break
	if(index not in testingIndex and binLabel.count(label) > 3):
		testingIndex.append(index)

mainDF = pd.read_csv("dataset.csv", delimiter=";", skiprows = 1) 
mainDF = mainDF.drop(columns = ['ID', 'Motorway'])
mainDF.to_csv('preprocess.csv', index = False)
binCSV  = mainDF.iloc[:, 0:14]
binCSV["label"] = binLabel

mask = [(i in testingIndex) for i in range(189)]

(binCSV.loc[mask]).to_csv('binarytest.csv', index = False)

mask = [(i not in testingIndex) for i in range(189)]

(binCSV.loc[mask]).to_csv('binarytrain.csv', index = False)