import pandas as pd
pd.options.mode.chained_assignment = None

mainDF = pd.read_csv("dataset.csv", delimiter=";", skiprows = 1) 
mainDF = mainDF.drop(columns = ['ID', 'Motorway'])

mainDF.to_csv('preprocess.csv', index = False)

binCSV  = mainDF.iloc[:, 0:14]
binLabel = []
for index, row in mainDF.iterrows():
	binLabel.append(str(row["Green frogs"]) + str(row["Brown frogs"]) + str(row["Common toad"]) + str(row["Fire-bellied toad"])
	 + str(row["Tree frog"]) + str(row["Common newt"]) + str(row["Great crested newt"]))

binCSV["label"] = binLabel
binCSV.to_csv('binary.csv', index = False)
