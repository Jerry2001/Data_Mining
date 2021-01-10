import pandas as pd

data = pd.read_csv("abalone.csv", delimiter=",")
data.insert(0, "I", 0)
data.insert(0, "F", 0)
data.insert(0, "M", 0)

for index, row in data.iterrows():
	genderCol = 0
	if(row.loc["Sex"] == "F"): genderCol = 1
	elif(row.loc["Sex"] =="I"): genderCol = 2  
	data.iloc[index, genderCol] = 1

data.drop("Sex", 1, inplace=True)
data.to_csv('onehot.csv', index = False)