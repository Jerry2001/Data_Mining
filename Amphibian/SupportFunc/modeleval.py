import pandas as pd
import numpy as np

def kfold(data, n = -1):
	if(n == -1): n = len(data)
	index = 0
	step = len(data) // n
	group = []
	while index < len(data):
		group.append(data.iloc[index:([index + step, len(data)][index + step < len(data) and index + step * 2 > len(data)]), :])
		if(index + step < len(data) and index + step * 2 > len(data)): break
		index += step
	toReturn = []
	for i in range(len(group)):
		cummulate = group[i]
		for j in range(len(group)):
			if(j != i): pd.concat([cummulate, group[j]], axis = 1)
		print(cummulate)
data = pd.read_csv("../Dataset/binary.csv", delimiter=",") 
print(kfold(data, 3))