import pandas as pd 

mainDF = pd.read_csv("preprocess.csv", delimiter=",")

hotDF = pd.DataFrame(columns = mainDF.iloc[:,:14].columns )

hotDF.insert(len(hotDF.columns), 'Amphibian', [])

for index, row in mainDF.iterrows():
	flag = 0
	for label in row.iloc[-7:].index:
		if(row[label] != 0):
			temp = (row.iloc[0:14]).copy()
			temp['Amphibian'] = label
			hotDF = hotDF.append(temp, ignore_index = True)
			flag += 1
	if(flag == 0):
		hotDF = hotDF.append(row.iloc[0:14], ignore_index = True)

hotDF.to_csv('onehot.csv', index = False)