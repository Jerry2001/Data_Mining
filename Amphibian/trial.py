import pandas as pd
import numpy as np
from sklearn.model_selection import KFold

data = pd.read_csv("Dataset/binary.csv", delimiter=",")
kf = KFold(n_splits = len(data), random_state = None, shuffle = False)

for train_index, test_index in kf.split(data):
	print(data.iloc[train_index], data.iloc[test_index])
	