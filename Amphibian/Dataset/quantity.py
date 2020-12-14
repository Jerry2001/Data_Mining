import pandas as pd
import numpy as np

testData = pd.read_csv("preprocesstest.csv", delimiter=",")
trainData = pd.read_csv("preprocesstrain.csv", delimiter=",")
labels = ("Green frog", "Brown frog", "Common toad", "Fire-bellied toad", "Tree frog", "Common newt", "Great crested newt")

print("Train Amphibian Quantity:")
for label in trainData.columns[-7:]:
	print("Numer of %s is %i" % (label, len(trainData[trainData[label] == 1])))

print("\nTest Amphibian Quantity:")
for label in testData.columns[-7:]:
	print("Numer of %s is %i" % (label, len(testData[testData[label] == 1])))

