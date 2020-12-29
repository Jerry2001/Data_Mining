import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import accuracy_score 
from sklearn.neural_network import MLPClassifier
import sys
import os

sys.path.append("../ANN")
sys.path.append("../DecisionTree")
sys.path.append("../SupportVectorMachine")

import binaryANNs 
import binaryDTs
import binarySVMs

print(binarySVMs.returnPredict())