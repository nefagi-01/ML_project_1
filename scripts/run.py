#Reproducibility: In your submission, you must provide a script run.py which produces exactly the
#same .csv predictions which you used in your best submission to the competition system.

# Useful starting lines
%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
import collections
%load_ext autoreload
%autoreload 2

from proj1_helpers import *
from implementations import *
DATA_TRAIN_PATH = '../data/train.csv'
y, tX, ids = load_csv_data(DATA_TRAIN_PATH)


#we normalize the data ranging from [0,1] instead of [-1,1] since it is a binary prediction 
#and it fits the structure for the logistic regression
y = (y-min(y))/(max(y)-min(y))
tX = remove_columns_null(0.70)
tX = remove outliers
tX = standardize(tX)


##cross ridge



##cross log


##testing accuracy

#generate submission