#Testing all method converge to correct value
import numpy as np
import matplotlib.pyplot as plt
import collections
from proj1_helpers import *
from implementations import *
DATA_TRAIN_PATH = 'data\\dummy_data.csv'
data = np.loadtxt(DATA_TRAIN_PATH, delimiter=",", skiprows=1, unpack=True)
tX = data[0]
y = data[1]
tX = build_poly(tX, 1, False)

#GD
initial_w=np.zeros(tX.shape[1])
max_iters=200
gamma=1e-1
w,loss = least_squares_GD(y,tX,initial_w,max_iters,gamma)
print(f'GD:\n\tw:{w}\n\tloss:{loss}')

#SDG
initial_w=np.zeros(tX.shape[1])
max_iters=600
gamma=1e-3
w,loss = least_squares_SGD(y,tX,initial_w,max_iters,gamma)
print(f'SDG:\n\tw:{w}\n\tloss:{loss}')

#LS
w,loss = least_squares(y,tX)
print(f'LS:\n\tw:{w}\n\tloss:{loss}')

#RR
lambda_=0.1
w,loss = ridge_regression(y,tX,lambda_)
print(f'RR:\n\tw:{w}\n\tloss:{loss}')

#LR
initial_w=np.zeros(tX.shape[1])
max_iters=100
gamma=1e-5
w,loss = logistic_regression(y,tX,initial_w,max_iters,gamma)
print(f'LR:\n\tw:{w}\n\tloss:{loss}')

#RLR
initial_w=np.zeros(tX.shape[1])
max_iters=100
gamma=1e-5
lambda_=0.01
w,loss = reg_logistic_regression(y,tX, lambda_, initial_w, max_iters,gamma)
print(f'RLR:\n\tw:{w}\n\tloss:{loss}')