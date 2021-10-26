import numpy as np
from proj1_helpers import *


#HELPERS
def calculate_mse(e):
    return 1/2*np.mean(e**2)


def compute_gradient(y, tx, w):
   #Compute the gradients
    err = y - tx.dot(w)
    grad = -tx.T.dot(err) / len(err)
    return grad, err

def sigmoid(t):
    return 1.0 / (1 + np.exp(-t))

def compute_loss_logistic(y, tx, w):
    pred = sigmoid(tx.dot(w))
    loss = y.T.dot(np.log(pred)) + (1 - y).T.dot(np.log(1 - pred))
    return np.squeeze(- loss)

def compute_gradient_logistic(y, tx, w):
    pred = sigmoid(tx.dot(w))
    grad = tx.T.dot(pred - y)
    return grad

def remap(y):
    return (y+1)/2


#ML METHODS
def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    #Gradient descent algorithm
    w = initial_w
    for n_iter in range(max_iters):
        # compute gradient
        grad, err = compute_gradient(y, tx, w)
        # gradient w update
        w = w - gamma * grad
        #print(f'w: {w}')
    loss = calculate_mse(err)
    return w,loss

def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    #Stochastic gradient descent
    w = initial_w
    batch_size = 1
    for n_iter in range(max_iters):
        for y_batch, tx_batch in batch_iter(y, tx, batch_size=batch_size, num_batches=1):
            # compute gradient
            grad, err = compute_gradient(y_batch, tx_batch, w)
            # update w
            w = w - gamma * grad
    # calculate loss
    loss = calculate_mse(err)
    return w, loss


def least_squares(y, tx):
    a = tx.T.dot(tx)
    b = tx.T.dot(y)
    w = np.linalg.solve(a, b)
    err = y - tx.dot(w)
    loss = calculate_mse(err)
    return w, loss

def ridge_regression(y, tx, lambda_):
    I = lambda_ * np.eye(tx.shape[1])
    a = tx.T.dot(tx) + I
    b = tx.T.dot(y)
    w = np.linalg.solve(a, b)
    err = y - tx.dot(w)
    loss = calculate_mse(err)
    return w, loss

def logistic_regression(y, tx, initial_w, max_iters, gamma):
    w=initial_w
    for n_iter in range(max_iters):
        # update w.
        grad = compute_gradient_logistic(y, tx, w)
        w = w - gamma * grad
        print(compute_loss_logistic(y,tx,w))
    loss = compute_loss_logistic(y, tx, w)
    return w, loss


def reg_logistic_regression(y,tx,lambda_,initial_w,max_iters,gamma):
    w=initial_w
    for n_iter in range(max_iters):
        # update w.
        grad = compute_gradient_logistic(y, tx, w) + 2 * lambda_ * w
        w = w - gamma * grad
    loss = compute_loss_logistic(y, tx, w) + lambda_ * np.squeeze(w.T.dot(w))
    return w, loss