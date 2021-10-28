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
    return np.reciprocal(1+np.exp(-t))

def compute_loss_logistic(y, tx, w):
    tmp = sigmoid(tx.dot(w))
    loss = y.T.dot(np.log(tmp)) + (1 - y).T.dot(np.log(1 - tmp))
    return -np.squeeze(loss)

def compute_gradient_logistic(y, tx, w):
    tmp = sigmoid(tx.dot(w))
    grad = tx.T.dot(tmp - y)
    return grad

def build_poly(x, degree):
    poly = np.ones((len(x), 1))
    for deg in range(1, degree+1):
        poly = np.c_[poly, np.power(x, deg)]
    return poly

def build_k_indices(y, k_fold, seed):
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval] for k in range(k_fold)]
    return np.array(k_indices)

def cross_validation(y, x, k_indices, k, lambda_, degree):
    # get k'th subgroup in test, others in train
    te_indice = k_indices[k]
    tr_indice = [y for i,x in enumerate(k_indices) for y in x if i!=k]
    y_te = y[te_indice]
    y_tr = y[tr_indice]
    x_te = x[te_indice]
    x_tr = x[tr_indice]
    # form data with polynomial degree
    tx_tr = build_poly(x_tr, degree)
    tx_te = build_poly(x_te, degree)
    # ridge regression
    w,_ = ridge_regression(y_tr, tx_tr, lambda_)
    # calculate the loss for train and test data
    e_tr = y_tr - tx_tr.dot(w)
    e_te = y_te - tx_te.dot(w)
    loss_tr = np.sqrt(2 * calculate_mse(e_tr))
    loss_te = np.sqrt(2 * calculate_mse(e_te))
    return loss_tr, loss_te,w

def apply_cross_validation(y,x,k_fold,degree,lambda_,seed):
    k_indices = build_k_indices(y, k_fold, seed)
    w_list=[]
    rmse_te_list=[]
    rmse_tr_list=[]
    for k in range(k_fold):
        loss_tr, loss_te, w = cross_validation(y, x, k_indices, k, lambda_, degree)
        w_list.append(w)
        rmse_te_list.append(loss_te)
        rmse_tr_list.append(loss_tr)
    rmse_te=np.mean(rmse_te_list)
    rmse_tr=np.mean(rmse_tr_list)
    return rmse_tr,rmse_te

def cross_validation_logistic(y, x, k_indices, k, degree, max_iters, gamma):
    # get k'th subgroup in test, others in train
    te_indice = k_indices[k]
    tr_indice = [y for i,x in enumerate(k_indices) for y in x if i!=k]
    y_te = y[te_indice]
    y_tr = y[tr_indice]
    x_te = x[te_indice]
    x_tr = x[tr_indice]
    # form data with polynomial degree
    tx_tr = build_poly(x_tr, degree)
    tx_te = build_poly(x_te, degree)
    initial_w=np.zeros(tx_tr.shape[1])
    # logistic regression
    w,_ = logistic_regression(y_tr, tx_tr, initial_w, max_iters, gamma)
    # calculate the loss for train and test data
    e_tr = y_tr - tx_tr.dot(w)
    e_te = y_te - tx_te.dot(w)
    loss_tr = np.sqrt(2 * calculate_mse(e_tr))
    loss_te = np.sqrt(2 * calculate_mse(e_te))
    return loss_tr, loss_te,w

def apply_cross_validation_logistic(y,x,k_fold,degree, max_iters, gamma,seed):
    k_indices = build_k_indices(y, k_fold, seed)
    w_list=[]
    rmse_te_list=[]
    rmse_tr_list=[]
    for k in range(k_fold):
        loss_tr, loss_te, w = cross_validation_logistic(y, x, k_indices, k, degree, max_iters, gamma)
        w_list.append(w)
        rmse_te_list.append(loss_te)
        rmse_tr_list.append(loss_tr)
    rmse_te=np.mean(rmse_te_list)
    rmse_tr=np.mean(rmse_tr_list)
    return rmse_tr,rmse_te



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