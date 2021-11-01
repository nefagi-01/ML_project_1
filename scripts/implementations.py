import numpy as np
from proj1_helpers import *
import collections

#HELPERS

def split_data(tX,y,ignore_y=False,print_=True):
    #split based on PRI_jet_num value
    tX_0=tX[tX[:,22]==0]
    tX_1=tX[tX[:,22]==1]
    tX_2=tX[tX[:,22]==2]
    tX_3=tX[tX[:,22]==3]
    y_list=None
    if print_:
        print('Subdatasets:')
    if not ignore_y:
        y_0=y[tX[:,22]==0]
        y_1=y[tX[:,22]==1]
        y_2=y[tX[:,22]==2]
        y_3=y[tX[:,22]==3]
        y_list=[y_0,y_1,y_2,y_3]
    tX_list=[tX_0,tX_1,tX_2,tX_3]
    #remove columns with all -999
    for j,x in enumerate(tX_list):
        remove_features=[22]
        for i in range(x.shape[1]):
            col=x[:,i]
            total=col.shape[0]
            counter_=collections.Counter(col)
            nulls=counter_[-999]
            if nulls==total:
                remove_features.append(i)
        tX_list[j]=np.delete(x,remove_features,1)
        if print_:
            print(f'\ttX_{j} shape: {tX_list[j].shape}')
            if not ignore_y:
                print(f'\ty_{j} shape: {y_list[j].shape}')
    #Remove outliers
    for j,x in enumerate(tX_list):
        k = 1
        for i in range(0,x.shape[1]):
            q1 = np.percentile(x[:,i],25)
            q2 = np.percentile(x[:,i],50)
            q3 = np.percentile(x[:,i],75)
            tX_list[j][:,i][(x[:,i] < q1 - k*(q3-q1))] = q2
            tX_list[j][:,i][(x[:,i] > q3 + k*(q3-q1))] = q2
    #Standardize
    for i,x in enumerate(tX_list):
        tX_list[i] = standardize(x)
    return tX_list, y_list


def calculate_mse(e):
    return 1/2*np.mean(e**2)


def compute_gradient(y, tx, w):
   #Compute the gradients
    err = y - tx.dot(w)
    grad = -tx.T.dot(err) / len(err)
    return grad, err

def sigmoid(t):
    ##Resolve overflow
    sigm=np.zeros(t.shape)
    positive=t>=0
    negative=t<0
    sigm[positive]=1/(1+np.exp(-t[positive]))
    sigm[negative]=np.exp(t[negative])/(1+np.exp(t[negative]))
    return sigm

def compute_loss_logistic(y, tx, w):
    #tmp = sigmoid(tx.dot(w))
    #loss = y.T.dot(np.log(tmp)) + (1 - y).T.dot(np.log(1 - tmp))
    #return -np.squeeze(loss)
    #Resolve overflow
    t=tx.dot(w)
    positive=t>=0
    negative=t<0
    first_term=np.sum(np.log(1+np.exp(-t[positive])))+np.sum(t[positive])+np.sum(np.log(1+np.exp(t[negative])))
    second_term=y.dot(tx.dot(w)) 
    loss=first_term-second_term
    return loss/y.shape[0]


def compute_gradient_logistic(y, tx, w):
    tmp = sigmoid(tx.dot(w))
    grad = tx.T.dot(tmp - y)
    return grad

def build_poly(x, degree, interaction):
    poly = np.ones((len(x), 1))
    for deg in range(1, degree+1):
        poly = np.c_[poly, np.power(x, deg)]
    ##interaction
    if interaction:
        for i in range(x.shape[1]):
            for j in range(i+1,x.shape[1]):
                interaction = x[:,i]*x[:,j]
                poly = np.c_[poly,interaction]
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
    tx_te = x[te_indice]
    tx_tr = x[tr_indice]
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
    rmse_te_list=[]
    rmse_tr_list=[]
    # form data with polynomial degree
    for k in range(k_fold):
        loss_tr, loss_te, w = cross_validation(y, x, k_indices, k, lambda_, degree)
        rmse_te_list.append(loss_te)
        rmse_tr_list.append(loss_tr)
    rmse_te=np.mean(rmse_te_list)
    rmse_tr=np.mean(rmse_tr_list)
    return rmse_tr,rmse_te

def cross_validation_logistic(y, x, k_indices, k, max_iters, gamma):
    # get k'th subgroup in test, others in train
    te_indice = k_indices[k]
    tr_indice = [y for i,x in enumerate(k_indices) for y in x if i!=k]
    y_te = y[te_indice]
    y_tr = y[tr_indice]
    tx_te = x[te_indice]
    tx_tr = x[tr_indice]
    initial_w=np.zeros(tx_tr.shape[1])
    # logistic regression
    w,_ = logistic_regression(y_tr, tx_tr, initial_w, max_iters, gamma)
    # calculate the loss for train and test data
    loss_tr = compute_loss_logistic(y_tr, tx_tr, w)
    loss_te = compute_loss_logistic(y_te, tx_te, w)
    return loss_tr, loss_te,w

def apply_cross_validation_logistic(y,x,k_fold, max_iters, gamma,seed):
    k_indices = build_k_indices(y, k_fold, seed)
    loss_te_list=[]
    loss_tr_list=[]
    # form data with polynomial degree
    #phi_x=build_poly(x, degree, interaction)
    for k in range(k_fold):
        loss_tr, loss_te, w = cross_validation_logistic(y, x, k_indices, k, max_iters, gamma)
        loss_te_list.append(loss_te)
        loss_tr_list.append(loss_tr)
    loss_te=np.mean(loss_te_list)
    loss_tr=np.mean(loss_tr_list)
    return loss_tr,loss_te



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
    aI = 2 * tx.shape[0] * lambda_ * np.identity(tx.shape[1])
    a = tx.T.dot(tx) + aI
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