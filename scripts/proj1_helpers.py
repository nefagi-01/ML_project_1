# -*- coding: utf-8 -*-
"""some helper functions for project 1."""
import csv
import numpy as np

def sigmoid(t):
    ##Resolve overflow
    sigm=np.zeros(t.shape)
    positive=t>=0
    negative=t<0
    sigm[positive]=1/(1+np.exp(-t[positive]))
    sigm[negative]=np.exp(t[negative])/(1+np.exp(t[negative]))
    return sigm

def load_csv_data(data_path, sub_sample=False):
    """Loads data and returns y (class labels), tX (features) and ids (event ids)"""
    y = np.genfromtxt(data_path, delimiter=",", skip_header=1, dtype=str, usecols=1)
    x = np.genfromtxt(data_path, delimiter=",", skip_header=1)
    ids = x[:, 0].astype(np.int)
    input_data = x[:, 2:]

    # convert class labels from strings to binary (-1,1)
    yb = np.ones(len(y))
    yb[np.where(y=='b')] = -1
    
    # sub-sample
    if sub_sample:
        yb = yb[::50]
        input_data = input_data[::50]
        ids = ids[::50]
        
    return yb, input_data, ids

def delete_columns_nulls(x,percentage):
    """remove the column contaning over a percentage of null values"""
    remove_features=[]
    for i in range(x.shape[1]):
        col=x[:,i]
        total=col.shape[0]
        counter_=collections.Counter(col)
        nulls=counter_[-999]
        null_percentage=round(nulls/total,2)
        print(f'NULL percentage is: {null_percentage}')
        if null_percentage>percentage:
            remove_features.append(i)
            
    x=np.delete(x,remove_features,1)
    return x

def remove_outliers(x):
    """remove the outliers in the code using the following formula and setting the outliers to the median"""
    k = 1
    for i in range(0,x.shape[1]):
        q1 = np.percentile(x[:,i],25)
        q2 = np.percentile(x[:,i],50)
        q3 = np.percentile(x[:,i],75)
        x[:,i][(x[:,i] < q1 - k*(q3-q1))] = q2
        x[:,i][(x[:,i] > q3 + k*(q3-q1))] = q2
    
    return x
        
        
def standardize(x):
    """standardize (feature scaling) the dataset x """
    for col in range(x.shape[1]):
        mean=np.mean(x[:,col])
        std=np.std(x[:,col])
        if std>0:
            x[:,col]=(x[:,col]-mean)/std
    return x
    


def predict_labels(weights, data, logistic=False):
    """Generates class predictions given weights, and a test data matrix"""
    #y_pred = sigmoid(np.dot(data, weights))
    if logistic==True:
        y_pred=sigmoid(data.dot(weights))
    else:
        y_pred=data.dot(weights)
    y_pred[np.where(y_pred <= 0.5)] = -1
    y_pred[np.where(y_pred > 0.5)] = 1
    
    return y_pred

def accuracy(y_pred,y_act):
    return y_pred[y_pred == y_act].shape[0]/y_act.shape[0]

def create_csv_submission(ids, y_pred, name):
    """
    Creates an output file in .csv format for submission to Kaggle or AIcrowd
    Arguments: ids (event ids associated with each prediction)
               y_pred (predicted class labels)
               name (string name of .csv output file to be created)
    """
    with open(name, 'w') as csvfile:
        fieldnames = ['Id', 'Prediction']
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({'Id':int(r1),'Prediction':int(r2)})
            
            
def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]
