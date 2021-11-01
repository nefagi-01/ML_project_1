import numpy as np
import matplotlib.pyplot as plt
import collections
from proj1_helpers import *
from implementations import *
DATA_TRAIN_PATH = 'train.csv'
y, tX, ids = load_csv_data(DATA_TRAIN_PATH)

#Subsample
N=1
y=y[::N]
tX=tX[::N,:]
print(f'Subsample: {N}\n\ttX shape:{tX.shape} y shape:{y.shape}')

#Transform y [-1,1] into [0,1]
y = (y-min(y))/(max(y)-min(y))

#Split dataset based on jet_num value. Preprocessing of the data including:
# - remove -999 columns when present
# - remove outliers
# - standardization 
tX_list,y_list=split_data(tX, y)


# Logistic Cross Validation - Best degree, gamma, and if interaction is needed
print('Logistic Cross Validation:')
degrees=np.arange(3)
k_fold=4
max_iters=100
gammas=np.logspace(-10,0,10)
interactions=[False,True]
tX_results_list_logistic=[]
for h,x in enumerate(tX_list):
    loss_tr_list=np.zeros((len(degrees),len(gammas),len(interactions)))
    loss_te_list=np.zeros((len(degrees),len(gammas),len(interactions)))
    print(f'\t-tX_{h} iteration')
    for i,D in enumerate(degrees):
        print(f'\t\tDegree: {i}')
        for k,interaction in enumerate(interactions):
            phi_x=build_poly(x, D, interaction)
            for j,gamma in enumerate(gammas):
                #compute loss with cross-validation
                loss_tr, loss_te=apply_cross_validation_logistic(y_list[h],phi_x,k_fold,max_iters,gamma,1)
                loss_tr_list[i,j,k]=loss_tr
                loss_te_list[i,j,k]=loss_te
    D_best_index, gamma_best_index, interaction_index=np.unravel_index(np.argmin(loss_te_list),loss_te_list.shape)
    gamma_best=gammas[gamma_best_index]
    D_best_logistic=degrees[D_best_index]
    interaction_logistic=interactions[interaction_index]
    print(f'\t\ttX_{h} Best degree logistic: {D_best_logistic}, best gamma logistic:{gamma_best}, interaction:{interaction_logistic}')
    tX_results_list_logistic.append({'D_best':D_best_logistic,'gamma_best':gamma_best,'interaction':interaction_logistic})

# Cross Validation Ridge Regression - Best degree, lambda_m, and if interaction is needed
print('Ridge Regression Cross Validation:')
degrees=np.arange(3)
lambdas=np.logspace(-10,0,10)
k_fold=4
interactions=[False,True]
tX_results_list_ridge=[]
for h,x in enumerate(tX_list):
    rmse_tr_list=np.zeros((len(degrees),len(lambdas),len(interactions)))
    rmse_te_list=np.zeros((len(degrees),len(lambdas),len(interactions)))
    print(f'\t- tX_{h} iteration')
    for i,D in enumerate(degrees):
        print(f'\t\tDegree: {D}')
        for k,interaction in enumerate(interactions):
            phi_x=build_poly(x, D, interaction)
            for j,lambda_ in enumerate(lambdas):
                #compute loss with cross-validation
                rmse_tr, rmse_te=apply_cross_validation(y_list[h],phi_x,k_fold,D,lambda_,1)
                rmse_tr_list[i,j,k]=rmse_tr
                rmse_te_list[i,j,k]=rmse_te
    D_best_index,lambda_best_index,interaction_index=np.unravel_index(np.argmin(rmse_te_list),rmse_te_list.shape)
    D_best_ridge=degrees[D_best_index]
    lambda_best_ridge=lambdas[lambda_best_index]
    interaction_ridge=interactions[interaction_index]
    print(f'\t\ttX_{h} Best degree ridge:{D_best_ridge}, best lambda_ ridge:{lambda_best_ridge}, interactions: {interaction_ridge}')
    tX_results_list_ridge.append({'D_best':D_best_ridge,'lambda_best':lambda_best_ridge,'interaction':interaction_ridge})

# Ridge Regression and Logistic accuracy on train data
weights_ridge=[]
weights_logistic=[]
for i,x in enumerate(tX_list):
    D_best_ridge=tX_results_list_ridge[i]['D_best']
    interaction_ridge=tX_results_list_ridge[i]['interaction']
    lambda_best_ridge=tX_results_list_ridge[i]['lambda_best']

    D_best_logistic=tX_results_list_logistic[i]['D_best']
    interaction_logistic=tX_results_list_logistic[i]['interaction']
    gamma_best=tX_results_list_logistic[i]['gamma_best']
    #Re-transform y data
    y_act=y_list[i]*2-1
    #Ridge
    phi_x_ridge=build_poly(x,D_best_ridge,interaction_ridge)
    w_ridge,_=ridge_regression(y_list[i],phi_x_ridge,lambda_best_ridge)
    weights_ridge.append(w_ridge)
    y_pred_ridge=predict_labels(w_ridge,phi_x_ridge)
    accuracy_ridge=accuracy(y_pred_ridge,y_act)
    #Logistic
    phi_x_logistic=build_poly(x, D_best_logistic, interaction_logistic)
    w_initial=np.zeros(phi_x_logistic.shape[1])
    max_iters=100
    w_logistic,loss_logistic=logistic_regression(y_list[i],phi_x_logistic,w_initial,max_iters,gamma_best)
    weights_logistic.append(w_logistic)
    y_pred_logistic=predict_labels(w_logistic,phi_x_logistic,logistic=True)
    accuracy_logistic=accuracy(y_pred_logistic,y_act)
    print(f'tX_{i} -- Accuracy ridge:{accuracy_ridge} Accuracy logistic:{accuracy_logistic}')

# Generate predictions and save ouput in csv format for submission:
DATA_TEST_PATH = 'test.csv'
_, tX_test, ids_test = load_csv_data(DATA_TEST_PATH)

#dummy predict for obtaining y_pred vector
w_dummy=np.zeros(tX_test.shape[1])
y_pred=predict_labels(w_dummy,tX_test)
tX_test_list,_=split_data(tX_test,y_pred,ignore_y=True,print_=False)

#Ridge submission
for i,x in enumerate(tX_results_list_ridge):
    D_best_ridge=tX_results_list_ridge[i]['D_best']
    interaction_ridge=tX_results_list_ridge[i]['interaction']
    lambda_best_ridge=tX_results_list_ridge[i]['lambda_best']

    tX_test_ridge=build_poly(tX_test_list[i],D_best_ridge,interaction_ridge)
    weights=weights_ridge[i]
    y_pred[tX_test[:,22]==i] = predict_labels(weights, tX_test_ridge)

OUTPUT_PATH = 'output_ridge.csv' # TODO: fill in desired name of output file for submission
create_csv_submission(ids_test, y_pred, OUTPUT_PATH)

#Logistic submission
for i,x in enumerate(tX_results_list_logistic):
    D_best_logistic=tX_results_list_logistic[i]['D_best']
    interaction_logistic=tX_results_list_logistic[i]['interaction']
    gamma_best=tX_results_list_logistic[i]['gamma_best']

    tX_test_logistic=build_poly(tX_test_list[i],D_best_logistic,interaction_logistic)
    weights=weights_logistic[i]
    y_pred[tX_test[:,22]==i]=predict_labels(weights,tX_test_logistic, logistic=True)
    
OUTPUT_PATH = 'output_logistic.csv' # TODO: fill in desired name of output file for submission
create_csv_submission(ids_test, y_pred, OUTPUT_PATH)