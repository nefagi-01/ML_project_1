import numpy as np
import matplotlib.pyplot as plt
import collections
from proj1_helpers import *
from implementations import *
print('Loading data:')
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
print('\nLogistic Cross Validation:')
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

#Logistic accuracy on train data
weights_logistic=[]
print('\nLogistic accuracy on train data:')
for i,x in enumerate(tX_list):
    D_best_logistic=tX_results_list_logistic[i]['D_best']
    interaction_logistic=tX_results_list_logistic[i]['interaction']
    gamma_best=tX_results_list_logistic[i]['gamma_best']
    #Re-transform y data
    y_act=y_list[i]*2-1
    #Logistic
    phi_x_logistic=build_poly(x, D_best_logistic, interaction_logistic)
    w_initial=np.zeros(phi_x_logistic.shape[1])
    max_iters=100
    w_logistic,loss_logistic=logistic_regression(y_list[i],phi_x_logistic,w_initial,max_iters,gamma_best)
    weights_logistic.append(w_logistic)
    y_pred_logistic=predict_labels(w_logistic,phi_x_logistic,logistic=True)
    accuracy_logistic=accuracy(y_pred_logistic,y_act)
    print(f'tX_{i} -- Accuracy logistic:{accuracy_logistic}')

# Generate predictions and save ouput in csv format for submission:
DATA_TEST_PATH = 'test.csv'
_, tX_test, ids_test = load_csv_data(DATA_TEST_PATH)

#dummy predict for obtaining y_pred vector
w_dummy=np.zeros(tX_test.shape[1])
y_pred=predict_labels(w_dummy,tX_test)
tX_test_list,_=split_data(tX_test,y_pred,ignore_y=True,print_=False)

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
print('\nSubmission output_logistic.csv created.') 
