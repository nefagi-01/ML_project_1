import numpy as np

def ridge_regression(y, tx, lamb):
    """implement ridge regression."""
    aI = 2* tx.shape[0] * lamb * np.identity(tx.shape[1])
    a = tx.T.dot(tx) + aI
    b = tx.T.dot(y)
    return np.linalg.solve(a, b)


#def logistic_regression(y, tx, initial w, max iters, gamma):
    """Logistic regression using gradient descent or SGD"""

#def reg_logistic_regression(y, tx, lambda , initial w, max iters, gamma):
    """Regularized logistic regression using gradient descent or SGD"""

