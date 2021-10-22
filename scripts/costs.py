import numpy as np


def calculate_mse(e):
    """Calculate the mse for vector e."""
    n = len(e)
    return 1/(2*n)*np.dot(e,e.T)


def calculate_mae(e):
    """Calculate the mae for vector e."""
    return np.mean(np.abs(e))


def compute_loss(y, tx, w):
    """Calculate the loss.

    You can calculate the loss using mse or mae.
    """
    e = y - tx.dot(w)
    return calculate_mse(e)
    # return calculate_mae(e)
