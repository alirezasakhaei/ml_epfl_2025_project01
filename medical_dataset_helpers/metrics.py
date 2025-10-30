import numpy as np


def accuracy_score(y_true, y_pred, normalize=True):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    
    score = np.sum(y_true == y_pred)
    
    if normalize:
        score = score / len(y_true)
    
    return score


def mean_squared_error(y_true, y_pred, squared=True):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    
    mse = np.mean((y_true - y_pred) ** 2)
    
    if squared:
        return mse
    else:
        return np.sqrt(mse)

