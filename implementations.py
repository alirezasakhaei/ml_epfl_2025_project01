import numpy as np
from typing import Tuple


def mean_squared_error_gd(
    y: np.ndarray, tx: np.ndarray, initial_w: np.ndarray, max_iters: int, gamma: float
) -> Tuple[np.ndarray, float]:
    """Gradient descent algorithm for MSE."""
    w = initial_w
    loss = (np.mean((tx.dot(w) - y) ** 2)) / 2
    for _ in range(max_iters):
        gradient = tx.T.dot(tx.dot(w) - y) / len(y)
        w = w - gamma * gradient
        loss = (np.mean((tx.dot(w) - y) ** 2)) / 2
    return w, loss




def mean_squared_error_sgd(
    y: np.ndarray, tx: np.ndarray, initial_w: np.ndarray, max_iters: int, gamma: float
) -> Tuple[np.ndarray, float]:
    w = initial_w
    n_samples = len(y)
    
    for _ in range(max_iters):
        rand_order = np.random.permutation(n_samples)
        for i in rand_order:
            gradient = tx[i].T.dot(tx[i].dot(w) - y[i])
            w = w - gamma * gradient
    
    loss = np.mean((y - tx.dot(w)) ** 2) / 2
    return w, loss


def least_squares(y: np.ndarray, tx: np.ndarray) -> Tuple[np.ndarray, float]:
    """Least squares regression using normal equations."""
    pass


def ridge_regression(
    y: np.ndarray, tx: np.ndarray, lambda_: float
) -> Tuple[np.ndarray, float]:
    """Ridge regression using normal equations."""
    pass


def logistic_regression(
    y: np.ndarray, tx: np.ndarray, initial_w: np.ndarray, max_iters: int, gamma: float
) -> Tuple[np.ndarray, float]:
    """Logistic regression using gradient descent."""
    pass


def reg_logistic_regression(
    y: np.ndarray,
    tx: np.ndarray,
    lambda_: float,
    initial_w: np.ndarray,
    max_iters: int,
    gamma: float,
) -> Tuple[np.ndarray, float]:
    """Regularized logistic regression using gradient descent."""
    pass
