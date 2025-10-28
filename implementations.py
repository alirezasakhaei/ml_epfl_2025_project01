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
    """Stochastic gradient descent algorithm for MSE."""
    pass


def least_squares(y: np.ndarray, tx: np.ndarray) -> Tuple[np.ndarray, float]:
    """Least squares regression using normal equations."""
    b = tx.T @ y
    A = tx.T @ tx
    w = np.linalg.solve(A, b)
    loss = np.mean((tx.dot(w) - y) ** 2) / 2
    return w, loss


def ridge_regression(
    y: np.ndarray, tx: np.ndarray, lambda_: float
) -> Tuple[np.ndarray, float]:
    """Ridge regression using normal equations."""
    b = tx.T @ y
    A = tx.T @ tx + lambda_ * np.eye(tx.shape[1])
    w = np.linalg.solve(A, b)
    loss = np.mean((tx.dot(w) - y) ** 2) / 2
    return w, loss


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
