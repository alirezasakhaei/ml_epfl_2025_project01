import numpy as np
from typing import Tuple


def mean_squared_error_gd(
    y: np.ndarray,
    tx: np.ndarray,
    initial_w: np.ndarray,
    max_iters: int,
    gamma: float
) -> Tuple[np.ndarray, float]:
    """Gradient descent algorithm for MSE."""
    pass


def mean_squared_error_sgd(
    y: np.ndarray,
    tx: np.ndarray,
    initial_w: np.ndarray,
    max_iters: int,
    gamma: float
) -> Tuple[np.ndarray, float]:
    """Stochastic gradient descent algorithm for MSE."""
    pass


def least_squares(
    y: np.ndarray,
    tx: np.ndarray
) -> Tuple[np.ndarray, float]:
    """Least squares regression using normal equations."""
    pass


def ridge_regression(
    y: np.ndarray,
    tx: np.ndarray,
    lambda_: float
) -> Tuple[np.ndarray, float]:
    """Ridge regression using normal equations."""
    pass


def logistic_regression(
    y: np.ndarray,
    tx: np.ndarray,
    initial_w: np.ndarray,
    max_iters: int,
    gamma: float
) -> Tuple[np.ndarray, float]:
    """Logistic regression using gradient descent."""
    pass


def reg_logistic_regression(
    y: np.ndarray,
    tx: np.ndarray,
    lambda_: float,
    initial_w: np.ndarray,
    max_iters: int,
    gamma: float
) -> Tuple[np.ndarray, float]:
    """Regularized logistic regression using gradient descent."""
    pass