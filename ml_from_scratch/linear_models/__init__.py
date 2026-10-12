"""Linear models estimated from an explicit empirical risk."""

from .linear_regression import LinearRegression
from .logistic_regression import LogisticRegression
from .softmax_regression import SoftmaxRegression

__all__ = ["LinearRegression", "LogisticRegression", "SoftmaxRegression"]
