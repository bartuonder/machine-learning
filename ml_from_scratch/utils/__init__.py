"""Data handling and evaluation utilities built only on NumPy."""

from .data_manipulation import (
    StandardScaler,
    add_intercept,
    standardize,
    train_test_split,
)
from .metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)

__all__ = [
    "StandardScaler",
    "accuracy_score",
    "add_intercept",
    "confusion_matrix",
    "f1_score",
    "precision_score",
    "r2_score",
    "recall_score",
    "roc_auc_score",
    "standardize",
    "train_test_split",
]
