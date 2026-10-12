"""Data handling and evaluation utilities built only on NumPy."""

from .cross_validation import cross_val_score, k_fold_split
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
    "cross_val_score",
    "confusion_matrix",
    "f1_score",
    "k_fold_split",
    "precision_score",
    "r2_score",
    "recall_score",
    "roc_auc_score",
    "standardize",
    "train_test_split",
]
