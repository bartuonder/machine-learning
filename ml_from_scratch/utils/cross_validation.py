"""K-fold partitions and cross-validated scores.

A score on a fold estimates predictive risk only when that fold was not
used to fit the model. Averaging the fold scores estimates the risk of
the training procedure, not of one particular fit.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any, Protocol

import numpy as np
from numpy.typing import ArrayLike, NDArray


class _Scorer(Protocol):
    """Estimator that can be refit and then scored on held-out rows."""

    def fit(self, X: ArrayLike, y: ArrayLike) -> Any:
        """Fit on a training fold."""

    def score(self, X: ArrayLike, y: ArrayLike) -> float:
        """Score a held-out fold."""


def k_fold_split(
    n_samples: int,
    n_splits: int = 5,
    shuffle: bool = True,
    random_state: int | None = None,
    y: ArrayLike | None = None,
) -> Iterator[tuple[NDArray[np.int64], NDArray[np.int64]]]:
    r"""Yield training and validation indices for each fold.

    The :math:`n` row indices are partitioned into :math:`K` folds whose
    sizes differ by at most one. Fold :math:`k` is the validation set and
    the other folds are the training set. The folds are disjoint and their
    union is every index.

    When ``y`` is given, the partition is stratified. Inside each class the
    rows are dealt round-robin across folds, so each fold's class counts
    differ from the others by at most one. Every class must then contain
    at least :math:`K` rows. Stratification requires ``shuffle=True``.

    Parameters
    ----------
    n_samples : int
        Number of rows.
    n_splits : int, default=5
        Number of folds :math:`K`. At least 2, and at most ``n_samples``.
    shuffle : bool, default=True
        Permute rows before cutting contiguous folds. Ignored as a separate
        switch when ``y`` is set: stratified folds are always shuffled
        inside each class.
    random_state : int or None, default=None
        Seed of the permutation.
    y : array-like of shape (n_samples,) or None, default=None
        Labels used only to balance class counts. ``None`` cuts the index
        vector without looking at a target.

    Yields
    ------
    train_idx, test_idx : ndarray of shape (n_train,) and (n_test,)
        Integer positions into the original row order.
    """

    if isinstance(n_samples, bool) or not isinstance(n_samples, int) or n_samples < 2:
        raise ValueError("n_samples must be an integer >= 2.")
    if isinstance(n_splits, bool) or not isinstance(n_splits, int) or not 2 <= n_splits <= n_samples:
        raise ValueError("n_splits must be an integer in 2..n_samples.")
    if not isinstance(shuffle, bool):
        raise TypeError("shuffle must be a bool.")
    rng = _generator(random_state)
    if y is None:
        indices = np.arange(n_samples, dtype=np.int64)
        if shuffle:
            rng.shuffle(indices)
        sizes = np.full(n_splits, n_samples // n_splits, dtype=np.int64)
        sizes[: n_samples % n_splits] += 1
        start = 0
        for size in sizes:
            stop = start + int(size)
            test_idx = indices[start:stop]
            train_idx = np.concatenate([indices[:start], indices[stop:]])
            yield train_idx, test_idx
            start = stop
        return

    if not shuffle:
        raise ValueError("Stratified K-fold requires shuffle=True.")
    labels = np.asarray(y).reshape(-1)
    if labels.shape[0] != n_samples:
        raise ValueError(
            "y must have one label per row. "
            f"Got {labels.shape[0]} labels and n_samples={n_samples}."
        )
    _classes, inverse = np.unique(labels, return_inverse=True)
    counts = np.bincount(inverse)
    if int(counts.min()) < n_splits:
        raise ValueError(
            "Stratified K-fold needs at least n_splits samples in every class."
        )
    fold_of_row = np.empty(n_samples, dtype=np.int64)
    for class_index in range(counts.size):
        members = np.flatnonzero(inverse == class_index)
        rng.shuffle(members)
        # Round-robin: sample t of a class goes to fold t mod K, so the
        # class counts across folds differ by at most one.
        fold_of_row[members] = np.arange(members.size, dtype=np.int64) % n_splits
    for fold in range(n_splits):
        test_idx = np.flatnonzero(fold_of_row == fold)
        train_idx = np.flatnonzero(fold_of_row != fold)
        yield train_idx, test_idx


def cross_val_score(
    estimator: _Scorer,
    X: ArrayLike,
    y: ArrayLike,
    n_splits: int = 5,
    shuffle: bool = True,
    random_state: int | None = None,
    stratify: bool = False,
) -> NDArray[np.float64]:
    r"""Refit ``estimator`` on each training fold and score the held-out fold.

    The returned vector has shape ``(n_splits,)``. Its mean is the
    K-fold estimate of the risk of the fitting procedure. The estimator
    is fitted in place: each call to ``fit`` replaces the previous fold's
    parameters, and the object is left fitted on the last training fold.

    Parameters
    ----------
    estimator : estimator
        Object with ``fit(X, y)`` and ``score(X, y)``. ``score`` defines
        the quantity being averaged.
    X : array-like of shape (n_samples, ...)
        Design. The first axis is the sample axis.
    y : array-like of shape (n_samples, ...)
        Targets aligned with ``X``.
    n_splits : int, default=5
        Number of folds.
    shuffle : bool, default=True
        Shuffle before assigning folds.
    random_state : int or None, default=None
        Seed of that shuffle.
    stratify : bool, default=False
        Balance class counts using ``y``. Requires ``shuffle=True`` and
        at least ``n_splits`` rows of every class.

    Returns
    -------
    ndarray of shape (n_splits,)
        Held-out score of each fold.
    """

    if not isinstance(stratify, bool):
        raise TypeError("stratify must be a bool.")
    features = np.asarray(X)
    targets = np.asarray(y)
    if features.shape[0] != targets.shape[0]:
        raise ValueError(
            "X and y must have the same number of rows. "
            f"Got {features.shape[0]} and {targets.shape[0]}."
        )
    scores = np.empty(n_splits, dtype=np.float64)
    folds = k_fold_split(
        int(features.shape[0]),
        n_splits=n_splits,
        shuffle=shuffle,
        random_state=random_state,
        y=targets if stratify else None,
    )
    for fold, (train_idx, test_idx) in enumerate(folds):
        estimator.fit(features[train_idx], targets[train_idx])
        scores[fold] = float(estimator.score(features[test_idx], targets[test_idx]))
    return scores


def _generator(random_state: int | None) -> np.random.Generator:
    """Return a generator local to this split."""

    if isinstance(random_state, bool) or not isinstance(random_state, (int, type(None))):
        raise TypeError("random_state must be an int or None.")
    return np.random.default_rng(random_state)
