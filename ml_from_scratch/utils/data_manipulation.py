"""Train/test partitioning and affine feature standardization.

Splitting and standardizing are part of the statistical procedure, not
just preprocessing. A test fold estimates predictive risk only when the
fit, including every estimate of location and scale, is a function of
the training fold alone.
"""

from __future__ import annotations

from numpy.typing import ArrayLike, NDArray

import numpy as np


def _generator(random_state: int | None) -> np.random.Generator:
    """Return a local random generator.

    A generator owned by the call keeps ``train_test_split`` from drawing
    on NumPy's global stream, so other code can shuffle arrays without
    changing this split.
    """

    if isinstance(random_state, bool) or not isinstance(random_state, (int, type(None))):
        raise TypeError("random_state must be an int or None.")
    return np.random.default_rng(random_state)


def _test_count(n_samples: int, test_size: float | int) -> int:
    """Translate ``test_size`` into a number of held-out rows."""

    if isinstance(test_size, bool) or not isinstance(test_size, (int, float)):
        raise TypeError("test_size must be a float in (0, 1) or an int count.")
    if isinstance(test_size, float):
        if not 0.0 < test_size < 1.0:
            raise ValueError("A fractional test_size must lie in (0, 1).")
        # ceil so that a fraction of a row still holds out one row, and so
        # the test fold is never smaller than the requested fraction.
        n_test = int(np.ceil(test_size * n_samples))
    else:
        n_test = test_size
    if n_test <= 0 or n_test >= n_samples:
        raise ValueError(
            "The test fold must contain at least one row and must leave "
            "at least one row in the training fold. Got "
            f"n_test={n_test} for n_samples={n_samples}."
        )
    return n_test


def _allocate_test_counts(counts: NDArray[np.int64], n_test: int) -> NDArray[np.int64]:
    """Spread ``n_test`` slots across classes by the largest-remainder method.

    Class ``c`` has claim ``counts[c] / n * n_test``. The integer part of
    that claim is assigned first. Each leftover slot goes to the class
    with the largest unused fractional claim that still has a member
    available. The resulting counts sum to ``n_test`` and never exceed
    the class size, so the held-out fraction inside every class tracks
    the overall held-out fraction as closely as integers allow.
    """

    total = int(counts.sum())
    raw = counts * (n_test / total)
    allocation = np.floor(raw).astype(np.int64)
    leftover = int(n_test - int(allocation.sum()))
    fractional = raw - allocation
    # mergesort is stable, so equal fractional claims keep the class order
    # produced by ``np.unique`` and the split stays reproducible.
    order = np.argsort(-fractional, kind="mergesort")
    position = 0
    n_classes = int(counts.size)
    while leftover > 0:
        if position >= n_classes:
            if np.all(allocation >= counts):
                raise RuntimeError("Stratified allocation exceeded class counts.")
            position = 0
        class_index = int(order[position])
        position += 1
        if allocation[class_index] < counts[class_index]:
            allocation[class_index] += 1
            leftover -= 1
    return allocation


def _stratified_indices(
    stratify: NDArray[np.generic],
    n_test: int,
    rng: np.random.Generator,
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Build train and test index arrays with class proportions preserved."""

    _classes, inverse = np.unique(stratify, return_inverse=True)
    counts = np.bincount(inverse).astype(np.int64)
    allocation = _allocate_test_counts(counts, n_test)
    train_parts: list[NDArray[np.int64]] = []
    test_parts: list[NDArray[np.int64]] = []
    # The loop is over classes, not over rows. Each body slices an index
    # vector, so the design matrix is never walked entry by entry.
    for class_index, n_class_test in enumerate(allocation):
        members = np.flatnonzero(inverse == class_index).astype(np.int64, copy=False)
        rng.shuffle(members)
        cut = int(n_class_test)
        test_parts.append(members[:cut])
        train_parts.append(members[cut:])
    train_idx = np.concatenate(train_parts)
    test_idx = np.concatenate(test_parts)
    # Membership is already fixed. Shuffling each fold removes the class
    # blocks left by walking classes in sorted order; otherwise the first
    # rows of a fold would all belong to the first class.
    rng.shuffle(train_idx)
    rng.shuffle(test_idx)
    return train_idx, test_idx


def train_test_split(
    X: ArrayLike,
    y: ArrayLike,
    test_size: float | int = 0.2,
    random_state: int | None = None,
    shuffle: bool = True,
    stratify: ArrayLike | None = None,
) -> tuple[NDArray[np.generic], NDArray[np.generic], NDArray[np.generic], NDArray[np.generic]]:
    """Hold out a subset of rows for evaluation.

    If the rows are an i.i.d. sample from the distribution on which the
    model will be used, the empirical risk on the test fold is an unbiased
    estimate of the predictive risk of a model that was fit on the training
    fold only. That claim is false when the scaler, the feature map, or
    the model itself has seen the test rows.

    Parameters
    ----------
    X : array-like of shape (n_samples, ...)
        Design whose first axis indexes independent samples.
    y : array-like of shape (n_samples, ...)
        Targets aligned with the first axis of ``X``.
    test_size : float or int, default=0.2
        Fraction of rows to hold out, or an absolute count. A fraction
        uses ``ceil(test_size * n_samples)``.
    random_state : int or None, default=None
        Seed for the permutation. ``None`` draws a non-reproducible seed.
    shuffle : bool, default=True
        Permute rows before cutting. When ``False``, the test fold is the
        suffix of the array, which is the chronological holdout for data
        stored in time order.
    stratify : array-like of shape (n_samples,) or None, default=None
        Discrete labels used to match class proportions across the two
        folds. Pass the target itself to stratify on ``y``. Requires
        ``shuffle=True``.

    Returns
    -------
    X_train, X_test, y_train, y_test
        Copies of the selected rows. Integer indexing copies, so later
        in-place edits of a fold do not write into the other fold or
        into the caller's arrays.
    """

    if not isinstance(shuffle, bool):
        raise TypeError("shuffle must be a bool.")
    features = np.asarray(X)
    targets = np.asarray(y)
    if features.ndim == 0 or targets.ndim == 0:
        raise ValueError("X and y must each have a sample axis.")
    if features.shape[0] != targets.shape[0]:
        raise ValueError(
            "X and y must have the same number of rows. "
            f"Got {features.shape[0]} and {targets.shape[0]}."
        )
    n_samples = int(features.shape[0])
    if n_samples < 2:
        raise ValueError("At least two samples are required to form two folds.")
    n_test = _test_count(n_samples, test_size)
    rng = _generator(random_state)

    if stratify is not None:
        if not shuffle:
            raise ValueError(
                "stratify requires shuffle=True so each class can be "
                "permuted before its own cut."
            )
        labels = np.asarray(stratify)
        if labels.ndim != 1 or labels.shape[0] != n_samples:
            raise ValueError(
                "stratify must be a 1-d array with one label per sample."
            )
        train_idx, test_idx = _stratified_indices(labels, n_test, rng)
    else:
        # Permute a position vector, then apply that same permutation to
        # X and y. Shuffling either matrix alone would break row alignment.
        indices = np.arange(n_samples)
        if shuffle:
            rng.shuffle(indices)
        n_train = n_samples - n_test
        train_idx = indices[:n_train]
        test_idx = indices[n_train:]

    return (
        features[train_idx],
        features[test_idx],
        targets[train_idx],
        targets[test_idx],
    )


class StandardScaler:
    r"""Affine map sending each training column to mean 0 and variance 1.

    For a feature column :math:`x_j` the training empirical measure
    defines

    .. math::

        \mu_j = \frac{1}{n}\sum_{i=1}^{n} x_{ij},

    .. math::

        \sigma_j
        = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(x_{ij} - \mu_j)^2},

    and the stored transform is :math:`z_{ij} = (x_{ij} - \mu_j) / \sigma_j`.

    The divisor is :math:`n`, the maximum-likelihood Gaussian variance,
    because :math:`\mu_j` and :math:`\sigma_j` are plugs into a fixed
    affine map. They are the mean and standard deviation of the empirical
    training distribution. Bessel's :math:`n - 1` correction answers a
    different question (unbiased estimation of a population variance for
    a test) and is not used.

    A constant column has :math:`\sigma_j = 0`. Division by that estimate
    is undefined, so the scale is stored as 1. After centering, the column
    is the zero vector, which is the only value identified by the data.

    Fit this object on the training fold. Transform the test fold with
    the stored :math:`(\mu, \sigma)`. Recomputing those statistics on a
    matrix that still contains the test rows lets the test sample shift
    the origin and the unit of every coefficient.

    Attributes
    ----------
    mean_ : ndarray of shape (n_features,)
        Column means :math:`\mu`.
    scale_ : ndarray of shape (n_features,)
        Column standard deviations :math:`\sigma`, with exact zeros
        replaced by 1.
    n_features_in_ : int
        Number of columns seen in ``fit``.
    n_samples_seen_ : int
        Number of rows used to estimate ``mean_`` and ``scale_``.
    """

    def __init__(self) -> None:
        self.mean_: NDArray[np.float64] | None = None
        self.scale_: NDArray[np.float64] | None = None
        self.n_features_in_: int | None = None
        self.n_samples_seen_: int | None = None

    def fit(self, X: ArrayLike) -> StandardScaler:
        """Estimate per-column mean and population standard deviation.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training design. Integer input is upcast to float64 because
            centering large values in float32 cancels the significant
            digits that standardization is supposed to preserve.

        Returns
        -------
        self
        """

        design = self._validate_unfitted(X)
        # Mean down the sample axis. ``mean_`` has shape (n_features,) and
        # is the row vector μ^T that every sample will be compared against.
        self.mean_ = design.mean(axis=0)
        # ddof=0 divides by n. This is the root-mean-square deviation from
        # the empirical mean, not the unbiased sample standard deviation.
        scale = design.std(axis=0, ddof=0)
        self.scale_ = np.where(scale == 0.0, 1.0, scale)
        self.n_features_in_ = int(design.shape[1])
        self.n_samples_seen_ = int(design.shape[0])
        return self

    def transform(self, X: ArrayLike) -> NDArray[np.float64]:
        r"""Apply the stored map :math:`z = (x - \mu) / \sigma`.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Matrix whose columns use the same order as in ``fit``.

        Returns
        -------
        ndarray of shape (n_samples, n_features)
            Standardized rows.
        """

        design = self._validate_fitted(X)
        assert self.mean_ is not None and self.scale_ is not None
        # ``mean_`` and ``scale_`` have shape (n_features,). NumPy aligns
        # them with the last axis of ``design``, which is the feature axis.
        # The subtraction is the matrix form of X - 1 μ^T: one mean per
        # column, broadcast down every row. The division rescales each
        # column by its own σ and does not mix features.
        return (design - self.mean_) / self.scale_

    def fit_transform(self, X: ArrayLike) -> NDArray[np.float64]:
        """Fit on ``X`` and return the standardized training matrix."""

        return self.fit(X).transform(X)

    def inverse_transform(self, Z: ArrayLike) -> NDArray[np.float64]:
        r"""Invert the stored map, :math:`x = z \sigma + \mu`.

        On a column whose training standard deviation was 0, ``transform``
        wrote zeros and this method writes the original constant ``mean_``
        back. That column never carried more information than its mean.

        Parameters
        ----------
        Z : array-like of shape (n_samples, n_features)
            Standardized rows.

        Returns
        -------
        ndarray of shape (n_samples, n_features)
            Rows on the original scale.
        """

        standardized = self._validate_fitted(Z)
        assert self.mean_ is not None and self.scale_ is not None
        # Multiply by σ first, then add μ. Both parameter vectors broadcast
        # across rows, which is the inverse of the column-wise affine map.
        return standardized * self.scale_ + self.mean_

    def _validate_unfitted(self, X: ArrayLike) -> NDArray[np.float64]:
        design = np.asarray(X, dtype=np.float64)
        if design.ndim != 2:
            raise ValueError(
                "X must have shape (n_samples, n_features); "
                f"got ndim={design.ndim}."
            )
        if design.shape[0] < 1 or design.shape[1] < 1:
            raise ValueError("X must contain at least one row and one column.")
        if not np.isfinite(design).all():
            raise ValueError("X must contain only finite values.")
        return design

    def _validate_fitted(self, X: ArrayLike) -> NDArray[np.float64]:
        if self.mean_ is None or self.scale_ is None or self.n_features_in_ is None:
            raise RuntimeError("StandardScaler must be fitted before use.")
        design = np.asarray(X, dtype=np.float64)
        if design.ndim != 2:
            raise ValueError(
                "X must have shape (n_samples, n_features); "
                f"got ndim={design.ndim}."
            )
        if design.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Expected {self.n_features_in_} features, got {design.shape[1]}."
            )
        if not np.isfinite(design).all():
            raise ValueError("X must contain only finite values.")
        return design

    def __repr__(self) -> str:
        if self.n_features_in_ is None:
            return "StandardScaler()"
        return f"StandardScaler(n_features_in_={self.n_features_in_})"


def standardize(
    X: ArrayLike,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Standardize columns using statistics computed on this same matrix.

    This is an in-sample transform. Call it on a training matrix. To
    standardize held-out rows, use :class:`StandardScaler` fitted on the
    training fold and call ``transform`` on the rest.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Matrix to center and rescale.

    Returns
    -------
    Z : ndarray of shape (n_samples, n_features)
        Column-standardized copy of ``X``.
    mean : ndarray of shape (n_features,)
        Column means.
    scale : ndarray of shape (n_features,)
        Column standard deviations used in the division.
    """

    scaler = StandardScaler().fit(X)
    assert scaler.mean_ is not None and scaler.scale_ is not None
    return scaler.transform(X), scaler.mean_.copy(), scaler.scale_.copy()


def add_intercept(X: ArrayLike) -> NDArray[np.float64]:
    r"""Prepend a column of ones so an intercept is a coordinate of :math:`\theta`.

    The linear predictor with a bias is :math:`\theta_0 + x^\top\theta_{1:}`.
    Stacking a one in front of every row makes that map a single product
    :math:`\tilde{x}^\top\theta`, which is the form the normal equations
    and the gradient are written in.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Feature matrix, unmodified.

    Returns
    -------
    ndarray of shape (n_samples, n_features + 1)
        Design matrix whose column 0 is identically 1.
    """

    design = np.asarray(X, dtype=np.float64)
    if design.ndim != 2:
        raise ValueError(
            "X must have shape (n_samples, n_features); "
            f"got ndim={design.ndim}."
        )
    ones = np.ones((design.shape[0], 1), dtype=np.float64)
    # Concatenation allocates a new matrix. The caller's X is not written.
    return np.concatenate((ones, design), axis=1)
