"""Principal components by SVD and by the covariance eigendecomposition.

Both paths diagonalize the same centered scatter matrix. SVD avoids forming
that matrix, whose condition number is the square of the condition number
of the centered data.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

SolverName = Literal["svd", "eigen"]


class PCA:
    r"""Orthogonal projection onto the leading principal subspace.

    Let :math:`X \in \mathbb{R}^{n \times d}` have rows :math:`x_i` and
    column mean :math:`\bar{x}`. The centered matrix is
    :math:`X_c = X - 1\bar{x}^\top`. The unbiased sample covariance is

    .. math::

        S = \frac{1}{n - 1} X_c^\top X_c \in \mathbb{R}^{d \times d}.

    A principal component is a unit eigenvector of :math:`S`. If
    :math:`S v_k = \lambda_k v_k` with :math:`\lambda_1 \ge \cdots`, the
    coordinate of a centered row on component :math:`k` is
    :math:`(x - \bar{x})^\top v_k`. Stacking the first :math:`r` eigenvectors
    as the rows of :math:`W \in \mathbb{R}^{r \times d}`, the projection is

    .. math::

        Z = X_c W^\top \in \mathbb{R}^{n \times r}.

    The eigenvalue :math:`\lambda_k` is the variance of column :math:`k` of
    :math:`Z` under the :math:`1/(n-1)` normalization. The explained
    variance ratio is :math:`\lambda_k / \sum_{j=1}^{d} \lambda_j`.

    **SVD.** The compact SVD :math:`X_c = U \Sigma V^\top` has right
    singular vectors equal to the eigenvectors of :math:`X_c^\top X_c`,
    and :math:`\lambda_k = \sigma_k^2 / (n - 1)`. This factorization does
    not form :math:`S`, so it does not square the condition number of
    :math:`X_c`. It is the default.

    **Eigendecomposition.** ``method="eigen"`` forms :math:`S` and calls
    a symmetric eigensolver. The eigenvalues and singular values describe
    the same subspace. Signs of the axes are fixed by making the largest
    absolute loading of each component positive, so the two solvers agree
    when the leading eigenvalues are distinct.

    Parameters
    ----------
    n_components : int, float, or None, default=None
        How many axes to keep. An int is that count. A float in ``(0, 1)``
        is the smallest count whose cumulative explained-variance ratio
        reaches the float. ``None`` keeps every positive-rank direction,
        at most ``min(n_samples, n_features)``.
    whiten : bool, default=False
        Scale each projected coordinate by :math:`1/\sqrt{\lambda_k}` so
        the columns of :math:`Z` have unit sample variance.
    method : {"svd", "eigen"}, default="svd"
        Which factorization produces the axes.

    Attributes
    ----------
    components_ : ndarray of shape (n_components, n_features)
        Rows are the principal axes :math:`v_k^\top`.
    explained_variance_ : ndarray of shape (n_components,)
        Eigenvalues :math:`\lambda_k` of the unbiased covariance.
    explained_variance_ratio_ : ndarray of shape (n_components,)
        :math:`\lambda_k` divided by the total variance.
    singular_values_ : ndarray of shape (n_components,)
        Singular values :math:`\sigma_k` of the centered data.
    mean_ : ndarray of shape (n_features,)
        Column mean removed before the factorization.
    noise_variance_ : float
        Mean of the discarded eigenvalues. Zero when every component is kept.
    n_components_ : int
        Rank actually retained.
    n_features_in_ : int
        Columns of ``X``.
    n_samples_ : int
        Rows of the last ``fit``.
    """

    def __init__(
        self,
        n_components: int | float | None = None,
        whiten: bool = False,
        method: SolverName = "svd",
    ) -> None:
        if n_components is not None and isinstance(n_components, bool):
            raise TypeError("n_components must be an int, a float, or None.")
        if isinstance(n_components, float) and not 0.0 < n_components < 1.0:
            raise ValueError("A fractional n_components must lie in (0, 1).")
        if isinstance(n_components, int) and n_components < 1:
            raise ValueError("n_components must be at least 1.")
        if not isinstance(whiten, bool):
            raise TypeError("whiten must be a bool.")
        if method not in ("svd", "eigen"):
            raise ValueError("method must be 'svd' or 'eigen'.")
        self.n_components = n_components
        self.whiten = whiten
        self.method: SolverName = method
        self.components_: NDArray[np.float64] | None = None
        self.explained_variance_: NDArray[np.float64] | None = None
        self.explained_variance_ratio_: NDArray[np.float64] | None = None
        self.singular_values_: NDArray[np.float64] | None = None
        self.mean_: NDArray[np.float64] | None = None
        self.noise_variance_: float | None = None
        self.n_components_: int | None = None
        self.n_features_in_: int | None = None
        self.n_samples_: int | None = None

    def fit(self, X: ArrayLike) -> PCA:
        """Estimate the principal axes of ``X``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data matrix. At least two rows are required because the
            unbiased covariance divides by :math:`n - 1`. A one-dimensional
            array is one feature column.

        Returns
        -------
        self
        """

        features = _validate_X(X)
        n_samples, n_features = int(features.shape[0]), int(features.shape[1])
        if n_samples < 2:
            raise ValueError("PCA requires at least two samples.")
        self.n_samples_ = n_samples
        self.n_features_in_ = n_features
        self.mean_ = features.mean(axis=0)
        # Broadcasting the mean across rows is X - 1 μ^T.
        centered = features - self.mean_
        if self.method == "svd":
            singular, axes = _svd_axes(centered)
            eigenvalues = singular**2 / (n_samples - 1)
        else:
            eigenvalues, axes, singular = _eigen_axes(centered, n_samples)
        total = float(np.sum(eigenvalues))
        rank = int(np.sum(eigenvalues > np.finfo(np.float64).eps * max(total, 1.0)))
        rank = max(rank, 1)
        n_keep = _component_count(self.n_components, eigenvalues, min(n_samples, n_features))
        self.n_components_ = n_keep
        self.components_ = _flip_signs(axes[:n_keep])
        self.explained_variance_ = eigenvalues[:n_keep].copy()
        if total <= 0.0:
            self.explained_variance_ratio_ = np.zeros(n_keep, dtype=np.float64)
        else:
            self.explained_variance_ratio_ = self.explained_variance_ / total
        self.singular_values_ = singular[:n_keep].copy()
        leftover = eigenvalues[n_keep:]
        self.noise_variance_ = float(np.mean(leftover)) if leftover.size else 0.0
        return self

    def transform(self, X: ArrayLike) -> NDArray[np.float64]:
        r"""Project rows onto the principal subspace.

        The map is :math:`Z = (X - 1\bar{x}^\top) W^\top`. With whitening,
        column :math:`k` is then divided by :math:`\sqrt{\lambda_k}`.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Rows on the same scale as ``fit``.

        Returns
        -------
        ndarray of shape (n_samples, n_components)
            Principal coordinates.
        """

        self._check_fitted()
        assert (
            self.components_ is not None
            and self.mean_ is not None
            and self.explained_variance_ is not None
            and self.n_features_in_ is not None
        )
        features = _validate_X(X, allow_single_row=True)
        if features.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Expected {self.n_features_in_} features, got {features.shape[1]}."
            )
        # (n, d) @ (d, r) = (n, r). Each column of the product is the
        # inner product of a centered row with one principal axis.
        projected = (features - self.mean_) @ self.components_.T
        if self.whiten:
            scale = np.sqrt(np.maximum(self.explained_variance_, 0.0))
            projected = projected / np.maximum(scale, np.finfo(np.float64).tiny)
        return projected

    def inverse_transform(self, Z: ArrayLike) -> NDArray[np.float64]:
        r"""Map principal coordinates back to the feature space.

        .. math::

            \hat{X} = Z W + 1\bar{x}^\top

        when whitening is off. Whitened coordinates are multiplied by
        :math:`\sqrt{\lambda_k}` first, which undoes the scaling in
        :meth:`transform`. Directions that were discarded are not recovered:
        :math:`\hat{X}` lies in the affine hull of the retained axes.

        Parameters
        ----------
        Z : array-like of shape (n_samples, n_components)
            Coordinates produced by :meth:`transform`.

        Returns
        -------
        ndarray of shape (n_samples, n_features)
            Reconstruction.
        """

        self._check_fitted()
        assert (
            self.components_ is not None
            and self.mean_ is not None
            and self.explained_variance_ is not None
        )
        scores = np.asarray(Z, dtype=np.float64)
        if scores.ndim != 2 or scores.shape[1] != self.components_.shape[0]:
            raise ValueError(
                "Z must have shape (n_samples, n_components); "
                f"got {scores.shape}."
            )
        if not np.isfinite(scores).all():
            raise ValueError("Z must contain only finite values.")
        if self.whiten:
            scores = scores * np.sqrt(np.maximum(self.explained_variance_, 0.0))
        # (n, r) @ (r, d) rebuilds the centered rows in the column space of W^T.
        return scores @ self.components_ + self.mean_

    def fit_transform(self, X: ArrayLike) -> NDArray[np.float64]:
        """Fit and return the projection of ``X``."""

        return self.fit(X).transform(X)

    def _check_fitted(self) -> None:
        if self.components_ is None:
            raise RuntimeError("PCA must be fitted before use.")

    def __repr__(self) -> str:
        return (
            f"PCA(n_components={self.n_components!r}, whiten={self.whiten}, "
            f"method={self.method!r})"
        )


def _svd_axes(
    centered: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    r"""Return singular values and principal axes from the centered data.

    ``full_matrices=False`` yields :math:`V^\top` of shape
    ``(min(n, d), d)``. Its rows are the eigenvectors of
    :math:`X_c^\top X_c`, ordered by decreasing singular value.
    """

    _u, singular, vt = np.linalg.svd(centered, full_matrices=False)
    return singular, vt


def _eigen_axes(
    centered: NDArray[np.float64],
    n_samples: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    r"""Return eigenvalues, axes, and singular values of the covariance.

    :math:`S = X_c^\top X_c / (n - 1)` is symmetric. ``eigh`` returns its
    eigenvalues in ascending order, so they are reversed. The matching
    singular value of :math:`X_c` is :math:`\sqrt{\lambda (n - 1)}`.
    """

    covariance = (centered.T @ centered) / (n_samples - 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 0.0)
    # Columns of the eigenvector matrix are axes. Transpose them into rows
    # so the layout matches the rows of V^T from the SVD path.
    axes = eigenvectors[:, order].T
    singular = np.sqrt(eigenvalues * (n_samples - 1))
    return eigenvalues, axes, singular


def _component_count(
    request: int | float | None,
    eigenvalues: NDArray[np.float64],
    maximum: int,
) -> int:
    """Translate ``n_components`` into a count no larger than ``maximum``."""

    available = int(eigenvalues.shape[0])
    limit = min(available, maximum)
    if request is None:
        return limit
    if isinstance(request, int) and not isinstance(request, bool):
        if request > limit:
            raise ValueError(f"n_components={request} exceeds the available rank {limit}.")
        return request
    total = float(np.sum(eigenvalues))
    if total <= 0.0:
        return 1
    cumulative = np.cumsum(eigenvalues) / total
    count = int(np.searchsorted(cumulative, float(request), side="left") + 1)
    return min(max(count, 1), limit)


def _flip_signs(components: NDArray[np.float64]) -> NDArray[np.float64]:
    """Make the largest-magnitude loading of each axis positive.

    Eigenvectors and singular vectors are defined up to a sign. A shared
    rule lets the SVD and the covariance eigensolver return the same
    matrix when the spectrum is simple.
    """

    if components.size == 0:
        return components
    pivot = np.argmax(np.abs(components), axis=1)
    signs = np.sign(components[np.arange(components.shape[0]), pivot])
    signs[signs == 0.0] = 1.0
    return components * signs[:, None]


def _validate_X(X: ArrayLike, allow_single_row: bool = False) -> NDArray[np.float64]:
    """Return a finite float64 design."""

    features = np.asarray(X, dtype=np.float64)
    if features.ndim == 1:
        features = features.reshape(-1, 1)
    if features.ndim != 2:
        raise ValueError(
            "X must have shape (n_samples, n_features); "
            f"got ndim={features.ndim}."
        )
    if features.shape[1] < 1 or features.shape[0] < 1:
        raise ValueError("X must contain at least one row and one column.")
    if not allow_single_row and features.shape[0] < 1:
        raise ValueError("X must contain at least one row.")
    if not np.isfinite(features).all():
        raise ValueError("X must contain only finite values.")
    return features
