"""Gaussian discriminant analysis, shared or per-class covariance.

The class-conditional density is Gaussian. Bayes' rule turns the fitted
densities into a classifier. A shared covariance makes the decision
boundary linear, the same function class as logistic regression, with
the generative rather than the discriminative MLE.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..utils.metrics import accuracy_score

CovarianceMode = Literal["shared", "separate"]


class GaussianDiscriminantAnalysis:
    r"""Generative classifier with Gaussian class-conditional densities.

    Labels follow a categorical distribution,
    :math:`P(y = k) = \pi_k`. Given the label,

    .. math::

        x \mid y = k \sim \mathcal{N}(\mu_k, \Sigma_k).

    The posterior is Bayes' rule. In log space the unnormalized score of
    class :math:`k` is

    .. math::

        \delta_k(x)
        = \log \pi_k
          - \tfrac{1}{2}\log\det\Sigma_k
          - \tfrac{1}{2}(x - \mu_k)^\top \Sigma_k^{-1}(x - \mu_k).

    ``predict_proba`` normalizes :math:`\exp(\delta_k)` across classes.
    The maximum is the Bayes classifier under this model.

    **Shared covariance.** ``covariance="shared"`` sets every
    :math:`\Sigma_k` equal to the pooled MLE

    .. math::

        \Sigma
        = \frac{1}{n}\sum_{k=1}^{K}
          \sum_{i: y_i = k}(x_i - \mu_k)(x_i - \mu_k)^\top
          + \epsilon I.

    The quadratic term :math:`x^\top\Sigma^{-1}x` no longer depends on
    :math:`k`, so :math:`\delta_k` collapses to a linear function

    .. math::

        \delta_k(x)
        = x^\top\Sigma^{-1}\mu_k
          - \tfrac{1}{2}\mu_k^\top\Sigma^{-1}\mu_k
          + \log\pi_k.

    That is the same family of decision boundaries as multinomial logistic
    regression. The parameters are different: GDA maximizes the joint
    likelihood :math:`p(x, y)`, while softmax regression maximizes the
    conditional likelihood :math:`p(y \mid x)`.

    **Separate covariances.** ``covariance="separate"`` estimates a
    covariance per class (quadratic discriminant analysis). The boundary
    between two classes is quadratic.

    :math:`\mu_k` is the class mean and :math:`\pi_k = n_k / n`. The
    divisor in :math:`\Sigma` is :math:`n`, the joint Gaussian MLE, not
    :math:`n - K`.

    Parameters
    ----------
    covariance : {"shared", "separate"}, default="shared"
        One pooled covariance, or one covariance per class.
    reg_covar : float, default=1e-6
        Ridge :math:`\epsilon` added to every covariance.

    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
        Sorted training labels.
    priors_ : ndarray of shape (n_classes,)
        :math:`\pi_k`.
    means_ : ndarray of shape (n_classes, n_features)
        :math:`\mu_k`.
    covariance_ : ndarray of shape (n_features, n_features) or None
        Pooled :math:`\Sigma` when ``covariance="shared"``.
    covariances_ : ndarray of shape (n_classes, n_features, n_features)
        The matrix used for class :math:`k`. Copies of ``covariance_``
        when the covariance is shared.
    linear_coef_ : ndarray of shape (n_classes, n_features) or None
        :math:`\Sigma^{-1}\mu_k` when the covariance is shared.
    linear_intercept_ : ndarray of shape (n_classes,) or None
        :math:`-\frac{1}{2}\mu_k^\top\Sigma^{-1}\mu_k + \log\pi_k`
        when the covariance is shared.
    n_features_in_ : int
        Columns of ``X``.
    """

    def __init__(
        self,
        covariance: CovarianceMode = "shared",
        reg_covar: float = 1e-6,
    ) -> None:
        if covariance not in ("shared", "separate"):
            raise ValueError("covariance must be 'shared' or 'separate'.")
        self.covariance: CovarianceMode = covariance
        self.reg_covar = _positive_float("reg_covar", reg_covar)
        self.classes_: NDArray[Any] | None = None
        self.priors_: NDArray[np.float64] | None = None
        self.means_: NDArray[np.float64] | None = None
        self.covariance_: NDArray[np.float64] | None = None
        self.covariances_: NDArray[np.float64] | None = None
        self.linear_coef_: NDArray[np.float64] | None = None
        self.linear_intercept_: NDArray[np.float64] | None = None
        self.n_features_in_: int | None = None

    def fit(self, X: ArrayLike, y: ArrayLike) -> GaussianDiscriminantAnalysis:
        """Estimate priors, means, and covariances.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training design. A one-dimensional array is one feature column.
        y : array-like of shape (n_samples,)
            Class labels.

        Returns
        -------
        self
        """

        features, labels = _validate_xy(X, y)
        classes, codes = np.unique(labels, return_inverse=True)
        n_samples, n_features = features.shape
        n_classes = int(classes.size)
        counts = np.bincount(codes, minlength=n_classes).astype(np.float64)
        if np.any(counts < 1.0):
            raise ValueError("Every class must contain at least one sample.")
        priors = counts / n_samples
        means = np.zeros((n_classes, n_features), dtype=np.float64)
        eye = np.eye(n_features, dtype=np.float64)
        scatter = np.zeros((n_features, n_features), dtype=np.float64)
        covariances = np.empty((n_classes, n_features, n_features), dtype=np.float64)
        for class_index in range(n_classes):
            members = features[codes == class_index]
            means[class_index] = members.mean(axis=0)
            centered = members - means[class_index]
            # Gram matrix of the centered class. Shape (d, d).
            class_scatter = centered.T @ centered
            if self.covariance == "separate":
                covariances[class_index] = class_scatter / counts[class_index] + self.reg_covar * eye
            else:
                scatter += class_scatter
        self.classes_ = classes
        self.priors_ = priors
        self.means_ = means
        self.n_features_in_ = int(n_features)
        if self.covariance == "shared":
            pooled = scatter / n_samples + self.reg_covar * eye
            self.covariance_ = pooled
            self.covariances_ = np.repeat(pooled[None, :, :], n_classes, axis=0)
            # Σ^{-1} μ_k as rows. solve(Σ, μ^T) avoids forming the inverse.
            solved = np.linalg.solve(pooled, means.T).T
            self.linear_coef_ = solved
            self.linear_intercept_ = -0.5 * np.sum(means * solved, axis=1) + np.log(priors)
        else:
            self.covariance_ = None
            self.covariances_ = covariances
            self.linear_coef_ = None
            self.linear_intercept_ = None
        return self

    def predict_proba(self, X: ArrayLike) -> NDArray[np.float64]:
        """Return :math:`P(y = k \\mid x)` under the fitted Gaussians.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Rows on the scale used in ``fit``.

        Returns
        -------
        ndarray of shape (n_samples, n_classes)
            Posterior probabilities. Each row sums to 1.
        """

        features = self._validate_fitted(X)
        assert self.priors_ is not None and self.means_ is not None and self.covariances_ is not None
        n_samples = int(features.shape[0])
        n_classes = int(self.priors_.shape[0])
        log_joint = np.empty((n_samples, n_classes), dtype=np.float64)
        log_prior = np.log(self.priors_)
        for class_index in range(n_classes):
            log_joint[:, class_index] = log_prior[class_index] + _log_gaussian(
                features,
                self.means_[class_index],
                self.covariances_[class_index],
            )
        log_marginal = _logsumexp(log_joint, axis=1)
        return np.exp(log_joint - log_marginal[:, None])

    def predict(self, X: ArrayLike) -> NDArray[Any]:
        """Return the class with the largest posterior probability.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Rows on the scale used in ``fit``.

        Returns
        -------
        ndarray of shape (n_samples,)
            Labels from ``classes_``.
        """

        self._check_fitted()
        assert self.classes_ is not None
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

    def score(self, X: ArrayLike, y: ArrayLike) -> float:
        """Return the classification accuracy of ``predict(X)``."""

        return accuracy_score(y, self.predict(X))

    def _validate_fitted(self, X: ArrayLike) -> NDArray[np.float64]:
        self._check_fitted()
        assert self.n_features_in_ is not None
        features = _validate_X(X)
        if features.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Expected {self.n_features_in_} features, got {features.shape[1]}."
            )
        return features

    def _check_fitted(self) -> None:
        if self.means_ is None or self.classes_ is None:
            raise RuntimeError("GaussianDiscriminantAnalysis must be fitted before use.")

    def __repr__(self) -> str:
        return f"GaussianDiscriminantAnalysis(covariance={self.covariance!r})"


def _log_gaussian(
    features: NDArray[np.float64],
    mean: NDArray[np.float64],
    covariance: NDArray[np.float64],
) -> NDArray[np.float64]:
    r"""Return :math:`\log\mathcal{N}(x_i\mid\mu,\Sigma)` for every row.

    The quadratic form is a solve against :math:`\Sigma`, shape
    ``(n_samples, n_features)`` in and ``(n_samples,)`` out.
    """

    centered = features - mean
    sign, logdet = np.linalg.slogdet(covariance)
    if sign <= 0.0 or not np.isfinite(logdet):
        raise FloatingPointError("A class covariance is not positive definite.")
    solved = np.linalg.solve(covariance, centered.T).T
    quadratic = np.sum(centered * solved, axis=1)
    dimension = features.shape[1]
    return -0.5 * (dimension * np.log(2.0 * np.pi) + logdet + quadratic)


def _logsumexp(values: NDArray[np.float64], axis: int) -> NDArray[np.float64]:
    """Stable log of a sum of exponentials along ``axis``."""

    peak = np.max(values, axis=axis, keepdims=True)
    total = np.sum(np.exp(values - peak), axis=axis)
    return np.squeeze(peak, axis=axis) + np.log(total)


def _validate_X(X: ArrayLike) -> NDArray[np.float64]:
    """Return a finite float64 design."""

    features = np.asarray(X, dtype=np.float64)
    if features.ndim == 1:
        features = features.reshape(-1, 1)
    if features.ndim != 2:
        raise ValueError(
            "X must have shape (n_samples, n_features); "
            f"got ndim={features.ndim}."
        )
    if features.shape[0] < 1 or features.shape[1] < 1:
        raise ValueError("X must contain at least one row and one column.")
    if not np.isfinite(features).all():
        raise ValueError("X must contain only finite values.")
    return features


def _validate_xy(
    X: ArrayLike,
    y: ArrayLike,
) -> tuple[NDArray[np.float64], NDArray[Any]]:
    """Return a design and a one-dimensional label vector."""

    features = _validate_X(X)
    labels = np.asarray(y)
    if labels.ndim == 2 and labels.shape[1] == 1:
        labels = labels.reshape(-1)
    if labels.ndim != 1:
        raise ValueError("y must be a one-dimensional label vector.")
    if features.shape[0] != labels.shape[0]:
        raise ValueError(
            "X and y must have the same number of rows. "
            f"Got {features.shape[0]} and {labels.shape[0]}."
        )
    if labels.dtype.kind == "f" and not np.isfinite(np.asarray(labels, dtype=np.float64)).all():
        raise ValueError("y must contain only finite values.")
    return features, labels


def _positive_float(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.floating)):
        raise TypeError(f"{name} must be a real number.")
    number = float(value)
    if not np.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be finite and positive.")
    return number
