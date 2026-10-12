"""Naive Bayes classifiers.

Each feature is treated as independent given the class. The joint density
then factors, and the class posterior is a product of one-dimensional
likelihoods times the class prior.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..utils.metrics import accuracy_score


class GaussianNaiveBayes:
    r"""Gaussian naive Bayes.

    The model is Gaussian discriminant analysis with a diagonal covariance.
    Given the class, feature :math:`j` is independent of the others:

    .. math::

        x_j \mid y = k \sim \mathcal{N}(\mu_{kj}, \sigma_{kj}^2),
        \qquad
        \log p(x \mid y = k)
        = \sum_{j=1}^{d} \log \mathcal{N}(x_j \mid \mu_{kj}, \sigma_{kj}^2).

    The posterior is Bayes' rule with this product likelihood and
    :math:`\pi_k = n_k / n`. :math:`\mu_{kj}` is the class mean of feature
    :math:`j`. The class variance is the MLE

    .. math::

        \sigma_{kj}^2
        = \frac{1}{n_k}\sum_{i: y_i = k}(x_{ij} - \mu_{kj})^2

    plus ``var_smoothing`` times the largest feature variance in the whole
    training matrix. That floor keeps a constant feature from sending the
    Gaussian density to a spike.

    Parameters
    ----------
    var_smoothing : float, default=1e-9
        Fraction of the largest empirical feature variance added to every
        :math:`\sigma_{kj}^2`.

    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
        Sorted training labels.
    class_prior_ : ndarray of shape (n_classes,)
        :math:`\pi_k`.
    theta_ : ndarray of shape (n_classes, n_features)
        Means :math:`\mu_{kj}`.
    var_ : ndarray of shape (n_classes, n_features)
        Smoothed variances :math:`\sigma_{kj}^2`.
    n_features_in_ : int
        Columns of ``X``.
    """

    def __init__(self, var_smoothing: float = 1e-9) -> None:
        self.var_smoothing = _nonnegative_float("var_smoothing", var_smoothing)
        self.classes_: NDArray[Any] | None = None
        self.class_prior_: NDArray[np.float64] | None = None
        self.theta_: NDArray[np.float64] | None = None
        self.var_: NDArray[np.float64] | None = None
        self.n_features_in_: int | None = None

    def fit(self, X: ArrayLike, y: ArrayLike) -> GaussianNaiveBayes:
        """Estimate a diagonal Gaussian for each class.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Continuous features.
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
        means = np.zeros((n_classes, n_features), dtype=np.float64)
        variances = np.zeros((n_classes, n_features), dtype=np.float64)
        for class_index in range(n_classes):
            members = features[codes == class_index]
            means[class_index] = members.mean(axis=0)
            deviations = members - means[class_index]
            variances[class_index] = np.mean(deviations * deviations, axis=0)
        scale = float(np.var(features, axis=0).max())
        floor = self.var_smoothing * scale if scale > 0.0 else self.var_smoothing
        self.classes_ = classes
        self.class_prior_ = counts / n_samples
        self.theta_ = means
        self.var_ = np.maximum(variances + floor, np.finfo(np.float64).tiny)
        self.n_features_in_ = int(n_features)
        return self

    def predict_log_proba(self, X: ArrayLike) -> NDArray[np.float64]:
        r"""Return :math:`\log P(y = k \mid x)`.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Rows on the scale used in ``fit``.

        Returns
        -------
        ndarray of shape (n_samples, n_classes)
            Log posteriors.
        """

        features = self._validate_fitted(X)
        assert self.theta_ is not None and self.var_ is not None and self.class_prior_ is not None
        # Broadcast features (n, 1, d) against parameters (1, K, d). The
        # sum over the feature axis is the independence assumption: the
        # joint log-likelihood is a sum of scalar Gaussian log-likelihoods.
        deviation = features[:, None, :] - self.theta_[None, :, :]
        quadratic = np.sum(deviation * deviation / self.var_[None, :, :], axis=2)
        log_partition = np.sum(np.log(2.0 * np.pi * self.var_), axis=1)
        log_likelihood = -0.5 * (log_partition[None, :] + quadratic)
        log_joint = np.log(self.class_prior_)[None, :] + log_likelihood
        log_marginal = _logsumexp(log_joint, axis=1)
        return log_joint - log_marginal[:, None]

    def predict_proba(self, X: ArrayLike) -> NDArray[np.float64]:
        """Return :math:`P(y = k \\mid x)`.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Rows on the scale used in ``fit``.

        Returns
        -------
        ndarray of shape (n_samples, n_classes)
            Posterior probabilities. Each row sums to 1.
        """

        return np.exp(self.predict_log_proba(X))

    def predict(self, X: ArrayLike) -> NDArray[Any]:
        """Return the class with the largest posterior probability."""

        self._check_fitted()
        assert self.classes_ is not None
        return self.classes_[np.argmax(self.predict_log_proba(X), axis=1)]

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
        if self.theta_ is None or self.classes_ is None:
            raise RuntimeError("GaussianNaiveBayes must be fitted before use.")

    def __repr__(self) -> str:
        return f"GaussianNaiveBayes(var_smoothing={self.var_smoothing})"


class BernoulliNaiveBayes:
    r"""Bernoulli naive Bayes for binary features.

    Feature :math:`j` is a coin toss given the class:

    .. math::

        P(x_j = 1 \mid y = k) = \theta_{kj},
        \qquad
        \log p(x \mid y = k)
        = \sum_{j=1}^{d}
          \left[
            x_j\log\theta_{kj} + (1 - x_j)\log(1 - \theta_{kj})
          \right].

    With Laplace smoothing :math:`\alpha`,

    .. math::

        \theta_{kj}
        = \frac{\#\{i : y_i = k,\, x_{ij} = 1\} + \alpha}
               {n_k + 2\alpha}.

    :math:`\alpha = 1` is the uniform Beta prior. Features must lie in
    :math:`\{0, 1\}`.

    Parameters
    ----------
    alpha : float, default=1.0
        Additive smoothing :math:`\alpha`. Zero is allowed and is the MLE.

    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
        Sorted training labels.
    class_prior_ : ndarray of shape (n_classes,)
        :math:`\pi_k`.
    feature_log_prob_ : ndarray of shape (n_classes, n_features)
        :math:`\log\theta_{kj}`.
    feature_count_ : ndarray of shape (n_classes, n_features)
        Unsmoothed counts of the positive value.
    n_features_in_ : int
        Columns of ``X``.
    """

    def __init__(self, alpha: float = 1.0) -> None:
        self.alpha = _nonnegative_float("alpha", alpha)
        self.classes_: NDArray[Any] | None = None
        self.class_prior_: NDArray[np.float64] | None = None
        self.feature_log_prob_: NDArray[np.float64] | None = None
        self.feature_count_: NDArray[np.float64] | None = None
        self.n_features_in_: int | None = None
        self._log_complement: NDArray[np.float64] | None = None

    def fit(self, X: ArrayLike, y: ArrayLike) -> BernoulliNaiveBayes:
        """Estimate a Bernoulli parameter per class and feature.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Binary features in ``{0, 1}``.
        y : array-like of shape (n_samples,)
            Class labels.

        Returns
        -------
        self
        """

        features, labels = _validate_xy(X, y)
        if not np.all((features == 0.0) | (features == 1.0)):
            raise ValueError("BernoulliNaiveBayes requires features in {0, 1}.")
        classes, codes = np.unique(labels, return_inverse=True)
        n_samples, n_features = features.shape
        n_classes = int(classes.size)
        indicators = np.eye(n_classes, dtype=np.float64)[codes]
        counts = indicators.sum(axis=0)
        # (K, n) @ (n, d) sums each feature inside each class.
        positive = indicators.T @ features
        theta = (positive + self.alpha) / (counts[:, None] + 2.0 * self.alpha)
        self.classes_ = classes
        self.class_prior_ = counts / n_samples
        self.feature_count_ = positive
        with np.errstate(divide="ignore"):
            self.feature_log_prob_ = np.log(theta)
            self._log_complement = np.log(1.0 - theta)
        self.n_features_in_ = int(n_features)
        return self

    def predict_log_proba(self, X: ArrayLike) -> NDArray[np.float64]:
        r"""Return :math:`\log P(y = k \mid x)`.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Binary rows.

        Returns
        -------
        ndarray of shape (n_samples, n_classes)
            Log posteriors.
        """

        features = self._validate_binary(X)
        assert (
            self.feature_log_prob_ is not None
            and self._log_complement is not None
            and self.class_prior_ is not None
        )
        # x log θ + (1 - x) log(1 - θ), summed over features by two Gram products.
        log_likelihood = features @ self.feature_log_prob_.T
        log_likelihood = log_likelihood + (1.0 - features) @ self._log_complement.T
        log_joint = np.log(self.class_prior_)[None, :] + log_likelihood
        return log_joint - _logsumexp(log_joint, axis=1)[:, None]

    def predict_proba(self, X: ArrayLike) -> NDArray[np.float64]:
        """Return :math:`P(y = k \\mid x)`."""

        return np.exp(self.predict_log_proba(X))

    def predict(self, X: ArrayLike) -> NDArray[Any]:
        """Return the class with the largest posterior probability."""

        self._check_fitted()
        assert self.classes_ is not None
        return self.classes_[np.argmax(self.predict_log_proba(X), axis=1)]

    def score(self, X: ArrayLike, y: ArrayLike) -> float:
        """Return the classification accuracy of ``predict(X)``."""

        return accuracy_score(y, self.predict(X))

    def _validate_binary(self, X: ArrayLike) -> NDArray[np.float64]:
        self._check_fitted()
        assert self.n_features_in_ is not None
        features = _validate_X(X)
        if features.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Expected {self.n_features_in_} features, got {features.shape[1]}."
            )
        if not np.all((features == 0.0) | (features == 1.0)):
            raise ValueError("BernoulliNaiveBayes requires features in {0, 1}.")
        return features

    def _check_fitted(self) -> None:
        if self.feature_log_prob_ is None or self.classes_ is None:
            raise RuntimeError("BernoulliNaiveBayes must be fitted before use.")

    def __repr__(self) -> str:
        return f"BernoulliNaiveBayes(alpha={self.alpha})"


class MultinomialNaiveBayes:
    r"""Multinomial naive Bayes for nonnegative count features.

    A row is a bag of feature counts. Given the class, the counts follow a
    multinomial whose event probabilities factor across features in the
    log-likelihood used for classification:

    .. math::

        \log p(x \mid y = k)
        = \sum_{j=1}^{d} x_j \log \theta_{kj}
          + \log \frac{(\sum_j x_j)!}{\prod_j x_j!}.

    The factorial term does not depend on :math:`k`, so it cancels in the
    class posterior. The smoothed event probabilities are

    .. math::

        \theta_{kj}
        = \frac{\sum_{i: y_i = k} x_{ij} + \alpha}
               {\sum_{i: y_i = k}\sum_{t=1}^{d} x_{it} + \alpha d}.

    Parameters
    ----------
    alpha : float, default=1.0
        Additive smoothing :math:`\alpha`.

    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
        Sorted training labels.
    class_prior_ : ndarray of shape (n_classes,)
        :math:`\pi_k`.
    feature_log_prob_ : ndarray of shape (n_classes, n_features)
        :math:`\log\theta_{kj}`.
    feature_count_ : ndarray of shape (n_classes, n_features)
        Unsmoothed summed counts.
    n_features_in_ : int
        Columns of ``X``.
    """

    def __init__(self, alpha: float = 1.0) -> None:
        self.alpha = _nonnegative_float("alpha", alpha)
        self.classes_: NDArray[Any] | None = None
        self.class_prior_: NDArray[np.float64] | None = None
        self.feature_log_prob_: NDArray[np.float64] | None = None
        self.feature_count_: NDArray[np.float64] | None = None
        self.n_features_in_: int | None = None

    def fit(self, X: ArrayLike, y: ArrayLike) -> MultinomialNaiveBayes:
        """Estimate multinomial event probabilities.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Nonnegative count features.
        y : array-like of shape (n_samples,)
            Class labels.

        Returns
        -------
        self
        """

        features, labels = _validate_xy(X, y)
        if np.any(features < 0.0):
            raise ValueError("MultinomialNaiveBayes requires nonnegative features.")
        classes, codes = np.unique(labels, return_inverse=True)
        n_samples, n_features = features.shape
        n_classes = int(classes.size)
        indicators = np.eye(n_classes, dtype=np.float64)[codes]
        class_count = indicators.sum(axis=0)
        feature_count = indicators.T @ features
        total = feature_count.sum(axis=1)
        denominator = total[:, None] + self.alpha * n_features
        if np.any(denominator <= 0.0):
            raise ValueError(
                "A class has no counts and alpha is 0, so the multinomial "
                "probabilities are undefined. Use alpha > 0."
            )
        theta = (feature_count + self.alpha) / denominator
        self.classes_ = classes
        self.class_prior_ = class_count / n_samples
        self.feature_count_ = feature_count
        with np.errstate(divide="ignore"):
            self.feature_log_prob_ = np.log(theta)
        self.n_features_in_ = int(n_features)
        return self

    def predict_log_proba(self, X: ArrayLike) -> NDArray[np.float64]:
        r"""Return :math:`\log P(y = k \mid x)`.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Nonnegative count rows.

        Returns
        -------
        ndarray of shape (n_samples, n_classes)
            Log posteriors. The multinomial coefficient is omitted because
            it does not depend on the class.
        """

        features = self._validate_counts(X)
        assert self.feature_log_prob_ is not None and self.class_prior_ is not None
        log_joint = np.log(self.class_prior_)[None, :] + features @ self.feature_log_prob_.T
        return log_joint - _logsumexp(log_joint, axis=1)[:, None]

    def predict_proba(self, X: ArrayLike) -> NDArray[np.float64]:
        """Return :math:`P(y = k \\mid x)`."""

        return np.exp(self.predict_log_proba(X))

    def predict(self, X: ArrayLike) -> NDArray[Any]:
        """Return the class with the largest posterior probability."""

        self._check_fitted()
        assert self.classes_ is not None
        return self.classes_[np.argmax(self.predict_log_proba(X), axis=1)]

    def score(self, X: ArrayLike, y: ArrayLike) -> float:
        """Return the classification accuracy of ``predict(X)``."""

        return accuracy_score(y, self.predict(X))

    def _validate_counts(self, X: ArrayLike) -> NDArray[np.float64]:
        self._check_fitted()
        assert self.n_features_in_ is not None
        features = _validate_X(X)
        if features.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Expected {self.n_features_in_} features, got {features.shape[1]}."
            )
        if np.any(features < 0.0):
            raise ValueError("MultinomialNaiveBayes requires nonnegative features.")
        return features

    def _check_fitted(self) -> None:
        if self.feature_log_prob_ is None or self.classes_ is None:
            raise RuntimeError("MultinomialNaiveBayes must be fitted before use.")

    def __repr__(self) -> str:
        return f"MultinomialNaiveBayes(alpha={self.alpha})"


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


def _nonnegative_float(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.floating)):
        raise TypeError(f"{name} must be a real number.")
    number = float(value)
    if not np.isfinite(number) or number < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative.")
    return number
