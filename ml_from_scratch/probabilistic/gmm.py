"""Gaussian mixtures fit by expectation-maximization.

The observed density is a weighted sum of Gaussians. EM alternates the
posterior responsibility of each component with the weighted Gaussian MLE.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..unsupervised.k_means import KMeans


class GaussianMixture:
    r"""Maximum-likelihood mixture of full-covariance Gaussians.

    A draw is a component index :math:`z \in \{1, \ldots, K\}` and then a
    point from that component. With weights :math:`\pi_k`, means
    :math:`\mu_k \in \mathbb{R}^d`, and covariances
    :math:`\Sigma_k \in \mathbb{R}^{d \times d}`,

    .. math::

        p(x) = \sum_{k=1}^{K} \pi_k \, \mathcal{N}(x \mid \mu_k, \Sigma_k).

    The component label is missing, so the marginal log-likelihood

    .. math::

        \ell(\theta)
        = \sum_{i=1}^{n}
          \log \sum_{k=1}^{K}
          \pi_k \, \mathcal{N}(x_i \mid \mu_k, \Sigma_k)

    has no closed-form maximizer. EM builds a lower bound by the posterior
    responsibilities and maximizes that bound. The E-step, in log space, is

    .. math::

        \log \gamma_{ik}
        = \log \pi_k + \log \mathcal{N}(x_i \mid \mu_k, \Sigma_k)
          - \log \sum_{j=1}^{K}
            \pi_j \, \mathcal{N}(x_i \mid \mu_j, \Sigma_j).

    :math:`\gamma` has shape ``(n, K)`` and each row sums to 1. The M-step
    is the weighted Gaussian MLE. With :math:`N_k = \sum_i \gamma_{ik}`,

    .. math::

        \pi_k = \frac{N_k}{n},
        \qquad
        \mu_k = \frac{1}{N_k} \sum_{i=1}^{n} \gamma_{ik} x_i,

    .. math::

        \Sigma_k
        = \frac{1}{N_k}
          \sum_{i=1}^{n}
          \gamma_{ik} (x_i - \mu_k)(x_i - \mu_k)^\top
          + \epsilon I.

    The ridge :math:`\epsilon I` keeps every covariance invertible when a
    component collapses onto a lower-dimensional subset. Initialization is
    a K-Means fit: the centers become :math:`\mu_k`, the cluster
    frequencies become :math:`\pi_k`, and each :math:`\Sigma_k` is the
    covariance of the rows K-Means assigned to that center.

    Parameters
    ----------
    n_components : int, default=2
        Number of Gaussians :math:`K`.
    max_iter : int, default=100
        Maximum EM cycles.
    tol : float, default=1e-4
        Stop when the average log-likelihood rises by less than this.
    reg_covar : float, default=1e-6
        Diagonal ridge :math:`\epsilon` added to each :math:`\Sigma_k`.
    n_init : int, default=1
        Independent runs from different K-Means initializations. The run
        with the highest log-likelihood is kept.
    random_state : int or None, default=None
        Seed passed to K-Means.

    Attributes
    ----------
    weights_ : ndarray of shape (n_components,)
        Mixture weights :math:`\pi`.
    means_ : ndarray of shape (n_components, n_features)
        Component means :math:`\mu_k`.
    covariances_ : ndarray of shape (n_components, n_features, n_features)
        Component covariances :math:`\Sigma_k`, including the ridge.
    n_iter_ : int
        EM cycles used by the best run.
    converged_ : bool
        Whether the best run met ``tol`` before ``max_iter``.
    lower_bound_ : float
        Average log-likelihood :math:`\ell / n` at the best parameters.
    log_likelihood_history_ : list of float
        Average log-likelihood at the start of the best run and after
        each M-step.
    n_features_in_ : int
        Columns of ``X``.
    """

    def __init__(
        self,
        n_components: int = 2,
        max_iter: int = 100,
        tol: float = 1e-4,
        reg_covar: float = 1e-6,
        n_init: int = 1,
        random_state: int | None = None,
    ) -> None:
        if isinstance(n_components, bool) or not isinstance(n_components, int) or n_components < 1:
            raise ValueError("n_components must be an integer >= 1.")
        if isinstance(max_iter, bool) or not isinstance(max_iter, int) or max_iter < 1:
            raise ValueError("max_iter must be an integer >= 1.")
        if isinstance(n_init, bool) or not isinstance(n_init, int) or n_init < 1:
            raise ValueError("n_init must be an integer >= 1.")
        if isinstance(random_state, bool) or not isinstance(random_state, (int, type(None))):
            raise TypeError("random_state must be an int or None.")
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = _nonnegative_float("tol", tol)
        self.reg_covar = _positive_float("reg_covar", reg_covar)
        self.n_init = n_init
        self.random_state = random_state
        self.weights_: NDArray[np.float64] | None = None
        self.means_: NDArray[np.float64] | None = None
        self.covariances_: NDArray[np.float64] | None = None
        self.n_iter_: int = 0
        self.converged_: bool = False
        self.lower_bound_: float | None = None
        self.log_likelihood_history_: list[float] = []
        self.n_features_in_: int | None = None

    def fit(self, X: ArrayLike) -> GaussianMixture:
        """Run EM from one or more K-Means initializations.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Observations. A one-dimensional array is one feature column.

        Returns
        -------
        self
        """

        features = _validate_X(X)
        n_samples, n_features = int(features.shape[0]), int(features.shape[1])
        if n_samples < self.n_components:
            raise ValueError(
                f"n_samples={n_samples} is smaller than n_components={self.n_components}."
            )
        self.n_features_in_ = n_features
        best_bound = -np.inf
        best: tuple[
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            int,
            bool,
            list[float],
        ] | None = None
        for restart in range(self.n_init):
            seed = None if self.random_state is None else self.random_state + restart
            weights, means, covariances = _initialize_from_kmeans(
                features,
                self.n_components,
                self.reg_covar,
                seed,
            )
            outcome = self._em(features, weights, means, covariances)
            if outcome[0] > best_bound:
                best_bound = outcome[0]
                best = outcome[1:]
        assert best is not None
        (
            self.weights_,
            self.means_,
            self.covariances_,
            self.n_iter_,
            self.converged_,
            self.log_likelihood_history_,
        ) = best
        self.lower_bound_ = float(best_bound)
        return self

    def predict_proba(self, X: ArrayLike) -> NDArray[np.float64]:
        r"""Return the posterior responsibility of each component.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Rows on the scale used in ``fit``.

        Returns
        -------
        ndarray of shape (n_samples, n_components)
            :math:`\gamma_{ik} = P(z_i = k \mid x_i)`. Rows sum to 1.
        """

        features = self._validate_fitted(X)
        _log_joint, log_responsibility = self._log_responsibility(features)
        return np.exp(log_responsibility)

    def predict(self, X: ArrayLike) -> NDArray[np.int64]:
        """Return the component with the largest responsibility.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Rows on the scale used in ``fit``.

        Returns
        -------
        ndarray of shape (n_samples,)
            MAP component indices.
        """

        return np.argmax(self.predict_proba(X), axis=1).astype(np.int64, copy=False)

    def score(self, X: ArrayLike) -> float:
        """Return the average log-density of the rows in ``X``."""

        return float(np.mean(self.score_samples(X)))

    def score_samples(self, X: ArrayLike) -> NDArray[np.float64]:
        r"""Return :math:`\log p(x_i)` for each row.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Rows on the scale used in ``fit``.

        Returns
        -------
        ndarray of shape (n_samples,)
            Log mixture density.
        """

        features = self._validate_fitted(X)
        log_joint, _responsibility = self._log_responsibility(features)
        return _logsumexp(log_joint, axis=1)

    def _em(
        self,
        features: NDArray[np.float64],
        weights: NDArray[np.float64],
        means: NDArray[np.float64],
        covariances: NDArray[np.float64],
    ) -> tuple[
        float,
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        int,
        bool,
        list[float],
    ]:
        """Run EM from one initialization and return the winning state."""

        self.weights_ = weights
        self.means_ = means
        self.covariances_ = covariances
        history = [self.score(features)]
        converged = False
        n_iter = 0
        for n_iter in range(1, self.max_iter + 1):
            _log_joint, log_responsibility = self._log_responsibility(features)
            responsibility = np.exp(log_responsibility)
            self._m_step(features, responsibility)
            history.append(self.score(features))
            if history[-1] - history[-2] <= self.tol:
                converged = True
                break
        assert self.weights_ is not None and self.means_ is not None and self.covariances_ is not None
        return (
            history[-1],
            self.weights_.copy(),
            self.means_.copy(),
            self.covariances_.copy(),
            n_iter,
            converged,
            history,
        )

    def _m_step(self, features: NDArray[np.float64], responsibility: NDArray[np.float64]) -> None:
        r"""Maximize the expected complete log-likelihood in the parameters.

        :math:`N_k` is the sum of column :math:`k` of :math:`\gamma`. The
        mean update is the matrix-vector product :math:`\gamma_{\cdot k}^\top X`,
        divided by :math:`N_k`. The covariance update scales each centered
        row by :math:`\sqrt{\gamma_{ik}}` and takes the Gram matrix, which
        equals :math:`\sum_i \gamma_{ik}(x_i-\mu_k)(x_i-\mu_k)^\top`
        without a Python loop over rows.
        """

        assert self.weights_ is not None and self.means_ is not None and self.covariances_ is not None
        n_samples, n_features = features.shape
        counts = responsibility.sum(axis=0)
        counts = np.maximum(counts, 1e-12)
        self.weights_ = counts / n_samples
        # (K, n) @ (n, d) = (K, d). Row k is the responsibility-weighted sum.
        self.means_ = (responsibility.T @ features) / counts[:, None]
        eye = np.eye(n_features, dtype=np.float64)
        for component in range(self.n_components):
            centered = features - self.means_[component]
            scale = np.sqrt(responsibility[:, component])[:, None]
            weighted = centered * scale
            covariance = (weighted.T @ weighted) / counts[component]
            self.covariances_[component] = covariance + self.reg_covar * eye

    def _log_responsibility(
        self,
        features: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        r"""Return joint log-densities and row-normalized log responsibilities.

        Both arrays have shape ``(n_samples, n_components)``. The
        normalization subtracts :math:`\log\sum_k \exp(\log \pi_k + \log
        \mathcal{N}_{ik})`, computed with a max subtraction so a large
        Gaussian density does not overflow the exponential.
        """

        assert self.weights_ is not None and self.means_ is not None and self.covariances_ is not None
        n_samples = int(features.shape[0])
        log_joint = np.empty((n_samples, self.n_components), dtype=np.float64)
        log_weights = np.full(self.n_components, -np.inf, dtype=np.float64)
        positive = self.weights_ > 0.0
        log_weights[positive] = np.log(self.weights_[positive])
        for component in range(self.n_components):
            log_joint[:, component] = log_weights[component] + _log_gaussian(
                features,
                self.means_[component],
                self.covariances_[component],
            )
        log_marginal = _logsumexp(log_joint, axis=1)
        return log_joint, log_joint - log_marginal[:, None]

    def _validate_fitted(self, X: ArrayLike) -> NDArray[np.float64]:
        if self.weights_ is None or self.n_features_in_ is None:
            raise RuntimeError("GaussianMixture must be fitted before use.")
        features = _validate_X(X)
        if features.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Expected {self.n_features_in_} features, got {features.shape[1]}."
            )
        return features

    def __repr__(self) -> str:
        return f"GaussianMixture(n_components={self.n_components}, reg_covar={self.reg_covar})"


def _initialize_from_kmeans(
    features: NDArray[np.float64],
    n_components: int,
    reg_covar: float,
    random_state: int | None,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Set mixture parameters from one K-Means partition."""

    model = KMeans(n_clusters=n_components, n_init=1, random_state=random_state)
    labels = model.fit_predict(features)
    n_samples, n_features = features.shape
    means = np.asarray(model.cluster_centers_, dtype=np.float64)
    weights = np.empty(n_components, dtype=np.float64)
    covariances = np.empty((n_components, n_features, n_features), dtype=np.float64)
    eye = np.eye(n_features, dtype=np.float64)
    for component in range(n_components):
        members = features[labels == component]
        weights[component] = members.shape[0] / n_samples
        if members.shape[0] == 0:
            covariances[component] = reg_covar * eye
            continue
        centered = members - means[component]
        if members.shape[0] == 1:
            covariances[component] = reg_covar * eye
        else:
            covariances[component] = (centered.T @ centered) / members.shape[0] + reg_covar * eye
    weights = np.maximum(weights, 0.0)
    total = float(weights.sum())
    if total <= 0.0:
        weights[:] = 1.0 / n_components
    else:
        weights /= total
    return weights, means, covariances


def _log_gaussian(
    features: NDArray[np.float64],
    mean: NDArray[np.float64],
    covariance: NDArray[np.float64],
) -> NDArray[np.float64]:
    r"""Return :math:`\log \mathcal{N}(x_i \mid \mu, \Sigma)` for every row.

    .. math::

        \log \mathcal{N}
        = -\frac{1}{2}\left(
            d \log 2\pi + \log\det\Sigma + (x-\mu)^\top \Sigma^{-1}(x-\mu)
          \right).

    The quadratic form is :math:`\sum_a \delta_{ia} z_{ia}` with
    :math:`\Sigma z_i^\top = \delta_i^\top`, one triangular solve for every
    row. ``features`` has shape ``(n, d)`` and the return value has shape
    ``(n,)``.
    """

    centered = features - mean
    sign, logdet = np.linalg.slogdet(covariance)
    if sign <= 0.0 or not np.isfinite(logdet):
        raise FloatingPointError("A component covariance is not positive definite.")
    solved = np.linalg.solve(covariance, centered.T).T
    quadratic = np.sum(centered * solved, axis=1)
    dimension = features.shape[1]
    return -0.5 * (dimension * np.log(2.0 * np.pi) + logdet + quadratic)


def _logsumexp(values: NDArray[np.float64], axis: int) -> NDArray[np.float64]:
    """Stable log of a sum of exponentials along ``axis``."""

    peak = np.max(values, axis=axis, keepdims=True)
    stable = np.where(np.isfinite(peak), peak, 0.0)
    total = np.sum(np.exp(values - stable), axis=axis)
    reduced = np.squeeze(stable, axis=axis)
    return reduced + np.log(total)


def _validate_X(X: ArrayLike) -> NDArray[np.float64]:
    """Return a finite float64 design of shape ``(n_samples, n_features)``."""

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


def _positive_float(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.floating)):
        raise TypeError(f"{name} must be a real number.")
    number = float(value)
    if not np.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be finite and positive.")
    return number


def _nonnegative_float(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.floating)):
        raise TypeError(f"{name} must be a real number.")
    number = float(value)
    if not np.isfinite(number) or number < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative.")
    return number
