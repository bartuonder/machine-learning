"""Lloyd's algorithm with K-Means++ initialization.

The objective is the within-cluster sum of squares. Distances are the
squared Euclidean norm, expanded so every assignment is one matrix product.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def pairwise_squared_distances(
    X: ArrayLike,
    Y: ArrayLike,
) -> NDArray[np.float64]:
    r"""Return squared Euclidean distances between two sets of rows.

    For :math:`X \in \mathbb{R}^{n \times d}` and
    :math:`Y \in \mathbb{R}^{m \times d}`,

    .. math::

        \|x_i - y_j\|^2
        = \|x_i\|^2 + \|y_j\|^2 - 2\, x_i^\top y_j.

    The three terms are an ``(n, 1)`` vector of row norms, an ``(1, m)``
    vector of row norms, and the Gram matrix :math:`X Y^\top` of shape
    ``(n, m)``. Broadcasting adds the norms onto that Gram matrix. The
    same identity with a square root would not change
    :math:`\arg\min_j`, and the sum of these squares is the k-means
    objective.

    Parameters
    ----------
    X : array-like of shape (n, d)
        First set of rows.
    Y : array-like of shape (m, d)
        Second set of rows.

    Returns
    -------
    ndarray of shape (n, m)
        Squared distances. Entries that fall slightly below zero from
        roundoff are clipped to zero.
    """

    left = np.asarray(X, dtype=np.float64)
    right = np.asarray(Y, dtype=np.float64)
    if left.ndim != 2 or right.ndim != 2:
        raise ValueError("X and Y must be two-dimensional.")
    if left.shape[1] != right.shape[1]:
        raise ValueError(
            "X and Y must have the same number of columns. "
            f"Got {left.shape[1]} and {right.shape[1]}."
        )
    left_sq = np.sum(left * left, axis=1)[:, None]
    right_sq = np.sum(right * right, axis=1)[None, :]
    # Gram matrix (n, m). Subtracting twice this inner product is the
    # cross term in the expanded square.
    gram = left @ right.T
    return np.maximum(left_sq + right_sq - 2.0 * gram, 0.0)


class KMeans:
    r"""Hard clustering by Lloyd's algorithm.

    Given centers :math:`\mu_1, \ldots, \mu_K`, the objective is the
    within-cluster inertia

    .. math::

        J(\mu, c)
        = \sum_{i=1}^{n} \|x_i - \mu_{c_i}\|^2,
        \qquad
        c_i \in \{1, \ldots, K\}.

    Lloyd alternates two exact minimizations. With the centers fixed, the
    optimal label of row :math:`i` is the nearest center. With the labels
    fixed, the optimal center of a cluster is the mean of its rows,
    because the squared Euclidean norm is minimized by the sample mean.
    Each full step weakly decreases :math:`J`. The procedure stops at a
    local minimum; different initial centers can produce different ones.

    K-Means++ draws the first center uniformly and each later center with
    probability proportional to its squared distance to the nearest center
    already chosen. Those probabilities are :math:`D(x)^2 / \sum_i D(x_i)^2`,
    which is what makes a bad initialization unlikely: points far from the
    current set are preferred.

    Parameters
    ----------
    n_clusters : int, default=8
        Number of centers :math:`K`.
    init : {"k-means++", "random"}, default="k-means++"
        How the first centers of each restart are drawn.
    n_init : int, default=10
        Independent Lloyd runs. The run with the smallest inertia is kept.
    max_iter : int, default=300
        Maximum assignment-and-update cycles per run.
    tol : float, default=1e-4
        Stop a run when the sum of squared center movements is at most
        this value.
    random_state : int or None, default=None
        Seed for center initialization.

    Attributes
    ----------
    cluster_centers_ : ndarray of shape (n_clusters, n_features)
        Centers :math:`\mu_k` of the best run.
    labels_ : ndarray of shape (n_samples,)
        Assignment of each training row.
    inertia_ : float
        :math:`J` at the best run.
    n_iter_ : int
        Lloyd cycles used by the best run.
    n_features_in_ : int
        Columns of ``X``.
    """

    def __init__(
        self,
        n_clusters: int = 8,
        init: str = "k-means++",
        n_init: int = 10,
        max_iter: int = 300,
        tol: float = 1e-4,
        random_state: int | None = None,
    ) -> None:
        if isinstance(n_clusters, bool) or not isinstance(n_clusters, int) or n_clusters < 1:
            raise ValueError("n_clusters must be an integer >= 1.")
        if init not in ("k-means++", "random"):
            raise ValueError("init must be 'k-means++' or 'random'.")
        if isinstance(n_init, bool) or not isinstance(n_init, int) or n_init < 1:
            raise ValueError("n_init must be an integer >= 1.")
        if isinstance(max_iter, bool) or not isinstance(max_iter, int) or max_iter < 1:
            raise ValueError("max_iter must be an integer >= 1.")
        if isinstance(random_state, bool) or not isinstance(random_state, (int, type(None))):
            raise TypeError("random_state must be an int or None.")
        self.n_clusters = n_clusters
        self.init = init
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = _nonnegative_float("tol", tol)
        self.random_state = random_state
        self.cluster_centers_: NDArray[np.float64] | None = None
        self.labels_: NDArray[np.int64] | None = None
        self.inertia_: float | None = None
        self.n_iter_: int = 0
        self.n_features_in_: int | None = None

    def fit(self, X: ArrayLike) -> KMeans:
        """Minimize inertia from several initializations.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Rows to cluster. A one-dimensional array is one feature column.

        Returns
        -------
        self
        """

        features = _validate_X(X)
        n_samples = int(features.shape[0])
        if n_samples < self.n_clusters:
            raise ValueError(
                f"n_samples={n_samples} is smaller than n_clusters={self.n_clusters}."
            )
        self.n_features_in_ = int(features.shape[1])
        rng = np.random.default_rng(self.random_state)
        best_inertia = np.inf
        best_centers: NDArray[np.float64] | None = None
        best_labels: NDArray[np.int64] | None = None
        best_iters = 0
        for _restart in range(self.n_init):
            centers = self._initialize(features, rng)
            labels, centers, inertia, n_iter = self._lloyd(features, centers)
            if inertia < best_inertia:
                best_inertia = inertia
                best_centers = centers
                best_labels = labels
                best_iters = n_iter
        assert best_centers is not None and best_labels is not None
        self.cluster_centers_ = best_centers
        self.labels_ = best_labels
        self.inertia_ = float(best_inertia)
        self.n_iter_ = best_iters
        return self

    def predict(self, X: ArrayLike) -> NDArray[np.int64]:
        """Return the index of the nearest center for each row.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Rows on the scale used in ``fit``.

        Returns
        -------
        ndarray of shape (n_samples,)
            Cluster indices in ``0, ..., n_clusters - 1``.
        """

        distances = self.transform(X)
        return np.argmin(distances, axis=1).astype(np.int64, copy=False)

    def fit_predict(self, X: ArrayLike) -> NDArray[np.int64]:
        """Fit and return the training assignments."""

        self.fit(X)
        assert self.labels_ is not None
        return self.labels_

    def transform(self, X: ArrayLike) -> NDArray[np.float64]:
        """Return squared distance from each row to each center.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Rows on the scale used in ``fit``.

        Returns
        -------
        ndarray of shape (n_samples, n_clusters)
            Squared Euclidean distances.
        """

        self._check_fitted()
        assert self.cluster_centers_ is not None and self.n_features_in_ is not None
        features = _validate_X(X)
        if features.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Expected {self.n_features_in_} features, got {features.shape[1]}."
            )
        return pairwise_squared_distances(features, self.cluster_centers_)

    def score(self, X: ArrayLike) -> float:
        """Return the negative inertia of ``X`` under the fitted centers.

        Higher is better. The value is :math:`-J(X)`, with :math:`J` the
        within-cluster sum of squares.
        """

        distances = self.transform(X)
        labels = np.argmin(distances, axis=1)
        return -float(np.sum(distances[np.arange(labels.shape[0]), labels]))

    def _initialize(
        self,
        features: NDArray[np.float64],
        rng: np.random.Generator,
    ) -> NDArray[np.float64]:
        """Draw ``n_clusters`` initial centers."""

        n_samples = int(features.shape[0])
        if self.init == "random":
            chosen = rng.choice(n_samples, size=self.n_clusters, replace=False)
            return features[chosen].copy()
        return _kmeans_plus_plus(features, self.n_clusters, rng)

    def _lloyd(
        self,
        features: NDArray[np.float64],
        centers: NDArray[np.float64],
    ) -> tuple[NDArray[np.int64], NDArray[np.float64], float, int]:
        """Alternate nearest-center assignment and mean updates."""

        labels = np.zeros(features.shape[0], dtype=np.int64)
        n_iter = 0
        for n_iter in range(1, self.max_iter + 1):
            distances = pairwise_squared_distances(features, centers)
            labels = np.argmin(distances, axis=1).astype(np.int64, copy=False)
            updated = _means_from_labels(features, labels, self.n_clusters, distances)
            # Sum of squared Euclidean movements of the K centers.
            shift = float(np.sum((updated - centers) ** 2))
            centers = updated
            if shift <= self.tol:
                break
        distances = pairwise_squared_distances(features, centers)
        labels = np.argmin(distances, axis=1).astype(np.int64, copy=False)
        inertia = float(np.sum(distances[np.arange(labels.shape[0]), labels]))
        return labels, centers, inertia, n_iter

    def _check_fitted(self) -> None:
        if self.cluster_centers_ is None:
            raise RuntimeError("KMeans must be fitted before use.")

    def __repr__(self) -> str:
        return f"KMeans(n_clusters={self.n_clusters}, init={self.init!r}, n_init={self.n_init})"


def _kmeans_plus_plus(
    features: NDArray[np.float64],
    n_clusters: int,
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    """Sample centers with probability proportional to squared distance.

    The first center is uniform. After :math:`t` centers are chosen, the
    sampling weights are the squared distances to the nearest of those
    :math:`t` centers, computed by one call to
    :func:`pairwise_squared_distances` and a minimum over the center axis.
    """

    n_samples = int(features.shape[0])
    centers = np.empty((n_clusters, features.shape[1]), dtype=np.float64)
    first = int(rng.integers(0, n_samples))
    centers[0] = features[first]
    closest = pairwise_squared_distances(features, centers[:1]).ravel()
    for cluster in range(1, n_clusters):
        total = float(closest.sum())
        if total <= 0.0:
            # Every row sits on a chosen center. The remaining centers
            # may repeat a row; the squared-distance weights have vanished.
            index = int(rng.integers(0, n_samples))
        else:
            probabilities = closest / total
            index = int(rng.choice(n_samples, p=probabilities))
        centers[cluster] = features[index]
        distance = pairwise_squared_distances(features, centers[cluster : cluster + 1]).ravel()
        closest = np.minimum(closest, distance)
    return centers


def _means_from_labels(
    features: NDArray[np.float64],
    labels: NDArray[np.int64],
    n_clusters: int,
    distances: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Return cluster means, reseeding any empty cluster.

    The one-hot indicator of the labels has shape ``(n_samples, K)``.
    Its transpose times ``X`` sums the rows of each cluster in one product,
    and dividing by the cluster counts is the mean. An empty column of the
    indicator produces a zero sum. That center is moved to a row whose
    squared distance to its own center is largest, so the next assignment
    step has a place to put an outlier.
    """

    indicators = np.eye(n_clusters, dtype=np.float64)[labels]
    counts = indicators.sum(axis=0)
    sums = indicators.T @ features
    centers = np.zeros((n_clusters, features.shape[1]), dtype=np.float64)
    occupied = counts > 0.0
    centers[occupied] = sums[occupied] / counts[occupied, None]
    empty = np.flatnonzero(~occupied)
    if empty.size:
        assigned = distances[np.arange(labels.shape[0]), labels]
        farthest = np.argsort(-assigned)
        for slot, cluster in enumerate(empty):
            centers[cluster] = features[int(farthest[slot % farthest.size])]
    return centers


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


def _nonnegative_float(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.floating)):
        raise TypeError(f"{name} must be a real number.")
    number = float(value)
    if not np.isfinite(number) or number < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative.")
    return number
