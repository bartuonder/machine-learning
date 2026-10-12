"""Bootstrap aggregation of classification trees with random split sets.

Each tree is grown on a multinomial bootstrap sample. At every node the
split search is restricted to a fresh subset of columns, which is what
makes the trees disagree with one another.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..utils.metrics import accuracy_score
from .decision_tree import DecisionTreeClassifier

MaxFeatures = int | float | Literal["sqrt", "log2"] | None


class _RandomFeatureTree(DecisionTreeClassifier):
    """Classification tree whose split search draws a column subset.

    The impurity, the gain, and the threshold rule are unchanged. The only
    difference from :class:`DecisionTreeClassifier` is which columns are
    offered to that search.
    """

    def __init__(
        self,
        rng: np.random.Generator,
        n_features_per_split: int,
        criterion: Literal["gini", "entropy"] = "gini",
        max_depth: int | None = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        min_impurity_decrease: float = 0.0,
    ) -> None:
        super().__init__(
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_impurity_decrease=min_impurity_decrease,
        )
        if n_features_per_split < 1:
            raise ValueError("n_features_per_split must be positive.")
        self._rng = rng
        self._n_features_per_split = n_features_per_split

    def _candidate_features(self, n_features: int) -> NDArray[np.int64]:
        """Draw ``m_try`` column indices without replacement.

        The draw is uniform among subsets of that size. Sorting the draw
        restores the smallest-index tie break inside the subset. When the
        subset is the full column set, no random numbers are consumed, so
        a forest with ``max_features=None`` follows the same splits as one
        :class:`DecisionTreeClassifier`.
        """

        if self._n_features_per_split >= n_features:
            return np.arange(n_features, dtype=np.int64)
        chosen = self._rng.choice(n_features, size=self._n_features_per_split, replace=False)
        chosen.sort()
        return chosen.astype(np.int64, copy=False)


class RandomForestClassifier:
    r"""Bagged classification trees with a random feature subset at each split.

    Draw :math:`B` trees. Tree :math:`b` is fit on indices
    :math:`I_b = (i_{b1}, \ldots, i_{b,n_b})` sampled from
    :math:`\{0,\ldots,n-1\}`. With the default bootstrap those draws are
    independent and uniform with replacement, so the multiplicity of row
    :math:`i` is :math:`\mathrm{Binomial}(n_b, 1/n)`. A row that appears
    twice is two observations: it contributes twice to every leaf count.
    That is the multinomial bootstrap. The probability that a given row
    is missing from one draw of size :math:`n` is :math:`(1 - 1/n)^n`,
    which converges to :math:`1/e`. Those missing rows are the
    out-of-bag sample.

    The forest prediction is the average of the trees' leaf distributions.
    If :math:`f_b(x) \in \Delta^{K-1}` is tree :math:`b`'s vector of class
    frequencies,

    .. math::

        \hat{P}(y = k \mid x)
        = \frac{1}{B}\sum_{b=1}^{B} f_b(x)_k.

    ``predict`` returns the class with the largest coordinate. Averaging
    the distributions, rather than averaging hard votes, keeps the
    probability that a tree assigned to its chosen leaf.

    Bagging alone, with every tree allowed to use every feature, drives
    the variance of the average down only until the correlation between
    trees remains. For identically distributed trees with pairwise
    correlation :math:`\rho` and variance :math:`\sigma^2`,

    .. math::

        \mathrm{Var}(\bar{f})
        = \rho\sigma^2 + \frac{1 - \rho}{B}\sigma^2.

    As :math:`B \to \infty` the second term vanishes and the variance
    floors at :math:`\rho\sigma^2`. A random forest lowers :math:`\rho` by
    offering each node only :math:`m_{\mathrm{try}}` columns, drawn
    uniformly without replacement. The information-gain comparison is the
    one in :class:`~ml_from_scratch.tree_models.decision_tree.DecisionTreeClassifier`,
    restricted to that column subset. The default
    :math:`m_{\mathrm{try}} = \lfloor\sqrt{d}\rfloor`.

    Parameters
    ----------
    n_estimators : int, default=100
        Number of trees :math:`B`.
    criterion : {"gini", "entropy"}, default="gini"
        Split impurity.
    max_depth : int or None, default=None
        Passed to every tree. ``None`` grows each tree until a leaf
        constraint stops it.
    min_samples_split : int, default=2
        Smallest node that a tree may split.
    min_samples_leaf : int, default=1
        Smallest side of a tree split.
    min_impurity_decrease : float, default=0.0
        Minimum sample-weighted information gain required of a tree split.
    max_features : {"sqrt", "log2"}, int, float, or None, default="sqrt"
        Columns drawn at each node. An int is that count. A float in
        ``(0, 1]`` is a fraction of ``n_features``, rounded up. ``"sqrt"``
        is :math:`\lfloor\sqrt{d}\rfloor` and ``"log2"`` is
        :math:`\lfloor\log_2 d\rfloor`, both at least 1. ``None`` uses
        every column.
    bootstrap : bool, default=True
        Sample training rows with replacement. ``False`` samples without
        replacement.
    max_samples : int, float, or None, default=None
        Size of each tree's row sample. ``None`` uses ``n_samples``. A
        float in ``(0, 1]`` is a fraction of ``n_samples``, rounded up.
        With ``bootstrap=False`` the size cannot exceed ``n_samples``.
    random_state : int or None, default=None
        Seed of the ``SeedSequence`` that spawns one generator per tree.
    oob_score : bool, default=False
        Estimate accuracy on rows that missed each tree's bootstrap
        sample. Requires ``bootstrap=True``. The estimate averages, for
        each row, only the trees that excluded it, then scores the class
        of highest average probability. Rows that happened to sit in every
        bootstrap sample are left out of the average.

    Attributes
    ----------
    estimators_ : list of DecisionTreeClassifier
        Fitted trees, length ``n_estimators``.
    classes_ : ndarray of shape (n_classes,)
        Sorted training labels. Column ``k`` of ``predict_proba`` is
        ``classes_[k]``.
    n_features_in_ : int
        Columns of ``X``.
    n_features_per_split_ : int
        Resolved :math:`m_{\mathrm{try}}`.
    n_samples_ : int
        Rows of the last ``fit``.
    feature_importances_ : ndarray of shape (n_features,)
        Average of the trees' normalized importances, renormalized to sum
        to 1 when any tree split. A tree that never split contributes a
        zero vector.
    oob_score_ : float or None
        Out-of-bag accuracy when ``oob_score`` is true and at least one
        row was excluded from some tree. ``None`` otherwise.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        criterion: Literal["gini", "entropy"] = "gini",
        max_depth: int | None = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        min_impurity_decrease: float = 0.0,
        max_features: MaxFeatures = "sqrt",
        bootstrap: bool = True,
        max_samples: int | float | None = None,
        random_state: int | None = None,
        oob_score: bool = False,
    ) -> None:
        if isinstance(n_estimators, bool) or not isinstance(n_estimators, int) or n_estimators < 1:
            raise ValueError("n_estimators must be an integer >= 1.")
        if criterion not in ("gini", "entropy"):
            raise ValueError("criterion must be 'gini' or 'entropy'.")
        if not isinstance(bootstrap, bool):
            raise TypeError("bootstrap must be a bool.")
        if not isinstance(oob_score, bool):
            raise TypeError("oob_score must be a bool.")
        if oob_score and not bootstrap:
            raise ValueError("oob_score requires bootstrap=True.")
        if isinstance(random_state, bool) or not isinstance(random_state, (int, type(None))):
            raise TypeError("random_state must be an int or None.")
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_decrease = min_impurity_decrease
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.max_samples = max_samples
        self.random_state = random_state
        self.oob_score = oob_score
        self.estimators_: list[_RandomFeatureTree] = []
        self.classes_: NDArray[Any] | None = None
        self.n_features_in_: int | None = None
        self.n_features_per_split_: int | None = None
        self.n_samples_: int | None = None
        self.feature_importances_: NDArray[np.float64] | None = None
        self.oob_score_: float | None = None

    def fit(self, X: ArrayLike, y: ArrayLike) -> RandomForestClassifier:
        """Fit ``n_estimators`` trees on bootstrap samples.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Numeric design. A one-dimensional array is one feature column.
        y : array-like of shape (n_samples,)
            Class labels.

        Returns
        -------
        self
        """

        features, labels = _validate_xy(X, y)
        classes = np.unique(labels)
        n_samples, n_features = int(features.shape[0]), int(features.shape[1])
        n_draw = _sample_count(self.max_samples, n_samples, self.bootstrap)
        n_try = _features_per_split(self.max_features, n_features)
        # One child generator per tree. Spawning them up front makes tree
        # ``b`` independent of how many random numbers tree ``b - 1`` used
        # while searching splits.
        generators = [
            np.random.default_rng(seed)
            for seed in np.random.SeedSequence(self.random_state).spawn(self.n_estimators)
        ]
        self.estimators_ = []
        self.classes_ = classes
        self.n_features_in_ = n_features
        self.n_features_per_split_ = n_try
        self.n_samples_ = n_samples
        self.feature_importances_ = np.zeros(n_features, dtype=np.float64)
        self.oob_score_ = None
        oob_sum = np.zeros((n_samples, classes.size), dtype=np.float64)
        oob_count = np.zeros(n_samples, dtype=np.float64)

        for rng in generators:
            indices = _draw_indices(rng, n_samples, n_draw, self.bootstrap)
            tree = _RandomFeatureTree(
                rng=rng,
                n_features_per_split=n_try,
                criterion=self.criterion,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                min_impurity_decrease=self.min_impurity_decrease,
            )
            # Integer indexing copies the bootstrap rows. Repeated indices
            # become repeated observations, which is the bootstrap weight.
            tree.fit(features[indices], labels[indices])
            self.estimators_.append(tree)
            self.feature_importances_ += tree.feature_importances_
            if self.oob_score:
                membership = np.bincount(indices, minlength=n_samples)
                missing = np.flatnonzero(membership == 0)
                if missing.size:
                    # Vote with this tree only where it did not train.
                    oob_sum[missing] += _align_probabilities(
                        tree.predict_proba(features[missing]),
                        tree.classes_,
                        classes,
                    )
                    oob_count[missing] += 1.0

        total_importance = float(self.feature_importances_.sum())
        if total_importance > 0.0:
            self.feature_importances_ /= total_importance
        if self.oob_score:
            seen = oob_count > 0.0
            if np.any(seen):
                averaged = oob_sum[seen] / oob_count[seen, None]
                predicted = classes[np.argmax(averaged, axis=1)]
                self.oob_score_ = accuracy_score(labels[seen], predicted)
        return self

    def predict_proba(self, X: ArrayLike) -> NDArray[np.float64]:
        """Return the average of the trees' class-frequency vectors.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Rows on the feature scale used in ``fit``.

        Returns
        -------
        ndarray of shape (n_samples, n_classes)
            Averaged probabilities. Each row sums to 1.
        """

        self._check_fitted()
        assert self.classes_ is not None and self.n_features_in_ is not None
        features = _validate_X(X, self.n_features_in_)
        total = np.zeros((features.shape[0], self.classes_.size), dtype=np.float64)
        for tree in self.estimators_:
            total += _align_probabilities(tree.predict_proba(features), tree.classes_, self.classes_)
        # Each tree's row already sums to 1, so dividing by B preserves that.
        total /= len(self.estimators_)
        return total

    def predict(self, X: ArrayLike) -> NDArray[Any]:
        """Return the class with the largest averaged probability.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Rows on the feature scale used in ``fit``.

        Returns
        -------
        ndarray of shape (n_samples,)
            Labels from ``classes_``. Ties go to the earlier class.
        """

        self._check_fitted()
        assert self.classes_ is not None
        codes = np.argmax(self.predict_proba(X), axis=1)
        return self.classes_[codes]

    def score(self, X: ArrayLike, y: ArrayLike) -> float:
        """Return the classification accuracy of ``predict(X)``."""

        return accuracy_score(y, self.predict(X))

    def _check_fitted(self) -> None:
        if not self.estimators_ or self.classes_ is None:
            raise RuntimeError("RandomForestClassifier must be fitted before use.")

    def __repr__(self) -> str:
        return (
            f"RandomForestClassifier(n_estimators={self.n_estimators}, "
            f"max_features={self.max_features!r}, bootstrap={self.bootstrap})"
        )


def _features_per_split(max_features: MaxFeatures, n_features: int) -> int:
    """Resolve ``max_features`` to a column count in ``1, ..., n_features``."""

    if max_features is None:
        return n_features
    if max_features == "sqrt":
        return max(1, int(np.sqrt(n_features)))
    if max_features == "log2":
        if n_features <= 1:
            return 1
        return max(1, int(np.log2(n_features)))
    if isinstance(max_features, bool):
        raise TypeError("max_features must be 'sqrt', 'log2', a count, a fraction, or None.")
    if isinstance(max_features, (int, np.integer)):
        count = int(max_features)
        if count < 1 or count > n_features:
            raise ValueError(
                f"max_features={count} is outside 1..{n_features}."
            )
        return count
    if isinstance(max_features, (float, np.floating)):
        fraction = float(max_features)
        if not 0.0 < fraction <= 1.0:
            raise ValueError("A fractional max_features must lie in (0, 1].")
        return max(1, int(np.ceil(fraction * n_features)))
    raise TypeError("max_features must be 'sqrt', 'log2', a count, a fraction, or None.")


def _sample_count(
    max_samples: int | float | None,
    n_samples: int,
    bootstrap: bool,
) -> int:
    """Resolve the number of row draws that build one tree."""

    if max_samples is None:
        return n_samples
    if isinstance(max_samples, bool):
        raise TypeError("max_samples must be an int, a float, or None.")
    if isinstance(max_samples, (int, np.integer)):
        count = int(max_samples)
    elif isinstance(max_samples, (float, np.floating)):
        fraction = float(max_samples)
        if not 0.0 < fraction <= 1.0:
            raise ValueError("A fractional max_samples must lie in (0, 1].")
        count = int(np.ceil(fraction * n_samples))
    else:
        raise TypeError("max_samples must be an int, a float, or None.")
    if count < 1:
        raise ValueError("max_samples must draw at least one row.")
    if not bootstrap and count > n_samples:
        raise ValueError(
            "Without replacement, max_samples cannot exceed n_samples."
        )
    return count


def _draw_indices(
    rng: np.random.Generator,
    n_samples: int,
    n_draw: int,
    bootstrap: bool,
) -> NDArray[np.int64]:
    """Draw the row indices for one tree.

    With replacement this is one call to the uniform integer generator,
    not a Python loop over draws. Without replacement it is one sample of
    a combination, shuffled into the order the tree will see.
    """

    if bootstrap:
        return rng.integers(0, n_samples, size=n_draw, dtype=np.int64)
    return rng.choice(n_samples, size=n_draw, replace=False).astype(np.int64, copy=False)


def _align_probabilities(
    proba: NDArray[np.float64],
    tree_classes: NDArray[Any],
    forest_classes: NDArray[Any],
) -> NDArray[np.float64]:
    """Place a tree's probability columns into the forest's class order.

    A bootstrap sample can miss a class. That tree's ``predict_proba``
    then has fewer columns. ``searchsorted`` finds the matching columns
    because both class arrays are sorted. A missing class stays at
    probability 0, and the copied columns still sum to 1.
    """

    aligned = np.zeros((proba.shape[0], forest_classes.size), dtype=np.float64)
    columns = np.searchsorted(forest_classes, tree_classes)
    aligned[:, columns] = proba
    return aligned


def _validate_xy(
    X: ArrayLike,
    y: ArrayLike,
) -> tuple[NDArray[np.float64], NDArray[Any]]:
    """Return a float64 design and a one-dimensional label vector."""

    features = np.asarray(X, dtype=np.float64)
    labels = np.asarray(y)
    if features.ndim == 1:
        features = features.reshape(-1, 1)
    if features.ndim != 2:
        raise ValueError(
            "X must have shape (n_samples, n_features); "
            f"got ndim={features.ndim}."
        )
    if labels.ndim == 2 and labels.shape[1] == 1:
        labels = labels.reshape(-1)
    if labels.ndim != 1:
        raise ValueError("y must be a one-dimensional label vector.")
    if features.shape[0] != labels.shape[0]:
        raise ValueError(
            "X and y must have the same number of rows. "
            f"Got {features.shape[0]} and {labels.shape[0]}."
        )
    if features.shape[0] < 1 or features.shape[1] < 1:
        raise ValueError("X must contain at least one row and one column.")
    if not np.isfinite(features).all():
        raise ValueError("X must contain only finite values.")
    if labels.dtype.kind == "f" and not np.isfinite(labels).all():
        raise ValueError("y must contain only finite values.")
    return features, labels


def _validate_X(X: ArrayLike, n_features: int) -> NDArray[np.float64]:
    """Return a float64 design with the training column count."""

    features = np.asarray(X, dtype=np.float64)
    if features.ndim == 1:
        features = features.reshape(-1, 1)
    if features.ndim != 2:
        raise ValueError(
            "X must have shape (n_samples, n_features); "
            f"got ndim={features.ndim}."
        )
    if features.shape[1] != n_features:
        raise ValueError(f"Expected {n_features} features, got {features.shape[1]}.")
    if not np.isfinite(features).all():
        raise ValueError("X must contain only finite values.")
    return features
