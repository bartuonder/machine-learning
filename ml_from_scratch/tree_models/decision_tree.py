"""Classification trees grown by greedy impurity reduction.

A node splits on one numeric threshold. The split is chosen by information
gain under Gini impurity or Shannon entropy. Recursion builds the left and
right children from the rows that fall on each side of that threshold.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..utils.metrics import accuracy_score

CriterionName = Literal["gini", "entropy"]


class TreeNode:
    """One node of a classification tree.

    ``predicted_class`` is the majority label. Ties go to the earlier entry
    of ``classes_``, which is the smaller label in sorted order. ``counts``
    follows that same class order. An internal node stores the feature,
    the threshold, and the information gain of the split that created its
    children. A leaf has ``left is None``.

    The rule at an internal node is ``X[:, feature] <= threshold`` on the
    left and the complement on the right.
    """

    def __init__(
        self,
        counts: NDArray[np.float64],
        impurity: float,
        n_samples: int,
        predicted_class: Any,
        information_gain: float = 0.0,
        feature: int | None = None,
        threshold: float | None = None,
        left: TreeNode | None = None,
        right: TreeNode | None = None,
    ) -> None:
        self.counts = counts
        self.impurity = impurity
        self.n_samples = n_samples
        self.predicted_class = predicted_class
        self.information_gain = information_gain
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right

    @property
    def is_leaf(self) -> bool:
        """Return whether this node has no children."""

        return self.left is None


def gini_impurity(class_counts: ArrayLike) -> float:
    r"""Return the Gini impurity of one empirical class distribution.

    With :math:`p_k = n_k / n` the class frequencies in ``class_counts``,

    .. math::

        G(p) = 1 - \sum_k p_k^2 = \sum_k p_k(1 - p_k).

    Draw two labels independently from :math:`p`. :math:`G(p)` is the
    probability that they differ, because :math:`\sum_k p_k^2` is the
    probability that they match. A pure node has :math:`G = 0`. The
    maximum on :math:`K` classes is :math:`1 - 1/K`, at the uniform
    distribution.

    A count of zero contributes nothing. Negative counts are rejected.

    Parameters
    ----------
    class_counts : array-like of shape (n_classes,)
        Nonnegative class totals. They do not need to be normalized;
        the empirical distribution divides by their sum. A vector that
        already sums to 1 is a valid input.

    Returns
    -------
    float
        Gini impurity.
    """

    return float(_impurity_from_counts(class_counts, "gini"))


def entropy(class_counts: ArrayLike) -> float:
    r"""Return the Shannon entropy of one empirical class distribution.

    .. math::

        H(p) = - \sum_k p_k \log_2 p_k,
        \qquad 0\log 0 := 0.

    :math:`H(p)` is the expected number of bits needed to name a class
    drawn from :math:`p`. A pure node has entropy 0. Any other base
    multiplies :math:`H` by a positive constant, and information gain by
    the same constant, so the identity of the best split does not depend
    on the base. Base 2 reports the impurity in bits.

    Parameters
    ----------
    class_counts : array-like of shape (n_classes,)
        Nonnegative class totals.

    Returns
    -------
    float
        Entropy in bits.
    """

    return float(_impurity_from_counts(class_counts, "entropy"))


def information_gain(
    parent_counts: ArrayLike,
    left_counts: ArrayLike,
    right_counts: ArrayLike,
    criterion: CriterionName = "gini",
) -> float:
    r"""Return the impurity drop of one binary partition.

    .. math::

        \mathrm{IG}
        = I(p)
          - \frac{n_L}{n} I(p_L)
          - \frac{n_R}{n} I(p_R).

    :math:`I` is Gini impurity or entropy. For entropy this is the mutual
    information between the class and the left/right indicator: it is
    :math:`H(Y) - H(Y \mid \mathrm{split})`. For Gini it is the same
    weighted drop with :math:`I = G`, which is not a mutual information.

    ``left_counts + right_counts`` must equal ``parent_counts``. That is
    the statement that the children are a partition of the parent.

    Parameters
    ----------
    parent_counts, left_counts, right_counts : array-like of shape (n_classes,)
        Nonnegative class totals.
    criterion : {"gini", "entropy"}, default="gini"
        Impurity :math:`I`.

    Returns
    -------
    float
        Information gain, in the units of ``criterion``.
    """

    if criterion not in ("gini", "entropy"):
        raise ValueError("criterion must be 'gini' or 'entropy'.")
    parent = np.asarray(parent_counts, dtype=np.float64).reshape(-1)
    left = np.asarray(left_counts, dtype=np.float64).reshape(-1)
    right = np.asarray(right_counts, dtype=np.float64).reshape(-1)
    if parent.shape != left.shape or parent.shape != right.shape:
        raise ValueError("Parent and child count vectors must share a shape.")
    if not np.allclose(parent, left + right):
        raise ValueError("Left and right counts must sum to the parent counts.")
    total = float(parent.sum())
    if total <= 0.0:
        raise ValueError("The parent node must contain at least one sample.")
    parent_impurity = float(_impurity_from_counts(parent, criterion))
    left_weight = float(left.sum()) / total
    right_weight = float(right.sum()) / total
    left_impurity = float(_impurity_from_counts(left, criterion))
    right_impurity = float(_impurity_from_counts(right, criterion))
    return parent_impurity - left_weight * left_impurity - right_weight * right_impurity


class DecisionTreeClassifier:
    r"""Greedy binary classification tree.

    The fit grows a tree by recursion. A call receives the set of training
    rows that reached the current node and either returns a leaf or

    1. chooses the numeric threshold with the largest information gain,
    2. builds the left child on the rows with ``x_j <= t``,
    3. builds the right child on the rows with ``x_j > t``.

    The impurity of a node is a function of its class counts only. The
    gain of a candidate threshold is

    .. math::

        \mathrm{IG}(j, t)
        = I(p)
          - \frac{n_L}{n} I(p_L)
          - \frac{n_R}{n} I(p_R),

    where both children are defined by :math:`(j, t)` and :math:`I` is
    :func:`gini_impurity` or :func:`entropy`. The parent term does not
    depend on :math:`(j, t)`, so minimizing the weighted child impurity
    is the same decision as maximizing :math:`\mathrm{IG}`.

    A threshold is eligible when both sides contain at least
    ``min_samples_leaf`` rows and the sample-weighted gain is at least
    ``min_impurity_decrease``. The weight is :math:`n_{\mathrm{node}} / N`,
    the fraction of the training sample that reached the node, so a split
    deep in a small leaf has to improve impurity by more, in absolute
    terms, than the same split at the root. The scan accepts a split only
    when :math:`\mathrm{IG} > 0` as well. A pure node, a constant feature,
    and a pattern such as XOR — whose every single-feature threshold has
    gain zero — therefore remain leaves. The algorithm does not look ahead
    to a second split that would become useful only after a zero-gain
    first split.

    Ties break toward the smallest feature index, and within a feature
    toward the leftmost eligible threshold. The search is deterministic.

    Inside one feature the class counts of every threshold are prefixes of
    a single cumulative sum of the label indicators, after the column is
    sorted. The loop in the search runs over features, because each
    feature has its own order. It does not run over rows.

    Parameters
    ----------
    criterion : {"gini", "entropy"}, default="gini"
        Impurity :math:`I`.
    max_depth : int or None, default=None
        Deepest allowed split. ``None`` grows until another stopping rule
        fires. Depth 0 is a single leaf: the majority class on the
        training labels. The root has depth 0.
    min_samples_split : int, default=2
        Smallest node size that may be split.
    min_samples_leaf : int, default=1
        Smallest number of rows allowed on either side of a split.
    min_impurity_decrease : float, default=0.0
        Minimum value of :math:`(n_{\mathrm{node}} / N) \mathrm{IG}`
        required to split.

    Attributes
    ----------
    tree_ : TreeNode
        Root of the fitted tree.
    classes_ : ndarray of shape (n_classes,)
        Sorted training labels. Column ``k`` of ``predict_proba`` is
        ``classes_[k]``.
    n_features_in_ : int
        Columns of ``X``.
    n_samples_ : int
        Rows of the last ``fit``.
    n_classes_ : int
        ``classes_.size``.
    depth_ : int
        Depth of the deepest node. A leaf-only tree has depth 0.
    n_leaves_ : int
        Number of leaves.
    feature_importances_ : ndarray of shape (n_features,)
        Share of the total sample-weighted information gain attributed to
        each feature. The weights sum to 1 when the root splits, and they
        are 0 when the root is a leaf.
    """

    def __init__(
        self,
        criterion: CriterionName = "gini",
        max_depth: int | None = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        min_impurity_decrease: float = 0.0,
    ) -> None:
        if criterion not in ("gini", "entropy"):
            raise ValueError("criterion must be 'gini' or 'entropy'.")
        if max_depth is not None and (
            isinstance(max_depth, bool) or not isinstance(max_depth, int) or max_depth < 0
        ):
            raise ValueError("max_depth must be a nonnegative integer or None.")
        if (
            isinstance(min_samples_split, bool)
            or not isinstance(min_samples_split, int)
            or min_samples_split < 2
        ):
            raise ValueError("min_samples_split must be an integer >= 2.")
        if (
            isinstance(min_samples_leaf, bool)
            or not isinstance(min_samples_leaf, int)
            or min_samples_leaf < 1
        ):
            raise ValueError("min_samples_leaf must be an integer >= 1.")
        self.criterion: CriterionName = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_decrease = _nonnegative_float(
            "min_impurity_decrease",
            min_impurity_decrease,
        )
        self.tree_: TreeNode | None = None
        self.classes_: NDArray[Any] | None = None
        self.n_features_in_: int | None = None
        self.n_samples_: int | None = None
        self.n_classes_: int | None = None
        self.depth_: int = 0
        self.n_leaves_: int = 0
        self.feature_importances_: NDArray[np.float64] | None = None
        self._X: NDArray[np.float64] | None = None
        self._y: NDArray[np.int64] | None = None

    def fit(self, X: ArrayLike, y: ArrayLike) -> DecisionTreeClassifier:
        """Grow the tree from the impurity criterion.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Numeric design. A one-dimensional array is one feature column.
        y : array-like of shape (n_samples,)
            Class labels. One class is allowed: the tree is then a leaf.

        Returns
        -------
        self
        """

        features, labels = _validate_xy(X, y)
        classes, codes = np.unique(labels, return_inverse=True)
        self.classes_ = classes
        self.n_classes_ = int(classes.size)
        self.n_features_in_ = int(features.shape[1])
        self.n_samples_ = int(features.shape[0])
        self.depth_ = 0
        self.n_leaves_ = 0
        self.feature_importances_ = np.zeros(features.shape[1], dtype=np.float64)
        # Held only while the recursion reads rows. Cleared before return
        # so a fitted tree does not keep the training matrix alive.
        self._X = features
        self._y = codes.astype(np.int64, copy=False)
        self.tree_ = self._build(np.arange(features.shape[0], dtype=np.int64), depth=0)
        self._X = None
        self._y = None
        assert self.feature_importances_ is not None
        total_gain = float(self.feature_importances_.sum())
        if total_gain > 0.0:
            self.feature_importances_ /= total_gain
        return self

    def predict_proba(self, X: ArrayLike) -> NDArray[np.float64]:
        """Return the training class frequencies of the leaf each row reaches.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Rows on the same feature scale used in ``fit``.

        Returns
        -------
        ndarray of shape (n_samples, n_classes)
            Leaf frequencies. Column ``k`` corresponds to ``classes_[k]``.
            Each row sums to 1.
        """

        self._check_fitted()
        assert self.tree_ is not None and self.n_classes_ is not None
        features = self._validate_X(X)
        probabilities = np.zeros((features.shape[0], self.n_classes_), dtype=np.float64)
        self._fill_probabilities(
            self.tree_,
            features,
            np.arange(features.shape[0], dtype=np.int64),
            probabilities,
        )
        return probabilities

    def predict(self, X: ArrayLike) -> NDArray[Any]:
        """Return the majority class of the leaf each row reaches.

        Ties in a leaf were broken toward the earlier class when the node
        was built. ``argmax`` on the leaf frequencies uses that same rule,
        because equal counts stay equal after division by the node size
        and ``argmax`` returns the first maximum.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Rows on the same feature scale used in ``fit``.

        Returns
        -------
        ndarray of shape (n_samples,)
            Predicted labels.
        """

        self._check_fitted()
        assert self.classes_ is not None
        codes = np.argmax(self.predict_proba(X), axis=1)
        return self.classes_[codes]

    def score(self, X: ArrayLike, y: ArrayLike) -> float:
        """Return the classification accuracy of ``predict(X)``."""

        return accuracy_score(y, self.predict(X))

    def _build(self, indices: NDArray[np.int64], depth: int) -> TreeNode:
        """Return the node fitted to ``indices``, then its children."""

        assert self._y is not None and self.classes_ is not None and self.n_classes_ is not None
        codes = self._y[indices]
        counts = np.bincount(codes, minlength=self.n_classes_).astype(np.float64)
        impurity = float(_impurity_from_counts(counts, self.criterion))
        # argmax returns the first maximum, so a tied majority is the
        # earlier class in the sorted ``classes_`` array.
        predicted = self.classes_[int(np.argmax(counts))]
        node = TreeNode(
            counts=counts,
            impurity=impurity,
            n_samples=int(indices.size),
            predicted_class=predicted,
        )
        self.depth_ = max(self.depth_, depth)
        if self._stop(indices.size, depth, impurity):
            self.n_leaves_ += 1
            return node
        split = self._best_split(indices, impurity)
        if split is None:
            self.n_leaves_ += 1
            return node
        feature, threshold, gain = split
        assert self._X is not None and self.feature_importances_ is not None
        assert self.n_samples_ is not None
        node.feature = feature
        node.threshold = threshold
        node.information_gain = gain
        # (n_node / N) * IG is this node's contribution to the feature's
        # total weighted gain. Normalization to a share happens after the
        # whole tree exists.
        self.feature_importances_[feature] += (indices.size / self.n_samples_) * gain
        go_left = self._X[indices, feature] <= threshold
        node.left = self._build(indices[go_left], depth + 1)
        node.right = self._build(indices[~go_left], depth + 1)
        return node

    def _stop(self, n_samples: int, depth: int, impurity: float) -> bool:
        """Return whether the current node must be a leaf."""

        if impurity <= 0.0:
            return True
        if self.max_depth is not None and depth >= self.max_depth:
            return True
        if n_samples < self.min_samples_split:
            return True
        if n_samples < 2 * self.min_samples_leaf:
            return True
        return False

    def _best_split(
        self,
        indices: NDArray[np.int64],
        parent_impurity: float,
    ) -> tuple[int, float, float] | None:
        """Return ``(feature, threshold, information_gain)`` or no split.

        For one sorted column, ``cumulative[i]`` is the class-count vector
        of the first ``i + 1`` rows. A legal cut sits only between two
        different adjacent values, so a run of ties is never broken: those
        rows carry the same feature value and cannot fall on different
        sides of a threshold.
        """

        assert (
            self._X is not None
            and self._y is not None
            and self.n_classes_ is not None
            and self.n_samples_ is not None
        )
        features = self._X[indices]
        codes = self._y[indices]
        n_rows = int(codes.size)
        best_gain = 0.0
        best: tuple[int, float, float] | None = None
        # One pass per column. The ordering that makes prefix counts equal
        # the left side of a threshold is different for every feature.
        for feature in range(features.shape[1]):
            column = features[:, feature]
            order = np.argsort(column, kind="mergesort")
            sorted_values = column[order]
            sorted_codes = codes[order]
            if sorted_values[0] == sorted_values[-1]:
                continue
            change = sorted_values[:-1] != sorted_values[1:]
            positions = np.flatnonzero(change)
            left_sizes = positions + 1
            right_sizes = n_rows - left_sizes
            eligible = (left_sizes >= self.min_samples_leaf) & (
                right_sizes >= self.min_samples_leaf
            )
            if not np.any(eligible):
                continue
            positions = positions[eligible]
            left_sizes = left_sizes[eligible]
            right_sizes = right_sizes[eligible]
            # Indicator rows are vertices of the simplex. Their prefix sums
            # are the left-child class counts at every cut.
            one_hot = np.zeros((n_rows, self.n_classes_), dtype=np.float64)
            one_hot[np.arange(n_rows), sorted_codes] = 1.0
            cumulative = np.cumsum(one_hot, axis=0)
            left_counts = cumulative[positions]
            right_counts = cumulative[-1] - left_counts
            left_impurity = _impurity_from_counts(left_counts, self.criterion)
            right_impurity = _impurity_from_counts(right_counts, self.criterion)
            weighted = (left_sizes * left_impurity + right_sizes * right_impurity) / n_rows
            gain = parent_impurity - weighted
            scaled_gain = (n_rows / self.n_samples_) * gain
            admissible = scaled_gain >= self.min_impurity_decrease
            if not np.any(admissible):
                continue
            scored = np.where(admissible, gain, -np.inf)
            local = int(np.argmax(scored))
            chosen_gain = float(scored[local])
            # Strict inequality keeps an earlier feature, and within this
            # feature the leftmost cut, when several cuts share the gain.
            if chosen_gain > best_gain:
                position = int(positions[local])
                left_value = float(sorted_values[position])
                right_value = float(sorted_values[position + 1])
                threshold = left_value + (right_value - left_value) / 2.0
                best_gain = chosen_gain
                best = (feature, threshold, chosen_gain)
        return best

    def _fill_probabilities(
        self,
        node: TreeNode,
        features: NDArray[np.float64],
        indices: NDArray[np.int64],
        probabilities: NDArray[np.float64],
    ) -> None:
        """Write leaf frequencies for every row in ``indices``.

        The recursive calls follow the tree. The comparison that decides
        left versus right is one vectorized predicate on the rows that
        reached this node.
        """

        if node.left is None or node.right is None or node.feature is None or node.threshold is None:
            probabilities[indices] = node.counts / node.n_samples
            return
        go_left = features[indices, node.feature] <= node.threshold
        left_indices = indices[go_left]
        right_indices = indices[~go_left]
        if left_indices.size:
            self._fill_probabilities(node.left, features, left_indices, probabilities)
        if right_indices.size:
            self._fill_probabilities(node.right, features, right_indices, probabilities)

    def _check_fitted(self) -> None:
        if self.tree_ is None or self.classes_ is None or self.n_features_in_ is None:
            raise RuntimeError("DecisionTreeClassifier must be fitted before use.")

    def _validate_X(self, X: ArrayLike) -> NDArray[np.float64]:
        design = np.asarray(X, dtype=np.float64)
        if design.ndim == 1:
            design = design.reshape(-1, 1)
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
        return (
            f"DecisionTreeClassifier(criterion={self.criterion!r}, "
            f"max_depth={self.max_depth})"
        )


def _impurity_from_counts(
    class_counts: ArrayLike,
    criterion: CriterionName,
) -> NDArray[np.float64]:
    """Return impurity for one count vector or for a batch of them.

    A 1-d input has shape ``(n_classes,)`` and produces a 0-d result. A
    2-d input has shape ``(n_candidates, n_classes)`` and produces one
    impurity per row. The 2-d path is the split scan: every candidate
    threshold is one row, and the reduction over classes is a single sum.
    """

    if criterion not in ("gini", "entropy"):
        raise ValueError("criterion must be 'gini' or 'entropy'.")
    counts = np.asarray(class_counts, dtype=np.float64)
    if counts.ndim == 1:
        counts = counts.reshape(1, -1)
        squeeze = True
    elif counts.ndim == 2:
        squeeze = False
    else:
        raise ValueError("class counts must be a vector or a matrix of vectors.")
    if counts.size == 0 or counts.shape[1] == 0:
        raise ValueError("class counts must contain at least one class.")
    if np.any(counts < 0.0):
        raise ValueError("class counts must be nonnegative.")
    totals = counts.sum(axis=1, keepdims=True)
    # Divide only where the node is nonempty. An empty candidate is not
    # used by the tree; the public helpers reject a parent with total 0.
    probabilities = np.divide(
        counts,
        totals,
        out=np.zeros_like(counts),
        where=totals > 0.0,
    )
    if criterion == "gini":
        impurity = 1.0 - np.sum(probabilities * probabilities, axis=1)
    else:
        logged = np.zeros_like(probabilities)
        positive = probabilities > 0.0
        # The masked assignment implements 0 log 0 = 0. log2 is bits;
        # a change of base would scale every gain by the same constant.
        logged[positive] = probabilities[positive] * np.log2(probabilities[positive])
        impurity = -np.sum(logged, axis=1)
    if squeeze:
        return impurity.reshape(())
    return impurity


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


def _nonnegative_float(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.floating)):
        raise TypeError(f"{name} must be a real number.")
    number = float(value)
    if not np.isfinite(number) or number < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative.")
    return number
