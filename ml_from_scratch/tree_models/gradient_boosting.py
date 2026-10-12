"""Second-order gradient boosting with regularized regression trees.

Each round fits a tree to the Newton step of the training loss. The split
score is the reduction in the second-order Taylor expansion, which is the
gain used by XGBoost. Squared error and binary logistic regression are the
two losses.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..optimization.losses import BinaryCrossEntropy, MeanSquaredError, sigmoid
from ..utils.metrics import accuracy_score, r2_score


class BoostingNode:
    """One node of a second-order regression tree.

    ``weight`` is the optimal constant prediction for the rows in the node,

    .. math::

        w = -\\frac{\\sum_i g_i}{\\sum_i h_i + \\lambda}.

    An internal node also stores the feature, the threshold, and the gain
    of the split that created its children. Prediction walks to a leaf and
    returns that leaf's ``weight``. The left branch is
    ``X[:, feature] <= threshold``.
    """

    def __init__(
        self,
        weight: float,
        gradient_sum: float,
        hessian_sum: float,
        n_samples: int,
        gain: float = 0.0,
        feature: int | None = None,
        threshold: float | None = None,
        left: BoostingNode | None = None,
        right: BoostingNode | None = None,
    ) -> None:
        self.weight = weight
        self.gradient_sum = gradient_sum
        self.hessian_sum = hessian_sum
        self.n_samples = n_samples
        self.gain = gain
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right

    @property
    def is_leaf(self) -> bool:
        """Return whether this node has no children."""

        return self.left is None


def leaf_weight(gradient_sum: float, hessian_sum: float, reg_lambda: float) -> float:
    r"""Return the minimizer of one leaf's quadratic objective.

    On a leaf with gradient sum :math:`G` and Hessian sum :math:`H`, the
    second-order objective in the leaf weight :math:`w` is

    .. math::

        \frac{1}{2}(H + \lambda) w^2 + G w.

    The derivative :math:`(H + \lambda) w + G` vanishes at

    .. math::

        w^\star = -\frac{G}{H + \lambda},

    provided :math:`H + \lambda > 0`. :math:`\lambda \ge 0` is the L2
    penalty on the leaf weight. These sums are over the rows in the leaf,
    with one summand per row: they are not divided by the sample size.
    Dividing :math:`G` and :math:`H` by :math:`m` would change :math:`w`
    whenever :math:`\lambda > 0`.

    Parameters
    ----------
    gradient_sum : float
        :math:`G = \sum_{i \in \mathrm{leaf}} g_i`.
    hessian_sum : float
        :math:`H = \sum_{i \in \mathrm{leaf}} h_i`.
    reg_lambda : float
        Nonnegative penalty :math:`\lambda`.

    Returns
    -------
    float
        Optimal leaf weight.
    """

    if isinstance(reg_lambda, bool) or not isinstance(reg_lambda, (int, float, np.floating)):
        raise TypeError("reg_lambda must be a real number.")
    penalty = float(reg_lambda)
    if not np.isfinite(penalty) or penalty < 0.0:
        raise ValueError("reg_lambda must be finite and nonnegative.")
    numerator = float(gradient_sum)
    denominator = float(hessian_sum) + penalty
    if not np.isfinite(numerator) or not np.isfinite(denominator):
        raise ValueError("gradient_sum and hessian_sum must be finite.")
    if denominator <= 0.0:
        raise ValueError(
            "Leaf weight is undefined when the summed Hessian plus reg_lambda is not positive."
        )
    return -numerator / denominator


def second_order_gain(
    gradient_left: float,
    hessian_left: float,
    gradient_right: float,
    hessian_right: float,
    reg_lambda: float,
    gamma: float = 0.0,
) -> float:
    r"""Return the objective reduction of one binary split.

    With :math:`G_L, H_L` the sums on the left child, :math:`G_R, H_R` the
    sums on the right, and :math:`G = G_L + G_R`, :math:`H = H_L + H_R`,

    .. math::

        \mathrm{Gain}
        = \frac{1}{2}\left[
            \frac{G_L^2}{H_L + \lambda}
            + \frac{G_R^2}{H_R + \lambda}
            - \frac{G^2}{H + \lambda}
          \right]
          - \gamma.

    The expression in brackets is how much the sum of leaf objectives
    falls when one constant weight is replaced by two. :math:`\gamma \ge 0`
    is the cost of adding that extra leaf to the penalty
    :math:`\Omega = \gamma T + \frac{1}{2}\lambda\sum_j w_j^2`, where
    :math:`T` is the number of leaves. A split is useful when the gain
    is positive.

    Parameters
    ----------
    gradient_left, hessian_left, gradient_right, hessian_right : float
        Sums of per-example derivatives on each side of the threshold.
    reg_lambda : float
        Nonnegative :math:`\lambda`.
    gamma : float, default=0.0
        Nonnegative split cost :math:`\gamma`.

    Returns
    -------
    float
        Gain. Positive means the split lowers the regularized objective.
    """

    gains = _gain_array(
        np.asarray([gradient_left], dtype=np.float64),
        np.asarray([hessian_left], dtype=np.float64),
        np.asarray([gradient_right], dtype=np.float64),
        np.asarray([hessian_right], dtype=np.float64),
        reg_lambda,
        gamma,
    )
    return float(gains[0])


class _NewtonTree:
    r"""Regression tree whose splits maximize :func:`second_order_gain`.

    Parameters
    ----------
    max_depth : int or None
        Deepest split. The root has depth 0. ``None`` does not cap depth.
    min_samples_split : int
        Smallest node that may be split.
    min_samples_leaf : int
        Smallest number of rows on either side of a split.
    min_child_weight : float
        Smallest summed Hessian allowed on either side of a split.
    reg_lambda : float
        :math:`\lambda` in the leaf weight and the gain.
    gamma : float
        :math:`\gamma` subtracted from the gain.
    """

    def __init__(
        self,
        max_depth: int | None = 3,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        min_child_weight: float = 0.0,
        reg_lambda: float = 1.0,
        gamma: float = 0.0,
    ) -> None:
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_child_weight = min_child_weight
        self.reg_lambda = reg_lambda
        self.gamma = gamma
        self.tree_: BoostingNode | None = None
        self.feature_importances_: NDArray[np.float64] | None = None
        self.n_features_in_: int | None = None
        self.depth_: int = 0
        self.n_leaves_: int = 0
        self._X: NDArray[np.float64] | None = None
        self._gradient: NDArray[np.float64] | None = None
        self._hessian: NDArray[np.float64] | None = None

    def fit(
        self,
        X: NDArray[np.float64],
        gradient: NDArray[np.float64],
        hessian: NDArray[np.float64],
    ) -> _NewtonTree:
        """Grow the tree on per-example derivatives of shape ``(n_samples,)``."""

        if gradient.shape != (X.shape[0],) or hessian.shape != (X.shape[0],):
            raise ValueError("gradient and hessian must each have shape (n_samples,).")
        if not np.isfinite(gradient).all() or not np.isfinite(hessian).all():
            raise ValueError("gradient and hessian must be finite.")
        if np.any(hessian < 0.0):
            raise ValueError("Per-example Hessians must be nonnegative.")
        self.n_features_in_ = int(X.shape[1])
        self.depth_ = 0
        self.n_leaves_ = 0
        self.feature_importances_ = np.zeros(X.shape[1], dtype=np.float64)
        self._X = X
        self._gradient = gradient
        self._hessian = hessian
        self.tree_ = self._build(np.arange(X.shape[0], dtype=np.int64), depth=0)
        self._X = None
        self._gradient = None
        self._hessian = None
        assert self.feature_importances_ is not None
        total = float(self.feature_importances_.sum())
        if total > 0.0:
            self.feature_importances_ /= total
        return self

    def predict(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return the leaf weight of every row. Shape ``(n_samples,)``."""

        if self.tree_ is None or self.n_features_in_ is None:
            raise RuntimeError("The boosting tree must be fitted before predict.")
        if X.ndim != 2 or X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Expected shape (n_samples, {self.n_features_in_}), got {X.shape}."
            )
        weights = np.empty(X.shape[0], dtype=np.float64)
        self._fill(self.tree_, X, np.arange(X.shape[0], dtype=np.int64), weights)
        return weights

    def _build(self, indices: NDArray[np.int64], depth: int) -> BoostingNode:
        """Return the node for ``indices`` and, when a split pays, its children."""

        assert self._gradient is not None and self._hessian is not None
        gradient = self._gradient[indices]
        hessian = self._hessian[indices]
        gradient_sum = float(gradient.sum())
        hessian_sum = float(hessian.sum())
        weight = leaf_weight(gradient_sum, hessian_sum, self.reg_lambda)
        node = BoostingNode(
            weight=weight,
            gradient_sum=gradient_sum,
            hessian_sum=hessian_sum,
            n_samples=int(indices.size),
        )
        self.depth_ = max(self.depth_, depth)
        if self._stop(indices.size, depth, hessian_sum):
            self.n_leaves_ += 1
            return node
        split = self._best_split(indices)
        if split is None:
            self.n_leaves_ += 1
            return node
        feature, threshold, gain = split
        assert self._X is not None and self.feature_importances_ is not None
        node.feature = feature
        node.threshold = threshold
        node.gain = gain
        self.feature_importances_[feature] += gain
        go_left = self._X[indices, feature] <= threshold
        node.left = self._build(indices[go_left], depth + 1)
        node.right = self._build(indices[~go_left], depth + 1)
        return node

    def _stop(self, n_samples: int, depth: int, hessian_sum: float) -> bool:
        """Return whether this node cannot host two legal children."""

        if self.max_depth is not None and depth >= self.max_depth:
            return True
        if n_samples < self.min_samples_split:
            return True
        if n_samples < 2 * self.min_samples_leaf:
            return True
        # Two children would each need summed Hessian ``min_child_weight``.
        if hessian_sum < 2.0 * self.min_child_weight:
            return True
        return False

    def _best_split(
        self,
        indices: NDArray[np.int64],
    ) -> tuple[int, float, float] | None:
        """Return ``(feature, threshold, gain)`` or no split.

        Sorting a column turns every threshold's left child into a prefix.
        The prefix sums of :math:`g` and :math:`h` are therefore
        :math:`G_L` and :math:`H_L` at every cut, in one ``cumsum`` per
        feature. The loop is over columns because each column has its own
        order.
        """

        assert self._X is not None and self._gradient is not None and self._hessian is not None
        features = self._X[indices]
        gradient = self._gradient[indices]
        hessian = self._hessian[indices]
        n_rows = int(indices.size)
        best_gain = 0.0
        best: tuple[int, float, float] | None = None
        for feature in range(features.shape[1]):
            column = features[:, feature]
            order = np.argsort(column, kind="mergesort")
            sorted_values = column[order]
            if sorted_values[0] == sorted_values[-1]:
                continue
            change = sorted_values[:-1] != sorted_values[1:]
            positions = np.flatnonzero(change)
            if positions.size == 0:
                continue
            left_sizes = positions + 1
            right_sizes = n_rows - left_sizes
            gradient_prefix = np.cumsum(gradient[order])
            hessian_prefix = np.cumsum(hessian[order])
            gradient_left = gradient_prefix[positions]
            hessian_left = hessian_prefix[positions]
            gradient_right = gradient_prefix[-1] - gradient_left
            hessian_right = hessian_prefix[-1] - hessian_left
            eligible = (
                (left_sizes >= self.min_samples_leaf)
                & (right_sizes >= self.min_samples_leaf)
                & (hessian_left >= self.min_child_weight)
                & (hessian_right >= self.min_child_weight)
            )
            gain = _gain_array(
                gradient_left,
                hessian_left,
                gradient_right,
                hessian_right,
                self.reg_lambda,
                self.gamma,
            )
            gain = np.where(eligible, gain, -np.inf)
            if not np.any(gain > 0.0):
                continue
            local = int(np.argmax(gain))
            chosen = float(gain[local])
            # Strict inequality keeps an earlier feature, and argmax keeps
            # the leftmost cut, when several cuts share a gain.
            if chosen > best_gain:
                position = int(positions[local])
                left_value = float(sorted_values[position])
                right_value = float(sorted_values[position + 1])
                threshold = left_value + (right_value - left_value) / 2.0
                best_gain = chosen
                best = (feature, threshold, chosen)
        return best

    def _fill(
        self,
        node: BoostingNode,
        features: NDArray[np.float64],
        indices: NDArray[np.int64],
        weights: NDArray[np.float64],
    ) -> None:
        """Write leaf weights for the rows in ``indices``."""

        if node.left is None or node.right is None or node.feature is None or node.threshold is None:
            weights[indices] = node.weight
            return
        go_left = features[indices, node.feature] <= node.threshold
        left_indices = indices[go_left]
        right_indices = indices[~go_left]
        if left_indices.size:
            self._fill(node.left, features, left_indices, weights)
        if right_indices.size:
            self._fill(node.right, features, right_indices, weights)


class _Booster:
    r"""Shared stagewise loop for regression and binary classification.

    Subclasses define the per-example derivatives and the scalar training
    loss. The additive model is

    .. math::

        F_0(x) = \bar{F},
        \qquad
        F_t(x) = F_{t-1}(x) + \eta\, f_t(x),

    where :math:`f_t` is a :class:`_NewtonTree` fit to
    :math:`(g_i, h_i)` evaluated at :math:`F_{t-1}`, and :math:`\eta` is
    the learning rate. :math:`\eta` scales the leaf weights after the tree
    is built. It does not enter the gain, so the tree is the Newton
    direction and :math:`\eta` chooses how far to travel along it.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int | None = 3,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        min_child_weight: float = 0.0,
        reg_lambda: float = 1.0,
        gamma: float = 0.0,
        subsample: float = 1.0,
        random_state: int | None = None,
    ) -> None:
        if isinstance(n_estimators, bool) or not isinstance(n_estimators, int) or n_estimators < 1:
            raise ValueError("n_estimators must be an integer >= 1.")
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
        if isinstance(random_state, bool) or not isinstance(random_state, (int, type(None))):
            raise TypeError("random_state must be an int or None.")
        self.n_estimators = n_estimators
        self.learning_rate = _positive_float("learning_rate", learning_rate)
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_child_weight = _nonnegative_float("min_child_weight", min_child_weight)
        self.reg_lambda = _nonnegative_float("reg_lambda", reg_lambda)
        self.gamma = _nonnegative_float("gamma", gamma)
        self.subsample = _unit_fraction("subsample", subsample)
        self.random_state = random_state
        self.estimators_: list[_NewtonTree] = []
        self.base_score_: float | None = None
        self.loss_history_: list[float] = []
        self.feature_importances_: NDArray[np.float64] | None = None
        self.n_features_in_: int | None = None
        self.n_samples_: int | None = None

    def _fit_stages(
        self,
        features: NDArray[np.float64],
        targets: NDArray[np.float64],
        base_score: float,
    ) -> None:
        """Add ``n_estimators`` trees starting from a constant score."""

        n_samples, n_features = int(features.shape[0]), int(features.shape[1])
        self.n_features_in_ = n_features
        self.n_samples_ = n_samples
        self.base_score_ = float(base_score)
        self.estimators_ = []
        self.feature_importances_ = np.zeros(n_features, dtype=np.float64)
        raw = np.full(n_samples, self.base_score_, dtype=np.float64)
        self.loss_history_ = [self._risk(targets, raw)]
        rng = np.random.default_rng(self.random_state)
        n_draw = max(1, int(np.ceil(self.subsample * n_samples)))

        for _stage in range(self.n_estimators):
            gradient, hessian = self._derivatives(targets, raw)
            if self.subsample < 1.0:
                # A without-replacement subset estimates the full-sample
                # sums G and H. The tree is still evaluated on every row
                # when it is added to F.
                chosen = rng.choice(n_samples, size=n_draw, replace=False)
                tree_x = features[chosen]
                tree_g = gradient[chosen]
                tree_h = hessian[chosen]
            else:
                tree_x = features
                tree_g = gradient
                tree_h = hessian
            tree = _NewtonTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                min_child_weight=self.min_child_weight,
                reg_lambda=self.reg_lambda,
                gamma=self.gamma,
            )
            tree.fit(tree_x, tree_g, tree_h)
            raw = raw + self.learning_rate * tree.predict(features)
            self.loss_history_.append(self._risk(targets, raw))
            self.estimators_.append(tree)
            assert tree.feature_importances_ is not None
            self.feature_importances_ += tree.feature_importances_

        total = float(self.feature_importances_.sum())
        if total > 0.0:
            self.feature_importances_ /= total

    def _raw_predict(self, features: NDArray[np.float64]) -> NDArray[np.float64]:
        """Evaluate :math:`F_T(x)` for every row."""

        if self.base_score_ is None or self.n_features_in_ is None:
            raise RuntimeError("The booster must be fitted before prediction.")
        if features.ndim != 2 or features.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Expected shape (n_samples, {self.n_features_in_}), got {features.shape}."
            )
        raw = np.full(features.shape[0], self.base_score_, dtype=np.float64)
        for tree in self.estimators_:
            raw += self.learning_rate * tree.predict(features)
        return raw

    def _derivatives(
        self,
        targets: NDArray[np.float64],
        raw: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return per-example ``(g, h)`` of shape ``(n_samples,)`` each."""

        raise NotImplementedError

    def _risk(self, targets: NDArray[np.float64], raw: NDArray[np.float64]) -> float:
        """Return the averaged training loss at ``raw``."""

        raise NotImplementedError


class GradientBoostingRegressor(_Booster):
    r"""Second-order boosting of squared error.

    The per-example loss is the Gaussian term

    .. math::

        \ell(y, F) = \frac{1}{2}(y - F)^2.

    Its derivatives with respect to the prediction :math:`F` are

    .. math::

        g = F - y, \qquad h = 1.

    :class:`~ml_from_scratch.optimization.losses.MeanSquaredError` is the
    average of this :math:`\ell`. The tree receives the unaveraged
    :math:`g` and :math:`h`, then sums them. The recorded ``loss_history_``
    is the averaged loss, so entry ``t`` is
    :meth:`MeanSquaredError.loss` of :math:`F_t`.

    The constant start :math:`F_0` is the sample mean of ``y``. At that
    point :math:`\sum_i g_i = 0`, so the root leaf weight of the first
    tree is 0. Splits can still have positive gain, because a child sum
    :math:`G_L` need not vanish.

    Parameters
    ----------
    n_estimators : int, default=100
        Number of trees.
    learning_rate : float, default=0.1
        Shrinkage :math:`\eta` applied to each finished tree.
    max_depth : int or None, default=3
        Maximum split depth of each tree. ``0`` fits a constant leaf.
    min_samples_split : int, default=2
        Smallest node that may be split.
    min_samples_leaf : int, default=1
        Smallest number of rows on either side of a split.
    min_child_weight : float, default=0.0
        Smallest summed Hessian on either side. For this loss the Hessian
        of a row is 1, so this is a minimum row count in addition to
        ``min_samples_leaf``.
    reg_lambda : float, default=1.0
        L2 penalty :math:`\lambda` on leaf weights.
    gamma : float, default=0.0
        Split cost :math:`\gamma`.
    subsample : float, default=1.0
        Fraction of rows, drawn without replacement, used to build each
        tree. ``1`` uses every row. The tree is then applied to all rows.
    random_state : int or None, default=None
        Seed of the subsample draws.

    Attributes
    ----------
    estimators_ : list
        Fitted second-order trees.
    base_score_ : float
        Constant :math:`F_0`.
    loss_history_ : list of float
        Averaged squared loss at :math:`F_0, F_1, \ldots, F_T`.
    feature_importances_ : ndarray of shape (n_features,)
        Share of total split gain. Zeros when no tree split.
    n_features_in_ : int
        Columns of ``X``.
    n_samples_ : int
        Rows of the last ``fit``.
    """

    def fit(self, X: ArrayLike, y: ArrayLike) -> GradientBoostingRegressor:
        """Fit the additive expansion of squared error.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Numeric design. A one-dimensional array is one feature column.
        y : array-like of shape (n_samples,)
            Real-valued targets.

        Returns
        -------
        self
        """

        features, targets = _validate_regression_xy(X, y)
        self._fit_stages(features, targets, base_score=float(targets.mean()))
        return self

    def predict(self, X: ArrayLike) -> NDArray[np.float64]:
        """Return :math:`F_T(x)` for each row.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Rows on the feature scale used in ``fit``.

        Returns
        -------
        ndarray of shape (n_samples,)
            Predicted targets.
        """

        self._check_fitted()
        assert self.n_features_in_ is not None
        return self._raw_predict(_validate_X(X, self.n_features_in_))

    def score(self, X: ArrayLike, y: ArrayLike) -> float:
        """Return :math:`R^2` of ``predict(X)`` against ``y``."""

        return r2_score(y, self.predict(X))

    def _derivatives(
        self,
        targets: NDArray[np.float64],
        raw: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return :math:`g = F - y` and :math:`h = 1`, both shape ``(n_samples,)``."""

        return raw - targets, np.ones_like(raw)

    def _risk(self, targets: NDArray[np.float64], raw: NDArray[np.float64]) -> float:
        """Return the averaged Gaussian loss, matching :class:`MeanSquaredError`."""

        return MeanSquaredError.loss(raw, targets)

    def _check_fitted(self) -> None:
        if self.base_score_ is None:
            raise RuntimeError("GradientBoostingRegressor must be fitted before use.")

    def __repr__(self) -> str:
        return (
            f"GradientBoostingRegressor(n_estimators={self.n_estimators}, "
            f"learning_rate={self.learning_rate}, max_depth={self.max_depth})"
        )


class GradientBoostingClassifier(_Booster):
    r"""Second-order boosting of the Bernoulli negative log-likelihood.

    The score :math:`F` is the logit. The success probability is
    :math:`\sigma(F)`, and the per-example loss is

    .. math::

        \ell(y, F) = \operatorname{softplus}(F) - y F,
        \qquad y \in \{0, 1\}.

    Differentiating with respect to :math:`F` gives the same cancellation
    as logistic regression,

    .. math::

        g = \sigma(F) - y,
        \qquad
        h = \sigma(F)\,(1 - \sigma(F)).

    :class:`~ml_from_scratch.optimization.losses.BinaryCrossEntropy`
    averages this :math:`\ell`. Its logit-space gradient equals :math:`g / m`.
    The tree sums the unaveraged :math:`g` and :math:`h`. ``loss_history_``
    stores the average, via :meth:`BinaryCrossEntropy.loss_with_logits`.

    :math:`F_0` is the logit of the training frequency of the positive
    class, clipped into ``(10^{-6}, 1 - 10^{-6})`` before the logit so a
    class that is almost pure does not start at an infinite score. The
    positive class is ``classes_[1]``, the second label in sorted order.

    ``predict_proba`` returns :math:`(\sigma(-F), \sigma(F))`. ``predict``
    returns the positive class when :math:`\sigma(F) \ge` ``threshold``.

    Parameters match :class:`GradientBoostingRegressor`, with one addition:

    threshold : float, default=0.5
        Probability cutoff used by ``predict``. The boundary predicts the
        positive class.

    Attributes
    ----------
    estimators_ : list
        Fitted second-order trees.
    classes_ : ndarray of shape (2,)
        Sorted labels. Column 1 of ``predict_proba`` is ``classes_[1]``.
    base_score_ : float
        Constant logit :math:`F_0`.
    loss_history_ : list of float
        Mean Bernoulli negative log-likelihood at each :math:`F_t`.
    feature_importances_ : ndarray of shape (n_features,)
        Share of total split gain.
    n_features_in_ : int
        Columns of ``X``.
    n_samples_ : int
        Rows of the last ``fit``.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int | None = 3,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        min_child_weight: float = 0.0,
        reg_lambda: float = 1.0,
        gamma: float = 0.0,
        subsample: float = 1.0,
        threshold: float = 0.5,
        random_state: int | None = None,
    ) -> None:
        super().__init__(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_child_weight=min_child_weight,
            reg_lambda=reg_lambda,
            gamma=gamma,
            subsample=subsample,
            random_state=random_state,
        )
        if isinstance(threshold, bool) or not isinstance(threshold, (int, float, np.floating)):
            raise TypeError("threshold must be a real number.")
        if not 0.0 <= float(threshold) <= 1.0:
            raise ValueError("threshold must lie in [0, 1].")
        self.threshold = float(threshold)
        self.classes_: NDArray[Any] | None = None

    def fit(self, X: ArrayLike, y: ArrayLike) -> GradientBoostingClassifier:
        """Fit the additive expansion of the Bernoulli log-likelihood.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Numeric design. A one-dimensional array is one feature column.
        y : array-like of shape (n_samples,)
            Binary labels.

        Returns
        -------
        self
        """

        features, labels = _validate_xy(X, y)
        classes = np.unique(labels)
        if classes.size != 2:
            raise ValueError(
                "GradientBoostingClassifier fits a Bernoulli likelihood and "
                f"needs exactly two classes. Observed {classes.size}: {classes.tolist()}."
            )
        targets = (labels == classes[1]).astype(np.float64)
        frequency = float(np.clip(targets.mean(), 1e-6, 1.0 - 1e-6))
        base = float(np.log(frequency / (1.0 - frequency)))
        self.classes_ = classes
        self._fit_stages(features, targets, base_score=base)
        return self

    def decision_function(self, X: ArrayLike) -> NDArray[np.float64]:
        """Return the logit :math:`F_T(x)`.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Rows on the feature scale used in ``fit``.

        Returns
        -------
        ndarray of shape (n_samples,)
            Log-odds of ``classes_[1]``.
        """

        self._check_fitted()
        assert self.n_features_in_ is not None
        return self._raw_predict(_validate_X(X, self.n_features_in_))

    def predict_proba(self, X: ArrayLike) -> NDArray[np.float64]:
        r"""Return class probabilities from the boosted logit.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Rows on the feature scale used in ``fit``.

        Returns
        -------
        ndarray of shape (n_samples, 2)
            Columns are :math:`\sigma(-F)` and :math:`\sigma(F)`.
        """

        logits = self.decision_function(X)
        positive = sigmoid(logits)
        negative = sigmoid(-logits)
        return np.column_stack((negative, positive))

    def predict(self, X: ArrayLike) -> NDArray[Any]:
        """Return the class whose probability meets ``threshold``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Rows on the feature scale used in ``fit``.

        Returns
        -------
        ndarray of shape (n_samples,)
            Labels from ``classes_``.
        """

        self._check_fitted()
        assert self.classes_ is not None
        positive = self.predict_proba(X)[:, 1] >= self.threshold
        predicted = np.empty(positive.shape[0], dtype=self.classes_.dtype)
        predicted[positive] = self.classes_[1]
        predicted[~positive] = self.classes_[0]
        return predicted

    def score(self, X: ArrayLike, y: ArrayLike) -> float:
        """Return the classification accuracy of ``predict(X)``."""

        return accuracy_score(y, self.predict(X))

    def _derivatives(
        self,
        targets: NDArray[np.float64],
        raw: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        r"""Return :math:`g = \sigma(F) - y` and :math:`h = \sigma(F)(1-\sigma(F))`."""

        probability = sigmoid(raw)
        gradient = probability - targets
        hessian = np.maximum(probability * (1.0 - probability), 0.0)
        return gradient, hessian

    def _risk(self, targets: NDArray[np.float64], raw: NDArray[np.float64]) -> float:
        """Return the mean Bernoulli negative log-likelihood."""

        return BinaryCrossEntropy.loss_with_logits(raw, targets)

    def _check_fitted(self) -> None:
        if self.base_score_ is None or self.classes_ is None:
            raise RuntimeError("GradientBoostingClassifier must be fitted before use.")

    def __repr__(self) -> str:
        return (
            f"GradientBoostingClassifier(n_estimators={self.n_estimators}, "
            f"learning_rate={self.learning_rate}, max_depth={self.max_depth})"
        )


def _gain_array(
    gradient_left: NDArray[np.float64],
    hessian_left: NDArray[np.float64],
    gradient_right: NDArray[np.float64],
    hessian_right: NDArray[np.float64],
    reg_lambda: float,
    gamma: float,
) -> NDArray[np.float64]:
    """Evaluate the split gain at many cuts. Shape ``(n_cuts,)``.

    The parent sums are the sums of the two children, so this is the gain
    of a partition. Division is suppressed when a denominator is not
    positive; those cuts receive a non-finite gain and lose the ``argmax``.
    """

    penalty = float(reg_lambda)
    split_cost = float(gamma)
    left_denom = hessian_left + penalty
    right_denom = hessian_right + penalty
    parent_denom = hessian_left + hessian_right + penalty
    left_term = np.divide(
        gradient_left * gradient_left,
        left_denom,
        out=np.zeros_like(gradient_left, dtype=np.float64),
        where=left_denom > 0.0,
    )
    right_term = np.divide(
        gradient_right * gradient_right,
        right_denom,
        out=np.zeros_like(gradient_right, dtype=np.float64),
        where=right_denom > 0.0,
    )
    parent_gradient = gradient_left + gradient_right
    parent_term = np.divide(
        parent_gradient * parent_gradient,
        parent_denom,
        out=np.zeros_like(parent_gradient, dtype=np.float64),
        where=parent_denom > 0.0,
    )
    gain = 0.5 * (left_term + right_term - parent_term) - split_cost
    legal = (left_denom > 0.0) & (right_denom > 0.0) & (parent_denom > 0.0)
    return np.where(legal, gain, -np.inf)


def _validate_regression_xy(
    X: ArrayLike,
    y: ArrayLike,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return a float64 design and a one-dimensional real target."""

    features, targets = _validate_xy(X, y)
    values = np.asarray(targets, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("y must be a one-dimensional target vector.")
    if not np.isfinite(values).all():
        raise ValueError("y must contain only finite values.")
    return features, values


def _validate_xy(
    X: ArrayLike,
    y: ArrayLike,
) -> tuple[NDArray[np.float64], NDArray[Any]]:
    """Return a float64 design and a one-dimensional target vector."""

    features = np.asarray(X, dtype=np.float64)
    targets = np.asarray(y)
    if features.ndim == 1:
        features = features.reshape(-1, 1)
    if features.ndim != 2:
        raise ValueError(
            "X must have shape (n_samples, n_features); "
            f"got ndim={features.ndim}."
        )
    if targets.ndim == 2 and targets.shape[1] == 1:
        targets = targets.reshape(-1)
    if targets.ndim != 1:
        raise ValueError("y must be a one-dimensional vector.")
    if features.shape[0] != targets.shape[0]:
        raise ValueError(
            "X and y must have the same number of rows. "
            f"Got {features.shape[0]} and {targets.shape[0]}."
        )
    if features.shape[0] < 1 or features.shape[1] < 1:
        raise ValueError("X must contain at least one row and one column.")
    if not np.isfinite(features).all():
        raise ValueError("X must contain only finite values.")
    if targets.dtype.kind == "f" and not np.isfinite(np.asarray(targets, dtype=np.float64)).all():
        raise ValueError("y must contain only finite values.")
    return features, targets


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


def _unit_fraction(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.floating)):
        raise TypeError(f"{name} must be a real number.")
    number = float(value)
    if not np.isfinite(number) or number <= 0.0 or number > 1.0:
        raise ValueError(f"{name} must lie in (0, 1].")
    return number
