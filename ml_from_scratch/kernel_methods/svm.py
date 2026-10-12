"""Binary support-vector machine in the dual, trained by SMO.

The primal separates points with a maximum margin in a feature space.
The dual and a kernel never build that feature map. SMO optimizes two
Lagrange multipliers at a time under the box constraints and the linear
equality that comes from the bias.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

KernelName = Literal["linear", "poly", "polynomial", "rbf"]


def linear_kernel(X: ArrayLike, Y: ArrayLike) -> NDArray[np.float64]:
    r"""Return the Gram matrix :math:`K_{ij} = x_i^\top y_j`.

    Parameters
    ----------
    X : array-like of shape (n, d)
        Left rows.
    Y : array-like of shape (m, d)
        Right rows.

    Returns
    -------
    ndarray of shape (n, m)
        Inner products.
    """

    left, right = _as_design(X), _as_design(Y)
    return left @ right.T


def polynomial_kernel(
    X: ArrayLike,
    Y: ArrayLike,
    gamma: float,
    degree: int,
    coef0: float,
) -> NDArray[np.float64]:
    r"""Return :math:`K_{ij} = (\gamma x_i^\top y_j + r)^p`.

    Parameters
    ----------
    X, Y : array-like of shape (n, d) and (m, d)
        Rows whose inner products enter the kernel.
    gamma : float
        Scale :math:`\gamma` of the inner product.
    degree : int
        Polynomial degree :math:`p`.
    coef0 : float
        Offset :math:`r`.

    Returns
    -------
    ndarray of shape (n, m)
        Polynomial Gram matrix.
    """

    return (gamma * linear_kernel(X, Y) + coef0) ** degree


def rbf_kernel(X: ArrayLike, Y: ArrayLike, gamma: float) -> NDArray[np.float64]:
    r"""Return :math:`K_{ij} = \exp(-\gamma \|x_i - y_j\|^2)`.

    The squared distance is expanded as
    :math:`\|x\|^2 + \|y\|^2 - 2 x^\top y`, with shapes ``(n, 1)``,
    ``(1, m)``, and ``(n, m)``. A tiny negative value from roundoff is
    clipped to zero before the exponential.

    Parameters
    ----------
    X, Y : array-like of shape (n, d) and (m, d)
        Rows.
    gamma : float
        Inverse-width :math:`\gamma`.

    Returns
    -------
    ndarray of shape (n, m)
        RBF Gram matrix.
    """

    left, right = _as_design(X), _as_design(Y)
    left_sq = np.sum(left * left, axis=1)[:, None]
    right_sq = np.sum(right * right, axis=1)[None, :]
    distance = np.maximum(left_sq + right_sq - 2.0 * (left @ right.T), 0.0)
    return np.exp(-gamma * distance)


class SVM:
    r"""Soft-margin binary SVM trained by sequential minimal optimization.

    In a feature map :math:`\phi`, the primal soft-margin problem is

    .. math::

        \min_{w, b, \xi}\;
        \frac{1}{2}\|w\|^2 + C \sum_{i=1}^{n} \xi_i

    subject to :math:`y_i(w^\top\phi(x_i) + b) \ge 1 - \xi_i` and
    :math:`\xi_i \ge 0`, with labels :math:`y_i \in \{-1, +1\}`. The dual
    depends on :math:`\phi` only through :math:`K_{ij} = \phi(x_i)^\top\phi(x_j)`:

    .. math::

        \max_{\alpha}\;
        \sum_{i=1}^{n} \alpha_i
        - \frac{1}{2}
          \sum_{i=1}^{n}\sum_{j=1}^{n}
          \alpha_i \alpha_j y_i y_j K_{ij}

    subject to :math:`0 \le \alpha_i \le C` and
    :math:`\sum_i \alpha_i y_i = 0`. The decision function on a new point is

    .. math::

        f(x) = \sum_{i=1}^{n} \alpha_i y_i K(x_i, x) + b.

    Only rows with :math:`\alpha_i > 0` are support vectors. The
    Karush-Kuhn-Tucker conditions at a multiplier are

    .. math::

        \alpha_i = 0 &\Rightarrow y_i f(x_i) \ge 1, \\
        0 < \alpha_i < C &\Rightarrow y_i f(x_i) = 1, \\
        \alpha_i = C &\Rightarrow y_i f(x_i) \le 1.

    SMO keeps every multiplier feasible and optimizes a pair
    :math:`(\alpha_i, \alpha_j)` in closed form. The equality constraint
    forces :math:`y_i\Delta\alpha_i + y_j\Delta\alpha_j = 0`. With
    :math:`\eta = K_{ii} + K_{jj} - 2 K_{ij}` and errors
    :math:`E_t = f(x_t) - y_t`,

    .. math::

        \alpha_j \leftarrow
        \Pi_{[L, H]}\!\left(
            \alpha_j + y_j \frac{E_i - E_j}{\eta}
        \right),

    and :math:`\alpha_i` moves by the opposite labeled step. :math:`[L, H]`
    is the slice of the step that also respects both box constraints.
    :math:`\eta` is :math:`\|\phi(x_i) - \phi(x_j)\|^2`, hence nonnegative
    for a valid kernel.

    The linear, polynomial, and RBF kernels are

    .. math::

        x^\top z,
        \qquad
        (\gamma x^\top z + r)^p,
        \qquad
        \exp(-\gamma \|x - z\|^2).

    ``gamma="scale"`` sets :math:`\gamma = 1 / (d \,\mathrm{Var}(X))` and
    ``gamma="auto"`` sets :math:`\gamma = 1/d`. There is no
    ``predict_proba``: the dual produces a signed margin, not a calibrated
    class probability.

    Parameters
    ----------
    C : float, default=1.0
        Penalty on margin violations. Larger :math:`C` fits the training
        labels more tightly.
    kernel : {"rbf", "linear", "poly"}, default="rbf"
        Kernel :math:`K`. ``"polynomial"`` is accepted as an alias of
        ``"poly"``.
    gamma : {"scale", "auto"} or float, default="scale"
        Kernel width used by ``"rbf"`` and ``"poly"``.
    degree : int, default=3
        Degree :math:`p` of the polynomial kernel.
    coef0 : float, default=1.0
        Offset :math:`r` of the polynomial kernel.
    tol : float, default=1e-3
        KKT slack. A multiplier is treated as stationary when
        :math:`y_i f(x_i)` is within ``tol`` of the condition its box
        bound requires.
    max_iter : int, default=500
        Maximum passes over KKT violators.

    Attributes
    ----------
    support_ : ndarray of shape (n_support,)
        Training indices with :math:`\alpha_i > 0`.
    support_vectors_ : ndarray of shape (n_support, n_features)
        Those rows.
    dual_coef_ : ndarray of shape (n_support,)
        :math:`\alpha_i y_i` on the support.
    alpha_ : ndarray of shape (n_samples,)
        Every dual multiplier, in training order.
    intercept_ : float
        Bias :math:`b`.
    classes_ : ndarray of shape (2,)
        ``classes_[0]`` is the negative label and ``classes_[1]`` is the
        positive label, in sorted order. :math:`f(x) \ge 0` predicts
        ``classes_[1]``.
    n_iter_ : int
        SMO passes actually run.
    converged_ : bool
        Whether the last pass found no KKT violator.
    n_features_in_ : int
        Columns of ``X``.
    gamma_ : float or None
        Resolved :math:`\gamma`. ``None`` for the linear kernel.
    """

    def __init__(
        self,
        C: float = 1.0,
        kernel: KernelName = "rbf",
        gamma: float | Literal["scale", "auto"] = "scale",
        degree: int = 3,
        coef0: float = 1.0,
        tol: float = 1e-3,
        max_iter: int = 500,
    ) -> None:
        if kernel not in ("linear", "poly", "polynomial", "rbf"):
            raise ValueError("kernel must be 'linear', 'poly', or 'rbf'.")
        if isinstance(degree, bool) or not isinstance(degree, int) or degree < 1:
            raise ValueError("degree must be an integer >= 1.")
        if isinstance(max_iter, bool) or not isinstance(max_iter, int) or max_iter < 1:
            raise ValueError("max_iter must be an integer >= 1.")
        self.C = _positive_float("C", C)
        self.kernel: KernelName = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = _real_float("coef0", coef0)
        self.tol = _positive_float("tol", tol)
        self.max_iter = max_iter
        self.support_: NDArray[np.int64] | None = None
        self.support_vectors_: NDArray[np.float64] | None = None
        self.dual_coef_: NDArray[np.float64] | None = None
        self.alpha_: NDArray[np.float64] | None = None
        self.intercept_: float = 0.0
        self.classes_: NDArray[Any] | None = None
        self.n_iter_: int = 0
        self.converged_: bool = False
        self.n_features_in_: int | None = None
        self.gamma_: float | None = None
        self._signed_labels: NDArray[np.float64] | None = None

    def fit(self, X: ArrayLike, y: ArrayLike) -> SVM:
        """Solve the dual by SMO.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training rows. A one-dimensional array is one feature column.
        y : array-like of shape (n_samples,)
            Binary labels. Exactly two distinct values are required.

        Returns
        -------
        self
        """

        features, labels = _validate_xy(X, y)
        classes = np.unique(labels)
        if classes.size != 2:
            raise ValueError(
                "SVM fits a binary margin and needs exactly two classes. "
                f"Observed {classes.size}: {classes.tolist()}."
            )
        signed = np.where(labels == classes[1], 1.0, -1.0)
        self.classes_ = classes
        self.n_features_in_ = int(features.shape[1])
        self._signed_labels = signed
        self.gamma_ = None if self.kernel == "linear" else _resolve_gamma(self.gamma, features)
        gram = self._gram(features, features)
        alpha, bias, n_iter, converged = _smo(
            gram,
            signed,
            C=self.C,
            tol=self.tol,
            max_iter=self.max_iter,
        )
        self.alpha_ = alpha
        self.intercept_ = bias
        self.n_iter_ = n_iter
        self.converged_ = converged
        support = np.flatnonzero(alpha > 1e-8)
        if support.size == 0:
            # Every multiplier stayed at 0. Keep a single row so prediction
            # still has a defined kernel expansion; its coefficient is zero
            # and the bias carries the decision.
            support = np.array([0], dtype=np.int64)
        self.support_ = support.astype(np.int64, copy=False)
        self.support_vectors_ = features[self.support_].copy()
        self.dual_coef_ = alpha[self.support_] * signed[self.support_]
        return self

    def decision_function(self, X: ArrayLike) -> NDArray[np.float64]:
        r"""Return the signed margin :math:`f(x)`.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Rows on the feature scale used in ``fit``.

        Returns
        -------
        ndarray of shape (n_samples,)
            Positive values vote for ``classes_[1]``.
        """

        self._check_fitted()
        assert (
            self.support_vectors_ is not None
            and self.dual_coef_ is not None
            and self.n_features_in_ is not None
        )
        features = _validate_X(X)
        if features.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Expected {self.n_features_in_} features, got {features.shape[1]}."
            )
        # (n_test, n_support) @ (n_support,) is the kernel expansion of w^T φ(x).
        return self._gram(features, self.support_vectors_) @ self.dual_coef_ + self.intercept_

    def predict(self, X: ArrayLike) -> NDArray[Any]:
        """Return the label on the positive side of the margin.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Rows on the feature scale used in ``fit``.

        Returns
        -------
        ndarray of shape (n_samples,)
            Labels from ``classes_``. A margin of zero predicts the
            positive class.
        """

        self._check_fitted()
        assert self.classes_ is not None
        positive = self.decision_function(X) >= 0.0
        predicted = np.empty(positive.shape[0], dtype=self.classes_.dtype)
        predicted[positive] = self.classes_[1]
        predicted[~positive] = self.classes_[0]
        return predicted

    def score(self, X: ArrayLike, y: ArrayLike) -> float:
        """Return the fraction of rows whose predicted label matches ``y``."""

        observed = np.asarray(y).reshape(-1)
        predicted = self.predict(X)
        if observed.shape[0] != predicted.shape[0]:
            raise ValueError(
                "X and y must have the same number of rows. "
                f"Got {predicted.shape[0]} and {observed.shape[0]}."
            )
        return float(np.mean(observed == predicted))

    def _gram(self, X: NDArray[np.float64], Y: NDArray[np.float64]) -> NDArray[np.float64]:
        """Evaluate the configured kernel on two designs."""

        if self.kernel == "linear":
            return linear_kernel(X, Y)
        assert self.gamma_ is not None
        if self.kernel == "rbf":
            return rbf_kernel(X, Y, self.gamma_)
        return polynomial_kernel(X, Y, self.gamma_, self.degree, self.coef0)

    def _check_fitted(self) -> None:
        if self.support_vectors_ is None or self.classes_ is None:
            raise RuntimeError("SVM must be fitted before use.")

    def __repr__(self) -> str:
        return f"SVM(C={self.C}, kernel={self.kernel!r}, gamma={self.gamma!r})"


def _smo(
    gram: NDArray[np.float64],
    signed: NDArray[np.float64],
    C: float,
    tol: float,
    max_iter: int,
) -> tuple[NDArray[np.float64], float, int, bool]:
    r"""Maximize the dual over pairs of multipliers.

    The Gram matrix has shape ``(n, n)``. Each accepted pair is an exact
    maximization on the line :math:`y_i\Delta\alpha_i + y_j\Delta\alpha_j = 0`
    inside the box :math:`[0, C]^2`. The pass itself is sequential because
    the next pair is chosen from the errors of the updated expansion.
    """

    n_samples = int(signed.shape[0])
    alpha = np.zeros(n_samples, dtype=np.float64)
    bias = 0.0
    decision = np.zeros(n_samples, dtype=np.float64)
    converged = False
    n_iter = 0
    for n_iter in range(1, max_iter + 1):
        decision = gram @ (alpha * signed) + bias
        violators = _kkt_violators(alpha, decision, signed, C, tol)
        if violators.size == 0:
            converged = True
            break
        errors = decision - signed
        changed = False
        order = _unbound_first(violators, alpha, C)
        for index in order:
            if _optimize_pair(int(index), gram, signed, alpha, errors, C):
                changed = True
                decision = gram @ (alpha * signed) + bias
                errors = decision - signed
                bias = _bias_from_margin(gram, signed, alpha, C, bias)
                decision = gram @ (alpha * signed) + bias
                errors = decision - signed
        if not changed:
            break
    return alpha, bias, n_iter, converged


def _optimize_pair(
    first: int,
    gram: NDArray[np.float64],
    signed: NDArray[np.float64],
    alpha: NDArray[np.float64],
    errors: NDArray[np.float64],
    C: float,
) -> bool:
    """Try the partner with the largest error gap, then a few more."""

    gaps = np.abs(errors - errors[first])
    gaps[first] = -1.0
    # Largest error gap first: that pair gives the longest Newton step when
    # the box allows it. Bound multipliers often clip the step to zero, so
    # the scan continues until some partner actually moves.
    partners = np.argsort(-gaps)
    for partner in partners:
        if int(partner) == first:
            continue
        if _update_pair(first, int(partner), gram, signed, alpha, errors, C):
            return True
    return False


def _update_pair(
    first: int,
    second: int,
    gram: NDArray[np.float64],
    signed: NDArray[np.float64],
    alpha: NDArray[np.float64],
    errors: NDArray[np.float64],
    C: float,
) -> bool:
    """Apply the analytic two-multiplier step, in place on ``alpha``."""

    y1 = float(signed[first])
    y2 = float(signed[second])
    a1 = float(alpha[first])
    a2 = float(alpha[second])
    if y1 == y2:
        low = max(0.0, a1 + a2 - C)
        high = min(C, a1 + a2)
    else:
        low = max(0.0, a2 - a1)
        high = min(C, C + a2 - a1)
    if high - low < 1e-12:
        return False
    eta = float(gram[first, first] + gram[second, second] - 2.0 * gram[first, second])
    if eta <= 1e-12:
        return False
    updated = a2 + y2 * (float(errors[first]) - float(errors[second])) / eta
    if updated > high:
        updated = high
    elif updated < low:
        updated = low
    if abs(updated - a2) < 1e-8:
        return False
    # y1 Δα1 + y2 Δα2 = 0, so the first multiplier takes the opposite step.
    alpha[second] = updated
    alpha[first] = a1 + y1 * y2 * (a2 - updated)
    alpha[first] = min(max(float(alpha[first]), 0.0), C)
    return True


def _bias_from_margin(
    gram: NDArray[np.float64],
    signed: NDArray[np.float64],
    alpha: NDArray[np.float64],
    C: float,
    fallback: float,
) -> float:
    r"""Set :math:`b` from the KKT equality on free support vectors.

    A multiplier strictly inside :math:`(0, C)` lies on the margin, so
    :math:`y_i f(x_i) = 1`. With the kernel expansion written without the
    bias, that equality is :math:`b = y_i - \sum_j \alpha_j y_j K_{ij}`.
    The mean over every free support vector is the value used here. If
    every multiplier is at a bound, the bias is not pinned to a point and
    the previous value is kept.
    """

    free = (alpha > 1e-8) & (alpha < C - 1e-8)
    if not np.any(free):
        return fallback
    expansion = gram[free] @ (alpha * signed)
    return float(np.mean(signed[free] - expansion))


def _kkt_violators(
    alpha: NDArray[np.float64],
    decision: NDArray[np.float64],
    signed: NDArray[np.float64],
    C: float,
    tol: float,
) -> NDArray[np.int64]:
    r"""Return indices whose box-constrained KKT condition fails.

    :math:`y_i f(x_i) < 1 - \mathrm{tol}` while :math:`\alpha_i < C` means
    the multiplier can still grow. :math:`y_i f(x_i) > 1 + \mathrm{tol}`
    while :math:`\alpha_i > 0` means it can still shrink.
    """

    functional = signed * decision
    can_grow = (alpha < C - 1e-8) & (functional < 1.0 - tol)
    can_shrink = (alpha > 1e-8) & (functional > 1.0 + tol)
    return np.flatnonzero(can_grow | can_shrink)


def _unbound_first(
    violators: NDArray[np.int64],
    alpha: NDArray[np.float64],
    C: float,
) -> NDArray[np.int64]:
    """List violators with free multipliers ahead of bound ones."""

    values = alpha[violators]
    free = (values > 1e-8) & (values < C - 1e-8)
    return np.concatenate([violators[free], violators[~free]])


def _resolve_gamma(
    gamma: float | Literal["scale", "auto"],
    features: NDArray[np.float64],
) -> float:
    """Turn the ``gamma`` argument into a positive float."""

    if gamma == "auto":
        return 1.0 / features.shape[1]
    if gamma == "scale":
        variance = float(np.var(features))
        if variance <= 0.0:
            return 1.0
        return 1.0 / (features.shape[1] * variance)
    return _positive_float("gamma", gamma)


def _as_design(X: ArrayLike) -> NDArray[np.float64]:
    """Return a two-dimensional float64 array."""

    features = np.asarray(X, dtype=np.float64)
    if features.ndim != 2:
        raise ValueError("Kernel inputs must have shape (n_samples, n_features).")
    return features


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
    """Return a design and a one-dimensional label vector of equal length."""

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


def _real_float(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.floating)):
        raise TypeError(f"{name} must be a real number.")
    number = float(value)
    if not np.isfinite(number):
        raise ValueError(f"{name} must be finite.")
    return number
