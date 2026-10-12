"""Bernoulli MLE by gradient descent and by Newton-Raphson.

Rows of ``X`` are examples. The success probability is the sigmoid of a
linear score, and the training objective is the averaged negative
log-likelihood already implemented in
:class:`~ml_from_scratch.optimization.losses.BinaryCrossEntropy`.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..optimization.losses import BinaryCrossEntropy, sigmoid
from ..optimization.optimizers import Adam, SGD
from ..utils.data_manipulation import add_intercept
from ..utils.metrics import accuracy_score

MethodName = Literal["gradient_descent", "newton"]
OptimizerName = Literal["sgd", "adam"]

# Armijo constant for the Newton line search. The full Newton step is
# accepted whenever it satisfies this sufficient-decrease test.
_ARMIJO = 1e-4
_MAX_BACKTRACKS = 40


class LogisticRegression:
    r"""Binary logistic regression, the Bernoulli GLM with a logit link.

    For a label :math:`y \in \{0, 1\}` and a score :math:`z = x^\top\theta`,

    .. math::

        P(y = 1 \mid x) = \sigma(z) = \frac{1}{1 + e^{-z}}.

    The Bernoulli likelihood of one row is
    :math:`\sigma(z)^y (1 - \sigma(z))^{1 - y}`. The training objective is
    the average negative log-likelihood, optionally with the same ridge
    penalty used by :class:`~ml_from_scratch.linear_models.linear_regression.LinearRegression`:

    .. math::

        J(\theta)
        = -\frac{1}{m}\sum_{i=1}^{m}
          \left[
            y_i\log\sigma(z_i) + (1 - y_i)\log(1 - \sigma(z_i))
          \right]
          + \frac{\lambda}{2m}\lVert P\theta\rVert_2^2.

    :math:`P` is diagonal, 0 on the intercept and 1 on every slope. The
    intercept stays unpenalized so the baseline log-odds are not shrunk
    toward 0. The loss is evaluated in logit coordinates,
    :math:`\operatorname{softplus}(z) - y z`, which is the same :math:`J`
    and does not pass a saturated probability through a logarithm.

    Differentiating under the sigmoid cancellation
    :math:`\partial J / \partial z_i = (p_i - y_i) / m` gives the score

    .. math::

        \nabla J(\theta)
        = \frac{1}{m} X^\top(p - y) + \frac{\lambda}{m} P\theta,
        \qquad p = \sigma(X\theta).

    Setting the unpenalized score to zero is the likelihood equation
    :math:`X^\top(p - y) = 0`: every column of the design is orthogonal to
    the residual between the predicted probability and the label.

    A second differentiation uses
    :math:`\partial p_i / \partial z_i = p_i(1 - p_i)`, the Bernoulli
    variance. With :math:`S = \operatorname{diag}(p(1 - p))`,

    .. math::

        \nabla^2 J(\theta)
        = \frac{1}{m} X^\top S X + \frac{\lambda}{m} P.

    :math:`X^\top S X` is the Hessian of the summed negative log-likelihood.
    The factor :math:`1/m` is present because :math:`J` averages. It
    cancels in the Newton step. Writing :math:`g = \nabla J` and
    :math:`H = \nabla^2 J`,

    .. math::

        H^{-1} g
        = (X^\top S X + \lambda P)^{-1}
          \left(X^\top(p - y) + \lambda P\theta\right).

    **Newton-Raphson.** ``method="newton"`` repeats
    :math:`\theta \leftarrow \theta - \alpha H^{-1}\nabla J`. The search
    direction is that Newton direction. :math:`\alpha` starts at 1, which
    is the classical Newton-Raphson step, and is halved until the Armijo
    test accepts it. Convexity makes the Newton direction a descent
    direction whenever :math:`H` is positive definite. :math:`H` is positive
    semidefinite for :math:`\lambda = 0` and positive definite on the slope
    block for :math:`\lambda > 0`. Linearly separable data send the
    unpenalized MLE to infinity; a positive :math:`\lambda` keeps a finite
    maximizer.

    **Gradient descent.** ``method="gradient_descent"`` feeds the same
    :math:`\nabla J` to heavy-ball SGD or Adam. A mini-batch re-averages
    only the data term. The penalty gradient :math:`(\lambda / m) P\theta`
    uses the full training size on every batch, so each step estimates
    :math:`\nabla J`. Newton always uses the full sample: the Hessian of a
    batch is not the Hessian of :math:`J`.

    Parameters
    ----------
    method : {"newton", "gradient_descent"}, default="newton"
        Newton-Raphson on :math:`H`, or first-order minimization of
        :math:`J`.
    learning_rate : float, default=0.1
        Step size for gradient descent. Unused by Newton.
    max_iter : int, default=1000
        Maximum Newton steps, or maximum passes over the sample for
        gradient descent. Newton usually stops much earlier.
    tol : float, default=1e-6
        Newton stops when :math:`\lVert\nabla J\rVert_2` falls to this
        value, and also when the accepted step is shorter than this value.
        Gradient descent stops when the absolute change in :math:`J` over
        an epoch falls below it.
    fit_intercept : bool, default=True
        Estimate an unpenalized intercept.
    l2 : float, default=0.0
        Ridge coefficient :math:`\lambda` in :math:`J`.
    optimizer : {"sgd", "adam"}, default="sgd"
        Recurrence used by gradient descent. Weight decay on that
        recurrence stays 0; ridge is applied inside :math:`\nabla J`.
    momentum : float, default=0.0
        Heavy-ball coefficient for ``optimizer="sgd"``.
    batch_size : int or None, default=None
        Rows per gradient estimate. ``None`` uses the whole sample.
        Ignored by Newton.
    threshold : float, default=0.5
        ``predict`` returns the positive class when
        :math:`P(y = 1 \mid x)` is at least this value. The boundary
        belongs to the positive class. ``0.5`` is the maximum a posteriori
        rule when the two classes have equal misclassification cost.
    random_state : int or None, default=None
        Seed for mini-batch order. Each ``fit`` starts a new generator.

    Attributes
    ----------
    coef_ : ndarray of shape (n_features,)
        Slope block of ``theta_``. This is a view.
    intercept_ : float
        ``theta_[0]`` when an intercept is fit, otherwise 0.
    theta_ : ndarray of shape (n_parameters,)
        Parameter vector. Index 0 is the intercept when
        ``fit_intercept`` is true.
    classes_ : ndarray of shape (2,)
        ``classes_[0]`` is the negative class and ``classes_[1]`` is the
        positive class, in sorted label order. ``predict_proba`` column
        ``k`` is the probability of ``classes_[k]``.
    n_features_in_ : int
        Column count of ``X``, excluding the intercept.
    n_samples_ : int
        Rows used in the last ``fit``.
    n_iter_ : int
        Parameter updates actually applied.
    loss_history_ : list of float
        :math:`J` at initialization, then :math:`J` after each accepted
        update. Newton records one value per accepted step. Gradient
        descent records one value per epoch.
    grad_norm_ : float
        :math:`\lVert\nabla J\rVert_2` at the returned parameter.
    step_size_history_ : list of float
        Accepted Newton step lengths. Empty after gradient descent.
        A value of 1 is a full Newton-Raphson step.
    solver_used_ : str
        ``"newton"`` or ``"gradient_descent"``.
    optimizer_ : SGD, Adam, or None
        Update object used by gradient descent.
    """

    def __init__(
        self,
        method: MethodName = "newton",
        learning_rate: float = 0.1,
        max_iter: int = 1000,
        tol: float = 1e-6,
        fit_intercept: bool = True,
        l2: float = 0.0,
        optimizer: OptimizerName = "sgd",
        momentum: float = 0.0,
        batch_size: int | None = None,
        threshold: float = 0.5,
        random_state: int | None = None,
    ) -> None:
        if method not in ("newton", "gradient_descent"):
            raise ValueError("method must be 'newton' or 'gradient_descent'.")
        if optimizer not in ("sgd", "adam"):
            raise ValueError("optimizer must be 'sgd' or 'adam'.")
        if not isinstance(fit_intercept, bool):
            raise TypeError("fit_intercept must be a bool.")
        if isinstance(max_iter, bool) or not isinstance(max_iter, int) or max_iter < 1:
            raise ValueError("max_iter must be an integer >= 1.")
        if batch_size is not None and (
            isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1
        ):
            raise ValueError("batch_size must be a positive integer or None.")
        if isinstance(random_state, bool) or not isinstance(random_state, (int, type(None))):
            raise TypeError("random_state must be an int or None.")
        if isinstance(threshold, bool) or not isinstance(threshold, (int, float, np.floating)):
            raise TypeError("threshold must be a real number.")
        if not 0.0 <= float(threshold) <= 1.0:
            raise ValueError("threshold must lie in [0, 1].")

        self.method: MethodName = method
        self.learning_rate = _positive_float("learning_rate", learning_rate)
        self.max_iter = max_iter
        self.tol = _nonnegative_float("tol", tol)
        self.fit_intercept = fit_intercept
        self.l2 = _nonnegative_float("l2", l2)
        self.optimizer: OptimizerName = optimizer
        self.momentum = _momentum(momentum)
        self.batch_size = batch_size
        self.threshold = float(threshold)
        self.random_state = random_state

        self.coef_: NDArray[np.float64] | None = None
        self.theta_: NDArray[np.float64] | None = None
        self.classes_: NDArray[Any] | None = None
        self.n_features_in_: int | None = None
        self.n_samples_: int | None = None
        self.n_iter_: int = 0
        self.loss_history_: list[float] = []
        self.grad_norm_: float | None = None
        self.step_size_history_: list[float] = []
        self.solver_used_: str | None = None
        self.optimizer_: SGD | Adam | None = None

    def fit(self, X: ArrayLike, y: ArrayLike) -> LogisticRegression:
        """Maximize the Bernoulli likelihood, optionally with ridge.

        ``X`` and ``y`` are read and not written. Labels must contain
        exactly two classes. The positive class is the larger one in
        sorted order.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training design. A one-dimensional array is one feature column.
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
                "LogisticRegression fits a Bernoulli likelihood and needs "
                f"exactly two classes. Observed {classes.size}: {classes.tolist()}."
            )
        # Map the sorted positive class to 1 and the other class to 0.
        # The likelihood is written for a Bernoulli trial, not for the
        # original label encoding.
        targets = (labels == classes[1]).astype(np.float64)
        self._reset_fit_state()
        self.classes_ = classes
        self.n_features_in_ = int(features.shape[1])
        self.n_samples_ = int(features.shape[0])
        design = add_intercept(features) if self.fit_intercept else np.array(
            features, dtype=np.float64, copy=True
        )
        if self.method == "newton":
            self._fit_newton(design, targets)
        else:
            self._fit_gradient_descent(design, targets)
        self._publish_coefficients()
        assert self.theta_ is not None
        self.grad_norm_ = float(
            np.linalg.norm(self._gradient(design, targets, self.theta_, targets.shape[0]))
        )
        return self

    def decision_function(self, X: ArrayLike) -> NDArray[np.float64]:
        r"""Return the logit :math:`\theta_0 + x^\top\theta_{1:}`.

        This is the natural parameter of the Bernoulli. Probabilities are
        ``sigmoid`` of these scores, not a second model.
        """

        self._check_fitted()
        features = self._validate_X(X)
        assert self.coef_ is not None and self.theta_ is not None
        bias = float(self.theta_[0]) if self.fit_intercept else 0.0
        return features @ self.coef_ + bias

    def predict_proba(self, X: ArrayLike) -> NDArray[np.float64]:
        r"""Return :math:`P(y = c \mid x)` for both classes.

        Column 0 is the negative class and column 1 is the positive class,
        matching ``classes_``. The two columns are
        :math:`\sigma(-z)` and :math:`\sigma(z)`. Evaluating both through
        the stable sigmoid keeps a saturated logit from requiring a
        subtraction ``1 - p`` on a probability that has already overflowed
        to 0 or 1.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Rows on the feature scale used in ``fit``.

        Returns
        -------
        ndarray of shape (n_samples, 2)
            Class probabilities. Each row sums to 1.
        """

        logits = self.decision_function(X)
        positive = sigmoid(logits)
        negative = sigmoid(-logits)
        return np.column_stack((negative, positive))

    def predict(self, X: ArrayLike) -> NDArray[Any]:
        """Return the class whose predicted probability meets ``threshold``.

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

    def _fit_newton(
        self,
        design: NDArray[np.float64],
        targets: NDArray[np.float64],
    ) -> None:
        """Run damped Newton-Raphson from the origin."""

        theta = np.zeros(design.shape[1], dtype=np.float64)
        self.solver_used_ = "newton"
        self.step_size_history_ = []
        self.loss_history_ = [
            self._require_finite_loss(self._objective(design, targets, theta))
        ]
        n_samples = int(targets.shape[0])

        for _epoch in range(self.max_iter):
            gradient = self._gradient(design, targets, theta, n_samples)
            self._require_finite_gradient(gradient)
            if float(np.linalg.norm(gradient)) <= self.tol:
                break
            hessian = self._hessian(design, targets, theta, n_samples)
            step = self._newton_step(hessian, gradient)
            slope = float(gradient @ step)
            if slope <= 0.0:
                raise FloatingPointError(
                    "The Newton system did not produce a descent direction. "
                    "The Hessian is singular when the data are separated or "
                    "when a column of X repeats the intercept. Increase l2, "
                    "or remove the collinear column."
                )
            alpha = self._backtrack(design, targets, theta, step, slope, self.loss_history_[-1])
            theta -= alpha * step
            loss = self._require_finite_loss(self._objective(design, targets, theta))
            self.loss_history_.append(loss)
            self.step_size_history_.append(alpha)
            self.n_iter_ += 1
            if float(np.linalg.norm(alpha * step)) <= self.tol:
                break

        self.theta_ = theta

    def _fit_gradient_descent(
        self,
        design: NDArray[np.float64],
        targets: NDArray[np.float64],
    ) -> None:
        """Minimize ``J`` from the origin with SGD or Adam."""

        theta = np.zeros(design.shape[1], dtype=np.float64)
        optimizer = self._make_optimizer()
        self.optimizer_ = optimizer
        self.solver_used_ = "gradient_descent"
        self.step_size_history_ = []
        self.loss_history_ = [
            self._require_finite_loss(self._objective(design, targets, theta))
        ]
        n_samples = int(targets.shape[0])
        batch_size = n_samples if self.batch_size is None else min(self.batch_size, n_samples)
        rng = np.random.default_rng(self.random_state)

        for epoch in range(self.max_iter):
            if batch_size >= n_samples:
                gradient = self._gradient(design, targets, theta, n_samples)
                self._require_finite_gradient(gradient)
                optimizer.step(theta, gradient)
            else:
                # Steps are sequential because each batch reads the θ
                # produced by the previous batch. One batch is one GEMM.
                permutation = rng.permutation(n_samples)
                for start in range(0, n_samples, batch_size):
                    batch = permutation[start : start + batch_size]
                    gradient = self._gradient(
                        design[batch],
                        targets[batch],
                        theta,
                        n_samples,
                    )
                    self._require_finite_gradient(gradient)
                    optimizer.step(theta, gradient)
            loss = self._require_finite_loss(self._objective(design, targets, theta))
            self.loss_history_.append(loss)
            self.n_iter_ = epoch + 1
            if abs(self.loss_history_[-1] - self.loss_history_[-2]) < self.tol:
                break

        self.theta_ = theta

    def _gradient(
        self,
        design_batch: NDArray[np.float64],
        targets_batch: NDArray[np.float64],
        theta: NDArray[np.float64],
        n_samples: int,
    ) -> NDArray[np.float64]:
        """Return ∇J, or an unbiased mini-batch estimate of its data term.

        ``gradient_with_logits`` averages over the rows it is given, so on
        a batch of size ``b`` it returns ``(p - y) / b``. The product
        ``X_batch.T @`` that vector is the chain rule through ``z = X θ``:
        column ``j`` of ``X`` is ``∂z / ∂θ_j``.

        The ridge term is ``(λ / (2m)) ||P θ||^2`` with ``m`` the full
        training size. Its gradient does not depend on which rows were
        sampled, so it is added in full inside every batch.
        """

        logits = design_batch @ theta
        data_gradient = design_batch.T @ BinaryCrossEntropy.gradient_with_logits(
            logits,
            targets_batch,
        )
        if self.l2 == 0.0:
            return data_gradient
        if self.fit_intercept:
            penalty = np.zeros_like(theta)
            penalty[1:] = (self.l2 / n_samples) * theta[1:]
            return data_gradient + penalty
        return data_gradient + (self.l2 / n_samples) * theta

    def _hessian(
        self,
        design: NDArray[np.float64],
        targets: NDArray[np.float64],
        theta: NDArray[np.float64],
        n_samples: int,
    ) -> NDArray[np.float64]:
        """Return the Hessian of the averaged objective ``J``.

        ``targets`` is unused by the curvature: for a Bernoulli model in
        logit coordinates the second derivative depends on the predicted
        variance :math:`p(1-p)`, not on the observed label. The label
        enters only the gradient. The argument is kept so the Hessian and
        the gradient are called with the same training pair.
        """

        del targets
        logits = design @ theta
        probabilities = sigmoid(logits)
        # S_ii = p_i (1 - p_i). Scaling row i of X by sqrt(S_ii) builds
        # S^{1/2} X, and the Gram matrix of that scaled design is X^T S X
        # without an m×m diagonal matrix.
        variance = np.maximum(probabilities * (1.0 - probabilities), 0.0)
        scaled = design * np.sqrt(variance)[:, None]
        curvature = scaled.T @ scaled
        hessian = curvature / n_samples
        if self.l2 == 0.0:
            return hessian
        mask = np.ones(hessian.shape[0], dtype=np.float64)
        if self.fit_intercept:
            mask[0] = 0.0
        # Derivative of (λ/m) P θ is (λ/m) P. P is the same mask as in ∇J.
        return hessian + (self.l2 / n_samples) * np.diag(mask)

    def _objective(
        self,
        design: NDArray[np.float64],
        targets: NDArray[np.float64],
        theta: NDArray[np.float64],
    ) -> float:
        """Return the regularized Bernoulli empirical risk ``J(θ)``."""

        data_loss = BinaryCrossEntropy.loss_with_logits(design @ theta, targets)
        if self.l2 == 0.0:
            return data_loss
        slopes = theta[1:] if self.fit_intercept else theta
        penalty = (self.l2 / (2.0 * targets.shape[0])) * float(slopes @ slopes)
        return data_loss + penalty

    def _newton_step(
        self,
        hessian: NDArray[np.float64],
        gradient: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Solve :math:`H s = \\nabla J`.

        ``solve`` uses an LU factorization and does not form :math:`H^{-1}`.
        A singular Hessian, which is the separable and collinear case, falls
        back to the minimum-norm solution of the same linear system.
        """

        try:
            return np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            return np.linalg.lstsq(hessian, gradient, rcond=None)[0]

    def _backtrack(
        self,
        design: NDArray[np.float64],
        targets: NDArray[np.float64],
        theta: NDArray[np.float64],
        step: NDArray[np.float64],
        slope: float,
        current_loss: float,
    ) -> float:
        """Choose α so that θ - α s decreases ``J``.

        The directional derivative of :math:`α \\mapsto J(θ - α s)` at
        :math:`α = 0` is :math:`-\\nabla J^\\top s`. The test

        .. math::

            J(\\theta - \\alpha s)
            \\le J(\\theta) - c \\alpha \\nabla J^\\top s

        with :math:`c = 10^{-4}` is the Armijo condition. :math:`\\alpha = 1`
        is the Newton-Raphson step and is the first candidate.
        """

        alpha = 1.0
        for _attempt in range(_MAX_BACKTRACKS):
            candidate = theta - alpha * step
            candidate_loss = self._objective(design, targets, candidate)
            decrease = _ARMIJO * alpha * slope
            if np.isfinite(candidate_loss) and candidate_loss <= current_loss - decrease:
                return alpha
            alpha *= 0.5
        raise FloatingPointError(
            "Newton backtracking did not find a step that decreases J. "
            "Standardize the columns of X or increase l2. Separable data "
            "make the unpenalized likelihood unbounded."
        )

    def _make_optimizer(self) -> SGD | Adam:
        """Build an optimizer that does not add a second ridge term."""

        if self.optimizer == "sgd":
            return SGD(learning_rate=self.learning_rate, momentum=self.momentum)
        return Adam(learning_rate=self.learning_rate)

    @property
    def intercept_(self) -> float:
        """Return the bias stored in ``theta_``."""

        if self.theta_ is None:
            raise RuntimeError("LogisticRegression must be fitted before use.")
        if self.fit_intercept:
            return float(self.theta_[0])
        return 0.0

    def _publish_coefficients(self) -> None:
        """Point ``coef_`` at the slope block of ``theta_``."""

        assert self.theta_ is not None
        if self.fit_intercept:
            self.coef_ = self.theta_[1:]
        else:
            self.coef_ = self.theta_

    def _reset_fit_state(self) -> None:
        """Drop the previous fit so a second call cannot reuse its classes."""

        self.coef_ = None
        self.theta_ = None
        self.classes_ = None
        self.n_iter_ = 0
        self.loss_history_ = []
        self.grad_norm_ = None
        self.step_size_history_ = []
        self.solver_used_ = None
        self.optimizer_ = None

    def _check_fitted(self) -> None:
        if self.coef_ is None or self.theta_ is None or self.classes_ is None:
            raise RuntimeError("LogisticRegression must be fitted before use.")

    def _validate_X(self, X: ArrayLike) -> NDArray[np.float64]:
        features = np.asarray(X, dtype=np.float64)
        if features.ndim == 1:
            features = features.reshape(-1, 1)
        if features.ndim != 2:
            raise ValueError(
                "X must have shape (n_samples, n_features); "
                f"got ndim={features.ndim}."
            )
        if features.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Expected {self.n_features_in_} features, got {features.shape[1]}."
            )
        if not np.isfinite(features).all():
            raise ValueError("X must contain only finite values.")
        return features

    def _require_finite_loss(self, loss: float) -> float:
        if not np.isfinite(loss):
            raise FloatingPointError(
                "The Bernoulli objective became non-finite. Standardize the "
                "columns of X, reduce learning_rate, or increase l2."
            )
        return float(loss)

    def _require_finite_gradient(self, gradient: NDArray[np.float64]) -> None:
        if not np.isfinite(gradient).all():
            raise FloatingPointError(
                "A gradient entry became non-finite. Standardize the columns "
                "of X, reduce learning_rate, or increase l2."
            )

    def __repr__(self) -> str:
        return (
            f"LogisticRegression(method={self.method!r}, fit_intercept={self.fit_intercept}, "
            f"l2={self.l2})"
        )


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


def _momentum(value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.floating)):
        raise TypeError("momentum must be a real number.")
    number = float(value)
    if not np.isfinite(number) or number < 0.0 or number >= 1.0:
        raise ValueError("momentum must lie in [0, 1).")
    return number
