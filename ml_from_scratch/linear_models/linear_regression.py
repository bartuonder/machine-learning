"""Ordinary least squares, by the normal equation or by gradient descent.

Rows of ``X`` are examples and columns are features. In that layout the
empirical risk :math:`J(\\theta)` has gradient
:math:`(1/m) X^\\top (X\\theta - y)`. Some lecture notes store each
example as a column; the products below assume the opposite, row-wise
layout, which is the one used in the CS229 programming exercises.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..optimization.losses import MeanSquaredError
from ..optimization.optimizers import Adam, SGD
from ..utils.data_manipulation import add_intercept
from ..utils.metrics import r2_score

MethodName = Literal["normal_equation", "gradient_descent"]
OptimizerName = Literal["sgd", "adam"]

# Beyond this condition number the explicit solve of the normal equations
# is a poor floating-point procedure: forming X^T X has already squared
# the condition number of X. The pseudoinverse path is still the normal
# equation, with the inverse replaced by the Moore-Penrose inverse.
_COND_THRESHOLD = 1e12


class LinearRegression:
    r"""Linear conditional mean fit by least squares.

    The model is :math:`\mathbb{E}[y \mid x] = \theta_0 + x^\top\theta_{1:}`
    when ``fit_intercept`` is true, and :math:`x^\top\theta` otherwise.
    Stacking a leading column of ones into the design matrix :math:`X`
    writes either predictor as :math:`X\theta`.

    **Likelihood.** Homoscedastic Gaussian noise,
    :math:`y \mid x \sim \mathcal{N}(x^\top\theta, \sigma^2)`, has average
    negative log-likelihood equal to a positive multiple of the residual
    sum of squares plus a constant. The minimizer over :math:`\theta` is
    therefore the Gaussian maximum-likelihood estimator. The profile
    maximum-likelihood estimate of the noise variance, stored as
    ``sigma2_``, is :math:`\mathrm{RSS}/n`. The unbiased estimator divides
    by :math:`n - \mathrm{rank}(X)` instead; that quantity is not stored.

    **Gauss-Markov.** Gaussianity is not required for a different
    optimality claim. If the conditional mean really is linear, the errors
    have mean zero and are uncorrelated with common variance, then this
    estimator is best linear unbiased: no other estimator that is linear
    in :math:`y` and unbiased for :math:`\theta` has a smaller variance.

    **Objective.** With sample size :math:`m` and penalty :math:`\lambda`,

    .. math::

        J(\theta)
        = \frac{1}{2m}\lVert X\theta - y\rVert_2^2
          + \frac{\lambda}{2m}\lVert P\theta\rVert_2^2.

    :math:`P` is the diagonal matrix that is 0 on the intercept coordinate
    and 1 on every slope. The intercept is unpenalized so that a shift
    :math:`y \mapsto y + c` is absorbed entirely by :math:`\theta_0`.
    Penalizing :math:`\theta_0` would pin the fit to a particular origin
    of :math:`y`.

    The gradient and the normal equation obtained by setting it to zero are

    .. math::

        \nabla J(\theta)
        = \frac{1}{m} X^\top(X\theta - y) + \frac{\lambda}{m} P\theta,

    .. math::

        (X^\top X + \lambda P)\,\theta = X^\top y.

    The factor :math:`1/m` cancels when the gradient is set to zero, so
    :math:`\lambda` in :math:`J` and :math:`\lambda` added to the Gram
    matrix are the same number. Multiplying :math:`J` by :math:`2m` shows
    that this critical point is also the minimizer of the un-normalized
    problem :math:`\lVert X\theta - y\rVert_2^2 + \lambda\lVert P\theta\rVert_2^2`.

    **Normal equation.** ``method="normal_equation"`` forms the Gram
    system and solves it. The Gram matrix squares the condition number of
    :math:`X`, so a moderately ill-conditioned design becomes a badly
    ill-conditioned square system. When that system is singular or its
    condition number exceeds ``1e12``, the solver switches to the
    Moore-Penrose inverse of the same Gram matrix. In exact arithmetic
    that is the minimum-Euclidean-norm minimizer of :math:`J`. Forming
    :math:`X^\top X` is the step that loses digits; a QR or SVD of
    :math:`X` itself is the more stable algorithm for the same residual
    sum of squares. This class keeps the normal equation visible rather
    than hiding it behind a factorization of :math:`X`.

    **Gradient descent.** ``method="gradient_descent"`` starts at
    :math:`\theta = 0` and applies :class:`~ml_from_scratch.optimization.optimizers.SGD`
    or :class:`~ml_from_scratch.optimization.optimizers.Adam` to
    :math:`\nabla J`. Zero is the prior mode that the ridge term encodes.
    The least-squares objective is a convex quadratic. It is strictly
    convex, and the minimizer is unique, when the regularized Gram matrix
    is positive definite.

    For full-batch steps with momentum 0 the gradient is Lipschitz with
    constant :math:`L = \lambda_{\max}(X^\top X + \lambda P) / m`. Every
    learning rate in :math:`(0, 2/L)` converges. Columns of very different
    scales make :math:`L` large along some coordinates and small along
    others; standardizing columns before gradient descent keeps one
    learning rate meaningful. The normal equation does not need that
    rescaling to be correct, though it helps its conditioning.

    ``batch_size=None`` uses the full sample, which is gradient descent on
    :math:`J`. A smaller batch replaces the data term by the average over
    the batch, which is an unbiased estimate of the full data gradient
    when the batch is a uniform subset. The penalty term does not depend
    on a row index, so its gradient :math:`(\lambda/m) P\theta` is added
    in full on every batch. Each step estimates :math:`\nabla J`. An epoch
    of :math:`m/b` such steps is that many updates, not one full-batch update.

    ``predict_proba`` is not defined. The Gaussian model returns a
    conditional mean, and a conditional mean is not a class probability.

    Parameters
    ----------
    method : {"normal_equation", "gradient_descent"}, default="normal_equation"
        Closed form, or iterative minimization of ``J``.
    learning_rate : float, default=0.01
        Step size for gradient descent. Unused by the normal equation.
    max_iter : int, default=1000
        Maximum number of passes over the training sample.
    tol : float, default=1e-6
        Stop when the absolute change in the full-sample objective over
        one epoch falls below this value. On a mini-batch path the
        full-sample objective is still what is tested. ``tol=0`` runs for
        ``max_iter`` epochs unless two successive values are identical.
    fit_intercept : bool, default=True
        Estimate an unpenalized intercept.
    l2 : float, default=0.0
        Ridge coefficient :math:`\lambda` in the objective above.
    optimizer : {"sgd", "adam"}, default="sgd"
        Which recurrence consumes :math:`\nabla J`. Unused by the normal
        equation. ``"sgd"`` with ``batch_size=None`` is full-batch
        gradient descent; with a smaller batch it is mini-batch SGD.
        Adam's weight decay is left at 0. The ridge term is applied inside
        :math:`\nabla J` so the intercept can be excluded.
    momentum : float, default=0.0
        Heavy-ball coefficient for ``optimizer="sgd"``.
    batch_size : int or None, default=None
        Rows per gradient estimate. ``None`` uses the whole sample.
        The last batch of an epoch may be shorter.
    random_state : int or None, default=None
        Seed of the permutation that builds mini-batches. Each ``fit``
        starts a new generator from this seed.

    Attributes
    ----------
    coef_ : ndarray of shape (n_features,)
        Slope block of ``theta_``. This is a view, not a copy.
    intercept_ : float
        ``theta_[0]`` when an intercept is fit, and 0 otherwise. This is
        a read-only view of that coordinate, so predictions and ``theta_``
        cannot disagree about the bias.
    theta_ : ndarray of shape (n_parameters,)
        Parameter vector used by ``predict``. Index 0 is the intercept
        when ``fit_intercept`` is true.
    n_features_in_ : int
        Column count of ``X``, excluding the intercept column.
    n_samples_ : int
        Rows used in the last ``fit``.
    n_iter_ : int
        Epochs actually run. Zero for the normal equation.
    loss_history_ : list of float
        For gradient descent, ``J`` at initialization followed by ``J``
        after each epoch. For the normal equation, a one-element list
        containing ``J`` at the solution.
    sigma2_ : float
        Gaussian maximum-likelihood noise variance, residual sum of
        squares divided by ``n_samples_``.
    solver_used_ : str
        ``"normal_equation"``, ``"normal_equation_pseudoinverse"``, or
        ``"gradient_descent"``.
    rank_ : int or None
        Numerical rank of the regularized Gram matrix. ``None`` after
        gradient descent, which never forms that matrix.
    cond_ : float or None
        Condition number of the regularized Gram matrix, or infinity if
        it is singular.
    singular_ : bool or None
        Whether that rank is below the number of parameters.
    ill_conditioned_ : bool or None
        Whether ``cond_`` exceeds ``1e12``.
    gram_singular_values_ : ndarray or None
        Singular values of the regularized Gram matrix, largest first.
        For this positive semidefinite matrix they are also its
        eigenvalues.
    optimizer_ : SGD, Adam, or None
        The update object used by gradient descent. Its ``n_steps``
        attribute counts parameter updates, which is larger than
        ``n_iter_`` when each epoch contains several batches.
    """

    def __init__(
        self,
        method: MethodName = "normal_equation",
        learning_rate: float = 0.01,
        max_iter: int = 1000,
        tol: float = 1e-6,
        fit_intercept: bool = True,
        l2: float = 0.0,
        optimizer: OptimizerName = "sgd",
        momentum: float = 0.0,
        batch_size: int | None = None,
        random_state: int | None = None,
    ) -> None:
        if method not in ("normal_equation", "gradient_descent"):
            raise ValueError(
                "method must be 'normal_equation' or 'gradient_descent'."
            )
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
        if isinstance(random_state, bool) or not isinstance(
            random_state, (int, type(None))
        ):
            raise TypeError("random_state must be an int or None.")

        self.method: MethodName = method
        self.learning_rate = _positive_float("learning_rate", learning_rate)
        self.max_iter = max_iter
        self.tol = _nonnegative_float("tol", tol)
        self.fit_intercept = fit_intercept
        self.l2 = _nonnegative_float("l2", l2)
        self.optimizer: OptimizerName = optimizer
        self.momentum = _momentum(momentum)
        self.batch_size = batch_size
        self.random_state = random_state

        self.coef_: NDArray[np.float64] | None = None
        self.theta_: NDArray[np.float64] | None = None
        self.n_features_in_: int | None = None
        self.n_samples_: int | None = None
        self.n_iter_: int = 0
        self.loss_history_: list[float] = []
        self.sigma2_: float | None = None
        self.solver_used_: str | None = None
        self.rank_: int | None = None
        self.cond_: float | None = None
        self.singular_: bool | None = None
        self.ill_conditioned_: bool | None = None
        self.gram_singular_values_: NDArray[np.float64] | None = None
        self.optimizer_: SGD | Adam | None = None

    def fit(self, X: ArrayLike, y: ArrayLike) -> LinearRegression:
        """Estimate ``theta_`` from samples.

        ``X`` and ``y`` are read and not written. Gradient descent records
        the full-sample objective, including the ridge term, after every
        epoch.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training design. A one-dimensional array is treated as one
            feature column.
        y : array-like of shape (n_samples,)
            Real-valued targets. A single column of shape
            ``(n_samples, 1)`` is accepted and flattened.

        Returns
        -------
        self
        """

        features, targets = _validate_xy(X, y)
        self._reset_fit_state()
        self.n_features_in_ = int(features.shape[1])
        self.n_samples_ = int(features.shape[0])
        # The ones column is the feature that multiplies the intercept, so
        # the same gradient formula covers both the bias and the slopes.
        design = add_intercept(features) if self.fit_intercept else np.array(
            features, dtype=np.float64, copy=True
        )
        if self.method == "normal_equation":
            self._fit_normal_equation(design, targets)
        else:
            self._fit_gradient_descent(design, targets)
        self._publish_coefficients()
        assert self.theta_ is not None
        residual = targets - design @ self.theta_
        # RSS / n is the Gaussian MLE of σ². Dividing by n - rank would
        # be the unbiased estimator; rank is available as ``rank_`` after
        # a normal-equation fit.
        self.sigma2_ = float(residual @ residual) / float(targets.shape[0])
        return self

    def predict(self, X: ArrayLike) -> NDArray[np.float64]:
        r"""Return the conditional mean :math:`\theta_0 + X_{\mathrm{raw}}\theta_{1:}`.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Rows on the same feature scale used in ``fit``. A
            one-dimensional array is one feature column, not one row.

        Returns
        -------
        ndarray of shape (n_samples,)
            Predicted means.
        """

        self._check_fitted()
        features = self._validate_X(X)
        assert self.coef_ is not None and self.theta_ is not None
        # One GEMM with the slope block, then a broadcast add of the bias.
        # Building an explicit column of ones would repeat the same product
        # by θ_0 once per row. The bias is read from theta_ so the public
        # intercept_ cannot drift away from the parameter vector.
        bias = float(self.theta_[0]) if self.fit_intercept else 0.0
        return features @ self.coef_ + bias

    def score(self, X: ArrayLike, y: ArrayLike) -> float:
        r"""Return :math:`R^2` of ``predict(X)`` against ``y``.

        See :func:`ml_from_scratch.utils.metrics.r2_score`.
        """

        predictions = self.predict(X)
        observed = np.asarray(y, dtype=np.float64).reshape(-1)
        if observed.shape[0] != predictions.shape[0]:
            raise ValueError(
                "X and y must have the same number of rows. "
                f"Got {predictions.shape[0]} and {observed.shape[0]}."
            )
        if not np.isfinite(observed).all():
            raise ValueError("y must contain only finite values.")
        return r2_score(observed, predictions)

    def _fit_normal_equation(
        self,
        design: NDArray[np.float64],
        targets: NDArray[np.float64],
    ) -> None:
        """Solve :math:`(X^T X + λ P) θ = X^T y`."""

        # G_jk is the dot product of parameter-column j with column k:
        # the uncentered second-moment matrix of the design. Ill-scaled
        # columns make G ill-conditioned because those dots mix units.
        gram = design.T @ design
        # X^T y is the cross-moment between the design and the target.
        rhs = design.T @ targets
        if self.l2 > 0.0:
            penalty_diag = np.ones(gram.shape[0], dtype=np.float64)
            if self.fit_intercept:
                # Coordinate 0 of θ is the intercept. Leaving its penalty
                # at 0 is the matrix P in the docstring.
                penalty_diag[0] = 0.0
            gram = gram + self.l2 * np.diag(penalty_diag)

        singular_values = np.linalg.svd(gram, compute_uv=False)
        self.gram_singular_values_ = singular_values
        largest = float(singular_values[0])
        smallest = float(singular_values[-1])
        self.cond_ = float("inf") if smallest == 0.0 else largest / smallest
        tolerance = largest * max(gram.shape) * float(np.finfo(np.float64).eps)
        self.rank_ = int(np.sum(singular_values > tolerance))
        self.singular_ = self.rank_ < gram.shape[0]
        self.ill_conditioned_ = (not np.isfinite(self.cond_)) or self.cond_ > _COND_THRESHOLD

        if self.singular_ or self.ill_conditioned_:
            # lstsq returns the min-norm solution of G θ = X^T y. For the
            # singular normal equation that is the min-norm least-squares
            # coefficient vector, in exact arithmetic.
            solution = np.linalg.lstsq(gram, rhs, rcond=None)[0]
            self.solver_used_ = "normal_equation_pseudoinverse"
        else:
            try:
                # solve uses an LU factorization. It does not form G^{-1}.
                solution = np.linalg.solve(gram, rhs)
                self.solver_used_ = "normal_equation"
            except np.linalg.LinAlgError:
                solution = np.linalg.lstsq(gram, rhs, rcond=None)[0]
                self.solver_used_ = "normal_equation_pseudoinverse"
                self.singular_ = True

        if not np.isfinite(solution).all():
            raise FloatingPointError(
                "The normal equation produced a non-finite coefficient vector."
            )
        self.theta_ = np.asarray(solution, dtype=np.float64).copy()
        self.n_iter_ = 0
        self.loss_history_ = [self._objective(design, targets, self.theta_)]

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
        self.loss_history_ = [
            self._require_finite_loss(self._objective(design, targets, theta))
        ]
        n_samples = int(targets.shape[0])
        batch_size = n_samples if self.batch_size is None else min(self.batch_size, n_samples)
        # A new generator per fit makes ``random_state`` reproduce the
        # batch sequence even if this instance is fitted twice.
        rng = np.random.default_rng(self.random_state)

        for epoch in range(self.max_iter):
            if batch_size >= n_samples:
                gradient = self._gradient(design, targets, theta, n_samples)
                self._require_finite_gradient(gradient)
                optimizer.step(theta, gradient)
            else:
                # The epoch is sequential because θ at batch t + 1 depends
                # on θ at batch t. Within a batch the gradient is one GEMM.
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
        """Return ∇J, or an unbiased mini-batch estimate of it.

        ``MeanSquaredError.gradient`` averages over the rows it is given.
        On a batch of size ``b`` that average is ``(ŷ - y) / b``. The
        product ``X_batch.T @`` that vector sums, over the batch, each
        residual coordinate times the corresponding design row. The chain
        rule for ``ŷ = X θ`` is exactly that product: column ``j`` of
        ``X`` is ``∂ŷ / ∂θ_j``.

        The ridge gradient uses the full training size ``n_samples`` even
        inside a batch. It is the derivative of
        ``(λ / (2m)) ||P θ||^2``, a term that was defined on the whole
        empirical risk and does not get re-averaged over the batch.
        """

        prediction = design_batch @ theta
        data_gradient = design_batch.T @ MeanSquaredError.gradient(prediction, targets_batch)
        if self.l2 == 0.0:
            return data_gradient
        if self.fit_intercept:
            penalty = np.zeros_like(theta)
            # Slice 1: drops the intercept coordinate of P.
            penalty[1:] = (self.l2 / n_samples) * theta[1:]
            return data_gradient + penalty
        return data_gradient + (self.l2 / n_samples) * theta

    def _objective(
        self,
        design: NDArray[np.float64],
        targets: NDArray[np.float64],
        theta: NDArray[np.float64],
    ) -> float:
        """Return the regularized empirical risk ``J(θ)``."""

        data_loss = MeanSquaredError.loss(design @ theta, targets)
        if self.l2 == 0.0:
            return data_loss
        slopes = theta[1:] if self.fit_intercept else theta
        penalty = (self.l2 / (2.0 * targets.shape[0])) * float(slopes @ slopes)
        return data_loss + penalty

    def _make_optimizer(self) -> SGD | Adam:
        """Build a stateless-penalty optimizer.

        Ridge is applied in :meth:`_gradient`, where the intercept can be
        masked. ``weight_decay`` on the optimizer would shrink every
        coordinate, including the bias, and would double-count ``l2``.
        """

        if self.optimizer == "sgd":
            return SGD(learning_rate=self.learning_rate, momentum=self.momentum)
        return Adam(learning_rate=self.learning_rate)

    @property
    def intercept_(self) -> float:
        """Return the bias stored in ``theta_``.

        With ``fit_intercept=False`` the model has no bias parameter and
        this is 0. Before ``fit``, there is no parameter vector yet.
        """

        if self.theta_ is None:
            raise RuntimeError("LinearRegression must be fitted before use.")
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
        """Clear results so a second ``fit`` cannot keep the first fit's rank."""

        self.coef_ = None
        self.theta_ = None
        self.n_iter_ = 0
        self.loss_history_ = []
        self.sigma2_ = None
        self.solver_used_ = None
        self.rank_ = None
        self.cond_ = None
        self.singular_ = None
        self.ill_conditioned_ = None
        self.gram_singular_values_ = None
        self.optimizer_ = None

    def _check_fitted(self) -> None:
        if self.coef_ is None or self.theta_ is None or self.n_features_in_ is None:
            raise RuntimeError("LinearRegression must be fitted before use.")

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
                "The least-squares objective became non-finite during "
                f"gradient descent (learning_rate={self.learning_rate}). "
                "Standardize the columns of X or reduce learning_rate. "
                "For full-batch gradient descent with momentum 0, any "
                "step size in (0, 2/L) converges, where L is the largest "
                "eigenvalue of (X^T X + λ P) / m."
            )
        return float(loss)

    def _require_finite_gradient(self, gradient: NDArray[np.float64]) -> None:
        if not np.isfinite(gradient).all():
            raise FloatingPointError(
                "A gradient entry became non-finite during gradient descent. "
                "Standardize the columns of X or reduce learning_rate."
            )

    def __repr__(self) -> str:
        return (
            f"LinearRegression(method={self.method!r}, fit_intercept={self.fit_intercept}, "
            f"l2={self.l2})"
        )


def _validate_xy(
    X: ArrayLike,
    y: ArrayLike,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return a 2-d design and a 1-d target, both float64."""

    features = np.asarray(X, dtype=np.float64)
    targets = np.asarray(y, dtype=np.float64)
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
        raise ValueError("y must be a one-dimensional target vector.")
    if features.shape[0] != targets.shape[0]:
        raise ValueError(
            "X and y must have the same number of rows. "
            f"Got {features.shape[0]} and {targets.shape[0]}."
        )
    if features.shape[0] < 1 or features.shape[1] < 1:
        raise ValueError("X must contain at least one row and one column.")
    if not np.isfinite(features).all() or not np.isfinite(targets).all():
        raise ValueError("X and y must contain only finite values.")
    return features, targets


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
