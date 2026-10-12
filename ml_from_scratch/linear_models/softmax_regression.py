"""Multinomial logistic regression, the softmax GLM.

Rows of ``X`` are examples. Class probabilities are the softmax of a
linear score per class, and the training objective is the averaged
multinomial negative log-likelihood.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..optimization.losses import SoftmaxCrossEntropy, softmax
from ..optimization.optimizers import Adam, SGD
from ..utils.data_manipulation import add_intercept
from ..utils.metrics import accuracy_score

OptimizerName = Literal["sgd", "adam"]


class SoftmaxRegression:
    r"""K-class logistic regression.

    For a feature vector :math:`x \in \mathbb{R}^d` and a parameter
    :math:`\theta_k` for each class :math:`k \in \{1, \ldots, K\}`,

    .. math::

        P(y = k \mid x)
        = \frac{\exp(\theta_k^\top x)}{\sum_{j=1}^{K} \exp(\theta_j^\top x)}.

    Stacking :math:`\theta_k` as the columns of
    :math:`\Theta \in \mathbb{R}^{d \times K}` makes the score matrix
    :math:`Z = X\Theta`, with :math:`X \in \mathbb{R}^{m \times d}` storing
    one example per row. An intercept is an extra column of ones, so it is
    a row of :math:`\Theta`. The training objective is the multinomial
    negative log-likelihood plus a ridge penalty on the slopes,

    .. math::

        J(\Theta)
        = -\frac{1}{m}\sum_{i=1}^{m}\log p_{i, y_i}
          + \frac{\lambda}{2m}\lVert P\Theta\rVert_F^2.

    :math:`P` zeros the intercept row and leaves every slope row untouched.
    The softmax is unchanged if the same vector is added to every
    :math:`\theta_k`. The likelihood alone therefore does not identify
    :math:`\Theta`. The ridge term pins that shift. When :math:`K = 2`,
    only the difference :math:`\theta_1 - \theta_0` enters the probability,
    and that difference is the weight vector of binary logistic regression.

    The gradient in logit space is :math:`(P_{\mathrm{prob}} - Y) / m`.
    The chain rule through :math:`Z = X\Theta` is

    .. math::

        \nabla_\Theta J
        = \frac{1}{m} X^\top(P_{\mathrm{prob}} - Y)
          + \frac{\lambda}{m} P\Theta.

    :math:`P_{\mathrm{prob}}` and :math:`Y` have shape ``(m, K)``. Setting
    the unpenalized gradient to zero is the likelihood equation
    :math:`X^\top(P_{\mathrm{prob}} - Y) = 0`.

    The Hessian is a :math:`Kd \times Kd` block matrix. Block
    :math:`(k, \ell)` is
    :math:`\frac{1}{m} X^\top \operatorname{diag}(p_k(\mathbf{1}_{k=\ell} - p_\ell)) X`.
    This class minimizes :math:`J` with gradient descent or Adam rather than
    forming that block matrix. A mini-batch replaces only the data term by
    an average over the batch. The penalty uses the full training size
    :math:`m` on every step.

    Parameters
    ----------
    learning_rate : float, default=0.1
        Step size.
    max_iter : int, default=200
        Maximum passes over the training sample.
    tol : float, default=1e-6
        Stop when the absolute change in :math:`J` over an epoch falls
        below this value.
    fit_intercept : bool, default=True
        Estimate an unpenalized intercept for every class.
    l2 : float, default=0.0
        Ridge coefficient :math:`\lambda`.
    optimizer : {"sgd", "adam"}, default="sgd"
        Recurrence applied to :math:`\nabla J`. Its weight decay stays 0;
        the ridge term is inside the gradient so the intercept row can be
        excluded.
    momentum : float, default=0.0
        Heavy-ball coefficient for ``optimizer="sgd"``.
    batch_size : int or None, default=None
        Rows per gradient estimate. ``None`` uses the whole sample.
    random_state : int or None, default=None
        Seed of the mini-batch permutation.

    Attributes
    ----------
    coef_ : ndarray of shape (n_classes, n_features)
        Slope rows. Row :math:`k` is :math:`\theta_k` without the intercept.
        This is a view of ``theta_``.
    intercept_ : ndarray of shape (n_classes,)
        Intercept of each class. Zeros when ``fit_intercept`` is false.
    theta_ : ndarray of shape (n_parameters, n_classes)
        Full parameter matrix. Row 0 is the intercept when
        ``fit_intercept`` is true.
    classes_ : ndarray of shape (n_classes,)
        Sorted training labels. Column ``k`` of ``predict_proba`` is
        ``classes_[k]``.
    n_iter_ : int
        Epochs actually run.
    loss_history_ : list of float
        :math:`J` at initialization and after each epoch.
    n_features_in_ : int
        Columns of ``X``, excluding the intercept.
    n_samples_ : int
        Rows of the last ``fit``.
    """

    def __init__(
        self,
        learning_rate: float = 0.1,
        max_iter: int = 200,
        tol: float = 1e-6,
        fit_intercept: bool = True,
        l2: float = 0.0,
        optimizer: OptimizerName = "sgd",
        momentum: float = 0.0,
        batch_size: int | None = None,
        random_state: int | None = None,
    ) -> None:
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
        self.intercept_: NDArray[np.float64] | None = None
        self.theta_: NDArray[np.float64] | None = None
        self.classes_: NDArray[Any] | None = None
        self.n_iter_: int = 0
        self.loss_history_: list[float] = []
        self.n_features_in_: int | None = None
        self.n_samples_: int | None = None

    def fit(self, X: ArrayLike, y: ArrayLike) -> SoftmaxRegression:
        """Minimize the multinomial negative log-likelihood.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training design. A one-dimensional array is one feature column.
        y : array-like of shape (n_samples,)
            Class labels. At least two classes are required.

        Returns
        -------
        self
        """

        features, labels = _validate_xy(X, y)
        classes, codes = np.unique(labels, return_inverse=True)
        if classes.size < 2:
            raise ValueError("SoftmaxRegression needs at least two classes.")
        self.classes_ = classes
        self.n_features_in_ = int(features.shape[1])
        self.n_samples_ = int(features.shape[0])
        design = add_intercept(features) if self.fit_intercept else np.array(
            features, dtype=np.float64, copy=True
        )
        theta = np.zeros((design.shape[1], classes.size), dtype=np.float64)
        updater = self._make_optimizer()
        self.loss_history_ = [self._objective(design, codes, theta)]
        self.n_iter_ = 0
        n_samples = int(codes.shape[0])
        batch_size = n_samples if self.batch_size is None else min(self.batch_size, n_samples)
        rng = np.random.default_rng(self.random_state)
        for epoch in range(self.max_iter):
            if batch_size >= n_samples:
                gradient = self._gradient(design, codes, theta, n_samples)
                updater.step(theta, gradient)
            else:
                permutation = rng.permutation(n_samples)
                for start in range(0, n_samples, batch_size):
                    batch = permutation[start : start + batch_size]
                    gradient = self._gradient(design[batch], codes[batch], theta, n_samples)
                    updater.step(theta, gradient)
            loss = self._objective(design, codes, theta)
            if not np.isfinite(loss):
                raise FloatingPointError(
                    "The multinomial objective became non-finite. "
                    "Standardize X, reduce learning_rate, or increase l2."
                )
            self.loss_history_.append(float(loss))
            self.n_iter_ = epoch + 1
            if abs(self.loss_history_[-1] - self.loss_history_[-2]) < self.tol:
                break
        self.theta_ = theta
        if self.fit_intercept:
            self.intercept_ = theta[0]
            self.coef_ = theta[1:].T
        else:
            self.intercept_ = np.zeros(classes.size, dtype=np.float64)
            self.coef_ = theta.T
        return self

    def decision_function(self, X: ArrayLike) -> NDArray[np.float64]:
        """Return the class scores :math:`X\\Theta` before the softmax.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Rows on the scale used in ``fit``.

        Returns
        -------
        ndarray of shape (n_samples, n_classes)
            Logits. Column ``k`` belongs to ``classes_[k]``.
        """

        self._check_fitted()
        assert self.coef_ is not None and self.intercept_ is not None and self.n_features_in_ is not None
        features = _validate_X(X)
        if features.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Expected {self.n_features_in_} features, got {features.shape[1]}."
            )
        # coef_ is (K, d), so its transpose is the (d, K) parameter matrix.
        return features @ self.coef_.T + self.intercept_

    def predict_proba(self, X: ArrayLike) -> NDArray[np.float64]:
        """Return :math:`P(y = k \\mid x)` for every class.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Rows on the scale used in ``fit``.

        Returns
        -------
        ndarray of shape (n_samples, n_classes)
            Softmax probabilities. Each row sums to 1.
        """

        return softmax(self.decision_function(X))

    def predict(self, X: ArrayLike) -> NDArray[Any]:
        """Return the class with the largest probability.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Rows on the scale used in ``fit``.

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

    def _gradient(
        self,
        design_batch: NDArray[np.float64],
        codes_batch: NDArray[np.int64],
        theta: NDArray[np.float64],
        n_samples: int,
    ) -> NDArray[np.float64]:
        """Return :math:`\\nabla J` for this batch.

        ``SoftmaxCrossEntropy.gradient`` averages over the rows it sees, so
        on a batch of size :math:`b` it equals
        :math:`(P - Y) / b`. The product ``X_batch.T @`` that matrix is the
        chain rule for :math:`Z = X\\Theta`: it has shape
        ``(n_parameters, n_classes)``. The ridge gradient uses the full
        training size, not :math:`b`.
        """

        scores = design_batch @ theta
        data_gradient = design_batch.T @ SoftmaxCrossEntropy.gradient(scores, codes_batch)
        if self.l2 == 0.0:
            return data_gradient
        penalty = np.zeros_like(theta)
        slope = slice(1, None) if self.fit_intercept else slice(None)
        penalty[slope] = (self.l2 / n_samples) * theta[slope]
        return data_gradient + penalty

    def _objective(
        self,
        design: NDArray[np.float64],
        codes: NDArray[np.int64],
        theta: NDArray[np.float64],
    ) -> float:
        """Return the regularized multinomial risk."""

        data_loss = SoftmaxCrossEntropy.loss(design @ theta, codes)
        if self.l2 == 0.0:
            return data_loss
        slopes = theta[1:] if self.fit_intercept else theta
        penalty = (self.l2 / (2.0 * codes.shape[0])) * float(np.sum(slopes * slopes))
        return data_loss + penalty

    def _make_optimizer(self) -> SGD | Adam:
        """Build an optimizer that does not add a second ridge term."""

        if self.optimizer == "adam":
            return Adam(learning_rate=self.learning_rate)
        return SGD(learning_rate=self.learning_rate, momentum=self.momentum)

    def _check_fitted(self) -> None:
        if self.theta_ is None or self.classes_ is None:
            raise RuntimeError("SoftmaxRegression must be fitted before use.")

    def __repr__(self) -> str:
        return (
            f"SoftmaxRegression(fit_intercept={self.fit_intercept}, l2={self.l2}, "
            f"optimizer={self.optimizer!r})"
        )


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
