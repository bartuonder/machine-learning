"""One-hidden-layer network trained as empirical risk minimization.

The hidden layer is an affine map followed by a pointwise nonlinearity.
The output layer is linear. Classification uses softmax cross-entropy and
regression uses squared error, both from the shared loss module. Parameter
updates are the existing SGD and Adam steps.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..optimization.losses import MeanSquaredError, SoftmaxCrossEntropy, softmax
from ..optimization.optimizers import Adam, SGD
from ..utils.metrics import accuracy_score, r2_score

ActivationName = Literal["relu", "sigmoid"]
TaskName = Literal["classification", "regression"]
OptimizerName = Literal["sgd", "adam"]


class MLP:
    r"""Multilayer perceptron with one hidden layer.

    For a row :math:`x \in \mathbb{R}^d` and hidden width :math:`h`,

    .. math::

        a = W_1^\top x + b_1,
        \qquad
        h(x) = \varphi(a),
        \qquad
        z = W_2^\top h(x) + b_2.

    In matrix form, with :math:`X \in \mathbb{R}^{m \times d}`,
    :math:`W_1 \in \mathbb{R}^{d \times h}`, and
    :math:`W_2 \in \mathbb{R}^{h \times K}`,

    .. math::

        A = X W_1 + 1 b_1^\top,
        \qquad
        H = \varphi(A),
        \qquad
        Z = H W_2 + 1 b_2^\top.

    :math:`\varphi` is ReLU or the logistic sigmoid, applied entrywise.
    Classification takes :math:`K` equal to the number of classes and
    minimizes the softmax cross-entropy of :math:`Z`. Regression takes
    :math:`K = 1` and minimizes the averaged squared error. An optional
    ridge penalty applies to :math:`W_1` and :math:`W_2` and not to the
    biases:

    .. math::

        J
        = J_{\mathrm{data}}(Z, y)
          + \frac{\lambda}{2m}\left(\lVert W_1\rVert_F^2 + \lVert W_2\rVert_F^2\right).

    Backpropagation is the chain rule. Writing :math:`G_Z = \partial J / \partial Z`,

    .. math::

        \frac{\partial J}{\partial W_2} = H^\top G_Z,
        \qquad
        \frac{\partial J}{\partial b_2} = G_Z^\top 1,

    .. math::

        G_H = G_Z W_2^\top,
        \qquad
        G_A = G_H \odot \varphi'(A),
        \qquad
        \frac{\partial J}{\partial W_1} = X^\top G_A.

    :math:`G_Z` already averages over the rows it was computed on, so these
    products are the gradients of the averaged empirical risk. The ReLU
    derivative is the indicator :math:`a > 0`. The sigmoid derivative is
    :math:`\varphi(a)(1 - \varphi(a))`.

    Each parameter array has its own SGD or Adam state. A single optimizer
    cannot hold moments for arrays of different shapes. Weight decay on
    those optimizers stays zero; :math:`\lambda` is added to the weight
    gradients so the biases remain unpenalized.

    Parameters
    ----------
    hidden_units : int, default=32
        Hidden width :math:`h`.
    activation : {"relu", "sigmoid"}, default="relu"
        Hidden nonlinearity :math:`\varphi`.
    task : {"classification", "regression"}, default="classification"
        Softmax cross-entropy, or squared error with one output.
    learning_rate : float, default=0.01
        Step size passed to every parameter's optimizer.
    max_iter : int, default=200
        Maximum passes over the training sample.
    tol : float, default=1e-6
        Stop when the absolute change in :math:`J` over an epoch falls
        below this value.
    batch_size : int or None, default=None
        Rows per gradient estimate. ``None`` uses the whole sample.
    optimizer : {"adam", "sgd"}, default="adam"
        Update rule. ``"sgd"`` uses ``momentum``.
    momentum : float, default=0.0
        Heavy-ball coefficient for SGD.
    l2 : float, default=0.0
        Ridge coefficient :math:`\lambda` on the two weight matrices.
    random_state : int or None, default=None
        Seed of the initial weights and of mini-batch order.

    Attributes
    ----------
    weights_input_ : ndarray of shape (n_features, hidden_units)
        :math:`W_1`.
    bias_hidden_ : ndarray of shape (hidden_units,)
        :math:`b_1`.
    weights_output_ : ndarray of shape (hidden_units, n_outputs)
        :math:`W_2`.
    bias_output_ : ndarray of shape (n_outputs,)
        :math:`b_2`.
    classes_ : ndarray of shape (n_classes,) or None
        Sorted labels for classification. ``None`` for regression.
    loss_history_ : list of float
        :math:`J` at initialization and after each epoch.
    n_iter_ : int
        Epochs actually run.
    n_features_in_ : int
        Columns of ``X``.
    n_outputs_ : int
        Columns of :math:`Z`.
    """

    def __init__(
        self,
        hidden_units: int = 32,
        activation: ActivationName = "relu",
        task: TaskName = "classification",
        learning_rate: float = 0.01,
        max_iter: int = 200,
        tol: float = 1e-6,
        batch_size: int | None = None,
        optimizer: OptimizerName = "adam",
        momentum: float = 0.0,
        l2: float = 0.0,
        random_state: int | None = None,
    ) -> None:
        if isinstance(hidden_units, bool) or not isinstance(hidden_units, int) or hidden_units < 1:
            raise ValueError("hidden_units must be an integer >= 1.")
        if activation not in ("relu", "sigmoid"):
            raise ValueError("activation must be 'relu' or 'sigmoid'.")
        if task not in ("classification", "regression"):
            raise ValueError("task must be 'classification' or 'regression'.")
        if optimizer not in ("sgd", "adam"):
            raise ValueError("optimizer must be 'sgd' or 'adam'.")
        if isinstance(max_iter, bool) or not isinstance(max_iter, int) or max_iter < 1:
            raise ValueError("max_iter must be an integer >= 1.")
        if batch_size is not None and (
            isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1
        ):
            raise ValueError("batch_size must be a positive integer or None.")
        if isinstance(random_state, bool) or not isinstance(random_state, (int, type(None))):
            raise TypeError("random_state must be an int or None.")
        self.hidden_units = hidden_units
        self.activation: ActivationName = activation
        self.task: TaskName = task
        self.learning_rate = _positive_float("learning_rate", learning_rate)
        self.max_iter = max_iter
        self.tol = _nonnegative_float("tol", tol)
        self.batch_size = batch_size
        self.optimizer: OptimizerName = optimizer
        self.momentum = _momentum(momentum)
        self.l2 = _nonnegative_float("l2", l2)
        self.random_state = random_state
        self.weights_input_: NDArray[np.float64] | None = None
        self.bias_hidden_: NDArray[np.float64] | None = None
        self.weights_output_: NDArray[np.float64] | None = None
        self.bias_output_: NDArray[np.float64] | None = None
        self.classes_: NDArray[Any] | None = None
        self.loss_history_: list[float] = []
        self.n_iter_: int = 0
        self.n_features_in_: int | None = None
        self.n_outputs_: int | None = None

    def fit(self, X: ArrayLike, y: ArrayLike) -> MLP:
        """Minimize :math:`J` from a random initialization.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training design. A one-dimensional array is one feature column.
        y : array-like of shape (n_samples,)
            Class labels, or real-valued regression targets.

        Returns
        -------
        self
        """

        features, targets, classes, codes = _prepare_targets(X, y, self.task)
        n_samples, n_features = features.shape
        n_outputs = 1 if self.task == "regression" else int(classes.size)  # type: ignore[union-attr]
        if self.task == "classification" and n_outputs < 2:
            raise ValueError("Classification needs at least two classes.")
        self.n_features_in_ = int(n_features)
        self.n_outputs_ = n_outputs
        self.classes_ = classes
        rng = np.random.default_rng(self.random_state)
        self.weights_input_ = _initialize_weights(rng, n_features, self.hidden_units, self.activation)
        self.bias_hidden_ = np.zeros(self.hidden_units, dtype=np.float64)
        self.weights_output_ = _initialize_weights(rng, self.hidden_units, n_outputs, self.activation)
        self.bias_output_ = np.zeros(n_outputs, dtype=np.float64)
        updaters = {name: self._make_optimizer() for name in ("input", "hidden_bias", "output", "output_bias")}
        observed: NDArray[np.float64] | NDArray[np.int64]
        observed = targets if self.task == "regression" else codes  # type: ignore[assignment]
        self.loss_history_ = [self._objective(features, observed)]
        self.n_iter_ = 0
        batch_size = n_samples if self.batch_size is None else min(self.batch_size, n_samples)
        for epoch in range(self.max_iter):
            if batch_size >= n_samples:
                gradients = self._gradients(features, observed, n_samples)
                self._apply(updaters, gradients)
            else:
                permutation = rng.permutation(n_samples)
                for start in range(0, n_samples, batch_size):
                    batch = permutation[start : start + batch_size]
                    gradients = self._gradients(features[batch], observed[batch], n_samples)
                    self._apply(updaters, gradients)
            loss = self._objective(features, observed)
            if not np.isfinite(loss):
                raise FloatingPointError(
                    "The network objective became non-finite. "
                    "Reduce learning_rate or increase l2."
                )
            self.loss_history_.append(float(loss))
            self.n_iter_ = epoch + 1
            if abs(self.loss_history_[-1] - self.loss_history_[-2]) < self.tol:
                break
        return self

    def decision_function(self, X: ArrayLike) -> NDArray[np.float64]:
        """Return the output-layer scores :math:`Z`.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Rows on the scale used in ``fit``.

        Returns
        -------
        ndarray of shape (n_samples, n_outputs)
            Logits for classification, or the regression prediction in one
            column.
        """

        features = self._validate_fitted(X)
        _hidden, scores = self._forward(features)
        return scores

    def predict_proba(self, X: ArrayLike) -> NDArray[np.float64]:
        """Return softmax probabilities for a classification network.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Rows on the scale used in ``fit``.

        Returns
        -------
        ndarray of shape (n_samples, n_classes)
            Class probabilities. Column ``k`` is ``classes_[k]``.
        """

        if self.task != "classification":
            raise AttributeError("predict_proba is defined for task='classification'.")
        return softmax(self.decision_function(X))

    def predict(self, X: ArrayLike) -> NDArray[Any]:
        """Return class labels, or the regression prediction.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Rows on the scale used in ``fit``.

        Returns
        -------
        ndarray of shape (n_samples,)
            Labels from ``classes_``, or real-valued predictions.
        """

        scores = self.decision_function(X)
        if self.task == "regression":
            return scores.reshape(-1)
        assert self.classes_ is not None
        return self.classes_[np.argmax(scores, axis=1)]

    def score(self, X: ArrayLike, y: ArrayLike) -> float:
        """Return accuracy for classification, or :math:`R^2` for regression."""

        predicted = self.predict(X)
        if self.task == "regression":
            return r2_score(y, predicted)
        return accuracy_score(y, predicted)

    def _forward(
        self,
        features: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return hidden activations and output scores."""

        assert (
            self.weights_input_ is not None
            and self.bias_hidden_ is not None
            and self.weights_output_ is not None
            and self.bias_output_ is not None
        )
        pre_activation = features @ self.weights_input_ + self.bias_hidden_
        hidden = _activate(pre_activation, self.activation)
        scores = hidden @ self.weights_output_ + self.bias_output_
        return hidden, scores

    def _objective(
        self,
        features: NDArray[np.float64],
        observed: NDArray[np.float64] | NDArray[np.int64],
    ) -> float:
        """Return the regularized empirical risk on these rows."""

        _hidden, scores = self._forward(features)
        data_loss = _data_loss(scores, observed, self.task)
        return data_loss + self._penalty(features.shape[0])

    def _gradients(
        self,
        features: NDArray[np.float64],
        observed: NDArray[np.float64] | NDArray[np.int64],
        n_samples: int,
    ) -> dict[str, NDArray[np.float64]]:
        """Return gradients of :math:`J` for each parameter array."""

        assert (
            self.weights_input_ is not None
            and self.bias_hidden_ is not None
            and self.weights_output_ is not None
            and self.bias_output_ is not None
        )
        pre_activation = features @ self.weights_input_ + self.bias_hidden_
        hidden = _activate(pre_activation, self.activation)
        scores = hidden @ self.weights_output_ + self.bias_output_
        output_gradient = _output_gradient(scores, observed, self.task)
        # H^T G_Z has shape (h, K). Each column is the hidden activation
        # correlated with one output residual.
        grad_output = hidden.T @ output_gradient
        grad_output_bias = output_gradient.sum(axis=0)
        hidden_gradient = output_gradient @ self.weights_output_.T
        pre_gradient = hidden_gradient * _activation_gradient(pre_activation, hidden, self.activation)
        grad_input = features.T @ pre_gradient
        grad_hidden_bias = pre_gradient.sum(axis=0)
        if self.l2 != 0.0:
            scale = self.l2 / n_samples
            grad_input = grad_input + scale * self.weights_input_
            grad_output = grad_output + scale * self.weights_output_
        return {
            "input": grad_input,
            "hidden_bias": grad_hidden_bias,
            "output": grad_output,
            "output_bias": grad_output_bias,
        }

    def _apply(
        self,
        updaters: dict[str, SGD | Adam],
        gradients: dict[str, NDArray[np.float64]],
    ) -> None:
        """Step every parameter with its own optimizer state."""

        assert (
            self.weights_input_ is not None
            and self.bias_hidden_ is not None
            and self.weights_output_ is not None
            and self.bias_output_ is not None
        )
        updaters["input"].step(self.weights_input_, gradients["input"])
        updaters["hidden_bias"].step(self.bias_hidden_, gradients["hidden_bias"])
        updaters["output"].step(self.weights_output_, gradients["output"])
        updaters["output_bias"].step(self.bias_output_, gradients["output_bias"])

    def _penalty(self, n_samples: int) -> float:
        r"""Return :math:`(\lambda / (2m))(\lVert W_1\rVert_F^2 + \lVert W_2\rVert_F^2)`."""

        if self.l2 == 0.0:
            return 0.0
        assert self.weights_input_ is not None and self.weights_output_ is not None
        weight = float(np.sum(self.weights_input_ * self.weights_input_))
        weight += float(np.sum(self.weights_output_ * self.weights_output_))
        return (self.l2 / (2.0 * n_samples)) * weight

    def _make_optimizer(self) -> SGD | Adam:
        """Build a fresh optimizer with no extra weight decay."""

        if self.optimizer == "adam":
            return Adam(learning_rate=self.learning_rate)
        return SGD(learning_rate=self.learning_rate, momentum=self.momentum)

    def _validate_fitted(self, X: ArrayLike) -> NDArray[np.float64]:
        if self.weights_input_ is None or self.n_features_in_ is None:
            raise RuntimeError("MLP must be fitted before use.")
        features = _validate_X(X)
        if features.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Expected {self.n_features_in_} features, got {features.shape[1]}."
            )
        return features

    def __repr__(self) -> str:
        return (
            f"MLP(hidden_units={self.hidden_units}, activation={self.activation!r}, "
            f"task={self.task!r})"
        )


def _activate(pre_activation: NDArray[np.float64], activation: ActivationName) -> NDArray[np.float64]:
    r"""Apply :math:`\varphi` entrywise."""

    if activation == "relu":
        return np.maximum(pre_activation, 0.0)
    return _sigmoid(pre_activation)


def _activation_gradient(
    pre_activation: NDArray[np.float64],
    hidden: NDArray[np.float64],
    activation: ActivationName,
) -> NDArray[np.float64]:
    r"""Return :math:`\varphi'(A)`, same shape as the pre-activation."""

    if activation == "relu":
        return (pre_activation > 0.0).astype(np.float64)
    return hidden * (1.0 - hidden)


def _sigmoid(values: NDArray[np.float64]) -> NDArray[np.float64]:
    """Stable logistic function, any shape."""

    out = np.empty_like(values)
    positive = values >= 0.0
    out[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    negative = np.exp(values[~positive])
    out[~positive] = negative / (1.0 + negative)
    return out


def _data_loss(
    scores: NDArray[np.float64],
    observed: NDArray[np.float64] | NDArray[np.int64],
    task: TaskName,
) -> float:
    """Return the data term of :math:`J`."""

    if task == "regression":
        return MeanSquaredError.loss(scores.reshape(-1), observed)
    return SoftmaxCrossEntropy.loss(scores, observed)


def _output_gradient(
    scores: NDArray[np.float64],
    observed: NDArray[np.float64] | NDArray[np.int64],
    task: TaskName,
) -> NDArray[np.float64]:
    r"""Return :math:`\partial J / \partial Z` with shape ``(n_batch, n_outputs)``."""

    if task == "regression":
        flat = MeanSquaredError.gradient(scores.reshape(-1), observed)
        return flat.reshape(scores.shape)
    return SoftmaxCrossEntropy.gradient(scores, observed)


def _initialize_weights(
    rng: np.random.Generator,
    fan_in: int,
    fan_out: int,
    activation: ActivationName,
) -> NDArray[np.float64]:
    """Draw :math:`W` so the hidden pre-activations start at order-one scale.

    ReLU uses :math:`\\sqrt{2 / d_{\\mathrm{in}}}`. Sigmoid uses
    :math:`\\sqrt{1 / d_{\\mathrm{in}}}`.
    """

    scale = np.sqrt((2.0 if activation == "relu" else 1.0) / fan_in)
    return rng.normal(0.0, scale, size=(fan_in, fan_out))


def _prepare_targets(
    X: ArrayLike,
    y: ArrayLike,
    task: TaskName,
) -> tuple[NDArray[np.float64], NDArray[np.float64] | None, NDArray[Any] | None, NDArray[np.int64] | None]:
    """Return features, regression targets, classes, and integer codes."""

    features = _validate_X(X)
    labels = np.asarray(y)
    if labels.ndim == 2 and labels.shape[1] == 1:
        labels = labels.reshape(-1)
    if labels.ndim != 1 or labels.shape[0] != features.shape[0]:
        raise ValueError("y must be a one-dimensional vector aligned with the rows of X.")
    if task == "regression":
        targets = labels.astype(np.float64, copy=False)
        if not np.isfinite(targets).all():
            raise ValueError("Regression targets must be finite.")
        return features, targets, None, None
    if labels.dtype.kind == "f" and not np.isfinite(np.asarray(labels, dtype=np.float64)).all():
        raise ValueError("y must contain only finite values.")
    classes, codes = np.unique(labels, return_inverse=True)
    return features, None, classes, codes.astype(np.int64, copy=False)


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
