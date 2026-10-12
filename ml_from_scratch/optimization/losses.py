"""Losses, their parameterizations, and the gradients used for fitting.

A loss here is an empirical risk: a mean of per-example negative
log-likelihoods, or a convex surrogate of the 0-1 loss. Gradients are
with respect to the prediction that the loss sees, not with respect to
model parameters. The chain rule that carries this gradient through a
linear predictor lives with the model, because that step depends on the
design matrix.
"""

from __future__ import annotations

from numpy.typing import ArrayLike, NDArray

import numpy as np


def sigmoid(z: ArrayLike) -> NDArray[np.float64]:
    r"""Return the logistic function, the mean of a Bernoulli GLM.

    .. math::

        \sigma(z) = \frac{1}{1 + e^{-z}}.

    This is the inverse of the logit link. It maps a real score to the
    success probability of a Bernoulli random variable. The derivative
    :math:`\sigma'(z) = \sigma(z)\,(1 - \sigma(z))` is what cancels the
    denominator of the Bernoulli negative log-likelihood when the loss is
    differentiated in logit coordinates.

    The implementation splits on the sign of ``z``. For ``z >= 0`` it
    evaluates ``exp(-z)``, which is at most 1. For ``z < 0`` it evaluates
    ``exp(z)``, which is at most 1. Either way the exponential sees a
    non-positive argument, so a large positive or large negative score
    cannot overflow.

    Parameters
    ----------
    z : array-like
        Real-valued logits, any shape.

    Returns
    -------
    ndarray
        Probabilities in ``(0, 1)``, same shape as ``z``.
    """

    logits = np.asarray(z, dtype=np.float64)
    probabilities = np.empty_like(logits, dtype=np.float64)
    positive = logits >= 0.0
    probabilities[positive] = 1.0 / (1.0 + np.exp(-logits[positive]))
    negative_exp = np.exp(logits[~positive])
    probabilities[~positive] = negative_exp / (1.0 + negative_exp)
    return probabilities


class MeanSquaredError:
    r"""Empirical risk of a homoscedastic Gaussian linear model.

    If :math:`y \mid x \sim \mathcal{N}(x^\top\theta, \sigma^2)`, the
    average negative log-likelihood is

    .. math::

        \frac{1}{2\sigma^2 m}\sum_{i=1}^{m}(\hat{y}_i - y_i)^2
        + \frac{1}{2}\log(2\pi\sigma^2).

    The variance and the additive constant do not change the minimizer
    over :math:`\hat{y}` (equivalently, over :math:`\theta`). Dropping
    them and keeping the factor :math:`1/2` from the Gaussian density
    leaves the training objective

    .. math::

        J(\hat{y}, y)
        = \frac{1}{2m}\sum_{i=1}^{m}(\hat{y}_i - y_i)^2.

    Differentiating with respect to the prediction gives

    .. math::

        \frac{\partial J}{\partial \hat{y}}
        = \frac{1}{m}(\hat{y} - y).

    The factor :math:`1/2` is why the gradient is the averaged residual
    and not twice that residual. Gaussianity is the likelihood
    justification. The same objective is the ordinary-least-squares
    criterion, and the Gauss-Markov theorem gives it a second
    justification that does not assume Gaussian noise: among linear
    unbiased estimators it has minimum variance when the errors are
    uncorrelated and homoscedastic.

    The average runs over every element of the arrays. A target of shape
    ``(m,)`` averages over the sample. A target of shape ``(m, k)``
    averages over samples and outputs together.
    """

    def __call__(self, y_pred: ArrayLike, y_true: ArrayLike) -> float:
        """Return :meth:`loss`."""

        return self.loss(y_pred, y_true)

    @staticmethod
    def loss(y_pred: ArrayLike, y_true: ArrayLike) -> float:
        r"""Return :math:`J = \frac{1}{2m}\lVert\hat{y} - y\rVert_2^2`.

        Parameters
        ----------
        y_pred : array-like
            Predictions :math:`\hat{y}`.
        y_true : array-like
            Observations :math:`y`, broadcastable to ``y_pred`` by a
            reshape when the two arrays contain the same number of
            elements.

        Returns
        -------
        float
            Averaged Gaussian objective.
        """

        predicted, observed = _align(y_pred, y_true)
        residual = predicted - observed
        # ``sum(square)`` is the squared Frobenius norm for any shape.
        # Overflow is an infinite objective, not a separate warning: callers
        # that minimize J test ``isfinite`` and stop with their own error.
        with np.errstate(over="ignore"):
            return float(np.sum(residual * residual) / (2.0 * predicted.size))

    @staticmethod
    def gradient(y_pred: ArrayLike, y_true: ArrayLike) -> NDArray[np.float64]:
        r"""Return :math:`\partial J / \partial\hat{y} = (\hat{y} - y) / m`.

        Parameters
        ----------
        y_pred, y_true : array-like
            Predictions and observations. See :meth:`loss`.

        Returns
        -------
        ndarray
            Gradient in prediction space, same shape as ``y_pred``.
        """

        predicted, observed = _align(y_pred, y_true)
        # Division by m is the derivative of the sample average. It is
        # applied elementwise because each coordinate of ŷ appears in
        # exactly one squared term. Overflow yields a non-finite gradient.
        with np.errstate(over="ignore"):
            return (predicted - observed) / predicted.size


class BinaryCrossEntropy:
    r"""Average Bernoulli negative log-likelihood.

    A label :math:`y \in [0, 1]` with success probability :math:`\hat{y}`
    has log-likelihood

    .. math::

        \ell = y\log\hat{y} + (1 - y)\log(1 - \hat{y}).

    Soft targets in ``(0, 1)`` are allowed: they are the same criterion
    with the empirical label replaced by a conditional probability. The
    training objective is the average negative log-likelihood

    .. math::

        J(\hat{y}, y)
        = -\frac{1}{m}\sum_{i=1}^{m}
          \left[
            y_i\log\hat{y}_i + (1 - y_i)\log(1 - \hat{y}_i)
          \right].

    In probability coordinates the gradient is

    .. math::

        \frac{\partial J}{\partial\hat{y}}
        = \frac{1}{m}
          \frac{\hat{y} - y}{\hat{y}\,(1 - \hat{y})}.

    That expression is unstable for training. A probability sitting next
    to 0 or 1 makes the denominator vanish, and forming the probability
    with a sigmoid overflows long before that. The parameterization used
    for fitting writes :math:`\hat{y} = \sigma(z)` and differentiates the
    same negative log-likelihood with respect to the logit :math:`z`:

    .. math::

        \frac{\partial J}{\partial z}
        = \frac{\partial J}{\partial\hat{y}}
          \frac{\partial\hat{y}}{\partial z}
        = \frac{1}{m}
          \frac{\sigma(z) - y}{\sigma(z)\,(1 - \sigma(z))}
          \cdot \sigma(z)\,(1 - \sigma(z))
        = \frac{1}{m}(\sigma(z) - y).

    The sigmoid derivative cancels the denominator. :meth:`gradient_with_logits`
    is that reduced expression. :meth:`loss_with_logits` evaluates the same
    :math:`J` with a stable softplus identity,

    .. math::

        -\ell = \operatorname{softplus}(z) - y z,
        \quad
        \operatorname{softplus}(z) = \log(1 + e^{z}).
    """

    def __call__(self, y_pred: ArrayLike, y_true: ArrayLike) -> float:
        """Return :meth:`loss`."""

        return self.loss(y_pred, y_true)

    @staticmethod
    def loss(y_pred: ArrayLike, y_true: ArrayLike) -> float:
        r"""Return Bernoulli negative log-likelihood in probability coordinates.

        Predictions must lie in the open interval ``(0, 1)``. Values
        outside that interval are not Bernoulli probabilities, and clipping
        them would replace :math:`J` by a different function whose gradient
        is zero outside the clip. Real-valued model scores should use
        :meth:`loss_with_logits`.

        Parameters
        ----------
        y_pred : array-like
            Probabilities :math:`\hat{y} \in (0, 1)`.
        y_true : array-like
            Targets in ``[0, 1]``.

        Returns
        -------
        float
            Mean negative log-likelihood.
        """

        predicted, observed = _align(y_pred, y_true)
        _check_unit_interval(observed, "targets", closed=True)
        _check_unit_interval(predicted, "predictions", closed=False)
        # Each term is the log probability of one Bernoulli observation.
        # The mean is the empirical risk under the training measure.
        log_likelihood = observed * np.log(predicted) + (1.0 - observed) * np.log(
            1.0 - predicted
        )
        return float(-np.mean(log_likelihood))

    @staticmethod
    def gradient(y_pred: ArrayLike, y_true: ArrayLike) -> NDArray[np.float64]:
        r"""Return :math:`\partial J/\partial\hat{y}`.

        Parameters
        ----------
        y_pred, y_true : array-like
            Probabilities in ``(0, 1)`` and targets in ``[0, 1]``.

        Returns
        -------
        ndarray
            Gradient in probability space, same shape as ``y_pred``.
        """

        predicted, observed = _align(y_pred, y_true)
        _check_unit_interval(observed, "targets", closed=True)
        _check_unit_interval(predicted, "predictions", closed=False)
        # Quotient rule on -y log p - (1-y) log(1-p), then divide by m
        # because J averages the per-example terms.
        return (predicted - observed) / (
            predicted.size * predicted * (1.0 - predicted)
        )

    @staticmethod
    def loss_with_logits(logits: ArrayLike, y_true: ArrayLike) -> float:
        r"""Return the Bernoulli negative log-likelihood from logits.

        Uses :math:`\operatorname{softplus}(z) - y z`, which equals
        :math:`-\ell(\sigma(z), y)`. ``softplus`` is evaluated as
        ``max(z, 0) + log1p(exp(-abs(z)))`` so that ``exp`` is only ever
        applied to a non-positive number.

        Parameters
        ----------
        logits : array-like
            Real scores :math:`z` before the sigmoid.
        y_true : array-like
            Targets in ``[0, 1]``.

        Returns
        -------
        float
            Mean negative log-likelihood.
        """

        scores, observed = _align(logits, y_true)
        _check_unit_interval(observed, "targets", closed=True)
        softplus = np.maximum(scores, 0.0) + np.log1p(np.exp(-np.abs(scores)))
        return float(np.mean(softplus - observed * scores))

    @staticmethod
    def gradient_with_logits(logits: ArrayLike, y_true: ArrayLike) -> NDArray[np.float64]:
        r"""Return :math:`\partial J/\partial z = (\sigma(z) - y) / m`.

        Parameters
        ----------
        logits, y_true : array-like
            Real scores and targets in ``[0, 1]``.

        Returns
        -------
        ndarray
            Gradient in logit space, same shape as ``logits``.
        """

        scores, observed = _align(logits, y_true)
        _check_unit_interval(observed, "targets", closed=True)
        # See the class docstring for the cancellation of σ(z) (1-σ(z)).
        return (sigmoid(scores) - observed) / scores.size


def softmax(logits: ArrayLike) -> NDArray[np.float64]:
    r"""Return row-wise softmax probabilities.

    For a score matrix :math:`Z \in \mathbb{R}^{m \times K}`,

    .. math::

        \operatorname{softmax}(Z)_{ik}
        = \frac{e^{z_{ik}}}{\sum_{j=1}^{K} e^{z_{ij}}}.

    Each row is shifted by its maximum before the exponential, so a large
    shared offset cannot overflow. The shift cancels in the ratio. The
    result has the same shape as ``logits``, and each row sums to 1.

    Parameters
    ----------
    logits : array-like of shape (n_samples, n_classes)
        Real scores.

    Returns
    -------
    ndarray of shape (n_samples, n_classes)
        Class probabilities.
    """

    scores = _as_logit_matrix(logits)
    shifted = scores - np.max(scores, axis=1, keepdims=True)
    unnormalized = np.exp(shifted)
    return unnormalized / np.sum(unnormalized, axis=1, keepdims=True)


def log_softmax(logits: ArrayLike) -> NDArray[np.float64]:
    r"""Return the log of :func:`softmax` without forming the probabilities.

    .. math::

        \log \operatorname{softmax}(Z)_{ik}
        = z_{ik} - \log \sum_{j=1}^{K} e^{z_{ij}}.

    The row maximum is subtracted inside the log-sum-exp.

    Parameters
    ----------
    logits : array-like of shape (n_samples, n_classes)
        Real scores.

    Returns
    -------
    ndarray of shape (n_samples, n_classes)
        Log probabilities.
    """

    scores = _as_logit_matrix(logits)
    shifted = scores - np.max(scores, axis=1, keepdims=True)
    return shifted - np.log(np.sum(np.exp(shifted), axis=1, keepdims=True))


class SoftmaxCrossEntropy:
    r"""Average multinomial negative log-likelihood.

    A label :math:`y` taking values in :math:`\{1, \ldots, K\}` with class
    probabilities :math:`p = \operatorname{softmax}(z)` has log-likelihood
    :math:`\sum_k y_k \log p_k`, where :math:`y` is a one-hot (or soft)
    vector. The training objective is

    .. math::

        J(Z, y)
        = -\frac{1}{m}\sum_{i=1}^{m}\sum_{k=1}^{K} y_{ik}\log p_{ik},
        \qquad
        p_i = \operatorname{softmax}(z_i).

    Differentiating through the normalization produces the same cancellation
    as Bernoulli logistic regression:

    .. math::

        \frac{\partial J}{\partial Z} = \frac{1}{m}(P - Y).

    :math:`P` and :math:`Y` have shape ``(m, K)``. The factor :math:`1/m`
    is the derivative of the average over rows, not over classes: each row
    contributes one log-probability. When :math:`K = 2` and the two score
    columns differ by a single logit, this is binary cross-entropy.

    Integer labels of shape ``(m,)`` are the class index in
    ``0, \ldots, K - 1``. A target matrix of shape ``(m, K)`` is a one-hot
    encoding or a matrix of soft labels whose rows sum to 1.
    """

    def __call__(self, logits: ArrayLike, y_true: ArrayLike) -> float:
        """Return :meth:`loss`."""

        return self.loss(logits, y_true)

    @staticmethod
    def loss(logits: ArrayLike, y_true: ArrayLike) -> float:
        r"""Return :math:`J = -\frac{1}{m}\sum_i \log p_{i, y_i}`.

        Parameters
        ----------
        logits : array-like of shape (n_samples, n_classes)
            Scores :math:`Z` before the softmax.
        y_true : array-like of shape (n_samples,) or (n_samples, n_classes)
            Integer class indices, or a target distribution per row.

        Returns
        -------
        float
            Mean negative log-likelihood.
        """

        scores = _as_logit_matrix(logits)
        log_probabilities = log_softmax(scores)
        targets = _classification_targets(y_true, scores.shape[0], scores.shape[1])
        if targets.ndim == 1:
            chosen = log_probabilities[np.arange(scores.shape[0]), targets]
            return float(-np.mean(chosen))
        # Row-wise dot of the target distribution with the log probabilities,
        # then the average over samples.
        return float(-np.mean(np.sum(targets * log_probabilities, axis=1)))

    @staticmethod
    def gradient(logits: ArrayLike, y_true: ArrayLike) -> NDArray[np.float64]:
        r"""Return :math:`\partial J/\partial Z = (P - Y) / m`.

        Parameters
        ----------
        logits, y_true : array-like
            Scores of shape ``(n_samples, n_classes)`` and integer labels
            or a target matrix.

        Returns
        -------
        ndarray of shape (n_samples, n_classes)
            Gradient in logit space.
        """

        scores = _as_logit_matrix(logits)
        probabilities = softmax(scores)
        targets = _classification_targets(y_true, scores.shape[0], scores.shape[1])
        if targets.ndim == 1:
            # Subtract 1 from the true-class column. That is Y in one-hot
            # form without allocating a dense indicator when labels are indices.
            probabilities[np.arange(scores.shape[0]), targets] -= 1.0
            residual = probabilities
        else:
            residual = probabilities - targets
        return residual / scores.shape[0]


class HingeLoss:
    r"""Convex surrogate of the 0-1 classification loss.

    With labels :math:`y_i \in \{-1, +1\}` and a real score
    :math:`\hat{y}_i`, the hinge objective is

    .. math::

        J(\hat{y}, y)
        = \frac{1}{m}\sum_{i=1}^{m}\max(0,\, 1 - y_i\hat{y}_i).

    The term :math:`1\{y_i\hat{y}_i \le 0\}` is the 0-1 loss. The hinge
    is a convex upper bound on that indicator once the decision rule is
    ``sign(ŷ)``. Its subgradient with respect to the score is

    .. math::

        \frac{\partial J}{\partial\hat{y}_i}
        =
        \begin{cases}
            -y_i / m & \text{if } y_i\hat{y}_i < 1, \\
            0 & \text{if } y_i\hat{y}_i \ge 1.
        \end{cases}

    At the kink :math:`y_i\hat{y}_i = 1` every value between :math:`-y_i/m`
    and 0 is a valid subgradient. This implementation returns 0 there, so
    an example sitting exactly on the margin contributes no update.

    Targets in ``{0, 1}`` are mapped by :math:`y \mapsto 2y - 1` before
    the formula is applied. The mathematical object is always the signed
    label.
    """

    def __call__(self, y_pred: ArrayLike, y_true: ArrayLike) -> float:
        """Return :meth:`loss`."""

        return self.loss(y_pred, y_true)

    @staticmethod
    def loss(y_pred: ArrayLike, y_true: ArrayLike) -> float:
        r"""Return :math:`\frac{1}{m}\sum_i\max(0, 1 - y_i\hat{y}_i)`.

        Parameters
        ----------
        y_pred : array-like
            Real-valued scores.
        y_true : array-like
            Labels in ``{0, 1}`` or ``{-1, +1}``.

        Returns
        -------
        float
            Mean hinge.
        """

        predicted, observed = _align(y_pred, y_true)
        signed = _signed_labels(observed)
        margins = signed * predicted
        # ``maximum`` is the pointwise hinge. The mean is the empirical risk.
        return float(np.mean(np.maximum(0.0, 1.0 - margins)))

    @staticmethod
    def gradient(y_pred: ArrayLike, y_true: ArrayLike) -> NDArray[np.float64]:
        r"""Return a subgradient :math:`\partial J/\partial\hat{y}`.

        Parameters
        ----------
        y_pred, y_true : array-like
            Scores and labels in ``{0, 1}`` or ``{-1, +1}``.

        Returns
        -------
        ndarray
            Subgradient in score space, same shape as ``y_pred``.
        """

        predicted, observed = _align(y_pred, y_true)
        signed = _signed_labels(observed)
        margins = signed * predicted
        violated = margins < 1.0
        subgradient = np.zeros_like(predicted)
        # The mask selects every violated margin at once. The contribution
        # of one violated example to the averaged objective is -y / m.
        subgradient[violated] = -signed[violated] / predicted.size
        return subgradient


def mean_squared_error(y_pred: ArrayLike, y_true: ArrayLike) -> float:
    """Return :meth:`MeanSquaredError.loss`."""

    return MeanSquaredError.loss(y_pred, y_true)


def mean_squared_error_gradient(
    y_pred: ArrayLike,
    y_true: ArrayLike,
) -> NDArray[np.float64]:
    """Return :meth:`MeanSquaredError.gradient`."""

    return MeanSquaredError.gradient(y_pred, y_true)


def binary_cross_entropy(y_pred: ArrayLike, y_true: ArrayLike) -> float:
    """Return :meth:`BinaryCrossEntropy.loss`."""

    return BinaryCrossEntropy.loss(y_pred, y_true)


def binary_cross_entropy_gradient(
    y_pred: ArrayLike,
    y_true: ArrayLike,
) -> NDArray[np.float64]:
    """Return :meth:`BinaryCrossEntropy.gradient`."""

    return BinaryCrossEntropy.gradient(y_pred, y_true)


def binary_cross_entropy_with_logits(logits: ArrayLike, y_true: ArrayLike) -> float:
    """Return :meth:`BinaryCrossEntropy.loss_with_logits`."""

    return BinaryCrossEntropy.loss_with_logits(logits, y_true)


def binary_cross_entropy_with_logits_gradient(
    logits: ArrayLike,
    y_true: ArrayLike,
) -> NDArray[np.float64]:
    """Return :meth:`BinaryCrossEntropy.gradient_with_logits`."""

    return BinaryCrossEntropy.gradient_with_logits(logits, y_true)


def softmax_cross_entropy(logits: ArrayLike, y_true: ArrayLike) -> float:
    """Return :meth:`SoftmaxCrossEntropy.loss`."""

    return SoftmaxCrossEntropy.loss(logits, y_true)


def softmax_cross_entropy_gradient(
    logits: ArrayLike,
    y_true: ArrayLike,
) -> NDArray[np.float64]:
    """Return :meth:`SoftmaxCrossEntropy.gradient`."""

    return SoftmaxCrossEntropy.gradient(logits, y_true)


def hinge_loss(y_pred: ArrayLike, y_true: ArrayLike) -> float:
    """Return :meth:`HingeLoss.loss`."""

    return HingeLoss.loss(y_pred, y_true)


def hinge_loss_gradient(y_pred: ArrayLike, y_true: ArrayLike) -> NDArray[np.float64]:
    """Return :meth:`HingeLoss.gradient`."""

    return HingeLoss.gradient(y_pred, y_true)


def _as_logit_matrix(logits: ArrayLike) -> NDArray[np.float64]:
    """Return scores as a finite array of shape ``(n_samples, n_classes)``."""

    scores = np.asarray(logits, dtype=np.float64)
    if scores.ndim != 2 or scores.shape[0] < 1 or scores.shape[1] < 1:
        raise ValueError(
            "Logits must have shape (n_samples, n_classes); "
            f"got {scores.shape}."
        )
    if not np.isfinite(scores).all():
        raise ValueError("Logits must be finite.")
    return scores


def _classification_targets(
    y_true: ArrayLike,
    n_samples: int,
    n_classes: int,
) -> NDArray[np.int64] | NDArray[np.float64]:
    """Return integer codes ``(n_samples,)`` or a distribution ``(n_samples, n_classes)``."""

    targets = np.asarray(y_true)
    if targets.ndim == 2 and targets.shape[1] == 1:
        targets = targets.reshape(-1)
    if targets.ndim == 1:
        if targets.shape[0] != n_samples:
            raise ValueError(
                "Integer labels must have one entry per row. "
                f"Got {targets.shape[0]} labels and {n_samples} rows."
            )
        if targets.dtype.kind == "f" and not np.all(targets == np.floor(targets)):
            raise ValueError("Class indices must be integers.")
        codes = targets.astype(np.int64, copy=False)
        if np.any(codes < 0) or np.any(codes >= n_classes):
            raise ValueError(
                f"Class indices must lie in 0..{n_classes - 1}."
            )
        return codes
    if targets.ndim == 2 and targets.shape == (n_samples, n_classes):
        distribution = targets.astype(np.float64, copy=False)
        if not np.isfinite(distribution).all():
            raise ValueError("Target distributions must be finite.")
        if np.any(distribution < 0.0):
            raise ValueError("Target distributions must be nonnegative.")
        row_sums = distribution.sum(axis=1)
        if not np.allclose(row_sums, 1.0):
            raise ValueError("Each target row must sum to 1.")
        return distribution
    raise ValueError(
        "Targets must have shape (n_samples,) or (n_samples, n_classes). "
        f"Got {targets.shape}."
    )


def _align(
    y_pred: ArrayLike,
    y_true: ArrayLike,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Cast both arrays to float64 and reshape targets onto predictions.

    A target of shape ``(m,)`` may be paired with predictions of shape
    ``(m, 1)`` when both contain the same number of elements. Any other
    disagreement is an error: silently broadcasting a shorter target would
    repeat labels and change the empirical risk.
    """

    predicted = np.asarray(y_pred, dtype=np.float64)
    observed = np.asarray(y_true, dtype=np.float64)
    if predicted.shape != observed.shape:
        if predicted.size != observed.size:
            raise ValueError(
                "Predictions and targets must contain the same number of "
                f"elements. Got shapes {predicted.shape} and {observed.shape}."
            )
        observed = np.reshape(observed, predicted.shape)
    if predicted.size == 0:
        raise ValueError("A loss is undefined on an empty sample.")
    if not np.isfinite(predicted).all() or not np.isfinite(observed).all():
        raise ValueError("Loss inputs must be finite.")
    return predicted, observed


def _check_unit_interval(
    values: NDArray[np.float64],
    name: str,
    *,
    closed: bool,
) -> None:
    """Require values in ``[0, 1]`` or in ``(0, 1)``."""

    if closed:
        valid = (values >= 0.0) & (values <= 1.0)
        interval = "[0, 1]"
    else:
        valid = (values > 0.0) & (values < 1.0)
        interval = "(0, 1)"
    if not np.all(valid):
        raise ValueError(f"Bernoulli {name} must lie in {interval}.")


def _signed_labels(y_true: NDArray[np.float64]) -> NDArray[np.float64]:
    """Map ``{0, 1}`` labels to ``{-1, +1}`` and pass signed labels through."""

    labels = set(np.unique(y_true).tolist())
    if labels <= {-1.0, 0.0, 1.0} and -1.0 in labels and 0.0 in labels:
        raise ValueError(
            "Hinge targets must be entirely in {0, 1} or entirely in {-1, +1}."
        )
    if labels <= {0.0, 1.0}:
        # 0 -> -1 and 1 -> +1. The affine map is the unique one that sends
        # the probability labels onto the signed labels the hinge uses.
        return 2.0 * y_true - 1.0
    if labels <= {-1.0, 1.0}:
        return y_true
    raise ValueError(
        "Hinge targets must be in {0, 1} or {-1, +1}. "
        f"Observed {sorted(labels)}."
    )
