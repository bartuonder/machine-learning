"""First-order parameter updates and a central-difference gradient check.

:class:`SGD` and :class:`Adam` do not evaluate a loss and do not sample
rows. The caller supplies a gradient of the empirical risk at the current
parameters. That split keeps the statistical definition of the gradient
in the model and the recurrence that consumes it in the optimizer.

Momentum and Adam keep state between calls. One instance follows one
parameter array. Call :meth:`reset` before pointing the same instance at
a new parameter vector.
"""

from __future__ import annotations

from collections.abc import Callable

from numpy.typing import ArrayLike, NDArray

import numpy as np


class SGD:
    r"""Heavy-ball update on a Euclidean parameter.

    Given a gradient :math:`g_t = \nabla J(\theta_{t-1})` and velocity
    :math:`v_0 = 0`,

    .. math::

        v_t = \mu v_{t-1} + g_t,

    .. math::

        \theta_t = \theta_{t-1} - \alpha v_t.

    :math:`\alpha > 0` is the learning rate and :math:`\mu \in [0, 1)` is
    the momentum. With :math:`\mu = 0` the velocity is the gradient and
    the update is plain gradient descent, or stochastic gradient descent
    when :math:`g_t` itself is a mini-batch estimate.

    Optional weight decay adds the gradient of
    :math:`\frac{\lambda}{2}\lVert\theta\rVert_2^2` before the velocity
    update:

    .. math::

        g_t \leftarrow g_t + \lambda\theta_{t-1}.

    This penalizes every coordinate, including a bias stored in the same
    vector. A model that must leave its intercept unpenalized should fold
    that masked penalty into the gradient it supplies and leave
    ``weight_decay`` at 0.

    Parameters
    ----------
    learning_rate : float, default=0.01
        Step size :math:`\alpha`.
    momentum : float, default=0.0
        Heavy-ball coefficient :math:`\mu` in ``[0, 1)``.
    weight_decay : float, default=0.0
        Coupled penalty coefficient :math:`\lambda`.
    """

    def __init__(
        self,
        learning_rate: float = 0.01,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
    ) -> None:
        self.learning_rate = _positive_float("learning_rate", learning_rate)
        self.momentum = _unit_interval("momentum", momentum, upper_open=True)
        self.weight_decay = _nonnegative_float("weight_decay", weight_decay)
        self.n_steps = 0
        self._velocity: NDArray[np.float64] | None = None

    def step(self, param: NDArray[np.float64], grad: ArrayLike) -> None:
        """Move ``param`` in place along the heavy-ball direction.

        Parameters
        ----------
        param : ndarray
            Writable floating-point parameter. Updated in place.
        grad : array-like
            Gradient of the scalar objective at ``param``, same shape.
            This array is not modified.
        """

        parameter = _writable_parameter(param)
        gradient = _gradient_array(grad, parameter.shape)
        if self.weight_decay != 0.0:
            # λ θ is ∇(λ/2 ||θ||^2). Adding it here couples the penalty to
            # the same velocity as the data gradient.
            gradient = gradient + self.weight_decay * parameter
        if self.momentum == 0.0:
            parameter -= self.learning_rate * gradient
        else:
            if self._velocity is None:
                self._velocity = np.zeros_like(parameter)
            elif self._velocity.shape != parameter.shape:
                raise RuntimeError(
                    "SGD momentum state has a different shape from param. "
                    "Call reset() before reusing the optimizer."
                )
            # Scale the stored velocity, then add the new gradient. The
            # buffer allocated on the first step is reused after that.
            self._velocity *= self.momentum
            self._velocity += gradient
            parameter -= self.learning_rate * self._velocity
        self.n_steps += 1

    def reset(self) -> None:
        """Drop the velocity and the step counter."""

        self.n_steps = 0
        self._velocity = None

    def __repr__(self) -> str:
        return (
            f"SGD(learning_rate={self.learning_rate}, momentum={self.momentum}, "
            f"weight_decay={self.weight_decay})"
        )


class Adam:
    r"""Adam with the decoupled weight decay of AdamW.

    Adam maintains exponential moving averages of the gradient and of the
    entrywise squared gradient. With :math:`m_0 = 0`, :math:`v_0 = 0`,
    and :math:`\beta_1, \beta_2 \in [0, 1)`,

    .. math::

        m_t = \beta_1 m_{t-1} + (1 - \beta_1)\, g_t,

    .. math::

        v_t = \beta_2 v_{t-1} + (1 - \beta_2)\, g_t^{\odot 2}.

    Initializing both moments at zero biases them toward zero. Under a
    stationary-gradient calculation the bias factors are
    :math:`1 - \beta_1^t` and :math:`1 - \beta_2^t`, so the corrected
    moments are

    .. math::

        \hat{m}_t = \frac{m_t}{1 - \beta_1^t},
        \qquad
        \hat{v}_t = \frac{v_t}{1 - \beta_2^t}.

    The parameter step is an adaptive trust-region style move,

    .. math::

        \theta_t
        = \theta_{t-1}
          - \alpha\,
            \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}.

    The recursion stores :math:`(m_t, v_t)`, the biased moments. The hats
    are local to the step. Writing a hat back into the buffer would apply
    the bias correction a second time on the next call.

    At step 1 the correction is exact for any gradient:
    :math:`m_1 = (1-\beta_1) g_1` and :math:`\hat{m}_1 = g_1`.

    Weight decay, when nonzero, is the decoupled AdamW step

    .. math::

        \theta \leftarrow (1 - \alpha\lambda)\,\theta

    applied before the adaptive step. It is not added inside :math:`g_t`.
    Coupled decay would be divided by :math:`\sqrt{\hat{v}_t}` and would
    shrink coordinates with small second moments by more than the
    regularization coefficient says. Decoupled decay subtracts
    :math:`\alpha\lambda\theta` in parameter space, on every coordinate
    including a bias that shares this vector.

    On a deterministic full-batch gradient the moments are a diagonal
    preconditioner with bias correction. The stochastic analysis of Adam
    assumes noisy gradients; the algebra of the update is the same either
    way.

    Parameters
    ----------
    learning_rate : float, default=0.001
        Step size :math:`\alpha`.
    beta1 : float, default=0.9
        Decay of the first moment.
    beta2 : float, default=0.999
        Decay of the second moment.
    eps : float, default=1e-8
        Stabilizer :math:`\epsilon` inside the denominator.
    weight_decay : float, default=0.0
        Decoupled coefficient :math:`\lambda`.
    """

    def __init__(
        self,
        learning_rate: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        self.learning_rate = _positive_float("learning_rate", learning_rate)
        self.beta1 = _unit_interval("beta1", beta1, upper_open=True)
        self.beta2 = _unit_interval("beta2", beta2, upper_open=True)
        self.eps = _positive_float("eps", eps)
        self.weight_decay = _nonnegative_float("weight_decay", weight_decay)
        self.n_steps = 0
        self._moment: NDArray[np.float64] | None = None
        self._second_moment: NDArray[np.float64] | None = None

    def step(self, param: NDArray[np.float64], grad: ArrayLike) -> None:
        """Move ``param`` in place using bias-corrected moments.

        Parameters
        ----------
        param : ndarray
            Writable floating-point parameter. Updated in place.
        grad : array-like
            Gradient at ``param``, same shape. This array is not modified.
        """

        parameter = _writable_parameter(param)
        gradient = _gradient_array(grad, parameter.shape)
        if self._moment is None or self._second_moment is None:
            self._moment = np.zeros_like(parameter)
            self._second_moment = np.zeros_like(parameter)
        elif self._moment.shape != parameter.shape:
            raise RuntimeError(
                "Adam moment state has a different shape from param. "
                "Call reset() before reusing the optimizer."
            )

        self.n_steps += 1
        # In-place polynomial in the previous moment. The new gradient is
        # only read, so the caller's grad buffer stays intact.
        self._moment *= self.beta1
        self._moment += (1.0 - self.beta1) * gradient
        self._second_moment *= self.beta2
        self._second_moment += (1.0 - self.beta2) * np.square(gradient)

        # Local corrected copies. The buffers keep the biased moments
        # because that is the state the next exponential average expects.
        first_correction = 1.0 - self.beta1**self.n_steps
        second_correction = 1.0 - self.beta2**self.n_steps
        moment_hat = self._moment / first_correction
        second_hat = self._second_moment / second_correction

        if self.weight_decay != 0.0:
            parameter *= 1.0 - self.learning_rate * self.weight_decay
        parameter -= self.learning_rate * moment_hat / (np.sqrt(second_hat) + self.eps)

    def reset(self) -> None:
        """Drop both moments and the step counter."""

        self.n_steps = 0
        self._moment = None
        self._second_moment = None

    def __repr__(self) -> str:
        return (
            f"Adam(learning_rate={self.learning_rate}, beta1={self.beta1}, "
            f"beta2={self.beta2}, eps={self.eps}, weight_decay={self.weight_decay})"
        )


def finite_difference_gradient(
    loss_fn: Callable[[NDArray[np.float64]], float],
    theta: ArrayLike,
    eps: float = 1e-6,
) -> NDArray[np.float64]:
    r"""Approximate :math:`\nabla J(\theta)` by a central difference.

    .. math::

        \frac{\partial J}{\partial\theta_j}
        \approx
        \frac{J(\theta + \varepsilon e_j) - J(\theta - \varepsilon e_j)}{2\varepsilon}.

    The truncation error of the central difference is
    :math:`O(\varepsilon^2)`. ``eps=1e-6`` is a practical default in
    float64 for losses of moderate scale: smaller steps drown in roundoff,
    larger steps leave a visible quadratic bias.

    The loop runs over coordinates of :math:`\theta`. A partial derivative
    probes one coordinate at a time. It is not a loop over data rows; each
    probe calls ``loss_fn`` on a full parameter vector.

    Parameters
    ----------
    loss_fn : callable
        Maps a parameter array of the same shape as ``theta`` to a scalar
        loss.
    theta : array-like
        Point at which the gradient is estimated.
    eps : float, default=1e-6
        Positive perturbation.

    Returns
    -------
    ndarray
        Numerical gradient, same shape as ``theta``.
    """

    step = _positive_float("eps", eps)
    point = np.asarray(theta, dtype=np.float64)
    gradient = np.empty_like(point)
    flat_point = point.ravel()
    flat_gradient = gradient.ravel()
    for index in range(flat_point.size):
        basis = np.zeros_like(point)
        basis.ravel()[index] = step
        flat_gradient[index] = (loss_fn(point + basis) - loss_fn(point - basis)) / (
            2.0 * step
        )
    return gradient


def gradient_check(
    analytic: ArrayLike,
    numerical: ArrayLike,
) -> float:
    r"""Return the relative discrepancy between two gradients.

    .. math::

        \frac{\lVert g_{\mathrm{analytic}} - g_{\mathrm{numeric}}\rVert_2}
             {\lVert g_{\mathrm{analytic}}\rVert_2
              + \lVert g_{\mathrm{numeric}}\rVert_2}.

    A value around ``1e-7`` or smaller, for a central difference with a
    well-chosen step, is the usual numerical evidence that an analytic
    gradient matches the scalar loss. Both arguments are flattened before
    the norms are taken, so a column vector and a row vector of the same
    entries do not match unless their shapes already agree.

    Parameters
    ----------
    analytic, numerical : array-like
        Two estimates of the same gradient. Shapes must be equal.

    Returns
    -------
    float
        Relative discrepancy. Zero when both gradients are zero.
    """

    analytic_gradient = np.asarray(analytic, dtype=np.float64)
    numerical_gradient = np.asarray(numerical, dtype=np.float64)
    if analytic_gradient.shape != numerical_gradient.shape:
        raise ValueError(
            "Gradients must share a shape. "
            f"Got {analytic_gradient.shape} and {numerical_gradient.shape}."
        )
    difference = float(np.linalg.norm(analytic_gradient - numerical_gradient))
    scale = float(np.linalg.norm(analytic_gradient) + np.linalg.norm(numerical_gradient))
    if scale == 0.0:
        return 0.0
    return difference / scale


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


def _unit_interval(name: str, value: float, *, upper_open: bool) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.floating)):
        raise TypeError(f"{name} must be a real number.")
    number = float(value)
    upper_ok = number < 1.0 if upper_open else number <= 1.0
    if not np.isfinite(number) or number < 0.0 or not upper_ok:
        bracket = "[0, 1)" if upper_open else "[0, 1]"
        raise ValueError(f"{name} must lie in {bracket}.")
    return number


def _writable_parameter(param: NDArray[np.float64]) -> NDArray[np.float64]:
    if not isinstance(param, np.ndarray):
        raise TypeError("param must be a NumPy array so the step can update it in place.")
    if not np.issubdtype(param.dtype, np.floating):
        raise TypeError("param must be a floating-point array.")
    if not param.flags.writeable:
        raise ValueError("param must be writable; the update is in place.")
    return param


def _gradient_array(grad: ArrayLike, shape: tuple[int, ...]) -> NDArray[np.float64]:
    gradient = np.asarray(grad, dtype=np.float64)
    if gradient.shape != shape:
        raise ValueError(
            f"grad shape {gradient.shape} does not match param shape {shape}."
        )
    if not np.isfinite(gradient).all():
        raise ValueError("grad must contain only finite values.")
    return gradient
