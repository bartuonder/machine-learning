"""Scores for classification decisions and real-valued predictions.

Classification scores are functions of the confusion counts. Regression
uses the coefficient of determination. ROC-AUC is the Mann-Whitney form
of the area under the receiver operating characteristic, which is a
probability statement about pairs of examples and does not require a
threshold.
"""

from __future__ import annotations

from typing import Any, Literal

from numpy.typing import ArrayLike, NDArray

import numpy as np

Average = Literal["binary", "macro", "micro", "weighted"]


def accuracy_score(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Return the fraction of rows whose predicted label equals the target.

    For a single-label problem this is
    :math:`(TP + TN) / n` in the binary case and, more generally,
    :math:`\\mathrm{trace}(C) / n` where :math:`C` is the confusion
    matrix. It treats every class error as equally costly.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Observed labels.
    y_pred : array-like of shape (n_samples,)
        Predicted labels.

    Returns
    -------
    float
        Accuracy in ``[0, 1]``.
    """

    observed, predicted = _align_labels(y_true, y_pred)
    # Boolean mean is the empirical measure of the agreement event.
    return float(np.mean(observed == predicted))


def confusion_matrix(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    labels: ArrayLike | None = None,
) -> NDArray[np.int64]:
    r"""Count co-occurrences of true and predicted labels.

    With an indicator matrix :math:`I^{\mathrm{true}}_{ic} = 1\{y_i = c\}`
    and the analogous matrix for predictions,

    .. math::

        C = (I^{\mathrm{true}})^\top I^{\mathrm{pred}}.

    Entry :math:`C_{ck}` is the number of rows whose true label is class
    ``c`` and whose predicted label is class ``k``. Rows follow the order
    of ``labels``.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Observed labels.
    y_pred : array-like of shape (n_samples,)
        Predicted labels.
    labels : array-like of shape (n_classes,) or None, default=None
        Class order. The default is the sorted unique union of the two
        arguments. A label that appears in the data and is absent from
        ``labels`` is an error, because dropping it would change every
        rate computed from ``C``.

    Returns
    -------
    ndarray of shape (n_classes, n_classes)
        Integer confusion counts.
    """

    observed, predicted = _align_labels(y_true, y_pred)
    if labels is None:
        label_list = np.unique(np.concatenate([observed, predicted]))
    else:
        label_list = np.asarray(labels)
        if label_list.ndim != 1:
            raise ValueError("labels must be one-dimensional.")
        if np.unique(label_list).size != label_list.size:
            raise ValueError("labels must be unique.")
    # Broadcasting the comparison against every class builds the indicator
    # matrices in the product above. Shape (n_samples, n_classes).
    true_indicator = (observed[:, None] == label_list[None, :]).astype(np.float64)
    pred_indicator = (predicted[:, None] == label_list[None, :]).astype(np.float64)
    if true_indicator.sum() != observed.size or pred_indicator.sum() != predicted.size:
        raise ValueError(
            "Every sample needs exactly one label from `labels`. "
            "A missing or duplicated label would drop or double-count rows."
        )
    # Contract the sample axis. float64 keeps counts exact through 2^53.
    counts = true_indicator.T @ pred_indicator
    return counts.astype(np.int64)


def precision_score(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    *,
    pos_label: Any = 1,
    average: Average | None = "binary",
    labels: ArrayLike | None = None,
    zero_division: float = 0.0,
) -> float | NDArray[np.float64]:
    r"""Return precision, :math:`TP / (TP + FP)`.

    Precision is the empirical probability that a row predicted as the
    class of interest actually belongs to it. It ignores false negatives,
    so a classifier that rarely fires can score well while missing most
    positive rows.

    Parameters
    ----------
    y_true, y_pred : array-like of shape (n_samples,)
        Observed and predicted labels.
    pos_label : label, default=1
        Class treated as positive when ``average="binary"``.
    average : {"binary", "macro", "micro", "weighted"} or None
        ``"binary"`` scores ``pos_label`` and requires at most two labels
        in the union of ``y_true`` and ``y_pred``. ``"macro"`` is the
        unweighted mean of per-class precisions. ``"weighted"`` weights
        those precisions by the number of true rows of each class.
        ``"micro"`` pools TP and FP across classes before dividing.
        ``None`` returns one precision per class.
    labels : array-like or None, default=None
        Class order used for every average other than ``"binary"``.
    zero_division : float, default=0.0
        Value used when a denominator is zero.

    Returns
    -------
    float or ndarray of shape (n_classes,)
        Precision.
    """

    return _score_from_counts(
        "precision",
        y_true,
        y_pred,
        pos_label=pos_label,
        average=average,
        labels=labels,
        zero_division=zero_division,
    )


def recall_score(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    *,
    pos_label: Any = 1,
    average: Average | None = "binary",
    labels: ArrayLike | None = None,
    zero_division: float = 0.0,
) -> float | NDArray[np.float64]:
    r"""Return recall, :math:`TP / (TP + FN)`.

    Recall is the empirical probability that a row of the class of
    interest is recovered by the classifier. It ignores false positives.

    Parameters match :func:`precision_score`.
    """

    return _score_from_counts(
        "recall",
        y_true,
        y_pred,
        pos_label=pos_label,
        average=average,
        labels=labels,
        zero_division=zero_division,
    )


def f1_score(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    *,
    pos_label: Any = 1,
    average: Average | None = "binary",
    labels: ArrayLike | None = None,
    zero_division: float = 0.0,
) -> float | NDArray[np.float64]:
    r"""Return the F1 score, the harmonic mean of precision and recall.

    .. math::

        F_1 = \frac{2 PR}{P + R}.

    The harmonic mean is small when either argument is small, so a model
    cannot hide a weak precision behind a strong recall. When
    :math:`P + R = 0` the score is ``zero_division``.

    Parameters match :func:`precision_score`. For a single-label problem,
    micro-averaged F1 equals accuracy: each row contributes one true
    positive when it is correct, and one false positive together with one
    false negative when it is not, so the pooled ratio is the accuracy.
    """

    return _score_from_counts(
        "f1",
        y_true,
        y_pred,
        pos_label=pos_label,
        average=average,
        labels=labels,
        zero_division=zero_division,
    )


def roc_auc_score(y_true: ArrayLike, y_score: ArrayLike) -> float:
    r"""Return the area under the receiver operating characteristic.

    Let a positive row and a negative row be drawn uniformly and
    independently. The AUC is the probability that the positive row
    receives the higher score, with ties counted as one half:

    .. math::

        \mathrm{AUC}
        = \frac{
            \sum_{i: y_i = 1} r_i - n_+(n_+ + 1) / 2
          }{n_+ n_-}.

    Here :math:`r_i` is the 1-based average rank of score :math:`s_i`
    (average ranks split ties evenly), :math:`n_+` is the number of
    positive rows, and :math:`n_-` is the number of negative rows. This
    is the normalized Mann-Whitney statistic. It equals
    :math:`\int_0^1 \mathrm{TPR}(\mathrm{FPR}^{-1}(t))\,dt`, the area
    under the ROC curve.

    Labels must be binary in ``{0, 1}`` or ``{-1, 1}``. The label ``1``
    is the positive class. Scores may be any real-valued ranking; only
    their order enters the statistic. A single class makes the
    probability undefined, and this function raises.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Binary labels.
    y_score : array-like of shape (n_samples,)
        Real-valued scores. Larger scores rank as more positive.

    Returns
    -------
    float
        AUC in ``[0, 1]``.
    """

    observed = np.asarray(y_true).reshape(-1)
    scores = np.asarray(y_score, dtype=np.float64).reshape(-1)
    if observed.shape != scores.shape:
        raise ValueError(
            "y_true and y_score must have the same length. "
            f"Got {observed.size} and {scores.size}."
        )
    if observed.size == 0:
        raise ValueError("ROC-AUC is undefined for an empty sample.")
    if not np.isfinite(scores).all():
        raise ValueError("y_score must contain only finite values.")
    positive = _as_positive_mask(observed)
    n_pos = int(positive.sum())
    n_neg = int(positive.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        raise ValueError("ROC-AUC is undefined when only one class is present.")
    ranks = _average_ranks(scores)
    # Sum of 1-based ranks of the positive class. Boolean indexing selects
    # those ranks in one pass; it does not loop over pairs.
    sum_positive_ranks = float(ranks[positive].sum())
    return (sum_positive_ranks - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def r2_score(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    r"""Return the coefficient of determination.

    .. math::

        R^2 = 1 - \frac{\sum_i (y_i - \hat{y}_i)^2}{\sum_i (y_i - \bar{y})^2}.

    The denominator is the residual sum of squares of the intercept-only
    model that predicts the sample mean. :math:`R^2 = 1` is a perfect
    fit, :math:`R^2 = 0` matches the mean predictor, and a negative value
    is worse than predicting the mean.

    When the targets are constant the denominator is zero. The score is
    then 1 if the predictions equal that constant, and 0 otherwise. A
    constant target makes the usual ratio undefined; this convention
    still reports whether the predictor recovered the only value present.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Observed targets.
    y_pred : array-like of shape (n_samples,)
        Predicted targets.

    Returns
    -------
    float
        Coefficient of determination.
    """

    observed, predicted = _align_floats(y_true, y_pred)
    residual = observed - predicted
    # v @ v is the squared Euclidean norm, one reduction instead of a
    # Python accumulation of squares.
    ss_residual = float(residual @ residual)
    centered = observed - observed.mean()
    ss_total = float(centered @ centered)
    if ss_total == 0.0:
        return 1.0 if ss_residual == 0.0 else 0.0
    return 1.0 - ss_residual / ss_total


def _score_from_counts(
    kind: Literal["precision", "recall", "f1"],
    y_true: ArrayLike,
    y_pred: ArrayLike,
    *,
    pos_label: Any,
    average: Average | None,
    labels: ArrayLike | None,
    zero_division: float,
) -> float | NDArray[np.float64]:
    if not np.isfinite(zero_division):
        raise ValueError("zero_division must be finite.")
    observed, predicted = _align_labels(y_true, y_pred)
    if average == "binary":
        return _binary_score(
            kind,
            observed,
            predicted,
            pos_label=pos_label,
            zero_division=float(zero_division),
        )
    if average not in ("macro", "micro", "weighted", None):
        raise ValueError(
            "average must be 'binary', 'macro', 'micro', 'weighted', or None."
        )

    if labels is None:
        label_list: ArrayLike = np.unique(np.concatenate([observed, predicted]))
    else:
        label_list = labels
    matrix = confusion_matrix(observed, predicted, labels=label_list)
    true_positive = np.diag(matrix).astype(np.float64)
    # Axis 0 sums rows, so it totals each predicted column: TP + FP.
    predicted_positive = matrix.sum(axis=0).astype(np.float64)
    # Axis 1 sums columns, so it totals each true row: TP + FN.
    support = matrix.sum(axis=1).astype(np.float64)
    false_positive = predicted_positive - true_positive
    false_negative = support - true_positive

    if average == "micro":
        return _micro_score(
            kind,
            true_positive=true_positive,
            false_positive=false_positive,
            false_negative=false_negative,
            zero_division=float(zero_division),
        )

    precision = _safe_div(true_positive, true_positive + false_positive, zero_division)
    recall = _safe_div(true_positive, true_positive + false_negative, zero_division)
    f1 = _safe_div(2.0 * precision * recall, precision + recall, zero_division)
    per_class = {"precision": precision, "recall": recall, "f1": f1}[kind]
    if average is None:
        return per_class
    if average == "macro":
        return float(np.mean(per_class))
    total = float(support.sum())
    if total == 0.0:
        return float(zero_division)
    # Weight by the true-class count so a rare class does not move the
    # average as much as a frequent one. This is a support-weighted mean
    # of the per-class scores, which is a different functional from micro
    # averaging (micro pools the counts first).
    return float(np.sum(per_class * support) / total)


def _binary_score(
    kind: Literal["precision", "recall", "f1"],
    y_true: NDArray[np.generic],
    y_pred: NDArray[np.generic],
    *,
    pos_label: Any,
    zero_division: float,
) -> float:
    present = np.unique(np.concatenate([y_true, y_pred]))
    if present.size > 2:
        raise ValueError(
            "average='binary' requires at most two labels. "
            f"Observed {present.tolist()}. Use average='macro', "
            "'micro', or 'weighted'."
        )
    if pos_label not in set(present.tolist()):
        raise ValueError(
            f"pos_label={pos_label!r} is not present in y_true or y_pred. "
            f"Observed labels: {present.tolist()}."
        )
    true_positive_mask = y_true == pos_label
    pred_positive_mask = y_pred == pos_label
    true_positive = float(np.sum(true_positive_mask & pred_positive_mask))
    false_positive = float(np.sum(~true_positive_mask & pred_positive_mask))
    false_negative = float(np.sum(true_positive_mask & ~pred_positive_mask))
    precision = _safe_div(
        np.asarray(true_positive),
        np.asarray(true_positive + false_positive),
        zero_division,
    )
    recall = _safe_div(
        np.asarray(true_positive),
        np.asarray(true_positive + false_negative),
        zero_division,
    )
    if kind == "precision":
        return float(precision)
    if kind == "recall":
        return float(recall)
    f1 = _safe_div(
        np.asarray(2.0 * float(precision) * float(recall)),
        np.asarray(float(precision) + float(recall)),
        zero_division,
    )
    return float(f1)


def _micro_score(
    kind: Literal["precision", "recall", "f1"],
    *,
    true_positive: NDArray[np.float64],
    false_positive: NDArray[np.float64],
    false_negative: NDArray[np.float64],
    zero_division: float,
) -> float:
    tp_sum = float(true_positive.sum())
    fp_sum = float(false_positive.sum())
    fn_sum = float(false_negative.sum())
    precision = float(_safe_div(np.asarray(tp_sum), np.asarray(tp_sum + fp_sum), zero_division))
    recall = float(_safe_div(np.asarray(tp_sum), np.asarray(tp_sum + fn_sum), zero_division))
    if kind == "precision":
        return precision
    if kind == "recall":
        return recall
    return float(
        _safe_div(
            np.asarray(2.0 * precision * recall),
            np.asarray(precision + recall),
            zero_division,
        )
    )


def _safe_div(
    numerator: NDArray[np.float64],
    denominator: NDArray[np.float64],
    zero_division: float,
) -> NDArray[np.float64]:
    """Divide, writing ``zero_division`` wherever the denominator vanishes."""

    numerator = np.asarray(numerator, dtype=np.float64)
    denominator = np.asarray(denominator, dtype=np.float64)
    result = np.divide(
        numerator,
        denominator,
        out=np.full(np.broadcast(numerator, denominator).shape, zero_division, dtype=np.float64),
        where=denominator != 0.0,
    )
    return result


def _average_ranks(scores: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return 1-based ranks, averaging the ranks inside every tied block.

    ``np.unique`` sorts the distinct scores and returns an inverse index
    from each row back to its distinct value. A tied block that occupies
    1-based positions ``start`` through ``end`` receives the midpoint of
    that interval. Looking the midpoint up with the inverse index writes
    a rank for every row without a Python loop over the sample.
    """

    _values, inverse, counts = np.unique(scores, return_inverse=True, return_counts=True)
    ends = np.cumsum(counts)
    starts = ends - counts + 1
    average = (starts.astype(np.float64) + ends.astype(np.float64)) / 2.0
    return average[inverse]


def _as_positive_mask(y_true: NDArray[np.generic]) -> NDArray[np.bool_]:
    """Map ``{0, 1}`` or ``{-1, 1}`` onto a boolean mask for the positive class."""

    if y_true.dtype == bool:
        return y_true.astype(bool, copy=False)
    labels = set(np.unique(y_true).tolist())
    if not labels or not labels <= {0, 1, 0.0, 1.0, False, True}:
        if labels <= {-1, 1, -1.0, 1.0}:
            return y_true == 1
        raise ValueError(
            "roc_auc_score expects binary labels in {0, 1} or {-1, 1}. "
            f"Observed {sorted(labels, key=str)}."
        )
    return y_true == 1


def _align_labels(
    y_true: ArrayLike,
    y_pred: ArrayLike,
) -> tuple[NDArray[np.generic], NDArray[np.generic]]:
    observed = np.asarray(y_true).reshape(-1)
    predicted = np.asarray(y_pred).reshape(-1)
    if observed.shape != predicted.shape:
        raise ValueError(
            "y_true and y_pred must have the same length. "
            f"Got {observed.size} and {predicted.size}."
        )
    if observed.size == 0:
        raise ValueError("The score is undefined for an empty sample.")
    return observed, predicted


def _align_floats(
    y_true: ArrayLike,
    y_pred: ArrayLike,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    observed = np.asarray(y_true, dtype=np.float64).reshape(-1)
    predicted = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    if observed.shape != predicted.shape:
        raise ValueError(
            "y_true and y_pred must have the same length. "
            f"Got {observed.size} and {predicted.size}."
        )
    if observed.size == 0:
        raise ValueError("The score is undefined for an empty sample.")
    if not np.isfinite(observed).all() or not np.isfinite(predicted).all():
        raise ValueError("Regression metrics require finite values.")
    return observed, predicted
