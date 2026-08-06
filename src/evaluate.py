"""Reusable model-evaluation and report-generation functions."""

import json
from pathlib import Path
from typing import Any, Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def _to_numpy(
    values: Any,
) -> np.ndarray:
    """Convert supported array-like values into a NumPy array."""

    return np.asarray(
        values
    )


def validate_binary_evaluation_inputs(
    y_true: Any,
    y_pred: Any,
    y_probability: Any,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Validate and normalize binary classification inputs."""

    true_array = _to_numpy(
        y_true
    )

    prediction_array = _to_numpy(
        y_pred
    )

    probability_array = _to_numpy(
        y_probability
    )

    if not (
        len(true_array)
        == len(prediction_array)
        == len(probability_array)
    ):
        raise ValueError(
            "y_true, y_pred, and y_probability "
            "must have equal lengths."
        )

    if len(true_array) == 0:
        raise ValueError(
            "Evaluation inputs cannot be empty."
        )

    if not set(
        np.unique(
            true_array
        )
    ).issubset(
        {
            0,
            1,
        }
    ):
        raise ValueError(
            "y_true must contain binary values 0 and 1."
        )

    if not set(
        np.unique(
            prediction_array
        )
    ).issubset(
        {
            0,
            1,
        }
    ):
        raise ValueError(
            "y_pred must contain binary values 0 and 1."
        )

    if np.isnan(
        probability_array
    ).any():
        raise ValueError(
            "y_probability contains missing values."
        )

    if (
        (
            probability_array
            < 0
        ).any()
        or (
            probability_array
            > 1
        ).any()
    ):
        raise ValueError(
            "Probabilities must be between 0 and 1."
        )

    return (
        true_array,
        prediction_array,
        probability_array,
    )


def calculate_classification_metrics(
    y_true: Any,
    y_pred: Any,
    y_probability: Any,
) -> dict[str, float | int]:
    """
    Calculate binary classification metrics.

    Parameters
    ----------
    y_true:
        Actual target values.

    y_pred:
        Predicted class labels.

    y_probability:
        Predicted positive-class probabilities.

    Returns
    -------
    dict[str, float | int]
        Metrics and confusion-matrix counts.
    """

    (
        true_array,
        prediction_array,
        probability_array,
    ) = validate_binary_evaluation_inputs(
        y_true,
        y_pred,
        y_probability,
    )

    matrix = confusion_matrix(
        true_array,
        prediction_array,
        labels=[
            0,
            1,
        ],
    )

    true_negative, false_positive, false_negative, true_positive = (
        matrix.ravel()
    )

    metrics: dict[str, float | int] = {
        "accuracy": float(
            accuracy_score(
                true_array,
                prediction_array,
            )
        ),
        "precision": float(
            precision_score(
                true_array,
                prediction_array,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                true_array,
                prediction_array,
                zero_division=0,
            )
        ),
        "f1": float(
            f1_score(
                true_array,
                prediction_array,
                zero_division=0,
            )
        ),
        "roc_auc": float(
            roc_auc_score(
                true_array,
                probability_array,
            )
        ),
        "average_precision": float(
            average_precision_score(
                true_array,
                probability_array,
            )
        ),
        "true_negative": int(
            true_negative
        ),
        "false_positive": int(
            false_positive
        ),
        "false_negative": int(
            false_negative
        ),
        "true_positive": int(
            true_positive
        ),
    }

    return metrics


def save_metrics_json(
    metrics: Mapping[str, Any],
    output_path: Path,
) -> None:
    """Save metrics or configuration values to JSON."""

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        mode="w",
        encoding="utf-8",
    ) as file:
        json.dump(
            dict(
                metrics
            ),
            file,
            indent=4,
        )


def save_predictions_csv(
    features: pd.DataFrame,
    y_true: pd.Series,
    y_pred: np.ndarray,
    y_probability: np.ndarray,
    output_path: Path,
) -> None:
    """Save input rows with targets and model predictions."""

    prediction_dataframe = (
        features
        .copy()
        .reset_index(
            drop=False
        )
        .rename(
            columns={
                "index": "source_index",
            }
        )
    )

    prediction_dataframe[
        "actual_target"
    ] = (
        y_true
        .reset_index(
            drop=True
        )
    )

    prediction_dataframe[
        "predicted_target"
    ] = y_pred

    prediction_dataframe[
        "positive_probability"
    ] = y_probability

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    prediction_dataframe.to_csv(
        output_path,
        index=False,
    )


def plot_confusion_matrix(
    y_true: Any,
    y_pred: Any,
    output_path: Path,
    title: str,
) -> None:
    """Create and save a binary confusion matrix."""

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=[
            0,
            1,
        ],
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=[
            "Class 0",
            "Class 1",
        ],
    )

    figure, axis = plt.subplots(
        figsize=(
            7,
            6,
        )
    )

    display.plot(
        ax=axis,
        values_format="d",
    )

    axis.set_title(
        title
    )

    figure.tight_layout()

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )


def plot_roc_curves(
    model_probabilities: Mapping[str, Any],
    y_true: Any,
    output_path: Path,
) -> None:
    """Create one ROC curve containing one or more models."""

    figure, axis = plt.subplots(
        figsize=(
            8,
            6,
        )
    )

    for model_name, probability_values in (
        model_probabilities.items()
    ):
        false_positive_rate, true_positive_rate, _ = (
            roc_curve(
                y_true,
                probability_values,
            )
        )

        auc_value = roc_auc_score(
            y_true,
            probability_values,
        )

        axis.plot(
            false_positive_rate,
            true_positive_rate,
            label=(
                f"{model_name} "
                f"(AUC = {auc_value:.3f})"
            ),
        )

    axis.plot(
        [
            0,
            1,
        ],
        [
            0,
            1,
        ],
        linestyle="--",
        label="Random classifier",
    )

    axis.set_xlabel(
        "False Positive Rate"
    )

    axis.set_ylabel(
        "True Positive Rate"
    )

    axis.set_title(
        "ROC Curve"
    )

    axis.legend()

    figure.tight_layout()

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )


def plot_precision_recall_curves(
    model_probabilities: Mapping[str, Any],
    y_true: Any,
    output_path: Path,
) -> None:
    """Create one precision-recall curve for one or more models."""

    figure, axis = plt.subplots(
        figsize=(
            8,
            6,
        )
    )

    for model_name, probability_values in (
        model_probabilities.items()
    ):
        precision_values, recall_values, _ = (
            precision_recall_curve(
                y_true,
                probability_values,
            )
        )

        average_precision = (
            average_precision_score(
                y_true,
                probability_values,
            )
        )

        axis.plot(
            recall_values,
            precision_values,
            label=(
                f"{model_name} "
                f"(AP = {average_precision:.3f})"
            ),
        )

    positive_rate = float(
        np.mean(
            y_true
        )
    )

    axis.axhline(
        positive_rate,
        linestyle="--",
        label=(
            "Positive-class prevalence "
            f"({positive_rate:.3f})"
        ),
    )

    axis.set_xlabel(
        "Recall"
    )

    axis.set_ylabel(
        "Precision"
    )

    axis.set_title(
        "Precision-Recall Curve"
    )

    axis.legend()

    figure.tight_layout()

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )
