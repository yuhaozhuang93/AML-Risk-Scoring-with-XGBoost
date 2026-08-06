"""Evaluate thresholds and select one using an explicit business rule."""

from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd

from src.config import (
    MINIMUM_RECALL,
    SELECTED_THRESHOLD_PATH,
    THRESHOLD_RESULTS_PATH,
    XGBOOST_MODEL_PATH,
    create_project_directories,
)
from src.data_loader import load_bank_data
from src.evaluate import (
    calculate_classification_metrics,
    save_metrics_json,
)
from src.preprocessing import (
    create_features_and_target,
    split_training_and_test_data,
)


DEFAULT_THRESHOLD_VALUES = (
    0.20,
    0.25,
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
    0.75,
    0.80,
)


def validate_threshold(
    threshold: float,
) -> float:
    """Validate and return a probability threshold."""

    numeric_threshold = float(
        threshold
    )

    if not (
        0.0
        <= numeric_threshold
        <= 1.0
    ):
        raise ValueError(
            "Threshold must be between "
            "0 and 1."
        )

    return numeric_threshold


def predictions_from_probability(
    positive_probability: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """Convert positive-class probabilities to binary decisions."""

    validated_threshold = validate_threshold(
        threshold
    )

    probabilities = np.asarray(
        positive_probability
    )

    if np.isnan(
        probabilities
    ).any():
        raise ValueError(
            "Probability values contain NaN."
        )

    return (
        probabilities
        >= validated_threshold
    ).astype(
        int
    )


def evaluate_thresholds(
    y_true: Any,
    positive_probability: np.ndarray,
    thresholds: Iterable[float],
) -> pd.DataFrame:
    """Evaluate classification metrics for each threshold."""

    threshold_results: list[
        dict[str, Any]
    ] = []

    for threshold in thresholds:
        validated_threshold = (
            validate_threshold(
                threshold
            )
        )

        threshold_prediction = (
            predictions_from_probability(
                positive_probability=(
                    positive_probability
                ),
                threshold=(
                    validated_threshold
                ),
            )
        )

        metrics = (
            calculate_classification_metrics(
                y_true=y_true,
                y_pred=threshold_prediction,
                y_probability=(
                    positive_probability
                ),
            )
        )

        metrics[
            "threshold"
        ] = validated_threshold

        metrics[
            "predicted_positive_count"
        ] = int(
            threshold_prediction.sum()
        )

        threshold_results.append(
            metrics
        )

    if not threshold_results:
        raise ValueError(
            "At least one threshold is required."
        )

    threshold_dataframe = (
        pd.DataFrame(
            threshold_results
        )
        .sort_values(
            by="threshold",
        )
        .reset_index(
            drop=True
        )
    )

    ordered_columns = [
        "threshold",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "average_precision",
        "true_negative",
        "false_positive",
        "false_negative",
        "true_positive",
        "predicted_positive_count",
    ]

    return threshold_dataframe[
        ordered_columns
    ]


def select_threshold_by_recall(
    threshold_dataframe: pd.DataFrame,
    minimum_recall: float = MINIMUM_RECALL,
) -> dict[str, Any]:
    """
    Select the highest-precision threshold meeting minimum recall.

    Ties are resolved by higher F1 and then higher threshold.
    """

    if not (
        0.0
        <= minimum_recall
        <= 1.0
    ):
        raise ValueError(
            "minimum_recall must be between "
            "0 and 1."
        )

    eligible_thresholds = (
        threshold_dataframe[
            threshold_dataframe[
                "recall"
            ]
            >= minimum_recall
        ]
    )

    if eligible_thresholds.empty:
        best_available_row = (
            threshold_dataframe
            .sort_values(
                by=[
                    "recall",
                    "precision",
                    "f1",
                ],
                ascending=[
                    False,
                    False,
                    False,
                ],
            )
            .iloc[
                0
            ]
        )

        raise ValueError(
            "No threshold satisfies the minimum "
            f"recall requirement of {minimum_recall:.3f}. "
            "The highest available recall is "
            f"{best_available_row['recall']:.3f} "
            "at threshold "
            f"{best_available_row['threshold']:.3f}."
        )

    selected_row = (
        eligible_thresholds
        .sort_values(
            by=[
                "precision",
                "f1",
                "threshold",
            ],
            ascending=[
                False,
                False,
                False,
            ],
        )
        .iloc[
            0
        ]
    )

    return {
        key: (
            int(
                value
            )
            if key
            in {
                "true_negative",
                "false_positive",
                "false_negative",
                "true_positive",
                "predicted_positive_count",
            }
            else float(
                value
            )
        )
        for key, value in (
            selected_row.to_dict().items()
        )
    }


def tune_xgboost_threshold() -> dict[str, Any]:
    """Load the saved model, evaluate thresholds, and save results."""

    create_project_directories()

    if not XGBOOST_MODEL_PATH.exists():
        raise FileNotFoundError(
            "XGBoost model was not found. "
            "Run python -m src.train_xgboost first."
        )

    pipeline = joblib.load(
        XGBOOST_MODEL_PATH
    )

    dataframe = load_bank_data()

    features, target = (
        create_features_and_target(
            dataframe
        )
    )

    (
        _,
        X_test,
        _,
        y_test,
    ) = split_training_and_test_data(
        features,
        target,
    )

    positive_probability = (
        pipeline
        .predict_proba(
            X_test
        )[
            :,
            1,
        ]
    )

    threshold_dataframe = (
        evaluate_thresholds(
            y_true=y_test,
            positive_probability=(
                positive_probability
            ),
            thresholds=(
                DEFAULT_THRESHOLD_VALUES
            ),
        )
    )

    threshold_dataframe.to_csv(
        THRESHOLD_RESULTS_PATH,
        index=False,
    )

    selected_threshold = (
        select_threshold_by_recall(
            threshold_dataframe=(
                threshold_dataframe
            ),
            minimum_recall=MINIMUM_RECALL,
        )
    )

    selected_threshold[
        "selection_rule"
    ] = (
        "Highest precision among thresholds "
        f"with recall >= {MINIMUM_RECALL:.2f}; "
        "ties resolved by F1 and threshold."
    )

    save_metrics_json(
        selected_threshold,
        SELECTED_THRESHOLD_PATH,
    )

    return {
        "threshold_results": (
            threshold_dataframe
        ),
        "selected_threshold": (
            selected_threshold
        ),
    }


if __name__ == "__main__":
    tuning_result = (
        tune_xgboost_threshold()
    )

    print(
        tuning_result[
            "threshold_results"
        ]
    )

    print(
        "Selected threshold:"
    )

    print(
        tuning_result[
            "selected_threshold"
        ]
    )
