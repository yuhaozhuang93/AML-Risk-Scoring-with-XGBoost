"""Load the saved Pipeline and make single or batch predictions."""

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import pandas as pd

from src.config import (
    DEFAULT_THRESHOLD,
    SELECTED_THRESHOLD_PATH,
    XGBOOST_MODEL_PATH,
)


MODEL_VERSION = "xgboost_v1"


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


def load_prediction_pipeline(
    model_path: Path = XGBOOST_MODEL_PATH,
) -> Any:
    """Load the saved preprocessing-and-model Pipeline."""

    if not model_path.exists():
        raise FileNotFoundError(
            "Model file was not found. "
            f"Expected path: {model_path}"
        )

    return joblib.load(
        model_path
    )


def load_selected_threshold(
    threshold_path: Path = SELECTED_THRESHOLD_PATH,
    fallback_threshold: float = DEFAULT_THRESHOLD,
) -> float:
    """Load the selected threshold or use the configured fallback."""

    if not threshold_path.exists():
        return validate_threshold(
            fallback_threshold
        )

    with threshold_path.open(
        mode="r",
        encoding="utf-8",
    ) as file:
        threshold_config = json.load(
            file
        )

    if "threshold" not in threshold_config:
        raise ValueError(
            "Selected-threshold file does not "
            "contain a 'threshold' field."
        )

    return validate_threshold(
        threshold_config[
            "threshold"
        ]
    )


def assign_risk_band(
    probability: float,
) -> str:
    """
    Convert a probability to a demonstration risk band.

    These boundaries are educational placeholders and are
    not approved AML risk-policy thresholds.
    """

    numeric_probability = float(
        probability
    )

    if not (
        0.0
        <= numeric_probability
        <= 1.0
    ):
        raise ValueError(
            "Probability must be between "
            "0 and 1."
        )

    if numeric_probability >= 0.75:
        return "High"

    if numeric_probability >= 0.40:
        return "Medium"

    return "Low"


def validate_input_dataframe(
    pipeline: Any,
    input_dataframe: pd.DataFrame,
) -> None:
    """Check that prediction inputs contain the expected raw features."""

    if input_dataframe.empty:
        raise ValueError(
            "Prediction input cannot be empty."
        )

    fitted_preprocessor = (
        pipeline.named_steps[
            "preprocessor"
        ]
    )

    expected_features = list(
        fitted_preprocessor.feature_names_in_
    )

    missing_features = [
        feature
        for feature in expected_features
        if feature not in input_dataframe.columns
    ]

    extra_features = [
        feature
        for feature in input_dataframe.columns
        if feature not in expected_features
    ]

    if missing_features:
        raise ValueError(
            "Prediction input is missing features: "
            f"{missing_features}"
        )

    if extra_features:
        raise ValueError(
            "Prediction input contains unexpected "
            f"features: {extra_features}"
        )


def predict_dataframe(
    pipeline: Any,
    input_dataframe: pd.DataFrame,
    threshold: float,
) -> list[dict[str, Any]]:
    """Predict every row in an input DataFrame."""

    validated_threshold = validate_threshold(
        threshold
    )

    validate_input_dataframe(
        pipeline,
        input_dataframe,
    )

    fitted_preprocessor = (
        pipeline.named_steps[
            "preprocessor"
        ]
    )

    expected_features = list(
        fitted_preprocessor.feature_names_in_
    )

    ordered_input = input_dataframe[
        expected_features
    ].copy()

    positive_probabilities = (
        pipeline
        .predict_proba(
            ordered_input
        )[
            :,
            1,
        ]
    )

    predictions = (
        positive_probabilities
        >= validated_threshold
    ).astype(
        int
    )

    results: list[
        dict[str, Any]
    ] = []

    for row_number, (
        prediction,
        probability,
    ) in enumerate(
        zip(
            predictions,
            positive_probabilities,
            strict=True,
        )
    ):
        results.append(
            {
                "row_number": row_number,
                "prediction": int(
                    prediction
                ),
                "positive_probability": float(
                    probability
                ),
                "risk_band": assign_risk_band(
                    probability
                ),
                "threshold": (
                    validated_threshold
                ),
                "model_version": (
                    MODEL_VERSION
                ),
            }
        )

    return results


def predict_single_record(
    pipeline: Any,
    record: Mapping[str, Any],
    threshold: float,
) -> dict[str, Any]:
    """Predict one raw banking record."""

    input_dataframe = pd.DataFrame(
        [
            dict(
                record
            )
        ]
    )

    results = predict_dataframe(
        pipeline=pipeline,
        input_dataframe=input_dataframe,
        threshold=threshold,
    )

    return results[
        0
    ]


def predict_batch_records(
    pipeline: Any,
    records: Sequence[
        Mapping[
            str,
            Any,
        ]
    ],
    threshold: float,
) -> list[dict[str, Any]]:
    """Predict multiple raw banking records."""

    if not records:
        raise ValueError(
            "records cannot be empty."
        )

    input_dataframe = pd.DataFrame(
        [
            dict(
                record
            )
            for record in records
        ]
    )

    return predict_dataframe(
        pipeline=pipeline,
        input_dataframe=input_dataframe,
        threshold=threshold,
    )


def main() -> None:
    """Run one demonstration prediction."""

    pipeline = load_prediction_pipeline()

    threshold = load_selected_threshold()

    sample_record = {
        "age": 42,
        "job": "management",
        "marital": "married",
        "education": "tertiary",
        "default": "no",
        "balance": 8500,
        "housing": "no",
        "loan": "no",
        "contact": "cellular",
        "day": 15,
        "month": "may",
        "campaign": 2,
        "pdays": -1,
        "previous": 0,
        "poutcome": "unknown",
    }

    prediction_result = (
        predict_single_record(
            pipeline=pipeline,
            record=sample_record,
            threshold=threshold,
        )
    )

    print(
        json.dumps(
            prediction_result,
            indent=4,
        )
    )


if __name__ == "__main__":
    main()
