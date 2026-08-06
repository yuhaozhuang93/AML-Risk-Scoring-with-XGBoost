"""Train, evaluate, and save the Logistic Regression baseline."""

from typing import Any

import joblib
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.config import (
    BASELINE_METRICS_PATH,
    BASELINE_MODEL_PATH,
    RANDOM_STATE,
    REPORTS_DIR,
    create_project_directories,
)
from src.data_loader import load_bank_data
from src.evaluate import (
    calculate_classification_metrics,
    plot_confusion_matrix,
    save_metrics_json,
    save_predictions_csv,
)
from src.preprocessing import (
    build_preprocessor,
    create_features_and_target,
    identify_feature_types,
    split_training_and_test_data,
)


def build_baseline_pipeline(
    training_features: pd.DataFrame,
) -> Pipeline:
    """Build an unfitted Logistic Regression Pipeline."""

    (
        numeric_features,
        categorical_features,
    ) = identify_feature_types(
        training_features
    )

    preprocessor = build_preprocessor(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
    )

    classifier = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )

    return Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor,
            ),
            (
                "classifier",
                classifier,
            ),
        ]
    )


def train_baseline_model() -> dict[str, Any]:
    """Train, evaluate, and save the baseline Pipeline."""

    create_project_directories()

    dataframe = load_bank_data()

    features, target = (
        create_features_and_target(
            dataframe
        )
    )

    (
        X_train,
        X_test,
        y_train,
        y_test,
    ) = split_training_and_test_data(
        features,
        target,
    )

    pipeline = build_baseline_pipeline(
        X_train
    )

    pipeline.fit(
        X_train,
        y_train,
    )

    y_pred = pipeline.predict(
        X_test
    )

    y_probability = (
        pipeline
        .predict_proba(
            X_test
        )[
            :,
            1,
        ]
    )

    metrics = calculate_classification_metrics(
        y_true=y_test,
        y_pred=y_pred,
        y_probability=y_probability,
    )

    joblib.dump(
        pipeline,
        BASELINE_MODEL_PATH,
    )

    save_metrics_json(
        metrics,
        BASELINE_METRICS_PATH,
    )

    save_predictions_csv(
        features=X_test,
        y_true=y_test,
        y_pred=y_pred,
        y_probability=y_probability,
        output_path=(
            REPORTS_DIR
            / "baseline_predictions.csv"
        ),
    )

    plot_confusion_matrix(
        y_true=y_test,
        y_pred=y_pred,
        output_path=(
            REPORTS_DIR
            / "baseline_confusion_matrix.png"
        ),
        title=(
            "Logistic Regression "
            "Confusion Matrix"
        ),
    )

    return {
        "pipeline": pipeline,
        "metrics": metrics,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "y_pred": y_pred,
        "y_probability": y_probability,
    }


if __name__ == "__main__":
    training_result = (
        train_baseline_model()
    )

    print(
        "Baseline training completed."
    )

    print(
        training_result[
            "metrics"
        ]
    )
