"""Train, validate, compare, and save the XGBoost Pipeline."""

from typing import Any

import joblib
import pandas as pd

from sklearn.model_selection import (
    StratifiedKFold,
    cross_validate,
)
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.config import (
    BASELINE_MODEL_PATH,
    MODEL_COMPARISON_PATH,
    RANDOM_STATE,
    REPORTS_DIR,
    XGBOOST_METRICS_PATH,
    XGBOOST_MODEL_PATH,
    create_project_directories,
)
from src.data_loader import load_bank_data
from src.evaluate import (
    calculate_classification_metrics,
    plot_confusion_matrix,
    plot_precision_recall_curves,
    plot_roc_curves,
    save_metrics_json,
    save_predictions_csv,
)
from src.preprocessing import (
    build_preprocessor,
    create_features_and_target,
    identify_feature_types,
    split_training_and_test_data,
)


def calculate_scale_pos_weight(
    training_target: pd.Series,
) -> float:
    """
    Calculate negative-to-positive class weight.

    Parameters
    ----------
    training_target:
        Binary training target.

    Returns
    -------
    float
        Negative-row count divided by positive-row count.
    """

    negative_count = int(
        (
            training_target
            == 0
        ).sum()
    )

    positive_count = int(
        (
            training_target
            == 1
        ).sum()
    )

    if negative_count == 0:
        raise ValueError(
            "Training data contains no "
            "negative-class rows."
        )

    if positive_count == 0:
        raise ValueError(
            "Training data contains no "
            "positive-class rows."
        )

    return (
        negative_count
        / positive_count
    )


def build_xgboost_pipeline(
    training_features: pd.DataFrame,
    training_target: pd.Series,
) -> Pipeline:
    """Build an unfitted preprocessing-and-XGBoost Pipeline."""

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

    scale_pos_weight = (
        calculate_scale_pos_weight(
            training_target
        )
    )

    classifier = XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        min_child_weight=1,
        subsample=0.80,
        colsample_bytree=0.80,
        objective="binary:logistic",
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        tree_method="hist",
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


def run_cross_validation(
    pipeline: Pipeline,
    training_features: pd.DataFrame,
    training_target: pd.Series,
) -> dict[str, Any]:
    """Run five-fold stratified cross-validation."""

    cross_validation = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
        "average_precision": "average_precision",
    }

    results = cross_validate(
        estimator=pipeline,
        X=training_features,
        y=training_target,
        cv=cross_validation,
        scoring=scoring,
        n_jobs=-1,
        return_train_score=True,
    )

    summary: dict[str, Any] = {}

    for metric_name in scoring:
        training_scores = results[
            f"train_{metric_name}"
        ]

        validation_scores = results[
            f"test_{metric_name}"
        ]

        summary[
            f"train_{metric_name}_mean"
        ] = float(
            training_scores.mean()
        )

        summary[
            f"validation_{metric_name}_mean"
        ] = float(
            validation_scores.mean()
        )

        summary[
            f"validation_{metric_name}_std"
        ] = float(
            validation_scores.std()
        )

    summary[
        "fit_time_mean_seconds"
    ] = float(
        results[
            "fit_time"
        ].mean()
    )

    return summary


def compare_with_baseline(
    X_test: pd.DataFrame,
    y_test: pd.Series,
    xgboost_metrics: dict[str, Any],
) -> pd.DataFrame:
    """Compare XGBoost with a previously saved baseline model."""

    comparison_rows = [
        {
            "model": "XGBoost",
            **xgboost_metrics,
        }
    ]

    if BASELINE_MODEL_PATH.exists():
        baseline_pipeline = joblib.load(
            BASELINE_MODEL_PATH
        )

        baseline_prediction = (
            baseline_pipeline
            .predict(
                X_test
            )
        )

        baseline_probability = (
            baseline_pipeline
            .predict_proba(
                X_test
            )[
                :,
                1,
            ]
        )

        baseline_metrics = (
            calculate_classification_metrics(
                y_true=y_test,
                y_pred=baseline_prediction,
                y_probability=baseline_probability,
            )
        )

        comparison_rows.insert(
            0,
            {
                "model": (
                    "Logistic Regression"
                ),
                **baseline_metrics,
            },
        )

    comparison_dataframe = (
        pd.DataFrame(
            comparison_rows
        )
    )

    comparison_dataframe.to_csv(
        MODEL_COMPARISON_PATH,
        index=False,
    )

    return comparison_dataframe


def train_xgboost_model(
    run_cv: bool = True,
) -> dict[str, Any]:
    """Train, evaluate, report, and save the XGBoost Pipeline."""

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

    pipeline = build_xgboost_pipeline(
        training_features=X_train,
        training_target=y_train,
    )

    cross_validation_summary = {}

    if run_cv:
        cross_validation_summary = (
            run_cross_validation(
                pipeline=pipeline,
                training_features=X_train,
                training_target=y_train,
            )
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

    metrics_with_validation = {
        **metrics,
        "scale_pos_weight": float(
            calculate_scale_pos_weight(
                y_train
            )
        ),
        "cross_validation": (
            cross_validation_summary
        ),
    }

    joblib.dump(
        pipeline,
        XGBOOST_MODEL_PATH,
    )

    save_metrics_json(
        metrics_with_validation,
        XGBOOST_METRICS_PATH,
    )

    save_predictions_csv(
        features=X_test,
        y_true=y_test,
        y_pred=y_pred,
        y_probability=y_probability,
        output_path=(
            REPORTS_DIR
            / "sample_predictions.csv"
        ),
    )

    plot_confusion_matrix(
        y_true=y_test,
        y_pred=y_pred,
        output_path=(
            REPORTS_DIR
            / "xgboost_confusion_matrix.png"
        ),
        title="XGBoost Confusion Matrix",
    )

    model_probabilities = {
        "XGBoost": y_probability,
    }

    if BASELINE_MODEL_PATH.exists():
        baseline_pipeline = joblib.load(
            BASELINE_MODEL_PATH
        )

        model_probabilities[
            "Logistic Regression"
        ] = (
            baseline_pipeline
            .predict_proba(
                X_test
            )[
                :,
                1,
            ]
        )

    plot_roc_curves(
        model_probabilities=model_probabilities,
        y_true=y_test,
        output_path=(
            REPORTS_DIR
            / "roc_curve.png"
        ),
    )

    plot_precision_recall_curves(
        model_probabilities=model_probabilities,
        y_true=y_test,
        output_path=(
            REPORTS_DIR
            / "precision_recall_curve.png"
        ),
    )

    comparison_dataframe = compare_with_baseline(
        X_test=X_test,
        y_test=y_test,
        xgboost_metrics=metrics,
    )

    return {
        "pipeline": pipeline,
        "metrics": metrics_with_validation,
        "comparison": comparison_dataframe,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "y_pred": y_pred,
        "y_probability": y_probability,
    }


if __name__ == "__main__":
    training_result = (
        train_xgboost_model(
            run_cv=True
        )
    )

    print(
        "XGBoost training completed."
    )

    print(
        training_result[
            "metrics"
        ]
    )

    print(
        training_result[
            "comparison"
        ]
    )
