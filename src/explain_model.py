"""Generate XGBoost feature-importance and SHAP explanations."""

from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from sklearn.pipeline import Pipeline

from src.config import (
    RANDOM_STATE,
    REPORTS_DIR,
    XGBOOST_MODEL_PATH,
    create_project_directories,
)
from src.data_loader import load_bank_data
from src.preprocessing import (
    create_features_and_target,
    split_training_and_test_data,
)


SHAP_SAMPLE_SIZE = 1000

TOP_FEATURE_COUNT = 20


def get_fitted_pipeline_components(
    pipeline: Pipeline,
) -> tuple[Any, Any]:
    """Return fitted preprocessor and classifier steps."""

    required_steps = {
        "preprocessor",
        "classifier",
    }

    missing_steps = (
        required_steps
        - set(
            pipeline.named_steps
        )
    )

    if missing_steps:
        raise ValueError(
            "Pipeline is missing required steps: "
            f"{sorted(missing_steps)}"
        )

    fitted_preprocessor = (
        pipeline.named_steps[
            "preprocessor"
        ]
    )

    fitted_classifier = (
        pipeline.named_steps[
            "classifier"
        ]
    )

    return (
        fitted_preprocessor,
        fitted_classifier,
    )


def get_transformed_feature_names(
    pipeline: Pipeline,
) -> np.ndarray:
    """Get feature names produced by the fitted preprocessor."""

    fitted_preprocessor, _ = (
        get_fitted_pipeline_components(
            pipeline
        )
    )

    feature_names = (
        fitted_preprocessor
        .get_feature_names_out()
    )

    return np.asarray(
        feature_names,
        dtype=str,
    )


def build_feature_importance_dataframe(
    pipeline: Pipeline,
) -> pd.DataFrame:
    """Create a sorted XGBoost feature-importance table."""

    (
        _,
        fitted_classifier,
    ) = get_fitted_pipeline_components(
        pipeline
    )

    feature_names = (
        get_transformed_feature_names(
            pipeline
        )
    )

    importance_values = np.asarray(
        fitted_classifier.feature_importances_,
        dtype=float,
    )

    if (
        len(feature_names)
        != len(importance_values)
    ):
        raise ValueError(
            "Feature-name count does not match "
            "importance-value count."
        )

    feature_importance_dataframe = (
        pd.DataFrame(
            {
                "feature": feature_names,
                "importance": (
                    importance_values
                ),
            }
        )
        .sort_values(
            by="importance",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )

    return feature_importance_dataframe


def plot_feature_importance(
    feature_importance_dataframe: pd.DataFrame,
) -> None:
    """Save a horizontal plot of the top feature importances."""

    top_features = (
        feature_importance_dataframe
        .head(
            TOP_FEATURE_COUNT
        )
        .sort_values(
            by="importance",
            ascending=True,
        )
    )

    figure, axis = plt.subplots(
        figsize=(
            10,
            8,
        )
    )

    axis.barh(
        top_features[
            "feature"
        ],
        top_features[
            "importance"
        ],
    )

    axis.set_xlabel(
        "XGBoost Feature Importance"
    )

    axis.set_ylabel(
        "Transformed Feature"
    )

    axis.set_title(
        f"Top {TOP_FEATURE_COUNT} "
        "XGBoost Features"
    )

    figure.tight_layout()

    figure.savefig(
        REPORTS_DIR
        / "feature_importance.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )


def transform_features_for_classifier(
    pipeline: Pipeline,
    features: pd.DataFrame,
) -> pd.DataFrame:
    """Transform raw features into the classifier input space."""

    fitted_preprocessor, _ = (
        get_fitted_pipeline_components(
            pipeline
        )
    )

    transformed_features = (
        fitted_preprocessor
        .transform(
            features
        )
    )

    if hasattr(
        transformed_features,
        "toarray",
    ):
        transformed_features = (
            transformed_features
            .toarray()
        )

    feature_names = (
        get_transformed_feature_names(
            pipeline
        )
    )

    if (
        transformed_features.shape[
            1
        ]
        != len(feature_names)
    ):
        raise ValueError(
            "Transformed feature count does "
            "not match feature-name count."
        )

    return pd.DataFrame(
        transformed_features,
        columns=feature_names,
        index=features.index,
    )


def sample_transformed_features(
    transformed_features: pd.DataFrame,
    sample_size: int = SHAP_SAMPLE_SIZE,
) -> pd.DataFrame:
    """Create a reproducible SHAP sample."""

    if sample_size <= 0:
        raise ValueError(
            "sample_size must be positive."
        )

    actual_sample_size = min(
        sample_size,
        len(
            transformed_features
        ),
    )

    return (
        transformed_features
        .sample(
            n=actual_sample_size,
            random_state=RANDOM_STATE,
        )
    )


def create_shap_explanations(
    pipeline: Pipeline,
    transformed_sample: pd.DataFrame,
) -> tuple[Any, Any]:
    """Create a TreeExplainer and SHAP Explanation values."""

    _, fitted_classifier = (
        get_fitted_pipeline_components(
            pipeline
        )
    )

    explainer = shap.TreeExplainer(
        fitted_classifier
    )

    shap_values = explainer(
        transformed_sample
    )

    return (
        explainer,
        shap_values,
    )


def save_shap_summary_plot(
    shap_values: Any,
    transformed_sample: pd.DataFrame,
) -> None:
    """Save the global SHAP summary plot."""

    shap.summary_plot(
        shap_values,
        transformed_sample,
        show=False,
        max_display=TOP_FEATURE_COUNT,
    )

    plt.tight_layout()

    plt.savefig(
        REPORTS_DIR
        / "shap_summary.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


def save_single_record_waterfall_plot(
    shap_values: Any,
    sample_position: int = 0,
) -> None:
    """Save a SHAP waterfall plot for one sample row."""

    if not (
        0
        <= sample_position
        < len(
            shap_values
        )
    ):
        raise IndexError(
            "sample_position is outside the "
            "available SHAP rows."
        )

    shap.plots.waterfall(
        shap_values[
            sample_position
        ],
        max_display=15,
        show=False,
    )

    plt.tight_layout()

    plt.savefig(
        REPORTS_DIR
        / "sample_shap_waterfall.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


def explain_xgboost_model() -> dict[str, Any]:
    """Generate feature-importance and SHAP report files."""

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
        _,
    ) = split_training_and_test_data(
        features,
        target,
    )

    feature_importance_dataframe = (
        build_feature_importance_dataframe(
            pipeline
        )
    )

    feature_importance_dataframe.to_csv(
        REPORTS_DIR
        / "feature_importance.csv",
        index=False,
    )

    plot_feature_importance(
        feature_importance_dataframe
    )

    transformed_test_features = (
        transform_features_for_classifier(
            pipeline=pipeline,
            features=X_test,
        )
    )

    transformed_sample = (
        sample_transformed_features(
            transformed_test_features
        )
    )

    _, shap_values = (
        create_shap_explanations(
            pipeline=pipeline,
            transformed_sample=(
                transformed_sample
            ),
        )
    )

    save_shap_summary_plot(
        shap_values=shap_values,
        transformed_sample=(
            transformed_sample
        ),
    )

    save_single_record_waterfall_plot(
        shap_values=shap_values,
        sample_position=0,
    )

    return {
        "feature_importance": (
            feature_importance_dataframe
        ),
        "shap_sample": (
            transformed_sample
        ),
        "shap_values": (
            shap_values
        ),
    }


if __name__ == "__main__":
    explanation_result = (
        explain_xgboost_model()
    )

    print(
        "Model explanation completed."
    )

    print(
        explanation_result[
            "feature_importance"
        ]
        .head(
            TOP_FEATURE_COUNT
        )
    )
