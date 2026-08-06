"""Feature, target, preprocessing, and split utilities."""

from typing import Sequence

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
)

from src.config import (
    LEAKAGE_COLUMNS,
    RANDOM_STATE,
    TARGET_COLUMN,
    TARGET_MAPPING,
    TEST_SIZE,
)


def create_features_and_target(
    dataframe: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Create model features and a binary target.

    Parameters
    ----------
    dataframe:
        Raw bank marketing dataset.

    Returns
    -------
    tuple[pd.DataFrame, pd.Series]
        Feature DataFrame and integer target Series.

    Raises
    ------
    ValueError
        If required columns are absent, target values cannot
        be mapped, or feature and target lengths differ.
    """

    if TARGET_COLUMN not in dataframe.columns:
        raise ValueError(
            f"Target column "
            f"'{TARGET_COLUMN}' is missing."
        )

    missing_leakage_columns = [
        column
        for column in LEAKAGE_COLUMNS
        if column not in dataframe.columns
    ]

    if missing_leakage_columns:
        raise ValueError(
            "Expected leakage columns are missing: "
            f"{missing_leakage_columns}"
        )

    target = (
        dataframe[
            TARGET_COLUMN
        ]
        .map(
            TARGET_MAPPING
        )
    )

    if target.isnull().any():
        unmapped_values = (
            dataframe.loc[
                target.isnull(),
                TARGET_COLUMN,
            ]
            .drop_duplicates()
            .tolist()
        )

        raise ValueError(
            "Target contains unmapped values: "
            f"{unmapped_values}"
        )

    target = target.astype(
        "int64"
    )

    columns_to_drop = [
        TARGET_COLUMN,
        *LEAKAGE_COLUMNS,
    ]

    features = dataframe.drop(
        columns=columns_to_drop,
    )

    if features.empty:
        raise ValueError(
            "No model features remain after "
            "dropping target and leakage columns."
        )

    if len(features) != len(target):
        raise ValueError(
            "Feature and target row counts "
            "do not match."
        )

    return (
        features,
        target,
    )


def split_training_and_test_data(
    features: pd.DataFrame,
    target: pd.Series,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    pd.Series,
]:
    """
    Create the reproducible stratified train/test split.

    Parameters
    ----------
    features:
        Model feature DataFrame.

    target:
        Binary target Series.

    Returns
    -------
    tuple
        X_train, X_test, y_train, and y_test.
    """

    if len(features) != len(target):
        raise ValueError(
            "Feature and target row counts "
            "do not match."
        )

    if target.nunique() != 2:
        raise ValueError(
            "The target must contain exactly "
            "two classes."
        )

    return train_test_split(
        features,
        target,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=target,
    )


def identify_feature_types(
    training_features: pd.DataFrame,
) -> tuple[list[str], list[str]]:
    """
    Identify numeric and categorical feature names.

    Parameters
    ----------
    training_features:
        Training feature DataFrame.

    Returns
    -------
    tuple[list[str], list[str]]
        Numeric and categorical column names.
    """

    numeric_features = (
        training_features
        .select_dtypes(
            include="number"
        )
        .columns
        .tolist()
    )

    categorical_features = (
        training_features
        .select_dtypes(
            include=[
                "object",
                "category",
                "bool",
            ]
        )
        .columns
        .tolist()
    )

    identified_features = (
        numeric_features
        + categorical_features
    )

    unidentified_features = [
        column
        for column in training_features.columns
        if column not in identified_features
    ]

    if unidentified_features:
        raise ValueError(
            "Unsupported feature dtypes were found: "
            f"{unidentified_features}"
        )

    if not numeric_features:
        raise ValueError(
            "No numeric features were identified."
        )

    if not categorical_features:
        raise ValueError(
            "No categorical features were identified."
        )

    return (
        numeric_features,
        categorical_features,
    )


def build_preprocessor(
    numeric_features: Sequence[str],
    categorical_features: Sequence[str],
) -> ColumnTransformer:
    """
    Build numeric and categorical preprocessing pipelines.

    Parameters
    ----------
    numeric_features:
        Numeric feature names.

    categorical_features:
        Categorical feature names.

    Returns
    -------
    ColumnTransformer
        Unfitted preprocessing transformer.
    """

    if not numeric_features:
        raise ValueError(
            "numeric_features cannot be empty."
        )

    if not categorical_features:
        raise ValueError(
            "categorical_features cannot be empty."
        )

    numeric_transformer = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                ),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent",
                ),
            ),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=True,
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_transformer,
                list(
                    numeric_features
                ),
            ),
            (
                "categorical",
                categorical_transformer,
                list(
                    categorical_features
                ),
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )

    return preprocessor
