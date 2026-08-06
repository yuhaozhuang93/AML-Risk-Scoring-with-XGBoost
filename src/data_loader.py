"""Functions for loading and validating the bank marketing dataset."""

from pathlib import Path

import pandas as pd

from src.config import (
    BANK_DATA_PATH,
    TARGET_COLUMN,
)


def load_bank_data(
    file_path: Path = BANK_DATA_PATH,
) -> pd.DataFrame:
    """
    Load the UCI Bank Marketing dataset.

    Parameters
    ----------
    file_path:
        Path to the semicolon-delimited CSV file.

    Returns
    -------
    pd.DataFrame
        Validated banking dataset.

    Raises
    ------
    FileNotFoundError
        If the CSV file does not exist.

    ValueError
        If the dataset is empty, lacks the target column,
        or contains duplicated column names.
    """

    if not file_path.exists():
        raise FileNotFoundError(
            "Bank dataset was not found. "
            f"Expected path: {file_path}"
        )

    dataframe = pd.read_csv(
        file_path,
        sep=";",
    )

    if dataframe.empty:
        raise ValueError(
            "The loaded bank dataset is empty."
        )

    if TARGET_COLUMN not in dataframe.columns:
        raise ValueError(
            f"Required target column "
            f"'{TARGET_COLUMN}' is missing."
        )

    if dataframe.columns.duplicated().any():
        duplicated_columns = (
            dataframe.columns[
                dataframe.columns.duplicated()
            ]
            .tolist()
        )

        raise ValueError(
            "Duplicated column names were found: "
            f"{duplicated_columns}"
        )

    return dataframe
