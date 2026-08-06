"""Central configuration for the AML risk-scoring learning project."""

from pathlib import Path
from typing import Final


PROJECT_ROOT: Final[Path] = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)

DATA_DIR: Final[Path] = (
    PROJECT_ROOT
    / "data"
)

RAW_DATA_DIR: Final[Path] = (
    DATA_DIR
    / "raw"
)

PROCESSED_DATA_DIR: Final[Path] = (
    DATA_DIR
    / "processed"
)

MODELS_DIR: Final[Path] = (
    PROJECT_ROOT
    / "models"
)

REPORTS_DIR: Final[Path] = (
    PROJECT_ROOT
    / "reports"
)

BANK_DATA_PATH: Final[Path] = (
    RAW_DATA_DIR
    / "bank-full.csv"
)

BASELINE_MODEL_PATH: Final[Path] = (
    MODELS_DIR
    / "baseline_pipeline.pkl"
)

XGBOOST_MODEL_PATH: Final[Path] = (
    MODELS_DIR
    / "xgboost_pipeline.pkl"
)

BASELINE_METRICS_PATH: Final[Path] = (
    REPORTS_DIR
    / "baseline_metrics.json"
)

XGBOOST_METRICS_PATH: Final[Path] = (
    REPORTS_DIR
    / "xgboost_metrics.json"
)

MODEL_COMPARISON_PATH: Final[Path] = (
    REPORTS_DIR
    / "model_comparison.csv"
)

THRESHOLD_RESULTS_PATH: Final[Path] = (
    REPORTS_DIR
    / "threshold_results.csv"
)

SELECTED_THRESHOLD_PATH: Final[Path] = (
    REPORTS_DIR
    / "selected_threshold.json"
)

TARGET_COLUMN: Final[str] = "y"

TARGET_MAPPING: Final[dict[str, int]] = {
    "no": 0,
    "yes": 1,
}

LEAKAGE_COLUMNS: Final[list[str]] = [
    "duration",
]

TEST_SIZE: Final[float] = 0.20

RANDOM_STATE: Final[int] = 42

DEFAULT_THRESHOLD: Final[float] = 0.50

MINIMUM_RECALL: Final[float] = 0.75


def create_project_directories() -> None:
    """Create output directories required by training and reporting."""

    for directory in [
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        MODELS_DIR,
        REPORTS_DIR,
    ]:
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )
