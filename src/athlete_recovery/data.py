"""Data loading and panel preprocessing for the athlete recovery project.

The notebook in this repository is the narrative analysis. This module is the
reusable implementation behind that story: it loads the zipped dataset, cleans
the panel, and constructs the time-series variables used by the ODE and EM
stages.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ColumnSpec:
    """Canonical dataset mapping used throughout the analysis."""

    athlete_id: str = "athlete_id"
    time_var: str = "session_id"
    injury_var: str = "injury_occurred"
    workload: str = "training_load"
    workload_intensity: str = "training_intensity"
    workload_duration: str = "training_duration"
    fatigue: str = "fatigue_index"
    recovery: str = "recovery_score"
    sleep: str = "sleep_quality"
    stress: str = "stress_level"
    hydration: str = "hydration_level"
    heart_rate: str = "heart_rate"
    muscle_activity: str = "muscle_activity"
    jump_height: str = "jump_height"
    gait_speed: str = "gait_speed"
    range_of_motion: str = "range_of_motion"
    sport_type: str = "sport_type"
    gender: str = "gender"
    age: str = "age"
    bmi: str = "bmi"


def load_dataset_from_zip(zip_path: Path) -> pd.DataFrame:
    """Load the first CSV stored in the dataset zip archive."""

    with zipfile.ZipFile(zip_path) as archive:
        csv_members = [name for name in archive.namelist() if name.lower().endswith(".csv")]
        if not csv_members:
            raise FileNotFoundError(f"No CSV file found inside {zip_path}.")
        with archive.open(csv_members[0]) as handle:
            return pd.read_csv(handle)


def find_dataset_zip(project_root: Path) -> Path:
    """Locate the single dataset archive expected at the repository root."""

    zip_candidates = sorted(project_root.glob("*.zip"))
    if not zip_candidates:
        raise FileNotFoundError(f"No dataset zip file found in {project_root}.")
    return zip_candidates[0]


def validate_required_columns(frame: pd.DataFrame, columns: ColumnSpec) -> None:
    """Fail early if the dataset is missing columns that drive the core model."""

    required = {
        columns.athlete_id,
        columns.time_var,
        columns.injury_var,
        columns.workload,
        columns.workload_intensity,
        columns.workload_duration,
        columns.fatigue,
        columns.recovery,
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")


def preprocess_panel(
    raw_df: pd.DataFrame,
    columns: ColumnSpec,
    smoothing_window: int = 3,
) -> pd.DataFrame:
    """Clean the panel and construct the variables used in the analysis.

    The main design choices mirror the report:
    - impute numeric variables with athlete-level medians first, then global medians;
    - impute categorical variables with the global mode;
    - build a workload input ``u_t`` and a fatigue proxy ``y_t``;
    - add both centered and trailing rolling means of fatigue;
    - mark next-session injury and injury onset.
    """

    validate_required_columns(raw_df, columns)

    df = raw_df.copy().sort_values([columns.athlete_id, columns.time_var]).reset_index(drop=True)

    numeric_columns = df.select_dtypes(include=np.number).columns.tolist()
    protected_numeric = {columns.athlete_id, columns.injury_var, columns.time_var}
    impute_numeric = [column for column in numeric_columns if column not in protected_numeric]
    for column in impute_numeric:
        if df[column].isna().any():
            df[column] = df.groupby(columns.athlete_id)[column].transform(lambda s: s.fillna(s.median()))
            df[column] = df[column].fillna(df[column].median())

    categorical_columns = [column for column in df.columns if column not in numeric_columns]
    for column in categorical_columns:
        if df[column].isna().any():
            mode = df[column].mode(dropna=True)
            if not mode.empty:
                df[column] = df[column].fillna(mode.iloc[0])

    df["u_t"] = df[columns.workload]
    df["u_intensity_x_duration"] = df[columns.workload_intensity] * df[columns.workload_duration]
    df["y_t"] = df[columns.fatigue]

    group = df.groupby(columns.athlete_id)["y_t"]
    df["y_t_smooth_centered"] = group.transform(
        lambda s: s.rolling(window=smoothing_window, min_periods=1, center=True).mean()
    )
    df["y_t_smooth_past"] = group.transform(
        lambda s: s.rolling(window=smoothing_window, min_periods=1).mean()
    )
    df["y_t_smooth"] = df["y_t_smooth_centered"]
    df["u_ma_3"] = df.groupby(columns.athlete_id)["u_t"].transform(
        lambda s: s.rolling(window=3, min_periods=1).mean()
    )

    df["delta_t"] = df.groupby(columns.athlete_id)[columns.time_var].diff().astype(float).replace(0, np.nan)
    df["dy_dt"] = df.groupby(columns.athlete_id)["y_t_smooth"].diff() / df["delta_t"]

    df["prev_injury"] = df.groupby(columns.athlete_id)[columns.injury_var].shift(1)
    df["next_injury"] = df.groupby(columns.athlete_id)[columns.injury_var].shift(-1)
    df["next_injured"] = (df["next_injury"] == 2).astype(float)
    df["injury_onset"] = df[columns.injury_var].eq(2) & df["prev_injury"].fillna(0).lt(2)

    return df


def panel_overview(raw_df: pd.DataFrame, df: pd.DataFrame, columns: ColumnSpec) -> pd.DataFrame:
    """Create a compact audit table used in the report and README."""

    overall_missing_pct = 100.0 * raw_df.isna().sum().sum() / raw_df.size
    missingness = raw_df.isna().mean().mul(100).sort_values(ascending=False)
    panel_counts = df.groupby(columns.athlete_id).size()
    session_gaps = df.groupby(columns.athlete_id)[columns.time_var].diff().dropna()

    largest_missing = missingness.iloc[0]
    largest_missing_name = missingness.index[0]
    has_nonunit_gaps = bool((session_gaps != 1.0).any()) if not session_gaps.empty else False

    return pd.DataFrame(
        {
            "metric": [
                "observations",
                "athletes",
                "sessions per athlete (mean)",
                "sessions per athlete (min-max)",
                "session index span",
                "overall missingness (%) before imputation",
                "largest column missingness (%) before imputation",
                "non-unit session gaps detected",
            ],
            "value": [
                f"{len(df):,}",
                f"{df[columns.athlete_id].nunique()}",
                f"{panel_counts.mean():.2f}",
                f"{panel_counts.min()}-{panel_counts.max()}",
                f"{df[columns.time_var].min()}-{df[columns.time_var].max()}",
                f"{overall_missing_pct:.3f}",
                f"{largest_missing:.3f} ({largest_missing_name})",
                "Yes" if has_nonunit_gaps else "No",
            ],
        }
    )
