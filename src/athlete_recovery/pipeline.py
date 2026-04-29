"""High-level pipeline for the ODE + EM recovery analysis."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .data import ColumnSpec, find_dataset_zip, load_dataset_from_zip, panel_overview, preprocess_panel
from .dynamics import (
    athlete_transition_features,
    build_transition_panel,
    grouped_cv_table,
    onset_window_summary,
    summarize_transition_models,
)
from .mixture import EMResult, build_athlete_feature_frame, fit_primary_em
from .plots import plot_em_model_selection, plot_em_structure, plot_onset_profiles, save_figure


@dataclass
class CoreAnalysisResult:
    """Bundle the main tables and EM result returned by the pipeline."""

    panel_overview: pd.DataFrame
    transition_summary: pd.DataFrame
    derivative_summary: pd.DataFrame
    cv_model_comparison: pd.DataFrame
    athlete_transition_detail: pd.DataFrame
    athlete_transition_summary: pd.DataFrame
    onset_summary: pd.DataFrame
    onset_count: int
    em_result: EMResult


def run_core_analysis(project_root: Path) -> CoreAnalysisResult:
    """Run the main restricted ODE + EM workflow used in the report."""

    columns = ColumnSpec()
    zip_path = find_dataset_zip(project_root)
    raw_df = load_dataset_from_zip(zip_path)
    df = preprocess_panel(raw_df, columns)

    overview = panel_overview(raw_df, df, columns)
    transition_df = build_transition_panel(df, columns)
    transition_outputs = summarize_transition_models(transition_df, columns.athlete_id)

    cv_frame = transition_df[
        [
            columns.athlete_id,
            "u_t",
            "y_t_smooth_past",
            "y_next_smooth_past",
            columns.recovery,
            columns.sleep,
            columns.stress,
        ]
    ].dropna()
    feature_map = {
        "Intercept only": [],
        "Workload only": ["u_t"],
        "Persistence only": ["y_t_smooth_past"],
        "ODE transition": ["y_t_smooth_past", "u_t"],
        "Contextual benchmark": ["y_t_smooth_past", "u_t", columns.recovery, columns.sleep, columns.stress],
    }
    cv_model_comparison = grouped_cv_table(
        cv_frame,
        response="y_next_smooth_past",
        feature_map=feature_map,
        group_col=columns.athlete_id,
    )

    athlete_detail, athlete_summary = athlete_transition_features(
        transition_outputs["smoothed_data"],
        df,
        columns,
    )
    onset_summary, onset_count = onset_window_summary(df, columns)

    feature_frame, primary_columns, validation_columns, profile_columns = build_athlete_feature_frame(
        df,
        athlete_detail,
        columns,
    )
    em_result = fit_primary_em(
        df=df,
        feature_frame=feature_frame,
        primary_feature_columns=primary_columns,
        validation_feature_columns=validation_columns,
        profile_feature_columns=profile_columns,
        columns=columns,
    )

    return CoreAnalysisResult(
        panel_overview=overview,
        transition_summary=transition_outputs["transition_summary"],
        derivative_summary=transition_outputs["derivative_summary"],
        cv_model_comparison=cv_model_comparison,
        athlete_transition_detail=athlete_detail,
        athlete_transition_summary=athlete_summary,
        onset_summary=onset_summary,
        onset_count=onset_count,
        em_result=em_result,
    )


def write_core_outputs(result: CoreAnalysisResult, output_dir: Path) -> None:
    """Write the selected GitHub-ready tables and figures."""

    output_dir.mkdir(parents=True, exist_ok=True)

    table_map = {
        "table_panel_overview.csv": result.panel_overview,
        "table_transition_model_justification.csv": result.transition_summary,
        "table_cv_model_comparison.csv": result.cv_model_comparison,
        "table_athlete_transition_detail.csv": result.athlete_transition_detail,
        "table_athlete_transition_summary.csv": result.athlete_transition_summary,
        "table_injury_onset_window_summary.csv": result.onset_summary,
        "table_em_model_selection_primary.csv": result.em_result.selection_table,
        "table_em_cluster_summary_primary.csv": result.em_result.cluster_summary,
        "table_em_validation_continuous.csv": result.em_result.validation_continuous,
        "table_em_validation_categorical.csv": result.em_result.validation_categorical,
        "table_em_cluster_onset_summary.csv": result.em_result.cluster_onset_summary,
        "table_em_cluster_transition_summary.csv": result.em_result.cluster_transition_summary,
    }
    for filename, frame in table_map.items():
        frame.to_csv(output_dir / filename, index=False)

    save_figure(plot_em_model_selection(result.em_result), output_dir / "figure_8_em_extension.png")
    save_figure(plot_em_structure(result.em_result), output_dir / "figure_8b_em_robustness.png")
    save_figure(plot_onset_profiles(result.em_result), output_dir / "figure_8d_em_onset_profiles.png")
