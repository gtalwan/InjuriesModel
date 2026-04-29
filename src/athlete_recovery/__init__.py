"""Reusable code for the athlete recovery ODE + EM analysis."""

from .data import ColumnSpec, find_dataset_zip, load_dataset_from_zip, panel_overview, preprocess_panel
from .dynamics import (
    athlete_transition_features,
    build_transition_panel,
    fit_clustered_transition,
    fit_demeaned_transition,
    grouped_cv_table,
    onset_window_summary,
    summarize_transition_models,
)
from .mixture import EMResult, build_athlete_feature_frame, fit_primary_em, repeated_gmm_selection
from .pipeline import CoreAnalysisResult, run_core_analysis, write_core_outputs

__all__ = [
    "ColumnSpec",
    "CoreAnalysisResult",
    "EMResult",
    "athlete_transition_features",
    "build_athlete_feature_frame",
    "build_transition_panel",
    "find_dataset_zip",
    "fit_clustered_transition",
    "fit_demeaned_transition",
    "fit_primary_em",
    "grouped_cv_table",
    "load_dataset_from_zip",
    "onset_window_summary",
    "panel_overview",
    "preprocess_panel",
    "repeated_gmm_selection",
    "run_core_analysis",
    "summarize_transition_models",
    "write_core_outputs",
]
