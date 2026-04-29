"""ODE-oriented feature engineering and transition models."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold

from .data import ColumnSpec


def build_transition_panel(df: pd.DataFrame, columns: ColumnSpec) -> pd.DataFrame:
    """Build the one-step-ahead panel used to justify the discrete ODE."""

    transition_df = df[
        [
            columns.athlete_id,
            columns.time_var,
            columns.injury_var,
            "injury_onset",
            "u_t",
            "y_t",
            "y_t_smooth_past",
            columns.recovery,
            columns.sleep,
            columns.stress,
        ]
    ].copy()
    transition_df["y_next_raw"] = transition_df.groupby(columns.athlete_id)["y_t"].shift(-1)
    transition_df["y_next_smooth_past"] = transition_df.groupby(columns.athlete_id)["y_t_smooth_past"].shift(-1)
    transition_df["dy_next_smooth_past"] = (
        transition_df["y_next_smooth_past"] - transition_df["y_t_smooth_past"]
    )
    return transition_df


def fit_clustered_transition(
    frame: pd.DataFrame,
    response: str,
    predictors: list[str],
    cluster_col: str,
):
    """Fit OLS with cluster-robust standard errors at the athlete level."""

    model_frame = frame[[cluster_col, response] + predictors].dropna().copy()
    design = sm.add_constant(model_frame[predictors], has_constant="add")
    result = sm.OLS(model_frame[response], design).fit(
        cov_type="cluster",
        cov_kwds={"groups": model_frame[cluster_col]},
    )
    return result, model_frame


def fit_demeaned_transition(
    frame: pd.DataFrame,
    response: str,
    predictors: list[str],
    group_col: str,
):
    """Remove athlete means before regression to isolate within-athlete signal."""

    model_frame = frame[[group_col, response] + predictors].dropna().copy()
    demeaned_map: dict[str, str] = {}
    for column in [response] + predictors:
        demeaned_column = f"{column}_dm"
        model_frame[demeaned_column] = model_frame[column] - model_frame.groupby(group_col)[column].transform("mean")
        demeaned_map[column] = demeaned_column

    design = model_frame[[demeaned_map[column] for column in predictors]]
    result = sm.OLS(model_frame[demeaned_map[response]], design).fit(
        cov_type="cluster",
        cov_kwds={"groups": model_frame[group_col]},
    )
    return result, model_frame, demeaned_map


def summarize_transition_models(
    transition_df: pd.DataFrame,
    athlete_id_col: str,
) -> dict[str, object]:
    """Fit the core ODE-motivated regressions and return report-ready summaries."""

    raw_result, raw_data = fit_clustered_transition(
        transition_df,
        response="y_next_raw",
        predictors=["y_t", "u_t"],
        cluster_col=athlete_id_col,
    )
    smoothed_result, smoothed_data = fit_clustered_transition(
        transition_df,
        response="y_next_smooth_past",
        predictors=["y_t_smooth_past", "u_t"],
        cluster_col=athlete_id_col,
    )
    within_result, within_data, demeaned_map = fit_demeaned_transition(
        transition_df,
        response="y_next_smooth_past",
        predictors=["y_t_smooth_past", "u_t"],
        group_col=athlete_id_col,
    )
    derivative_result, derivative_data = fit_clustered_transition(
        transition_df,
        response="dy_next_smooth_past",
        predictors=["u_t", "y_t_smooth_past"],
        cluster_col=athlete_id_col,
    )

    transition_summary = pd.DataFrame(
        [
            {
                "specification": "Raw observed transition",
                "response": "y_(t+1)",
                "alpha_hat": raw_result.params["u_t"],
                "alpha_se": raw_result.bse["u_t"],
                "rho_hat": raw_result.params["y_t"],
                "rho_se": raw_result.bse["y_t"],
                "implied_k": 1 - raw_result.params["y_t"],
                "adj_r_squared": raw_result.rsquared_adj,
                "n_obs": len(raw_data),
            },
            {
                "specification": "Past-smoothed transition",
                "response": "y~_(t+1)",
                "alpha_hat": smoothed_result.params["u_t"],
                "alpha_se": smoothed_result.bse["u_t"],
                "rho_hat": smoothed_result.params["y_t_smooth_past"],
                "rho_se": smoothed_result.bse["y_t_smooth_past"],
                "implied_k": 1 - smoothed_result.params["y_t_smooth_past"],
                "adj_r_squared": smoothed_result.rsquared_adj,
                "n_obs": len(smoothed_data),
            },
            {
                "specification": "Past-smoothed transition, within-athlete",
                "response": "y~_(t+1) demeaned",
                "alpha_hat": within_result.params[demeaned_map["u_t"]],
                "alpha_se": within_result.bse[demeaned_map["u_t"]],
                "rho_hat": within_result.params[demeaned_map["y_t_smooth_past"]],
                "rho_se": within_result.bse[demeaned_map["y_t_smooth_past"]],
                "implied_k": 1 - within_result.params[demeaned_map["y_t_smooth_past"]],
                "adj_r_squared": within_result.rsquared_adj,
                "n_obs": len(within_data),
            },
        ]
    )

    derivative_summary = pd.DataFrame(
        [
            {
                "specification": "Past-smoothed derivative regression",
                "response": "delta y~_t",
                "alpha_hat": derivative_result.params["u_t"],
                "alpha_se": derivative_result.bse["u_t"],
                "y_coefficient_hat": derivative_result.params["y_t_smooth_past"],
                "y_coefficient_se": derivative_result.bse["y_t_smooth_past"],
                "implied_k": -derivative_result.params["y_t_smooth_past"],
                "adj_r_squared": derivative_result.rsquared_adj,
                "n_obs": len(derivative_data),
            }
        ]
    )

    return {
        "raw_result": raw_result,
        "smoothed_result": smoothed_result,
        "within_result": within_result,
        "derivative_result": derivative_result,
        "transition_summary": transition_summary,
        "derivative_summary": derivative_summary,
        "smoothed_data": smoothed_data,
    }


def grouped_cv_table(
    frame: pd.DataFrame,
    response: str,
    feature_map: dict[str, list[str]],
    group_col: str,
    n_splits: int = 5,
) -> pd.DataFrame:
    """Compare simple and ODE-style one-step models under grouped CV."""

    rows = []
    splitter = GroupKFold(n_splits=n_splits)
    groups = frame[group_col]

    for label, predictors in feature_map.items():
        y_true: list[float] = []
        y_pred: list[float] = []

        for train_idx, test_idx in splitter.split(frame, groups=groups):
            train = frame.iloc[train_idx]
            test = frame.iloc[test_idx]
            if predictors:
                model = LinearRegression().fit(train[predictors], train[response])
                predictions = model.predict(test[predictors])
            else:
                predictions = np.repeat(train[response].mean(), len(test))
            y_true.extend(test[response].tolist())
            y_pred.extend(predictions.tolist())

        y_true_array = np.asarray(y_true)
        y_pred_array = np.asarray(y_pred)
        rows.append(
            {
                "model": label,
                "rmse": mean_squared_error(y_true_array, y_pred_array) ** 0.5,
                "mae": mean_absolute_error(y_true_array, y_pred_array),
                "r_squared": r2_score(y_true_array, y_pred_array),
            }
        )

    return pd.DataFrame(rows)


def athlete_transition_features(
    smoothed_transition_data: pd.DataFrame,
    df: pd.DataFrame,
    columns: ColumnSpec,
    min_observations: int = 20,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit the smoothed transition separately within each athlete.

    These per-athlete coefficients become the main dynamic inputs for the EM
    stage. The computation is simple OLS because the clustering unit is the
    athlete rather than the session.
    """

    rows = []
    for athlete, athlete_frame in smoothed_transition_data.groupby(columns.athlete_id):
        if len(athlete_frame) < min_observations:
            continue

        design = sm.add_constant(athlete_frame[["y_t_smooth_past", "u_t"]], has_constant="add")
        result = sm.OLS(athlete_frame["y_next_smooth_past"], design).fit()
        full_athlete_frame = df[df[columns.athlete_id] == athlete]
        k_hat = 1 - result.params["y_t_smooth_past"]

        rows.append(
            {
                columns.athlete_id: athlete,
                "n_obs": len(athlete_frame),
                "alpha_hat": result.params["u_t"],
                "rho_hat": result.params["y_t_smooth_past"],
                "k_hat": k_hat,
                "half_life_sessions": np.log(2) / k_hat if k_hat > 0 else np.nan,
                "r_squared": result.rsquared,
                "injury_rate_any": (full_athlete_frame[columns.injury_var] > 0).mean(),
                "injury_rate_injured": (full_athlete_frame[columns.injury_var] == 2).mean(),
                "mean_recovery_score": full_athlete_frame[columns.recovery].mean(),
                "mean_sleep_quality": full_athlete_frame[columns.sleep].mean(),
                "mean_workload": full_athlete_frame["u_t"].mean(),
                "sport_type": full_athlete_frame[columns.sport_type].iloc[0],
                "gender": full_athlete_frame[columns.gender].iloc[0],
            }
        )

    detail = pd.DataFrame(rows)
    summary = pd.DataFrame(
        {
            "metric": [
                "athletes fit",
                "median alpha_hat",
                "median rho_hat",
                "median k_hat",
                "median half-life (sessions)",
                "median athlete R^2",
                "share alpha_hat > 0",
                "share 0 < rho_hat < 1",
                "share k_hat > 0",
            ],
            "value": [
                int(len(detail)),
                detail["alpha_hat"].median(),
                detail["rho_hat"].median(),
                detail["k_hat"].median(),
                detail["half_life_sessions"].median(),
                detail["r_squared"].median(),
                (detail["alpha_hat"] > 0).mean(),
                ((detail["rho_hat"] > 0) & (detail["rho_hat"] < 1)).mean(),
                (detail["k_hat"] > 0).mean(),
            ],
        }
    )
    return detail, summary


def onset_window_summary(
    df: pd.DataFrame,
    columns: ColumnSpec,
    window: int = 3,
) -> tuple[pd.DataFrame, int]:
    """Summarize workload and fatigue around injury onset."""

    rows = []
    onset_count = 0
    for athlete, athlete_df in df.groupby(columns.athlete_id):
        athlete_df = athlete_df.reset_index(drop=True)
        onset_indices = athlete_df.index[athlete_df["injury_onset"]].tolist()
        onset_count += len(onset_indices)
        for idx in onset_indices:
            for rel_session in range(-window, window + 1):
                j = idx + rel_session
                if 0 <= j < len(athlete_df):
                    rows.append(
                        {
                            columns.athlete_id: athlete,
                            "rel_session": rel_session,
                            "u_t": athlete_df.loc[j, "u_t"],
                            "y_t": athlete_df.loc[j, "y_t"],
                        }
                    )

    onset_df = pd.DataFrame(rows)
    summary = onset_df.groupby("rel_session")[["u_t", "y_t"]].agg(["mean", "median"]).reset_index()
    summary.columns = ["rel_session", "u_t_mean", "u_t_median", "y_t_mean", "y_t_median"]
    return summary, onset_count
