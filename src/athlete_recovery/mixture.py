"""EM / Gaussian-mixture modeling for latent recovery profiles."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, f_oneway, kruskal
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm

from .data import ColumnSpec


@dataclass
class EMResult:
    """Container for the main EM outputs used by the report and README."""

    feature_frame: pd.DataFrame
    selection_table: pd.DataFrame
    primary_k: int
    assignments: pd.DataFrame
    cluster_summary: pd.DataFrame
    validation_continuous: pd.DataFrame
    validation_categorical: pd.DataFrame
    cluster_onset_summary: pd.DataFrame
    cluster_transition_summary: pd.DataFrame
    profile_heatmap_z: pd.DataFrame
    cluster_label_order: list[str]
    cluster_palette: dict[str, str]
    primary_feature_columns: list[str]
    validation_feature_columns: list[str]


def build_athlete_feature_frame(
    df: pd.DataFrame,
    athlete_transition_detail: pd.DataFrame,
    columns: ColumnSpec,
) -> tuple[pd.DataFrame, list[str], list[str], list[str]]:
    """Assemble the athlete-level matrix used for clustering and validation."""

    athlete_validation = (
        df.groupby(columns.athlete_id)
        .agg(
            mean_stress_level=(columns.stress, "mean"),
            mean_hydration=(columns.hydration, "mean"),
            mean_heart_rate=(columns.heart_rate, "mean"),
            mean_muscle_activity=(columns.muscle_activity, "mean"),
            mean_jump_height=(columns.jump_height, "mean"),
            mean_gait_speed=(columns.gait_speed, "mean"),
            mean_range_of_motion=(columns.range_of_motion, "mean"),
            age=(columns.age, "first"),
            bmi=(columns.bmi, "first"),
        )
        .reset_index()
    )

    onset_rows = []
    for athlete, athlete_df in df.groupby(columns.athlete_id):
        athlete_df = athlete_df.reset_index(drop=True)
        onset_indices = athlete_df.index[athlete_df["injury_onset"]].tolist()
        per_onset_rows = []
        for idx in onset_indices:
            if idx + 5 < len(athlete_df):
                per_onset_rows.append(
                    {
                        "onset_y": athlete_df.loc[idx, "y_t"],
                        "drop_y_1": athlete_df.loc[idx, "y_t"] - athlete_df.loc[idx + 1, "y_t"],
                        "drop_y_3": athlete_df.loc[idx, "y_t"] - athlete_df.loc[idx + 3, "y_t"],
                        "drop_y_5": athlete_df.loc[idx, "y_t"] - athlete_df.loc[idx + 5, "y_t"],
                        "rec_post3": athlete_df.loc[idx + 1 : idx + 3, columns.recovery].mean(),
                        "rec_post5": athlete_df.loc[idx + 1 : idx + 5, columns.recovery].mean(),
                        "u_drop_3": athlete_df.loc[idx, "u_t"] - athlete_df.loc[idx + 3, "u_t"],
                    }
                )

        athlete_features = {columns.athlete_id: athlete, "onset_count": len(per_onset_rows)}
        if per_onset_rows:
            onset_frame = pd.DataFrame(per_onset_rows)
            athlete_features.update(onset_frame.mean().to_dict())
            athlete_features["drop_y_1_sd"] = onset_frame["drop_y_1"].std(ddof=0)
            athlete_features["drop_y_3_sd"] = onset_frame["drop_y_3"].std(ddof=0)
            athlete_features["drop_y_5_sd"] = onset_frame["drop_y_5"].std(ddof=0)
        else:
            athlete_features.update(
                {
                    "onset_y": np.nan,
                    "drop_y_1": np.nan,
                    "drop_y_3": np.nan,
                    "drop_y_5": np.nan,
                    "rec_post3": np.nan,
                    "rec_post5": np.nan,
                    "u_drop_3": np.nan,
                    "drop_y_1_sd": np.nan,
                    "drop_y_3_sd": np.nan,
                    "drop_y_5_sd": np.nan,
                }
            )
        onset_rows.append(athlete_features)

    onset_features = pd.DataFrame(onset_rows)
    feature_frame = (
        athlete_transition_detail.merge(onset_features, on=columns.athlete_id, how="left")
        .merge(athlete_validation, on=columns.athlete_id, how="left")
    )

    primary_feature_columns = [
        "alpha_hat",
        "half_life_sessions",
        "r_squared",
        "injury_rate_injured",
        "mean_recovery_score",
    ]
    validation_feature_columns = [
        "mean_sleep_quality",
        "mean_workload",
        "mean_stress_level",
        "mean_hydration",
        "mean_heart_rate",
        "mean_muscle_activity",
        "mean_jump_height",
        "mean_gait_speed",
        "mean_range_of_motion",
        "drop_y_5",
        "rec_post3",
        "u_drop_3",
        "age",
        "bmi",
    ]
    profile_feature_columns = primary_feature_columns + [
        "mean_sleep_quality",
        "mean_workload",
        "rec_post3",
        "drop_y_5",
        "u_drop_3",
    ]
    return feature_frame, primary_feature_columns, validation_feature_columns, profile_feature_columns


def safe_silhouette(X: np.ndarray, labels: np.ndarray) -> float:
    """Return NaN instead of failing when silhouette is undefined."""

    try:
        return silhouette_score(X, labels)
    except Exception:
        return np.nan


def repeated_gmm_selection(
    X: np.ndarray,
    max_components: int = 6,
    covariance_type: str = "full",
    seeds: Sequence[int] = tuple(range(15)),
    n_init: int = 20,
) -> tuple[pd.DataFrame, dict[int, dict[str, object]]]:
    """Run repeated GMM fits to separate stable solutions from lucky starts."""

    selection_rows: list[dict[str, float]] = []
    best_models: dict[int, dict[str, object]] = {}

    for n_components in range(1, max_components + 1):
        run_rows = []
        label_runs = []
        best_bic = np.inf
        best_model = None
        best_labels = None
        best_probabilities = None
        best_silhouette = np.nan

        for seed in seeds:
            gmm = GaussianMixture(
                n_components=n_components,
                covariance_type=covariance_type,
                n_init=n_init,
                random_state=seed,
            )
            gmm.fit(X)
            probabilities = gmm.predict_proba(X)
            labels = gmm.predict(X)
            bic = gmm.bic(X)
            aic = gmm.aic(X)
            mean_max_prob = probabilities.max(axis=1).mean()
            mean_entropy = -(probabilities * np.log(np.clip(probabilities, 1e-12, 1))).sum(axis=1).mean()
            silhouette_value = np.nan if n_components == 1 else safe_silhouette(X, labels)

            run_rows.append(
                {
                    "seed": seed,
                    "n_components": n_components,
                    "bic": bic,
                    "aic": aic,
                    "mean_max_prob": mean_max_prob,
                    "mean_entropy": mean_entropy,
                    "silhouette": silhouette_value,
                }
            )

            if bic < best_bic:
                best_bic = bic
                best_model = gmm
                best_labels = labels
                best_probabilities = probabilities
                best_silhouette = silhouette_value

            if n_components > 1:
                label_runs.append(labels)

        run_frame = pd.DataFrame(run_rows)
        if n_components > 1:
            ari_scores = [
                adjusted_rand_score(label_runs[i], label_runs[j])
                for i in range(len(label_runs))
                for j in range(i + 1, len(label_runs))
            ]
            mean_ari = float(np.mean(ari_scores))
            min_ari = float(np.min(ari_scores))
        else:
            mean_ari = np.nan
            min_ari = np.nan

        selection_rows.append(
            {
                "n_components": n_components,
                "best_bic": run_frame["bic"].min(),
                "median_bic": run_frame["bic"].median(),
                "q25_bic": run_frame["bic"].quantile(0.25),
                "q75_bic": run_frame["bic"].quantile(0.75),
                "best_aic": run_frame["aic"].min(),
                "median_max_prob": run_frame["mean_max_prob"].median(),
                "median_entropy": run_frame["mean_entropy"].median(),
                "best_silhouette": best_silhouette,
                "mean_ari": mean_ari,
                "min_ari": min_ari,
            }
        )
        best_models[n_components] = {
            "model": best_model,
            "labels": best_labels,
            "probabilities": best_probabilities,
        }

    return pd.DataFrame(selection_rows), best_models


def select_stable_component_count(selection_table: pd.DataFrame, stability_threshold: float = 0.90) -> int:
    """Choose the smallest well-fitting stable mixture instead of the luckiest run."""

    stable_candidates = selection_table.loc[
        (selection_table["n_components"] > 1)
        & (selection_table["mean_ari"].fillna(0) >= stability_threshold)
    ].copy()
    if stable_candidates.empty:
        return int(selection_table.sort_values("best_bic").iloc[0]["n_components"])
    return int(stable_candidates.sort_values(["best_bic", "median_bic"]).iloc[0]["n_components"])


def _cluster_palette(cluster_labels: Sequence[str]) -> dict[str, str]:
    palette_values = ["#2a9d8f", "#e76f51", "#577590", "#8d99ae", "#bc6c25", "#6a4c93"]
    return {label: palette_values[idx] for idx, label in enumerate(cluster_labels)}


def fit_primary_em(
    df: pd.DataFrame,
    feature_frame: pd.DataFrame,
    primary_feature_columns: list[str],
    validation_feature_columns: list[str],
    profile_feature_columns: list[str],
    columns: ColumnSpec,
    max_components: int = 6,
    seeds: Sequence[int] = tuple(range(15)),
    n_init: int = 20,
    stability_threshold: float = 0.90,
) -> EMResult:
    """Fit the restricted EM model and compute the main validation outputs."""

    feature_block = feature_frame[primary_feature_columns].fillna(feature_frame[primary_feature_columns].median())
    scaler = StandardScaler()
    X_primary = scaler.fit_transform(feature_block)

    selection_table, best_models = repeated_gmm_selection(
        X_primary,
        max_components=max_components,
        covariance_type="full",
        seeds=seeds,
        n_init=n_init,
    )
    primary_k = select_stable_component_count(selection_table, stability_threshold=stability_threshold)

    pca = PCA(n_components=2, random_state=0)
    pca_coords = pca.fit_transform(X_primary)

    assignments = feature_frame.copy()
    assignments["cluster"] = best_models[primary_k]["labels"]
    assignments["cluster_prob"] = best_models[primary_k]["probabilities"].max(axis=1)
    assignments["cluster_entropy"] = -(
        best_models[primary_k]["probabilities"]
        * np.log(np.clip(best_models[primary_k]["probabilities"], 1e-12, 1))
    ).sum(axis=1)
    assignments["pc1"] = pca_coords[:, 0]
    assignments["pc2"] = pca_coords[:, 1]

    cluster_order = (
        assignments.groupby("cluster")["half_life_sessions"].mean().sort_values().index.tolist()
    )
    if primary_k == 3:
        label_map = {
            cluster_order[0]: "Fast recovery / lower burden",
            cluster_order[1]: "Intermediate recovery / higher burden",
            cluster_order[2]: "Slow recovery / high persistence",
        }
    else:
        label_map = {cluster_id: f"Recovery profile {i + 1}" for i, cluster_id in enumerate(cluster_order)}

    assignments["cluster_label"] = assignments["cluster"].map(label_map)
    cluster_label_order = [label_map[cluster_id] for cluster_id in cluster_order]
    cluster_palette = _cluster_palette(cluster_label_order)

    cluster_summary = (
        assignments.groupby("cluster_label")
        .agg(
            athletes=(columns.athlete_id, "size"),
            mean_alpha_hat=("alpha_hat", "mean"),
            mean_half_life=("half_life_sessions", "mean"),
            mean_r_squared=("r_squared", "mean"),
            mean_injury_rate=("injury_rate_injured", "mean"),
            mean_recovery_score=("mean_recovery_score", "mean"),
            mean_sleep_quality=("mean_sleep_quality", "mean"),
            mean_workload=("mean_workload", "mean"),
            mean_rec_post3=("rec_post3", "mean"),
            mean_drop_y_5=("drop_y_5", "mean"),
            mean_u_drop_3=("u_drop_3", "mean"),
            mean_assignment_prob=("cluster_prob", "mean"),
        )
        .reindex(cluster_label_order)
        .reset_index()
    )

    profile_heatmap_z = (
        assignments.groupby("cluster_label")[profile_feature_columns]
        .mean()
        .reindex(cluster_label_order)
        .apply(lambda column: (column - column.mean()) / column.std(ddof=0), axis=0)
    )

    validation_rows = []
    for variable in validation_feature_columns:
        group_values = [
            group[variable].dropna().values
            for _, group in assignments.groupby("cluster_label")
        ]
        if any(len(values) == 0 for values in group_values):
            continue

        _, anova_p = f_oneway(*group_values)
        _, kruskal_p = kruskal(*group_values)
        grand_mean = assignments[variable].mean()
        ss_between = sum(len(values) * (values.mean() - grand_mean) ** 2 for values in group_values)
        ss_total = ((assignments[variable] - grand_mean) ** 2).sum()
        eta_sq = ss_between / ss_total if ss_total > 0 else np.nan
        validation_rows.append(
            {
                "variable": variable,
                "anova_pvalue": anova_p,
                "kruskal_pvalue": kruskal_p,
                "eta_squared": eta_sq,
            }
        )
    validation_continuous = (
        pd.DataFrame(validation_rows).sort_values("eta_squared", ascending=False).reset_index(drop=True)
    )

    categorical_rows = []
    for categorical_variable in [columns.sport_type, columns.gender]:
        table = pd.crosstab(assignments["cluster_label"], assignments[categorical_variable])
        chi2, pvalue, _, _ = chi2_contingency(table)
        n_total = table.values.sum()
        phi2 = chi2 / n_total
        r_dim, c_dim = table.shape
        cramers_v = np.sqrt(phi2 / max(min(r_dim - 1, c_dim - 1), 1))
        categorical_rows.append(
            {
                "variable": categorical_variable,
                "chi_square_pvalue": pvalue,
                "cramers_v": cramers_v,
            }
        )
    validation_categorical = pd.DataFrame(categorical_rows)

    cluster_onset_summary = cluster_onset_trajectories(df, assignments, columns, cluster_label_order)
    cluster_transition_summary = cluster_specific_transitions(df, assignments, columns, cluster_label_order)

    return EMResult(
        feature_frame=feature_frame,
        selection_table=selection_table,
        primary_k=primary_k,
        assignments=assignments,
        cluster_summary=cluster_summary,
        validation_continuous=validation_continuous,
        validation_categorical=validation_categorical,
        cluster_onset_summary=cluster_onset_summary,
        cluster_transition_summary=cluster_transition_summary,
        profile_heatmap_z=profile_heatmap_z,
        cluster_label_order=cluster_label_order,
        cluster_palette=cluster_palette,
        primary_feature_columns=primary_feature_columns,
        validation_feature_columns=validation_feature_columns,
    )


def cluster_onset_trajectories(
    df: pd.DataFrame,
    assignments: pd.DataFrame,
    columns: ColumnSpec,
    cluster_label_order: Sequence[str],
) -> pd.DataFrame:
    """Summarize session-level behavior around injury onset within each profile."""

    df_with_cluster = df.merge(
        assignments[[columns.athlete_id, "cluster_label"]],
        on=columns.athlete_id,
        how="left",
    )
    rows = []
    for _, athlete_df in df_with_cluster.groupby(columns.athlete_id):
        athlete_df = athlete_df.reset_index(drop=True)
        cluster_label = athlete_df["cluster_label"].iloc[0]
        onset_indices = athlete_df.index[athlete_df["injury_onset"]].tolist()
        for idx in onset_indices:
            for rel_session in range(-2, 4):
                j = idx + rel_session
                if 0 <= j < len(athlete_df):
                    rows.append(
                        {
                            "cluster_label": cluster_label,
                            "rel_session": rel_session,
                            "u_t": athlete_df.loc[j, "u_t"],
                            "y_t": athlete_df.loc[j, "y_t"],
                            "recovery_score": athlete_df.loc[j, columns.recovery],
                        }
                    )

    return (
        pd.DataFrame(rows)
        .groupby(["cluster_label", "rel_session"])[["u_t", "y_t", "recovery_score"]]
        .mean()
        .reset_index()
    )


def cluster_specific_transitions(
    df: pd.DataFrame,
    assignments: pd.DataFrame,
    columns: ColumnSpec,
    cluster_label_order: Sequence[str],
) -> pd.DataFrame:
    """Compare pooled and profile-specific transition behavior."""

    transition_panel = df[
        [columns.athlete_id, "u_t", "y_t_smooth_past", columns.recovery, columns.sleep, columns.injury_var]
    ].copy()
    transition_panel["y_next_smooth_past"] = transition_panel.groupby(columns.athlete_id)["y_t_smooth_past"].shift(-1)
    transition_panel = transition_panel.dropna().copy()
    transition_panel = transition_panel.merge(
        assignments[[columns.athlete_id, "cluster_label"]],
        on=columns.athlete_id,
        how="left",
    )

    pooled_model = sm.OLS(
        transition_panel["y_next_smooth_past"],
        sm.add_constant(transition_panel[["y_t_smooth_past", "u_t"]], has_constant="add"),
    ).fit()
    transition_panel["pooled_pred"] = pooled_model.predict(
        sm.add_constant(transition_panel[["y_t_smooth_past", "u_t"]], has_constant="add")
    )
    transition_panel["pooled_residual"] = transition_panel["y_next_smooth_past"] - transition_panel["pooled_pred"]

    rows = []
    for cluster_label in cluster_label_order:
        cluster_frame = transition_panel.loc[transition_panel["cluster_label"] == cluster_label]
        cluster_model = sm.OLS(
            cluster_frame["y_next_smooth_past"],
            sm.add_constant(cluster_frame[["y_t_smooth_past", "u_t"]], has_constant="add"),
        ).fit()
        cluster_pred = cluster_model.predict(
            sm.add_constant(cluster_frame[["y_t_smooth_past", "u_t"]], has_constant="add")
        )
        rows.append(
            {
                "cluster_label": cluster_label,
                "sessions": len(cluster_frame),
                "alpha_hat_cluster": cluster_model.params["u_t"],
                "rho_hat_cluster": cluster_model.params["y_t_smooth_past"],
                "k_hat_cluster": 1 - cluster_model.params["y_t_smooth_past"],
                "adj_r_squared_cluster": cluster_model.rsquared_adj,
                "rmse_cluster_specific_fit": np.sqrt(np.mean((cluster_frame["y_next_smooth_past"] - cluster_pred) ** 2)),
                "rmse_pooled_fit_same_rows": np.sqrt(np.mean((cluster_frame["y_next_smooth_past"] - cluster_frame["pooled_pred"]) ** 2)),
                "mean_pooled_residual": cluster_frame["pooled_residual"].mean(),
                "mae_pooled_residual": cluster_frame["pooled_residual"].abs().mean(),
                "mean_recovery_score": cluster_frame[columns.recovery].mean(),
                "mean_sleep_quality": cluster_frame[columns.sleep].mean(),
                "injured_session_rate": (cluster_frame[columns.injury_var] == 2).mean(),
            }
        )
    return pd.DataFrame(rows)
