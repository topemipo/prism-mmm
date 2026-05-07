import hashlib
import numpy as np
import pandas as pd
import mlflow


def get_data_hash(df: pd.DataFrame) -> str:
    return hashlib.md5(pd.util.hash_pandas_object(df).values.tobytes()).hexdigest()[:8]


def log_transformation_run(
    params: np.ndarray,
    media_cols: list[str],
    mode: str,
    baseline_nrmse: float,
    best_nrmse: float,
    df_transformed: pd.DataFrame,
    transformed_csv_path: str,
) -> None:
    """Log hyperparameters, NRMSE metrics, and the transformed CSV artifact."""
    mlflow.log_param("mode", mode)
    mlflow.log_param("data_hash", get_data_hash(df_transformed))

    for i, ch in enumerate(media_cols):
        mlflow.log_param(f"{ch}_decay_rate", round(float(params[i * 3]),     4))
        mlflow.log_param(f"{ch}_K",          round(float(params[i * 3 + 1]), 4))
        mlflow.log_param(f"{ch}_S",          round(float(params[i * 3 + 2]), 4))

    mlflow.log_metric("baseline_nrmse",  baseline_nrmse)
    mlflow.log_metric("best_nrmse",      best_nrmse)
    mlflow.log_metric("nrmse_improvement", round(baseline_nrmse - best_nrmse, 6))

    mlflow.log_artifact(transformed_csv_path, artifact_path="data")


def log_model_validity(results) -> None:
    """Step 1 — F-statistic, R², Adj R²."""
    mlflow.log_metric("f_statistic",  round(results.fvalue,        4))
    mlflow.log_metric("f_pvalue",     round(results.f_pvalue,      6))
    mlflow.log_metric("r_squared",    round(results.rsquared,       4))
    mlflow.log_metric("adj_r_squared", round(results.rsquared_adj, 4))
    mlflow.log_metric("r2_adj_gap",   round(results.rsquared - results.rsquared_adj, 4))
    mlflow.log_param("f_test_pass",   results.f_pvalue < 0.05)


def log_line_assumptions(
    dw_stat: float,
    bp_pval: float,
    jb_pval: float,
    jb_skew: float,
    jb_kurtosis: float,
) -> None:
    """Step 2 — LINE assumption test results."""
    mlflow.log_metric("durbin_watson",       round(dw_stat,     4))
    mlflow.log_metric("breusch_pagan_pval",  round(bp_pval,     4))
    mlflow.log_metric("jarque_bera_pval",    round(jb_pval,     4))
    mlflow.log_metric("jb_skewness",         round(jb_skew,     4))
    mlflow.log_metric("jb_excess_kurtosis",  round(jb_kurtosis, 4))

    mlflow.log_param("dw_pass",  1.5 < dw_stat < 2.5)
    mlflow.log_param("bp_pass",  bp_pval > 0.05)
    mlflow.log_param("jb_pass",  jb_pval > 0.05)


def log_variable_health(results, vif_df: pd.DataFrame) -> None:
    """Step 3 — p-values summary + VIF summary. Full tables saved as artifacts."""
    insignificant  = results.pvalues[results.pvalues >= 0.05].index.tolist()
    media_insig    = [v for v in insignificant if "adstock" in v]

    mlflow.log_metric("insignificant_variable_count", len(insignificant))
    mlflow.log_param("insignificant_media_channels",
                     ", ".join(media_insig) if media_insig else "None")

    high_vif = vif_df[vif_df["Flag"].str.startswith("HIGH")]["Feature"].tolist()
    mod_vif  = vif_df[vif_df["Flag"].str.startswith("MODERATE")]["Feature"].tolist()

    mlflow.log_metric("vif_max",            round(vif_df["VIF"].max(), 2))
    mlflow.log_metric("vif_high_count",     len(high_vif))
    mlflow.log_metric("vif_moderate_count", len(mod_vif))
    mlflow.log_param("vif_high_features",   ", ".join(high_vif) if high_vif else "None")
    mlflow.log_param("vif_moderate_features", ", ".join(mod_vif) if mod_vif else "None")

    coef_df = pd.DataFrame({
        "feature":     results.params.index,
        "coefficient": results.params.values,
        "t_statistic": results.tvalues.values,
        "p_value":     results.pvalues.values,
    }).round(4)
    coef_df.to_csv("/tmp/coefficients.csv", index=False)
    vif_df.to_csv("/tmp/vif_table.csv", index=False)
    mlflow.log_artifact("/tmp/coefficients.csv", artifact_path="tables")
    mlflow.log_artifact("/tmp/vif_table.csv",    artifact_path="tables")


def log_prediction_accuracy(
    rse: float,
    rmse: float,
    mape: float,
    mean_y: float,
) -> None:
    """Step 4 — RSE, RMSE, MAPE."""
    mlflow.log_metric("rse",      round(rse,  2))
    mlflow.log_metric("rmse",     round(rmse, 2))
    mlflow.log_metric("mape",     round(mape, 4))
    mlflow.log_metric("rse_pct",  round(rse  / mean_y * 100, 2))
    mlflow.log_metric("rmse_pct", round(rmse / mean_y * 100, 2))


def log_anomalies(
    outliers: np.ndarray,
    high_lev: np.ndarray,
    influential: np.ndarray,
) -> None:
    """Step 5 — Counts as metrics, indices as tags."""
    mlflow.log_metric("outlier_count",     len(outliers))
    mlflow.log_metric("high_leverage_count", len(high_lev))
    mlflow.log_metric("influential_count", len(influential))

    both = np.intersect1d(outliers, influential)
    mlflow.log_metric("outlier_and_influential_count", len(both))

    mlflow.set_tag("outlier_indices",     str(list(map(int, outliers))))
    mlflow.set_tag("influential_indices", str(list(map(int, influential))))
    mlflow.set_tag("both_indices",        str(list(map(int, both))))


def log_diagnostic_plots(plot_dir: str) -> None:
    """Log all saved diagnostic plots as artifacts."""
    import os
    for fname in os.listdir(plot_dir):
        if fname.endswith(".png"):
            mlflow.log_artifact(os.path.join(plot_dir, fname), artifact_path="plots")
