import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


def decompose_revenue(
    results,
    X: pd.DataFrame,
    y: pd.Series,
    dates: pd.Series,
    media_sat_cols: list[str],
    control_cols: list[str],
    channel_labels: dict[str, str] | None = None,
) -> pd.DataFrame:
    """
    Decompose fitted revenue into weekly contributions per channel.

    Columns returned:
        DATE, one column per media channel, Non-media Drivers,
        Baseline + Non-media, Fitted, Actual.

    Args:
        results: Fitted statsmodels OLS results object.
        X: Design matrix used in fitting (without constant column).
        y: Actual revenue series.
        dates: DATE series aligned to X.
        media_sat_cols: Column names of the transformed media predictors in X.
        control_cols: Column names of the control variables in X.
        channel_labels: Human-readable names, e.g.
            {'tv_S_adstock_norm_sat': 'TV'}. If None, uses column names as-is.

    Returns:
        DataFrame with 208 rows and columns:
        DATE | TV | OOH | ... | Non-media Drivers | Baseline + Non-media | Fitted | Actual
    """
    if channel_labels is None:
        channel_labels = {col: col for col in media_sat_cols}

    params = results.params
    decomp = pd.DataFrame(index=X.index)

    decomp['DATE'] = dates.values

    for col in media_sat_cols:
        label         = channel_labels.get(col, col)
        coef          = params.get(col, 0.0)
        decomp[label] = coef * X[col].values

    non_media = np.zeros(len(X))
    for col in control_cols:
        if col in X.columns:
            coef       = params.get(col, 0.0)
            non_media += coef * X[col].values
    decomp['Non-media Drivers'] = non_media

    pure_baseline                  = params.get('const', 0.0)
    decomp['Baseline + Non-media'] = pure_baseline + non_media

    decomp['Fitted'] = results.fittedvalues.values
    decomp['Actual'] = y.values

    max_error = (
        decomp.drop(columns=['DATE', 'Baseline + Non-media', 'Fitted', 'Actual']).sum(axis=1)
        + pure_baseline
        - results.fittedvalues
    ).abs().max()
    if max_error > 1.0:
        logger.warning(f"Decomposition reconstruction error: £{max_error:.2f}")

    logger.info(f"Decomposition complete. Max reconstruction error: £{max_error:.4f}")
    return decomp


def calculate_media_roi(
    decomp: pd.DataFrame,
    df_raw: pd.DataFrame,
    media_spend_cols: list[str],
    channel_labels: dict[str, str]) -> pd.DataFrame:
    """
    Calculate return on investment per channel.

    ROI = total incremental revenue attributed to channel / total spend on channel.

    Args:
        decomp: Output from decompose_revenue().
        df_raw: Original DataFrame with raw spend columns.
        media_spend_cols: Raw spend column names (e.g. 'tv_S').
        channel_labels: Mapping from raw spend column to label used in decomp.

    Returns:
        DataFrame with columns: Channel, Total Spend, Revenue Contribution, ROI.
        Sorted by ROI descending.
    """
    rows = []
    for col in media_spend_cols:
        label                = channel_labels.get(col, col)
        total_spend          = float(df_raw[col].sum())
        revenue_contribution = float(decomp[label].sum()) if label in decomp.columns else 0.0
        roi                  = revenue_contribution / total_spend if total_spend > 0 else 0.0
        rows.append({
            'Channel':              label,
            'Total Spend':          round(total_spend, 0),
            'Revenue Contribution': round(revenue_contribution, 0),
            'ROI':                  round(roi, 2)
        })

    return pd.DataFrame(rows).sort_values('ROI', ascending=False).reset_index(drop=True)