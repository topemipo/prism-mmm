import numpy as np
import pandas as pd

def geometric_adstock(spend: np.ndarray, decay_rate: float) -> np.ndarray:
    """
    Apply geometric adstock transformation to a media spend series.

    The adstocked value at time t is:
        adstock[t] = spend[t] + decay_rate * adstock[t-1]

    This models advertising carryover: spend in week t continues to
    influence sales in weeks t+1, t+2, ... but that effect decays
    geometrically at `decay_rate` per week.

    Args:
        spend: Raw weekly media spend values (1D array).
        decay_rate: Decay rate in [0, 1).
            0.0 = no carryover (each week is independent).
            0.5 = half the previous week's effect carries forward.
            0.9 = strong carryover — advertising lingers for many weeks.

    Returns:
        Adstocked spend series, same length as input.

    Raises:
        ValueError: If decay_rate is not in [0, 1).
    """
    if not 0.0 <= decay_rate < 1.0:
        raise ValueError(f"decay_rate must be in [0, 1]. Got: {decay_rate}")

    adstocked = np.zeros(len(spend))
    adstocked[0] = spend[0]

    for t in range(1, len(spend)):
        adstocked[t] = spend[t] + decay_rate * adstocked[t - 1]

    return adstocked


def apply_adstock_to_channels(
    df: pd.DataFrame,
    channel_cols: list[str],
    decay_rates: dict[str, float]) -> pd.DataFrame:
    """
    Apply geometric adstock to multiple channels with per-channel decay rates.

    Args:
        df: Input DataFrame containing raw spend columns.
        channel_cols: Column names to transform.
        decay_rates: Mapping of column name to decay rate.
            Channels not in this dict receive a default rate of 0.5.

    Returns:
        New DataFrame with adstocked columns added (originals preserved).
        New columns are named: {original_col}_adstock
    """
    df_out = df.copy()
    for col in channel_cols:
        rate = decay_rates.get(col, 0.5)
        df_out[f"{col}_adstock"] = geometric_adstock(df[col].values, rate)
    return df_out


def hill_saturation(x: np.ndarray, K: float, S: float) -> np.ndarray:
    """
    Apply Hill saturation function to model diminishing returns.

    The Hill function:
        hill(x) = x^S / (K^S + x^S)

    Output is always in [0, 1]. This requires x to be normalised to [0, 1]
    before calling this function (see apply_saturation_to_channels).

    Args:
        x: Input values, expected in [0, 1] range (normalised spend).
        K: Half-saturation point. The spend level at which response reaches
           50% of maximum. Higher K = channel saturates at higher spend levels.
           Must be positive.
        S: Shape/slope parameter.
           S < 1 = concave curve (immediate diminishing returns from £1 spent).
           S = 1 = Michaelis-Menten curve.
           S > 1 = S-shaped curve (slow start, rapid growth, then saturation).
           Must be positive.

    Returns:
        Saturated values in [0, 1].

    Raises:
        ValueError: If K or S are not positive.
    """
    if K <= 0:
        raise ValueError(f"K must be positive. Got: {K}")
    if S <= 0:
        raise ValueError(f"S must be positive. Got: {S}")

    # ensures all cells in spend channel is greater than or equal to zero
    x_safe = np.where(x < 0, 0.0, x)

    return (x_safe ** S) / (K ** S + x_safe ** S)


def apply_saturation_to_channels(
    df: pd.DataFrame,
    normalised_cols: list[str],
    K_values: dict[str, float],
    S_values: dict[str, float]) -> pd.DataFrame:
    """
    Apply Hill saturation to normalised adstocked columns.

    Args:
        df: DataFrame containing normalised adstock columns.
        normalised_cols: Column names to transform (should be in [0, 1]).
        K_values: Mapping of column name to K parameter.
        S_values: Mapping of column name to S parameter.

    Returns:
        New DataFrame with saturated columns added.
        New columns are named: {original_col}_sat
    """
    df_out = df.copy()
    for col in normalised_cols:
        K = K_values.get(col, 0.5)
        S = S_values.get(col, 2.0)
        df_out[f"{col}_sat"] = hill_saturation(df[col].values, K=K, S=S)
    return df_out


def apply_pipeline(
    df: pd.DataFrame,
    params: np.ndarray,
    media_cols: list[str],
    adstock_cols: list[str],
    norm_cols: list[str],
) -> pd.DataFrame:
    """
    Apply adstock → normalise → Hill saturation for a flat params array.

    params layout (3 values × n channels):
        [decay_rate, K, S] for each channel in media_cols order.

    Args:
        df: Raw DataFrame containing media spend columns.
        params: Flat array of [decay_rate, K, S] per channel.
        media_cols: Raw spend column names (e.g. ['tv_S', 'ooh_S']).
        adstock_cols: Adstocked column names (e.g. ['tv_S_adstock']).
        norm_cols: Normalised adstock column names (e.g. ['tv_S_adstock_norm']).

    Returns:
        DataFrame with adstocked, normalised, and saturated columns added.
    """
    decay_rates = {ch:       params[i * 3]     for i, ch       in enumerate(media_cols)}
    K_values    = {norm_col: params[i * 3 + 1] for i, norm_col in enumerate(norm_cols)}
    S_values    = {norm_col: params[i * 3 + 2] for i, norm_col in enumerate(norm_cols)}

    df_a = apply_adstock_to_channels(df, media_cols, decay_rates)
    df_n = normalise_cols(df_a, adstock_cols)
    df_s = apply_saturation_to_channels(df_n, norm_cols, K_values, S_values)
    return df_s


def objective(
    params: np.ndarray,
    df_raw: pd.DataFrame,
    media_cols: list[str],
    adstock_cols: list[str],
    norm_cols: list[str],
    sat_cols: list[str],
    control_cols: list[str],
    val_weeks: int = 20,
) -> float:
    """
    Validation NRMSE for a given set of transformation parameters.

    Fits OLS on the training split and evaluates on the held-out validation
    split. Used as the objective function for differential_evolution.

    Args:
        params: Flat array of [decay_rate, K, S] per channel.
        df_raw: Raw DataFrame (pre-transformation).
        media_cols: Raw spend column names.
        adstock_cols: Adstocked column names.
        norm_cols: Normalised adstock column names.
        sat_cols: Saturated column names used as model predictors.
        control_cols: Control variable column names.
        val_weeks: Number of weeks to hold out for validation.

    Returns:
        NRMSE on the validation split. Returns 1.0 on any error so invalid
        parameter combinations are penalised without crashing the search.
    """
    import statsmodels.api as sm
    from model import build_design_matrix, fit_ols

    try:
        df_t  = apply_pipeline(df_raw, params, media_cols, adstock_cols, norm_cols)
        train = df_t.iloc[:-val_weeks].copy()
        val   = df_t.iloc[-val_weeks:].copy()

        X_train = build_design_matrix(train, sat_cols, control_cols)
        y_train = train['revenue'].reset_index(drop=True)
        res     = fit_ols(X_train, y_train)

        X_val       = build_design_matrix(val, sat_cols, control_cols)
        X_val_const = sm.add_constant(X_val, has_constant='add')
        y_pred      = res.predict(X_val_const)
        y_val       = val['revenue'].values

        nrmse = np.sqrt(np.mean((y_val - y_pred) ** 2)) / (y_val.max() - y_val.min())
        return float(nrmse)
    except Exception:
        return 1.0


def normalise_cols(
    df: pd.DataFrame,
    cols_to_normalise: list[str]) -> tuple[pd.DataFrame, list[str]]:
    """
    Normalise media columns to [0, 1] range.

    Required before applying Hill saturation because the K parameter
    assumes inputs are proportions between 0 and 1, not raw pound values.
    Each column is divided by its own maximum value so that the busiest
    week for each channel becomes 1.0 and all other weeks are proportions
    of that maximum.

    This function works for both transformation orderings:
        - Adstock first:      pass adstocked columns  (e.g. tv_S_adstock)
        - Saturation first:   pass raw spend columns  (e.g. tv_S)

    The output column name is always {input_col}_norm, which naturally
    reflects whichever ordering was used.

    Args:
        df: DataFrame containing the columns to normalise.
        cols_to_normalise: List of column names to normalise.

    Returns:
        Tuple of:
            - DataFrame with normalised columns added (originals preserved).
              New columns follow the pattern {input_col}_norm.
            - List of the new normalised column names.

    Raises:
        ValueError: If any column contains only zeros or negative values,
                    making normalisation impossible.
    """
    df_out     = df.copy()

    for col in cols_to_normalise:
        col_max  = df_out[col].max()

        if col_max <= 0:
            raise ValueError(
                f"Column '{col}' has a maximum value of {col_max}. "
                f"Cannot normalise a column that is entirely zero or negative — "
                f"check that the correct columns are being passed in."
            )

        df_out[f"{col}_norm"] = df_out[col] / col_max
    return df_out


    