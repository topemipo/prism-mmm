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