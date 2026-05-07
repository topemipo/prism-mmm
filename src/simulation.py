import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)


def simulate_revenue(
    allocation: dict[str, float],
    roi_table: pd.DataFrame,
    baseline_revenue: float
) -> dict:
    """
    Project total revenue given a media budget allocation.

    Uses a linear projection: projected_revenue = baseline + sum(spend * ROI).

    Args:
        allocation: Channel name to spend amount mapping.
        roi_table: Output from calculate_media_roi().
        baseline_revenue: Total baseline revenue from decomposition.

    Returns:
        Dictionary with keys: total_budget, projected_revenue,
        incremental_from_media, baseline_revenue, blended_roi.
    """
    roi_map       = dict(zip(roi_table['Channel'], roi_table['ROI']))
    incremental   = sum(allocation.get(ch, 0) * roi for ch, roi in roi_map.items())
    total_revenue = baseline_revenue + incremental
    total_spend   = sum(allocation.values())
    blended_roi   = incremental / total_spend if total_spend > 0 else 0.0

    return {
        'total_budget':           total_spend,
        'projected_revenue':      total_revenue,
        'incremental_from_media': incremental,
        'baseline_revenue':       baseline_revenue,
        'blended_roi':            blended_roi
    }


def optimise_budget(
    total_budget: float,
    roi_table: pd.DataFrame,
    baseline_revenue: float,
    min_share: float = 0.02,
    max_share: float = 0.70
) -> dict:
    """
    Find the channel allocation that maximises projected revenue.

    Because simulate_revenue uses fixed ROI values the objective is linear,
    so the optimal solution is greedy: allocate budget to channels in
    descending ROI order, up to max_share each, with min_share guaranteed
    to every channel.

    Args:
        total_budget: Total media budget to allocate (£).
        roi_table: Output from calculate_media_roi().
        baseline_revenue: Non-media baseline revenue.
        min_share: Minimum fraction per channel (default 2%).
        max_share: Maximum fraction per channel (default 70%).

    Returns:
        Dictionary with: optimal_allocation, projected_revenue,
        incremental_from_media, optimisation_success.
    """
    ranked    = roi_table.sort_values('ROI', ascending=False)
    channels  = ranked['Channel'].tolist()
    n         = len(channels)

    shares    = {ch: min_share for ch in channels}
    remaining = 1.0 - n * min_share

    for ch in channels:
        headroom    = max_share - shares[ch]
        add         = min(headroom, remaining)
        shares[ch] += add
        remaining  -= add
        if remaining <= 1e-10:
            break

    optimal_allocation = {ch: round(shares[ch] * total_budget, 2) for ch in shares}
    projected = simulate_revenue(optimal_allocation, roi_table, baseline_revenue)

    return {
        'optimal_allocation':     optimal_allocation,
        'projected_revenue':      projected['projected_revenue'],
        'incremental_from_media': projected['incremental_from_media'],
        'optimisation_success':   True
    }