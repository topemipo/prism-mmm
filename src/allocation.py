import numpy as np
import pandas as pd
from scipy.optimize import minimize

from transformations import geometric_adstock, hill_saturation

def steady_state_adstock(spend, decay, n_periods=200):
    """
    Steady-state adstock under sustained weekly spend.
    Simulates n_periods of constant spend through geometric_adstock
    and returns the converged final value.
    """
    constant_series = np.full(n_periods, spend)
    adstock_series = geometric_adstock(constant_series, decay)
    return adstock_series[-1]


def response(spend, decay, coef, K, S, adstock_max):
    """
    Predicted weekly revenue contribution from a channel
    at a given £ spend level (assuming sustained weekly spend).
    
    Uses simulation through geometric_adstock to compute the
    steady-state adstock, rather than a closed-form formula.
    """
    adstock_steady = steady_state_adstock(spend, decay)
    normalised_spend = adstock_steady / adstock_max
    saturated = hill_saturation(normalised_spend, K=K, S=S)
    return coef * saturated


# def response(spend, decay, coef, K, S, adstock_max):
#     """
#     Predicted weekly revenue contribution from a channel
#     at a given £ spend level.
    
#     spend:        £ spend per week (raw, not adstocked or normalised)
#     decay:        carryover effect of this channel
#     coef:         the OLS coefficient for this channel
#     K, S:         Hill saturation parameters
#     adstock_max:  the normalisation factor for this channel
    
#     NOTE: This is a simplification — it treats current spend as both
#     the immediate spend AND the steady-state adstock level. In a real
#     Robyn allocator you'd handle adstock dynamics over multiple weeks.
#     For a 2-hour take-home this approximation is standard.
#     """
#     adstock_steady = spend / (1 - decay)
#     normalised_spend = adstock_steady / adstock_max
#     saturated = hill_saturation(normalised_spend, K=K, S=S)
#     return coef * saturated


def marginal_roi(spend, coef, K, S, adstock_max, decay, increment=1.0):
    """
    Marginal ROI: extra revenue per extra £1 of spend at this spend level.
    Computed numerically as a small finite difference.
    """
    response_now   = response(spend, decay, coef, K, S, adstock_max)
    response_after = response(spend + increment, decay, coef, K, S, adstock_max)
    return (response_after - response_now) / increment


def total_response(spend_array, media_cols, coefs, hill_params, channel_state):
    """
    Total weekly revenue contribution across all channels at a given spend allocation.
    
    spend_array:   numpy array of length 5, one £ spend per channel
    media_cols:    list of channel keys (e.g. MEDIA_COLS), defines order of spend_array
    coefs:         dict mapping channel key to OLS coefficient
    hill_params:   dict mapping channel key to {'decay', 'K', 'S'}
    channel_state: dict mapping channel key to state info (we need 'adstock_max')
    
    Returns: total predicted weekly response in £
    """
    total = 0.0
    for i, col in enumerate(media_cols):
        spend = spend_array[i]
        params = hill_params[col]
        state = channel_state[col]
        
        channel_response = response(
            spend=spend,
            coef=coefs[col],
            K=params['K'],
            S=params['S'],
            adstock_max=state['adstock_max'],
            decay=params['decay'],
        )
        total += channel_response
    
    return total


def neg_total_response(spend_array, *args):
    """
    Negative of total_response. scipy.optimize.minimize finds minima,
    so to maximise response we minimise its negative.
    """
    return -total_response(spend_array, *args)


def make_bounds(current_spend, lower_pct, upper_pct):
    """Build bounds list given a percentage range from current spend."""
    bounds = []
    for current in current_spend:
        lower = max(0, current * lower_pct)
        upper = current * upper_pct
        bounds.append((lower, upper))
    return bounds


def budget_constraint(spend_array, target_budget):
    """
    Equality constraint: total spend - target = 0
    scipy enforces this == 0 by treating it as 'fun(x) = 0' equality.
    """
    return spend_array.sum() - target_budget


def run_optimisation(
    target_budget,
    scenario_label,
    current_spend,
    bounds,
    media_cols,
    coefs,
    hill_params,
    channel_state,
):
    """
    Run the optimiser for a given target budget and return the result.
    """
    constraint = {
        'type': 'eq',
        'fun': budget_constraint,
        'args': (target_budget,),
    }
    
    result = minimize(
        fun=neg_total_response,
        x0=current_spend,
        args=(media_cols, coefs, hill_params, channel_state),
        method='SLSQP',
        bounds=bounds,
        constraints=[constraint],
        options={'ftol': 1e-8, 'maxiter': 200, 'disp': False},
    )
    
    return {
        'scenario':      scenario_label,
        'target_budget': target_budget,
        'spend':         result.x,
        'response':      -result.fun,
        'success':       result.success,
        'message':       result.message,
    }


def build_comparison(
    scenario,
    current_spend,
    media_cols,
    coefs,
    hill_params,
    channel_state,
    channel_labels,
):
    """
    Take a scenario result dict and produce a per-channel comparison DataFrame
    plus a headline summary.

    uplift is calculated with reference to the inital (original) response.
    if you want to get uplift with reference to somehting else, it's outside the scope of this function
    """
    optimised_spend = scenario['spend']
    optimised_total_response = scenario['response']
    
    rows = []
    for i, col in enumerate(media_cols):
        current = current_spend[i]
        recommended = optimised_spend[i]
        change_pct = (recommended - current) / current * 100
        
        current_resp = response(
            spend=current,
            coef=coefs[col],
            K=hill_params[col]['K'],
            S=hill_params[col]['S'],
            adstock_max=channel_state[col]['adstock_max'],
            decay=hill_params[col]['decay'],
        )
        recommended_resp = response(
            spend=recommended,
            coef=coefs[col],
            K=hill_params[col]['K'],
            S=hill_params[col]['S'],
            adstock_max=channel_state[col]['adstock_max'],
            decay=hill_params[col]['decay'],
        )
        
        rows.append({
            'Channel':           channel_labels[col],
            'Current Spend':     round(current, 0),
            'Recommended Spend': round(recommended, 0),
            'Change %':          round(change_pct, 1),
            'Current Response':  round(current_resp, 0),
            'New Response':      round(recommended_resp, 0),
        })
    
    comparison_df = pd.DataFrame(rows)
    
    current_total_response = total_response(
        current_spend, media_cols, coefs, hill_params, channel_state
    )
    uplift = optimised_total_response - current_total_response
    uplift_pct = uplift / current_total_response * 100
    
    summary = {
        'scenario':               scenario['scenario'],
        'target_budget':          scenario['target_budget'],
        'current_total_response': current_total_response,
        'new_total_response':     optimised_total_response,
        'uplift':                 uplift,
        'uplift_pct':             uplift_pct,
        'total_spend_check':      optimised_spend.sum(),
    }
    
    return comparison_df, summary