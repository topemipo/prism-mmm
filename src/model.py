import pandas as pd
import numpy as np
import statsmodels.api as sm
import logging

logger = logging.getLogger(__name__)


def build_design_matrix(
    df: pd.DataFrame,
    sat_cols: list[str],
    control_cols: list[str]) -> pd.DataFrame:
    """
    Assemble the full design matrix for OLS regression.

    All preparation (adstock, normalisation, saturation, encoding of
    control variables) is done upstream in the transformations notebook.
    This function simply selects and combines the relevant columns.

    Args:
        df: Transformed DataFrame (output from transformations notebook).
        sat_cols: Names of the saturated media predictor columns.
        control_cols: Names of the prepared control variable columns
                      (e.g. competitor_sales_norm, is_holiday, event
                      dummies, month dummies).

    Returns:
        Design matrix X (without constant — statsmodels adds it separately).
    """
    X = df[sat_cols + control_cols].copy().reset_index(drop=True).astype(float)
    logger.info(f"Design matrix assembled: {X.shape[0]} rows, {X.shape[1]} columns")
    return X


def fit_ols(
    X: pd.DataFrame,
    y: pd.Series,
    cov_type: str = 'nonrobust') -> sm.regression.linear_model.RegressionResultsWrapper:
    """
    Fit an OLS model with a constant term added by statsmodels.

    Args:
        X: Design matrix (predictors, without constant).
        y: Target variable (revenue).

    Returns:
        Fitted statsmodels OLS results object.
    """
    X_with_const = sm.add_constant(X)
    model = sm.OLS(y, X_with_const)
    results = model.fit(cov_type=cov_type)
    logger.info(f"OLS fitted. R²={results.rsquared:.4f}, Adj R²={results.rsquared_adj:.4f}")
    return results


def fit_wls(
    X: pd.DataFrame,
    y: pd.Series,
) -> sm.regression.linear_model.RegressionResultsWrapper:
    """
    Fit a two-step Feasible WLS model.

    Step 1: OLS to get fitted values.
    Step 2: WLS with weights = 1 / fitted_values², down-weighting weeks
            where revenue (and thus variance) is high.

    Args:
        X: Design matrix (predictors, without constant).
        y: Target variable (revenue).

    Returns:
        Fitted statsmodels WLS results object.
    """
    X_with_const = sm.add_constant(X)

    ols_fitted = sm.OLS(y, X_with_const).fit().fittedvalues
    weights = 1.0 / (ols_fitted ** 2)

    model   = sm.WLS(y, X_with_const, weights=weights)
    results = model.fit()
    logger.info(f"WLS fitted. R²={results.rsquared:.4f}, Adj R²={results.rsquared_adj:.4f}")
    return results