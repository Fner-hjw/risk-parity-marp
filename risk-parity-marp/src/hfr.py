"""
src/hfr.py — Hedge Fund Replicator (Hasanhodzic & Lo, 2007).

Replicates hedge fund index returns using liquid factor-mimicking portfolios.
The core idea is a two-step mapping: hedge fund returns → factor returns →
tradable asset returns, allowing us to build a synthetic hedge fund index
from easily traded ETFs.

Public API
----------
build_factor_replicating_portfolios(asset_returns, factors)
    Constrained regression: each factor → portfolio of tradable assets.

replicate_hfrx(hedge_fund_returns, factor_returns, method="constrained")
    Replicate a hedge fund index using factor exposures.

build_hfrx_synthetic_index(factor_weights, factor_returns, asset_returns)
    Two-step synthetic index: hedge fund → factors → tradable assets.

compare_replication_quality(original_returns, synthetic_returns)
    Correlation, R², tracking error, and information ratio.

build_factor_mimicking_portfolios(asset_returns)
    Self-contained: constructs factor returns from ETFs, then builds
    factor-mimicking portfolios.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .conventions import RISK_FREE_ANNUAL


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def _daily_rf() -> float:
    """Convert the annual risk-free rate (from conventions) to daily."""
    return (1.0 + RISK_FREE_ANNUAL) ** (1.0 / 252.0) - 1.0


def _constrained_nnls(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Non-negative least squares with sum-to-1 constraint.

    Solves  min_w ||X w - y||²   subject to  w_i >= 0, sum(w) = 1.

    Uses SLSQP over a quadratic objective, consistent with the rest of
    the project's optimisation stack.
    """
    n_assets = X.shape[1]

    def objective(w: np.ndarray) -> float:
        residuals = X @ w - y
        return float(np.sum(residuals ** 2))

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bounds = [(0.0, 1.0) for _ in range(n_assets)]

    result = minimize(
        objective,
        np.ones(n_assets) / n_assets,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 2000, "ftol": 1e-12},
    )

    if not result.success:
        warnings.warn(
            f"Constrained NNLS did not converge: {result.message}"
        )

    return result.x


# --------------------------------------------------------------------------
# Factor-mimicking portfolios
# --------------------------------------------------------------------------

def build_factor_replicating_portfolios(
    asset_returns: pd.DataFrame,
    factors: pd.DataFrame,
) -> dict:
    """Construct factor-mimicking portfolios via constrained regression.

    For each factor (e.g. mkt_rf, smb, hml), regress its daily returns onto
    the daily returns of tradable assets with non-negativity and sum-to-1
    constraints.  The resulting weight vector is a long-only portfolio of
    tradable assets that best tracks the factor.

    Parameters
    ----------
    asset_returns : DataFrame (T × N)
        Daily returns of tradable assets (ETFs).
    factors : DataFrame (T × K)
        Daily factor returns.  Each column is a factor to replicate.

    Returns
    -------
    dict[str, pd.Series]
        Mapping from factor name to a pd.Series of portfolio weights
        indexed by asset names.  Weights sum to 1 for each factor.
    """
    # Align dates and drop assets that have any NaN in the common window
    common_idx = asset_returns.index.intersection(factors.index)
    if len(common_idx) < 2:
        raise ValueError(
            "asset_returns and factors share fewer than 2 dates; "
            "cannot build replicating portfolios."
        )

    X = asset_returns.loc[common_idx].dropna(axis=1)
    if X.shape[1] == 0:
        raise ValueError("No assets remain after dropping NaN columns.")

    result: dict[str, pd.Series] = {}
    asset_names = X.columns.tolist()

    for factor_name in factors.columns:
        y = factors.loc[common_idx, factor_name]

        # Drop rows where either X or y has NaN
        mask = ~(np.isnan(X.values).any(axis=1) | np.isnan(y.values))
        X_clean = X.values[mask]
        y_clean = y.values[mask]

        n_obs, n_assets = X_clean.shape
        if n_obs < n_assets + 5:
            warnings.warn(
                f"Factor '{factor_name}': only {n_obs} clean observations "
                f"for {n_assets} assets.  Estimates may be unstable."
            )

        if n_obs < 2:
            # Not enough data — fall back to equal weight
            w = np.ones(n_assets) / n_assets
        else:
            w = _constrained_nnls(X_clean, y_clean)

        # Clean up: clip tiny negatives, renormalise
        w = np.clip(w, 0.0, None)
        total = w.sum()
        if total <= 0.0:
            w = np.ones(n_assets) / n_assets
        else:
            w = w / total

        result[factor_name] = pd.Series(w, index=asset_names, name=factor_name)

    return result


# --------------------------------------------------------------------------
# Hedge fund replication
# --------------------------------------------------------------------------

def replicate_hfrx(
    hedge_fund_returns: pd.Series,
    factor_returns: pd.DataFrame,
    method: str = "constrained",
) -> pd.Series:
    """Replicate a hedge fund index's returns using factor exposures.

    Parameters
    ----------
    hedge_fund_returns : Series (T,)
        Daily returns of the hedge fund index (e.g. HFRX Global).
    factor_returns : DataFrame (T × K)
        Daily factor returns (e.g. mkt_rf, smb, hml, bond, gold).
    method : {"constrained", "ridge"}
        ``"constrained"`` — non-negative least squares with sum-to-1
            constraint.  Produces a long-only factor portfolio.
        ``"ridge"`` — ridge regression (L2 penalty with alpha=0.01).
            Weights are clipped to be non-negative and renormalised.

    Returns
    -------
    pd.Series
        Weight vector indexed by factor names, summing to 1.
    """
    if method not in ("constrained", "ridge"):
        raise ValueError(
            f"Unknown method {method!r}; expected 'constrained' or 'ridge'."
        )

    # Align
    common_idx = hedge_fund_returns.index.intersection(factor_returns.index)
    y = hedge_fund_returns.loc[common_idx]
    X = factor_returns.loc[common_idx]

    # Drop rows with any NaN
    mask = ~(np.isnan(X.values).any(axis=1) | np.isnan(y.values))
    X_arr = X.values[mask]
    y_arr = y.values[mask]

    n_obs, n_factors = X_arr.shape
    factor_names = X.columns.tolist()

    if n_obs < n_factors + 5:
        warnings.warn(
            f"Only {n_obs} clean observations for {n_factors} factors.  "
            "Replication weights may be unstable."
        )

    if n_obs < 2:
        # Degenerate fallback
        w = np.ones(n_factors) / n_factors
    elif method == "constrained":
        w = _constrained_nnls(X_arr, y_arr)
    elif method == "ridge":
        alpha = 0.01
        K = X_arr.T @ X_arr + alpha * np.eye(n_factors)
        w = np.linalg.solve(K, X_arr.T @ y_arr)

    # Post-process: clip negatives, renormalise
    w = np.clip(w, 0.0, None)
    total = w.sum()
    if total <= 0.0:
        w = np.ones(n_factors) / n_factors
    else:
        w = w / total

    return pd.Series(w, index=factor_names, name="weight")


# --------------------------------------------------------------------------
# Synthetic hedge fund index (two-step)
# --------------------------------------------------------------------------

def build_hfrx_synthetic_index(
    factor_weights: pd.Series,
    factor_returns: pd.DataFrame,
    asset_returns: pd.DataFrame,
) -> pd.Series:
    """Build a synthetic hedge fund index via two-step replication.

    Step 1 (already done) : hedge fund return → factor exposures.
        ``factor_weights`` is the output of :func:`replicate_hfrx`.

    Step 2 (this function): factor exposures → tradable assets.
        Each factor is tracked by a factor-mimicking portfolio of ETFs.
        The synthetic index return is the weighted sum of those
        mimicking-portfolio returns.

    Parameters
    ----------
    factor_weights : Series
        Weights mapping the hedge fund to factors (from replicate_hfrx).
    factor_returns : DataFrame (T × K)
        Daily factor returns used to build mimicking portfolios.
    asset_returns : DataFrame (T × N)
        Daily returns of tradable ETFs.

    Returns
    -------
    pd.Series
        Daily synthetic hedge fund return series, indexed by date.
    """
    # Build asset-level weights for each factor
    factor_portfolios = build_factor_replicating_portfolios(
        asset_returns, factor_returns
    )

    # Only keep factors present in both the weight vector and portfolios
    available = [f for f in factor_weights.index if f in factor_portfolios]
    if not available:
        raise ValueError(
            "No overlap between factor_weights index and factors available "
            "in factor_returns.  Check factor naming consistency."
        )

    w_factors = factor_weights.loc[available]
    w_factors = w_factors / w_factors.sum()

    # Compute mimicking-portfolio returns for each factor
    common_idx = asset_returns.index.intersection(factor_returns.index)
    aligned_assets = asset_returns.loc[common_idx]

    mimicking: dict[str, pd.Series] = {}
    for factor_name in available:
        asset_w = factor_portfolios[factor_name]
        # Reindex to match the aligned asset columns
        aligned_w = asset_w.reindex(aligned_assets.columns, fill_value=0.0)
        mimicking[factor_name] = (
            aligned_assets * aligned_w.values
        ).sum(axis=1)

    mimicking_df = pd.DataFrame(mimicking, index=common_idx)

    # Combine across factors using the hedge-fund-level factor weights
    synthetic = (mimicking_df * w_factors.values).sum(axis=1)
    synthetic.name = "HFRX_synthetic"

    return synthetic


# --------------------------------------------------------------------------
# Replication quality diagnostics
# --------------------------------------------------------------------------

def compare_replication_quality(
    original_returns: pd.Series,
    synthetic_returns: pd.Series,
) -> dict:
    """Compute replication quality metrics.

    Parameters
    ----------
    original_returns : Series
        Original hedge fund index daily returns.
    synthetic_returns : Series
        Synthetic (replicated) daily returns.

    Returns
    -------
    dict
        Keys:
        - ``correlation`` : Pearson correlation between the two series.
        - ``r_squared``   : R² = correlation² (fraction of variance explained).
        - ``tracking_error`` : Annualised tracking error
          (std of return difference × sqrt(252)).
        - ``information_ratio`` : Annualised information ratio
          (mean difference / std difference × sqrt(252)).
    """
    # Align on common dates
    common_idx = original_returns.index.intersection(synthetic_returns.index)
    orig = original_returns.loc[common_idx].dropna()
    synth = synthetic_returns.loc[common_idx].dropna()

    # Intersect again after individual dropna
    clean_idx = orig.index.intersection(synth.index)
    orig = orig.loc[clean_idx]
    synth = synth.loc[clean_idx]

    n = len(orig)
    if n < 20:
        warnings.warn(
            f"Only {n} overlapping observations; "
            "replication quality estimates are unreliable."
        )
        return {
            "correlation": np.nan,
            "r_squared": np.nan,
            "tracking_error": np.nan,
            "information_ratio": np.nan,
        }

    corr = float(orig.corr(synth))
    r2 = corr ** 2

    diff = orig - synth
    te_daily = float(diff.std())
    te_annual = te_daily * np.sqrt(252.0)

    mean_diff_daily = float(diff.mean())
    ir = mean_diff_daily / te_daily if te_daily > 0.0 else 0.0
    ir_annual = ir * np.sqrt(252.0)

    return {
        "correlation": corr,
        "r_squared": r2,
        "tracking_error": te_annual,
        "information_ratio": ir_annual,
    }


# --------------------------------------------------------------------------
# Self-contained factor construction + mimicking
# --------------------------------------------------------------------------

def build_factor_mimicking_portfolios(
    asset_returns: pd.DataFrame,
) -> dict:
    """Construct factor returns from ETFs, then build mimicking portfolios.

    This is a self-contained pipeline that:
    1. Constructs economic factor returns directly from ETF returns.
    2. Builds factor-mimicking portfolios (long-only, sum-to-1) that track
       each constructed factor using the full ETF universe.

    Factor definitions
    ------------------
    - **mkt_rf** : equal-weight average of all equity ETFs minus the daily
      risk-free rate.
    - **smb** (size proxy) : CSI 500 (510500) return minus CSI 300 (510300)
      return.
    - **bond** : 5Y Treasury Bond ETF (511010) return minus the daily
      risk-free rate.
    - **gold** : Gold ETF (518880) return minus the daily risk-free rate.

    Parameters
    ----------
    asset_returns : DataFrame (T × N)
        Daily returns of tradable ETFs.  Expected to contain columns
        ``510300``, ``510500``, ``511010``, ``518880``, and optionally
        ``159980`` (commodity composite).

    Returns
    -------
    dict
        Keys:
        - ``"weights"`` : dict[str, pd.Series]
            Factor name → asset weight Series.
        - ``"factor_returns"`` : pd.DataFrame
            Constructed factor returns (T × K).
        - ``"mimicking_returns"`` : dict[str, pd.Series]
            Factor name → daily return of the factor-mimicking portfolio.
    """
    rf_daily = _daily_rf()
    columns = set(asset_returns.columns)
    constructed: dict[str, pd.Series] = {}

    # -- mkt_rf: equal-weight equity ETFs minus rf ---------------------------
    equity_etfs = [c for c in ("510300", "510500") if c in columns]
    if equity_etfs:
        mkt = asset_returns[equity_etfs].mean(axis=1) - rf_daily
        constructed["mkt_rf"] = mkt

    # -- smb: small-cap minus large-cap -------------------------------------
    if "510300" in columns and "510500" in columns:
        smb = asset_returns["510500"] - asset_returns["510300"]
        constructed["smb"] = smb

    # -- bond: Treasury bond ETF minus rf -----------------------------------
    if "511010" in columns:
        bond = asset_returns["511010"] - rf_daily
        constructed["bond"] = bond

    # -- gold: gold ETF minus rf --------------------------------------------
    if "518880" in columns:
        gold = asset_returns["518880"] - rf_daily
        constructed["gold"] = gold

    if not constructed:
        raise ValueError(
            "No factor could be constructed from the available columns "
            f"{sorted(columns)}.  Expected at least one of 510300, 510500, "
            "511010, 518880."
        )

    factor_returns = pd.DataFrame(constructed).dropna()

    # Build factor-mimicking portfolios using the full asset universe
    weights = build_factor_replicating_portfolios(asset_returns, factor_returns)

    # Compute the mimicking-portfolio return for each factor
    common_idx = asset_returns.index.intersection(factor_returns.index)
    aligned_assets = asset_returns.loc[common_idx]

    mimicking_returns: dict[str, pd.Series] = {}
    for factor_name, w_series in weights.items():
        aligned_w = w_series.reindex(aligned_assets.columns, fill_value=0.0)
        mimicking_returns[factor_name] = (
            aligned_assets * aligned_w.values
        ).sum(axis=1)

    return {
        "weights": weights,
        "factor_returns": factor_returns,
        "mimicking_returns": mimicking_returns,
    }
