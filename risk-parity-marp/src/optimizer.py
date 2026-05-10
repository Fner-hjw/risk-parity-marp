"""
src/optimizer.py — portfolio construction strategies.

Public API
----------
allocate_erc(returns, vol_target=None)
    Equal Risk Contribution weights via Newton iteration on risk budgeting.
    If vol_target is set, scales weights so the realised annualised vol equals
    the target (e.g. 0.10 = 10 %).

allocate_marp_replication(returns, marp_returns, method="ridge")
    Replicates the MARP index using OLS / Ridge / constrained optimisation.

allocate_equal_weight(returns)
    Plain 1/N portfolio.

All functions return a pd.Series indexed by asset names, summing to 1.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage as _linkage, leaves_list
from scipy.optimize import minimize
from scipy.spatial.distance import squareform

# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def _annualised_vol(returns: pd.DataFrame, weights: np.ndarray) -> float:
    portfolio_returns = returns @ weights
    return float(portfolio_returns.std() * np.sqrt(252))


def _risk_contributions(
    returns: pd.DataFrame, weights: np.ndarray
) -> np.ndarray:
    """Component risk (RC_i = w_i * (Cov * w)_i / portfolio_vol)."""
    cov = returns.cov().values
    port_vol = np.sqrt(weights @ cov @ weights)
    if port_vol == 0:
        return np.full_like(weights, 1 / len(weights))
    marginal_risk = cov @ weights
    rc = weights * marginal_risk / port_vol
    return rc


# --------------------------------------------------------------------------
# ERC — Equal Risk Contribution
# --------------------------------------------------------------------------

def allocate_erc(
    returns: pd.DataFrame,
    vol_target: float | None = None,
) -> pd.Series:
    """Equal Risk Contribution portfolio.

    Solves for weights that equalise each asset's risk contribution.
    Optional vol_target rescales the weights post-optimisation so that
    the realised annualised portfolio vol equals the target.
    """
    n = returns.shape[1]
    tickers = returns.columns.tolist()
    cov = returns.cov().values

    def objective(w):
        """Minimise sum of squared RC deviations from mean RC."""
        rc = _risk_contributions(returns, w)
        target_rc = rc.mean()
        return float(np.sum((rc - target_rc) ** 2))

    # Inverse-volatility initial guess: avoids the equal-weight saddle point
    # where SLSQP can terminate at nit=1 claiming success without moving.
    vols = returns.std().values * np.sqrt(252)
    if np.any(vols <= 0) or np.any(~np.isfinite(vols)):
        w0 = np.ones(n) / n
    else:
        inv_vol = 1.0 / vols
        w0 = inv_vol / inv_vol.sum()

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bounds = [(0.0, 1.0) for _ in range(n)]  # long only

    result = minimize(
        objective,
        w0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 2000, "ftol": 1e-12},
    )

    if not result.success:
        import warnings
        warnings.warn(
            f"ERC optimisation did not converge: {result.message} "
            f"(nit={result.nit}, fun={result.fun:.3e})"
        )

    w = result.x

    if vol_target is not None:
        port_vol = np.sqrt(w @ cov @ w) * np.sqrt(252)
        if port_vol > 0:
            w = w * (vol_target / port_vol)

    w = np.clip(w, 0, None)
    total = w.sum()
    if total <= 0:
        return allocate_equal_weight(returns)
    if total > 1.0:
        w = w / total
    return pd.Series(w, index=tickers)


# --------------------------------------------------------------------------
# MARP Replication
# --------------------------------------------------------------------------

def allocate_marp_replication(
    returns: pd.DataFrame,
    marp_returns: pd.Series,
    method: str = "ridge",
) -> pd.Series:
    """Replicate MARP using in-sample OLS/Ridge.

    method : {"ols", "ridge", "constrained"}
        ols        — plain least squares (may overfit).
        ridge      — Tikhonov regularised; alpha=0.01.
        constrained — non-negative least squares (long only, sum=1).
    """
    tickers = returns.columns.tolist()
    X = returns.values
    y = marp_returns.values

    # Align indices
    mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
    X, y = X[mask], y[mask]

    if method == "ols":
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)

    elif method == "ridge":
        alpha = 0.01
        K = X.T @ X + alpha * np.eye(X.shape[1])
        coef = np.linalg.solve(K, X.T @ y)

    elif method == "constrained":
        # Long-only sum-to-1 replication via SLSQP (lsq_linear has no equality constraint support).
        n_assets = X.shape[1]

        def _obj(w):
            r = X @ w - y
            return float(r @ r)

        sol = minimize(
            _obj,
            np.ones(n_assets) / n_assets,
            method="SLSQP",
            bounds=[(0.0, 1.0) for _ in range(n_assets)],
            constraints=[{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}],
            options={"maxiter": 2000, "ftol": 1e-12},
        )
        coef = sol.x

    else:
        raise ValueError(f"Unknown method: {method!r}")

    w = np.clip(coef, 0, None)
    total = w.sum()
    if total <= 0:
        import warnings
        warnings.warn("MARP replication produced all-negative weights; falling back to equal weight.")
        w = np.ones(n := len(tickers)) / n
    else:
        w = w / total

    return pd.Series(w, index=tickers)


# --------------------------------------------------------------------------
# Equal Weight (1/N)
# --------------------------------------------------------------------------

def allocate_equal_weight(returns: pd.DataFrame) -> pd.Series:
    """Simple 1/N portfolio."""
    n = returns.shape[1]
    return pd.Series(np.ones(n) / n, index=returns.columns)


# --------------------------------------------------------------------------
# Risk parity (vol-target only, no ERC iteration — faster variant)
# --------------------------------------------------------------------------

def allocate_vol_target(
    returns: pd.DataFrame,
    vol_target: float | None = 0.10,
) -> pd.Series:
    """Inverse-vol portfolio, optionally scaled to vol_target.

    Each asset i receives weight proportional to 1 / σ_i.
    If vol_target is set, scales so realised vol = vol_target (unlevered: sum ≤ 1).
    """
    tickers = returns.columns.tolist()
    vols = returns.std() * np.sqrt(252)
    inv_vol = 1.0 / vols.values
    w = inv_vol / inv_vol.sum()

    if vol_target is not None:
        cov = returns.cov().values
        port_vol = np.sqrt(w @ cov @ w) * np.sqrt(252)
        if port_vol > 0:
            w = w * (vol_target / port_vol)

    w = np.clip(w, 0, None)
    total = w.sum()
    if total <= 0:
        return allocate_equal_weight(returns)
    if total > 1.0:
        w = w / total
    return pd.Series(w, index=tickers)


# --------------------------------------------------------------------------
# HRP — Hierarchical Risk Parity  (Lopez de Prado 2016)
# --------------------------------------------------------------------------

def _get_quasi_diag(linkage_matrix: np.ndarray, n_leaves: int) -> np.ndarray:
    """Quasi-diagonal ordering from a linkage matrix."""
    return leaves_list(linkage_matrix)


def _get_inverse_var_weights(cov_subset: np.ndarray) -> np.ndarray:
    """Inverse-variance weights for a covariance sub-matrix."""
    diag = np.diag(cov_subset)
    if np.any(diag <= 0):
        n = len(diag)
        return np.ones(n) / n
    iv = 1.0 / diag
    return iv / iv.sum()


def _recursive_bisection(
    cov: np.ndarray, ordered_indices: np.ndarray
) -> pd.Series:
    """Recursive bisection that computes HRP weights."""

    def _bisect(indices: np.ndarray) -> pd.Series:
        if len(indices) == 1:
            return pd.Series([1.0], index=indices)

        mid = len(indices) // 2
        left = indices[:mid]
        right = indices[mid:]

        cov_left = cov[np.ix_(left, left)]
        cov_right = cov[np.ix_(right, right)]

        w_left = _get_inverse_var_weights(cov_left)
        w_right = _get_inverse_var_weights(cov_right)

        var_left = w_left @ cov_left @ w_left
        var_right = w_right @ cov_right @ w_right

        denom = var_left + var_right
        if denom == 0:
            alpha_left = alpha_right = 0.5
        else:
            # inverse-variance allocation: w_i ∝ 1 / V_i
            alpha_left = var_right / denom
            alpha_right = var_left / denom

        return pd.concat(
            [_bisect(left) * alpha_left, _bisect(right) * alpha_right]
        )

    return _bisect(ordered_indices)


def allocate_hrp(
    returns: pd.DataFrame,
    vol_target: float | None = None,
    linkage: str = "ward",
) -> pd.Series:
    """Hierarchical Risk Parity portfolio.

    Implements the HRP algorithm: correlation -> distance -> clustering ->
    quasi-diagonal ordering -> recursive inverse-variance bisection.

    Parameters
    ----------
    returns : pd.DataFrame
        Asset returns (columns = assets).
    vol_target : float or None
        If set, scale weights so annualised portfolio vol equals this target.
    linkage : str
        Linkage method passed to scipy.cluster.hierarchy.linkage.
    """
    tickers = returns.columns.tolist()
    n = len(tickers)
    cov = returns.cov()
    corr = returns.corr().values
    dist = np.sqrt(0.5 * (1.0 - corr))

    try:
        condensed = squareform(dist, checks=False)
        link_mat = _linkage(condensed, method=linkage)
    except Exception:
        return allocate_equal_weight(returns)

    ordered = _get_quasi_diag(link_mat, n)
    raw_weights = _recursive_bisection(cov.values, ordered)

    w = raw_weights.sort_index().values

    if vol_target is not None:
        port_vol = np.sqrt(w @ cov.values @ w) * np.sqrt(252)
        if port_vol > 0:
            w = w * (vol_target / port_vol)

    w = np.clip(w, 0, None)
    total = w.sum()
    if total <= 0:
        return allocate_equal_weight(returns)
    if total > 1.0:
        w = w / total
    return pd.Series(w, index=tickers)
