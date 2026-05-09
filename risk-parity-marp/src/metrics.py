"""
src/metrics.py — performance analytics and factor regression.

Functions
---------
performance_summary(results_dict, marp_returns=None)
    Prints a formatted table of key metrics for all strategies.

factor_regression(portfolio_returns, factors, rf=None)
    Runs OLS Fama-French 3-factor (or custom) regression.

tracking_error(portfolio_returns, benchmark_returns)
    Annualised tracking error vs benchmark.

plot_cumulative(results_dict, benchmark=None, save_path=None)
    Renders a cumulative NAV chart.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

try:
    from .conventions import OOS_START, RISK_FREE_ANNUAL
except ImportError:
    from src.conventions import OOS_START, RISK_FREE_ANNUAL


# --------------------------------------------------------------------------
# Core metrics
# --------------------------------------------------------------------------

def annualised_return(returns: pd.Series) -> float:
    n_years = len(returns) / 252
    cumulative = (1 + returns).prod()
    return float(cumulative ** (1 / n_years) - 1)


def annualised_vol(returns: pd.Series) -> float:
    return float(returns.std() * np.sqrt(252))


def max_drawdown(returns: pd.Series) -> float:
    cum = (1 + returns).cumprod()
    peak = cum.cummax()
    dd = (cum - peak) / peak
    return float(dd.min())


def sortino_ratio(
    returns: pd.Series, rf: float = 0.0, periods_per_year: int = 252
) -> float:
    ann_ret = annualised_return(returns) - rf
    downside = returns[returns < 0]
    if len(downside) == 0:
        return 0.0
    down_vol = float(downside.std() * np.sqrt(periods_per_year))
    return ann_ret / down_vol if down_vol > 0 else 0.0


def calmar_ratio(returns: pd.Series) -> float:
    ann_ret = annualised_return(returns)
    mdd = abs(max_drawdown(returns))
    return ann_ret / mdd if mdd > 0 else 0.0


def tracking_error(
    portfolio_returns: pd.Series, benchmark_returns: pd.Series
) -> float:
    diff = portfolio_returns - benchmark_returns
    diff = diff.dropna()
    return float(diff.std() * np.sqrt(252))


def information_ratio(
    portfolio_returns: pd.Series, benchmark_returns: pd.Series
) -> float:
    diff = portfolio_returns - benchmark_returns
    te = tracking_error(portfolio_returns, benchmark_returns)
    return float(diff.mean() * 252 / te) if te > 0 else 0.0


# --------------------------------------------------------------------------
# Performance summary
# --------------------------------------------------------------------------

def performance_summary(
    results_dict: dict[str, Any],
    marp_returns: pd.Series | None = None,
    rf: float | None = None,
    oos_start: str | None = None,
) -> pd.DataFrame:
    """Build a multi-metric comparison table for all backtest strategies.

    Metrics are computed on out-of-sample returns only (>= oos_start).
    """
    if rf is None:
        rf = RISK_FREE_ANNUAL
    if oos_start is None:
        oos_start = OOS_START

    rows = []
    for name, res in results_dict.items():
        rets = res.portfolio_returns.dropna()
        # Filter to OOS period
        rets = rets.loc[rets.index >= oos_start]
        if len(rets) < 60:
            continue
        mdd = max_drawdown(rets)

        row = {
            "Strategy": name,
            "Ann. Return": f"{annualised_return(rets):.2%}",
            "Ann. Vol":    f"{annualised_vol(rets):.2%}",
            "Sharpe":      f"{(annualised_return(rets) - rf) / annualised_vol(rets):.2f}",
            "Sortino":     f"{sortino_ratio(rets, rf):.2f}",
            "Calmar":      f"{calmar_ratio(rets):.2f}",
            "Max DD":      f"{mdd:.2%}",
            "Total Ret":   f"{(1 + rets).prod() - 1:.2%}",
        }

        if marp_returns is not None:
            # Track error only on OOS period
            b = marp_returns.loc[marp_returns.index >= oos_start].reindex(rets.index).dropna()
            aligned_rets = rets.reindex(b.index).dropna()
            aligned_b = b.reindex(aligned_rets.index).dropna()
            if len(aligned_rets) > 60:
                te = tracking_error(aligned_rets, aligned_b)
                ir = information_ratio(aligned_rets, aligned_b)
                row["Track. Error"] = f"{te:.2%}"
                row["Info Ratio"] = f"{ir:.2f}"

        rows.append(row)

    return pd.DataFrame(rows).set_index("Strategy")


# --------------------------------------------------------------------------
# Factor regression (Fama-French style)
# --------------------------------------------------------------------------

def factor_regression(
    portfolio_returns: pd.Series,
    factors: pd.DataFrame,
    rf: pd.Series | None = None,
) -> pd.DataFrame:
    """Run OLS cross-sectional factor regression.

    portfolio_returns : Series (T,)
    factors           : DataFrame (T × K) of factor returns (e.g. mkt, smb, hml)
    rf                : Series (T,) of risk-free rate (optional; subtracted first)

    Returns a DataFrame with coefficient estimates, t-stats, and R².
    """
    # Align and drop NaN
    df = pd.concat([portfolio_returns, factors], axis=1).dropna()
    y = df.iloc[:, 0].values
    X = df.iloc[:, 1:].values
    X = np.column_stack([np.ones(len(X)), X])  # add intercept

    # OLS
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    resid = y - X @ beta
    n, k = X.shape
    s2 = float(np.sum(resid**2) / max(n - k, 1))
    try:
        var_beta = s2 * np.linalg.inv(X.T @ X)
    except np.linalg.LinAlgError:
        var_beta = s2 * np.linalg.pinv(X.T @ X)
    se = np.sqrt(np.diag(var_beta))
    tstats = beta / se
    r2 = 1 - np.sum(resid**2) / np.sum((y - y.mean())**2)

    names = ["alpha"] + list(factors.columns)
    result = pd.DataFrame({
        "coef": beta,
        "t-stat": tstats,
        "se": se,
        "p-value": 2 * (1 - _t_cdf(np.abs(tstats), df=n - k)),
    }, index=names)

    result.attrs["R-squared"] = float(r2)
    result.attrs["N_obs"] = n
    return result


def _t_cdf(t: float, df: int) -> float:
    """Approximate Student's t CDF using SciPy if available, else fallback."""
    try:
        from scipy.stats import t as t_dist
        return t_dist.cdf(t, df)
    except ImportError:
        # Very rough normal approximation for large df
        import math
        x = t / np.sqrt(df / (df - 2))
        return math.erf(x / np.sqrt(2)) / 2 + 0.5


# --------------------------------------------------------------------------
# Plotting
# --------------------------------------------------------------------------

def plot_cumulative(
    results_dict: dict[str, Any],
    benchmark: pd.Series | None = None,
    benchmark_label: str = "CSI MARP",
    save_path: str | None = None,
    title: str = "Cumulative Performance",
):
    """Plot cumulative NAV curves for all strategies + optional benchmark."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mticker
    except ImportError:
        print("[metrics] matplotlib not installed; skipping plot.")
        return

    fig, ax = plt.subplots(figsize=(11, 5))

    colors = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0", "#F44336", "#00BCD4"]
    for i, (name, res) in enumerate(results_dict.items()):
        col = colors[i % len(colors)]
        cumm = res.cumulative
        cumm = cumm.dropna()
        ax.plot(cumm.index, cumm.values, label=name, color=col, linewidth=1.6)

    if benchmark is not None:
        bm = (1 + benchmark.dropna()).cumprod()
        bm = bm.reindex(results_dict[list(results_dict.keys())[0]].cumulative.index, method="ffill")
        ax.plot(bm.index, bm.values, label=benchmark_label,
                color="black", linewidth=1.8, linestyle="--")

    ax.set_title(title, fontsize=13)
    ax.set_ylabel("Cumulative Return (unit NAV)")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.25)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[metrics] saved {save_path}")

    return fig


def plot_drawdown(
    results_dict: dict[str, Any],
    save_path: str | None = None,
):
    """Plot drawdown series for all strategies."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mticker
    except ImportError:
        return

    fig, ax = plt.subplots(figsize=(11, 4))
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0", "#F44336", "#00BCD4"]

    for i, (name, res) in enumerate(results_dict.items()):
        dd = res.drawdown_series()
        ax.fill_between(dd.index, 0, dd.values, alpha=0.25, color=colors[i % len(colors)])
        ax.plot(dd.index, dd.values, label=name, color=colors[i % len(colors)], linewidth=1.2)

    ax.set_title("Drawdown", fontsize=13)
    ax.set_ylabel("Drawdown")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(decimals=0))
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(alpha=0.25)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[metrics] saved {save_path}")

    return fig
