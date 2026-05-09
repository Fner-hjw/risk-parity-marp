"""
src/viz.py -- publication-quality visualisation for risk-parity backtesting.

All functions return the matplotlib Figure object and optionally save to disk.
"""

from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .conventions import FIGURES, OOS_START, RISK_FREE_ANNUAL, ROOT


# ---------------------------------------------------------------------------
# Global style defaults
# ---------------------------------------------------------------------------

COLOUR_CYCLE = [
    "#2166AC",  # blue
    "#B2182B",  # red
    "#4DAF4A",  # green
    "#FF7F00",  # orange
    "#984EA3",  # purple
    "#A65628",  # brown
    "#F781BF",  # pink
    "#66C2A5",  # teal
]


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _ensure_matplotlib():
    """Lazy-import matplotlib + seaborn; raise if missing."""
    try:
        import matplotlib as _mpl
        import matplotlib.pyplot as _plt
        import seaborn as _sns
        return _mpl, _plt, _sns
    except ImportError as exc:
        raise ImportError(
            "matplotlib and seaborn are required for viz.py. "
            "Install with: pip install matplotlib seaborn"
        ) from exc


def _maybe_save(fig, save_path: str | Path | None, dpi: int = 200) -> None:
    """Save *fig* to *save_path* if not None."""
    if save_path is not None:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(path), dpi=dpi, bbox_inches="tight")
        print(f"[viz] saved {path}")


def _risk_budget_contributions(
    weights: pd.Series, returns: pd.DataFrame
) -> pd.Series:
    """Compute each asset's percentage risk contribution.

    RC_i = w_i * (Sigma @ w)_i / sqrt(w^T @ Sigma @ w)
    """
    cov = returns.cov().values
    w = weights.values.astype(float)
    port_vol = np.sqrt(w @ cov @ w)
    if port_vol == 0:
        n = len(w)
        return pd.Series(np.full(n, 1.0 / n), index=weights.index)
    marginal = cov @ w
    rc = w * marginal / port_vol
    return pd.Series(rc, index=weights.index)


# ===================================================================
# Public plotting functions
# ===================================================================


def set_style() -> None:
    """Apply a clean publication-grade matplotlib / seaborn style."""
    _, plt, sns = _ensure_matplotlib()

    sns.set_style("whitegrid")
    plt.rcParams.update({
        # Font
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "Palatino", "serif"],
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
        # Figure
        "figure.dpi": 150,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.1,
        # Axes
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": "--",
        # Lines / markers
        "lines.linewidth": 1.6,
        "lines.markeredgewidth": 0.0,
        "patch.edgecolor": "w",
        "patch.force_edgecolor": True,
        # Legend
        "legend.frameon": True,
        "legend.framealpha": 0.9,
        "legend.edgecolor": "#cccccc",
        # Colour
        "axes.prop_cycle": plt.cycler(color=COLOUR_CYCLE),
    })


# ---------------------------------------------------------------------------
# 1. Cumulative returns
# ---------------------------------------------------------------------------

def plot_cumulative_returns(
    results_dict: dict[str, Any],
    benchmark_series: pd.Series | None = None,
    benchmark_label: str = "CSI MARP 930929",
    title: str | None = None,
    save_path: str | Path | None = None,
    figsize: tuple[float, float] = (12, 6),
):
    """Cumulative NAV chart for backtest results with optional benchmark.

    Parameters
    ----------
    results_dict : dict of {name: BacktestResult}
    benchmark_series : pd.Series of daily benchmark *returns*, optional
    benchmark_label : str
    title : str, optional (auto-generated if None)
    save_path : path-like or None
    figsize : (width, height) in inches
    """
    _, plt, _ = _ensure_matplotlib()

    fig, ax = plt.subplots(figsize=figsize)

    for i, (name, res) in enumerate(results_dict.items()):
        cum = res.cumulative.dropna()
        ax.plot(
            cum.index, cum.values,
            label=name,
            color=COLOUR_CYCLE[i % len(COLOUR_CYCLE)],
            linewidth=1.8,
        )

    if benchmark_series is not None:
        bm_cum = (1 + benchmark_series.dropna()).cumprod()
        # reindex to match the first strategy's timeline
        first_key = next(iter(results_dict))
        bm_cum = bm_cum.reindex(
            results_dict[first_key].cumulative.index, method="ffill"
        ).dropna()
        ax.plot(
            bm_cum.index, bm_cum.values,
            label=benchmark_label,
            color="black", linewidth=2.0, linestyle="--", alpha=0.85,
        )

    # -- Key-event annotations -------------------------------------------------
    oos_dt = pd.Timestamp(OOS_START)
    if oos_dt > results_dict[next(iter(results_dict))].cumulative.index[0]:
        ax.axvline(x=oos_dt, color="grey", linestyle=":", linewidth=1.2, alpha=0.7)
        ax.annotate(
            "Out-of-Sample",
            xy=(oos_dt, ax.get_ylim()[1] * 0.95),
            xytext=(8, 0), textcoords="offset points",
            fontsize=8, color="grey", ha="left", va="top",
        )

    # -- Formatting ------------------------------------------------------------
    ax.set_ylabel("Cumulative NAV")
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda y, _: f"{y:.2f}x")
    )
    if title is None:
        title = "Cumulative Performance: Risk-Parity Strategies vs CSI MARP 930929"
    ax.set_title(title)
    ax.legend(loc="upper left")
    fig.tight_layout()

    _maybe_save(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# 2. Drawdowns
# ---------------------------------------------------------------------------

def plot_drawdowns(
    results_dict: dict[str, Any],
    save_path: str | Path | None = None,
    figsize: tuple[float, float] = (12, 5),
):
    """Drawdown chart with filled regions for each strategy.

    Parameters
    ----------
    results_dict : dict of {name: BacktestResult}
    save_path : path-like or None
    figsize : (width, height)
    """
    _, plt, _ = _ensure_matplotlib()

    fig, ax = plt.subplots(figsize=figsize)

    for i, (name, res) in enumerate(results_dict.items()):
        dd = res.drawdown_series().dropna()
        colour = COLOUR_CYCLE[i % len(COLOUR_CYCLE)]
        ax.fill_between(dd.index, 0, dd.values, alpha=0.18, color=colour)
        ax.plot(dd.index, dd.values, label=name, color=colour, linewidth=1.2)

    ax.axhline(y=0, color="black", linewidth=0.5, linestyle="-")
    ax.set_ylabel("Drawdown")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.set_title("Drawdown Analysis")
    ax.legend(loc="lower left")
    fig.tight_layout()

    _maybe_save(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# 3. Rolling metrics (Sharpe + Vol)
# ---------------------------------------------------------------------------

def plot_rolling_metrics(
    results_dict: dict[str, Any],
    window: int = 252,
    save_dir: str | Path | None = None,
):
    """Rolling Sharpe ratio and rolling annualised volatility subplots.

    Parameters
    ----------
    results_dict : dict of {name: BacktestResult}
    window : int, lookback in trading days (default 252 = 1 year)
    save_dir : directory to save the figure, optional
    """
    _, plt, _ = _ensure_matplotlib()

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(12, 8), sharex=True,
        gridspec_kw={"hspace": 0.15},
    )

    for i, (name, res) in enumerate(results_dict.items()):
        colour = COLOUR_CYCLE[i % len(COLOUR_CYCLE)]
        rets = res.portfolio_returns.dropna()

        # Rolling annualised return and vol
        roll_ann_ret = rets.rolling(window).mean() * 252
        roll_ann_vol = rets.rolling(window).std() * np.sqrt(252)
        roll_sharpe = (roll_ann_ret - RISK_FREE_ANNUAL) / roll_ann_vol

        ax1.plot(roll_sharpe.index, roll_sharpe.values, label=name,
                 color=colour, linewidth=1.3)
        ax2.plot(roll_ann_vol.index, roll_ann_vol.values, label=name,
                 color=colour, linewidth=1.3)

    # ---- Sharpe subplot ----
    ax1.axhline(y=0, color="black", linewidth=0.5, linestyle="-")
    ax1.set_ylabel(f"Rolling {window}D Sharpe Ratio")
    ax1.set_title(f"Rolling Metrics ({window}-day window)")
    ax1.legend(loc="upper left", ncol=2)

    # ---- Vol subplot ----
    ax2.axhline(y=0, color="black", linewidth=0.5, linestyle="-")
    ax2.set_ylabel(f"Rolling {window}D Ann. Vol")
    ax2.set_xlabel("Date")

    # Shared formatting
    for ax in (ax1, ax2):
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.2f}"))

    fig.align_ylabels()
    fig.tight_layout()

    if save_dir is not None:
        _maybe_save(fig, Path(save_dir) / "rolling_metrics.png")
    return fig


# ---------------------------------------------------------------------------
# 4. Weight heatmap
# ---------------------------------------------------------------------------

def plot_weight_heatmap(
    result: Any,
    save_path: str | Path | None = None,
    figsize: tuple[float, float] = (14, 6),
):
    """Heatmap of portfolio weights over time for a single backtest result.

    Parameters
    ----------
    result : BacktestResult
    save_path : path-like or None
    figsize : (width, height)
    """
    _, plt, sns = _ensure_matplotlib()

    w = result.weights.dropna(how="all")
    if w.empty:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No weight data available", ha="center", va="center",
                transform=ax.transAxes, fontsize=14)
        _maybe_save(fig, save_path)
        return fig

    # Resample to monthly for cleaner display
    w_monthly = w.resample("M").last().dropna(how="all")
    if w_monthly.empty:
        w_monthly = w

    # Build the heatmap data matrix: rows = dates, cols = assets
    data = w_monthly.T  # assets × dates

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        data,
        ax=ax,
        cmap="YlOrRd",
        vmin=0, vmax=0.6,
        annot=False,
        cbar_kws={"label": "Weight", "shrink": 0.8},
        linewidths=0.0,
    )

    # Label formatting
    ax.set_title(f"Weight Allocation Over Time — {result.name}")
    ax.set_xlabel("Rebalance Date")
    ax.set_ylabel("Asset")

    # Thin x-axis ticks (too many dates)
    n_dates = data.shape[1]
    step = max(1, n_dates // 12)
    tick_positions = list(range(0, n_dates, step))
    tick_labels = [data.columns[i].strftime("%Y-%m") for i in tick_positions]
    ax.set_xticks([p + 0.5 for p in tick_positions])
    ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=8)

    fig.tight_layout()
    _maybe_save(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# 5. Risk contribution bar chart
# ---------------------------------------------------------------------------

def plot_risk_contribution(
    weights: pd.Series,
    returns: pd.DataFrame,
    save_path: str | Path | None = None,
):
    """Horizontal bar chart of percentage risk contributions for a single rebalance.

    Parameters
    ----------
    weights : pd.Series indexed by asset ticker, summing to 1
    returns : pd.DataFrame of daily returns used to estimate covariance
    save_path : path-like or None
    """
    _, plt, _ = _ensure_matplotlib()

    rc = _risk_budget_contributions(weights, returns)

    # Sort descending
    rc = rc.sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(8, 5))

    colours = [
        COLOUR_CYCLE[i % len(COLOUR_CYCLE)] for i in range(len(rc))
    ]
    bars = ax.barh(rc.index.astype(str), rc.values, color=colours, height=0.6)

    # Annotate percentages
    for bar, val in zip(bars, rc.values):
        ax.text(
            bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
            f"{val:.1%}", va="center", fontsize=10,
        )

    ax.axvline(x=1.0 / len(weights), color="grey", linestyle="--",
               linewidth=1.0, alpha=0.7, label="Equal weight target")
    ax.set_xlabel("Risk Contribution (%)")
    ax.set_title("Portfolio Risk Contribution by Asset")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()

    _maybe_save(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# 6. Factor exposure bar chart
# ---------------------------------------------------------------------------

def plot_factor_exposure(
    factor_reg_result: pd.DataFrame,
    save_path: str | Path | None = None,
):
    """Bar chart of factor betas with 95 % confidence-interval error bars.

    Parameters
    ----------
    factor_reg_result : pd.DataFrame from metrics.factor_regression()
        Index includes "alpha" and factor names; columns: coef, t-stat, se, p-value.
    save_path : path-like or None
    """
    _, plt, _ = _ensure_matplotlib()

    # Exclude alpha row
    df = factor_reg_result.drop("alpha", errors="ignore")
    if df.empty:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.text(0.5, 0.5, "No factor exposures to display", ha="center",
                va="center", transform=ax.transAxes, fontsize=14)
        _maybe_save(fig, save_path)
        return fig

    names = df.index.astype(str).tolist()
    betas = df["coef"].values
    ses = df["se"].values
    pvals = df.get("p-value", pd.Series(np.ones(len(df)), index=df.index)).values
    ci = 1.96 * ses  # 95 % confidence interval half-width

    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(8, 5))

    # Colour by significance
    bar_colours = []
    for p in pvals:
        if p < 0.01:
            bar_colours.append("#2166AC")
        elif p < 0.05:
            bar_colours.append("#92C5DE")
        else:
            bar_colours.append("#D1E5F0")

    ax.bar(x, betas, color=bar_colours, edgecolor="white", linewidth=0.8)
    ax.errorbar(x, betas, yerr=ci, fmt="none", ecolor="black",
                capsize=4, linewidth=1.2)

    # Significance stars
    for i, (b, p) in enumerate(zip(betas, pvals)):
        if p < 0.01:
            star = "***"
        elif p < 0.05:
            star = "**"
        elif p < 0.10:
            star = "*"
        else:
            star = ""
        offset = ci[i] + 0.0005 if b >= 0 else -(ci[i] + 0.002)
        ax.text(i, b + offset, star, ha="center", fontsize=12, va="bottom" if b >= 0 else "top")

    ax.axhline(y=0, color="black", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=11)
    ax.set_ylabel("Factor Beta")
    ax.set_title("Factor Exposure — Regression Coefficients (95% CI)")

    # Add R-squared annotation
    r2 = factor_reg_result.attrs.get("R-squared", None)
    if r2 is not None:
        ax.text(
            0.98, 0.95, f"$R^2$ = {r2:.3f}",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=10, bbox=dict(boxstyle="round,pad=0.3",
                                   facecolor="white", alpha=0.8),
        )

    fig.tight_layout()
    _maybe_save(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# 7. Regime performance comparison
# ---------------------------------------------------------------------------

def plot_regime_performance(
    results_dict: dict[str, Any],
    regime_dates: dict[str, tuple[str, str]],
    save_path: str | Path | None = None,
):
    """Grouped bar chart of annualised returns by market regime.

    Parameters
    ----------
    results_dict : dict of {name: BacktestResult}
    regime_dates : dict of {regime_label: (start_date_str, end_date_str)}
    save_path : path-like or None
    """
    _, plt, _ = _ensure_matplotlib()

    regimes = list(regime_dates.keys())
    strategy_names = list(results_dict.keys())
    n_regimes = len(regimes)
    n_strats = len(strategy_names)

    # Compute annualised return per strategy per regime
    data = np.zeros((n_strats, n_regimes))
    for j, (rlabel, (start, end)) in enumerate(regime_dates.items()):
        start_dt = pd.Timestamp(start)
        end_dt = pd.Timestamp(end)
        for i, (sname, res) in enumerate(results_dict.items()):
            rets = res.portfolio_returns.loc[start_dt:end_dt].dropna()
            if len(rets) < 20:
                data[i, j] = np.nan
                continue
            n_years = len(rets) / 252
            cum = (1 + rets).prod()
            data[i, j] = float(cum ** (1 / max(n_years, 0.1)) - 1)

    # Plot grouped bars
    x = np.arange(n_regimes)
    width = 0.8 / n_strats

    fig, ax = plt.subplots(figsize=(max(8, n_regimes * 3), 5.5))

    for i, sname in enumerate(strategy_names):
        offset = (i - (n_strats - 1) / 2) * width
        ax.bar(x + offset, data[i, :], width=width * 0.9, label=sname,
               color=COLOUR_CYCLE[i % len(COLOUR_CYCLE)], edgecolor="white",
               linewidth=0.5)

    ax.axhline(y=0, color="black", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(
        ["\n".join(textwrap.wrap(lbl, 15)) for lbl in regimes], fontsize=10
    )
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.1%}"))
    ax.set_ylabel("Annualised Return")
    ax.set_title("Strategy Performance by Market Regime")
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()

    _maybe_save(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# 8. Annual returns calendar heatmap
# ---------------------------------------------------------------------------

def plot_annual_returns_heatmap(
    returns_series: pd.Series,
    save_path: str | Path | None = None,
):
    """Calendar-style heatmap of monthly returns (year vs month).

    Parameters
    ----------
    returns_series : pd.Series of daily portfolio returns
    save_path : path-like or None
    """
    _, plt, sns = _ensure_matplotlib()

    rets = returns_series.dropna()
    if rets.empty:
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.text(0.5, 0.5, "No return data available", ha="center", va="center",
                transform=ax.transAxes, fontsize=14)
        _maybe_save(fig, save_path)
        return fig

    # Compute monthly cumulative returns
    monthly = rets.resample("M").apply(lambda x: (1 + x).prod() - 1)
    monthly.index = monthly.index.to_period("M")

    # Build pivot: rows=years, cols=months
    years = sorted(set(m.index.year for m in monthly.index))
    months = list(range(1, 13))
    month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    data = np.full((len(years), 12), np.nan)
    for i, yr in enumerate(years):
        for m in range(1, 13):
            key = pd.Period(year=yr, month=m, freq="M")
            if key in monthly.index:
                data[i, m - 1] = monthly.loc[key]

    # Annotation matrix
    annot = np.empty_like(data, dtype=object)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = data[i, j]
            annot[i, j] = f"{v:.1%}" if not np.isnan(v) else ""

    fig, ax = plt.subplots(figsize=(10, max(3, len(years) * 0.6)))
    cmap = sns.diverging_palette(10, 130, s=80, l=55, as_cmap=True)

    sns.heatmap(
        data,
        annot=annot,
        fmt="",
        cmap=cmap,
        center=0,
        vmin=-0.15, vmax=0.15,
        xticklabels=month_labels,
        yticklabels=[str(y) for y in years],
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "Monthly Return", "shrink": 0.8},
        ax=ax,
    )

    ax.set_title("Monthly Returns Calendar Heatmap")
    ax.set_xlabel("Month")
    ax.set_ylabel("Year")
    fig.tight_layout()

    _maybe_save(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# 9. Benchmark comparison table
# ---------------------------------------------------------------------------

def compare_to_benchmark_table(
    results_dict: dict[str, Any],
    benchmark_returns: pd.Series,
    save_path: str | Path | None = None,
):
    """Render a formatted comparison table as a matplotlib table.

    Parameters
    ----------
    results_dict : dict of {name: BacktestResult}
    benchmark_returns : pd.Series of daily benchmark returns
    save_path : path-like or None
    """
    _, plt, _ = _ensure_matplotlib()

    from .metrics import (
        annualised_return,
        annualised_vol,
        calmar_ratio,
        information_ratio,
        max_drawdown,
        sortino_ratio,
        tracking_error,
    )

    # Build table data
    rows: list[list[str]] = []
    col_labels = [
        "Strategy",
        "Ann. Ret",
        "Ann. Vol",
        "Sharpe",
        "Sortino",
        "Calmar",
        "Max DD",
        "Total Ret",
        "Track. Err",
        "Info Ratio",
    ]

    for name, res in results_dict.items():
        rets = res.portfolio_returns.dropna()
        ann_ret = annualised_return(rets)
        ann_vol = annualised_vol(rets)
        sharpe = (ann_ret - RISK_FREE_ANNUAL) / ann_vol if ann_vol > 0 else 0.0
        sortino = sortino_ratio(rets, rf=RISK_FREE_ANNUAL)
        calmar = calmar_ratio(rets)
        mdd = max_drawdown(rets)
        total_ret = (1 + rets).prod() - 1

        bm = benchmark_returns.reindex(rets.index).dropna()
        aligned = pd.concat([rets, bm], axis=1).dropna()
        te = tracking_error(aligned.iloc[:, 0], aligned.iloc[:, 1])
        ir = information_ratio(aligned.iloc[:, 0], aligned.iloc[:, 1])

        rows.append([
            name,
            f"{ann_ret:.2%}",
            f"{ann_vol:.2%}",
            f"{sharpe:.2f}",
            f"{sortino:.2f}",
            f"{calmar:.2f}",
            f"{mdd:.2%}",
            f"{total_ret:.2%}",
            f"{te:.2%}",
            f"{ir:.2f}",
        ])

    # Also add benchmark row
    bm_ann_ret = annualised_return(benchmark_returns.dropna())
    bm_ann_vol = annualised_vol(benchmark_returns.dropna())
    bm_sharpe = (bm_ann_ret - RISK_FREE_ANNUAL) / bm_ann_vol if bm_ann_vol > 0 else 0.0
    bm_sortino = sortino_ratio(benchmark_returns.dropna(), rf=RISK_FREE_ANNUAL)
    bm_calmar = calmar_ratio(benchmark_returns.dropna())
    bm_mdd = max_drawdown(benchmark_returns.dropna())
    bm_total = (1 + benchmark_returns.dropna()).prod() - 1

    rows.append([
        "CSI MARP 930929",
        f"{bm_ann_ret:.2%}",
        f"{bm_ann_vol:.2%}",
        f"{bm_sharpe:.2f}",
        f"{bm_sortino:.2f}",
        f"{bm_calmar:.2f}",
        f"{bm_mdd:.2%}",
        f"{bm_total:.2%}",
        "—",
        "—",
    ])

    # Build matplotlib table
    n_rows = len(rows)
    n_cols = len(col_labels)
    fig_width = max(14, n_cols * 1.5)
    fig_height = max(2.5, n_rows * 0.45 + 1.0)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis("off")

    table = ax.table(
        cellText=rows,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
    )

    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.4)

    for (row, col), cell in table.get_celld().items():
        if row == 0:
            # Header row
            cell.set_facecolor("#333333")
            cell.set_text_props(color="white", fontweight="bold", fontsize=9)
        elif row == n_rows:
            # Benchmark row (last)
            cell.set_facecolor("#E8E8E8")
            cell.set_text_props(fontweight="bold")
        else:
            cell.set_facecolor("#FAFAFA" if row % 2 == 0 else "white")

    ax.set_title(
        "Performance Comparison: Strategies vs CSI MARP 930929",
        fontsize=13, fontweight="bold", pad=20,
    )
    fig.tight_layout()

    _maybe_save(fig, save_path)
    return fig
