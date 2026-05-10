"""
src/backtest.py — walk-forward backtest engine.

Usage
-----
results = run_backtest(
    returns,             # pd.DataFrame, daily returns of all assets
    marp_returns=None,  # pd.Series, benchmark to replicate (optional)
    lookback=756,       # 3 years × 252 trading days
    rebal_days=252,     # rebalance annually
    strategies={
        "ERC_5":   lambda r, m: allocate_erc(r, vol_target=0.05),
        "RP_5":    lambda r, m: allocate_vol_target(r, vol_target=0.05),
        "Equal":   lambda r, m: allocate_equal_weight(r),
        "MARP_rep": lambda r, m: allocate_marp_replication(r, m, method="ridge"),
    },
)

Returns dict of BacktestResult objects keyed by strategy name.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from .optimizer import allocate_erc, allocate_equal_weight, allocate_marp_replication, allocate_vol_target

try:
    from .conventions import RISK_FREE_ANNUAL
except ImportError:
    from src.conventions import RISK_FREE_ANNUAL


# --------------------------------------------------------------------------
# Data structures
# --------------------------------------------------------------------------

@dataclass
class BacktestResult:
    name: str
    weights: pd.DataFrame  # index=rebalance_dates, columns=asset names
    portfolio_returns: pd.Series  # daily out-of-sample returns
    cumulative: pd.Series  # unit NAV (starts at 1.0)
    lookback: int
    rebal_days: int

    @property
    def total_return(self) -> float:
        return float(self.cumulative.iloc[-1] - 1.0)

    @property
    def annualised_return(self) -> float:
        rets = self.portfolio_returns.dropna()
        if len(rets) == 0:
            return 0.0
        n_years = len(rets) / 252
        total = (1 + rets).prod()
        return float(total ** (1 / n_years) - 1.0)

    @property
    def annualised_vol(self) -> float:
        return float(self.portfolio_returns.std() * np.sqrt(252))

    @property
    def sharpe(self) -> float:
        ann_ret = self.annualised_return
        ann_vol = self.annualised_vol
        if ann_vol <= 0:
            return 0.0
        return float((ann_ret - RISK_FREE_ANNUAL) / ann_vol)

    def summary_df(self) -> pd.DataFrame:
        """Single-row summary of key metrics."""
        return pd.DataFrame([{
            "Strategy": self.name,
            "Ann. Return": f"{self.annualised_return:.2%}",
            "Ann. Vol":    f"{self.annualised_vol:.2%}",
            "Sharpe":      f"{self.sharpe:.2f}",
            "Max Drawdown": f"{self.max_drawdown:.2%}",
            "Total Ret":   f"{self.total_return:.2%}",
        }])

    @property
    def max_drawdown(self) -> float:
        cum = (1 + self.portfolio_returns.dropna()).cumprod()
        cummax = cum.cummax()
        drawdown = (cum - cummax) / cummax
        return float(drawdown.min())

    def drawdown_series(self) -> pd.Series:
        cum = (1 + self.portfolio_returns.dropna()).cumprod()
        cummax = cum.cummax()
        return (cum - cummax) / cummax


# --------------------------------------------------------------------------
# Core engine
# --------------------------------------------------------------------------

def run_backtest(
    returns: pd.DataFrame,
    marp_returns: pd.Series | None = None,
    lookback: int = 756,
    rebal_days: int = 252,
    strategies: dict[str, Callable] | None = None,
    min_rebal_assets: int = 5,
) -> dict[str, BacktestResult]:
    """Walk-forward backtest.

    Parameters
    ----------
    returns : DataFrame (T × N)
        Daily excess returns of each asset.
    marp_returns : Series (T,)
        Benchmark index daily returns (for MARP replication strategy).
    lookback : int
        Number of trading days used for in-sample weight estimation.
    rebal_days : int
        Number of trading days between rebalances.
    strategies : dict of callable
        Maps strategy name → weight allocation function.
        Each function receives (returns_in_sample, marp_in_sample) and
        returns a pd.Series of weights indexed by asset names.
    min_rebal_assets : int
        Minimum number of non-NaN assets required to run optimisation.
    """

    if strategies is None:
        strategies = _default_strategies()

    # Align MARP to returns index
    if marp_returns is not None:
        marp_returns = marp_returns.reindex(returns.index).dropna()

    results: dict[str, BacktestResult] = {}

    # Determine rebalance dates (end of lookback, then every rebal_days)
    first_rebal = lookback
    rebal_dates_all = list(range(first_rebal, len(returns) - rebal_days, rebal_days))
    if not rebal_dates_all:
        raise ValueError(
            f"Data too short for lookback={lookback}, rebal_days={rebal_days}"
        )

    for strat_name, strat_fn in strategies.items():
        all_weights: list[pd.Series] = []
        all_oos_returns: list[float] = []

        for i, rebal_idx in enumerate(rebal_dates_all):
            # In-sample window
            is_start = rebal_idx - lookback
            is_end   = rebal_idx
            ret_is   = returns.iloc[is_start:is_end].dropna(axis=1)

            # Skip if not enough assets or data
            if ret_is.shape[1] < min_rebal_assets or ret_is.shape[0] < 252:
                continue

            # MARP in-sample (aligned)
            marp_is = None
            if marp_returns is not None:
                marp_is = marp_returns.iloc[is_start:is_end].dropna()
                # Drop any returns column that is all NaN over IS window
                valid_assets = ret_is.columns[~ret_is.isna().all()]
                ret_is = ret_is[valid_assets]

            # Compute weights
            try:
                w = strat_fn(ret_is, marp_is)
            except Exception:
                continue

            # Align weights to available assets
            w = w.reindex(ret_is.columns).fillna(0.0)
            w = np.clip(w, 0, None)
            w_sum = w.sum()
            if w_sum <= 0:
                continue
            if w_sum > 1.0:
                w = w / w_sum

            # Out-of-sample window
            oos_start = rebal_idx
            oos_end   = (
                rebal_dates_all[i + 1]
                if i + 1 < len(rebal_dates_all)
                else len(returns)
            )
            ret_oos = returns.iloc[oos_start:oos_end][w.index]

            # Portfolio returns: assets + cash earning risk-free rate
            rf_daily = RISK_FREE_ANNUAL / 252.0
            cash_weight = max(0.0, 1.0 - w.sum())
            port_ret = (ret_oos * w.values).sum(axis=1, skipna=True) + cash_weight * rf_daily
            port_ret = port_ret.dropna()

            # Record weights as a single-row DataFrame keyed by rebalance date
            rebal_date = returns.index[rebal_idx]
            all_weights.append(pd.DataFrame([w.values], index=[rebal_date], columns=w.index))
            all_oos_returns.append(port_ret)

        if not all_weights:
            raise RuntimeError(f"Strategy '{strat_name}' produced no rebalancing windows.")

        # Assemble weight DataFrame (forward-fill between rebalances)
        weight_df = (
            pd.concat(all_weights, axis=0)
            .reindex(returns.index, method="ffill")
            .dropna(how="all")
            .loc[returns.index[rebal_dates_all[0]]:]
        )

        # Assemble portfolio returns series
        portfolio_returns = pd.concat(all_oos_returns)
        portfolio_returns = portfolio_returns[~portfolio_returns.index.duplicated(keep="first")]

        # Cumulative NAV starting at 1
        cumulative = (1 + portfolio_returns).cumprod()
        # Extend to full date range of data (last known value)
        full_cumulative = cumulative.reindex(returns.index, method="ffill").fillna(1.0)

        results[strat_name] = BacktestResult(
            name=strat_name,
            weights=weight_df,
            portfolio_returns=portfolio_returns,
            cumulative=full_cumulative,
            lookback=lookback,
            rebal_days=rebal_days,
        )

    return results


def _default_strategies() -> dict[str, Callable]:
    return {
        "ERC_5":      lambda r, m: allocate_erc(r, vol_target=0.05),
        "Equal":      lambda r, m: allocate_equal_weight(r),
        "MARP_rep":   lambda r, m: allocate_marp_replication(r, m, method="ridge") if m is not None else allocate_equal_weight(r),
    }
