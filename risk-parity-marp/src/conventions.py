"""Project-wide conventions and constants.

All agents (data / strategy / backtest / research) read from this file
to ensure consistent universe, date ranges, schema, and parameters.
"""
from __future__ import annotations
from pathlib import Path

# ---- Paths ----------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw"
DATA_CLEAN = ROOT / "data" / "clean"
DATA_OOS = ROOT / "data" / "oos"
DATA_FACTORS = ROOT / "data" / "factors"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"

# ---- Universe -------------------------------------------------------------
# Tradable ETFs (constituent legs of our risk parity portfolio)
ETF_UNIVERSE: dict[str, str] = {
    "510300": "CSI 300 ETF (large-cap equity)",
    "510500": "CSI 500 ETF (mid-cap equity)",
    "511010": "5Y Treasury Bond ETF",
    "518880": "Gold ETF",
    "159980": "Commodity Composite ETF",
}

# Reference indices
REFERENCE_INDICES: dict[str, str] = {
    "930929": "CSI MARP (target index we replicate)",
    "000300": "CSI 300 (equity benchmark)",
    "000832": "CSI Treasury Bond (bond benchmark)",
}

# Mutual fund for star-fund replication appendix
STAR_FUND = "004685"

# ---- Time windows ---------------------------------------------------------
START_DATE = "2017-01-01"      # CSI MARP base date
IS_END = "2021-12-31"          # In-sample end (inclusive)
OOS_START = "2022-01-01"       # Out-of-sample start (NEVER touched during tuning)
END_DATE = "2025-12-31"        # End of OOS window

# ---- Strategy parameters --------------------------------------------------
TARGET_VOL_ANNUAL = 0.05       # CSI MARP-style 5% annual vol target
LOOKBACK_DAYS = 252            # rolling covariance lookback
REBALANCE = "M"                # monthly rebalance (pandas freq)
WEIGHT_BOUNDS = (0.0, 0.60)    # cap any single asset weight
MAX_LEVERAGE = 1.0             # no leverage in baseline (vol-target may scale)

# ---- Cost model -----------------------------------------------------------
COST_BPS_ROUNDTRIP = 10.0      # 10 bps round-trip transaction cost
SLIPPAGE_BPS = 2.0
RISK_FREE_ANNUAL = 0.022       # ~2.2% China 1Y deposit avg over horizon

# ---- Data schema ----------------------------------------------------------
# data/clean/prices.parquet: index=Date (DatetimeIndex), columns=ticker, values=adj_close
# data/clean/returns.parquet: same shape, daily simple returns
# data/factors/ff3_china.parquet: index=Date, columns=[mkt_rf, smb, hml, rf]


def trading_days_per_year() -> int:
    return 252
