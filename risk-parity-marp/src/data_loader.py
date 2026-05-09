"""Data extraction and cleaning module.

Pulls all required market data from AkShare + yfinance and saves canonical
parquet files following the schema defined in src/conventions.py.

Run as a script:  python -m src.data_loader

Data source notes:
  - ETFs (510300, 510500, 511010, 518880): yfinance (Yahoo Finance)
  - CSI MARP 930929:                   AkShare stock_zh_index_hist_csindex (csindex.com.cn API)
  - CSI 300 index (sh000300):           AkShare stock_zh_index_daily
  - Commodity ETF 159980 NAV:          AkShare fund_open_fund_info_em (East Money, different endpoint)
  - Mutual fund 004685 NAV:            AkShare fund_open_fund_info_em
  - Note: eastmoney ETF hist endpoint (fund_etf_hist_em) is blocked by proxy.
"""
from __future__ import annotations
import sys
import time
from pathlib import Path

import pandas as pd
import numpy as np

from src.conventions import (
    DATA_RAW, DATA_CLEAN, DATA_OOS, DATA_FACTORS,
    ETF_UNIVERSE, REFERENCE_INDICES, STAR_FUND,
    START_DATE, OOS_START, END_DATE, RISK_FREE_ANNUAL,
)

# --------------------------------------------------------------------------
# Yahoo Finance tickers (verified working; bypasses eastmoney proxy)
# --------------------------------------------------------------------------
YF_TICKERS = {
    "510300": "510300.SS",
    "510500": "510500.SS",
    "511010": "511010.SS",
    "518880": "518880.SS",
}

# --------------------------------------------------------------------------
# Retry helper
# --------------------------------------------------------------------------

def _retry(fn, *, name: str, attempts: int = 3, sleep: float = 3.0):
    last_err: Exception | None = None
    for i in range(attempts):
        try:
            return fn()
        except Exception as e:
            last_err = e
            print(f"   [retry {i+1}/{attempts}] {name}: {e}", file=sys.stderr)
            time.sleep(sleep)
    raise RuntimeError(f"{name} failed after {attempts} attempts: {last_err}")


# --------------------------------------------------------------------------
# Fetchers
# --------------------------------------------------------------------------

def fetch_etf_yf(ticker_yf: str, start: str, end: str) -> pd.Series:
    """Pull ETF daily adjusted close via yfinance.

    yfinance can return multi-level columns for single-ticker downloads.
    On empty results or TLS/SSL errors, retries by splitting the window.
    """
    import yfinance as yf

    def _download(s: str, e: str) -> pd.DataFrame:
        return yf.download(ticker_yf, start=s, end=e,
                           progress=False, auto_adjust=True)

    def _to_series(df: pd.DataFrame) -> pd.Series:
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] for c in df.columns]
        close = pd.to_numeric(df["Close"], errors="coerce")
        s = close.dropna()
        s.index = pd.to_datetime(s.index).tz_localize(None)
        return s

    # Try full range; if empty, split into two chunks
    df_full = _retry(lambda: _download(start, end), name=f"yfinance {ticker_yf}")
    if df_full.empty:
        mid = "2022-01-01"
        print(f"     [{ticker_yf}] empty result for {start}→{end}, splitting at {mid}")
        df1 = _retry(lambda: _download(start, mid), name=f"yfinance {ticker_yf} [chunk1]")
        df2 = _retry(lambda: _download(mid, end), name=f"yfinance {ticker_yf} [chunk2]")
        if df1.empty and df2.empty:
            raise RuntimeError(f"yfinance {ticker_yf}: all chunks returned empty data")
        df_full = pd.concat([df1, df2])

    return _to_series(df_full)


def fetch_csi_marp(start: str, end: str) -> pd.DataFrame:
    """Fetch official CSI MARP 930929 via the csindex API (separate from eastmoney)."""
    import akshare as ak
    df = _retry(
        lambda: ak.stock_zh_index_hist_csindex(
            symbol="930929",
            start_date=start.replace("-", ""),
            end_date=end.replace("-", ""),
        ),
        name="CSI MARP 930929",
    )
    df["date"] = pd.to_datetime(df["日期"])
    return df.sort_values("date").reset_index(drop=True)


def fetch_index_daily(symbol: str) -> pd.Series:
    """Fetch index daily close via AkShare.  symbol like 'sh000300'."""
    import akshare as ak
    df = _retry(
        lambda: ak.stock_zh_index_daily(symbol=symbol),
        name=f"index {symbol}",
    )
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date")["close"].astype(float)


def fetch_fund_nav(symbol: str) -> pd.Series:
    """Fetch open-end fund NAV via AkShare fund_open_fund_info_em."""
    import akshare as ak
    df = _retry(
        lambda: ak.fund_open_fund_info_em(symbol=symbol, indicator="单位净值走势"),
        name=f"fund_nav {symbol}",
    )
    df = df.rename(columns={"净值日期": "date", "单位净值": "close"})
    df["date"] = pd.to_datetime(df["date"])
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    return df.set_index("date")["close"].dropna()


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------

def assemble_panel(start: str, end: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pull everything and stitch into a single price panel.

    Returns:
        prices : DataFrame indexed by date, columns per ticker
        marp_full : full csindex frame for 930929 (provenance)
    """
    for d in (DATA_RAW, DATA_CLEAN, DATA_OOS, DATA_FACTORS):
        d.mkdir(parents=True, exist_ok=True)

    series: dict[str, pd.Series] = {}

    # 1. ETFs via yfinance
    for tkr_internal, tkr_yf in YF_TICKERS.items():
        print(f"-> fetching ETF {tkr_internal} ({tkr_yf})")
        s = fetch_etf_yf(tkr_yf, start, end)
        series[tkr_internal] = s.rename(tkr_internal)
        pd.DataFrame({"date": s.index, "close": s.values}).to_csv(
            DATA_RAW / f"{tkr_internal}.csv", index=False)
        print(f"   {tkr_internal}: {len(s)} rows, {s.index[0].date()} -> {s.index[-1].date()}")

    # 2. Commodity ETF 159980 (not on Yahoo; use AkShare fund_open_fund_info_em)
    print(f"-> fetching commodity ETF 159980 (fund NAV)")
    fund_nav_159980 = fetch_fund_nav("159980")
    fund_nav_159980.to_csv(DATA_RAW / "159980.csv")
    series["159980"] = fund_nav_159980.rename("159980")
    print(f"   159980: {len(series['159980'])} rows "
          f"({fund_nav_159980.index[0].date()} -> {fund_nav_159980.index[-1].date()})")

    # 2. CSI MARP 930929 via csindex
    print("-> fetching CSI MARP 930929 (csindex)")
    marp_full = fetch_csi_marp(start, end)
    marp_full.to_csv(DATA_RAW / "930929.csv", index=False)
    series["930929"] = (
        marp_full.set_index("date")["收盘"].dropna().astype(float).rename("930929")
    )
    print(f"   930929: {len(series['930929'])} rows")

    # 3. CSI 300 index
    print("-> fetching CSI 300 (sh000300)")
    csi300 = fetch_index_daily("sh000300")
    csi300.to_csv(DATA_RAW / "000300.csv")
    series["000300"] = csi300.rename("000300")
    print(f"   000300: {len(series['000300'])} rows")

    # 4. STAR fund NAV (004685 — star fund for benchmark replication)
    print(f"-> fetching STAR fund {STAR_FUND} NAV")
    fund_nav = fetch_fund_nav(STAR_FUND)
    fund_nav.to_csv(DATA_RAW / f"{STAR_FUND}.csv")
    series[STAR_FUND] = fund_nav.rename(STAR_FUND)
    print(f"   {STAR_FUND}: {len(series[STAR_FUND])} rows "
          f"({fund_nav.index[0].date()} -> {fund_nav.index[-1].date()})")

    # 5. STAR fund (004685 — same as STAR_FUND, avoid double-fetch)
    # (already fetched as STAR_FUND above)

    # Combine into wide panel on the union of dates
    prices = pd.concat(series.values(), axis=1)

    # Restrict to project window
    prices = prices.loc[(prices.index >= start) & (prices.index <= end)]

    # Forward-fill within each series (handles holidays where one market traded)
    prices = prices.ffill()

    # Drop rows where >50% NaN (pre-inception leading rows)
    keep = prices.notna().mean(axis=1) > 0.5
    prices = prices.loc[keep]

    # Final dtype
    prices = prices.astype(float)
    prices.index.name = "Date"
    prices = prices.sort_index()

    return prices, marp_full


def build_factors(prices: pd.DataFrame) -> pd.DataFrame:
    """Construct China FF3-style factor proxies.

    NOTE: this is a documented PROXY. Use CSMAR / Wind / RESSET in production.
      - mkt_rf: CSI 300 daily return − rf_daily  (large-cap proxy)
      - smb:    CSI 500 − CSI 300  (mid-large spread, NOT true SMB)
      - hml:    NaN  (not constructible from this universe)
      - rf:     2.2% annual / 252  (China 1Y deposit rate proxy)
    """
    rf_daily = RISK_FREE_ANNUAL / 252.0
    rets = prices.pct_change()
    factors = pd.DataFrame(index=rets.index)
    factors["mkt_rf"] = rets["000300"] - rf_daily
    factors["smb"]    = rets["510500"] - rets["510300"]
    factors["hml"]    = np.nan
    factors["rf"]     = rf_daily
    return factors


def write_schema():
    """Write data/clean/SCHEMA.md documenting each file."""
    lines = [
        "# Data Schema\n",
        "## Source Attribution\n",
        "All raw data pulled from:\n",
        "- **yfinance** (Yahoo Finance) for ETF adjusted close prices\n",
        "  - Ticket mapping: 510300→510300.SS, 510500→510500.SS, etc.\n",
        "- **AkShare** (akshare.com, wrapping East Money & CSI) for:\n",
        "  - CSI MARP 930929: `stock_zh_index_hist_csindex` (csindex.com.cn API)\n",
        "  - CSI 300 index: `stock_zh_index_daily` (East Money)\n",
        "  - Fund NAV data: `fund_open_fund_info_em` (East Money)\n",
        "\n## Files\n",
        "### `data/clean/prices.parquet`\n",
        "Daily adjusted close prices. Index = Date (DatetimeIndex, naive).\n",
        "Columns: 510300, 510500, 511010, 518880, 159980, 930929, 000300, 004685\n",
        "Missing values: forward-filled within each series. Leading NaN rows dropped.\n",
        "\n### `data/clean/returns.parquet`\n",
        "Same structure as prices.parquet but daily simple returns (pct_change).\n",
        "First row per column is NaN (no prior price) — expected.\n",
        "\n### `data/clean/marp_official.parquet`\n",
        "Full CSI csindex response for 930929 (all OHLCV columns preserved).\n",
        "Source: AkShare `stock_zh_index_hist_csindex`.\n",
        "\n### `data/oos/prices_oos.parquet` / `returns_oos.parquet`\n",
        "Locked out-of-sample slices (2022-01-01 onward). NEVER used during tuning.\n",
        "\n### `data/factors/china_ff3_proxy.parquet`\n",
        "Daily factor returns. Columns: mkt_rf, smb, hml (NaN), rf.\n",
        "**KNOWN PROXY ISSUES**:\n",
        "  - smb = CSI500−CSI300, not true small−big. Document in report.\n",
        "  - hml unavailable from this universe.\n",
        "  - rf = 2.2%/252 constant (not time-varying).\n",
    ]
    (DATA_CLEAN / "SCHEMA.md").write_text("".join(lines))


def main():
    print(f"== assembling panel {START_DATE} → {END_DATE} ==")
    prices, marp_full = assemble_panel(START_DATE, END_DATE)
    returns = prices.pct_change()

    # Save full series
    prices.to_parquet(DATA_CLEAN / "prices.parquet")
    returns.to_parquet(DATA_CLEAN / "returns.parquet")
    marp_full.to_parquet(DATA_CLEAN / "marp_official.parquet", index=False)

    # OOS slice (locked)
    prices_oos = prices.loc[prices.index >= OOS_START]
    returns_oos = returns.loc[returns.index >= OOS_START]
    prices_oos.to_parquet(DATA_OOS / "prices_oos.parquet")
    returns_oos.to_parquet(DATA_OOS / "returns_oos.parquet")

    # Factor proxies
    factors = build_factors(prices)
    factors.to_parquet(DATA_FACTORS / "china_ff3_proxy.parquet")

    # Schema documentation
    write_schema()

    # Status table
    print("\n== file write summary ==")
    print(f"prices.parquet     : {prices.shape[0]} rows × {prices.shape[1]} cols")
    print(f"  date range      : {prices.index.min().date()} → {prices.index.max().date()}")
    print(f"  per-column non-NaN coverage:")
    for c in prices.columns:
        nz = prices[c].notna().sum()
        first = prices[c].first_valid_index()
        last  = prices[c].last_valid_index()
        print(f"     {c:8} | {nz:5} pts | {first.date() if first else '?':10} → {last.date() if last else '?'}")
    print(f"\nprices_oos.parquet : {prices_oos.shape[0]} rows  (OOS locked)")
    print(f"factors            : {factors.shape}  ({list(factors.columns)})")

    # Acceptance checks
    import src.conventions as C
    assert prices.shape[1] == 8, f"expected 8 cols, got {prices.shape[1]}"
    assert prices.index.is_monotonic_increasing
    assert prices.index[0].year >= 2017
    assert prices.index[-1].year <= 2026
    print("\n[PASS] acceptance checks")


if __name__ == "__main__":
    main()
