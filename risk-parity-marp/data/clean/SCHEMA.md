# Data Schema
## Source Attribution
All raw data pulled from:
- **yfinance** (Yahoo Finance) for ETF adjusted close prices
  - Ticket mapping: 510300→510300.SS, 510500→510500.SS, etc.
- **AkShare** (akshare.com, wrapping East Money & CSI) for:
  - CSI MARP 930929: `stock_zh_index_hist_csindex` (csindex.com.cn API)
  - CSI 300 index: `stock_zh_index_daily` (East Money)
  - Fund NAV data: `fund_open_fund_info_em` (East Money)

## Files
### `data/clean/prices.parquet`
Daily adjusted close prices. Index = Date (DatetimeIndex, naive).
Columns: 510300, 510500, 511010, 518880, 159980, 930929, 000300, 004685
Missing values: forward-filled within each series. Leading NaN rows dropped.

### `data/clean/returns.parquet`
Same structure as prices.parquet but daily simple returns (pct_change).
First row per column is NaN (no prior price) — expected.

### `data/clean/marp_official.parquet`
Full CSI csindex response for 930929 (all OHLCV columns preserved).
Source: AkShare `stock_zh_index_hist_csindex`.

### `data/oos/prices_oos.parquet` / `returns_oos.parquet`
Locked out-of-sample slices (2022-01-01 onward). NEVER used during tuning.

### `data/factors/china_ff3_proxy.parquet`
Daily factor returns. Columns: mkt_rf, smb, hml (NaN), rf.
**KNOWN PROXY ISSUES**:
  - smb = CSI500−CSI300, not true small−big. Document in report.
  - hml unavailable from this universe.
  - rf = 2.2%/252 constant (not time-varying).
