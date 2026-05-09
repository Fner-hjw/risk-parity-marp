# A Tradable Multi-Asset Risk Parity Strategy

**Replicating CSI MARP (930929) and Benchmarking Against Hedge Fund Replication**

Fudan University - AIS Course Project - Prof. Sun Lin

---

## Project framing

This report constructs a **tradable** multi-asset risk-parity portfolio using liquid Chinese ETFs, with two reference benchmarks:

1. **CSI MARP 930929** — the official (non-tradable) Multi-Asset Risk Parity index we are replicating.
2. **A factor-based hedge fund replicator** — built in the Hasanhodzic & Lo (2007) tradition, in dialogue with the ProShares Hedge Replication ETF (Week 5 case) and AQR DELTA Strategy (Week 6 case).

The project addresses three of the eight "potential topics" in one report: risk parity index construction (Topic 6), hedge fund replication (Topic 5), and a quantitative multi-asset strategy (Topic 3) — and incidentally answers the gold question (Topic 1) through ablation.

## Course-pillar coverage

| Course objective | Where addressed in the report |
|---|---|
| Fund performance evaluation | Section 6 — return / vol / Sharpe / Sortino / max DD / Calmar / FF3+Carhart factor alpha |
| Portfolio construction with hedge funds | Sections 4-5 — ERC optimization, vol-target overlay, HRP extension |
| Hedge fund replication | Section 7 — factor replication of HFRX vs. our risk-parity portfolio |

## Strategy specification

- **Universe** — liquid A-share ETFs across equity, bond, commodity legs:
  - `510300.SH` Huatai-PB CSI 300 (large-cap equity)
  - `510500.SH` Southern CSI 500 (mid-cap equity)
  - `511010.SH` GTJA 5Y Treasury Bond
  - `518880.SH` Huaan Gold
  - `159980.SZ` NF Commodity (optional, post-2019)
- **Weighting** — Equal Risk Contribution (Maillard, Roncalli & Teïletche 2010)
- **Risk overlay** — annualized volatility target ≈ 5% (matching CSI MARP)
- **Rebalance** — monthly; transaction cost 10 bps + 2 bps slippage
- **In-sample** — 2017-01 to 2021-12 (5y) for parameter selection
- **Out-of-sample** — 2022-01 to 2025-12 (4y) for evaluation; never touched during tuning

## Original findings (planned)

1. **Tracking-error decomposition** of our tradable basket vs. official CSI MARP
2. **Hedge fund replication comparison**: factor-based replication of HFRX China Hedge Fund Index, side-by-side with our risk-parity output
3. **Regime split** of OOS: 2022 bear, 2023 sideways, 2024-2025 AI/equity rally — does risk parity deliver as advertised?
4. **Hierarchical Risk Parity (HRP, López de Prado 2016)** as robustness extension
5. **Cost stress test**: at what cost level does the strategy lose its Sharpe edge?

## Repository layout

```
risk-parity-marp/
├── data/
│   ├── raw/         <- as pulled from AkShare
│   ├── clean/       <- adjusted prices, aligned, parquet
│   ├── oos/         <- LOCKED 2022-01-01 onward
│   └── factors/     <- FF3 / Carhart China factors
├── notebooks/
│   ├── 01_data.ipynb
│   ├── 02_strategy.ipynb
│   ├── 03_backtest.ipynb
│   ├── 04_findings.ipynb
│   └── 05_report_figures.ipynb
├── src/
│   ├── data_loader.py
│   ├── optimizer.py        <- ERC + HRP + vol target
│   ├── backtest.py         <- walk-forward engine
│   ├── metrics.py          <- perf + factor regression
│   └── viz.py
└── reports/
    ├── report.md           <- 10-page final report
    └── figures/

```
