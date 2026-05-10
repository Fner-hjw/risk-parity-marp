# A Tradable Multi-Asset Risk Parity Strategy: Replicating CSI MARP 930929

**Fudan University — Alternative Investment Strategies, Prof. Sun Lin — May 2026**

---

## Abstract

The CSI Multi-Asset Risk Parity Index (930929) provides a transparent reference for diversified Chinese multi-asset investing but is not directly tradable. We construct a tradable proxy from five liquid onshore ETFs spanning equity, fixed income, and commodity asset classes, and we evaluate four allocation rules — Equal Risk Contribution (ERC), inverse-volatility risk parity, Hierarchical Risk Parity (HRP), and a ridge-regression replication of the official index — under a strict walk-forward protocol with the post-2021 data locked. Out-of-sample (Dec 2022–Dec 2025), the ERC portfolio with a 5% volatility overlay delivers an annualised return of 8.94% at 5.24% realised volatility, a Sharpe ratio of 1.28, and a maximum drawdown of −4.93%, halving the drawdown of an equal-weight benchmark while improving risk-adjusted performance. A factor decomposition shows that returns are not levered equity exposure (market β ≈ 0.25); a regime split confirms asymmetric protection across the 2023 sideways and 2024 rally regimes; and a gold-ablation test reveals that removing the gold leg degrades the Sharpe ratio by 0.64, identifying gold as the single largest contributor to risk-adjusted performance.

---

## 1. Introduction

The China Securities Index Multi-Asset Risk Parity Index (CSI MARP, code 930929), launched in January 2017, allocates risk budgets equally across equity, fixed income, and commodity sleeves. While the index has become a standard reference for multi-asset balance, its constituents include institutional fixed-income instruments and over-the-counter commodity exposures that retail and smaller institutional investors cannot access. This raises a practical question that this study addresses: *can the diversification properties of CSI MARP be reproduced using liquid exchange-traded funds, and do those properties survive out-of-sample?*

We answer this question in three parts. First, we construct a five-ETF universe and apply the Equal Risk Contribution algorithm of Maillard, Roncalli & Teïletche (2010) under a 5% volatility overlay. Second, we extend the analysis along two dimensions that test the robustness of the conclusion: (i) Hierarchical Risk Parity (López de Prado 2016) as an estimation-error robust alternative; and (ii) an ablation of the gold sleeve, which directly addresses the project brief's question on whether gold remains a useful portfolio asset. Third, we decompose the realised performance using both a factor regression and a regime split of the out-of-sample window.

The remainder of the report is organised as follows. Section 2 describes the data and the construction methodology. Section 3 reports the headline backtest. Section 4 conducts factor and regime analyses. Section 5 presents three robustness exercises. Section 6 discusses limitations, and Section 7 concludes.

---

## 2. Data and Methodology

### 2.1 Tradable Universe and Sample

We select five Shanghai/Shenzhen-listed ETFs that span the asset classes of CSI MARP while satisfying daily liquidity and full-sample availability constraints. The asset universe is reported in Table 1.

**Table 1 — Tradable asset universe.**

| Ticker | Name | Asset class | Inception |
|--------|------|-------------|-----------|
| 510300.SH | Huatai-PB CSI 300 ETF | Large-cap equity | 2012 |
| 510500.SH | Southern CSI 500 ETF | Mid-cap equity | 2013 |
| 511010.SH | GTJA 5Y Treasury Bond ETF | Fixed income | 2013 |
| 518880.SH | Huaan Gold ETF | Commodity (gold) | 2013 |
| 159980.SZ | Dacheng Commodity Composite ETF | Commodity (broad) | 2019 |

The sample is partitioned into a five-year in-sample window (2017-01-03 to 2021-12-31, 1,215 trading days) used for parameter selection and an out-of-sample (OOS) window beginning 2022-01-01. The OOS data is stored in a locked subdirectory and never accessed during weight estimation, hyperparameter selection, or strategy specification, eliminating look-ahead bias by construction. The broad commodity ETF (159980) only began trading 2019-10-25; combined with the 756-day (≈ three-year) lookback, this means the engine produces its first valid out-of-sample rebalance on 2022-12-05, so the realised OOS coverage is roughly three years (Dec 2022 – Dec 2025) rather than the full four-year window. Adjusted close prices for the ETFs are sourced from Yahoo Finance, the official CSI MARP and CSI 300 levels are sourced from China Securities Index Co. via the AkShare API, and a flat 2.2% per annum is used as the risk-free rate, approximating the average one-year deposit rate over the sample.

### 2.2 In-Sample Asset Statistics

Table 2 reports the asset characteristics that shape the optimisation. The cross-asset correlation structure is the foundation on which risk parity operates: the equity pair is highly correlated (ρ = 0.82), bonds are essentially uncorrelated with equities (ρ = 0.05), and gold sits between these two extremes (ρ = 0.15 with equity, 0.08 with bonds).

**Table 2 — In-sample asset statistics, 2017–2021.**

| Asset | Ann. return | Ann. vol | Sharpe | Max drawdown |
|-------|-------------|----------|--------|---------------|
| CSI 300 | 8.2% | 19.3% | 0.31 | −32.5% |
| CSI 500 | 5.1% | 22.1% | 0.14 | −38.7% |
| 5Y Treasury | 4.1% | 1.8% | 1.06 | −2.1% |
| Gold | 8.7% | 13.2% | 0.49 | −21.3% |
| Commodity | 6.3% | 14.8% | 0.29 | −25.1% |

### 2.3 Allocation Rules

We implement four allocation rules, all under a long-only, unlevered constraint $\sum_i w_i \le 1$ with any residual held in cash.

**Equal Risk Contribution (ERC).** Following Maillard, Roncalli & Teïletche (2010), we solve for weights such that each asset contributes equally to portfolio variance,

$$RC_i \;=\; w_i \cdot \frac{(\Sigma w)_i}{\sqrt{w^\top \Sigma w}} \;=\; \frac{1}{N},$$

via SLSQP minimisation of the squared deviation of risk contributions from their mean, subject to non-negativity. The resulting weights are then scaled by $\sigma_{\text{target}}/\sigma_{\text{port}}$ to a 5% annualised volatility target. We adopt 5% as a moderate, comparable target for a multi-asset balanced portfolio; the official CSI MARP index actually realises a much lower volatility (≈ 1.5% per annum, see Section 3) because its institutional fixed-income constituents are far less volatile than the listed bond ETF used here. When the natural ERC volatility on this universe exceeds 5%, scaling shrinks the risk-asset weight and the residual is held at the risk-free rate; when it is below 5%, the unlevered constraint binds and the portfolio operates at its natural volatility.

**Inverse-Volatility Risk Parity (RP).** A simpler heuristic — $w_i \propto 1/\sigma_i$ — followed by the same volatility-target rescaling. This rule ignores correlations and provides a baseline against which the full ERC optimisation can be evaluated.

**Hierarchical Risk Parity (HRP).** Following López de Prado (2016), correlations are mapped to a distance metric $d_{ij} = \sqrt{0.5(1 - \rho_{ij})}$, assets are clustered using Ward's linkage, and weights are determined by recursive bisection of the dendrogram with inverse-variance allocation at each split. HRP is included specifically as a test of robustness to estimation error in the covariance matrix, which is the principal weakness of mean-variance and ERC optimisation in small samples.

**MARP Replication (Ridge).** The official CSI MARP daily returns are projected onto the five-ETF universe via Tikhonov-regularised least squares,

$$\hat w \;=\; \arg\min_w \;\|R_{\text{MARP}} - R\,w\|_2^2 \,+\, \alpha\|w\|_2^2, \quad \alpha = 0.01,$$

with the resulting weights clipped to non-negativity and renormalised. This produces a tradable approximation of the official index from the same five-ETF universe.

### 2.4 Walk-Forward Protocol

All strategies are evaluated under an identical walk-forward protocol. At each rebalance the previous 756 trading days (≈ three years) are used to estimate inputs and compute weights; the resulting allocation is held for the next 252 trading days (annual rebalance); if optimisation fails (insufficient assets, singular covariance, non-convergence), the engine falls back to equal weight. Because the broad commodity ETF (159980) only became tradable in late 2019, the first lookback window with all five assets present begins in late 2019 and the first valid OOS rebalance therefore lands on 2022-12-05; subsequent rebalances follow at 252-trading-day intervals through the end of the sample (2025-12-31). The base engine reports gross-of-cost performance; transaction-cost sensitivity is examined separately in Section 5.3.

---

## 3. Empirical Results

### 3.1 Out-of-Sample Performance

Table 3 summarises the OOS performance (Dec 2022 – Dec 2025) of the four strategies, the equal-weight benchmark, and the official CSI MARP index. All Sharpe ratios use the 2.2% per annum risk-free rate. Headline metrics are reported gross of transaction costs; Section 5.3 documents the cost sensitivity.

**Table 3 — Out-of-sample performance, 2022-12 to 2025-12 (gross of costs).**

| Strategy | Ann. return | Ann. vol | Sharpe | Sortino | Calmar | Max DD |
|----------|------------:|---------:|-------:|--------:|-------:|-------:|
| ERC (5% target) | 8.94% | 5.24% | **1.28** | 1.63 | 1.81 | −4.93% |
| Inverse-vol RP | 7.37% | 3.18% | 1.63 | 2.17 | 3.40 | −2.17% |
| HRP | 4.03% | 1.45% | 1.26 | 1.76 | 3.12 | −1.29% |
| Equal weight | 12.86% | 9.08% | 1.17 | 1.53 | 1.34 | −9.63% |
| MARP replication | 18.48% | 9.07% | 1.80 | 2.29 | 3.67 | −5.04% |
| CSI MARP (official) | 5.32% | 1.51% | 2.07 | — | — | — |

Three observations follow. First, ERC achieves a 1.28 Sharpe at a realised volatility of 5.24%, almost exactly matching the 5% target — a direct verification that the volatility-overlay machinery operates as designed. The portfolio holds approximately 42% in cash on average; this is the consequence of the unlevered constraint, since the natural ERC volatility on the five-asset universe exceeds 5% throughout the OOS period. Second, ERC dominates equal weight on every risk-adjusted metric — the Sharpe is 0.11 higher and the maximum drawdown is roughly half. Third, the ridge-based MARP replication delivers the highest absolute Sharpe (1.80) but at a volatility of 9% it is not in fact a 5%-volatility product; it tracks the *shape* of the official index well while operating at the natural unlevered volatility of the constituents. The official CSI MARP figures are reported for completeness but realise only ≈ 1.5% volatility on average — well below any tradable target achievable on this ETF universe — because the underlying institutional fixed-income constituents are far less volatile than the listed bond ETF.

The inverse-volatility and HRP strategies both record higher Sharpe ratios than ERC but are constrained by their natural volatilities (3.18% and 1.45% respectively), which the unlevered framework cannot scale up to the 5% target. HRP's 1.45% volatility is the consequence of Ward-linkage clustering concentrating capital in the low-volatility bond cluster — a known sensitivity discussed in §5.3.

### 3.2 Cumulative Performance and Annual Decomposition

Figure 1 plots the OOS cumulative growth of one yuan invested in each strategy, alongside the official CSI MARP index. The visual story is consistent with Table 3: ERC tracks a path between the more aggressive equal-weight and MARP replication portfolios and the very conservative HRP and inverse-volatility paths, with the smoothest profile of any 5%-target strategy.

![Figure 1 — Cumulative out-of-sample performance, Dec 2022–Dec 2025.](figures/test_cumulative.png)

The annual decomposition (Table 4) makes the behaviour over the cycle explicit. The 2022 row is omitted because the engine has no portfolio returns prior to its first valid OOS rebalance on 2022-12-05. In the 2024 rally, ERC participated for 8.9%, well below equal weight's 13.6%, reflecting the cash buffer required by the volatility overlay; the same pattern holds in 2025.

**Table 4 — Annual total returns by strategy.**

| Year | ERC (5%) | Equal weight | MARP rep | CSI MARP |
|------|---------:|-------------:|---------:|---------:|
| 2023 | 1.9% | 0.0% | 5.4% | 5.3% |
| 2024 | 8.9% | 13.6% | 18.8% | 8.3% |
| 2025 | 16.0% | 26.5% | 32.6% | 5.4% |

---

## 4. Factor and Regime Analysis

### 4.1 Factor Decomposition

To establish whether the OOS performance reflects genuine diversification or merely a hidden equity tilt, we regress daily strategy returns on two tradable factors constructed from the same universe: the market excess return (CSI 300 minus the risk-free rate) and a size-style proxy (CSI 500 ETF minus CSI 300 ETF). A value factor is unavailable from this ETF universe and is omitted; this limitation is revisited in §6. Table 5 reports the OLS estimates over the OOS window with t-statistics.

**Table 5 — Factor regression on out-of-sample returns.**

| Strategy | α (annualised) | β_mkt | β_smb | R² |
|----------|--------------:|------:|------:|----:|
| ERC (5%) | 7.4% (5.0) | 0.25 (45.2) | 0.11 (13.2) | 0.76 |
| Equal weight | 10.1% (4.3) | 0.45 (50.7) | 0.19 (13.7) | 0.80 |
| MARP replication | 15.4% (4.2) | 0.37 (26.9) | 0.12 (5.4) | 0.52 |

t-statistics in parentheses. Because the strategies and the factor proxies are both built from the same five-ETF universe, the regressions are close to mechanical decompositions and the t-statistics are correspondingly large; they should be read as indicators of consistency rather than as classical inference. ERC's market beta of 0.25 is materially below the equal-weight value of 0.45, confirming that the bond, gold, and cash legs deliver real equity-risk diversification. The annualised alpha of 7.4% is an artefact of the cash leg earning the risk-free rate while the regression contains no cash factor; it reflects the un-modelled cash term, not skill. The R² of 0.76 indicates that the two-factor model captures most of the systematic variation, with the unexplained residual concentrated in bond and commodity beta.

### 4.2 Regime Decomposition

Table 6 splits the OOS window into three distinct regimes defined by CSI 300 price action; the engine begins producing portfolio returns on 2022-12-05, so the 2022 bear-market regime falls outside the realised OOS window and is not reported. During the 2023 sideways/correction regime, when CSI 300 fell 22.6%, ERC drew down only 2.2% — a ten-fold reduction in downside relative to a passive equity exposure and a 5.1 ppt advantage over the equal-weight portfolio. In the 2024 stimulus-driven rally, ERC participated for 11.8% versus equal weight's 19.3%, capturing roughly 56% of the upside.

**Table 6 — Strategy total returns by regime.**

| Regime | Period | CSI 300 | ERC (5%) | Equal | MARP rep |
|--------|--------|--------:|---------:|------:|---------:|
| 2023 Sideways | Feb 2023 – Jan 2024 | −22.6% | −2.2% | −7.3% | −0.5% |
| 2024 Rally | Feb – Oct 2024 | +21.0% | +11.8% | +19.3% | +23.1% |
| 2024–25 Consolidation | Nov 2024 – Dec 2025 | +19.0% | +15.5% | +25.2% | +31.7% |

The regime profile is consistent with the central claim of the risk parity literature: the strategy is designed to deliver asymmetric protection — sacrificing upside participation in exchange for reduced downside — and over a complete cycle this asymmetry compounds into superior risk-adjusted performance, which is precisely what Table 3 documents.

---

## 5. Robustness

### 5.1 Gold Ablation

The role of gold in 2024–2025 is a topical question for Chinese multi-asset investors given the elevated price level. We re-estimate the ERC strategy on the four-asset universe excluding the gold ETF; Table 7 reports the resulting performance against the full five-asset specification. Removing gold reduces the annualised return from 8.94% to 5.46%, leaves volatility essentially unchanged, deteriorates the Sharpe from 1.28 to 0.64, and worsens the maximum drawdown by 71 bps. The Sharpe deterioration of 0.64 is the single largest sensitivity in the entire study.

**Table 7 — Gold ablation, ERC (5%) strategy, OOS.**

| Metric | With gold | Without gold | Δ |
|--------|----------:|-------------:|----:|
| Annualised return | 8.94% | 5.46% | −3.48% |
| Annualised volatility | 5.24% | 5.09% | −0.15% |
| Sharpe ratio | 1.28 | 0.64 | −0.64 |
| Maximum drawdown | −4.93% | −5.64% | −0.71% |

Decomposing this effect by regime, gold's contribution is concentrated in the 2024–2025 rally, where its appreciation alongside equities lifted the cumulative path. The conclusion of the ablation is therefore unambiguous: in a multi-asset portfolio constructed under a risk-parity rule, gold remains a materially valuable diversifier through 2025.

### 5.2 Drawdown Profile

Figure 2 displays the OOS drawdown trajectories. ERC and the official CSI MARP exhibit shallow, mean-reverting drawdowns that recover within months; equal weight and MARP replication exhibit deeper, slower recoveries; HRP shows almost no drawdown but at the cost of negligible return. The pattern provides direct visual confirmation that the volatility overlay translates into the desired drawdown control.

![Figure 2 — Drawdown trajectories, OOS.](figures/test_drawdown.png)

### 5.3 Cost Sensitivity and HRP Diagnostic

Table 8 stress-tests the ERC Sharpe ratio against transaction costs, applied as a per-rebalance turnover charge against the gross-of-cost backtest. Because the engine rebalances annually and the realised OOS window contains only two such rebalances, the absolute cost drag is modest: even at 100 bps one-way — an order of magnitude above realistic friction — the Sharpe falls only to 1.20. At realistic Chinese-ETF friction (5–10 bps one-way), the impact is essentially undetectable.

**Table 8 — Cost sensitivity, ERC (5%).**

| One-way cost (bps) | 0 | 10 | 20 | 50 | 100 |
|---------------------:|------:|------:|------:|------:|------:|
| Annualised return | 8.94% | 8.89% | 8.84% | 8.70% | 8.47% |
| Sharpe | 1.28 | 1.28 | 1.27 | 1.24 | 1.20 |

The HRP underperformance noted in §3 reflects a known property of Ward-linkage clustering on assets with widely heterogeneous volatilities. With the bond ETF at ≈ 2% volatility and equity ETFs at ≈ 20%, the dendrogram concentrates capital in the low-volatility cluster and the recursive inverse-variance step amplifies this concentration further, leaving a portfolio that is unambiguously safe but has very limited risk-asset participation. Standard remedies in the literature — pre-processing with volatility parity prior to clustering, single or complete linkage, or a minimum-variance constraint at each bisection — are not implemented here in order to preserve the integrity of the locked OOS protocol, but all are natural extensions for future research.

---

## 6. Limitations

The study has six limitations worth acknowledging. *First*, the size factor is proxied by the spread between the CSI 500 ETF and the CSI 300 ETF, which captures a mid-cap minus large-cap differential rather than a true small-minus-big sort, and which uses ETF returns rather than the underlying index returns; the difference is small in practice (a few basis points of tracking error) but should be acknowledged. The value factor is unavailable from this ETF universe and is omitted entirely. Constructing canonical Fama–French factors for China would require security-level data from CSMAR, Wind, or RESSET. *Second*, five ETFs cannot fully span the CSI MARP constituent universe, which includes international equities, broader fixed-income exposures, and several commodity sub-classes. The broad commodity ETF (159980) only begins trading in late 2019, which materially limits early-sample diversification and pushes the first valid out-of-sample rebalance to December 2022. *Third*, the framework is unlevered by construction; when the natural volatility of an inverse-volatility or HRP portfolio falls below the 5% target, the unlevered constraint binds and the target cannot be reached. *Fourth*, the risk-free rate is held constant at 2.2% per annum; in reality, Chinese deposit rates varied between roughly 1.5% and 3.5% over the sample, and modelling this time variation would refine both the cash leg returns and the Sharpe estimates. *Fifth*, the headline performance metrics are reported gross of transaction costs; the cost-sensitivity stress test in §5.3 is run as a separate post-hoc deduction rather than being integrated into the main backtest engine, although the test indicates that this assumption is not material for the headline conclusion. *Sixth*, all ETFs in the universe survived to the end of the sample, so survivorship bias is not modelled; this is a minor concern for the established broad-market ETFs used here but would be a serious issue if the universe were extended to thematic or sector products. We also note that an extension comparing this risk-parity construction with hedge-fund factor-replication portfolios (in the spirit of Hasanhodzic & Lo, 2007) is a natural avenue for future work but was not pursued here, as a defensible comparison would require access to actual hedge-fund index return histories that are not available in our data sources.

---

## 7. Conclusion

We have shown that a tradable five-ETF approximation of the CSI MARP risk-parity construction, implemented under the Equal Risk Contribution rule with a 5% volatility overlay and evaluated under a strict locked-out-of-sample walk-forward protocol, delivers the diversification properties that risk parity promises. Out-of-sample (Dec 2022 – Dec 2025), the strategy produces a Sharpe ratio of 1.28, an annualised return of 8.94% at a realised volatility of 5.24%, and a maximum drawdown of −4.93% — outperforming the equal-weight benchmark on every risk-adjusted metric. The factor regression confirms the strategy is not levered equity exposure (market β = 0.25), the regime decomposition confirms asymmetric protection across the 2023 sideways and 2024 rally regimes, and the gold-ablation test identifies gold as the single largest contributor to risk-adjusted performance — which directly answers the project brief's gold-allocation question in the affirmative through 2025. The full pipeline — from data ingestion to weight optimisation to factor regression — runs end-to-end on five liquid Chinese ETFs and is therefore directly implementable for retail and smaller institutional investors who require multi-asset diversification but cannot access the institutional instruments that underlie the official index.

---

## References

Hasanhodzic, J., & Lo, A. W. (2007). Can hedge-fund returns be replicated? The linear case. *Journal of Investment Management*, 5(2), 5–45.

López de Prado, M. (2016). Building diversified portfolios that outperform out of sample. *Journal of Portfolio Management*, 42(4), 59–69.

Maillard, S., Roncalli, T., & Teïletche, J. (2010). The properties of equally weighted risk contribution portfolios. *Journal of Portfolio Management*, 36(4), 60–70.

Roncalli, T. (2013). *Introduction to Risk Parity and Budgeting*. Chapman and Hall/CRC.

China Securities Index Co., Ltd. (2017). *CSI Multi-Asset Risk Parity Index (930929) — Index Methodology*. csindex.com.cn.

China Securities Index Co., Ltd. CSI MARP 930929 and CSI 300 daily index levels, 2017–2025. csindex.com.cn.

East Money. CSI 300 index levels and open-ended fund net asset values, 2017–2025. eastmoney.com.

Yahoo Finance. Adjusted close prices for ETFs 510300.SH, 510500.SH, 511010.SH, 518880.SH, and 159980.SZ, 2017–2025. finance.yahoo.com.

---

*This report and all underlying code, data, and figures are original work produced for the Fudan University Alternative Investment Strategies course. All external data sources are attributed in the references.*
