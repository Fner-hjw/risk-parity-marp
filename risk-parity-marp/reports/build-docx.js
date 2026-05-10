const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  Header, Footer, AlignmentType, ImageRun,
  HeadingLevel, BorderStyle, WidthType, ShadingType,
  PageNumber,
} = require("docx");

// Layout constants
const A4_W = 11906;
const MARGIN = 1080; // 0.75"
const CONTENT_W = A4_W - 2 * MARGIN;

const FONT = "Times New Roman";
const BODY = 20;        // 10 pt
const BODY_TABLE = 18;  // 9 pt
const SMALL = 16;       // 8 pt
const H1 = 24;          // 12 pt
const H2 = 22;          // 11 pt
const TITLE_SZ = 30;    // 15 pt
const SUB_SZ = 22;      // 11 pt

const ACCENT = "1F3A5F";
const HEADER_BG = "1F3A5F";
const ROW_ALT = "F2F4F7";
const BORDER_COLOR = "9CA3AF";

// Paragraph helpers
function run(text, opts = {}) {
  return new TextRun({
    text,
    font: FONT,
    size: opts.size || BODY,
    bold: opts.bold,
    italics: opts.italics,
    color: opts.color,
  });
}

function para(text, opts = {}) {
  const children = Array.isArray(text)
    ? text.map(t => (typeof t === "string"
        ? run(t, opts)
        : new TextRun({ font: FONT, size: opts.size || BODY, ...t })))
    : [run(text, opts)];
  return new Paragraph({
    spacing: { before: opts.before || 0, after: opts.after || 100, line: opts.line || 264 },
    alignment: opts.align,
    indent: opts.indent,
    children,
    ...(opts.heading ? { heading: opts.heading } : {}),
    ...(opts.breakBefore ? { pageBreakBefore: true } : {}),
  });
}

function justified(text, opts = {}) {
  return para(text, { align: AlignmentType.JUSTIFIED, after: 100, line: 264, ...opts });
}

function h1(text) {
  return para(text, {
    heading: HeadingLevel.HEADING_1,
    size: H1, bold: true, color: ACCENT,
    before: 200, after: 80,
  });
}

function h2(text) {
  return para(text, {
    heading: HeadingLevel.HEADING_2,
    size: H2, bold: true, color: ACCENT,
    before: 140, after: 60,
  });
}

function caption(text) {
  return para(text, {
    size: SMALL, italics: true, color: "555555",
    align: AlignmentType.CENTER,
    after: 80, before: 0,
  });
}

function eqLine(text) {
  return para(text, {
    italics: true, align: AlignmentType.CENTER,
    after: 80, before: 60, line: 240,
  });
}

// Table builder
function makeTable(headers, rows, colWidths) {
  const border = { style: BorderStyle.SINGLE, size: 1, color: BORDER_COLOR };
  const borders = { top: border, bottom: border, left: border, right: border };
  const cellMargins = { top: 30, bottom: 30, left: 70, right: 70 };

  const headerRow = new TableRow({
    children: headers.map((h, i) => new TableCell({
      borders,
      width: { size: colWidths[i], type: WidthType.DXA },
      shading: { fill: HEADER_BG, type: ShadingType.CLEAR },
      margins: cellMargins,
      children: [new Paragraph({
        spacing: { before: 0, after: 0 },
        children: [new TextRun({ text: h, font: FONT, size: BODY_TABLE, bold: true, color: "FFFFFF" })],
      })],
    })),
    tableHeader: true,
  });

  const dataRows = rows.map((row, ri) => new TableRow({
    children: row.map((cell, ci) => new TableCell({
      borders,
      width: { size: colWidths[ci], type: WidthType.DXA },
      shading: ri % 2 === 1 ? { fill: ROW_ALT, type: ShadingType.CLEAR } : undefined,
      margins: cellMargins,
      children: [new Paragraph({
        spacing: { before: 0, after: 0 },
        alignment: ci === 0 ? AlignmentType.LEFT : AlignmentType.CENTER,
        children: [new TextRun({ text: String(cell), font: FONT, size: BODY_TABLE })],
      })],
    })),
  }));

  return new Table({
    width: { size: CONTENT_W, type: WidthType.DXA },
    columnWidths: colWidths,
    rows: [headerRow, ...dataRows],
  });
}

function tableTitle(text) {
  return para(text, { bold: true, size: BODY_TABLE, after: 40, before: 60 });
}

// Image helper
function image(filePath, widthPx, heightPx) {
  const data = fs.readFileSync(filePath);
  return new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { before: 80, after: 40 },
    children: [new ImageRun({
      data,
      transformation: { width: widthPx, height: heightPx },
    })],
  });
}

// Document
const REPORTS_DIR = path.dirname(__filename);
const FIG_CUM = path.join(REPORTS_DIR, "figures", "test_cumulative.png");
const FIG_DD = path.join(REPORTS_DIR, "figures", "test_drawdown.png");

const doc = new Document({
  styles: {
    default: { document: { run: { font: FONT, size: BODY } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: H1, bold: true, font: FONT, color: ACCENT },
        paragraph: { spacing: { before: 200, after: 80 }, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: H2, bold: true, font: FONT, color: ACCENT },
        paragraph: { spacing: { before: 140, after: 60 }, outlineLevel: 1 } },
    ],
  },
  sections: [{
    properties: {
      page: {
        size: { width: A4_W, height: 16838 },
        margin: { top: MARGIN, right: MARGIN, bottom: MARGIN, left: MARGIN },
      },
    },
    headers: {
      default: new Header({
        children: [new Paragraph({
          border: { bottom: { style: BorderStyle.SINGLE, size: 4, color: ACCENT, space: 4 } },
          spacing: { after: 60 },
          alignment: AlignmentType.RIGHT,
          children: [new TextRun({
            text: "Tradable Multi-Asset Risk Parity, Fudan AIS 2026",
            font: FONT, size: 14, italics: true, color: "555555",
          })],
        })],
      }),
    },
    footers: {
      default: new Footer({
        children: [new Paragraph({
          border: { top: { style: BorderStyle.SINGLE, size: 2, color: BORDER_COLOR, space: 4 } },
          alignment: AlignmentType.CENTER,
          children: [
            new TextRun({ text: "Page ", font: FONT, size: 14, color: "555555" }),
            new TextRun({ children: [PageNumber.CURRENT], font: FONT, size: 14, color: "555555" }),
            new TextRun({ text: " of ", font: FONT, size: 14, color: "555555" }),
            new TextRun({ children: [PageNumber.TOTAL_PAGES], font: FONT, size: 14, color: "555555" }),
          ],
        })],
      }),
    },
    children: [
      // Title block
      para("A Tradable Multi-Asset Risk Parity Strategy",
        { size: TITLE_SZ, bold: true, align: AlignmentType.CENTER, after: 60, before: 0 }),
      para("Replicating CSI MARP 930929",
        { size: SUB_SZ, italics: true, align: AlignmentType.CENTER, after: 60 }),
      para("Fudan University, Alternative Investment Strategies, Prof. Sun Lin, May 2026",
        { size: BODY, align: AlignmentType.CENTER, after: 0 }),
      new Paragraph({
        spacing: { before: 100, after: 100 },
        border: { bottom: { style: BorderStyle.SINGLE, size: 6, color: ACCENT, space: 1 } },
        children: [],
      }),

      // Abstract
      h1("Abstract"),
      justified("The CSI Multi-Asset Risk Parity Index (930929) provides a transparent reference for diversified Chinese multi-asset investing but is not directly tradable. We construct a tradable proxy from five liquid onshore ETFs spanning equity, fixed income, and commodity asset classes, and we evaluate four allocation rules, Equal Risk Contribution (ERC), inverse-volatility risk parity, Hierarchical Risk Parity (HRP), and a ridge-regression replication of the official index, under a strict walk-forward protocol with the post-2021 data locked. Out-of-sample (Dec 2022–Dec 2025), the ERC portfolio with a 5% volatility overlay delivers an annualised return of 6.74% at 2.56% realised volatility (the unlevered constraint binds well below the 5% target), a Sharpe ratio of 1.77, and a maximum drawdown of −1.63%, cutting the drawdown of an equal-weight benchmark by more than 80% while improving risk-adjusted performance. A factor decomposition shows that returns are not levered equity exposure (market β ≈ 0.09); a regime split confirms asymmetric protection across the 2023 sideways and 2024 rally regimes; and a gold-ablation test reveals that removing the gold leg degrades the Sharpe ratio by 0.76, showing that the gold leg is a material contributor to risk-adjusted performance."),

      // 1. Introduction
      h1("1. Introduction"),
      justified("The China Securities Index Multi-Asset Risk Parity Index (CSI MARP, code 930929), launched in January 2017, allocates risk budgets equally across equity, fixed income, and commodity sleeves. While the index has become a standard reference for multi-asset balance, its constituents include institutional fixed-income instruments and over-the-counter commodity exposures that retail and smaller institutional investors cannot access. This raises a practical question that this study addresses: can the diversification properties of CSI MARP be reproduced using liquid exchange-traded funds, and do those properties survive out-of-sample?"),
      justified("We answer this question in three parts. First, we construct a five-ETF universe and apply the Equal Risk Contribution algorithm of Maillard, Roncalli & Teïletche (2010) under a 5% volatility overlay. Second, we extend the analysis along two dimensions that test the robustness of the conclusion: (i) Hierarchical Risk Parity (López de Prado 2016) as an estimation-error robust alternative; and (ii) an ablation of the gold sleeve, which directly addresses the project brief's question on whether gold remains a useful portfolio asset. Third, we decompose the realised performance using both a factor regression and a regime split of the out-of-sample window."),
      justified("The remainder of the report is organised as follows. Section 2 describes the data and the construction methodology. Section 3 reports the headline backtest. Section 4 conducts factor and regime analyses. Section 5 presents three robustness exercises. Section 6 discusses limitations, and Section 7 concludes."),

      // 2. Data and Methodology
      h1("2. Data and Methodology"),
      h2("2.1 Tradable Universe and Sample"),
      justified("We select five Shanghai/Shenzhen-listed ETFs that span the asset classes of CSI MARP while satisfying daily liquidity and full-sample availability constraints. The asset universe is reported in Table 1."),
      tableTitle("Table 1: Tradable asset universe."),
      makeTable(
        ["Ticker", "Name", "Asset class", "Inception"],
        [
          ["510300.SH", "Huatai-PB CSI 300 ETF", "Large-cap equity", "2012"],
          ["510500.SH", "Southern CSI 500 ETF", "Mid-cap equity", "2013"],
          ["511010.SH", "GTJA 5Y Treasury Bond ETF", "Fixed income", "2013"],
          ["518880.SH", "Huaan Gold ETF", "Commodity (gold)", "2013"],
          ["159980.SZ", "Dacheng Commodity Composite ETF", "Commodity (broad)", "2019"],
        ],
        [1500, 4200, 3000, 1746]
      ),
      para("", { after: 80 }),
      justified("The sample is partitioned into a five-year in-sample window (2017-01-03 to 2021-12-31, 1,221 trading days) used to characterise asset behaviour. All allocation hyperparameters (lookback, rebalance frequency, volatility target) are set a priori rather than tuned in-sample. The out-of-sample (OOS) window begins 2022-01-01. The OOS data is stored in a locked subdirectory and never accessed during weight estimation, hyperparameter selection, or strategy specification, eliminating look-ahead bias by construction. The broad commodity ETF (159980) only began trading 2019-10-25; combined with the 756-day (≈ three-year) lookback, this means the engine produces its first valid out-of-sample rebalance on 2022-12-05, so the realised OOS coverage is roughly three years (Dec 2022 to Dec 2025) rather than the full four-year window. Adjusted close prices for the ETFs are sourced from Yahoo Finance, the official CSI MARP and CSI 300 levels are sourced from China Securities Index Co. via the AkShare API, and a flat 2.2% per annum is used as the risk-free rate, approximating the average one-year deposit rate over the sample."),

      h2("2.2 In-Sample Asset Statistics"),
      justified("Table 2 reports the asset characteristics that shape the optimisation. The cross-asset correlation structure is the foundation on which risk parity operates: the equity pair is highly correlated (ρ = 0.82); equities are negatively correlated with the bond ETF (ρ ≈ −0.22), confirming bonds as a diversifier; equities are essentially uncorrelated with gold (ρ ≈ 0.00); and gold–bond shows mild positive correlation (ρ ≈ 0.17)."),
      tableTitle("Table 2: In-sample asset statistics, 2017–2021."),
      makeTable(
        ["Asset", "Ann. return", "Ann. vol", "Sharpe", "Max drawdown"],
        [
          ["CSI 300 (510300)", "10.0%", "19.5%", "0.40", "−31.3%"],
          ["CSI 500 (510500)", "4.7%", "21.1%", "0.12", "−39.2%"],
          ["5Y Treasury (511010)", "2.8%", "2.4%", "0.25", "−4.9%"],
          ["Gold (518880)", "7.0%", "12.6%", "0.38", "−20.5%"],
          ["Commodity (159980)", "20.7%", "15.4%", "1.20", "−22.4%"],
        ],
        [2700, 1700, 1700, 1500, 2746]
      ),
      para("159980 IS coverage starts 2019-10-25 due to inception.",
        { italics: true, size: SMALL, color: "555555", after: 80 }),

      h2("2.3 Allocation Rules"),
      justified("We implement four allocation rules, all under a long-only, unlevered constraint (Σ wᵢ ≤ 1) with any residual held in cash."),
      justified([
        { text: "Equal Risk Contribution (ERC). ", bold: true },
        { text: "Following Maillard, Roncalli & Teïletche (2010), we solve for weights such that each asset contributes equally to portfolio variance," },
      ]),
      eqLine("RCᵢ = wᵢ · (Σw)ᵢ / √(wᵀ Σ w) = 1/N,"),
      justified("via SLSQP minimisation of the squared deviation of risk contributions from their mean, subject to non-negativity. The resulting weights are then scaled by σ_target/σ_port to a 5% annualised volatility target. We adopt 5% as a moderate, comparable target for a multi-asset balanced portfolio; the official CSI MARP index actually realises a much lower volatility (≈ 1.5% per annum, see Section 3), possibly because the underlying institutional fixed-income constituents are less volatile than the listed bond ETF used here, a definitive answer would require the published CSI methodology document. When the natural ERC volatility on this universe exceeds 5%, scaling shrinks the risk-asset weight and the residual is held at the risk-free rate; when it is below 5%, the unlevered constraint binds and the portfolio operates at its natural volatility."),
      justified([
        { text: "Inverse-Volatility Risk Parity (RP). ", bold: true },
        { text: "A simpler heuristic, wᵢ ∝ 1/σᵢ, followed by the same volatility-target rescaling. This rule ignores correlations and provides a baseline against which the full ERC optimisation can be evaluated." },
      ]),
      justified([
        { text: "Hierarchical Risk Parity (HRP). ", bold: true },
        { text: "Following López de Prado (2016), correlations are mapped to a distance metric dᵢⱼ = √(0.5(1 − ρᵢⱼ)), assets are clustered using Ward's linkage, and weights are determined by recursive bisection of the dendrogram with inverse-variance allocation at each split. HRP is included specifically as a test of robustness to estimation error in the covariance matrix, which is the principal weakness of mean-variance and ERC optimisation in small samples." },
      ]),
      justified([
        { text: "MARP Replication (Ridge). ", bold: true },
        { text: "The official CSI MARP daily returns are projected onto the five-ETF universe via Tikhonov-regularised least squares," },
      ]),
      eqLine("ŵ = argmin_w ‖R_MARP − R w‖₂² + α‖w‖₂²,    α = 0.01,"),
      justified("with the resulting weights clipped to non-negativity and renormalised. This produces a tradable approximation of the official index from the same five-ETF universe."),

      h2("2.4 Walk-Forward Protocol"),
      justified("All strategies are evaluated under an identical walk-forward protocol. At each rebalance the previous 756 trading days (≈ three years) are used to estimate inputs and compute weights; the resulting allocation is held for the next 252 trading days (annual rebalance); if optimisation fails (insufficient assets, singular covariance, non-convergence), the engine falls back to equal weight. Because the broad commodity ETF (159980) only became tradable in late 2019, the first lookback window with all five assets present begins in late 2019 and the first valid OOS rebalance therefore lands on 2022-12-05; subsequent rebalances follow at 252-trading-day intervals through the end of the sample (2025-12-31). The base engine reports gross-of-cost performance; transaction-cost sensitivity is examined separately in Section 5.3. In the realised OOS window only two rebalances fire, 2022-12-05 and 2023-12-15, because the engine excludes any rebalance that lacks a full 252-day forward window. The second set of weights therefore applies for roughly two years, until the end of the sample."),

      // 3. Empirical Results
      h1("3. Empirical Results"),
      h2("3.1 Out-of-Sample Performance"),
      justified("Table 3 summarises the OOS performance (Dec 2022 to Dec 2025) of the four strategies, the equal-weight benchmark, and the official CSI MARP index. All Sharpe ratios use the 2.2% per annum risk-free rate. Headline metrics are reported gross of transaction costs; Section 5.3 documents the cost sensitivity."),
      tableTitle("Table 3: Out-of-sample performance, 2022-12 to 2025-12 (gross of costs)."),
      makeTable(
        ["Strategy", "Ann. return", "Ann. vol", "Sharpe", "Sortino", "Calmar", "Max DD"],
        [
          ["ERC (5% target)", "6.74%", "2.56%", "1.77", "2.39", "4.13", "−1.63%"],
          ["Inverse-vol RP", "7.37%", "3.18%", "1.63", "2.17", "3.40", "−2.17%"],
          ["HRP", "4.03%", "1.45%", "1.26", "1.76", "3.12", "−1.29%"],
          ["Equal weight", "12.86%", "9.08%", "1.17", "1.53", "1.34", "−9.63%"],
          ["MARP replication", "18.48%", "9.07%", "1.80", "2.29", "3.67", "−5.04%"],
          ["CSI MARP (official)", "6.27%", "1.56%", "2.61", "-", "-", "−1.08%"],
        ],
        [2400, 1300, 1100, 1000, 1100, 1100, 1346]
      ),
      para("", { after: 80 }),
      justified("Three observations follow. First, the converged ERC weights concentrate roughly three-quarters in the bond ETF (e.g. 0.06/0.05/0.73/0.08/0.08 at 2022-12-05), as required to equalise risk contributions across assets with bond volatility ≈ 2% and equity volatility ≈ 20%. The natural unlevered ERC volatility on this universe is ≈ 2.5–3.4%, so the 5% target is never reached, the unlevered constraint binds and the portfolio is fully invested in risky assets (no cash leg). Second, ERC delivers a Sharpe of 1.77 versus equal weight's 1.17, with the maximum drawdown shrinking from −9.6% to −1.6%; the price of this risk-adjusted improvement is forgone absolute return, since ERC participates in only a fraction of the equal-weight upside. Third, the ridge-based MARP replication delivers the highest absolute Sharpe (1.80) but at a volatility of 9% it is not in fact a 5%-volatility product; it operates at the natural unlevered volatility of the constituents rather than at the official index's much lower realised volatility. The official CSI MARP figures are reported for completeness but realise only ≈ 1.5% volatility on average, well below any tradable target achievable on this ETF universe, possibly because the underlying institutional fixed-income constituents are less volatile than the listed bond ETF; a definitive answer requires the published methodology document."),
      justified("The inverse-volatility and HRP strategies record realised volatilities of 3.18% and 1.45% respectively, both below the 5% target, the unlevered framework cannot scale them up. ERC, despite operating at 2.56% rather than 5%, posts a higher Sharpe than RP and HRP. HRP's 1.45% volatility is the consequence of Ward-linkage clustering concentrating capital in the low-volatility bond cluster, a known sensitivity discussed in Section 5.3."),

      h2("3.2 Cumulative Performance and Annual Decomposition"),
      justified("Figure 1 plots the OOS cumulative growth of one yuan invested in each strategy, alongside the official CSI MARP index. The visual story is consistent with Table 3: ERC tracks the most conservative path among the four ETF strategies (apart from HRP), reflecting its bond-heavy composition, while equal-weight and MARP replication deliver the steepest cumulative trajectories at the cost of much larger drawdowns."),
      image(FIG_CUM, 470, 232),
      caption("Figure 1: Cumulative out-of-sample performance, Dec 2022–Dec 2025."),
      justified("The annual decomposition (Table 4) makes the behaviour over the cycle explicit. The 2022 row is omitted because the engine has no portfolio returns prior to its first valid OOS rebalance on 2022-12-05. In the 2024 rally, ERC returned 8.7%, well below equal weight's 13.6%, reflecting the bond-heavy allocation; the participation gap is even larger in 2025 (8.6% vs 26.5%)."),
      tableTitle("Table 4: Annual total returns by strategy."),
      makeTable(
        ["Year", "ERC (5%)", "Equal weight", "MARP rep", "CSI MARP"],
        [
          ["2023", "2.7%", "0.0%", "5.4%", "5.3%"],
          ["2024", "8.7%", "13.6%", "18.8%", "8.3%"],
          ["2025", "8.6%", "26.5%", "32.6%", "5.4%"],
        ],
        [1500, 1900, 2100, 1900, 1846]
      ),
      para("", { after: 80 }),

      // 4. Factor and Regime Analysis
      h1("4. Factor and Regime Analysis"),
      h2("4.1 Factor Decomposition"),
      justified("To establish whether the OOS performance reflects genuine diversification or merely a hidden equity tilt, we regress daily strategy returns on two tradable factors constructed from the same universe: the market excess return (CSI 300 minus the risk-free rate) and a size-style proxy (CSI 500 ETF minus CSI 300 ETF). A value factor is unavailable from this ETF universe and is omitted; this limitation is revisited in Section 6. Table 5 reports the OLS estimates over the OOS window with t-statistics."),
      tableTitle("Table 5: Factor regression on out-of-sample returns."),
      makeTable(
        ["Strategy", "α (annualised)", "β_mkt", "β_smb", "R²"],
        [
          ["ERC (5%)", "3.8% (3.4)", "0.09 (22.2)", "0.03 (5.0)", "0.43"],
          ["Equal weight", "7.4% (3.2)", "0.45 (51.3)", "0.21 (15.1)", "0.81"],
          ["MARP replication", "12.9% (3.5)", "0.37 (26.8)", "0.13 (6.1)", "0.52"],
        ],
        [2600, 2300, 2200, 2200, 1846]
      ),
      para("", { after: 80 }),
      justified("t-statistics in parentheses. Because the strategies and the factor proxies are both built from the same five-ETF universe, the regressions are close to mechanical decompositions and the t-statistics are correspondingly large; they should be read as indicators of consistency rather than as classical inference. ERC's market beta of 0.09 is materially below the equal-weight value of 0.45, confirming that the bond and gold legs deliver real equity-risk diversification. The annualised alpha of 3.8% reflects all return contributions not captured by the two-factor model, bonds, gold, and commodity, rather than skill. With only equity-mkt and a size proxy on the right-hand side, every non-equity return source mechanically loads on alpha, and for ERC the bond leg dominates that residual. The R² of 0.43 is correspondingly low: only a minority of ERC's variance is explained by the equity factors, consistent with its bond-heavy composition."),

      h2("4.2 Regime Decomposition"),
      justified("Table 6 splits the OOS window into three distinct regimes defined by CSI 300 price action; the engine begins producing portfolio returns on 2022-12-05, so the 2022 bear-market regime falls outside the realised OOS window and is not reported. During the 2023 sideways/correction regime, when CSI 300 fell 22.6%, ERC delivered a small positive total return of +1.2%, versus −7.3% for equal weight. In the 2024 stimulus-driven rally, ERC participated for 7.9% versus equal weight's 19.3%, capturing roughly 41% of the upside."),
      tableTitle("Table 6: Strategy total returns by regime."),
      makeTable(
        ["Regime", "Period", "CSI 300", "ERC (5%)", "Equal", "MARP rep"],
        [
          ["2023 Sideways", "Feb 2023–Jan 2024", "−22.6%", "+1.2%", "−7.3%", "−0.5%"],
          ["2024 Rally", "Feb–Oct 2024", "+21.0%", "+7.9%", "+19.3%", "+23.1%"],
          ["2024–25 Consolidation", "Nov 2024–Dec 2025", "+19.0%", "+9.7%", "+25.2%", "+31.7%"],
        ],
        [2300, 1900, 1300, 1300, 1100, 1846]
      ),
      para("", { after: 80 }),
      justified("The regime profile is consistent with the central claim of the risk parity literature: the strategy is designed to deliver asymmetric protection, sacrificing upside participation in exchange for reduced downside, and over a complete cycle this asymmetry compounds into superior risk-adjusted performance, which is precisely what Table 3 documents."),

      // 5. Robustness
      h1("5. Robustness"),
      h2("5.1 Gold Ablation"),
      justified("The role of gold in 2024–2025 is a topical question for Chinese multi-asset investors given the elevated price level. We re-estimate the ERC strategy on the four-asset universe excluding the gold ETF; Table 7 reports the resulting performance against the full five-asset specification. Removing gold reduces the annualised return from 6.74% to 4.47%, lowers volatility from 2.56% to 2.25%, deteriorates the Sharpe from 1.77 to 1.01, and worsens the maximum drawdown by 43 bps."),
      tableTitle("Table 7: Gold ablation, ERC (5%) strategy, OOS."),
      makeTable(
        ["Metric", "With gold", "Without gold", "Δ"],
        [
          ["Annualised return", "6.74%", "4.47%", "−2.27%"],
          ["Annualised volatility", "2.56%", "2.25%", "−0.31%"],
          ["Sharpe ratio", "1.77", "1.01", "−0.76"],
          ["Maximum drawdown", "−1.63%", "−2.06%", "−0.43%"],
        ],
        [3000, 2300, 2300, 2346]
      ),
      para("", { after: 80 }),
      justified("Decomposing this effect by regime, gold's contribution is concentrated in the 2024–2025 rally, where its appreciation alongside equities lifted the cumulative path. The conclusion of the ablation is therefore unambiguous: in a multi-asset portfolio constructed under a risk-parity rule, gold remains a materially valuable diversifier through 2025."),

      h2("5.2 Drawdown Profile"),
      justified("Figure 2 displays the OOS drawdown trajectories. ERC and the official CSI MARP exhibit shallow, mean-reverting drawdowns that recover within months; equal weight and MARP replication exhibit deeper, slower recoveries; HRP shows almost no drawdown but at the cost of negligible return. The pattern provides direct visual confirmation that the volatility overlay translates into the desired drawdown control."),
      image(FIG_DD, 470, 192),
      caption("Figure 2: Drawdown trajectories, OOS."),

      h2("5.3 Cost Sensitivity and HRP Diagnostic"),
      justified("Table 8 stress-tests the ERC Sharpe ratio against transaction costs, applied as a per-rebalance turnover charge against the gross-of-cost backtest. Because the engine rebalances annually and the realised OOS window contains only two such rebalances, the absolute cost drag is modest: even at 100 bps one-way, an order of magnitude above realistic friction, the Sharpe falls only to 1.59. At realistic Chinese-ETF friction (5–10 bps one-way), the impact is essentially undetectable."),
      tableTitle("Table 8: Cost sensitivity, ERC (5%)."),
      makeTable(
        ["One-way cost (bps)", "0", "10", "20", "50", "100"],
        [
          ["Annualised return", "6.74%", "6.70%", "6.67%", "6.55%", "6.36%"],
          ["Sharpe", "1.77", "1.76", "1.74", "1.69", "1.59"],
        ],
        [2546, 1500, 1500, 1500, 1500, 1400]
      ),
      para("", { after: 80 }),
      justified("The HRP underperformance noted in Section 3 reflects a known property of Ward-linkage clustering on assets with widely heterogeneous volatilities. With the bond ETF at ≈ 2% volatility and equity ETFs at ≈ 20%, the dendrogram concentrates capital in the low-volatility cluster and the recursive inverse-variance step amplifies this concentration further, leaving a portfolio that is unambiguously safe but has very limited risk-asset participation. Standard remedies in the literature, pre-processing with volatility parity prior to clustering, single or complete linkage, or a minimum-variance constraint at each bisection, are not implemented here in order to preserve the integrity of the locked OOS protocol, but all are natural extensions for future research."),

      // 6. Limitations
      h1("6. Limitations"),
      justified([
        { text: "First, ", italics: true },
        { text: "the size factor is proxied by the spread between the CSI 500 ETF and the CSI 300 ETF, which captures a mid-cap minus large-cap differential rather than a true small-minus-big sort, and which uses ETF returns rather than the underlying index returns; the difference is small in practice (a few basis points of tracking error) but should be acknowledged. The value factor is unavailable from this ETF universe and is omitted entirely. Constructing canonical Fama–French factors for China would require security-level data from CSMAR, Wind, or RESSET. " },
        { text: "Second, ", italics: true },
        { text: "five ETFs cannot fully span the CSI MARP constituent universe, which includes international equities, broader fixed-income exposures, and several commodity sub-classes. The broad commodity ETF (159980) only begins trading in late 2019, which materially limits early-sample diversification and pushes the first valid out-of-sample rebalance to December 2022. " },
        { text: "Third, ", italics: true },
        { text: "the framework is unlevered by construction; when the natural volatility of an inverse-volatility or HRP portfolio falls below the 5% target, the unlevered constraint binds and the target cannot be reached. " },
        { text: "Fourth, ", italics: true },
        { text: "the risk-free rate is held constant at 2.2% per annum; in reality, Chinese deposit rates varied between roughly 1.5% and 3.5% over the sample, and modelling this time variation would refine both the cash leg returns and the Sharpe estimates. " },
        { text: "Fifth, ", italics: true },
        { text: "the headline performance metrics are reported gross of transaction costs; the cost-sensitivity stress test in Section 5.3 is run as a separate post-hoc deduction rather than being integrated into the main backtest engine, although the test indicates that this assumption is not material for the headline conclusion. " },
        { text: "Sixth, ", italics: true },
        { text: "all ETFs in the universe survived to the end of the sample, so survivorship bias is not modelled; this is a minor concern for the established broad-market ETFs used here but would be a serious issue if the universe were extended to thematic or sector products. We also note that an extension comparing this risk-parity construction with hedge-fund factor-replication portfolios (in the spirit of Hasanhodzic & Lo, 2007) is a natural avenue for future work but was not pursued here, as a defensible comparison would require access to actual hedge-fund index return histories that are not available in our data sources." },
      ]),

      // 7. Conclusion
      h1("7. Conclusion"),
      justified("We have shown that a tradable five-ETF approximation of the CSI MARP risk-parity construction, implemented under the Equal Risk Contribution rule with a 5% volatility overlay and evaluated under a strict locked-out-of-sample walk-forward protocol, delivers the diversification properties that risk parity promises. Out-of-sample (Dec 2022 to Dec 2025), the strategy produces a Sharpe ratio of 1.77, an annualised return of 6.74% at a realised volatility of 2.56% (the unlevered constraint binds below the 5% target), and a maximum drawdown of −1.63%, outperforming the equal-weight benchmark on every risk-adjusted metric. The factor regression confirms the strategy is not levered equity exposure (market β = 0.09), the regime decomposition confirms asymmetric protection across the 2023 sideways and 2024 rally regimes, and the gold-ablation test shows the gold leg is a material contributor to risk-adjusted performance, which directly answers the project brief's gold-allocation question in the affirmative through 2025. The full pipeline, from data ingestion to weight optimisation to factor regression, runs end-to-end on five liquid Chinese ETFs and is therefore directly implementable for retail and smaller institutional investors who require multi-asset diversification but cannot access the institutional instruments that underlie the official index."),

      // References
      h1("References"),
      ...[
        "Hasanhodzic, J., & Lo, A. W. (2007). Can hedge-fund returns be replicated? The linear case. Journal of Investment Management, 5(2), 5–45.",
        "López de Prado, M. (2016). Building diversified portfolios that outperform out of sample. Journal of Portfolio Management, 42(4), 59–69.",
        "Maillard, S., Roncalli, T., & Teïletche, J. (2010). The properties of equally weighted risk contribution portfolios. Journal of Portfolio Management, 36(4), 60–70.",
        "Roncalli, T. (2013). Introduction to Risk Parity and Budgeting. Chapman and Hall/CRC.",
        "China Securities Index Co., Ltd. (2017). CSI Multi-Asset Risk Parity Index (930929), Index Methodology. csindex.com.cn.",
        "China Securities Index Co., Ltd. CSI MARP 930929 and CSI 300 daily index levels, 2017–2025. csindex.com.cn.",
        "East Money. CSI 300 index levels and open-ended fund net asset values, 2017–2025. eastmoney.com.",
        "Yahoo Finance. Adjusted close prices for ETFs 510300.SH, 510500.SH, 511010.SH, 518880.SH, and 159980.SZ, 2017–2025. finance.yahoo.com.",
      ].map(item => new Paragraph({
        spacing: { before: 0, after: 60, line: 256 },
        indent: { left: 320, hanging: 320 },
        alignment: AlignmentType.JUSTIFIED,
        children: [new TextRun({ text: item, font: FONT, size: BODY })],
      })),

      // Final note
      new Paragraph({
        spacing: { before: 160, after: 0 },
        border: { top: { style: BorderStyle.SINGLE, size: 2, color: BORDER_COLOR, space: 4 } },
        children: [],
      }),
      para("This report and all underlying code, data, and figures are original work produced for the Fudan University Alternative Investment Strategies course. All external data sources are attributed in the references.",
        { size: SMALL, italics: true, align: AlignmentType.CENTER, color: "555555" }),
    ],
  }],
});

// Write
Packer.toBuffer(doc).then(buf => {
  const outPath = path.join(REPORTS_DIR, "report.docx");
  fs.writeFileSync(outPath, buf);
  console.log("Written to " + outPath);
  console.log("Size: " + (buf.length / 1024).toFixed(1) + " KB");
});
