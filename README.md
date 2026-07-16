# Replication Package

**The Economics of Optional Royalties: Platform Competition, Enforcement, and Routing in NFT Markets**

This package reproduces every empirical result in Section 5 ("Empirical Model") and
Appendix B ("Empirical Results") of the paper, starting from the original Dune Analytics
trade-level data.

---

## 1. Contents

```
replication_package/
├── README.md                     <- this file
├── paper/                        <- the paper PDF
├── code/                         <- the 7 scripts that generate all empirical results
├── data/
│   ├── raw/                      <- original (un-cleaned) Dune Analytics data
│   └── clean/                    <- estimation dataset + cleaning metadata
└── results/                      <- pre-generated outputs (tables & figures)
```

---

## 2. Software environment

The code was run with:

| Package     | Version |
|-------------|---------|
| Python      | 3.14    |
| pandas      | 3.0.2   |
| numpy       | 2.4.4   |
| scipy       | 1.17.1  |
| matplotlib  | 3.10.8  |

Any recent versions of these packages should reproduce the results. Estimation and the
bootstrap use fixed random seeds, so results are deterministic.

---

## 3. Data

### `data/raw/MSM_data_clean_week2023-04-03.csv`
The original trade-level panel pulled from Dune Analytics. It aggregates all Ethereum
secondary NFT trades from 2022–2024 in which **both** the executing platform and the
collection's mint platform are OpenSea or Blur, collapsed to the
**platform × collection × week** level (43,690 data rows). Columns include weekly trades,
secondary volume (ETH), average trade price, stated royalty rate, platform fee rate,
royalties paid, the constructed **enforcement credibility** measure `κ`, the effective
total cost rate `τ`, volume/trade shares, the mint-platform proxy, and switching shares.
These are the "Constructed Measures" of Section 5.2.

### `data/clean/MSM_data_clean_week2023-04-03_medium_paired.csv`
The cleaned **estimation dataset** (Section 5.3, "Data Cleaning"). This is the
**Medium-Cleaned Paired Dataset** described in the paper: **14,648 rows = 7,324 paired
collection–week markets** (each market contributes exactly one Blur row and one OpenSea
row), spanning 826 collections over 92 weeks. `..._cleaning_metadata.json` records the
exact cleaning thresholds and the winsorization bounds; `..._cleaning_summary.csv` is a
short audit of rows kept/dropped.

> The paper uses **only** this "medium" cleaning. A stricter "strong" preset
> (4,287 markets) was explored for robustness but is **not** used for any result in the
> paper; it can be regenerated at any time with `--preset strong` (Step 0 below).

---

## 4. How to reproduce (run order)

Run every command **from the `replication_package/` root directory**. Each script takes
its input file(s) as command-line arguments and writes its outputs to a directory in the
current working directory (the directory names are noted below and match the pre-generated
copies in `results/`).

Replace `python` with the interpreter that has the packages from Section 2 (e.g. the
project virtual environment: `../.venv/bin/python`).

### Step 0 (optional) — Regenerate the clean dataset from raw data
**Paper: §5.1 (Empirical Data and Simplification), §5.2 (Constructed Measures), §5.3 (Data Cleaning).**
Reproduces `data/clean/...` from `data/raw/...`. **Not required** — the cleaned dataset is
already provided.

```bash
python code/preprocess_msm_data_paired.py \
    data/raw/MSM_data_clean_week2023-04-03.csv \
    data/clean/MSM_data_clean_week2023-04-03_medium_paired.csv \
    --preset medium
```

### Step 1 — Summary statistics and data figures  →  **Table 1, Figure 7**
**Paper: Table 1 in §5.3 (Data Cleaning); Figure 7 in Appendix §B.1 (Summary of NFT Data).**
```bash
python code/make_data_summary_and_figures.py \
    data/clean/MSM_data_clean_week2023-04-03_medium_paired.csv
```
Writes `data_summary_outputs/`:
- `summary_statistics.csv` / `summary_statistics.tex` → **Table 1** (Summary Statistics for the Medium-Cleaned Paired Dataset)
- `weekly_*.png` (7 plots) → **Figure 7** (Summary Plots of Post-Cleaning Data on a Weekly Basis)

### Step 2 — Structural estimation  →  **Table 2 (point estimates)**
**Paper: model in §5.4 (Empirical Structural Model), moments in §5.5 (Moments), estimator in §5.6 (Estimation); results reported in §5.7 (Estimation Results and Consistency).**
```bash
python code/paper_matching_estimator.py \
    data/clean/MSM_data_clean_week2023-04-03_medium_paired.csv
```
Writes `paper_match_estimates_v2/`:
- `params.csv` → **Table 2** point estimates: `δ_B=0.9166, λ=0.1934, φ_K=3.3751, k0=1.6105, k1=5.0, γ=0.8050`
- `fitted_markets.csv` → model-implied shares/switching for every market (input to Steps 4 & 5)
- `moment_comparison.csv` → data vs. model moments (Section 5.5)

### Step 3 — Bootstrap confidence intervals  →  **Table 2 (CI columns)**
**Paper: §5.7 (Estimation Results and Consistency) — the Boot Mean / Boot SD / percentile columns of Table 2.**
```bash
python code/bootstrap_v2_msm.py \
    data/clean/MSM_data_clean_week2023-04-03_medium_paired.csv \
    --n_boot 100 --seed 123 --bootstrap_unit collection
```
Writes `paper_match_estimates_v2_bootstrap_exact/`:
- `bootstrap_confidence_intervals.csv` → **Table 2** Boot Mean / Boot SD / 2.5% / 50% / 97.5% columns
- `bootstrap_draws.csv`, `bootstrap_metadata.json`, `baseline_params.csv`

> Note: this re-estimates the model on 100 resamples and is the slowest step.

### Step 4 — Estimation-fit plots  →  **Figures 1–6, 8, 9**
**Paper: Figures 1–6 in §5.7 (Estimation Results and Consistency); Figures 8–9 in Appendix §B.2 (Additional Results from Estimation).**
```bash
python code/make_paper_match_plots.py \
    paper_match_estimates_v2/params.csv \
    paper_match_estimates_v2/fitted_markets.csv
```
Writes `paper_match_v2_plots/`. Figure mapping:
- **Figure 1** (Distribution of Blur Share and Switching Share) → `hist_share_fit.png`, `hist_switch_fit.png`, `hist_share_gap_fit.png`
- **Figure 2** (Share Gap vs. Collection Royalty Rate) → `share_gap_alpha.png` (+ `_counts`)
- **Figure 3** (Share Gap vs. Δκ) → `share_gap_kappa_diff.png` (+ `_counts`)
- **Figure 4** (Share Gap vs. Δτ) → `share_gap_tau_diff.png` (+ `_counts`)
- **Figure 5** (Switching Share vs. Δτ) → `switch_tau_diff.png` (+ `_counts`)
- **Figure 6** (Subgroup Moments) → `subgroup_share_bars.png`, `subgroup_share_gap_bars.png`, `subgroup_switch_bars.png`
- **Figure 8** (Scatter Plots of Fit) → `scatter_*.png`
- **Figure 9** (Residuals after Estimation) → `*_resid.png`
- Also `params_bar.png` (visual of Table 2) and the binned-moment CSVs.

### Step 5 — Counterfactual analysis  →  **Tables 3, 4, 5, 6, 7, 8, 9**
**Paper: §5.8 (Counterfactual Analysis) — Table 3 in §5.8.1 (Aggregate Effects); Tables 4–5 in §5.8.2 (Group-Specific Counterfactual Effects); Tables 6–9 in Appendix §B.3 (Additional Results from Counterfactual Analysis).**
```bash
python code/run_v2_counterfactuals.py \
    paper_match_estimates_v2/params.csv \
    paper_match_estimates_v2/fitted_markets.csv
```
Writes `v2_counterfactuals_structured_extended_out/`. Table mapping:
- `summary_counterfactuals.csv` → **Table 3** (Aggregate Counterfactual Effects) and **Table 8** (Subgroup Counterfactual Effects)
- `structured_enforcement_counterfactuals.csv` → **Table 4** (Structured Enforcement Counterfactuals) and **Table 7**
- `structured_switching_counterfactuals.csv` → **Table 5** (Structured Switching Counterfactuals) and **Table 6**
- `alpha_bin_counterfactuals.csv` → **Table 9** (Counterfactual Outcomes by Royalty Bin)
- `subgroup_counterfactuals.csv`, `market_level_counterfactuals.csv` → supporting detail

The five counterfactual scenarios in Section 5.8 (OpenSea matches Blur enforcement; Full
enforcement on both; Remove switching frictions; Blur matches OpenSea enforcement; Zero
enforcement preference) plus two additional diagnostics (`zero_price_dependent_switching`,
`targeted_full_enforcement_high_royalty`) are all produced by this script.

### Step 6 (optional) — Counterfactual figures
Not shown as numbered figures in the paper, but provided for reference.
```bash
python code/plot_counterfactual_effects.py v2_counterfactuals_structured_extended_out
```
Writes plots into `v2_counterfactuals_structured_extended_out/counterfactual_figures/`.

---

## 5. Paper artifact → file index

| Paper artifact | Paper section | Script (Step) | Output file |
|---|---|---|---|
| Table 1 (summary stats) | §5.3 | `make_data_summary_and_figures.py` (1) | `results/data_summary_outputs/summary_statistics.csv` |
| Table 2 (estimates) | §5.7 (model §5.4–5.6) | `paper_matching_estimator.py` (2) | `results/paper_match_estimates_v2/params.csv` |
| Table 2 (bootstrap CIs) | §5.7 | `bootstrap_v2_msm.py` (3) | `results/paper_match_estimates_v2_bootstrap_exact/bootstrap_confidence_intervals.csv` |
| Table 3 (aggregate CF) | §5.8.1 | `run_v2_counterfactuals.py` (5) | `results/v2_counterfactuals_structured_extended_out/summary_counterfactuals.csv` |
| Table 8 (subgroup CF) | §B.3 | `run_v2_counterfactuals.py` (5) | `.../summary_counterfactuals.csv` |
| Table 4 (structured enforcement CF) | §5.8.2 | `run_v2_counterfactuals.py` (5) | `.../structured_enforcement_counterfactuals.csv` |
| Table 7 (structured enforcement CF, full) | §B.3 | `run_v2_counterfactuals.py` (5) | `.../structured_enforcement_counterfactuals.csv` |
| Table 5 (structured switching CF) | §5.8.2 | `run_v2_counterfactuals.py` (5) | `.../structured_switching_counterfactuals.csv` |
| Table 6 (structured switching CF, full) | §B.3 | `run_v2_counterfactuals.py` (5) | `.../structured_switching_counterfactuals.csv` |
| Table 9 (CF by royalty bin) | §B.3 | `run_v2_counterfactuals.py` (5) | `.../alpha_bin_counterfactuals.csv` |
| Figures 1–6 (estimation fit) | §5.7 | `make_paper_match_plots.py` (4) | `results/paper_match_v2_plots/*.png` |
| Figure 7 (data summary) | §B.1 | `make_data_summary_and_figures.py` (1) | `results/data_summary_outputs/weekly_*.png` |
| Figures 8–9 (scatter fit, residuals) | §B.2 | `make_paper_match_plots.py` (4) | `results/paper_match_v2_plots/scatter_*.png`, `*_resid.png` |

The pre-generated copies in `results/` were produced by exactly the commands above and
match the paper. (Re-running a step writes a fresh copy of that directory into the package
root, which can be diffed against the corresponding `results/` copy.)
