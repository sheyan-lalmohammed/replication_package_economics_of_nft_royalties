# make_paper_match_v2_plots.py
#
# Plotting companion for estimate_paper_match_v2.py
#
# Usage:
#   python make_paper_match_v2_plots.py paper_match_estimates_v2/params.csv paper_match_estimates_v2/fitted_markets.csv [outdir]
#
# This script creates:
#   - binned fit plots for share gap vs alpha_c
#   - binned fit plots for share gap vs kappa_diff
#   - binned fit plots for share gap vs tau_diff
#   - binned fit plots for switching vs tau_diff
#   - residual plots for all the above
#   - counts plots for all the above
#   - observed vs fitted scatter plots
#   - histograms of observed and fitted shares/switching
#   - subgroup bar charts: mint/non-mint, high/low price, tau parity, kappa parity
#   - optional local-fit scatter plots against tau_diff and kappa_diff
#   - parameter summary text file
#   - CSV exports of the binned moments used for plotting

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


plt.style.use('seaborn-v0_8-colorblind')


# Must match estimation code
ALPHA_BINS = np.array([0, 1, 2.5, 5, 7.5, 10, 15], dtype=float)
KAPPA_BINS = np.array([-1.0, -0.75, -0.5, -0.25, -0.10, 0.0, 0.10, 0.25, 0.5, 0.75, 1.0], dtype=float)
TAU_DIFF_BINS = np.array([-7.5, -5, -2.5, -1, -0.5, 0, 0.5, 1, 2.5, 5, 7.5], dtype=float)

TAU_PARITY_TOL = 0.50
KAPPA_PARITY_TOL = 0.05


# ----------------------------
# helpers
# ----------------------------
def _safe_mkdir(d):
    os.makedirs(d, exist_ok=True)

def _bin_centers(df):
    return 0.5 * (df["bin_lo"].to_numpy() + df["bin_hi"].to_numpy())

def _binned_means(x, y, bins):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    bin_id = np.digitize(x, bins) - 1
    K = len(bins) - 1

    rows = []
    for k in range(K):
        m = (bin_id == k) & np.isfinite(x) & np.isfinite(y)
        rows.append({
            "bin_lo": float(bins[k]),
            "bin_hi": float(bins[k + 1]),
            "count": int(m.sum()),
            "mean": float(np.mean(y[m])) if m.sum() > 0 else np.nan,
            "sd": float(np.std(y[m])) if m.sum() > 1 else np.nan,
        })
    return pd.DataFrame(rows)

def _merge_binned(data_df, model_df):
    out = data_df[["bin_lo", "bin_hi", "count", "mean", "sd"]].copy()
    out = out.rename(columns={"mean": "data_mean", "sd": "data_sd"})
    out["model_mean"] = model_df["mean"].to_numpy()
    out["model_sd"] = model_df["sd"].to_numpy()
    out["resid"] = out["model_mean"] - out["data_mean"]
    return out

def _subset_mean(arr, mask):
    arr = np.asarray(arr, dtype=float)
    mask = np.asarray(mask, dtype=bool)
    m = mask & np.isfinite(arr)
    if m.sum() == 0:
        return np.nan
    return float(np.mean(arr[m]))

def _subset_sd(arr, mask):
    arr = np.asarray(arr, dtype=float)
    mask = np.asarray(mask, dtype=bool)
    m = mask & np.isfinite(arr)
    if m.sum() <= 1:
        return np.nan
    return float(np.std(arr[m]))

def _safe_corr(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]
    y = y[m]
    if len(x) <= 2:
        return np.nan
    sx = np.std(x)
    sy = np.std(y)
    if sx < 1e-12 or sy < 1e-12:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


# ----------------------------
# data prep for plots
# ----------------------------
def prep_df(df):
    df = df.copy()

    needed = [
        "alpha_c", "kappa_diff", "tau_diff",
        "sB", "sO", "share_B_hat",
        "switch_obs", "switch_hat",
        "B_is_mint", "hi_price",
    ]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"fitted_markets.csv is missing required columns: {missing}")

    df["share_gap_data"] = df["sB"] - df["sO"]
    df["share_gap_model"] = 2.0 * df["share_B_hat"] - 1.0

    df["tau_parity"] = (np.abs(df["tau_diff"]) <= TAU_PARITY_TOL).astype(int)
    df["kappa_parity"] = (np.abs(df["kappa_diff"]) <= KAPPA_PARITY_TOL).astype(int)

    return df


# ----------------------------
# build binned tables
# ----------------------------
def build_binned_tables(df):
    out = {}

    # share gap vs alpha_c
    out["alpha_share"] = _merge_binned(
        _binned_means(df["alpha_c"], df["share_gap_data"], ALPHA_BINS),
        _binned_means(df["alpha_c"], df["share_gap_model"], ALPHA_BINS),
    )

    # share gap vs kappa_diff
    out["kappa_share"] = _merge_binned(
        _binned_means(df["kappa_diff"], df["share_gap_data"], KAPPA_BINS),
        _binned_means(df["kappa_diff"], df["share_gap_model"], KAPPA_BINS),
    )

    # share gap vs tau_diff
    out["tau_share"] = _merge_binned(
        _binned_means(df["tau_diff"], df["share_gap_data"], TAU_DIFF_BINS),
        _binned_means(df["tau_diff"], df["share_gap_model"], TAU_DIFF_BINS),
    )

    # switching vs tau_diff
    out["tau_switch"] = _merge_binned(
        _binned_means(df["tau_diff"], df["switch_obs"], TAU_DIFF_BINS),
        _binned_means(df["tau_diff"], df["switch_hat"], TAU_DIFF_BINS),
    )

    return out


# ----------------------------
# subgroup tables
# ----------------------------
def build_subgroup_table(df):
    mint = df["B_is_mint"].astype(bool).to_numpy()
    nonmint = ~mint
    hi_price = df["hi_price"].astype(bool).to_numpy()
    lo_price = ~hi_price
    tau_parity = df["tau_parity"].astype(bool).to_numpy()
    non_tau_parity = ~tau_parity
    kappa_parity = df["kappa_parity"].astype(bool).to_numpy()
    non_kappa_parity = ~kappa_parity

    groups = {
        "mint": mint,
        "nonmint": nonmint,
        "hi_price": hi_price,
        "lo_price": lo_price,
        "tau_parity": tau_parity,
        "non_tau_parity": non_tau_parity,
        "kappa_parity": kappa_parity,
        "non_kappa_parity": non_kappa_parity,
    }

    rows = []
    for name, mask in groups.items():
        rows.append({
            "group": name,
            "n": int(np.sum(mask)),
            "share_data": _subset_mean(df["sB"], mask),
            "share_model": _subset_mean(df["share_B_hat"], mask),
            "share_gap_data": _subset_mean(df["share_gap_data"], mask),
            "share_gap_model": _subset_mean(df["share_gap_model"], mask),
            "switch_data": _subset_mean(df["switch_obs"], mask),
            "switch_model": _subset_mean(df["switch_hat"], mask),
            "share_data_sd": _subset_sd(df["sB"], mask),
            "share_model_sd": _subset_sd(df["share_B_hat"], mask),
            "switch_data_sd": _subset_sd(df["switch_obs"], mask),
            "switch_model_sd": _subset_sd(df["switch_hat"], mask),
        })

    out = pd.DataFrame(rows)
    out["share_resid"] = out["share_model"] - out["share_data"]
    out["share_gap_resid"] = out["share_gap_model"] - out["share_gap_data"]
    out["switch_resid"] = out["switch_model"] - out["switch_data"]
    return out


# ----------------------------
# plotting functions
# ----------------------------
def plot_binned_comparison(df, title, xlabel, ylabel, outpath):
    x = _bin_centers(df)
    y_data = df["data_mean"].to_numpy()
    y_model = df["model_mean"].to_numpy()

    plt.figure()
    plt.plot(x, y_data, marker="o", linestyle="-", label="Data")
    plt.plot(x, y_model, marker="o", linestyle="--", label="Model")
    plt.axhline(0.0, linewidth=1)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()

def plot_binned_residuals(df, title, xlabel, outpath):
    x = _bin_centers(df)
    resid = df["resid"].to_numpy()

    plt.figure()
    plt.plot(x, resid, marker="o")
    plt.axhline(0.0, linewidth=1)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Model - Data")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()

def plot_binned_counts(df, title, xlabel, outpath):
    x = _bin_centers(df)
    widths = (df["bin_hi"] - df["bin_lo"]).to_numpy()
    counts = df["count"].to_numpy()

    plt.figure()
    plt.bar(x, counts, width=widths)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()

def plot_scatter_fit(obs, fit, title, xlabel, ylabel, outpath):
    obs = np.asarray(obs, dtype=float)
    fit = np.asarray(fit, dtype=float)
    m = np.isfinite(obs) & np.isfinite(fit)
    obs = obs[m]
    fit = fit[m]

    lo = min(obs.min(), fit.min()) if len(obs) else 0.0
    hi = max(obs.max(), fit.max()) if len(obs) else 1.0
    corr = _safe_corr(obs, fit)

    plt.figure()
    plt.scatter(obs, fit, alpha=0.5)
    plt.plot([lo, hi], [lo, hi], linestyle="--")
    if np.isfinite(corr):
        plt.title(f"{title}\nCorr = {corr:.3f}")
    else:
        plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()

def plot_hist_compare(data_arr, model_arr, title, xlabel, outpath):
    data_arr = np.asarray(data_arr, dtype=float)
    model_arr = np.asarray(model_arr, dtype=float)

    m1 = np.isfinite(data_arr)
    m2 = np.isfinite(model_arr)

    plt.figure()
    plt.hist(data_arr[m1], bins=30, alpha=0.6, label="Data")
    plt.hist(model_arr[m2], bins=30, alpha=0.6, label="Model")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()

def plot_series_against_driver(x, y_data, y_model, title, xlabel, ylabel, outpath, sort_x=False):
    x = np.asarray(x, dtype=float)
    y_data = np.asarray(y_data, dtype=float)
    y_model = np.asarray(y_model, dtype=float)

    m = np.isfinite(x) & np.isfinite(y_data) & np.isfinite(y_model)
    x = x[m]
    y_data = y_data[m]
    y_model = y_model[m]

    if sort_x and len(x) > 0:
        idx = np.argsort(x)
        x = x[idx]
        y_data = y_data[idx]
        y_model = y_model[idx]

    plt.figure()
    plt.scatter(x, y_data, alpha=0.25, label="Data")
    plt.scatter(x, y_model, alpha=0.25, label="Model")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()

def plot_subgroup_bars(df_sub, value_col_data, value_col_model, title, ylabel, outpath):
    groups = df_sub["group"].tolist()
    x = np.arange(len(groups))
    width = 0.38

    y_data = df_sub[value_col_data].to_numpy()
    y_model = df_sub[value_col_model].to_numpy()

    plt.figure(figsize=(10, 5))
    plt.bar(x - width / 2, y_data, width=width, label="Data")
    plt.bar(x + width / 2, y_model, width=width, label="Model")
    plt.axhline(0.0, linewidth=1)
    plt.xticks(x, groups, rotation=30, ha="right")
    plt.title(title)
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()

def plot_subgroup_residuals(df_sub, resid_col, title, ylabel, outpath):
    groups = df_sub["group"].tolist()
    x = np.arange(len(groups))
    y = df_sub[resid_col].to_numpy()

    plt.figure(figsize=(10, 5))
    plt.bar(x, y)
    plt.axhline(0.0, linewidth=1)
    plt.xticks(x, groups, rotation=30, ha="right")
    plt.title(title)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()

def plot_params_bar(params, outpath):
    keys = [k for k in ["delta_B", "lambda_S", "phi_K", "k0", "k1", "gamma_S"] if k in params]
    vals = [float(params[k]) for k in keys]

    plt.figure(figsize=(8, 4))
    plt.bar(np.arange(len(keys)), vals)
    plt.axhline(0.0, linewidth=1)
    plt.xticks(np.arange(len(keys)), keys, rotation=30, ha="right")
    plt.title("Estimated parameters")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


# ----------------------------
# summary text
# ----------------------------
def write_summary(params, df, subgroup_df, outpath):
    with open(outpath, "w") as f:
        f.write("PARAMETERS\n")
        for k in sorted(params.keys()):
            f.write(f"{k}: {params[k]}\n")

        f.write("\nFIT SUMMARY\n")
        corr_share = _safe_corr(df["sB"], df["share_B_hat"])
        corr_switch = _safe_corr(df["switch_obs"], df["switch_hat"])

        mae_share = float(np.nanmean(np.abs(df["share_B_hat"] - df["sB"])))
        mae_switch = float(np.nanmean(np.abs(df["switch_hat"] - df["switch_obs"])))
        mae_gap = float(np.nanmean(np.abs(df["share_gap_model"] - df["share_gap_data"])))

        f.write(f"corr(observed share, fitted share): {corr_share}\n")
        f.write(f"corr(observed switch, fitted switch): {corr_switch}\n")
        f.write(f"MAE share: {mae_share}\n")
        f.write(f"MAE switch: {mae_switch}\n")
        f.write(f"MAE share gap: {mae_gap}\n")

        f.write("\nSUBGROUP TABLE\n")
        f.write(subgroup_df.to_string(index=False))
        f.write("\n")


# ----------------------------
# main
# ----------------------------
def main(params_csv, fitted_markets_csv, outdir="paper_match_v2_plots"):
    _safe_mkdir(outdir)

    params = pd.read_csv(params_csv).iloc[0].to_dict()
    df = pd.read_csv(fitted_markets_csv)
    df = prep_df(df)

    # tables
    binned = build_binned_tables(df)
    subgroup_df = build_subgroup_table(df)

    # save binned tables
    binned["alpha_share"].to_csv(os.path.join(outdir, "binned_share_gap_alpha.csv"), index=False)
    binned["kappa_share"].to_csv(os.path.join(outdir, "binned_share_gap_kappa_diff.csv"), index=False)
    binned["tau_share"].to_csv(os.path.join(outdir, "binned_share_gap_tau_diff.csv"), index=False)
    binned["tau_switch"].to_csv(os.path.join(outdir, "binned_switch_tau_diff.csv"), index=False)
    subgroup_df.to_csv(os.path.join(outdir, "subgroup_moments.csv"), index=False)

    # parameter plot
    plot_params_bar(params, os.path.join(outdir, "params_bar.png"))

    # binned plots: share gap vs alpha
    plot_binned_comparison(
        binned["alpha_share"],
        "Share gap vs collection royalty rate",
        "alpha_c bin center",
        "Mean share gap (Blur - OpenSea)",
        os.path.join(outdir, "share_gap_alpha.png"),
    )
    plot_binned_residuals(
        binned["alpha_share"],
        "Residuals: share gap vs collection royalty rate",
        "alpha_c bin center",
        os.path.join(outdir, "share_gap_alpha_resid.png"),
    )
    plot_binned_counts(
        binned["alpha_share"],
        "Counts by collection royalty-rate bin",
        "alpha_c bin center",
        os.path.join(outdir, "share_gap_alpha_counts.png"),
    )

    # binned plots: share gap vs kappa_diff
    plot_binned_comparison(
        binned["kappa_share"],
        "Share gap vs enforcement-credibility difference",
        "kappa_B - kappa_O bin center",
        "Mean share gap (Blur - OpenSea)",
        os.path.join(outdir, "share_gap_kappa_diff.png"),
    )
    plot_binned_residuals(
        binned["kappa_share"],
        "Residuals: share gap vs enforcement-credibility difference",
        "kappa_B - kappa_O bin center",
        os.path.join(outdir, "share_gap_kappa_diff_resid.png"),
    )
    plot_binned_counts(
        binned["kappa_share"],
        "Counts by enforcement-credibility-difference bin",
        "kappa_B - kappa_O bin center",
        os.path.join(outdir, "share_gap_kappa_diff_counts.png"),
    )

    # binned plots: share gap vs tau_diff
    plot_binned_comparison(
        binned["tau_share"],
        "Share gap vs effective cost difference",
        "tau_B - tau_O bin center",
        "Mean share gap (Blur - OpenSea)",
        os.path.join(outdir, "share_gap_tau_diff.png"),
    )
    plot_binned_residuals(
        binned["tau_share"],
        "Residuals: share gap vs effective cost difference",
        "tau_B - tau_O bin center",
        os.path.join(outdir, "share_gap_tau_diff_resid.png"),
    )
    plot_binned_counts(
        binned["tau_share"],
        "Counts by effective cost-difference bin",
        "tau_B - tau_O bin center",
        os.path.join(outdir, "share_gap_tau_diff_counts.png"),
    )

    # binned plots: switching vs tau_diff
    plot_binned_comparison(
        binned["tau_switch"],
        "Switching vs effective cost difference",
        "tau_B - tau_O bin center",
        "Mean off-mint switching share",
        os.path.join(outdir, "switch_tau_diff.png"),
    )
    plot_binned_residuals(
        binned["tau_switch"],
        "Residuals: switching vs effective cost difference",
        "tau_B - tau_O bin center",
        os.path.join(outdir, "switch_tau_diff_resid.png"),
    )
    plot_binned_counts(
        binned["tau_switch"],
        "Counts by effective cost-difference bin",
        "tau_B - tau_O bin center",
        os.path.join(outdir, "switch_tau_diff_counts.png"),
    )

    # observed vs fitted scatters
    plot_scatter_fit(
        df["sB"],
        df["share_B_hat"],
        "Observed vs fitted Blur share",
        "Observed Blur share",
        "Fitted Blur share",
        os.path.join(outdir, "scatter_share_fit.png"),
    )
    plot_scatter_fit(
        df["switch_obs"],
        df["switch_hat"],
        "Observed vs fitted switching",
        "Observed switching share",
        "Fitted switching share",
        os.path.join(outdir, "scatter_switch_fit.png"),
    )
    plot_scatter_fit(
        df["share_gap_data"],
        df["share_gap_model"],
        "Observed vs fitted share gap",
        "Observed share gap",
        "Fitted share gap",
        os.path.join(outdir, "scatter_share_gap_fit.png"),
    )

    # histograms
    plot_hist_compare(
        df["sB"],
        df["share_B_hat"],
        "Distribution: observed vs fitted Blur share",
        "Blur share",
        os.path.join(outdir, "hist_share_fit.png"),
    )
    plot_hist_compare(
        df["switch_obs"],
        df["switch_hat"],
        "Distribution: observed vs fitted switching",
        "Switching share",
        os.path.join(outdir, "hist_switch_fit.png"),
    )
    plot_hist_compare(
        df["share_gap_data"],
        df["share_gap_model"],
        "Distribution: observed vs fitted share gap",
        "Share gap",
        os.path.join(outdir, "hist_share_gap_fit.png"),
    )

    # raw scatter clouds vs drivers
    plot_series_against_driver(
        df["tau_diff"],
        df["share_gap_data"],
        df["share_gap_model"],
        "Raw fit cloud: share gap vs tau_diff",
        "tau_B - tau_O",
        "Share gap",
        os.path.join(outdir, "cloud_share_gap_tau_diff.png"),
    )
    plot_series_against_driver(
        df["tau_diff"],
        df["switch_obs"],
        df["switch_hat"],
        "Raw fit cloud: switching vs tau_diff",
        "tau_B - tau_O",
        "Switching share",
        os.path.join(outdir, "cloud_switch_tau_diff.png"),
    )
    plot_series_against_driver(
        df["kappa_diff"],
        df["share_gap_data"],
        df["share_gap_model"],
        "Raw fit cloud: share gap vs kappa_diff",
        "kappa_B - kappa_O",
        "Share gap",
        os.path.join(outdir, "cloud_share_gap_kappa_diff.png"),
    )
    plot_series_against_driver(
        df["alpha_c"],
        df["share_gap_data"],
        df["share_gap_model"],
        "Raw fit cloud: share gap vs alpha_c",
        "alpha_c",
        "Share gap",
        os.path.join(outdir, "cloud_share_gap_alpha.png"),
    )

    # subgroup bar charts
    plot_subgroup_bars(
        subgroup_df,
        "share_data",
        "share_model",
        "Blur share by subgroup",
        "Mean Blur share",
        os.path.join(outdir, "subgroup_share_bars.png"),
    )
    plot_subgroup_bars(
        subgroup_df,
        "share_gap_data",
        "share_gap_model",
        "Share gap by subgroup",
        "Mean share gap",
        os.path.join(outdir, "subgroup_share_gap_bars.png"),
    )
    plot_subgroup_bars(
        subgroup_df,
        "switch_data",
        "switch_model",
        "Switching by subgroup",
        "Mean switching share",
        os.path.join(outdir, "subgroup_switch_bars.png"),
    )

    # subgroup residuals
    plot_subgroup_residuals(
        subgroup_df,
        "share_resid",
        "Residuals: Blur share by subgroup",
        "Model - Data",
        os.path.join(outdir, "subgroup_share_resid.png"),
    )
    plot_subgroup_residuals(
        subgroup_df,
        "share_gap_resid",
        "Residuals: share gap by subgroup",
        "Model - Data",
        os.path.join(outdir, "subgroup_share_gap_resid.png"),
    )
    plot_subgroup_residuals(
        subgroup_df,
        "switch_resid",
        "Residuals: switching by subgroup",
        "Model - Data",
        os.path.join(outdir, "subgroup_switch_resid.png"),
    )

    # parity-only focused scatters
    tau_parity_mask = df["tau_parity"].astype(bool)
    kappa_parity_mask = df["kappa_parity"].astype(bool)

    plot_scatter_fit(
        df.loc[tau_parity_mask, "sB"],
        df.loc[tau_parity_mask, "share_B_hat"],
        "Observed vs fitted Blur share (tau-parity subset)",
        "Observed Blur share",
        "Fitted Blur share",
        os.path.join(outdir, "scatter_share_fit_tau_parity.png"),
    )
    plot_scatter_fit(
        df.loc[kappa_parity_mask, "sB"],
        df.loc[kappa_parity_mask, "share_B_hat"],
        "Observed vs fitted Blur share (kappa-parity subset)",
        "Observed Blur share",
        "Fitted Blur share",
        os.path.join(outdir, "scatter_share_fit_kappa_parity.png"),
    )
    plot_scatter_fit(
        df.loc[tau_parity_mask, "switch_obs"],
        df.loc[tau_parity_mask, "switch_hat"],
        "Observed vs fitted switching (tau-parity subset)",
        "Observed switching share",
        "Fitted switching share",
        os.path.join(outdir, "scatter_switch_fit_tau_parity.png"),
    )

    # summary text
    write_summary(params, df, subgroup_df, os.path.join(outdir, "summary.txt"))

    print(f"Wrote plots to: {outdir}/")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python make_paper_match_v2_plots.py params.csv fitted_markets.csv [outdir]")
        sys.exit(1)

    outdir = sys.argv[3] if len(sys.argv) >= 4 else "paper_match_v2_plots"
    main(sys.argv[1], sys.argv[2], outdir=outdir)