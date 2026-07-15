# make_data_summary_and_figures.py
#
# Usage:
#   python make_data_summary_and_figures.py MSM_data_clean_week2023-04-03_medium_paired.csv
#
# Outputs:
#   data_summary_outputs/
#       summary_statistics.csv
#       summary_statistics.tex
#       weekly_total_volume_by_platform.png
#       weekly_total_trades_by_platform.png
#       weekly_mean_share_volume_by_platform.png
#       weekly_mean_enforcement_by_platform.png
#       weekly_mean_effective_cost_by_platform.png
#       weekly_switching_share.png
#       weekly_mean_trade_price_by_platform.png

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.style.use('seaborn-v0_8-colorblind')


OUTDIR = "data_summary_outputs"

# Variables to summarize in the thesis table
SUMMARY_VARS = [
    "trades",
    "total_secondary_volume_eth",
    "avg_trade_price_eth",
    "royalty_rate_pct",
    "platform_fee_rate_pct",
    "enforcement_credibility",
    "effective_total_cost_rate_pct",
    "share_volume",
    "share_trades",
    "switch_share_volume_ct",
    "switch_share_trades_ct",
    "ct_trades_total",
    "ct_volume_total_eth",
]

# Nice labels for LaTeX table
VAR_LABELS = {
    "trades": "Platform-week trades",
    "total_secondary_volume_eth": "Platform-week secondary volume (ETH)",
    "avg_trade_price_eth": "Average trade price (ETH)",
    "royalty_rate_pct": "Royalty rate (pct)",
    "platform_fee_rate_pct": "Platform fee rate (pct)",
    "enforcement_credibility": "Enforcement credibility",
    "effective_total_cost_rate_pct": "Effective total cost rate (pct)",
    "share_volume": "Volume share",
    "share_trades": "Trade share",
    "switch_share_volume_ct": "Switching share (volume)",
    "switch_share_trades_ct": "Switching share (trades)",
    "ct_trades_total": "Collection-week trades",
    "ct_volume_total_eth": "Collection-week volume (ETH)",
}


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def read_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["week"] = pd.to_datetime(df["week"], utc=True, errors="coerce")
    df = df.dropna(subset=["week"]).copy()
    return df


def make_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for var in SUMMARY_VARS:
        s = pd.to_numeric(df[var], errors="coerce")
        s = s[np.isfinite(s)]
        rows.append({
            "variable": var,
            "label": VAR_LABELS.get(var, var),
            "N": int(s.shape[0]),
            "mean": float(s.mean()),
            "sd": float(s.std(ddof=1)),
            "p25": float(s.quantile(0.25)),
            "median": float(s.quantile(0.50)),
            "p75": float(s.quantile(0.75)),
            "min": float(s.min()),
            "max": float(s.max()),
        })
    return pd.DataFrame(rows)


def write_latex_table(summary_df: pd.DataFrame, outpath: str) -> None:
    lines = []
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append("\\caption{Summary Statistics for the Medium-Cleaned Paired Dataset}")
    lines.append("\\label{tab:summary_stats}")
    lines.append("\\begin{tabular}{lrrrrrrr}")
    lines.append("\\hline")
    lines.append("Variable & N & Mean & SD & P25 & Median & P75 & Min / Max \\\\")
    lines.append("\\hline")

    for _, row in summary_df.iterrows():
        lines.append(
            f"{row['label']} & "
            f"{int(row['N'])} & "
            f"{row['mean']:.3f} & "
            f"{row['sd']:.3f} & "
            f"{row['p25']:.3f} & "
            f"{row['median']:.3f} & "
            f"{row['p75']:.3f} & "
            f"{row['min']:.3f} / {row['max']:.3f} \\\\"
        )

    lines.append("\\hline")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    with open(outpath, "w") as f:
        f.write("\n".join(lines))


def savefig(path: str) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()


def make_platform_time_series(df: pd.DataFrame) -> None:
    # 1. Weekly total volume by platform
    g = (
        df.groupby(["week", "platform"], as_index=False)["total_secondary_volume_eth"]
        .sum()
    )
    plt.figure(figsize=(10, 5))
    for plat, sub in g.groupby("platform"):
        plt.plot(sub["week"], sub["total_secondary_volume_eth"], label=plat)
    plt.title("Weekly total secondary volume by platform")
    plt.xlabel("Week")
    plt.ylabel("Total secondary volume (ETH)")
    plt.legend()
    savefig(os.path.join(OUTDIR, "weekly_total_volume_by_platform.png"))

    # 2. Weekly total trades by platform
    g = (
        df.groupby(["week", "platform"], as_index=False)["trades"]
        .sum()
    )
    plt.figure(figsize=(10, 5))
    for plat, sub in g.groupby("platform"):
        plt.plot(sub["week"], sub["trades"], label=plat)
    plt.title("Weekly total trades by platform")
    plt.xlabel("Week")
    plt.ylabel("Trades")
    plt.legend()
    savefig(os.path.join(OUTDIR, "weekly_total_trades_by_platform.png"))

    # 3. Weekly mean share_volume by platform
    g = (
        df.groupby(["week", "platform"], as_index=False)["share_volume"]
        .mean()
    )
    plt.figure(figsize=(10, 5))
    for plat, sub in g.groupby("platform"):
        plt.plot(sub["week"], sub["share_volume"], label=plat)
    plt.title("Weekly mean platform volume share")
    plt.xlabel("Week")
    plt.ylabel("Mean share_volume")
    plt.legend()
    savefig(os.path.join(OUTDIR, "weekly_mean_share_volume_by_platform.png"))

    # 4. Weekly mean enforcement by platform
    g = (
        df.groupby(["week", "platform"], as_index=False)["enforcement_credibility"]
        .mean()
    )
    plt.figure(figsize=(10, 5))
    for plat, sub in g.groupby("platform"):
        plt.plot(sub["week"], sub["enforcement_credibility"], label=plat)
    plt.title("Weekly mean enforcement credibility by platform")
    plt.xlabel("Week")
    plt.ylabel("Mean enforcement credibility")
    plt.legend()
    savefig(os.path.join(OUTDIR, "weekly_mean_enforcement_by_platform.png"))

    # 5. Weekly mean effective total cost by platform
    g = (
        df.groupby(["week", "platform"], as_index=False)["effective_total_cost_rate_pct"]
        .mean()
    )
    plt.figure(figsize=(10, 5))
    for plat, sub in g.groupby("platform"):
        plt.plot(sub["week"], sub["effective_total_cost_rate_pct"], label=plat)
    plt.title("Weekly mean effective total cost by platform")
    plt.xlabel("Week")
    plt.ylabel("Mean effective total cost rate (pct)")
    plt.legend()
    savefig(os.path.join(OUTDIR, "weekly_mean_effective_cost_by_platform.png"))

    # 6. Weekly mean average trade price by platform
    g = (
        df.groupby(["week", "platform"], as_index=False)["avg_trade_price_eth"]
        .mean()
    )
    plt.figure(figsize=(10, 5))
    for plat, sub in g.groupby("platform"):
        plt.plot(sub["week"], sub["avg_trade_price_eth"], label=plat)
    plt.title("Weekly mean average trade price by platform")
    plt.xlabel("Week")
    plt.ylabel("Mean avg trade price (ETH)")
    plt.legend()
    savefig(os.path.join(OUTDIR, "weekly_mean_trade_price_by_platform.png"))


def make_market_level_switching_figure(df: pd.DataFrame) -> None:
    # switch_share_volume_ct is duplicated across the two platform rows
    # so deduplicate to the collection-week market level first
    market_df = (
        df.sort_values(["market_id", "platform"])
          .drop_duplicates(subset=["market_id"])
          .copy()
    )

    g = market_df.groupby("week", as_index=False)["switch_share_volume_ct"].mean()

    plt.figure(figsize=(10, 5))
    plt.plot(g["week"], g["switch_share_volume_ct"])
    plt.title("Weekly mean switching share at the collection-week level")
    plt.xlabel("Week")
    plt.ylabel("Mean switch_share_volume_ct")
    savefig(os.path.join(OUTDIR, "weekly_switching_share.png"))


def main():
    if len(sys.argv) < 2:
        print("Usage: python make_data_summary_and_figures.py input.csv")
        sys.exit(1)

    input_csv = sys.argv[1]
    ensure_dir(OUTDIR)

    df = read_data(input_csv)

    # Basic sample metadata
    n_rows = len(df)
    n_markets = df["market_id"].nunique() if "market_id" in df.columns else np.nan
    n_collections = df["collection_address"].nunique() if "collection_address" in df.columns else np.nan
    n_weeks = df["week"].nunique()
    print(f"Rows: {n_rows:,}")
    print(f"Markets: {n_markets:,}")
    print(f"Collections: {n_collections:,}")
    print(f"Weeks: {n_weeks:,}")

    # Summary statistics
    summary_df = make_summary_table(df)
    summary_df.to_csv(os.path.join(OUTDIR, "summary_statistics.csv"), index=False)
    write_latex_table(summary_df, os.path.join(OUTDIR, "summary_statistics.tex"))

    # Figures
    make_platform_time_series(df)
    make_market_level_switching_figure(df)

    print(f"Wrote outputs to: {OUTDIR}/")


if __name__ == "__main__":
    main()