# plot_counterfactual_effects.py
#
# Make figures from the counterfactual output CSV files.
#
# Expected files:
#   summary_counterfactuals.csv
#   alpha_bin_counterfactuals.csv
#   subgroup_counterfactuals.csv
#   structured_enforcement_counterfactuals.csv
#   structured_switching_counterfactuals.csv
#
# Usage:
#   python plot_counterfactual_effects.py /path/to/counterfactual_output_dir
#
# Or with individual files:
#   python plot_counterfactual_effects.py . \
#       --summary summary_counterfactuals.csv \
#       --alpha alpha_bin_counterfactuals.csv \
#       --subgroup subgroup_counterfactuals.csv \
#       --enforcement structured_enforcement_counterfactuals.csv \
#       --switching structured_switching_counterfactuals.csv
#
# Output:
#   Creates a folder called "counterfactual_figures" inside the base directory
#   and writes PNG figures there.

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ----------------------------
# Helpers
# ----------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("base_dir", type=str, help="Directory containing counterfactual CSV outputs.")
    p.add_argument("--summary", type=str, default=None)
    p.add_argument("--alpha", type=str, default=None)
    p.add_argument("--subgroup", type=str, default=None)
    p.add_argument("--enforcement", type=str, default=None)
    p.add_argument("--switching", type=str, default=None)
    return p.parse_args()


def resolve_path(base_dir: str, explicit_path: str | None, default_name: str) -> str:
    if explicit_path is not None:
        return explicit_path
    return os.path.join(base_dir, default_name)


def safe_read_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    return pd.read_csv(path)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def clean_alpha_bin_labels(x):
    s = str(x)
    return s.replace("[", "").replace(")", "").replace("]", "").replace(",", " to ")


def sort_alpha_bins(df: pd.DataFrame, col: str = "alpha_bin") -> pd.DataFrame:
    out = df.copy()
    out["_alpha_sort"] = out[col].astype(str).str.extract(r"^\[?\(?\s*([-+]?\d*\.?\d+)")[0].astype(float)
    out = out.sort_values("_alpha_sort").drop(columns="_alpha_sort")
    return out


def savefig(path: str):
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()


# ----------------------------
# Plot 1: Top-line bar charts
# ----------------------------
def plot_summary(summary_df: pd.DataFrame, outdir: str):
    plot_df = summary_df.copy()

    pretty_names = {
        "equalize_enforcement": "Equalize enforcement",
        "opensea_matches_blur_enforcement": "OpenSea matches Blur",
        "full_enforcement_both": "Full enforcement both",
        "remove_switching_frictions": "Remove switching frictions",
        "blur_matches_opensea_enforcement": "Blur matches OpenSea",
        "zero_enforcement_preference": "Zero enforcement preference",
        "zero_price_dependent_switching": "Zero price-dependent switching",
        "targeted_full_enforcement_high_royalty": "Targeted full enforcement\n(high royalty)",
    }
    plot_df["scenario_label"] = plot_df["scenario"].map(pretty_names).fillna(plot_df["scenario"])

    metrics = [
        ("delta_mean_share_B", "Counterfactual effect on mean Blur share", "summary_delta_mean_share_B.png"),
        ("delta_mean_share_gap", "Counterfactual effect on mean share gap", "summary_delta_mean_share_gap.png"),
        ("delta_mean_switch", "Counterfactual effect on mean switching share", "summary_delta_mean_switch.png"),
    ]

    for col, title, fname in metrics:
        fig = plt.figure(figsize=(11, 5))
        plt.bar(plot_df["scenario_label"], plot_df[col])
        plt.axhline(0, linewidth=1)
        plt.title(title)
        plt.ylabel("Counterfactual change")
        plt.xticks(rotation=35, ha="right")
        savefig(os.path.join(outdir, fname))

    # Combined figure
    fig = plt.figure(figsize=(12, 6))
    x = np.arange(len(plot_df))
    width = 0.25
    plt.bar(x - width, plot_df["delta_mean_share_B"], width=width, label="Δ Mean Blur share")
    plt.bar(x, plot_df["delta_mean_share_gap"], width=width, label="Δ Mean share gap")
    plt.bar(x + width, plot_df["delta_mean_switch"], width=width, label="Δ Mean switching")
    plt.axhline(0, linewidth=1)
    plt.xticks(x, plot_df["scenario_label"], rotation=35, ha="right")
    plt.ylabel("Counterfactual change")
    plt.title("Top-line counterfactual effects")
    plt.legend()
    savefig(os.path.join(outdir, "summary_combined_effects.png"))


# ----------------------------
# Plot 2: Alpha-bin lines
# ----------------------------
def plot_alpha_bins(alpha_df: pd.DataFrame, outdir: str):
    alpha_df = sort_alpha_bins(alpha_df, "alpha_bin")

    pretty_names = {
        "equalize_enforcement": "Equalize enforcement",
        "opensea_matches_blur_enforcement": "OpenSea matches Blur",
        "full_enforcement_both": "Full enforcement both",
        "remove_switching_frictions": "Remove switching frictions",
        "blur_matches_opensea_enforcement": "Blur matches OpenSea",
        "zero_enforcement_preference": "Zero enforcement preference",
        "zero_price_dependent_switching": "Zero price-dependent switching",
        "targeted_full_enforcement_high_royalty": "Targeted full enforcement",
    }

    alpha_df["scenario_label"] = alpha_df["scenario"].map(pretty_names).fillna(alpha_df["scenario"])
    alpha_df["alpha_bin_label"] = alpha_df["alpha_bin"].map(clean_alpha_bin_labels)

    # 2a. Delta share gap by alpha bin
    fig = plt.figure(figsize=(11, 6))
    for scen, g in alpha_df.groupby("scenario_label", sort=False):
        g = sort_alpha_bins(g, "alpha_bin")
        plt.plot(g["alpha_bin_label"], g["delta_share_gap"], marker="o", label=scen)
    plt.axhline(0, linewidth=1)
    plt.title("Counterfactual effect on share gap across royalty bins")
    plt.ylabel("Δ Mean share gap")
    plt.xlabel("Royalty-rate bin")
    plt.xticks(rotation=35, ha="right")
    plt.legend()
    savefig(os.path.join(outdir, "alpha_bins_delta_share_gap.png"))

    # 2b. Delta Blur share by alpha bin
    fig = plt.figure(figsize=(11, 6))
    for scen, g in alpha_df.groupby("scenario_label", sort=False):
        g = sort_alpha_bins(g, "alpha_bin")
        plt.plot(g["alpha_bin_label"], g["delta_share_B"], marker="o", label=scen)
    plt.axhline(0, linewidth=1)
    plt.title("Counterfactual effect on Blur share across royalty bins")
    plt.ylabel("Δ Mean Blur share")
    plt.xlabel("Royalty-rate bin")
    plt.xticks(rotation=35, ha="right")
    plt.legend()
    savefig(os.path.join(outdir, "alpha_bins_delta_share_B.png"))

    # 2c. Delta switching by alpha bin
    fig = plt.figure(figsize=(11, 6))
    for scen, g in alpha_df.groupby("scenario_label", sort=False):
        g = sort_alpha_bins(g, "alpha_bin")
        plt.plot(g["alpha_bin_label"], g["delta_switch"], marker="o", label=scen)
    plt.axhline(0, linewidth=1)
    plt.title("Counterfactual effect on switching across royalty bins")
    plt.ylabel("Δ Mean switching")
    plt.xlabel("Royalty-rate bin")
    plt.xticks(rotation=35, ha="right")
    plt.legend()
    savefig(os.path.join(outdir, "alpha_bins_delta_switch.png"))


# ----------------------------
# Plot 3: Subgroup bars
# ----------------------------
def plot_subgroups(subgroup_df: pd.DataFrame, outdir: str):
    plot_df = subgroup_df.copy()

    pretty_names = {
        "equalize_enforcement": "Equalize enforcement",
        "opensea_matches_blur_enforcement": "OpenSea matches Blur",
        "full_enforcement_both": "Full enforcement both",
        "remove_switching_frictions": "Remove switching frictions",
        "blur_matches_opensea_enforcement": "Blur matches OpenSea",
        "zero_enforcement_preference": "Zero enforcement preference",
        "zero_price_dependent_switching": "Zero price-dependent switching",
        "targeted_full_enforcement_high_royalty": "Targeted full enforcement",
    }
    group_order = ["mint", "nonmint", "hi_price", "lo_price", "tau_parity", "kappa_parity"]

    for scen in plot_df["scenario"].unique():
        g = plot_df[plot_df["scenario"] == scen].copy()
        g["group"] = pd.Categorical(g["group"], categories=group_order, ordered=True)
        g = g.sort_values("group")
        label = pretty_names.get(scen, scen)

        # share_B
        fig = plt.figure(figsize=(9, 5))
        plt.bar(g["group"].astype(str), g["delta_share_B"])
        plt.axhline(0, linewidth=1)
        plt.title(f"{label}: subgroup effects on Blur share")
        plt.ylabel("Δ Mean Blur share")
        savefig(os.path.join(outdir, f"subgroup_delta_share_B_{scen}.png"))

        # share_gap
        fig = plt.figure(figsize=(9, 5))
        plt.bar(g["group"].astype(str), g["delta_share_gap"])
        plt.axhline(0, linewidth=1)
        plt.title(f"{label}: subgroup effects on share gap")
        plt.ylabel("Δ Mean share gap")
        savefig(os.path.join(outdir, f"subgroup_delta_share_gap_{scen}.png"))

        # switch
        fig = plt.figure(figsize=(9, 5))
        plt.bar(g["group"].astype(str), g["delta_switch"])
        plt.axhline(0, linewidth=1)
        plt.title(f"{label}: subgroup effects on switching")
        plt.ylabel("Δ Mean switching")
        savefig(os.path.join(outdir, f"subgroup_delta_switch_{scen}.png"))


# ----------------------------
# Plot 4: Structured enforcement
# ----------------------------
def plot_structured_enforcement(enf_df: pd.DataFrame, outdir: str):
    plot_df = enf_df.copy()

    # combined market-structure label
    plot_df["market_structure"] = (
        plot_df["royalty_group"].astype(str)
        + " | " + plot_df["kappa_group"].astype(str)
        + " | " + plot_df["mint_group"].astype(str)
    )

    order = [
        "low | Parity | Blur mint",
        "low | Parity | Blur nonmint",
        "low | Blur stronger | Blur mint",
        "low | Blur stronger | Blur nonmint",
        "low | OpenSea stronger | Blur mint",
        "low | OpenSea stronger | Blur nonmint",
        "medium | Parity | Blur mint",
        "medium | Parity | Blur nonmint",
        "medium | Blur stronger | Blur mint",
        "medium | Blur stronger | Blur nonmint",
        "medium | OpenSea stronger | Blur mint",
        "medium | OpenSea stronger | Blur nonmint",
        "high | Parity | Blur mint",
        "high | Parity | Blur nonmint",
        "high | Blur stronger | Blur mint",
        "high | Blur stronger | Blur nonmint",
        "high | OpenSea stronger | Blur mint",
        "high | OpenSea stronger | Blur nonmint",
    ]
    plot_df["market_structure"] = pd.Categorical(plot_df["market_structure"], categories=order, ordered=True)

    pretty_names = {
        "equalize_enforcement": "Equalize enforcement",
        "opensea_matches_blur_enforcement": "OpenSea matches Blur",
        "blur_matches_opensea_enforcement": "Blur matches OpenSea",
        "full_enforcement_both": "Full enforcement both",
        "targeted_full_enforcement_high_royalty": "Targeted full enforcement",
        "zero_enforcement_preference": "Zero enforcement preference",
    }
    plot_df["scenario_label"] = plot_df["scenario"].map(pretty_names).fillna(plot_df["scenario"])

    metrics = [
        ("mean_delta_share_B", "Structured enforcement effects on Blur share", "structured_enforcement_delta_share_B.png"),
        ("mean_delta_share_gap", "Structured enforcement effects on share gap", "structured_enforcement_delta_share_gap.png"),
        ("mean_delta_switch", "Structured enforcement effects on switching", "structured_enforcement_delta_switch.png"),
    ]

    for col, title, fname in metrics:
        pivot = plot_df.pivot(index="market_structure", columns="scenario_label", values=col).sort_index()
        fig = plt.figure(figsize=(14, 6))
        pivot.plot(kind="bar", ax=plt.gca())
        plt.axhline(0, linewidth=1)
        plt.title(title)
        plt.ylabel("Mean counterfactual change")
        plt.xlabel("Market structure")
        plt.xticks(rotation=45, ha="right")
        plt.legend(title="")
        savefig(os.path.join(outdir, fname))


# ----------------------------
# Plot 5: Structured switching
# ----------------------------
def plot_structured_switching(sw_df: pd.DataFrame, outdir: str):
    plot_df = sw_df.copy()

    plot_df["market_structure"] = (
        plot_df["price_group"].astype(str)
        + " | " + plot_df["mint_group"].astype(str)
        + " | " + plot_df["tau_parity_group"].astype(str)
    )

    order = [
        "low | Blur mint | tau parity",
        "low | Blur mint | non-parity",
        "low | Blur nonmint | tau parity",
        "low | Blur nonmint | non-parity",
        "high | Blur mint | tau parity",
        "high | Blur mint | non-parity",
        "high | Blur nonmint | tau parity",
        "high | Blur nonmint | non-parity",
    ]
    plot_df["market_structure"] = pd.Categorical(plot_df["market_structure"], categories=order, ordered=True)

    pretty_names = {
        "remove_switching_frictions": "Remove switching frictions",
        "zero_price_dependent_switching": "Zero price-dependent switching",
    }
    plot_df["scenario_label"] = plot_df["scenario"].map(pretty_names).fillna(plot_df["scenario"])

    metrics = [
        ("mean_delta_share_B", "Structured switching effects on Blur share", "structured_switching_delta_share_B.png"),
        ("mean_delta_share_gap", "Structured switching effects on share gap", "structured_switching_delta_share_gap.png"),
        ("mean_delta_switch", "Structured switching effects on switching", "structured_switching_delta_switch.png"),
    ]

    for col, title, fname in metrics:
        pivot = plot_df.pivot(index="market_structure", columns="scenario_label", values=col).sort_index()
        fig = plt.figure(figsize=(12, 6))
        pivot.plot(kind="bar", ax=plt.gca())
        plt.axhline(0, linewidth=1)
        plt.title(title)
        plt.ylabel("Mean counterfactual change")
        plt.xlabel("Market structure")
        plt.xticks(rotation=35, ha="right")
        plt.legend(title="")
        savefig(os.path.join(outdir, fname))


# ----------------------------
# Main
# ----------------------------
def main():
    args = parse_args()

    summary_path = resolve_path(args.base_dir, args.summary, "summary_counterfactuals.csv")
    alpha_path = resolve_path(args.base_dir, args.alpha, "alpha_bin_counterfactuals.csv")
    subgroup_path = resolve_path(args.base_dir, args.subgroup, "subgroup_counterfactuals.csv")
    enforcement_path = resolve_path(args.base_dir, args.enforcement, "structured_enforcement_counterfactuals.csv")
    switching_path = resolve_path(args.base_dir, args.switching, "structured_switching_counterfactuals.csv")

    outdir = os.path.join(args.base_dir, "counterfactual_figures")
    ensure_dir(outdir)

    summary_df = safe_read_csv(summary_path)
    alpha_df = safe_read_csv(alpha_path)
    subgroup_df = safe_read_csv(subgroup_path)
    enforcement_df = safe_read_csv(enforcement_path)
    switching_df = safe_read_csv(switching_path)

    plot_summary(summary_df, outdir)
    plot_alpha_bins(alpha_df, outdir)
    plot_subgroups(subgroup_df, outdir)
    plot_structured_enforcement(enforcement_df, outdir)
    plot_structured_switching(switching_df, outdir)

    print(f"Wrote figures to: {outdir}")
    for f in sorted(os.listdir(outdir)):
        print(f" - {f}")


if __name__ == "__main__":
    main()