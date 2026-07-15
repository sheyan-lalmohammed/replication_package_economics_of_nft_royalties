# run_v2_counterfactuals_structured_extended.py
#
# Plain-v2 counterfactuals + structured market summaries
# + 3 added counterfactuals:
#   6) zero_enforcement_preference        : phi_K = 0
#   7) zero_price_dependent_switching     : k1 = 0 (keep k0)
#   8) targeted_full_enforcement_high_royalty : set kappa_B = kappa_O = 1 only when alpha_c >= 7.5
#
# Existing scenarios:
#   1) equalize_enforcement
#   2) opensea_matches_blur_enforcement
#   3) full_enforcement_both
#   4) remove_switching_frictions
#   5) blur_matches_opensea_enforcement
#
# Outputs:
#   - summary_counterfactuals.csv
#   - alpha_bin_counterfactuals.csv
#   - subgroup_counterfactuals.csv
#   - market_level_counterfactuals.csv
#   - structured_enforcement_counterfactuals.csv
#   - structured_switching_counterfactuals.csv
#
# Usage:
#   python run_v2_counterfactuals_structured_extended.py params.csv fitted_markets.csv [outdir]

import os
import sys
import numpy as np
import pandas as pd
from dataclasses import dataclass


ALPHA_BINS = np.array([0, 1, 2.5, 5, 7.5, 10, 15], dtype=float)
TAU_PARITY_TOL = 0.50
KAPPA_PARITY_TOL = 0.05
HIGH_ROYALTY_THRESHOLD = 7.5


@dataclass
class Params:
    delta_B: float
    lambda_S: float
    phi_K: float
    k0: float
    k1: float
    gamma_S: float


def load_params(params_csv: str) -> Params:
    p = pd.read_csv(params_csv).iloc[0].to_dict()
    needed = ["delta_B", "lambda_S", "phi_K", "k0", "k1", "gamma_S"]
    missing = [k for k in needed if k not in p]
    if missing:
        raise ValueError(f"params.csv is missing required columns: {missing}")

    return Params(
        delta_B=float(p["delta_B"]),
        lambda_S=float(p["lambda_S"]),
        phi_K=float(p["phi_K"]),
        k0=float(p["k0"]),
        k1=float(p["k1"]),
        gamma_S=float(p["gamma_S"]),
    )


def _subset_mean(arr: pd.Series, mask: pd.Series) -> float:
    x = arr[mask]
    if len(x) == 0:
        return np.nan
    return float(np.mean(x))


def _assign_bins(x: pd.Series, bins: np.ndarray) -> pd.Categorical:
    return pd.cut(x, bins=bins, include_lowest=True, right=False)


def predict_shares_v2(wide: pd.DataFrame, par: Params):
    tau_B = wide["tau_B"].to_numpy()
    tau_O = wide["tau_O"].to_numpy()
    kappa_diff = wide["kappa_diff"].to_numpy()

    z_lpB = wide["z_lpB"].to_numpy()
    z_lpO = wide["z_lpO"].to_numpy()
    B_is_mint = wide["B_is_mint"].to_numpy().astype(int)

    off_B = par.k0 + par.k1 * z_lpB
    off_O = par.k0 + par.k1 * z_lpO
    off_B = np.maximum(off_B, 0.0)
    off_O = np.maximum(off_O, 0.0)

    tau_tilde_B = tau_B + np.where(B_is_mint == 1, 0.0, off_B)
    tau_tilde_O = tau_O + np.where(B_is_mint == 1, off_O, 0.0)
    d_tau_tilde = tau_tilde_B - tau_tilde_O

    V_S = par.delta_B - par.lambda_S * d_tau_tilde
    V_C = par.delta_B - par.lambda_S * d_tau_tilde + par.phi_K * kappa_diff

    sB_S = 1.0 / (1.0 + np.exp(-np.clip(V_S, -60, 60)))
    sB_C = 1.0 / (1.0 + np.exp(-np.clip(V_C, -60, 60)))

    sB_hat = par.gamma_S * sB_S + (1.0 - par.gamma_S) * sB_C
    switch_hat = np.where(B_is_mint == 1, 1.0 - sB_hat, sB_hat)

    return sB_hat, switch_hat


# ----------------------------
# Counterfactual builders
# ----------------------------
def make_equalize_enforcement(wide: pd.DataFrame) -> pd.DataFrame:
    cf = wide.copy()
    k_eq = 0.5 * (cf["kappa_B"] + cf["kappa_O"])
    cf["kappa_B"] = k_eq
    cf["kappa_O"] = k_eq
    cf["kappa_diff"] = cf["kappa_B"] - cf["kappa_O"]
    cf["tau_B"] = cf["beta_Blur"] + cf["alpha_c"] * cf["kappa_B"]
    cf["tau_O"] = cf["beta_OpenSea"] + cf["alpha_c"] * cf["kappa_O"]
    cf["tau_diff"] = cf["tau_B"] - cf["tau_O"]
    return cf


def make_opensea_matches_blur(wide: pd.DataFrame) -> pd.DataFrame:
    cf = wide.copy()
    cf["kappa_O"] = np.maximum(cf["kappa_O"], cf["kappa_B"])
    cf["kappa_diff"] = cf["kappa_B"] - cf["kappa_O"]
    cf["tau_B"] = cf["beta_Blur"] + cf["alpha_c"] * cf["kappa_B"]
    cf["tau_O"] = cf["beta_OpenSea"] + cf["alpha_c"] * cf["kappa_O"]
    cf["tau_diff"] = cf["tau_B"] - cf["tau_O"]
    return cf


def make_blur_matches_opensea(wide: pd.DataFrame) -> pd.DataFrame:
    cf = wide.copy()
    cf["kappa_B"] = np.maximum(cf["kappa_B"], cf["kappa_O"])
    cf["kappa_diff"] = cf["kappa_B"] - cf["kappa_O"]
    cf["tau_B"] = cf["beta_Blur"] + cf["alpha_c"] * cf["kappa_B"]
    cf["tau_O"] = cf["beta_OpenSea"] + cf["alpha_c"] * cf["kappa_O"]
    cf["tau_diff"] = cf["tau_B"] - cf["tau_O"]
    return cf


def make_full_enforcement_both(wide: pd.DataFrame) -> pd.DataFrame:
    cf = wide.copy()
    cf["kappa_B"] = 1.0
    cf["kappa_O"] = 1.0
    cf["kappa_diff"] = 0.0
    cf["tau_B"] = cf["beta_Blur"] + cf["alpha_c"] * 1.0
    cf["tau_O"] = cf["beta_OpenSea"] + cf["alpha_c"] * 1.0
    cf["tau_diff"] = cf["tau_B"] - cf["tau_O"]
    return cf


def make_targeted_full_enforcement_high_royalty(wide: pd.DataFrame) -> pd.DataFrame:
    cf = wide.copy()
    m = cf["alpha_c"] >= HIGH_ROYALTY_THRESHOLD
    cf.loc[m, "kappa_B"] = 1.0
    cf.loc[m, "kappa_O"] = 1.0
    cf["kappa_diff"] = cf["kappa_B"] - cf["kappa_O"]
    cf["tau_B"] = cf["beta_Blur"] + cf["alpha_c"] * cf["kappa_B"]
    cf["tau_O"] = cf["beta_OpenSea"] + cf["alpha_c"] * cf["kappa_O"]
    cf["tau_diff"] = cf["tau_B"] - cf["tau_O"]
    return cf


# ----------------------------
# Parameter-only counterfactuals
# ----------------------------
def make_no_switching_params(par: Params) -> Params:
    return Params(
        delta_B=par.delta_B,
        lambda_S=par.lambda_S,
        phi_K=par.phi_K,
        k0=0.0,
        k1=0.0,
        gamma_S=par.gamma_S,
    )


def make_zero_enforcement_preference_params(par: Params) -> Params:
    return Params(
        delta_B=par.delta_B,
        lambda_S=par.lambda_S,
        phi_K=0.0,
        k0=par.k0,
        k1=par.k1,
        gamma_S=par.gamma_S,
    )


def make_zero_price_dependent_switching_params(par: Params) -> Params:
    return Params(
        delta_B=par.delta_B,
        lambda_S=par.lambda_S,
        phi_K=par.phi_K,
        k0=par.k0,
        k1=0.0,
        gamma_S=par.gamma_S,
    )


# ----------------------------
# Summaries
# ----------------------------
def summarize_scenario(name: str, base_df: pd.DataFrame, cf_df: pd.DataFrame) -> dict:
    return {
        "scenario": name,
        "mean_share_B_base": float(base_df["share_B_hat"].mean()),
        "mean_share_B_cf": float(cf_df["share_B_hat"].mean()),
        "delta_mean_share_B": float(cf_df["share_B_hat"].mean() - base_df["share_B_hat"].mean()),
        "mean_share_gap_base": float(base_df["share_gap_hat"].mean()),
        "mean_share_gap_cf": float(cf_df["share_gap_hat"].mean()),
        "delta_mean_share_gap": float(cf_df["share_gap_hat"].mean() - base_df["share_gap_hat"].mean()),
        "mean_switch_base": float(base_df["switch_hat"].mean()),
        "mean_switch_cf": float(cf_df["switch_hat"].mean()),
        "delta_mean_switch": float(cf_df["switch_hat"].mean() - base_df["switch_hat"].mean()),

        "mean_share_B_tau_parity_base": _subset_mean(base_df["share_B_hat"], base_df["tau_parity"]),
        "mean_share_B_tau_parity_cf": _subset_mean(cf_df["share_B_hat"], cf_df["tau_parity"]),
        "delta_share_B_tau_parity": _subset_mean(cf_df["share_B_hat"], cf_df["tau_parity"]) - _subset_mean(base_df["share_B_hat"], base_df["tau_parity"]),

        "mean_share_B_kappa_parity_base": _subset_mean(base_df["share_B_hat"], base_df["kappa_parity"]),
        "mean_share_B_kappa_parity_cf": _subset_mean(cf_df["share_B_hat"], cf_df["kappa_parity"]),
        "delta_share_B_kappa_parity": _subset_mean(cf_df["share_B_hat"], cf_df["kappa_parity"]) - _subset_mean(base_df["share_B_hat"], base_df["kappa_parity"]),

        "mean_switch_mint_base": _subset_mean(base_df["switch_hat"], base_df["mint"]),
        "mean_switch_mint_cf": _subset_mean(cf_df["switch_hat"], cf_df["mint"]),
        "delta_switch_mint": _subset_mean(cf_df["switch_hat"], cf_df["mint"]) - _subset_mean(base_df["switch_hat"], base_df["mint"]),

        "mean_switch_nonmint_base": _subset_mean(base_df["switch_hat"], base_df["nonmint"]),
        "mean_switch_nonmint_cf": _subset_mean(cf_df["switch_hat"], cf_df["nonmint"]),
        "delta_switch_nonmint": _subset_mean(cf_df["switch_hat"], cf_df["nonmint"]) - _subset_mean(base_df["switch_hat"], base_df["nonmint"]),

        "mean_switch_hi_price_base": _subset_mean(base_df["switch_hat"], base_df["hi_price"]),
        "mean_switch_hi_price_cf": _subset_mean(cf_df["switch_hat"], cf_df["hi_price"]),
        "delta_switch_hi_price": _subset_mean(cf_df["switch_hat"], cf_df["hi_price"]) - _subset_mean(base_df["switch_hat"], base_df["hi_price"]),

        "mean_switch_lo_price_base": _subset_mean(base_df["switch_hat"], base_df["lo_price"]),
        "mean_switch_lo_price_cf": _subset_mean(cf_df["switch_hat"], cf_df["lo_price"]),
        "delta_switch_lo_price": _subset_mean(cf_df["switch_hat"], cf_df["lo_price"]) - _subset_mean(base_df["switch_hat"], base_df["lo_price"]),
    }


def alpha_bin_summary(name: str, base_df: pd.DataFrame, cf_df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame({"alpha_bin": _assign_bins(base_df["alpha_c"], ALPHA_BINS)})
    out["share_gap_base"] = base_df["share_gap_hat"]
    out["share_gap_cf"] = cf_df["share_gap_hat"]
    out["share_B_base"] = base_df["share_B_hat"]
    out["share_B_cf"] = cf_df["share_B_hat"]
    out["switch_base"] = base_df["switch_hat"]
    out["switch_cf"] = cf_df["switch_hat"]

    g = out.groupby("alpha_bin", observed=False).agg(
        n=("share_gap_base", "size"),
        mean_share_gap_base=("share_gap_base", "mean"),
        mean_share_gap_cf=("share_gap_cf", "mean"),
        mean_share_B_base=("share_B_base", "mean"),
        mean_share_B_cf=("share_B_cf", "mean"),
        mean_switch_base=("switch_base", "mean"),
        mean_switch_cf=("switch_cf", "mean"),
    ).reset_index()

    g["delta_share_gap"] = g["mean_share_gap_cf"] - g["mean_share_gap_base"]
    g["delta_share_B"] = g["mean_share_B_cf"] - g["mean_share_B_base"]
    g["delta_switch"] = g["mean_switch_cf"] - g["mean_switch_base"]
    g.insert(0, "scenario", name)
    return g


def subgroup_summary(name: str, base_df: pd.DataFrame, cf_df: pd.DataFrame) -> pd.DataFrame:
    groups = {
        "mint": base_df["mint"],
        "nonmint": base_df["nonmint"],
        "hi_price": base_df["hi_price"],
        "lo_price": base_df["lo_price"],
        "tau_parity": base_df["tau_parity"],
        "kappa_parity": base_df["kappa_parity"],
    }

    rows = []
    for group_name, mask in groups.items():
        rows.append({
            "scenario": name,
            "group": group_name,
            "n": int(mask.sum()),
            "mean_share_B_base": _subset_mean(base_df["share_B_hat"], mask),
            "mean_share_B_cf": _subset_mean(cf_df["share_B_hat"], mask),
            "delta_share_B": _subset_mean(cf_df["share_B_hat"], mask) - _subset_mean(base_df["share_B_hat"], mask),
            "mean_share_gap_base": _subset_mean(base_df["share_gap_hat"], mask),
            "mean_share_gap_cf": _subset_mean(cf_df["share_gap_hat"], mask),
            "delta_share_gap": _subset_mean(cf_df["share_gap_hat"], mask) - _subset_mean(base_df["share_gap_hat"], mask),
            "mean_switch_base": _subset_mean(base_df["switch_hat"], mask),
            "mean_switch_cf": _subset_mean(cf_df["switch_hat"], mask),
            "delta_switch": _subset_mean(cf_df["switch_hat"], mask) - _subset_mean(base_df["switch_hat"], mask),
        })
    return pd.DataFrame(rows)


# ----------------------------
# Structured summaries
# ----------------------------
def add_structure_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["kappa_diff_base"] = out["kappa_B_base"] - out["kappa_O_base"]

    out["royalty_group"] = pd.cut(
        out["alpha_c"],
        bins=[-np.inf, 2.5, 7.5, np.inf],
        labels=["low", "medium", "high"],
        right=False,
    )

    def kappa_group(x):
        if x > 0.10:
            return "Blur stronger"
        if x < -0.10:
            return "OpenSea stronger"
        return "Parity"

    out["kappa_group"] = out["kappa_diff_base"].map(kappa_group)
    out["mint_group"] = out["B_is_mint"].map({1: "Blur mint", 0: "Blur nonmint"})
    out["price_group"] = out["hi_price"].map({True: "high", False: "low"})
    out["tau_parity_group"] = np.where(np.abs(out["tau_diff_base"]) <= TAU_PARITY_TOL, "tau parity", "non-parity")
    return out


def structured_enforcement_summary(market_level_df: pd.DataFrame) -> pd.DataFrame:
    scenarios = [
        "equalize_enforcement",
        "opensea_matches_blur_enforcement",
        "blur_matches_opensea_enforcement",
        "full_enforcement_both",
        "targeted_full_enforcement_high_royalty",
        "zero_enforcement_preference",
    ]
    sub = market_level_df[market_level_df["scenario"].isin(scenarios)].copy()
    g = (
        sub.groupby(
            ["scenario", "royalty_group", "kappa_group", "mint_group"],
            observed=False
        )
        .agg(
            n=("delta_share_B", "size"),
            mean_delta_share_B=("delta_share_B", "mean"),
            mean_delta_share_gap=("delta_share_gap", "mean"),
            mean_delta_switch=("delta_switch", "mean"),
        )
        .reset_index()
    )
    return g


def structured_switching_summary(market_level_df: pd.DataFrame) -> pd.DataFrame:
    scenarios = [
        "remove_switching_frictions",
        "zero_price_dependent_switching",
    ]
    sub = market_level_df[market_level_df["scenario"].isin(scenarios)].copy()
    g = (
        sub.groupby(
            ["scenario", "price_group", "mint_group", "tau_parity_group"],
            observed=False
        )
        .agg(
            n=("delta_share_B", "size"),
            mean_delta_share_B=("delta_share_B", "mean"),
            mean_delta_share_gap=("delta_share_gap", "mean"),
            mean_delta_switch=("delta_switch", "mean"),
        )
        .reset_index()
    )
    return g


# ----------------------------
# Main
# ----------------------------
def main(params_csv: str, fitted_markets_csv: str, outdir: str):
    os.makedirs(outdir, exist_ok=True)

    par = load_params(params_csv)
    wide = pd.read_csv(fitted_markets_csv)

    needed = [
        "kappa_B", "kappa_O", "tau_B", "tau_O", "tau_diff",
        "beta_Blur", "beta_OpenSea", "alpha_c",
        "z_lpB", "z_lpO", "B_is_mint", "hi_price"
    ]
    missing = [c for c in needed if c not in wide.columns]
    if missing:
        raise ValueError(f"fitted_markets.csv is missing required columns for counterfactuals: {missing}")

    # Baseline
    base_share, base_switch = predict_shares_v2(wide, par)
    base = wide.copy()
    base["share_B_hat"] = base_share
    base["switch_hat"] = base_switch
    base["share_gap_hat"] = 2.0 * base["share_B_hat"] - 1.0

    base["mint"] = base["B_is_mint"].astype(bool)
    base["nonmint"] = ~base["mint"]
    base["hi_price"] = base["hi_price"].astype(bool)
    base["lo_price"] = ~base["hi_price"]
    base["tau_parity"] = np.abs(base["tau_diff"]) <= TAU_PARITY_TOL
    base["kappa_parity"] = np.abs(base["kappa_diff"]) <= KAPPA_PARITY_TOL

    scenarios = {}

    # 1) Equalize enforcement
    cf1 = make_equalize_enforcement(base)
    s1, w1 = predict_shares_v2(cf1, par)
    cf1["share_B_hat"] = s1
    cf1["switch_hat"] = w1
    cf1["share_gap_hat"] = 2.0 * cf1["share_B_hat"] - 1.0
    scenarios["equalize_enforcement"] = cf1

    # 2) OpenSea matches Blur
    cf2 = make_opensea_matches_blur(base)
    s2, w2 = predict_shares_v2(cf2, par)
    cf2["share_B_hat"] = s2
    cf2["switch_hat"] = w2
    cf2["share_gap_hat"] = 2.0 * cf2["share_B_hat"] - 1.0
    scenarios["opensea_matches_blur_enforcement"] = cf2

    # 3) Full enforcement both
    cf3 = make_full_enforcement_both(base)
    s3, w3 = predict_shares_v2(cf3, par)
    cf3["share_B_hat"] = s3
    cf3["switch_hat"] = w3
    cf3["share_gap_hat"] = 2.0 * cf3["share_B_hat"] - 1.0
    scenarios["full_enforcement_both"] = cf3

    # 4) Remove switching frictions entirely
    par4 = make_no_switching_params(par)
    cf4 = base.copy()
    s4, w4 = predict_shares_v2(cf4, par4)
    cf4["share_B_hat"] = s4
    cf4["switch_hat"] = w4
    cf4["share_gap_hat"] = 2.0 * cf4["share_B_hat"] - 1.0
    scenarios["remove_switching_frictions"] = cf4

    # 5) Blur matches OpenSea
    cf5 = make_blur_matches_opensea(base)
    s5, w5 = predict_shares_v2(cf5, par)
    cf5["share_B_hat"] = s5
    cf5["switch_hat"] = w5
    cf5["share_gap_hat"] = 2.0 * cf5["share_B_hat"] - 1.0
    scenarios["blur_matches_opensea_enforcement"] = cf5

    # 6) Zero enforcement preference
    par6 = make_zero_enforcement_preference_params(par)
    cf6 = base.copy()
    s6, w6 = predict_shares_v2(cf6, par6)
    cf6["share_B_hat"] = s6
    cf6["switch_hat"] = w6
    cf6["share_gap_hat"] = 2.0 * cf6["share_B_hat"] - 1.0
    scenarios["zero_enforcement_preference"] = cf6

    # 7) Zero price-dependent switching only
    par7 = make_zero_price_dependent_switching_params(par)
    cf7 = base.copy()
    s7, w7 = predict_shares_v2(cf7, par7)
    cf7["share_B_hat"] = s7
    cf7["switch_hat"] = w7
    cf7["share_gap_hat"] = 2.0 * cf7["share_B_hat"] - 1.0
    scenarios["zero_price_dependent_switching"] = cf7

    # 8) Targeted full enforcement only in high-royalty markets
    cf8 = make_targeted_full_enforcement_high_royalty(base)
    s8, w8 = predict_shares_v2(cf8, par)
    cf8["share_B_hat"] = s8
    cf8["switch_hat"] = w8
    cf8["share_gap_hat"] = 2.0 * cf8["share_B_hat"] - 1.0
    scenarios["targeted_full_enforcement_high_royalty"] = cf8

    # Add common masks to all scenarios
    for name, cf in scenarios.items():
        cf["mint"] = base["mint"]
        cf["nonmint"] = base["nonmint"]
        cf["hi_price"] = base["hi_price"]
        cf["lo_price"] = base["lo_price"]
        cf["tau_parity"] = np.abs(cf["tau_diff"]) <= TAU_PARITY_TOL
        cf["kappa_parity"] = np.abs(cf["kappa_diff"]) <= KAPPA_PARITY_TOL

    # Summaries
    summary_rows = []
    alpha_bin_rows = []
    subgroup_rows = []
    market_level_rows = []

    for name, cf in scenarios.items():
        summary_rows.append(summarize_scenario(name, base, cf))
        alpha_bin_rows.append(alpha_bin_summary(name, base, cf))
        subgroup_rows.append(subgroup_summary(name, base, cf))

        tmp = pd.DataFrame({
            "scenario": name,
            "market_id": base["market_id"] if "market_id" in base.columns else np.arange(len(base)),
            "week": base["week"] if "week" in base.columns else np.nan,
            "collection_address": base["collection_address"] if "collection_address" in base.columns else np.nan,
            "alpha_c": base["alpha_c"],
            "B_is_mint": base["B_is_mint"],
            "hi_price": base["hi_price"],

            "kappa_B_base": base["kappa_B"],
            "kappa_O_base": base["kappa_O"],
            "tau_B_base": base["tau_B"],
            "tau_O_base": base["tau_O"],
            "tau_diff_base": base["tau_diff"],
            "share_B_hat_base": base["share_B_hat"],
            "share_gap_hat_base": base["share_gap_hat"],
            "switch_hat_base": base["switch_hat"],

            "kappa_B_cf": cf["kappa_B"],
            "kappa_O_cf": cf["kappa_O"],
            "tau_B_cf": cf["tau_B"],
            "tau_O_cf": cf["tau_O"],
            "tau_diff_cf": cf["tau_diff"],
            "share_B_hat_cf": cf["share_B_hat"],
            "share_gap_hat_cf": cf["share_gap_hat"],
            "switch_hat_cf": cf["switch_hat"],
        })
        tmp["delta_share_B"] = tmp["share_B_hat_cf"] - tmp["share_B_hat_base"]
        tmp["delta_share_gap"] = tmp["share_gap_hat_cf"] - tmp["share_gap_hat_base"]
        tmp["delta_switch"] = tmp["switch_hat_cf"] - tmp["switch_hat_base"]
        market_level_rows.append(tmp)

    summary_df = pd.DataFrame(summary_rows)
    alpha_bin_df = pd.concat(alpha_bin_rows, axis=0, ignore_index=True)
    subgroup_df = pd.concat(subgroup_rows, axis=0, ignore_index=True)
    market_level_df = pd.concat(market_level_rows, axis=0, ignore_index=True)

    market_level_df = add_structure_columns(market_level_df)
    structured_enf_df = structured_enforcement_summary(market_level_df)
    structured_switch_df = structured_switching_summary(market_level_df)

    # Write outputs
    summary_df.to_csv(os.path.join(outdir, "summary_counterfactuals.csv"), index=False)
    alpha_bin_df.to_csv(os.path.join(outdir, "alpha_bin_counterfactuals.csv"), index=False)
    subgroup_df.to_csv(os.path.join(outdir, "subgroup_counterfactuals.csv"), index=False)
    market_level_df.to_csv(os.path.join(outdir, "market_level_counterfactuals.csv"), index=False)
    structured_enf_df.to_csv(os.path.join(outdir, "structured_enforcement_counterfactuals.csv"), index=False)
    structured_switch_df.to_csv(os.path.join(outdir, "structured_switching_counterfactuals.csv"), index=False)

    print(f"Wrote summary to: {os.path.join(outdir, 'summary_counterfactuals.csv')}")
    print(f"Wrote alpha-bin results to: {os.path.join(outdir, 'alpha_bin_counterfactuals.csv')}")
    print(f"Wrote subgroup results to: {os.path.join(outdir, 'subgroup_counterfactuals.csv')}")
    print(f"Wrote market-level results to: {os.path.join(outdir, 'market_level_counterfactuals.csv')}")
    print(f"Wrote structured enforcement results to: {os.path.join(outdir, 'structured_enforcement_counterfactuals.csv')}")
    print(f"Wrote structured switching results to: {os.path.join(outdir, 'structured_switching_counterfactuals.csv')}")
    print("\nTop-line summary:")
    print(
        summary_df[
            ["scenario", "delta_mean_share_B", "delta_mean_share_gap", "delta_mean_switch"]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python run_v2_counterfactuals_structured_extended.py params.csv fitted_markets.csv [outdir]")
        sys.exit(1)

    params_csv = sys.argv[1]
    fitted_markets_csv = sys.argv[2]
    outdir = sys.argv[3] if len(sys.argv) >= 4 else "v2_counterfactuals_structured_extended_out"
    main(params_csv, fitted_markets_csv, outdir)