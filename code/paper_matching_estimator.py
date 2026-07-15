import os
import sys
import math
import numpy as np
import pandas as pd

from dataclasses import dataclass
from typing import Dict, Tuple
from scipy.optimize import minimize


# ----------------------------
# CONFIG
# ----------------------------
PLATFORM_B = "Blur"
PLATFORM_O = "OpenSea"
START_WEEK = pd.Timestamp("2023-04-03", tz="UTC")

ALPHA_BINS = np.array([0, 1, 2.5, 5, 7.5, 10, 15], dtype=float)
KAPPA_BINS = np.array([-1.0, -0.75, -0.5, -0.25, -0.10, 0.0, 0.10, 0.25, 0.5, 0.75, 1.0], dtype=float)
TAU_DIFF_BINS = np.array([-7.5, -5, -2.5, -1, -0.5, 0, 0.5, 1, 2.5, 5, 7.5], dtype=float)

RIDGE = 1e-4
MIN_BIN_COUNT = 75

TAU_PARITY_TOL = 0.50
KAPPA_PARITY_TOL = 0.05

W_MOMENTS = {
    "share_gap_alpha": 1.0,
    "share_gap_kappa_diff": 1.5,
    "share_gap_tau_diff": 2.0,
    "switch_tau_diff": 1.5,

    "mean_share_B": 0.5,
    "mean_switch": 0.5,

    # subgroup moments
    "share_mint": 1.0,
    "share_nonmint": 1.0,
    "switch_mint": 1.0,
    "switch_nonmint": 1.0,

    "share_hi_price": 1.0,
    "share_lo_price": 1.0,
    "switch_hi_price": 1.5,
    "switch_lo_price": 1.5,

    "share_tau_parity": 2.0,
    "share_kappa_parity": 2.0,
    "switch_tau_parity": 1.5,

    "slope_share_tau": 2.0,
    "slope_switch_tau": 1.5,
    "slope_share_kappa": 1.5,
}

# weak regularization on boundary-prone channels
#REG_PHI = 0.01
#REG_K1 = 0.01


# ----------------------------
# HELPERS
# ----------------------------
def safe_logit(p: float) -> float:
    p = min(max(float(p), 1e-8), 1 - 1e-8)
    return math.log(p / (1 - p))

def inv_logit(x: float) -> float:
    x = float(np.clip(x, -60, 60))
    return 1.0 / (1.0 + math.exp(-x))

def clip(x: float, lo: float, hi: float) -> float:
    return float(min(max(float(x), lo), hi))

def zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    m = np.nanmean(x)
    s = np.nanstd(x)
    if (not np.isfinite(s)) or s < 1e-12:
        return np.zeros_like(x, dtype=float)
    return (x - m) / s

def _safe_slope(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]
    y = y[m]
    if len(x) <= 10:
        return 0.0
    vx = np.var(x)
    if (not np.isfinite(vx)) or vx < 1e-12:
        return 0.0
    return float(np.cov(x, y, bias=True)[0, 1] / (vx + 1e-12))

def _binned_means(x: np.ndarray, y: np.ndarray, bins: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    bin_id = np.digitize(x, bins) - 1
    K = len(bins) - 1
    means = np.full(K, np.nan)
    counts = np.zeros(K, dtype=float)
    for k in range(K):
        m = (bin_id == k) & np.isfinite(y)
        counts[k] = float(m.sum())
        if counts[k] > 0:
            means[k] = float(np.mean(y[m]))
    return means, counts


# ----------------------------
# DATA PREP
# ----------------------------
def load_and_pivot(path_csv: str) -> pd.DataFrame:
    df = pd.read_csv(path_csv)

    df["week"] = pd.to_datetime(df["week"], utc=True, errors="coerce")
    df = df.dropna(subset=["week"])
    df = df[df["week"] >= START_WEEK].copy()

    required = [
        "week",
        "collection_address",
        "platform",
        "share_volume",
        "platform_fee_rate_pct",
        "royalty_rate_pct",
        "enforcement_credibility",
        "avg_trade_price_eth",
        "is_mint_platform_proxy",
        "switch_share_volume_ct",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df[df["platform"].isin([PLATFORM_B, PLATFORM_O])].copy()

    num_cols = [
        "share_volume",
        "platform_fee_rate_pct",
        "royalty_rate_pct",
        "enforcement_credibility",
        "avg_trade_price_eth",
        "switch_share_volume_ct",
        "is_mint_platform_proxy",
    ]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=num_cols)
    df = df[np.isfinite(df[num_cols]).all(axis=1)].copy()
    df["share_volume"] = df["share_volume"].clip(1e-9, 1 - 1e-9)

    if "market_id" not in df.columns:
        df["market_id"] = df["collection_address"].astype(str) + "||" + df["week"].astype(str)

    # paper objects
    df["beta"] = df["platform_fee_rate_pct"]
    df["alpha"] = df["royalty_rate_pct"]
    df["kappa"] = df["enforcement_credibility"]
    df["tau"] = df["beta"] + df["alpha"] * df["kappa"]
    df["price"] = df["avg_trade_price_eth"].clip(1e-12)
    df["log_price"] = np.log(df["price"])

    wide = df.pivot_table(
        index=["market_id", "week", "collection_address"],
        columns="platform",
        values=[
            "share_volume",
            "beta",
            "alpha",
            "kappa",
            "tau",
            "price",
            "log_price",
            "is_mint_platform_proxy",
            "switch_share_volume_ct",
        ],
        aggfunc="first",
    )
    wide.columns = [f"{v}_{k}" for v, k in wide.columns]
    wide = wide.reset_index()

    needed = [
        f"share_volume_{PLATFORM_B}", f"share_volume_{PLATFORM_O}",
        f"alpha_{PLATFORM_B}", f"alpha_{PLATFORM_O}",
        f"kappa_{PLATFORM_B}", f"kappa_{PLATFORM_O}",
        f"tau_{PLATFORM_B}", f"tau_{PLATFORM_O}",
        f"price_{PLATFORM_B}", f"price_{PLATFORM_O}",
        f"log_price_{PLATFORM_B}", f"log_price_{PLATFORM_O}",
        f"is_mint_platform_proxy_{PLATFORM_B}", f"is_mint_platform_proxy_{PLATFORM_O}",
        f"switch_share_volume_ct_{PLATFORM_B}", f"switch_share_volume_ct_{PLATFORM_O}",
    ]
    wide = wide.dropna(subset=needed).copy()

    wide["alpha_c"] = wide[f"alpha_{PLATFORM_B}"]

    wide["sB"] = wide[f"share_volume_{PLATFORM_B}"]
    wide["sO"] = wide[f"share_volume_{PLATFORM_O}"]
    wide["share_gap"] = wide["sB"] - wide["sO"]

    wide["kappa_B"] = wide[f"kappa_{PLATFORM_B}"]
    wide["kappa_O"] = wide[f"kappa_{PLATFORM_O}"]
    wide["kappa_diff"] = wide["kappa_B"] - wide["kappa_O"]

    wide["tau_B"] = wide[f"tau_{PLATFORM_B}"]
    wide["tau_O"] = wide[f"tau_{PLATFORM_O}"]
    wide["tau_diff"] = wide["tau_B"] - wide["tau_O"]

    wide["pB"] = wide[f"price_{PLATFORM_B}"]
    wide["pO"] = wide[f"price_{PLATFORM_O}"]
    wide["lpB"] = wide[f"log_price_{PLATFORM_B}"]
    wide["lpO"] = wide[f"log_price_{PLATFORM_O}"]
    wide["z_lpB"] = zscore(wide["lpB"].to_numpy())
    wide["z_lpO"] = zscore(wide["lpO"].to_numpy())

    wide["B_is_mint"] = wide[f"is_mint_platform_proxy_{PLATFORM_B}"].astype(int)
    wide["switch_obs"] = wide[f"switch_share_volume_ct_{PLATFORM_B}"]

    # subgroup flags
    price_ref = 0.5 * (wide["lpB"] + wide["lpO"])
    med_price = float(np.nanmedian(price_ref))
    wide["hi_price"] = (price_ref >= med_price).astype(int)

    wide["tau_parity"] = (np.abs(wide["tau_diff"]) <= TAU_PARITY_TOL).astype(int)
    wide["kappa_parity"] = (np.abs(wide["kappa_diff"]) <= KAPPA_PARITY_TOL).astype(int)

    keep = np.isfinite(
        wide[
            [
                "sB", "sO", "share_gap",
                "alpha_c",
                "kappa_B", "kappa_O", "kappa_diff",
                "tau_B", "tau_O", "tau_diff",
                "pB", "pO", "lpB", "lpO", "z_lpB", "z_lpO",
                "switch_obs", "B_is_mint"
            ]
        ]
    ).all(axis=1)

    return wide.loc[keep].copy()


# ----------------------------
# MODEL
# ----------------------------
@dataclass
class Params:
    delta_B: float
    lambda_S: float
    phi_K: float
    k0: float
    k1: float
    gamma_S: float


def unpack_theta(theta: np.ndarray) -> Params:
    return Params(
        delta_B=clip(theta[0], -6.0, 6.0),
        lambda_S=float(np.exp(np.clip(theta[1], -8.0, 5.0))),   # positive
        phi_K=clip(theta[2], -8.0, 8.0),
        k0=float(np.exp(np.clip(theta[3], -8.0, 5.0))),         # positive
        k1=clip(theta[4], -5.0, 5.0),
        gamma_S=inv_logit(theta[5]),
    )


def predict_shares(wide: pd.DataFrame, par: Params) -> Tuple[np.ndarray, np.ndarray]:
    tau_B = wide["tau_B"].to_numpy()
    tau_O = wide["tau_O"].to_numpy()
    kappa_diff = wide["kappa_diff"].to_numpy()

    z_lpB = wide["z_lpB"].to_numpy()
    z_lpO = wide["z_lpO"].to_numpy()
    B_is_mint = wide["B_is_mint"].to_numpy().astype(int)

    # price-dependent off-mint switching cost using standardized log price
    off_B = par.k0 + par.k1 * z_lpB
    off_O = par.k0 + par.k1 * z_lpO

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
# MOMENTS
# ----------------------------
def _subset_mean(arr: np.ndarray, mask: np.ndarray) -> float:
    arr = np.asarray(arr, dtype=float)
    mask = np.asarray(mask, dtype=bool)
    m = mask & np.isfinite(arr)
    if m.sum() == 0:
        return 0.0
    return float(np.mean(arr[m]))

def compute_data_moments(wide: pd.DataFrame) -> Dict[str, np.ndarray]:
    share_gap = wide["share_gap"].to_numpy()
    switch_obs = wide["switch_obs"].to_numpy()
    alpha_c = wide["alpha_c"].to_numpy()
    kappa_diff = wide["kappa_diff"].to_numpy()
    tau_diff = wide["tau_diff"].to_numpy()

    share_gap_alpha, n_alpha = _binned_means(alpha_c, share_gap, ALPHA_BINS)
    share_gap_kappa, n_kappa = _binned_means(kappa_diff, share_gap, KAPPA_BINS)
    share_gap_tau, n_tau = _binned_means(tau_diff, share_gap, TAU_DIFF_BINS)
    switch_tau, _ = _binned_means(tau_diff, switch_obs, TAU_DIFF_BINS)

    mint = wide["B_is_mint"].to_numpy().astype(bool)
    nonmint = ~mint
    hi_price = wide["hi_price"].to_numpy().astype(bool)
    lo_price = ~hi_price
    tau_parity = wide["tau_parity"].to_numpy().astype(bool)
    kappa_parity = wide["kappa_parity"].to_numpy().astype(bool)

    return {
        "share_gap_alpha": share_gap_alpha,
        "count_alpha": n_alpha,

        "share_gap_kappa_diff": share_gap_kappa,
        "count_kappa_diff": n_kappa,

        "share_gap_tau_diff": share_gap_tau,
        "switch_tau_diff": switch_tau,
        "count_tau_diff": n_tau,

        "mean_share_B": np.array([float(np.mean(wide["sB"]))]),
        "mean_switch": np.array([float(np.mean(switch_obs))]),

        "share_mint": np.array([_subset_mean(wide["sB"].to_numpy(), mint)]),
        "share_nonmint": np.array([_subset_mean(wide["sB"].to_numpy(), nonmint)]),
        "switch_mint": np.array([_subset_mean(switch_obs, mint)]),
        "switch_nonmint": np.array([_subset_mean(switch_obs, nonmint)]),

        "share_hi_price": np.array([_subset_mean(wide["sB"].to_numpy(), hi_price)]),
        "share_lo_price": np.array([_subset_mean(wide["sB"].to_numpy(), lo_price)]),
        "switch_hi_price": np.array([_subset_mean(switch_obs, hi_price)]),
        "switch_lo_price": np.array([_subset_mean(switch_obs, lo_price)]),

        "share_tau_parity": np.array([_subset_mean(wide["sB"].to_numpy(), tau_parity)]),
        "share_kappa_parity": np.array([_subset_mean(wide["sB"].to_numpy(), kappa_parity)]),
        "switch_tau_parity": np.array([_subset_mean(switch_obs, tau_parity)]),

        "slope_share_tau": np.array([_safe_slope(tau_diff, share_gap)]),
        "slope_switch_tau": np.array([_safe_slope(tau_diff, switch_obs)]),
        "slope_share_kappa": np.array([_safe_slope(kappa_diff, share_gap)]),
    }


def compute_model_moments(wide: pd.DataFrame, par: Params) -> Dict[str, np.ndarray]:
    sB_hat, switch_hat = predict_shares(wide, par)
    share_gap_hat = 2.0 * sB_hat - 1.0

    alpha_c = wide["alpha_c"].to_numpy()
    kappa_diff = wide["kappa_diff"].to_numpy()
    tau_diff = wide["tau_diff"].to_numpy()

    share_gap_alpha, n_alpha = _binned_means(alpha_c, share_gap_hat, ALPHA_BINS)
    share_gap_kappa, n_kappa = _binned_means(kappa_diff, share_gap_hat, KAPPA_BINS)
    share_gap_tau, n_tau = _binned_means(tau_diff, share_gap_hat, TAU_DIFF_BINS)
    switch_tau, _ = _binned_means(tau_diff, switch_hat, TAU_DIFF_BINS)

    mint = wide["B_is_mint"].to_numpy().astype(bool)
    nonmint = ~mint
    hi_price = wide["hi_price"].to_numpy().astype(bool)
    lo_price = ~hi_price
    tau_parity = wide["tau_parity"].to_numpy().astype(bool)
    kappa_parity = wide["kappa_parity"].to_numpy().astype(bool)

    return {
        "share_gap_alpha": share_gap_alpha,
        "count_alpha": n_alpha,

        "share_gap_kappa_diff": share_gap_kappa,
        "count_kappa_diff": n_kappa,

        "share_gap_tau_diff": share_gap_tau,
        "switch_tau_diff": switch_tau,
        "count_tau_diff": n_tau,

        "mean_share_B": np.array([float(np.mean(sB_hat))]),
        "mean_switch": np.array([float(np.mean(switch_hat))]),

        "share_mint": np.array([_subset_mean(sB_hat, mint)]),
        "share_nonmint": np.array([_subset_mean(sB_hat, nonmint)]),
        "switch_mint": np.array([_subset_mean(switch_hat, mint)]),
        "switch_nonmint": np.array([_subset_mean(switch_hat, nonmint)]),

        "share_hi_price": np.array([_subset_mean(sB_hat, hi_price)]),
        "share_lo_price": np.array([_subset_mean(sB_hat, lo_price)]),
        "switch_hi_price": np.array([_subset_mean(switch_hat, hi_price)]),
        "switch_lo_price": np.array([_subset_mean(switch_hat, lo_price)]),

        "share_tau_parity": np.array([_subset_mean(sB_hat, tau_parity)]),
        "share_kappa_parity": np.array([_subset_mean(sB_hat, kappa_parity)]),
        "switch_tau_parity": np.array([_subset_mean(switch_hat, tau_parity)]),

        "slope_share_tau": np.array([_safe_slope(tau_diff, share_gap_hat)]),
        "slope_switch_tau": np.array([_safe_slope(tau_diff, switch_hat)]),
        "slope_share_kappa": np.array([_safe_slope(kappa_diff, share_gap_hat)]),
    }


def moment_residuals(m_data: Dict[str, np.ndarray], m_mod: Dict[str, np.ndarray]) -> np.ndarray:
    out = []

    def add_binned(name: str, count_name: str, weight: float):
        counts = m_data[count_name]
        use = counts >= MIN_BIN_COUNT
        if np.any(use):
            w = np.sqrt(np.clip(counts[use], 1.0, None))
            out.append(weight * (m_mod[name][use] - m_data[name][use]) * w)

    add_binned("share_gap_alpha", "count_alpha", W_MOMENTS["share_gap_alpha"])
    add_binned("share_gap_kappa_diff", "count_kappa_diff", W_MOMENTS["share_gap_kappa_diff"])
    add_binned("share_gap_tau_diff", "count_tau_diff", W_MOMENTS["share_gap_tau_diff"])
    add_binned("switch_tau_diff", "count_tau_diff", W_MOMENTS["switch_tau_diff"])

    scalar_names = [
        "mean_share_B", "mean_switch",
        "share_mint", "share_nonmint", "switch_mint", "switch_nonmint",
        "share_hi_price", "share_lo_price", "switch_hi_price", "switch_lo_price",
        "share_tau_parity", "share_kappa_parity", "switch_tau_parity",
        "slope_share_tau", "slope_switch_tau", "slope_share_kappa",
    ]

    for name in scalar_names:
        out.append(W_MOMENTS[name] * (m_mod[name] - m_data[name]))

    return np.concatenate([x.reshape(-1) for x in out])


def objective(theta: np.ndarray, wide: pd.DataFrame, m_data: Dict[str, np.ndarray]) -> float:
    par = unpack_theta(theta)
    m_mod = compute_model_moments(wide, par)
    r = moment_residuals(m_data, m_mod)

    if not np.isfinite(r).all():
        return 1e12

    loss = float(np.dot(r, r))
    loss += RIDGE * float(np.dot(theta, theta))

    # mild regularization on boundary-prone channels
    #loss += REG_PHI * (par.phi_K ** 2)
    #loss += REG_K1 * (par.k1 ** 2)

    return loss


# ----------------------------
# RUN
# ----------------------------
def run(path_csv: str, outdir: str = "paper_match_estimates_v2") -> None:
    os.makedirs(outdir, exist_ok=True)

    wide = load_and_pivot(path_csv)
    m_data = compute_data_moments(wide)

    theta0 = np.array([
        0.0,               # delta_B
        math.log(0.5),     # lambda_S > 0
        0.5,               # phi_K
        math.log(0.25),    # k0 > 0
        0.0,               # k1 on z(log price)
        safe_logit(0.65),  # gamma_S
    ], dtype=float)

    bounds = [
        (-6, 6),   # delta_B
        (-8, 5),   # log lambda_S
        (-8, 8),   # phi_K
        (-8, 5),   # log k0
        (-5, 5),   # k1
        (-8, 8),   # gamma_S logit
    ]

    res = minimize(
        objective,
        theta0,
        args=(wide, m_data),
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 4000, "ftol": 1e-10},
    )

    par = unpack_theta(res.x)
    sB_hat, switch_hat = predict_shares(wide, par)

    out = wide.copy()
    out["share_B_hat"] = sB_hat
    out["switch_hat"] = switch_hat
    out.to_csv(os.path.join(outdir, "fitted_markets.csv"), index=False)

    pd.DataFrame([{
        "delta_B": par.delta_B,
        "lambda_S": par.lambda_S,
        "phi_K": par.phi_K,
        "k0": par.k0,
        "k1": par.k1,
        "gamma_S": par.gamma_S,
        "objective": float(res.fun),
        "success": bool(res.success),
        "message": str(res.message),
    }]).to_csv(os.path.join(outdir, "params.csv"), index=False)

    # optional: save model/data moments
    m_mod = compute_model_moments(wide, par)
    rows = []
    for k in sorted(m_data.keys()):
        rows.append({
            "moment": k,
            "data": np.array2string(m_data[k], precision=6, separator=","),
            "model": np.array2string(m_mod[k], precision=6, separator=","),
        })
    pd.DataFrame(rows).to_csv(os.path.join(outdir, "moment_comparison.csv"), index=False)

    print("=== ESTIMATES ===")
    print(f"delta_B:  {par.delta_B:.6f}")
    print(f"lambda_S: {par.lambda_S:.6f}")
    print(f"phi_K:    {par.phi_K:.6f}")
    print(f"k0:       {par.k0:.6f}")
    print(f"k1:       {par.k1:.6f}")
    print(f"gamma_S:  {par.gamma_S:.6f}")
    print(f"Objective: {res.fun:.6f}")
    print(f"Converged: {res.success} | {res.message}")
    print(f"Wrote outputs to: {outdir}/")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python estimate_paper_match_v2.py input.csv")
        sys.exit(1)
    run(sys.argv[1])