# preprocess_msm_data_paired.py
#
# Cleans MSM_data_clean_week2023-04-03.csv for paired Blur/OpenSea estimation.
#
# IMPORTANT:
#   This version enforces pair-level survival:
#   if one platform row fails cleaning for a collection-week market,
#   the entire market is dropped from the final sample.
#
# Usage:
#   python preprocess_msm_data_paired.py input.csv output.csv
#
# Optional presets:
#   python preprocess_msm_data_paired.py input.csv output.csv --preset light
#   python preprocess_msm_data_paired.py input.csv output.csv --preset medium
#   python preprocess_msm_data_paired.py input.csv output.csv --preset strong
#
# Optional outputs:
#   --summary_csv path.csv
#   --summary_json path.json

import argparse
import json
import os
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


START_WEEK = pd.Timestamp("2023-04-03", tz="UTC")
PLATFORM_A = "Blur"
PLATFORM_B = "OpenSea"
PAIR_PLATFORMS = {PLATFORM_A, PLATFORM_B}

REQUIRED_COLUMNS = [
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
    "trades",
    "total_secondary_volume_eth",
    "share_trades",
    "switch_share_trades_ct",
    "ct_trades_total",
    "ct_volume_total_eth",
]

NUMERIC_COLUMNS = [
    "share_volume",
    "platform_fee_rate_pct",
    "royalty_rate_pct",
    "enforcement_credibility",
    "avg_trade_price_eth",
    "is_mint_platform_proxy",
    "switch_share_volume_ct",
    "trades",
    "total_secondary_volume_eth",
    "share_trades",
    "switch_share_trades_ct",
    "ct_trades_total",
    "ct_volume_total_eth",
]

SHARE_COLUMNS = [
    "share_volume",
    "share_trades",
    "switch_share_volume_ct",
    "switch_share_trades_ct",
]

WINSOR_COLUMNS = [
    "avg_trade_price_eth",
    "total_secondary_volume_eth",
    "ct_volume_total_eth",
    "trades",
    "ct_trades_total",
]


@dataclass
class CleaningPreset:
    name: str
    min_ct_trades_total: int
    min_ct_volume_total_eth: float
    min_platform_trades: int
    min_platform_volume_eth: float
    min_royalty_rate_pct: float
    max_royalty_rate_pct: float
    winsor_q_low: float
    winsor_q_high: float
    share_floor: float
    enforcement_cap_low: float
    enforcement_cap_high: float


PRESETS: Dict[str, CleaningPreset] = {
    "light": CleaningPreset(
        name="light",
        min_ct_trades_total=5,
        min_ct_volume_total_eth=1.0,
        min_platform_trades=1,
        min_platform_volume_eth=0.05,
        min_royalty_rate_pct=0.0,
        max_royalty_rate_pct=12.5,
        winsor_q_low=0.001,
        winsor_q_high=0.999,
        share_floor=1e-6,
        enforcement_cap_low=0.0,
        enforcement_cap_high=1.0,
    ),
    "medium": CleaningPreset(
        name="medium",
        min_ct_trades_total=10,
        min_ct_volume_total_eth=2.0,
        min_platform_trades=2,
        min_platform_volume_eth=0.10,
        min_royalty_rate_pct=0.5,
        max_royalty_rate_pct=10.5,
        winsor_q_low=0.005,
        winsor_q_high=0.995,
        share_floor=1e-6,
        enforcement_cap_low=0.0,
        enforcement_cap_high=1.0,
    ),
    "strong": CleaningPreset(
        name="strong",
        min_ct_trades_total=20,
        min_ct_volume_total_eth=5.0,
        min_platform_trades=3,
        min_platform_volume_eth=0.25,
        min_royalty_rate_pct=1.0,
        max_royalty_rate_pct=10.0,
        winsor_q_low=0.01,
        winsor_q_high=0.99,
        share_floor=1e-6,
        enforcement_cap_low=0.0,
        enforcement_cap_high=1.0,
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_csv", type=str)
    parser.add_argument("output_csv", type=str)
    parser.add_argument(
        "--preset",
        type=str,
        choices=sorted(PRESETS.keys()),
        default="medium",
        help="Cleaning aggressiveness preset.",
    )
    parser.add_argument(
        "--summary_csv",
        type=str,
        default=None,
        help="Optional path to write row-count summary CSV.",
    )
    parser.add_argument(
        "--summary_json",
        type=str,
        default=None,
        help="Optional path to write cleaning metadata JSON.",
    )
    return parser.parse_args()


def require_columns(df: pd.DataFrame, required: List[str]) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def coerce_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def winsorize_series(s: pd.Series, q_low: float, q_high: float) -> Tuple[pd.Series, float, float]:
    lo = float(s.quantile(q_low))
    hi = float(s.quantile(q_high))
    return s.clip(lo, hi), lo, hi


def add_summary_row(rows: List[Dict], step: str, before_n: int, after_n: int) -> None:
    rows.append(
        {
            "step": step,
            "before_rows": int(before_n),
            "after_rows": int(after_n),
            "dropped_rows": int(before_n - after_n),
        }
    )


def ensure_market_id(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "market_id" not in out.columns:
        out["market_id"] = out["collection_address"].astype(str) + "||" + out["week"].astype(str)
    return out


def paired_market_filter(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """
    Keep only market_ids where surviving rows contain exactly Blur and OpenSea, one each.
    If either platform row fails, the whole market is dropped.
    """
    out = df.copy()

    counts = (
        out.groupby("market_id")
        .agg(
            n_rows=("platform", "size"),
            n_platforms=("platform", "nunique"),
        )
        .reset_index()
    )

    platform_sets = (
        out.groupby("market_id")["platform"]
        .apply(lambda s: tuple(sorted(set(map(str, s)))))
        .reset_index(name="platform_set")
    )

    merged = counts.merge(platform_sets, on="market_id", how="left")

    good_ids = merged.loc[
        (merged["n_rows"] == 2)
        & (merged["n_platforms"] == 2)
        & (merged["platform_set"] == tuple(sorted(PAIR_PLATFORMS))),
        "market_id",
    ]

    dropped_market_ids = set(merged["market_id"]) - set(good_ids)

    meta = {
        "n_markets_total_pre_pair_filter": int(len(merged)),
        "n_markets_kept_post_pair_filter": int(len(good_ids)),
        "n_markets_dropped_post_pair_filter": int(len(dropped_market_ids)),
    }

    out = out[out["market_id"].isin(good_ids)].copy()
    return out, meta


def preprocess(df: pd.DataFrame, preset: CleaningPreset) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    summary_rows: List[Dict] = []
    meta: Dict = {
        "preset": asdict(preset),
        "winsor_bounds": {},
    }

    before = len(df)
    require_columns(df, REQUIRED_COLUMNS)
    add_summary_row(summary_rows, "require_columns", before, before)

    out = df.copy()

    # Parse week
    before = len(out)
    out["week"] = pd.to_datetime(out["week"], utc=True, errors="coerce")
    out = out.dropna(subset=["week"])
    add_summary_row(summary_rows, "parse_week_dropna", before, len(out))

    # Restrict time window
    before = len(out)
    out = out[out["week"] >= START_WEEK].copy()
    add_summary_row(summary_rows, "restrict_start_week", before, len(out))

    # Keep only target platforms up front
    before = len(out)
    out = out[out["platform"].isin([PLATFORM_A, PLATFORM_B])].copy()
    add_summary_row(summary_rows, "restrict_target_platforms", before, len(out))

    # Add market_id
    before = len(out)
    out = ensure_market_id(out)
    add_summary_row(summary_rows, "ensure_market_id", before, len(out))

    # Numeric conversion
    out = coerce_numeric(out, NUMERIC_COLUMNS)

    # Drop missing / non-finite on required numerics
    before = len(out)
    out = out.dropna(subset=NUMERIC_COLUMNS)
    out = out[np.isfinite(out[NUMERIC_COLUMNS]).all(axis=1)].copy()
    add_summary_row(summary_rows, "drop_missing_nonfinite_numeric", before, len(out))

    # Thin collection-week restriction
    before = len(out)
    out = out[out["ct_trades_total"] >= preset.min_ct_trades_total].copy()
    out = out[out["ct_volume_total_eth"] >= preset.min_ct_volume_total_eth].copy()
    add_summary_row(summary_rows, "drop_thin_collection_weeks", before, len(out))

    # Thin platform activity restriction
    before = len(out)
    out = out[out["trades"] >= preset.min_platform_trades].copy()
    out = out[out["total_secondary_volume_eth"] >= preset.min_platform_volume_eth].copy()
    add_summary_row(summary_rows, "drop_thin_platform_activity", before, len(out))

    # Plausible royalty range
    before = len(out)
    out = out[
        (out["royalty_rate_pct"] >= preset.min_royalty_rate_pct)
        & (out["royalty_rate_pct"] <= preset.max_royalty_rate_pct)
    ].copy()
    add_summary_row(summary_rows, "restrict_royalty_rate_range", before, len(out))

    # Bound enforcement credibility
    before = len(out)
    out["enforcement_credibility_raw"] = out["enforcement_credibility"]
    out["enforcement_credibility"] = out["enforcement_credibility"].clip(
        preset.enforcement_cap_low, preset.enforcement_cap_high
    )
    add_summary_row(summary_rows, "clip_enforcement_credibility", before, len(out))

    # Positive price / volume guards
    before = len(out)
    out = out[out["avg_trade_price_eth"] > 0].copy()
    out = out[out["ct_volume_total_eth"] > 0].copy()
    out = out[out["total_secondary_volume_eth"] > 0].copy()
    add_summary_row(summary_rows, "drop_nonpositive_price_volume", before, len(out))

    # Winsorize heavy tails
    for c in WINSOR_COLUMNS:
        before = len(out)
        out[c], lo, hi = winsorize_series(out[c], preset.winsor_q_low, preset.winsor_q_high)
        meta["winsor_bounds"][c] = {"low": lo, "high": hi}
        add_summary_row(summary_rows, f"winsorize_{c}", before, len(out))

    # Bound shares away from 0/1
    before = len(out)
    lo = preset.share_floor
    hi = 1.0 - preset.share_floor
    for c in SHARE_COLUMNS:
        out[c] = out[c].clip(lo, hi)
    add_summary_row(summary_rows, "clip_share_columns", before, len(out))

    # Recompute effective total cost for consistency, if present
    if "effective_total_cost_rate_pct" in out.columns:
        before = len(out)
        out["effective_total_cost_rate_pct"] = (
            out["platform_fee_rate_pct"] + out["royalty_rate_pct"] * out["enforcement_credibility"]
        )
        add_summary_row(summary_rows, "recompute_effective_total_cost_rate_pct", before, len(out))

    # PAIR-LEVEL ENFORCEMENT:
    # if either platform row fails, drop the entire market
    before = len(out)
    out, pair_meta = paired_market_filter(out)
    add_summary_row(summary_rows, "pair_level_keep_only_complete_markets", before, len(out))
    meta.update(pair_meta)

    # Useful derived diagnostics
    out["log_avg_trade_price_eth"] = np.log(out["avg_trade_price_eth"])

    summary_df = pd.DataFrame(summary_rows)

    meta["final_row_count"] = int(len(out))
    meta["final_market_count"] = int(out["market_id"].nunique()) if "market_id" in out.columns else None
    meta["n_unique_collections"] = int(out["collection_address"].nunique()) if "collection_address" in out.columns else None
    meta["n_unique_weeks"] = int(out["week"].nunique()) if "week" in out.columns else None
    meta["platform_counts"] = (
        out["platform"].value_counts(dropna=False).to_dict() if "platform" in out.columns else {}
    )

    # Audit stats
    audit_cols = [
        "avg_trade_price_eth",
        "ct_volume_total_eth",
        "total_secondary_volume_eth",
        "trades",
        "ct_trades_total",
        "royalty_rate_pct",
        "enforcement_credibility",
        "share_volume",
        "switch_share_volume_ct",
    ]
    meta["audit_stats"] = {}
    for c in audit_cols:
        if c in out.columns:
            s = out[c]
            meta["audit_stats"][c] = {
                "min": float(np.nanmin(s)),
                "p1": float(np.nanquantile(s, 0.01)),
                "p50": float(np.nanquantile(s, 0.50)),
                "p99": float(np.nanquantile(s, 0.99)),
                "max": float(np.nanmax(s)),
            }

    return out, summary_df, meta


def main() -> None:
    args = parse_args()
    preset = PRESETS[args.preset]

    df = pd.read_csv(args.input_csv)
    cleaned_df, summary_df, meta = preprocess(df, preset)

    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
    cleaned_df.to_csv(args.output_csv, index=False)

    summary_csv = args.summary_csv
    if summary_csv is None:
        root, _ = os.path.splitext(args.output_csv)
        summary_csv = f"{root}_cleaning_summary.csv"
    os.makedirs(os.path.dirname(summary_csv) or ".", exist_ok=True)
    summary_df.to_csv(summary_csv, index=False)

    summary_json = args.summary_json
    if summary_json is None:
        root, _ = os.path.splitext(args.output_csv)
        summary_json = f"{root}_cleaning_metadata.json"
    os.makedirs(os.path.dirname(summary_json) or ".", exist_ok=True)
    with open(summary_json, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Wrote cleaned paired data to: {args.output_csv}")
    print(f"Wrote cleaning summary to: {summary_csv}")
    print(f"Wrote cleaning metadata to: {summary_json}")
    print(f"Final rows: {len(cleaned_df):,}")
    print(f"Final paired markets: {cleaned_df['market_id'].nunique():,}")
    if len(cleaned_df) > 0:
        print(f"Rows per market should be 2 exactly. Check: {cleaned_df.groupby('market_id').size().value_counts().to_dict()}")


if __name__ == "__main__":
    main()