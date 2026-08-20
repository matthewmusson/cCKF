#!/usr/bin/env python3
"""Analyze cCKF gate diagnostic traces.

Reads the _traces.csv produced by CckfTrackFindingAlgorithm and the gate
weight blob's standardization params.  Produces three diagnostic outputs:

1. Feature distribution match: empirical vs training mean/std for all 26 features
2. Gate quality: accepted chi2 vs min-chi2 on surface at each step
3. Compound error: candidate chi2 distribution evolution over steps
"""
import argparse
import struct
import sys
from pathlib import Path

import numpy as np
import pandas as pd


FEATURE_NAMES = [
    "res0", "res1", "chol_S_00", "chol_S_10", "chol_S_11", "chi2_feat",
    "clus_s_u", "clus_s_v", "clus_q_tot",
    "clus_sigma_uu", "clus_sigma_uv", "clus_sigma_vv",
    "kappa_u", "kappa_v", "q_tilde", "n_window",
    "eta", "qop", "step_k_feat", "pathInX0",
    "pitch_u", "pitch_v", "thickness",
    "n_hits", "n_holes", "n_seq_holes",
]


def load_blob_stats(blob_path: str) -> tuple[np.ndarray, np.ndarray]:
    """Read standardization mean and std from a CCKF weight blob."""
    with open(blob_path, "rb") as f:
        magic = f.read(4)
        assert magic == b"CCKF", f"Bad magic: {magic}"
        version = struct.unpack("<I", f.read(4))[0]
        assert version == 1
        n_features = struct.unpack("<I", f.read(4))[0]
        _n_hidden = struct.unpack("<I", f.read(4))[0]
        _n_layers = struct.unpack("<I", f.read(4))[0]
        mu = np.frombuffer(f.read(n_features * 4), dtype=np.float32).copy()
        sigma = np.frombuffer(f.read(n_features * 4), dtype=np.float32).copy()
    return mu, sigma


def analysis_1_feature_distribution(df: pd.DataFrame, mu_train: np.ndarray,
                                     sigma_train: np.ndarray):
    """Compare empirical feature distribution against training stats."""
    print("\n" + "=" * 70)
    print("ANALYSIS 1: Feature Distribution Match")
    print("=" * 70)
    print("Do the raw features the C++ gate sees match the training distribution?")
    print("Ratio ~ 1.0 means match. >> 1 or << 1 means mismatch.\n")

    feat_cols = FEATURE_NAMES
    X = df[feat_cols].values

    print(f"{'idx':>3} {'feature':<16} {'emp_mean':>10} {'trn_mean':>10} "
          f"{'emp_std':>10} {'trn_std':>10} {'std_ratio':>10} {'status':<8}")
    print("-" * 86)

    for j, name in enumerate(feat_cols):
        emp_mean = np.nanmean(X[:, j])
        emp_std = np.nanstd(X[:, j])
        t_mean = mu_train[j]
        t_std = sigma_train[j]
        ratio = emp_std / t_std if t_std > 1e-12 else float("inf")
        status = "OK" if 0.1 < ratio < 10 else "MISMATCH"
        print(f"{j:3d} {name:<16} {emp_mean:10.4f} {t_mean:10.4f} "
              f"{emp_std:10.4f} {t_std:10.4f} {ratio:10.3f} {status:<8}")

    print(f"\nZ-score audit (z = (x - mu_train) / sigma_train):")
    print(f"{'idx':>3} {'feature':<16} {'|z|>3':>8} {'|z|>5':>8} {'|z|>10':>8} {'median_z':>10}")
    print("-" * 60)
    for j, name in enumerate(feat_cols):
        if sigma_train[j] < 1e-12:
            print(f"{j:3d} {name:<16} {'N/A':>8} {'N/A':>8} {'N/A':>8} {'N/A':>10}")
            continue
        z = (X[:, j] - mu_train[j]) / sigma_train[j]
        z_finite = z[np.isfinite(z)]
        if len(z_finite) == 0:
            continue
        frac3 = np.mean(np.abs(z_finite) > 3)
        frac5 = np.mean(np.abs(z_finite) > 5)
        frac10 = np.mean(np.abs(z_finite) > 10)
        med_z = np.median(z_finite)
        print(f"{j:3d} {name:<16} {frac3:8.3f} {frac5:8.3f} {frac10:8.3f} {med_z:10.3f}")


def analysis_2_gate_quality(df: pd.DataFrame):
    """Compare gate's accepted chi2 against classical best (min chi2)."""
    print("\n" + "=" * 70)
    print("ANALYSIS 2: Gate Quality vs Classical Selector")
    print("=" * 70)
    print("Does the gate choose good candidates? Compare accepted chi2 vs min chi2.\n")

    grouped = df.groupby(["seed", "step_k"])

    records = []
    for (seed, step_k), g in grouped:
        min_chi2 = g["chi2"].min()
        acc = g[g["accepted"] == 1]
        if len(acc) == 0:
            continue
        best_acc_chi2 = acc["chi2"].min()
        records.append({
            "seed": seed, "step_k": step_k,
            "min_chi2": min_chi2, "acc_chi2": best_acc_chi2,
            "ratio": best_acc_chi2 / max(min_chi2, 1e-6),
            "n_cands": g["n_cands_total"].iloc[0],
            "gate_chose_best": int(abs(best_acc_chi2 - min_chi2) < 0.01),
        })

    if not records:
        print("No accepted candidates found in traces.")
        return

    stats = pd.DataFrame(records)

    print(f"Per-step summary (grouped by step_k):\n")
    print(f"{'step':>5} {'n_obs':>6} {'mean_ratio':>11} {'med_ratio':>10} "
          f"{'chose_best%':>12} {'mean_acc_chi2':>14} {'mean_min_chi2':>14}")
    print("-" * 80)

    for step_k in sorted(stats["step_k"].unique()):
        s = stats[stats["step_k"] == step_k]
        print(f"{step_k:5d} {len(s):6d} {s['ratio'].mean():11.2f} "
              f"{s['ratio'].median():10.2f} "
              f"{100 * s['gate_chose_best'].mean():12.1f} "
              f"{s['acc_chi2'].mean():14.2f} {s['min_chi2'].mean():14.2f}")

    print(f"\nOverall: gate chose classical-best in "
          f"{100 * stats['gate_chose_best'].mean():.1f}% of steps")
    print(f"Overall mean accepted/min ratio: {stats['ratio'].mean():.2f}")


def analysis_3_compound_error(df: pd.DataFrame):
    """Check whether candidate chi2 distributions drift with step depth."""
    print("\n" + "=" * 70)
    print("ANALYSIS 3: Compound Error Detection")
    print("=" * 70)
    print("Does the chi2 distribution worsen with track depth?")
    print("Growing median chi2 across ALL candidates = predicted state is lost.\n")

    print(f"{'step':>5} {'n_cands':>8} {'min_chi2':>9} {'p25_chi2':>9} "
          f"{'med_chi2':>9} {'p75_chi2':>9} {'max_chi2':>9} {'mean_chi2':>10}")
    print("-" * 75)

    for step_k in sorted(df["step_k"].unique()):
        s = df[df["step_k"] == step_k]
        chi2 = s["chi2"].values
        print(f"{step_k:5d} {len(chi2):8d} {np.min(chi2):9.2f} "
              f"{np.percentile(chi2, 25):9.2f} {np.median(chi2):9.2f} "
              f"{np.percentile(chi2, 75):9.2f} {np.max(chi2):9.2f} "
              f"{np.mean(chi2):10.2f}")

    acc = df[df["accepted"] == 1]
    print(f"\nAccepted candidates only:")
    print(f"{'step':>5} {'n_acc':>8} {'med_chi2':>9} {'mean_chi2':>10}")
    print("-" * 35)
    for step_k in sorted(acc["step_k"].unique()):
        s = acc[acc["step_k"] == step_k]
        chi2 = s["chi2"].values
        print(f"{step_k:5d} {len(chi2):8d} {np.median(chi2):9.2f} "
              f"{np.mean(chi2):10.2f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("traces_csv", help="Path to _traces.csv")
    parser.add_argument("gate_blob", help="Path to gate.bin weight blob")
    args = parser.parse_args()

    mu_train, sigma_train = load_blob_stats(args.gate_blob)
    assert len(mu_train) == 26, f"Expected 26 features, got {len(mu_train)}"

    df = pd.read_csv(args.traces_csv)
    print(f"Loaded {len(df)} candidate rows from {args.traces_csv}")
    print(f"  Seeds traced: {df['seed'].nunique()}")
    print(f"  Steps per seed: {df.groupby('seed')['step_k'].nunique().describe()}")
    print(f"  Accepted fraction: {df['accepted'].mean():.3f}")

    analysis_1_feature_distribution(df, mu_train, sigma_train)
    analysis_2_gate_quality(df)
    analysis_3_compound_error(df)


if __name__ == "__main__":
    main()
