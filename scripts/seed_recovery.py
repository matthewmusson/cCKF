"""Pilot check 2 — Tight seed recovery under offline filtering of envelope (§6.6.2).

Identity: rounded (b, m, t) space-point triplet from ACTS ``seed.csv``.
Offline filter of envelope seeds toward Tight:
  * pT >= Tight seed_minPt
  * keep top ``num_seeds_per_spm`` by quality per middle-SP bucket

impactMax / sigmaScattering reshape the finder itself and cannot be fully
applied offline — recovery is therefore an upper bound on true subset
recoverability. Threshold ~98% per spec; report and stop if below.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


TRIPLET_COLS = ["bX", "bY", "bZ", "mX", "mY", "mZ", "tX", "tY", "tZ"]


def _load_seeds(seed_dir: Path, event: int) -> pd.DataFrame | None:
    # CsvSeedWriter fileName defaults to "seed.csv" → eventNNNNNNNNN-seed.csv
    candidates = list(seed_dir.glob(f"event*-seed.csv")) + list(
        seed_dir.glob(f"event*-seeds.csv")
    )
    for c in sorted(candidates):
        stem = c.name.split("-")[0]
        try:
            ev = int(stem.replace("event", ""))
        except ValueError:
            continue
        if ev == event:
            df = pd.read_csv(c, comment="#")
            # Normalize column names from CsvSeedWriter header
            # seed_id,particleId,pT,eta,phi,bX,bY,bZ,mX,mY,mZ,tX,tY,tZ,quality,...
            rename = {}
            for col in df.columns:
                cl = col.strip()
                rename[col] = cl
            df = df.rename(columns=rename)
            if "pT" not in df.columns and "pt" in df.columns:
                df = df.rename(columns={"pt": "pT"})
            return df
    return None


def triplet_key(row: pd.Series, ndigits: int = 3) -> tuple:
    vals = []
    for c in TRIPLET_COLS:
        vals.append(round(float(row[c]), ndigits))
    return tuple(vals)


def middle_key(row: pd.Series, ndigits: int = 3) -> tuple:
    return tuple(round(float(row[c]), ndigits) for c in ("mX", "mY", "mZ"))


def offline_filter_to_tight(
    envelope: pd.DataFrame,
    *,
    min_pt: float,
    max_seeds_per_spm: int,
) -> pd.DataFrame:
    df = envelope.copy()
    df = df[df["pT"] >= min_pt]
    if "quality" not in df.columns:
        # Keep all that pass pT if quality missing
        return df.reset_index(drop=True)

    kept_idx: list[int] = []
    buckets: dict[tuple, list[int]] = defaultdict(list)
    for i, row in df.iterrows():
        buckets[middle_key(row)].append(i)
    for idxs in buckets.values():
        # Higher quality is better in ACTS seed filter
        ranked = sorted(
            idxs, key=lambda i: float(df.loc[i, "quality"]), reverse=True
        )
        kept_idx.extend(ranked[:max_seeds_per_spm])
    return df.loc[kept_idx].reset_index(drop=True)


def recovery_for_event(
    tight: pd.DataFrame,
    envelope: pd.DataFrame,
    *,
    min_pt: float,
    max_seeds_per_spm: int,
) -> dict[str, Any]:
    missing = [c for c in TRIPLET_COLS + ["pT"] if c not in tight.columns]
    missing_e = [c for c in TRIPLET_COLS + ["pT"] if c not in envelope.columns]
    if missing or missing_e:
        return {
            "pass": False,
            "reason": f"seed.csv missing columns tight={missing} env={missing_e}",
            "tight_columns": list(tight.columns),
            "envelope_columns": list(envelope.columns),
        }

    tight_keys = {triplet_key(r) for _, r in tight.iterrows()}
    env_keys = {triplet_key(r) for _, r in envelope.iterrows()}
    filtered = offline_filter_to_tight(
        envelope, min_pt=min_pt, max_seeds_per_spm=max_seeds_per_spm
    )
    filt_keys = {triplet_key(r) for _, r in filtered.iterrows()}

    recovered = tight_keys & filt_keys
    in_envelope = tight_keys & env_keys
    n_tight = len(tight_keys)
    frac_in_env = len(in_envelope) / n_tight if n_tight else float("nan")
    frac_recovered = len(recovered) / n_tight if n_tight else float("nan")

    return {
        "n_tight": int(n_tight),
        "n_envelope": int(len(env_keys)),
        "n_filtered": int(len(filt_keys)),
        "n_tight_in_envelope": int(len(in_envelope)),
        "n_recovered": int(len(recovered)),
        "frac_tight_in_envelope": float(frac_in_env),
        "frac_recovered": float(frac_recovered),
        "pass": bool(frac_recovered >= 0.98) if n_tight else False,
        "threshold": 0.98,
    }


def probe_seed_recovery(
    tight_dir: Path,
    envelope_dir: Path,
    events: list[int],
    *,
    min_pt: float,
    max_seeds_per_spm: int,
) -> dict[str, Any]:
    per_event = {}
    for ev in events:
        t = _load_seeds(Path(tight_dir), ev)
        e = _load_seeds(Path(envelope_dir), ev)
        if t is None or e is None:
            per_event[str(ev)] = {
                "pass": False,
                "reason": f"missing seed.csv (tight={t is not None}, env={e is not None})",
            }
            continue
        per_event[str(ev)] = recovery_for_event(
            t, e, min_pt=min_pt, max_seeds_per_spm=max_seeds_per_spm
        )

    fracs = [
        v["frac_recovered"]
        for v in per_event.values()
        if "frac_recovered" in v and np.isfinite(v["frac_recovered"])
    ]
    overall = float(np.mean(fracs)) if fracs else float("nan")
    passed = bool(overall >= 0.98) if fracs else False
    return {
        "check": 2,
        "name": "tight_seed_recovery",
        "tight_dir": str(tight_dir),
        "envelope_dir": str(envelope_dir),
        "events": [int(e) for e in events],
        "filter": {
            "min_pt": float(min_pt),
            "max_seeds_per_spm": int(max_seeds_per_spm),
            "note": (
                "impactMax/sigmaScattering not applied offline — recovery is an "
                "upper bound on subset recoverability"
            ),
        },
        "per_event": per_event,
        "mean_frac_recovered": overall,
        "pass": bool(passed),
        "threshold": 0.98,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("tight_dir", type=Path)
    p.add_argument("envelope_dir", type=Path)
    p.add_argument("--events", type=int, nargs="+", default=[0, 1])
    p.add_argument("--min-pt", type=float, default=0.6876416997703818)
    p.add_argument("--max-seeds-per-spm", type=int, default=16)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()
    report = probe_seed_recovery(
        args.tight_dir,
        args.envelope_dir,
        args.events,
        min_pt=args.min_pt,
        max_seeds_per_spm=args.max_seeds_per_spm,
    )
    text = json.dumps(report, indent=2)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)


if __name__ == "__main__":
    main()
