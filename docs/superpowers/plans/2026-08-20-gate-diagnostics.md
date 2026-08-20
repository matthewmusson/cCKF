# Gate Diagnostic Trace & Analysis Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Capture per-candidate feature vectors at each CKF step for traced seeds, remove the chi2 ceiling, rebuild, run inference, and analyze whether the cluster moments fix resolved the distribution shift.

**Architecture:** Three changes to the C++ diagnostic infrastructure (TrackTrace redesign with representative candidate sampling, chi2 ceiling removal, CSV writer update), a rebuild on Modal, one diagnostic inference run, and a Python analysis script that produces three diagnostic outputs (feature distribution match, gate quality vs classical, compound error detection).

**Tech Stack:** C++ (ACTS header-only cCKF library), Python (numpy, pandas for analysis), Modal (build + run)

## Global Constraints

- Build runs on Modal via `modal run modal_build_acts.py::build_acts --force`
- Inference runs via `modal run modal_build_acts.py::run_cckf --events 1`
- The gate MLP architecture is 26→128→128→128→1 SiLU; feature order follows `cckf/features.py::GATE_FEATURES`
- Weight blob binary format: magic "CCKF", version 1, then `n_features` floats of `standardization_mean`, `n_features` floats of `standardization_std`, 4 Platt params, then weight/bias pairs
- Test data: events [32, 64); training data: events [0, 32). Never cross this boundary.
- The chi2 ceiling (`chi2Ceiling`) is being **removed** — the gate must stand alone
- No changes to the Kalman engine, propagator, or track containers

---

### Task 1: Redesign TrackTrace for representative candidate sampling

**Files:**
- Modify: `cCKF/acts_patches/cckf/CckfTimers.hpp:66-127`

**Interfaces:**
- Consumes: Nothing from other tasks
- Produces: `TrackTrace` struct with `addStep(uint32_t seed_idx, uint32_t step_k, const std::array<float, 26>* all_feats, const float* all_scores, const float* all_chi2s, std::size_t n_cands, std::size_t n_passed)` — called by Task 2's updated measurement selector

The current `TrackTrace::addStep` stores only one candidate per step (the first accepted). The new version stores up to 30 candidates per step per traced seed, selected to be representative of the full candidate distribution:

- **Slot 0**: Best accepted candidate (lowest chi2 among accepted). If none accepted, lowest-chi2 overall.
- **Slot 1**: Min-chi2 candidate across ALL candidates on the surface.
- **Slot 2**: Max-chi2 candidate across ALL candidates.
- **Slots 3–29**: Evenly spaced quantiles from chi2-sorted candidate list: index `floor(j * (N-1) / 26)` for j = 0..26. Deduplication with slots 0–2 is acceptable — the overlap confirms where the accepted/min/max sit within the distribution.

Each `CandEntry` stores: `float features[26]`, `float score`, `float chi2`, `bool accepted`.

Each `StepEntry` stores: `uint32_t step_k`, `uint32_t seed_idx`, `int n_cands_stored`, `int n_cands_total`, `int n_accepted`, `CandEntry cands[MAX_CANDS]`.

Constants: `MAX_TRACKS = 100`, `MAX_STEPS = 15`, `MAX_CANDS = 30`.

Memory: 100 × 15 × (20 + 30 × 116) ≈ 5.2 MB per thread. Acceptable for diagnostic builds.

- [ ] **Step 1: Replace the TrackTrace struct in CckfTimers.hpp**

Replace lines 66–127 of `cCKF/acts_patches/cckf/CckfTimers.hpp` (the entire `TrackTrace` struct including its doc comment) with:

```cpp
/// Per-track feature trace: for the first N seeds, logs a representative
/// sample of ALL candidates' raw features at every step. The sample
/// preserves the distribution shape: accepted (best chi2 among passed),
/// min-chi2, max-chi2, and evenly spaced chi2 quantiles.
struct TrackTrace {
  static constexpr int N_FEAT = 26;
  static constexpr int MAX_TRACKS = 100;
  static constexpr int MAX_STEPS = 15;
  static constexpr int MAX_CANDS = 30;

  struct CandEntry {
    float features[N_FEAT] = {};
    float score = 0;
    float chi2 = 0;
    bool accepted = false;
  };

  struct StepEntry {
    uint32_t step_k = 0;
    uint32_t seed_idx = 0;
    int n_cands_stored = 0;
    int n_cands_total = 0;
    int n_accepted = 0;
    CandEntry cands[MAX_CANDS] = {};
  };

  StepEntry entries[MAX_TRACKS][MAX_STEPS] = {};
  int step_counts[MAX_TRACKS] = {};
  int n_tracks = 0;

  void addStep(uint32_t seed_idx, uint32_t step_k,
               const std::array<float, N_FEAT>* all_feats,
               const float* all_scores, const float* all_chi2s,
               std::size_t n_cands, std::size_t n_passed) {
    // Find or create track slot
    int t = -1;
    for (int i = 0; i < n_tracks; ++i) {
      if (entries[i][0].seed_idx == seed_idx) { t = i; break; }
    }
    if (t < 0) {
      if (n_tracks >= MAX_TRACKS) return;
      t = n_tracks++;
    }
    int s = step_counts[t];
    if (s >= MAX_STEPS) return;
    auto& se = entries[t][s];
    se.step_k = step_k;
    se.seed_idx = seed_idx;
    se.n_cands_total = static_cast<int>(n_cands);
    se.n_accepted = static_cast<int>(n_passed);

    if (n_cands == 0) {
      se.n_cands_stored = 0;
      step_counts[t] = s + 1;
      return;
    }

    // Build sorted-by-chi2 index array (small N, insertion sort is fine)
    int sorted_idx[512];
    int N = static_cast<int>(n_cands);
    if (N > 512) N = 512;
    for (int i = 0; i < N; ++i) sorted_idx[i] = i;
    for (int i = 1; i < N; ++i) {
      int key = sorted_idx[i];
      float keyVal = all_chi2s[key];
      int j = i - 1;
      while (j >= 0 && all_chi2s[sorted_idx[j]] > keyVal) {
        sorted_idx[j + 1] = sorted_idx[j];
        --j;
      }
      sorted_idx[j + 1] = key;
    }

    // Pick representative indices into a selection buffer
    int sel[MAX_CANDS];
    int n_sel = 0;

    // Slot 0: best accepted (lowest chi2 among passed candidates)
    int best_acc = -1;
    for (int i = 0; i < N; ++i) {
      int idx = sorted_idx[i];
      if (static_cast<std::size_t>(idx) < n_passed) {
        best_acc = idx;
        break;
      }
    }
    sel[n_sel++] = (best_acc >= 0) ? best_acc : sorted_idx[0];

    if (N <= MAX_CANDS) {
      // Store all candidates — no subsampling needed
      for (int i = 0; i < N; ++i) {
        if (sorted_idx[i] == sel[0]) continue;
        sel[n_sel++] = sorted_idx[i];
      }
    } else {
      // Slot 1: min-chi2 overall
      if (sorted_idx[0] != sel[0]) sel[n_sel++] = sorted_idx[0];
      // Slot 2: max-chi2 overall
      if (sorted_idx[N - 1] != sel[0]) sel[n_sel++] = sorted_idx[N - 1];
      // Remaining slots: evenly spaced quantiles from sorted list
      int remaining = MAX_CANDS - n_sel;
      for (int j = 0; j < remaining && n_sel < MAX_CANDS; ++j) {
        int pick = sorted_idx[(j * (N - 1)) / (remaining - 1)];
        // Skip duplicates
        bool dup = false;
        for (int k = 0; k < n_sel; ++k) {
          if (sel[k] == pick) { dup = true; break; }
        }
        if (!dup) sel[n_sel++] = pick;
      }
    }

    se.n_cands_stored = n_sel;
    for (int c = 0; c < n_sel; ++c) {
      auto& ce = se.cands[c];
      int idx = sel[c];
      for (int j = 0; j < N_FEAT; ++j) ce.features[j] = all_feats[idx][j];
      ce.score = all_scores[idx];
      ce.chi2 = all_chi2s[idx];
      ce.accepted = (static_cast<std::size_t>(idx) < n_passed);
    }
    step_counts[t] = s + 1;
  }
};
```

- [ ] **Step 2: Verify CckfTimers compiles in isolation**

This is a header-only struct — verify by checking that the syntax is valid C++17. The real compilation test happens in Task 4 (build). Here, visually confirm:
- `std::array<float, N_FEAT>` requires `#include <array>` — already present at top of file (added earlier).
- `std::size_t` requires `<cstdint>` — already present.
- No other new includes needed.

- [ ] **Step 3: Commit**

```bash
cd /Users/matthewm/SURP/cCKF
git add acts_patches/cckf/CckfTimers.hpp
git commit -m "diag: redesign TrackTrace for representative candidate sampling

Store up to 30 candidates per step per traced seed (100 seeds).
Selection: best-accepted, min-chi2, max-chi2, then evenly spaced
chi2 quantiles. Stores all candidates when n_cands <= 30."
```

---

### Task 2: Remove chi2 ceiling and update trace callsite

**Files:**
- Modify: `cCKF/acts_patches/cckf/CckfMeasurementSelector.hpp:202,242-247`

**Interfaces:**
- Consumes: `TrackTrace::addStep` from Task 1 (new signature with `std::array<float, N_FEAT>*`, `n_cands`, `n_passed`)
- Produces: Same `select()` return type (iterator pair) — no interface change to callers

Two changes in `CckfMeasurementSelector::select()`:

1. **Remove chi2 ceiling** from the partition condition (line 202). The gate must stand alone — imposing a hard ceiling prevents us from seeing whether the CKF goes off-track.

2. **Update the trace callsite** (lines 242-247) to pass all candidates' features using the new `addStep` signature. Call it unconditionally (not gated on `passedCandidates > 0`) so we capture steps where the gate rejected everything.

- [ ] **Step 1: Remove chi2 ceiling from partition**

In `cCKF/acts_patches/cckf/CckfMeasurementSelector.hpp`, the partition condition at line 202 currently reads:

```cpp
      if (scores[i] >= m_config.gateThreshold &&
          chi2s[i] <= m_config.chi2Ceiling) {
```

Note: this may already have been partially edited in a prior session. If the file currently has only `scores[i] >= m_config.gateThreshold` (no chi2Ceiling), this step is already done — skip it. If the chi2Ceiling condition is present, replace the two-line condition with:

```cpp
      if (scores[i] >= m_config.gateThreshold) {
```

- [ ] **Step 2: Update the trace callsite**

In `CckfMeasurementSelector.hpp`, find the trace block (around lines 242-247):

```cpp
      // Track-level trace: log ALL candidates' features for first N seeds
      m_timers->track_trace.addStep(
          m_seedIndex, m_branchCtx.step_k,
          featCache.data(), scores.data(), chi2s.data(),
          candidates.size(), passedCandidates);
```

If this already matches the new `addStep` signature (takes `featCache.data()` as `const std::array<float, 26>*`, plus `scores.data()`, `chi2s.data()`, `n_cands`, `n_passed`), this step is already done.

If the old single-candidate signature is still there:

```cpp
      // Track-level trace: log the best accepted candidate's features
      // AND all-candidate chi2 stats for the first N seeds
      if (passedCandidates > 0) {
        m_timers->track_trace.addStep(
            m_seedIndex, m_branchCtx.step_k,
            featCache[0].data(), scores[0], chi2s[0],
            chi2s.data(), candidates.size());
      }
```

Replace it with:

```cpp
      // Track-level trace: log representative candidates for first N seeds
      m_timers->track_trace.addStep(
          m_seedIndex, m_branchCtx.step_k,
          featCache.data(), scores.data(), chi2s.data(),
          candidates.size(), passedCandidates);
```

Note: `featCache` is `std::vector<std::array<float, 26>>`, so `featCache.data()` returns `const std::array<float, 26>*`, matching the `addStep` parameter type.

- [ ] **Step 3: Commit**

```bash
cd /Users/matthewm/SURP/cCKF
git add acts_patches/cckf/CckfMeasurementSelector.hpp
git commit -m "diag: remove chi2 ceiling, pass all candidates to trace

Gate must stand alone without a hard chi2 cutoff so diagnostics
can see the full distribution shift. Pass all candidates' features,
scores, and chi2s to TrackTrace::addStep for representative sampling."
```

---

### Task 3: Update CSV writer for per-candidate output

**Files:**
- Modify: `cCKF/acts_patches/ActsExamples/TrackFinding/CckfTrackFindingAlgorithm.cpp:1158-1207`

**Interfaces:**
- Consumes: `TrackTrace` struct from Task 1 (new `StepEntry`/`CandEntry` layout with `n_cands_stored`, `n_cands_total`, `n_accepted`, and `cands[]`)
- Produces: `_traces.csv` file on the Modal data volume with one row per candidate

The current CSV writer outputs one row per step (only the accepted candidate). The new writer outputs one row per candidate per step, with columns that enable all three analyses.

- [ ] **Step 1: Replace the CSV writer block**

In `CckfTrackFindingAlgorithm.cpp`, find the block starting at line 1158:

```cpp
    // Track-level feature traces — raw features at every step for first N seeds
    if (!m_cfg.outputTimingPath.empty()) {
```

Replace everything from that line through the closing `}` of the `if (fout)` block and the `ACTS_INFO("  track traces: ..."` line (through approximately line 1207) with:

```cpp
    // Track-level feature traces — one row per candidate per step
    if (!m_cfg.outputTimingPath.empty()) {
      const auto& tt = timers.track_trace;
      std::string tracePath = m_cfg.outputTimingPath;
      auto dot = tracePath.rfind('.');
      if (dot != std::string::npos) {
        tracePath = tracePath.substr(0, dot) + "_traces.csv";
      } else {
        tracePath += "_traces.csv";
      }
      bool writeHdr = false;
      {
        std::ifstream probe(tracePath);
        writeHdr = !probe.good() ||
                   probe.peek() == std::ifstream::traits_type::eof();
      }
      std::ofstream fout(tracePath, std::ios::app);
      if (fout) {
        if (writeHdr) {
          fout << "event,seed,step_k,cand_idx,accepted,chi2,score,"
               << "n_cands_total,n_accepted,"
               << "res0,res1,chol_S_00,chol_S_10,chol_S_11,chi2_feat,"
               << "clus_s_u,clus_s_v,clus_q_tot,"
               << "clus_sigma_uu,clus_sigma_uv,clus_sigma_vv,"
               << "kappa_u,kappa_v,q_tilde,n_window,"
               << "eta,qop,step_k_feat,pathInX0,"
               << "pitch_u,pitch_v,thickness,"
               << "n_hits,n_holes,n_seq_holes\n";
        }
        for (int t = 0; t < tt.n_tracks; ++t) {
          for (int s = 0; s < tt.step_counts[t]; ++s) {
            const auto& se = tt.entries[t][s];
            for (int c = 0; c < se.n_cands_stored; ++c) {
              const auto& ce = se.cands[c];
              fout << ctx.eventNumber << "," << se.seed_idx << ","
                   << se.step_k << "," << c << ","
                   << (ce.accepted ? 1 : 0) << ","
                   << ce.chi2 << "," << ce.score << ","
                   << se.n_cands_total << "," << se.n_accepted;
              for (int j = 0; j < 26; ++j) {
                fout << "," << ce.features[j];
              }
              fout << "\n";
            }
          }
        }
      }
      ACTS_INFO("  track traces: " << tt.n_tracks << " seeds, written to "
                << tracePath);
    }
```

- [ ] **Step 2: Commit**

```bash
cd /Users/matthewm/SURP/cCKF
git add acts_patches/ActsExamples/TrackFinding/CckfTrackFindingAlgorithm.cpp
git commit -m "diag: CSV writer outputs one row per candidate per step

Columns: event, seed, step_k, cand_idx, accepted, chi2, score,
n_cands_total, n_accepted, then all 26 raw feature values.
Enables per-candidate feature distribution analysis."
```

---

### Task 4: Build and run diagnostic inference

**Files:**
- No new files created
- Uses: `cCKF/modal_build_acts.py` (existing `build_acts` and `run_cckf` functions)

**Interfaces:**
- Consumes: Committed C++ changes from Tasks 1–3
- Produces: `_traces.csv` on Modal data volume, ACTS log output with per-step chi2 table and standardization params

- [ ] **Step 1: Build patched ACTS on Modal**

```bash
cd /Users/matthewm/SURP/cCKF
modal run modal_build_acts.py::build_acts --force
```

Expected: build completes successfully. If it fails, check the error against the "Common Build Errors and Fixes" table in `cCKF/CLAUDE.md` and fix before retrying.

- [ ] **Step 2: Run cCKF inference on 1 event with diagnostics**

```bash
cd /Users/matthewm/SURP/cCKF
modal run modal_build_acts.py::run_cckf --events 1
```

Expected output includes:
- ACTS_INFO lines with all 26 `std[i ...]` standardization params (mean, std)
- ACTS_INFO lines with per-step chi2 table (`step K: accepted=... mean_chi2_acc=... total_cands=... mean_chi2_all=...`)
- ACTS_INFO line: `track traces: 100 seeds, written to ...`
- Gate diagnostics: `n_accepted=..., n_rejected=..., chi2 accepted: mean=... max=...`

- [ ] **Step 3: Download the traces CSV**

```bash
cd /Users/matthewm/SURP/cCKF
modal volume get surp-acts-data results/run_cckf_*/timing_traces.csv ./output/traces.csv
```

If the exact path differs, check with `modal volume ls surp-acts-data results/` to find the latest run directory, then look for `*_traces.csv`.

- [ ] **Step 4: Verify CSV structure**

```bash
head -1 output/traces.csv
wc -l output/traces.csv
```

Expected: header line matches the columns from Task 3. Row count should be approximately 100 seeds × 10–15 steps × 20–30 candidates ≈ 20,000–45,000 rows.

---

### Task 5: Analysis script — feature distribution match, gate quality, compound error

**Files:**
- Create: `cCKF/scripts/analyze_traces.py`

**Interfaces:**
- Consumes: `_traces.csv` from Task 4, weight blob's mu/sigma (either from ACTS log output or by reading the blob directly)
- Produces: Three printed diagnostic tables + interpretation guidance

This script answers three questions:
1. Did the cluster moments fix resolve the feature distribution mismatch? (feature distribution match)
2. Is the gate choosing the right candidates? (gate quality vs classical)
3. Is compound error building over track depth? (chi2 distribution vs step)

- [ ] **Step 1: Write the analysis script**

Create `cCKF/scripts/analyze_traces.py`:

```python
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
    print("Ratio ≈ 1.0 means match. >> 1 or << 1 means mismatch.\n")

    feat_cols = FEATURE_NAMES
    X = df[feat_cols].values  # (n_rows, 26)

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

    # Z-score audit: what fraction of values have |z| > 5?
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

    # Group by (seed, step_k) to get per-surface stats
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

    # Accepted-only vs all-candidates comparison
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
```

- [ ] **Step 2: Test the script can parse a synthetic CSV**

Create a small synthetic trace CSV to verify the script runs without errors:

```bash
cd /Users/matthewm/SURP/cCKF
python3 -c "
import csv, random
header = ['event','seed','step_k','cand_idx','accepted','chi2','score',
          'n_cands_total','n_accepted',
          'res0','res1','chol_S_00','chol_S_10','chol_S_11','chi2_feat',
          'clus_s_u','clus_s_v','clus_q_tot',
          'clus_sigma_uu','clus_sigma_uv','clus_sigma_vv',
          'kappa_u','kappa_v','q_tilde','n_window',
          'eta','qop','step_k_feat','pathInX0',
          'pitch_u','pitch_v','thickness',
          'n_hits','n_holes','n_seq_holes']
with open('/tmp/test_traces.csv', 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(header)
    for seed in range(5):
        for step in range(8):
            for c in range(10):
                row = [0, seed, step, c, int(c == 0), random.gauss(3, 2),
                       random.random(), 10, 1]
                row += [random.gauss(0, 1) for _ in range(26)]
                w.writerow(row)
"
```

Then run:

```bash
cd /Users/matthewm/SURP/cCKF

# Use a real gate blob if available, otherwise create a dummy one
python3 -c "
import struct, numpy as np
mu = np.zeros(26, dtype=np.float32)
sigma = np.ones(26, dtype=np.float32)
with open('/tmp/test_gate.bin', 'wb') as f:
    f.write(b'CCKF')
    f.write(struct.pack('<I', 1))
    f.write(struct.pack('<III', 26, 128, 3))
    f.write(mu.tobytes())
    f.write(sigma.tobytes())
    f.write(struct.pack('<ffff', 1, 0, 0, 0))
    # Dummy weights — not needed for stats reading
    for layer in [(128, 26), (128, 128), (128, 128), (1, 128)]:
        out, inp = layer
        f.write(np.zeros(out * inp, dtype=np.float32).tobytes())
        f.write(np.zeros(out, dtype=np.float32).tobytes())
"

python3 scripts/analyze_traces.py /tmp/test_traces.csv /tmp/test_gate.bin
```

Expected: three analysis tables print without errors. Values will be meaningless (random data) but format is correct.

- [ ] **Step 3: Commit**

```bash
cd /Users/matthewm/SURP/cCKF
git add scripts/analyze_traces.py
git commit -m "diag: add trace analysis script for feature/gate/compound diagnostics

Three analyses:
1. Feature distribution match (empirical vs training mu/sigma)
2. Gate quality (accepted chi2 vs min chi2 per step)
3. Compound error (chi2 distribution evolution over steps)"
```

---

### Task 6: Run analysis on real traces and interpret results

**Files:**
- No files created or modified — this is an analysis/interpretation task

**Interfaces:**
- Consumes: `output/traces.csv` from Task 4, gate weight blob from Modal volume
- Produces: Diagnostic conclusions that determine next steps

- [ ] **Step 1: Download the gate weight blob**

```bash
cd /Users/matthewm/SURP/cCKF
modal volume get surp-acts-data weights/gate.bin ./output/gate.bin
```

- [ ] **Step 2: Run the analysis**

```bash
cd /Users/matthewm/SURP/cCKF
python3 scripts/analyze_traces.py output/traces.csv output/gate.bin
```

- [ ] **Step 3: Interpret Analysis 1 (feature distribution match)**

Check the `std_ratio` column for each feature. Key features to watch:

| Feature index | Name | Expected if fix worked | Expected if fix failed |
|---|---|---|---|
| 9 | clus_sigma_uu | ratio ≈ 1.0 | ratio << 0.01 or >> 100 |
| 10 | clus_sigma_uv | ratio ≈ 1.0 | ratio << 0.01 or >> 100 |
| 11 | clus_sigma_vv | ratio ≈ 1.0 | ratio << 0.01 or >> 100 |
| 0–8, 12–25 | all others | ratio ≈ 1.0 (already matching) | ratio ≈ 1.0 (unchanged) |

In the z-score audit, no feature should have `|z| > 5` for more than ~1% of candidates. If features 9–11 show `|z| > 5` in > 10% of rows, the moments fix did not fully resolve the mismatch.

- [ ] **Step 4: Interpret Analysis 2 (gate quality)**

The `chose_best%` column shows how often the gate agrees with the classical chi2 selector. The `mean_ratio` shows how far off the gate's choice is on average.

| Metric | Healthy gate | Broken gate |
|---|---|---|
| chose_best% | > 60% | < 20% |
| mean_ratio at step 0 | < 2.0 | > 10 |
| mean_ratio trend | flat or decreasing | increasing with step |

If `chose_best%` is high at step 0 but drops at later steps, that's compound error (the gate works on good states but fails on corrupted ones).

- [ ] **Step 5: Interpret Analysis 3 (compound error)**

Compare `med_chi2` (all candidates) across steps:

| Pattern | Diagnosis | Next step |
|---|---|---|
| Flat (~2–5) at all steps | No compound error — gate issue only | Moments fix should suffice |
| Grows linearly with step | Moderate compound error | Moments fix + may need DAgger |
| Grows exponentially | Severe compound error | DAgger is required regardless of fix |

If `med_chi2` is flat but `mean_chi2` of accepted candidates is high, the gate is choosing wrong from good options — a feature/model issue, not compound error.

- [ ] **Step 6: Record findings**

Based on the three analyses, update `cCKF/experiments/LOG.md` with:
- Date, config used, event count
- Feature distribution match results (especially features 9–11)
- Gate quality summary
- Compound error verdict
- Decision: proceed to Pareto sweep, or further debugging needed
