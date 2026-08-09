# ACTS Instrumentation — Three Patches for cCKF Data Collection

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Persist three currently-discarded quantities per CKF TrackState — the full innovation covariance `S`, the material `X/X₀` accumulated since the previous measurement surface, and the cluster shape/charge features plus track incidence angles — so the cCKF gate feature vector (spec §8.2) can be built directly from `trackstates.root`.

**Architecture:** All three patches are additive instrumentation. Patch A reads quantities already computed inside `RootTrackStatesWriter` and writes three more branches. Patch B threads `MaterialSlab::thicknessInX0()` out of `detail::performMaterialInteraction` through a new field on `PointwiseMaterialEffects`, accumulates it in a single scalar on the CKF result, and stamps it onto each new TrackState via an optional dynamic column. Patch C adds a `ClusterContainer` read handle to the writer, joins clusters to track states by measurement index, and computes charge second moments and incidence angles through two pure helper functions that are unit-tested in isolation. The Kalman engine's numerical behaviour is unchanged by all three.

**Tech Stack:** C++20, ACTS (cmuchancel fork), CMake, Boost.Test (unit tests), ROOT / uproot (output verification), Python 3.10+ for the verification script.

## Global Constraints

- **Tier 2 task.** Do not modify any label definition, loss function, calibrator, value target, or training code. Nothing in this plan touches `specs`-governed learning code.
- **Do not run data collection.** Single-event smoke runs for verification only. Never open events `[32, 64)` — that is the held-out test set (CLAUDE.md, spec §6.1).
- **Three separate commits**, one per patch (A, B, C), each independently reviewable and independently revertable. Task 6's verification script and Task 1's report get their own commits.
- **Every new ROOT branch must be state-aligned**: exactly one entry pushed per TrackState, so the branch is index-parallel to `volume_id`, `layer_id`, `module_id`, `chi2`, and `pathLength`. Use `NaN` (float branches) or `0` (int branches) where a quantity is undefined. See the alignment trap in Task 1 — the existing `*_hit` branches do **not** satisfy this and must not be used as a template for push placement.
- **ACTS code style:** every new/modified file keeps the existing MPL-2.0 copyright header verbatim. Run `clang-format` (the repo's `.clang-format`) on every file you touch before committing. Doxygen `///` comments on new public functions and struct fields.
- **Numerical no-op requirement:** none of these patches may change track finding results. Task 6 verifies this by diffing track counts and χ² against a pre-patch baseline.
- **Naming:** new branch names exactly as specified — `S00_prt`, `S01_prt`, `S11_prt`, `pathInX0_interval`, `alpha_u`, `alpha_v`, `clus_size_u`, `clus_size_v`, `clus_qtot`, `clus_sigma_uu`, `clus_sigma_uv`, `clus_sigma_vv`.

## Repository Note — read before Task 1

The investigation below was performed against a local checkout of **upstream `acts-project/acts`, branch `main`**, at `/Users/matthewm/SURP/acts`. The patches target the **cmuchancel fork**. The fork is expected to match closely, but every line number cited is an upstream line number and must be re-anchored in the fork. Task 1 exists specifically to do that re-anchoring. **Do not skip Task 1**, and do not trust a line number in this document without confirming the quoted code is actually there.

Anchor by *quoted code string*, never by line number alone. Every task below gives you the exact string to `grep` for.

---

## Phase 1 Findings (pre-computed; Task 1 verifies against the fork)

These answer the four investigation questions. Task 1 turns this into the committed report.

### Q1 — Which of the six cluster features does geometric digitization compute and store?

| Feature | Status | Where |
|---|---|---|
| `s_u` | **Stored** as `Cluster::sizeLoc0` | `Examples/Framework/include/ActsExamples/EventData/Cluster.hpp:22`; filled `Examples/Algorithms/Digitization/src/DigitizationAlgorithm.cpp:410-412` (`bmax[0]-bmin[0]+1`) and, for merged clusters, `Examples/Algorithms/Digitization/src/ModuleClusters.cpp:307-308` |
| `s_v` | **Stored** as `Cluster::sizeLoc1` | same sites, index 1 |
| `Q_tot` | **Derivable, no new code needed** — `Cluster::sumActivations()` | `Cluster.hpp:37-41`; sums `Cell::activation`, which `DigitizationAlgorithm.cpp:283,306-307` sets to the post-smearing, post-threshold charge |
| `σ_uu` | **NOT computed anywhere in ACTS** | — |
| `σ_uv` | **NOT computed anywhere in ACTS** | — |
| `σ_vv` | **NOT computed anywhere in ACTS** | — |

**The three second moments are missing, but this does not block the patches.** They are fully derivable at write time from `Cluster::channels`. Each channel is an `ActsFatras::Segmentizer::ChannelSegment` (`Fatras/include/ActsFatras/Digitization/Segmentizer.hpp:66-80`) carrying:

- `bin` — the 2D readout bin index,
- `path2D` — the segment's start/end points **in local surface cartesian coordinates**, populated at `Fatras/src/Digitization/Segmentizer.cpp:53`, `:97`, `:160`,
- `activation` — the (smeared, thresholded) charge in that channel.

Charge-weighted second moments about the charge centroid follow directly from the `path2D` midpoints and `activation` weights, in physical length units, without needing the module's `BinUtility`. Patch C **computes** these; it does not copy them.

**Trap:** `Cluster` also declares `localEta`, `localPhi`, `globalEta`, `globalPhi`, `etaAngle`, `phiAngle`, `localDirection`, `lengthDirection` (`Cluster.hpp:27-35`). These look like the incidence angles we want. **They are only ever filled by `RootAthenaDumpReader`** (`Examples/Io/Root/src/RootAthenaDumpReader.cpp:369-378`), which reads ATLAS dumps. The Geant4 → `DigitizationAlgorithm` path never touches them, so in the ColliderML pipeline they are identically zero. Do not read them. Patch C computes the angles from the surface reference frame instead.

### Q2 — Are they accessible during CKF execution / at write time?

Not from the measurement or source link object itself. `ActsExamples::Cluster` lives in a **separate collection**, `ClusterContainer = std::vector<Cluster>` (`Cluster.hpp:45`), produced by `DigitizationAlgorithm` under the config key `outputClusters` (default `"clusters"`, declared `Examples/Algorithms/Digitization/include/ActsExamples/Digitization/DigitizationAlgorithm.hpp:47-48`, written via `m_outputClusters` at `:149`).

That collection **is in the event store**, so `RootTrackStatesWriter` can read it by adding a `ReadDataHandle<ClusterContainer>`. **The features are available in memory at write time. No CSV and no offline join are required.** This is the good case of Patch C.

### Q3 — What is the join key?

`Cluster.hpp:44` documents: *"Clusters have a one-to-one relation with measurements"* — `ClusterContainer` is indexed by measurement index.

The writer **already extracts that index**: at `Examples/Io/Root/src/RootTrackStatesWriter.cpp:412-419` it pulls `IndexSourceLink` off the track state and takes `const auto hitIdx = sl.index();`.

So the join is a direct array index: `clusters.at(sl.index())`. Nothing offline.

### Q4 — Is the surface local frame available at write time?

Yes, both ingredients are already in hand inside `writeT`:

- **Surface:** `const Acts::Surface& surface = state.referenceSurface();` — `RootTrackStatesWriter.cpp:364`. Plus `const Acts::GeometryContext& gctx = ctx.geoContext;` at `:313`.
- **Local frame:** `Surface::referenceFrame(gctx, position, direction)` — declared `Core/include/Acts/Surfaces/Surface.hpp:368`, implemented `Core/src/Surfaces/Surface.cpp:243`. Returns a `RotationMatrix3` whose three columns are the measurement-frame axes `(û, v̂, n̂)`.
- **Predicted direction:** `Acts::transformBoundToFreeParameters(surface, gctx, state.predicted())`, then `.segment<3>(Acts::eFreeDir0)`. The writer already performs exactly this transform at `RootTrackStatesWriter.cpp:608-610`.

So `d̂·û`, `d̂·v̂`, `d̂·n̂` are all computable at write time.

**Deliberate deviation from the task spec, flag to Matthew:** the task asks for the raw ratios `d̂·û / d̂·n̂` and `d̂·v̂ / d̂·n̂`. These diverge at grazing incidence (`d̂·n̂ → 0`), which produces infinities in a feature that feeds a network. This plan stores the **angles** instead:

```
alpha_u = atan2(d̂·û, d̂·n̂)
alpha_v = atan2(d̂·v̂, d̂·n̂)
```

These are bounded in `(-π, π]`, well-defined at grazing incidence, and carry identical information: `tan(alpha_u) == d̂·û / d̂·n̂`. If the raw ratio is genuinely wanted downstream it is one `std::tan` away. This is documented in the report and in the code comment.

### Q5 (extra, required by Patch B) — where does the CKF discard the material slab?

`detail::performMaterialInteraction(state, stepper, surface, updateMode, ...)` at `Core/include/Acts/Propagator/detail/PointwiseMaterialInteraction.hpp:239-262`:

```cpp
const Result<MaterialSlab> slabResult =
    evaluateMaterialSlab(state, stepper, surface, updateMode);
...
const MaterialSlab& slab = slabResult.value();
...
const PointwiseMaterialEffects effects = performMaterialInteraction(
    state, stepper, slab, noiseUpdateMode, multipleScattering, energyLoss);
```

`slab` goes out of scope. The returned `PointwiseMaterialEffects` (`:105-110`) carries only `eLoss`, `variancePhi`, `varianceTheta`, `varianceQoverP` — no `X/X₀`. `MaterialSlab::thicknessInX0()` is `Core/include/Acts/Material/MaterialSlab.hpp:112`.

The CKF actor calls it at exactly three sites in `Core/include/Acts/TrackFinding/CombinatorialKalmanFilter.hpp`:

| Site | Line | Context |
|---|---|---|
| `reset()` | 418-425 | `MaterialUpdateMode::PostUpdate` on the resumed branch's reference surface |
| `filter()` pre | 488-495 | `MaterialUpdateMode::PreUpdate` on the current surface |
| `filter()` post | 624-632 | `MaterialUpdateMode::PostUpdate` on the current surface |

Truly passive surfaces (no material) return early at `:475-478` **before** any material call, so they contribute nothing — which is correct and needs no special handling.

### Q6 (extra, required by Patch B) — do branch forks need per-branch accumulator storage?

**No — a single scalar suffices.** The CKF search is depth-first:

- Forking happens only in `processNewTrackStates` (`CombinatorialKalmanFilter.hpp:654-720`), which runs **after** the pre-update material call at `:488`. So all candidates on a surface inherit the same accumulated interval by construction — exactly as the task statement anticipated.
- Between two measurement surfaces there is exactly one branch propagating: `result.activeBranches.back()`.
- When the search backtracks to a different branch, `reset()` (`:395-440`) re-initialises the stepper at that branch's tip surface and applies `PostUpdate` material there. Zeroing the accumulator at the top of `reset()` and letting that post-update land in it reproduces precisely "material since the previous measurement surface" for the resumed branch.

So: one `double` on `CombinatorialKalmanFilterResult`, zeroed in `reset()` and zeroed again immediately after stamping at a measurement surface. **This makes Patch B substantially smaller than the 70–120 line estimate in the task statement** — roughly 40 lines across four files. No per-branch map, no inheritance logic.

### The alignment trap — affects all three patches

`RootTrackStatesWriter` has two families of per-state branches, and **they are not the same length**.

State-aligned (one push per TrackState, in the outer `for (const auto& state : track.trackStatesReversed())` loop): `volume_id`, `layer_id`, `module_id`, `stateType`, `pathLength`, `chi2`.

**Not** state-aligned: `dim_hit`, `res_x_hit`, `err_x_hit`, `pull_x_hit`, and their `_y` counterparts. These are pushed inside the parameter loop guarded by `if (ipar == ePredicted)`, and there is a bare `continue` at `RootTrackStatesWriter.cpp:630`:

```cpp
        if (!state.hasUncalibratedSourceLink()) {
          continue;
        }
```

which fires *after* `m_hasParams[ipar].push_back(...)` but *before* the `ipar == ePredicted` hit block at `:672`. Consequence: a **hole** state that nonetheless has valid predicted parameters gets no `dim_hit` entry at all, so `dim_hit.size() <= nStates`, and index `i` of `dim_hit` does not correspond to index `i` of `volume_id`.

**Every branch added by this plan is pushed in the outer state loop**, giving `size() == nStates` unconditionally. Task 6 asserts this. Do not follow the `dim_hit` push sites as a template.

---

## File Structure

**cmuchancel ACTS fork** (patches):

| File | Responsibility | Patch |
|---|---|---|
| `Examples/Io/Root/include/ActsExamples/Io/Root/RootTrackStatesWriter.hpp` | branch member declarations, cluster read handle, config field | A, B, C |
| `Examples/Io/Root/src/RootTrackStatesWriter.cpp` | branch registration, per-state fill, per-track clear | A, B, C |
| `Core/include/Acts/Propagator/detail/PointwiseMaterialInteraction.hpp` | carry `pathXOverX0` out of the slab evaluation | B |
| `Core/include/Acts/TrackFinding/CombinatorialKalmanFilter.hpp` | interval accumulator + column name constant + stamp-on-fork | B |
| `Examples/Algorithms/TrackFinding/src/TrackFindingAlgorithm.cpp` | register the optional track-state dynamic column | B |
| `Examples/Framework/include/ActsExamples/EventData/ClusterFeatures.hpp` | **new** — pure helpers: charge second moments, incidence angles | C |
| `Examples/Framework/src/EventData/ClusterFeatures.cpp` | **new** — their implementation | C |
| `Examples/Framework/CMakeLists.txt` | add the new source file | C |
| `Tests/UnitTests/Examples/EventData/ClusterFeaturesTests.cpp` | **new** — unit tests for both helpers | C |
| `Tests/UnitTests/Examples/EventData/CMakeLists.txt` | register the new test target | C |
| `Python/Examples/src/plugins/Root.cpp` | expose `inputClusters` to Python | C |
| `Python/Examples/python/reconstruction.py` | pass `inputClusters` when constructing the writer | C |

**cCKF repo** (report + verification):

| File | Responsibility |
|---|---|
| `docs/instrumentation/phase1_cluster_feature_availability.md` | **new** — the Phase 1 report deliverable |
| `scripts/instrumentation/check_trackstate_branches.py` | **new** — reads `trackstates.root`, asserts every new branch exists, is state-aligned, and is finite where expected |
| `docs/instrumentation/trackstate_branch_reference.md` | **new** — final summary: what is written directly, what (if anything) must be joined offline |

`ClusterFeatures.hpp/.cpp` is a new pair rather than an extension of `Cluster.hpp` for two reasons: the incidence-angle helper is pure geometry and has nothing to do with clusters, and putting both in a header with no ROOT/writer dependency makes them unit-testable without linking the ROOT IO library.

---

## Task 1: Fork reconnaissance and Phase 1 report

Re-anchor every claim above against the cmuchancel fork and commit the report. No production code changes.

**Files:**
- Create: `docs/instrumentation/phase1_cluster_feature_availability.md` (cCKF repo)

**Interfaces:**
- Consumes: nothing.
- Produces: a confirmed set of anchor strings for Tasks 2–5. If any anchor is missing from the fork, this task's output is the corrected anchor.

- [ ] **Step 1: Locate the fork checkout and record its revision**

```bash
# Adjust the path if the fork lives elsewhere; NERSC path per CLAUDE.md is
#   /global/cfs/cdirs/atlas/mussonm/cCKF/
# and Max Zhao's ACTS build is /global/cfs/cdirs/atlas/hrzhao/acts
ACTS_FORK=${ACTS_FORK:-$HOME/SURP/acts}
cd "$ACTS_FORK"
git remote -v
git rev-parse HEAD
git log --oneline -3
```

Record the remote URL and HEAD hash — they go in the report header. If `git remote -v` does not show a cmuchancel remote, **stop and ask Matthew which checkout is the fork** before proceeding. Do not patch upstream by accident.

- [ ] **Step 2: Verify all eleven anchor strings exist**

Run this script. Every line must print `OK`.

```bash
cd "$ACTS_FORK"
check () { # $1 = label, $2 = file, $3 = literal string
  if grep -qF "$3" "$2" 2>/dev/null; then
    echo "OK    $1"
  else
    echo "MISS  $1  ($2)"
  fi
}

W_H=Examples/Io/Root/include/ActsExamples/Io/Root/RootTrackStatesWriter.hpp
W_C=Examples/Io/Root/src/RootTrackStatesWriter.cpp
PMI=Core/include/Acts/Propagator/detail/PointwiseMaterialInteraction.hpp
CKF=Core/include/Acts/TrackFinding/CombinatorialKalmanFilter.hpp
TFA=Examples/Algorithms/TrackFinding/src/TrackFindingAlgorithm.cpp
CLU=Examples/Framework/include/ActsExamples/EventData/Cluster.hpp

check "A1 resCov"        "$W_C" 'const Acts::DynamicMatrix resCov = V + H * covariance * H.transpose();'
check "A2 chi2 push"     "$W_C" 'm_chi2.push_back(state.chi2());'
check "A3 dim_hit branch" "$W_C" 'm_outputTree->Branch("dim_hit", &m_dim_hit);'
check "A4 dim_hit clear" "$W_C" 'm_dim_hit.clear();'
check "A5 dim_hit member" "$W_H" 'std::vector<int> m_dim_hit;'
check "A6 surface"       "$W_C" 'const Acts::Surface& surface = state.referenceSurface();'
check "A7 srclink"       "$W_C" 'state.getUncalibratedSourceLink().template get<IndexSourceLink>();'
check "B1 effects struct" "$PMI" 'struct PointwiseMaterialEffects {'
check "B2 slab value"    "$PMI" 'const MaterialSlab& slab = slabResult.value();'
check "B3 ckf result"    "$CKF" 'struct CombinatorialKalmanFilterResult {'
check "B4 newBranches"   "$CKF" 'CkfTypes::BranchVector<TrackProxy> newBranches;'
check "B5 tfa column"    "$TFA" 'tracks.addColumn<unsigned int>("trackGroup");'
check "C1 sizeLoc0"      "$CLU" 'std::size_t sizeLoc0 = 0;'
check "C2 sumActivations" "$CLU" 'double sumActivations() const {'
check "C3 outputClusters" Examples/Algorithms/Digitization/include/ActsExamples/Digitization/DigitizationAlgorithm.hpp 'std::string outputClusters = "clusters";'
check "C4 thicknessInX0" Core/include/Acts/Material/MaterialSlab.hpp 'constexpr float thicknessInX0() const'
check "C5 referenceFrame" Core/include/Acts/Surfaces/Surface.hpp 'virtual RotationMatrix3 referenceFrame('
```

For any `MISS`, open the file, find the equivalent construct, and write the fork's actual string into the report's "anchor corrections" section. Later tasks use the corrected string.

- [ ] **Step 3: Confirm the alignment trap is present in the fork**

```bash
cd "$ACTS_FORK"
grep -n -B2 -A3 'if (!state.hasUncalibratedSourceLink()) {' Examples/Io/Root/src/RootTrackStatesWriter.cpp
```

Expected: **two** matches. One in the outer state loop (around line 389, followed by a block of `push_back(nan)` for truth quantities) and one inside the parameter loop (around line 630) whose body is a bare `continue;`. The second one is the trap. Record both line numbers in the report.

If the second occurrence is absent in the fork, note it — the `*_hit` branches would then be state-aligned and the trap discussion becomes informational only. The plan's approach (push in the outer loop) remains correct either way.

- [ ] **Step 4: Confirm the ODD geometric digi config produces clusters with channels**

```bash
cd "$ACTS_FORK"
python3 -c "
import json, sys
cfg = json.load(open('Examples/Configs/odd-digi-geometric-config.json'))
ents = cfg['acts-geometry-hierarchy-map'] and cfg['entries']
n_geo = sum(1 for e in ents if 'geometric' in e['value'])
n_dig = sum(1 for e in ents if e['value'].get('geometric', {}).get('digital', False))
print(f'entries={len(ents)} with_geometric={n_geo} digital=True count={n_dig}')
"
```

Expected: every entry has a `geometric` block, and `digital` is `false` throughout. `digital: false` matters — it means `weight = charge` rather than `weight = 1` in `DigitizationAlgorithm.cpp`, so `activation` carries real charge and `Q_tot` / the second moments are charge-weighted rather than hit-count-weighted. Record the numbers in the report.

Also confirm the config the cCKF pipeline actually uses:

```bash
cd ~/SURP/cCKF && grep -rn "digi.*config\|digiConfig\|odd-digi" digi_and_reco.py configs/ | head -20
```

Record which digi config file the pipeline resolves to. If it is *not* a geometric config (e.g. a smearing-only config), **that blocks Patch C's cluster features** — say so explicitly in the report and flag it to Matthew, because smearing digitization produces no clusters at all.

- [ ] **Step 5: Write the report**

Create `docs/instrumentation/phase1_cluster_feature_availability.md` in the cCKF repo. Use exactly this structure, filling the bracketed slots from Steps 1–4:

```markdown
# Phase 1 — Cluster Feature Availability for cCKF Gate Features

**Fork:** [remote URL]
**Revision:** [HEAD hash] ([subject line])
**Digi config in use:** [path resolved in Step 4]
**Date:** 2026-08-08

## Summary

Five of the six requested cluster features are available in memory at
TrackState write time. The three charge second moments are not computed
anywhere in ACTS but are fully derivable from data that is present, so
**nothing is blocked** and **no offline CSV join is required**.

## Q1 — What geometric digitization computes

| Feature | Status | Source |
|---|---|---|
| s_u | stored | `Cluster::sizeLoc0`, `Cluster.hpp:[N]`; filled `DigitizationAlgorithm.cpp:[N]` |
| s_v | stored | `Cluster::sizeLoc1`, same |
| Q_tot | derivable, zero new code | `Cluster::sumActivations()`, `Cluster.hpp:[N]` |
| sigma_uu | NOT computed | must be computed from `Cluster::channels` |
| sigma_uv | NOT computed | must be computed from `Cluster::channels` |
| sigma_vv | NOT computed | must be computed from `Cluster::channels` |

The moments are recoverable because each `Cluster::Cell`
(`ActsFatras::Segmentizer::ChannelSegment`, `Segmentizer.hpp:[N]`) carries
`path2D` (segment endpoints in local surface coordinates, filled in
`Segmentizer.cpp:[N]`) and `activation` (charge). Charge-weighted second
moments about the centroid follow directly, in length units.

Config check: [entries]/[with_geometric] entries carry a geometric block;
`digital` is false throughout, so activations are real charge.

**Trap:** `Cluster::localEta/localPhi/etaAngle/phiAngle/localDirection`
(`Cluster.hpp:[N]`) are filled *only* by `RootAthenaDumpReader.cpp:[N]`.
In the Geant4 -> DigitizationAlgorithm path they stay zero. Do not read them.

## Q2 — Accessibility at write time

Not on the measurement or source link. Clusters live in a separate
`ClusterContainer` (`Cluster.hpp:[N]`) published by `DigitizationAlgorithm`
under `outputClusters` (default `"clusters"`,
`DigitizationAlgorithm.hpp:[N]`). It is in the event store, so
`RootTrackStatesWriter` can read it with a `ReadDataHandle<ClusterContainer>`.
**Available in memory at write time.**

## Q3 — Join key

`ClusterContainer` is one-to-one with measurements, indexed by measurement
index (`Cluster.hpp:[N]`). The writer already extracts that index at
`RootTrackStatesWriter.cpp:[N]` as `sl.index()`. Join is `clusters.at(sl.index())`.
**No offline join needed.**

## Q4 — Incidence angles

Both ingredients are already local to `writeT`:
- `state.referenceSurface()` (`RootTrackStatesWriter.cpp:[N]`) and `ctx.geoContext`
- `Surface::referenceFrame(gctx, position, direction)` (`Surface.hpp:[N]`),
  whose columns are (u_hat, v_hat, n_hat)
- predicted direction via `transformBoundToFreeParameters`, already used at
  `RootTrackStatesWriter.cpp:[N]`

Computable. **Deviation:** we store `alpha_u = atan2(d.u, d.n)` and
`alpha_v = atan2(d.v, d.n)` rather than the raw ratios `d.u/d.n`, which
diverge at grazing incidence. Same information: `tan(alpha_u) == d.u/d.n`.

## Blocking issues

[None — or list them explicitly.]

## Anchor corrections vs. the plan document

[Any string from the plan's Task 1 Step 2 that came back MISS, with the
fork's actual equivalent. Write "none" if all OK.]
```

- [ ] **Step 6: Commit the report**

```bash
cd ~/SURP/cCKF
git add docs/instrumentation/phase1_cluster_feature_availability.md
git commit -m "docs: Phase 1 report on cluster feature availability for cCKF gate features"
```

---

## Task 2: Patch A — innovation covariance off-diagonals

Write `S00_prt`, `S01_prt`, `S11_prt` as state-aligned float branches.

**Files:**
- Modify: `Examples/Io/Root/include/ActsExamples/Io/Root/RootTrackStatesWriter.hpp` (member declarations)
- Modify: `Examples/Io/Root/src/RootTrackStatesWriter.cpp` (registration, fill, clear)
- Create: `scripts/instrumentation/check_trackstate_branches.py` (cCKF repo)

**Interfaces:**
- Consumes: anchor strings confirmed in Task 1.
- Produces:
  - ROOT branches `S00_prt`, `S01_prt`, `S11_prt`, each `std::vector<float>`, one entry per TrackState.
  - `scripts/instrumentation/check_trackstate_branches.py` exposing
    `check_file(path: str, tree: str = "trackstates") -> list[str]` returning a list
    of human-readable failure strings (empty list == pass), and a `main()` CLI
    entry point. Tasks 3–5 extend its `EXPECTED` table; Task 6 runs it.

The verification script is written first, in this task, because it is the test harness for all three patches.

- [ ] **Step 1: Write the failing verification script**

Create `scripts/instrumentation/check_trackstate_branches.py` in the cCKF repo:

```python
"""Verify cCKF instrumentation branches in a trackstates ROOT file.

Checks, for every branch added by the ACTS instrumentation patches:
  1. the branch exists in the tree;
  2. it is *state-aligned* -- per entry, its length equals ``nStates``;
  3. it is finite wherever the spec says it must be.

Exit code 0 means every check passed.

Usage
-----
    python scripts/instrumentation/check_trackstate_branches.py trackstates.root
    python scripts/instrumentation/check_trackstate_branches.py out.root --patches A B
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass

import awkward as ak
import numpy as np
import uproot


@dataclass(frozen=True)
class BranchSpec:
    """Expectation for one instrumentation branch.

    Parameters
    ----------
    name
        ROOT branch name.
    patch
        Which patch introduced it: ``"A"``, ``"B"`` or ``"C"``.
    finite_where
        Which entries must be finite. ``"measurement"`` means entries whose
        ``dim_hit``-equivalent state is a measurement; ``"all"`` means every
        entry; ``"any"`` means at least one finite entry in the file.
    """

    name: str
    patch: str
    finite_where: str = "any"


# Extended by Tasks 3, 4 and 5.
EXPECTED: list[BranchSpec] = [
    BranchSpec("S00_prt", "A", finite_where="any"),
    BranchSpec("S01_prt", "A", finite_where="any"),
    BranchSpec("S11_prt", "A", finite_where="any"),
]

# Branch known to be state-aligned in stock ACTS; used as the reference length.
REFERENCE_BRANCH = "volume_id"


def check_file(
    path: str,
    tree: str = "trackstates",
    patches: tuple[str, ...] = ("A", "B", "C"),
) -> list[str]:
    """Check one trackstates file and return a list of failure messages.

    Parameters
    ----------
    path
        Path to the ROOT file.
    tree
        Name of the TTree.
    patches
        Which patches' branches to require. Lets Task 2 run before Patches
        B and C exist.

    Returns
    -------
    list of str
        Human-readable failures. Empty means everything passed.
    """
    failures: list[str] = []
    specs = [s for s in EXPECTED if s.patch in patches]

    with uproot.open(f"{path}:{tree}") as t:
        available = set(t.keys())

        missing = [s.name for s in specs if s.name not in available]
        if missing:
            failures.append(f"missing branches: {sorted(missing)}")
        specs = [s for s in specs if s.name in available]
        if not specs:
            return failures

        if REFERENCE_BRANCH not in available:
            failures.append(f"reference branch {REFERENCE_BRANCH!r} not in tree")
            return failures

        arrays = t.arrays([REFERENCE_BRANCH] + [s.name for s in specs])
        n_states = ak.num(arrays[REFERENCE_BRANCH], axis=1)

        for spec in specs:
            col = arrays[spec.name]

            lengths = ak.num(col, axis=1)
            bad = ak.sum(lengths != n_states)
            if bad:
                failures.append(
                    f"{spec.name}: not state-aligned -- {bad} of "
                    f"{len(lengths)} tracks have length != nStates"
                )

            flat = np.asarray(ak.flatten(col))
            if flat.size == 0:
                failures.append(f"{spec.name}: no entries at all")
                continue

            finite = np.isfinite(flat)
            if spec.finite_where == "all" and not finite.all():
                failures.append(
                    f"{spec.name}: {int((~finite).sum())} of {flat.size} "
                    f"entries are non-finite, expected all finite"
                )
            elif spec.finite_where == "any" and not finite.any():
                failures.append(
                    f"{spec.name}: all {flat.size} entries are non-finite"
                )

    return failures


def main() -> int:
    """CLI entry point. Returns a process exit code."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", help="path to trackstates ROOT file")
    parser.add_argument("--tree", default="trackstates", help="TTree name")
    parser.add_argument(
        "--patches",
        nargs="+",
        default=["A", "B", "C"],
        choices=["A", "B", "C"],
        help="which patches' branches to require",
    )
    args = parser.parse_args()

    failures = check_file(args.path, args.tree, tuple(args.patches))
    if failures:
        print(f"FAIL ({len(failures)} problem(s)) in {args.path}")
        for f in failures:
            print(f"  - {f}")
        return 1

    print(f"PASS  all checks for patches {args.patches} in {args.path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Run it against a pre-patch file to verify it fails**

You need a baseline `trackstates.root` produced by the *unpatched* build. If one is not already on disk, produce it now — it also serves as the numerical-no-op baseline for Task 6.

```bash
cd ~/SURP/cCKF
# Enable the track states writer for a 1-event run. digi_and_reco.py currently
# passes writeTrackStates=False at several call sites; flip the one on the CKF
# path for this run only (see Task 6 Step 1 for the flag plumbing).
python digi_and_reco.py --events 1 --output-dir "$PWD/output/instr_baseline" \
    --write-track-states
python scripts/instrumentation/check_trackstate_branches.py \
    output/instr_baseline/trackstates_ckf.root --patches A
```

Expected: `FAIL (1 problem(s))` with `missing branches: ['S00_prt', 'S01_prt', 'S11_prt']`.

**`digi_and_reco.py` has no `--write-track-states` flag yet — add it now**, since every later task needs it. There are four `writeTrackStates=False` call sites (cCKF lines ~584, 653, 670, 694); only the CKF one changes.

In the argument parser (near the existing `create_base_parser("Digitization and reconstruction for ACTS")` call around line 140), add:

```python
    parser.add_argument(
        "--write-track-states",
        action="store_true",
        help=(
            "Write the per-TrackState ROOT tree from the CKF. Off by default "
            "because the tree is large; required for cCKF gate feature "
            "collection and for the instrumentation verifier."
        ),
    )
```

Then find the CKF reconstruction call — the one at line ~584, which is the `addCKFTracks`-style call on the main reconstruction path, **not** the fitter calls at ~653/670/694 — and change:

```python
            writeTrackStates=False,
```

to:

```python
            writeTrackStates=args.write_track_states,
```

Confirm you edited the right one:

```bash
cd ~/SURP/cCKF && grep -n -B12 "writeTrackStates" digi_and_reco.py | grep -n "CKF\|ckf\|addCKF\|writeTrackStates"
```

The line you changed should sit inside the CKF block. Leave the three fitter call sites at `False` — their track containers never register the `pathInX0_interval` column, so their output would be uniformly NaN for Patch B and is not wanted.

- [ ] **Step 3: Declare the members in the writer header**

In `RootTrackStatesWriter.hpp`, immediately after the `std::vector<float> m_pull_x_hit;` / `m_pull_y_hit;` declarations (upstream ~line 199), add:

```cpp
  // --- cCKF instrumentation (Patch A) -------------------------------------
  // Innovation (residual) covariance S = V + H C H^T evaluated with the
  // *predicted* state, i.e. the matrix that defines the CKF search ellipse.
  // Stock ACTS computes this internally but persists only sqrt of its
  // diagonal. cCKF needs the full 2x2 to build chol(S) for the gate feature
  // vector (spec 8.2).
  //
  // These are pushed once per track state, so they are index-parallel with
  // m_volumeID / m_layerID / m_chi2 / m_pathLength. They are deliberately NOT
  // pushed alongside the m_*_hit branches, which are not state-aligned.
  //
  /// S(0,0). NaN if the state has no predicted parameters or no measurement.
  std::vector<float> m_S00_prt;
  /// S(0,1). NaN as above, and NaN for 1D measurements where S is 1x1.
  std::vector<float> m_S01_prt;
  /// S(1,1). NaN as above, and NaN for 1D measurements where S is 1x1.
  std::vector<float> m_S11_prt;
```

- [ ] **Step 4: Register the branches**

In `RootTrackStatesWriter.cpp`, immediately after the line matching anchor A3
(`m_outputTree->Branch("dim_hit", &m_dim_hit);`), add:

```cpp
  // cCKF instrumentation (Patch A): full innovation covariance.
  m_outputTree->Branch("S00_prt", &m_S00_prt);
  m_outputTree->Branch("S01_prt", &m_S01_prt);
  m_outputTree->Branch("S11_prt", &m_S11_prt);
```

- [ ] **Step 5: Fill them in the outer state loop**

In `writeT`, immediately after the line matching anchor A2
(`m_chi2.push_back(state.chi2());`), add:

```cpp
      // cCKF instrumentation (Patch A): innovation covariance
      //   S = V + H C_pred H^T
      // computed here, in the outer per-state loop, so the branches stay
      // index-aligned with volume_id / layer_id / chi2 / pathLength. The
      // existing *_hit branches are computed in the parameter loop below,
      // which skips states without a source link, and are therefore shorter
      // than nStates -- do not copy that placement.
      {
        float s00 = nan;
        float s01 = nan;
        float s11 = nan;

        if (state.hasPredicted() && state.hasCalibrated()) {
          const Acts::DynamicMatrix H =
              state.projectorSubspaceHelper().fullProjector().topLeftCorner(
                  state.calibratedSize(), Acts::eBoundSize);
          const Acts::DynamicMatrix V = state.effectiveCalibratedCovariance();
          const Acts::DynamicMatrix S =
              V + H * state.predictedCovariance() * H.transpose();

          s00 = Acts::clampValue<float>(S(0, 0));
          // A 1D measurement (e.g. a strip) gives a 1x1 S; the off-diagonal
          // and S11 are genuinely undefined there, so they stay NaN.
          if (state.calibratedSize() >= 2) {
            s01 = Acts::clampValue<float>(S(0, 1));
            s11 = Acts::clampValue<float>(S(1, 1));
          }
        }

        m_S00_prt.push_back(s00);
        m_S01_prt.push_back(s01);
        m_S11_prt.push_back(s11);
      }
```

`nan` is already in scope — it is declared as `constexpr float nan = std::numeric_limits<float>::quiet_NaN();` at the top of `writeT` (upstream line 311).

- [ ] **Step 6: Clear them**

In the per-track reset block, immediately after the line matching anchor A4
(`m_dim_hit.clear();`), add:

```cpp
    // cCKF instrumentation (Patch A)
    m_S00_prt.clear();
    m_S01_prt.clear();
    m_S11_prt.clear();
```

- [ ] **Step 7: Build**

```bash
cd "$ACTS_FORK/../acts-build" 2>/dev/null || { echo "set your build dir"; exit 1; }
cmake --build . --target ActsExamplesIoRoot -j"$(nproc 2>/dev/null || sysctl -n hw.ncpu)"
```

Expected: compiles clean. If `topLeftCorner` on the projector complains about types, confirm you copied the expression verbatim from the existing `resCov` block (anchor A1) — it is the same call.

- [ ] **Step 8: Run one event and verify the branches pass**

```bash
cd ~/SURP/cCKF
python digi_and_reco.py --events 1 --output-dir "$PWD/output/instr_patchA" \
    --write-track-states
python scripts/instrumentation/check_trackstate_branches.py \
    output/instr_patchA/trackstates_ckf.root --patches A
```

Expected: `PASS  all checks for patches ['A'] in ...`.

- [ ] **Step 9: Verify S is consistent with the existing pull branches**

This catches a wrong projector or a transposed covariance, which the finiteness check would not.

```bash
cd ~/SURP/cCKF
python3 - <<'PY'
import awkward as ak, numpy as np, uproot
t = uproot.open("output/instr_patchA/trackstates_ckf.root:trackstates")
a = t.arrays(["S00_prt", "err_x_hit", "res_x_hit", "pull_x_hit"])
s00 = np.asarray(ak.flatten(a["S00_prt"]))
s00 = s00[np.isfinite(s00) & (s00 > 0)]
print(f"S00: n={s00.size} min={s00.min():.4g} median={np.median(s00):.4g} max={s00.max():.4g}")
assert (s00 > 0).all(), "S00 must be strictly positive (it is a variance)"
# S00 = V00 + (H C H^T)00 >= V00 = err_x_hit^2, so sqrt(S00) >= err_x_hit.
err = np.asarray(ak.flatten(a["err_x_hit"]))
err = err[np.isfinite(err)]
print(f"err_x_hit: median={np.median(err):.4g}")
print(f"sqrt(S00) median={np.median(np.sqrt(s00)):.4g}  (must be >= err median)")
assert np.median(np.sqrt(s00)) >= np.median(err), "sqrt(S00) below measurement error -- projector or covariance is wrong"
print("OK")
PY
```

Expected: prints `OK`. Note the two arrays have different lengths (the alignment trap), so this compares distributions, not element-by-element.

- [ ] **Step 10: Also check S01 symmetry and positive-definiteness**

```bash
cd ~/SURP/cCKF
python3 - <<'PY'
import awkward as ak, numpy as np, uproot
t = uproot.open("output/instr_patchA/trackstates_ckf.root:trackstates")
a = t.arrays(["S00_prt", "S01_prt", "S11_prt"])
s00 = np.asarray(ak.flatten(a["S00_prt"]))
s01 = np.asarray(ak.flatten(a["S01_prt"]))
s11 = np.asarray(ak.flatten(a["S11_prt"]))
m = np.isfinite(s00) & np.isfinite(s01) & np.isfinite(s11)
s00, s01, s11 = s00[m], s01[m], s11[m]
print(f"2D measurements: {s00.size}")
det = s00 * s11 - s01 * s01
bad = int((det <= 0).sum())
print(f"det(S) min={det.min():.4g}  non-positive-definite count={bad}")
assert bad == 0, "S must be positive definite everywhere"
rho = s01 / np.sqrt(s00 * s11)
print(f"correlation rho: min={rho.min():.3f} max={rho.max():.3f}")
assert np.all(np.abs(rho) <= 1.0 + 1e-6), "correlation out of [-1,1]"
print("OK")
PY
```

Expected: `OK`, with a non-trivial spread of `rho` — if every `rho` is exactly zero, the off-diagonal is not actually being read and the patch is a no-op.

- [ ] **Step 11: Format and commit**

```bash
cd "$ACTS_FORK"
clang-format -i \
  Examples/Io/Root/include/ActsExamples/Io/Root/RootTrackStatesWriter.hpp \
  Examples/Io/Root/src/RootTrackStatesWriter.cpp
git add Examples/Io/Root/include/ActsExamples/Io/Root/RootTrackStatesWriter.hpp \
        Examples/Io/Root/src/RootTrackStatesWriter.cpp
git commit -m "feat(cckf): write full innovation covariance S00/S01/S11 per track state

The CKF search ellipse is defined by S = V + H C H^T, which
RootTrackStatesWriter already computes but persists only as the sqrt of its
diagonal. cCKF's gate feature vector needs chol(S), so write all three
independent components.

Pushed in the outer per-state loop so the branches are index-parallel with
volume_id/layer_id/chi2/pathLength. NaN for states without predicted
parameters or a calibrated measurement; S01/S11 additionally NaN for 1D
measurements where S is 1x1."
```

```bash
cd ~/SURP/cCKF
git add scripts/instrumentation/check_trackstate_branches.py
git commit -m "test: add trackstates instrumentation branch verifier"
```

---

## Task 3: Patch B — accumulated X/X₀ per interval

Thread the material slab's radiation-length thickness out of the propagator, accumulate it between measurement surfaces, and stamp it on each TrackState.

**Files:**
- Modify: `Core/include/Acts/Propagator/detail/PointwiseMaterialInteraction.hpp`
- Modify: `Core/include/Acts/TrackFinding/CombinatorialKalmanFilter.hpp`
- Modify: `Examples/Algorithms/TrackFinding/src/TrackFindingAlgorithm.cpp`

**Interfaces:**
- Consumes: nothing from Task 2 (independent patch).
- Produces:
  - `Acts::detail::PointwiseMaterialEffects::pathXOverX0` — `double`, radiation lengths of the slab traversed at this interaction. Zero for vacuum and for the slab-taking overload, which never sees a surface.
  - `Acts::CkfConstants::kPathInX0Interval` — `constexpr std::string_view`, value `"pathInX0_interval"`. The name of the optional `double` track-state dynamic column.
  - `Acts::CombinatorialKalmanFilterResult<T>::accumulatedXOverX0` — `double`, running total since the last measurement surface.
  Task 4 reads the column from the writer.

- [ ] **Step 1: Add the field to `PointwiseMaterialEffects`**

In `Core/include/Acts/Propagator/detail/PointwiseMaterialInteraction.hpp`, at anchor B1 (`struct PointwiseMaterialEffects {`), add a fifth field:

```cpp
/// Struct to hold the material effects computed at a pointwise interaction
struct PointwiseMaterialEffects {
  double eLoss = 0;
  double variancePhi = 0;
  double varianceTheta = 0;
  double varianceQoverP = 0;
  /// cCKF instrumentation: thickness of the traversed material slab in
  /// radiation lengths, i.e. MaterialSlab::thicknessInX0().
  ///
  /// Only the surface-level performMaterialInteraction() overload sets this,
  /// because it is the only one that evaluates (and then discards) the slab.
  /// The slab-taking overload leaves it at zero -- callers that need it must
  /// go through the surface-level entry point. Zero for vacuum.
  double pathXOverX0 = 0;
};
```

- [ ] **Step 2: Populate it in the surface-level overload**

Still in `PointwiseMaterialInteraction.hpp`, in the `Result<PointwiseMaterialEffects> performMaterialInteraction(...const Surface& surface...)` overload. Find:

```cpp
  const PointwiseMaterialEffects effects = performMaterialInteraction(
      state, stepper, slab, noiseUpdateMode, multipleScattering, energyLoss);
```

Replace with:

```cpp
  PointwiseMaterialEffects effects = performMaterialInteraction(
      state, stepper, slab, noiseUpdateMode, multipleScattering, energyLoss);
  // cCKF instrumentation: the slab is in scope only here. Record its
  // radiation-length thickness before it is discarded. Note this is the
  // *slab* thickness in X0 as returned by the material decorator, which for a
  // surface-projected slab already accounts for the incidence angle.
  effects.pathXOverX0 = slab.thicknessInX0();
```

Only the `const` is dropped and one line added. Do not touch anything else in that function.

- [ ] **Step 3: Add the column-name constant and the accumulator**

In `Core/include/Acts/TrackFinding/CombinatorialKalmanFilter.hpp`, just before anchor B3 (`struct CombinatorialKalmanFilterResult {`), add the namespace:

```cpp
/// Constants for optional cCKF instrumentation columns on the CKF output.
namespace CkfConstants {

/// Name of the optional per-track-state dynamic column (type: double) holding
/// the material X/X0 accumulated since the previous measurement surface.
///
/// The CKF writes this column only if the caller registered it on the track
/// state backend; see TrackFindingAlgorithm. This keeps the column opt-in so
/// existing users pay nothing.
constexpr std::string_view kPathInX0Interval = "pathInX0_interval";

}  // namespace CkfConstants
```

Then inside `CombinatorialKalmanFilterResult`, after the `PathLimitReached pathLimitReached;` member, add:

```cpp
  /// cCKF instrumentation: material X/X0 accumulated since the previous
  /// measurement surface, for the branch currently being propagated.
  ///
  /// A single scalar is sufficient despite branching. The CKF search is
  /// depth-first, so exactly one branch propagates between two measurement
  /// surfaces; forking in processNewTrackStates() happens *after* the
  /// pre-update material call, so every candidate on a surface legitimately
  /// shares the same interval. When the search backtracks, reset() zeroes this
  /// and re-applies the resumed branch's post-update material, which
  /// re-establishes the correct interval start.
  double accumulatedXOverX0 = 0;
```

Ensure `#include <string_view>` is present in this header; add it to the include block if not.

- [ ] **Step 4: Zero and accumulate in `reset()`**

In `reset()`, find the material call (upstream ~line 418):

```cpp
      const Result<detail::PointwiseMaterialEffects> materialInteractionRes =
          detail::performMaterialInteraction(
              state, stepper, currentState.referenceSurface(),
```

Immediately **before** that statement insert:

```cpp
      // cCKF instrumentation: the search has backtracked to a different
      // branch. Its previous measurement surface is the one we are resuming
      // from, so the interval restarts here; the post-update material applied
      // just below is the first contribution to the new interval.
      result.accumulatedXOverX0 = 0;
```

and immediately **after** the `if (!materialInteractionRes.ok()) { ... }` error block insert:

```cpp
      result.accumulatedXOverX0 += materialInteractionRes->pathXOverX0;
```

- [ ] **Step 5: Accumulate at the pre-update call in `filter()`**

Find the pre-update call (upstream ~line 488) and its error block:

```cpp
      if (!materialInteractionPreRes.ok()) {
        ACTS_DEBUG("Material interaction failed during filter: "
                   << materialInteractionPreRes.error().message());
        return materialInteractionPreRes.error();
      }
```

Immediately after that closing brace, insert:

```cpp
      // cCKF instrumentation: material in front of this surface belongs to the
      // interval that ends at this surface.
      result.accumulatedXOverX0 += materialInteractionPreRes->pathXOverX0;
```

- [ ] **Step 6: Accumulate at the post-update call in `filter()`**

Find the post-update call (upstream ~line 624) and its error block:

```cpp
      if (!materialInteractionPostRes.ok()) {
        ACTS_DEBUG("Material interaction failed during filter: "
                   << materialInteractionPostRes.error().message());
        return materialInteractionPostRes.error();
      }
```

Immediately after that closing brace, insert:

```cpp
      // cCKF instrumentation: material behind this surface belongs to the
      // *next* interval. The accumulator was already zeroed when the states on
      // this surface were stamped in processNewTrackStates().
      result.accumulatedXOverX0 += materialInteractionPostRes->pathXOverX0;
```

- [ ] **Step 7: Stamp the new track states and reset the accumulator**

In `processNewTrackStates`, find anchor B4 and the loop that builds `newBranches`, ending with:

```cpp
      // Remove the root branch
      result.activeBranches.pop_back();
```

Immediately **before** that `pop_back()` line, insert:

```cpp
      // cCKF instrumentation: stamp every state created on this surface with
      // the material accumulated since the previous measurement surface. All
      // candidates share the value because propagation -- and therefore the
      // pre-update material -- happens before hit branching. Then restart the
      // interval; the post-update material applied after filtering will be the
      // first contribution to the next one.
      if (result.trackStates != nullptr &&
          result.trackStates->hasColumn(
              hashString(CkfConstants::kPathInX0Interval))) {
        for (TrackProxy branch : newBranches) {
          branch.outermostTrackState()
              .template component<double, hashString(
                                              CkfConstants::kPathInX0Interval)>() =
              result.accumulatedXOverX0;
        }
      }
      result.accumulatedXOverX0 = 0;
```

Note the reset is **outside** the `hasColumn` guard: the accumulator's interval semantics must not depend on whether anyone is recording it, otherwise enabling the column would change nothing but disabling it would silently produce a running total.

Confirm `hashString` is reachable — the header should already pull in `Acts/Utilities/HashedString.hpp` transitively; if the build complains, add `#include "Acts/Utilities/HashedString.hpp"`.

- [ ] **Step 8: Register the column in `TrackFindingAlgorithm`**

In `Examples/Algorithms/TrackFinding/src/TrackFindingAlgorithm.cpp`, immediately after anchor B5 (`tracks.addColumn<unsigned int>("trackGroup");` and the matching `tracksTemp` line), add:

```cpp
  // cCKF instrumentation: opt in to the CKF's per-track-state interval X/X0
  // column. Must exist on BOTH backends -- the CKF builds candidates in
  // tracksTemp and TrackProxy::copyFrom only carries a dynamic column across
  // if the destination backend also declares it.
  trackStateContainer->addColumn<double>(
      std::string(Acts::CkfConstants::kPathInX0Interval));
  trackStateContainerTemp->addColumn<double>(
      std::string(Acts::CkfConstants::kPathInX0Interval));
```

Add `#include "Acts/TrackFinding/CombinatorialKalmanFilter.hpp"` if the file does not already include it (it almost certainly does).

- [ ] **Step 9: Build**

```bash
cd "$ACTS_BUILD"
cmake --build . --target ActsExamplesTrackFinding -j"$(nproc 2>/dev/null || sysctl -n hw.ncpu)"
```

Expected: compiles clean. Two likely failures: (a) `hashString` not found → add the include from Step 7; (b) `component<double, ...>` complaining the key is not `constexpr` → confirm you wrote `hashString(CkfConstants::kPathInX0Interval)` with `kPathInX0Interval` declared `constexpr std::string_view`.

- [ ] **Step 10: Verify the column survives the temp→final copy**

This is the single most likely silent failure in Patch B: if `copyFrom` drops the column, every value reads back as zero. Test it before touching the writer.

```bash
cd "$ACTS_BUILD"
cmake --build . --target ActsUnitTestCombinatorialKalmanFilter -j4 2>/dev/null \
  && ./bin/ActsUnitTestCombinatorialKalmanFilter
```

If that target does not exist in the fork, skip it — Step 12 covers the same ground end to end. Do not spend more than a few minutes hunting for the target name.

- [ ] **Step 11: Add the writer branch**

Header — after the Patch A members in `RootTrackStatesWriter.hpp`:

```cpp
  // --- cCKF instrumentation (Patch B) -------------------------------------
  /// Material X/X0 traversed between the previous measurement surface and this
  /// one, read from the CKF's optional "pathInX0_interval" track state column.
  /// NaN if the column is absent (e.g. output of a fitter rather than the CKF).
  std::vector<float> m_pathInX0_interval;
```

Registration — after the Patch A `Branch` calls in `RootTrackStatesWriter.cpp`:

```cpp
  // cCKF instrumentation (Patch B): per-interval material budget.
  m_outputTree->Branch("pathInX0_interval", &m_pathInX0_interval);
```

Fill — in the outer state loop, immediately after the Patch A block from Task 2 Step 5:

```cpp
      // cCKF instrumentation (Patch B): accumulated X/X0 since the previous
      // measurement surface, stamped by the CKF actor. Optional column: NaN
      // when this track container did not come from an instrumented CKF.
      {
        constexpr auto kKey =
            Acts::hashString(Acts::CkfConstants::kPathInX0Interval);
        m_pathInX0_interval.push_back(
            state.has(kKey)
                ? Acts::clampValue<float>(state.template component<double, kKey>())
                : nan);
      }
```

Clear — after the Patch A clears:

```cpp
    // cCKF instrumentation (Patch B)
    m_pathInX0_interval.clear();
```

Add these includes to `RootTrackStatesWriter.cpp` if absent:

```cpp
#include "Acts/TrackFinding/CombinatorialKalmanFilter.hpp"
#include "Acts/Utilities/HashedString.hpp"
```

- [ ] **Step 12: Extend the verification script**

In `scripts/instrumentation/check_trackstate_branches.py`, extend `EXPECTED`:

```python
EXPECTED: list[BranchSpec] = [
    BranchSpec("S00_prt", "A", finite_where="any"),
    BranchSpec("S01_prt", "A", finite_where="any"),
    BranchSpec("S11_prt", "A", finite_where="any"),
    # Patch B: the CKF stamps every state, so this must be finite everywhere.
    BranchSpec("pathInX0_interval", "B", finite_where="all"),
]
```

- [ ] **Step 13: Build, run one event, verify**

```bash
cd "$ACTS_BUILD" && cmake --build . -j"$(nproc 2>/dev/null || sysctl -n hw.ncpu)"
cd ~/SURP/cCKF
python digi_and_reco.py --events 1 --output-dir "$PWD/output/instr_patchB" \
    --write-track-states
python scripts/instrumentation/check_trackstate_branches.py \
    output/instr_patchB/trackstates_ckf.root --patches A B
```

Expected: `PASS  all checks for patches ['A', 'B'] in ...`.

- [ ] **Step 14: Verify the values are physically sensible, not all zero**

The finiteness check passes trivially if every value is 0.0. This step is the real test.

```bash
cd ~/SURP/cCKF
python3 - <<'PY'
import awkward as ak, numpy as np, uproot
t = uproot.open("output/instr_patchB/trackstates_ckf.root:trackstates")
a = t.arrays(["pathInX0_interval", "volume_id", "layer_id"])
x = np.asarray(ak.flatten(a["pathInX0_interval"]))
vol = np.asarray(ak.flatten(a["volume_id"]))
print(f"n={x.size} zeros={int((x == 0).sum())} "
      f"min={x.min():.5g} median={np.median(x):.5g} max={x.max():.5g}")
assert np.isfinite(x).all(), "non-finite X/X0"
assert (x >= 0).all(), "X/X0 must be non-negative"
frac_zero = (x == 0).mean()
assert frac_zero < 0.5, (
    f"{frac_zero:.1%} of intervals are exactly zero -- the column is probably "
    "not being written (check the temp->final copyFrom in Step 10)"
)
# ODD layers are thin: a per-interval budget of order 1e-3 .. 1e-1 X0 is
# expected. A median above 1 would mean the accumulator is never reset.
assert np.median(x) < 1.0, "median X/X0 > 1 -- accumulator is not being reset"
print("per-volume median X/X0:")
for v in np.unique(vol):
    m = vol == v
    print(f"  volume {v:>3}: n={m.sum():>7} median={np.median(x[m]):.5g}")
print("OK")
PY
```

Expected: `OK`, a median in the `1e-3`–`1e-1` range, and a clear per-volume structure (endcap/barrel transitions traverse more material). A median above 1, or a monotonically growing distribution, means the reset in Step 7 is not firing.

- [ ] **Step 15: Format and commit**

```bash
cd "$ACTS_FORK"
clang-format -i \
  Core/include/Acts/Propagator/detail/PointwiseMaterialInteraction.hpp \
  Core/include/Acts/TrackFinding/CombinatorialKalmanFilter.hpp \
  Examples/Algorithms/TrackFinding/src/TrackFindingAlgorithm.cpp \
  Examples/Io/Root/include/ActsExamples/Io/Root/RootTrackStatesWriter.hpp \
  Examples/Io/Root/src/RootTrackStatesWriter.cpp
git add Core/include/Acts/Propagator/detail/PointwiseMaterialInteraction.hpp \
        Core/include/Acts/TrackFinding/CombinatorialKalmanFilter.hpp \
        Examples/Algorithms/TrackFinding/src/TrackFindingAlgorithm.cpp \
        Examples/Io/Root/include/ActsExamples/Io/Root/RootTrackStatesWriter.hpp \
        Examples/Io/Root/src/RootTrackStatesWriter.cpp
git commit -m "feat(cckf): record accumulated material X/X0 per measurement interval

performMaterialInteraction evaluated the material slab and discarded it,
keeping only the scattering variances. Carry thicknessInX0() out on
PointwiseMaterialEffects, accumulate it in the CKF actor, and stamp the total
onto every track state created at a measurement surface via an opt-in
'pathInX0_interval' dynamic column.

A single scalar accumulator is sufficient: the CKF search is depth-first, so
one branch propagates between measurement surfaces, forking happens after the
pre-update material call (hence all candidates on a surface legitimately share
the interval), and reset() re-establishes the interval start on backtrack.

The column must be registered on both the temporary and final track state
backends or copyFrom drops it."
```

```bash
cd ~/SURP/cCKF
git add scripts/instrumentation/check_trackstate_branches.py
git commit -m "test: check pathInX0_interval branch"
```

---

## Task 4: Patch C part 1 — pure helpers for cluster moments and incidence angles

Two side-effect-free functions with real unit tests, before any writer wiring. Written first because they contain all the actual maths in Patch C.

**Files:**
- Create: `Examples/Framework/include/ActsExamples/EventData/ClusterFeatures.hpp`
- Create: `Examples/Framework/src/EventData/ClusterFeatures.cpp`
- Modify: `Examples/Framework/CMakeLists.txt`
- Create: `Tests/UnitTests/Examples/EventData/ClusterFeaturesTests.cpp`
- Modify: `Tests/UnitTests/Examples/EventData/CMakeLists.txt`

**Interfaces:**
- Consumes: `ActsExamples::Cluster` (`Cluster.hpp`), `Acts::RotationMatrix3`, `Acts::Vector3`.
- Produces, in namespace `ActsExamples`:
  - `struct ClusterMoments { double sigmaUU; double sigmaUV; double sigmaVV; };`
  - `ClusterMoments clusterChargeMoments(const Cluster& cluster);`
  - `struct IncidenceAngles { double alphaU; double alphaV; };`
  - `IncidenceAngles incidenceAngles(const Acts::RotationMatrix3& referenceFrame, const Acts::Vector3& direction);`
  Task 5 calls both from `RootTrackStatesWriter`.

- [ ] **Step 1: Write the failing test**

Create `Tests/UnitTests/Examples/EventData/ClusterFeaturesTests.cpp`:

```cpp
// This file is part of the ACTS project.
//
// Copyright (C) 2016 CERN for the benefit of the ACTS project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include <boost/test/unit_test.hpp>

#include "Acts/Definitions/Algebra.hpp"
#include "ActsExamples/EventData/ClusterFeatures.hpp"

#include <cmath>
#include <numbers>

namespace {

/// Build a cluster cell whose path2D midpoint is (u, v) and whose charge is q.
ActsExamples::Cluster::Cell makeCell(double u, double v, double q) {
  // A degenerate segment: start == end == the cell position. The moment
  // helper uses the midpoint, so this places all the charge at exactly (u, v).
  return ActsExamples::Cluster::Cell({0, 0},
                                     {Acts::Vector2(u, v), Acts::Vector2(u, v)},
                                     q);
}

}  // namespace

BOOST_AUTO_TEST_SUITE(ClusterFeatures)

BOOST_AUTO_TEST_CASE(MomentsOfSingleCellAreZero) {
  ActsExamples::Cluster c;
  c.channels.push_back(makeCell(1.0, 2.0, 5.0));

  const auto m = ActsExamples::clusterChargeMoments(c);

  BOOST_CHECK_SMALL(m.sigmaUU, 1e-12);
  BOOST_CHECK_SMALL(m.sigmaUV, 1e-12);
  BOOST_CHECK_SMALL(m.sigmaVV, 1e-12);
}

BOOST_AUTO_TEST_CASE(MomentsOfEmptyClusterAreZero) {
  const ActsExamples::Cluster c;

  const auto m = ActsExamples::clusterChargeMoments(c);

  BOOST_CHECK_SMALL(m.sigmaUU, 1e-12);
  BOOST_CHECK_SMALL(m.sigmaUV, 1e-12);
  BOOST_CHECK_SMALL(m.sigmaVV, 1e-12);
}

BOOST_AUTO_TEST_CASE(MomentsOfTwoEqualCellsAlongU) {
  // Charges 1 at u = -1 and u = +1, both at v = 0.
  // Centroid u = 0, so sigma_uu = (1*1 + 1*1) / 2 = 1. No v spread.
  ActsExamples::Cluster c;
  c.channels.push_back(makeCell(-1.0, 0.0, 1.0));
  c.channels.push_back(makeCell(+1.0, 0.0, 1.0));

  const auto m = ActsExamples::clusterChargeMoments(c);

  BOOST_CHECK_CLOSE(m.sigmaUU, 1.0, 1e-9);
  BOOST_CHECK_SMALL(m.sigmaUV, 1e-12);
  BOOST_CHECK_SMALL(m.sigmaVV, 1e-12);
}

BOOST_AUTO_TEST_CASE(MomentsAreChargeWeighted) {
  // Charge 3 at u = 0, charge 1 at u = 4. Centroid u = (0*3 + 4*1)/4 = 1.
  // sigma_uu = (3*(0-1)^2 + 1*(4-1)^2) / 4 = (3 + 9) / 4 = 3.
  ActsExamples::Cluster c;
  c.channels.push_back(makeCell(0.0, 0.0, 3.0));
  c.channels.push_back(makeCell(4.0, 0.0, 1.0));

  const auto m = ActsExamples::clusterChargeMoments(c);

  BOOST_CHECK_CLOSE(m.sigmaUU, 3.0, 1e-9);
}

BOOST_AUTO_TEST_CASE(CrossMomentIsPositiveForDiagonalCluster) {
  // Charge on the u == v diagonal: sigma_uv must equal sigma_uu = sigma_vv.
  ActsExamples::Cluster c;
  c.channels.push_back(makeCell(-1.0, -1.0, 1.0));
  c.channels.push_back(makeCell(+1.0, +1.0, 1.0));

  const auto m = ActsExamples::clusterChargeMoments(c);

  BOOST_CHECK_CLOSE(m.sigmaUU, 1.0, 1e-9);
  BOOST_CHECK_CLOSE(m.sigmaVV, 1.0, 1e-9);
  BOOST_CHECK_CLOSE(m.sigmaUV, 1.0, 1e-9);
}

BOOST_AUTO_TEST_CASE(CrossMomentIsNegativeForAntiDiagonalCluster) {
  ActsExamples::Cluster c;
  c.channels.push_back(makeCell(-1.0, +1.0, 1.0));
  c.channels.push_back(makeCell(+1.0, -1.0, 1.0));

  const auto m = ActsExamples::clusterChargeMoments(c);

  BOOST_CHECK_CLOSE(m.sigmaUV, -1.0, 1e-9);
}

BOOST_AUTO_TEST_CASE(MomentsIgnoreNonPositiveCharge) {
  // A zero-charge cell must not shift the centroid or the moments.
  ActsExamples::Cluster c;
  c.channels.push_back(makeCell(-1.0, 0.0, 1.0));
  c.channels.push_back(makeCell(+1.0, 0.0, 1.0));
  c.channels.push_back(makeCell(100.0, 100.0, 0.0));

  const auto m = ActsExamples::clusterChargeMoments(c);

  BOOST_CHECK_CLOSE(m.sigmaUU, 1.0, 1e-9);
}

BOOST_AUTO_TEST_CASE(NormalIncidenceGivesZeroAngles) {
  // Identity frame: u_hat = x, v_hat = y, n_hat = z.
  const Acts::RotationMatrix3 frame = Acts::RotationMatrix3::Identity();
  const Acts::Vector3 dir(0.0, 0.0, 1.0);

  const auto a = ActsExamples::incidenceAngles(frame, dir);

  BOOST_CHECK_SMALL(a.alphaU, 1e-12);
  BOOST_CHECK_SMALL(a.alphaV, 1e-12);
}

BOOST_AUTO_TEST_CASE(FortyFiveDegreesInU) {
  const Acts::RotationMatrix3 frame = Acts::RotationMatrix3::Identity();
  const Acts::Vector3 dir = Acts::Vector3(1.0, 0.0, 1.0).normalized();

  const auto a = ActsExamples::incidenceAngles(frame, dir);

  BOOST_CHECK_CLOSE(a.alphaU, std::numbers::pi / 4, 1e-9);
  BOOST_CHECK_SMALL(a.alphaV, 1e-12);
}

BOOST_AUTO_TEST_CASE(TangentOfAngleRecoversTheRatio) {
  // The documented contract: tan(alpha_u) == (d.u_hat) / (d.n_hat).
  const Acts::RotationMatrix3 frame = Acts::RotationMatrix3::Identity();
  const Acts::Vector3 dir = Acts::Vector3(0.3, -0.5, 0.9).normalized();

  const auto a = ActsExamples::incidenceAngles(frame, dir);

  BOOST_CHECK_CLOSE(std::tan(a.alphaU), 0.3 / 0.9, 1e-9);
  BOOST_CHECK_CLOSE(std::tan(a.alphaV), -0.5 / 0.9, 1e-9);
}

BOOST_AUTO_TEST_CASE(GrazingIncidenceStaysFinite) {
  // d . n_hat == 0 exactly. The ratio form would be infinite; atan2 gives
  // +pi/2, which is what we want to persist.
  const Acts::RotationMatrix3 frame = Acts::RotationMatrix3::Identity();
  const Acts::Vector3 dir(1.0, 0.0, 0.0);

  const auto a = ActsExamples::incidenceAngles(frame, dir);

  BOOST_CHECK(std::isfinite(a.alphaU));
  BOOST_CHECK(std::isfinite(a.alphaV));
  BOOST_CHECK_CLOSE(a.alphaU, std::numbers::pi / 2, 1e-9);
}

BOOST_AUTO_TEST_CASE(RotatedFrameIsRespected) {
  // Frame rotated 90 degrees about z: u_hat = y, v_hat = -x, n_hat = z.
  Acts::RotationMatrix3 frame = Acts::RotationMatrix3::Identity();
  frame.col(0) = Acts::Vector3(0.0, 1.0, 0.0);
  frame.col(1) = Acts::Vector3(-1.0, 0.0, 0.0);
  frame.col(2) = Acts::Vector3(0.0, 0.0, 1.0);

  // Direction with a +x component: in the rotated frame that is -v.
  const Acts::Vector3 dir = Acts::Vector3(1.0, 0.0, 1.0).normalized();

  const auto a = ActsExamples::incidenceAngles(frame, dir);

  BOOST_CHECK_SMALL(a.alphaU, 1e-12);
  BOOST_CHECK_CLOSE(a.alphaV, -std::numbers::pi / 4, 1e-9);
}

BOOST_AUTO_TEST_SUITE_END()
```

- [ ] **Step 2: Register the test target**

In `Tests/UnitTests/Examples/EventData/CMakeLists.txt`, add after the existing `add_unittest` lines:

```cmake
add_unittest(ClusterFeatures ClusterFeaturesTests.cpp)
```

- [ ] **Step 3: Run the test to verify it fails**

```bash
cd "$ACTS_BUILD"
cmake . >/dev/null && cmake --build . --target ActsUnitTestClusterFeatures -j4
```

Expected: **build failure**, `fatal error: ActsExamples/EventData/ClusterFeatures.hpp: No such file or directory`.

- [ ] **Step 4: Write the header**

Create `Examples/Framework/include/ActsExamples/EventData/ClusterFeatures.hpp`:

```cpp
// This file is part of the ACTS project.
//
// Copyright (C) 2016 CERN for the benefit of the ACTS project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#pragma once

#include "Acts/Definitions/Algebra.hpp"
#include "ActsExamples/EventData/Cluster.hpp"

namespace ActsExamples {

/// Charge-weighted second central moments of a cluster's charge distribution,
/// in the surface's local (u, v) frame.
///
/// These are the sigma_uu / sigma_uv / sigma_vv entries of the cCKF gate
/// feature vector. ACTS digitization does not compute them, so they are
/// derived here from the per-channel charge deposits.
struct ClusterMoments {
  /// Second central moment along local u, in (length unit)^2.
  double sigmaUU = 0;
  /// Mixed second central moment, in (length unit)^2. Signed.
  double sigmaUV = 0;
  /// Second central moment along local v, in (length unit)^2.
  double sigmaVV = 0;
};

/// Compute the charge-weighted second central moments of a cluster.
///
/// Each channel contributes its charge (``Cell::activation``) at the midpoint
/// of its ``path2D`` segment, which the Fatras Segmentizer fills with the
/// entry/exit points of the track segment through that channel, expressed in
/// local surface cartesian coordinates. Working from ``path2D`` rather than
/// the bin index means the result is in physical length units and needs no
/// access to the module's BinUtility.
///
/// Channels with non-positive charge are skipped. A cluster with no channels,
/// or with zero total charge, yields all-zero moments -- these are population
/// moments, not sample moments, so a single-channel cluster correctly gives
/// exactly zero rather than an undefined value.
///
/// @param cluster The cluster whose channels carry the charge deposits
/// @return The three independent second central moments
ClusterMoments clusterChargeMoments(const Cluster& cluster);

/// Track incidence angles relative to a surface's local frame.
struct IncidenceAngles {
  /// atan2(d . u_hat, d . n_hat), in radians, in (-pi, pi].
  double alphaU = 0;
  /// atan2(d . v_hat, d . n_hat), in radians, in (-pi, pi].
  double alphaV = 0;
};

/// Compute the incidence angles of a direction against a surface local frame.
///
/// The angles satisfy ``tan(alphaU) == (d . u_hat) / (d . n_hat)`` and
/// likewise for v, but unlike the raw ratios they stay finite at grazing
/// incidence (``d . n_hat -> 0``), where the ratio diverges. Since a feature
/// vector cannot carry infinities, the angle is the form we persist.
///
/// @param referenceFrame Columns are the local axes (u_hat, v_hat, n_hat), as
///                       returned by ``Acts::Surface::referenceFrame``
/// @param direction Unit direction of the track at the surface, global frame
/// @return The two incidence angles in radians
IncidenceAngles incidenceAngles(const Acts::RotationMatrix3& referenceFrame,
                                const Acts::Vector3& direction);

}  // namespace ActsExamples
```

- [ ] **Step 5: Write the implementation**

Create `Examples/Framework/src/EventData/ClusterFeatures.cpp`:

```cpp
// This file is part of the ACTS project.
//
// Copyright (C) 2016 CERN for the benefit of the ACTS project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include "ActsExamples/EventData/ClusterFeatures.hpp"

#include <cmath>

namespace ActsExamples {

ClusterMoments clusterChargeMoments(const Cluster& cluster) {
  // Two-pass: centroid first, then moments about it. A one-pass
  // sum-of-squares formulation loses precision when the cluster sits far from
  // the local origin, which it routinely does on large sensors.
  double qTot = 0;
  double meanU = 0;
  double meanV = 0;

  for (const Cluster::Cell& cell : cluster.channels) {
    const double q = cell.activation;
    if (!(q > 0)) {
      continue;
    }
    // Midpoint of the track segment through this channel.
    const Acts::Vector2 p = 0.5 * (cell.path2D[0] + cell.path2D[1]);
    qTot += q;
    meanU += q * p[0];
    meanV += q * p[1];
  }

  if (!(qTot > 0)) {
    return {};
  }

  meanU /= qTot;
  meanV /= qTot;

  ClusterMoments moments;
  for (const Cluster::Cell& cell : cluster.channels) {
    const double q = cell.activation;
    if (!(q > 0)) {
      continue;
    }
    const Acts::Vector2 p = 0.5 * (cell.path2D[0] + cell.path2D[1]);
    const double du = p[0] - meanU;
    const double dv = p[1] - meanV;
    moments.sigmaUU += q * du * du;
    moments.sigmaUV += q * du * dv;
    moments.sigmaVV += q * dv * dv;
  }

  moments.sigmaUU /= qTot;
  moments.sigmaUV /= qTot;
  moments.sigmaVV /= qTot;

  return moments;
}

IncidenceAngles incidenceAngles(const Acts::RotationMatrix3& referenceFrame,
                                const Acts::Vector3& direction) {
  const double dU = direction.dot(referenceFrame.col(0));
  const double dV = direction.dot(referenceFrame.col(1));
  const double dN = direction.dot(referenceFrame.col(2));

  return {std::atan2(dU, dN), std::atan2(dV, dN)};
}

}  // namespace ActsExamples
```

- [ ] **Step 6: Add the source to the Framework library**

In `Examples/Framework/CMakeLists.txt`, find the `add_library(ActsExamplesFramework ...)` source list and add, in alphabetical position among the `src/EventData/` entries:

```cmake
    src/EventData/ClusterFeatures.cpp
```

If the CMakeLists uses a glob rather than an explicit list, no change is needed — verify by rerunning `cmake .` and checking the file is picked up.

- [ ] **Step 7: Run the tests and verify they pass**

```bash
cd "$ACTS_BUILD"
cmake . >/dev/null && cmake --build . --target ActsUnitTestClusterFeatures -j4
./bin/ActsUnitTestClusterFeatures --log_level=test_suite
```

Expected: `*** No errors detected`, 13 test cases run.

If `MomentsOfSingleCellAreZero` fails with a non-zero value, the `makeCell` helper's degenerate segment is not being handled — check that `path2D[0] == path2D[1]` gives a midpoint equal to the point itself.

- [ ] **Step 8: Format and commit**

```bash
cd "$ACTS_FORK"
clang-format -i \
  Examples/Framework/include/ActsExamples/EventData/ClusterFeatures.hpp \
  Examples/Framework/src/EventData/ClusterFeatures.cpp \
  Tests/UnitTests/Examples/EventData/ClusterFeaturesTests.cpp
git add Examples/Framework/include/ActsExamples/EventData/ClusterFeatures.hpp \
        Examples/Framework/src/EventData/ClusterFeatures.cpp \
        Examples/Framework/CMakeLists.txt \
        Tests/UnitTests/Examples/EventData/ClusterFeaturesTests.cpp \
        Tests/UnitTests/Examples/EventData/CMakeLists.txt
git commit -m "feat(cckf): add cluster charge-moment and incidence-angle helpers

ACTS digitization stores cluster sizes and per-channel charge but never
computes the charge second moments the cCKF gate features need. Derive them
from Cluster::channels, using each channel's path2D midpoint so the result is
in physical length units and needs no BinUtility lookup.

Incidence angles are stored as atan2(d.u, d.n) rather than the raw ratio
d.u/d.n, which diverges at grazing incidence. Same information --
tan(alpha_u) == d.u/d.n -- but always finite.

Pure functions with no ROOT or writer dependency, unit tested in isolation."
```

---

## Task 5: Patch C part 2 — wire cluster features and angles into the writer

**Files:**
- Modify: `Examples/Io/Root/include/ActsExamples/Io/Root/RootTrackStatesWriter.hpp`
- Modify: `Examples/Io/Root/src/RootTrackStatesWriter.cpp`
- Modify: `Examples/Io/Root/CMakeLists.txt` (only if `ActsExamplesFramework` is not already a link dependency — it is, so likely no change)
- Modify: `Python/Examples/src/plugins/Root.cpp`
- Modify: `Python/Examples/python/reconstruction.py`
- Modify: `scripts/instrumentation/check_trackstate_branches.py` (cCKF repo)

**Interfaces:**
- Consumes: `clusterChargeMoments`, `incidenceAngles`, `ClusterMoments`, `IncidenceAngles` from Task 4.
- Produces: ROOT branches `clus_size_u` (int), `clus_size_v` (int), `clus_qtot` (float), `clus_sigma_uu` (float), `clus_sigma_uv` (float), `clus_sigma_vv` (float), `alpha_u` (float), `alpha_v` (float) — all state-aligned. Plus a new writer config field `std::string inputClusters` (optional; empty disables the cluster branches).

- [ ] **Step 1: Extend the verification script (failing test)**

In `scripts/instrumentation/check_trackstate_branches.py`, extend `EXPECTED`:

```python
EXPECTED: list[BranchSpec] = [
    BranchSpec("S00_prt", "A", finite_where="any"),
    BranchSpec("S01_prt", "A", finite_where="any"),
    BranchSpec("S11_prt", "A", finite_where="any"),
    # Patch B: the CKF stamps every state, so this must be finite everywhere.
    BranchSpec("pathInX0_interval", "B", finite_where="all"),
    # Patch C: NaN on hole states (no cluster), so only "any".
    BranchSpec("clus_size_u", "C", finite_where="any"),
    BranchSpec("clus_size_v", "C", finite_where="any"),
    BranchSpec("clus_qtot", "C", finite_where="any"),
    BranchSpec("clus_sigma_uu", "C", finite_where="any"),
    BranchSpec("clus_sigma_uv", "C", finite_where="any"),
    BranchSpec("clus_sigma_vv", "C", finite_where="any"),
    # Patch C: angles need only a predicted state, which holes have too.
    BranchSpec("alpha_u", "C", finite_where="all"),
    BranchSpec("alpha_v", "C", finite_where="all"),
]
```

Run it against the Patch B output to confirm it now fails:

```bash
cd ~/SURP/cCKF
python scripts/instrumentation/check_trackstate_branches.py \
    output/instr_patchB/trackstates_ckf.root
```

Expected: `FAIL` with `missing branches: ['alpha_u', 'alpha_v', 'clus_qtot', ...]`.

- [ ] **Step 2: Add the config field and read handle to the writer header**

In `RootTrackStatesWriter.hpp`, inside `struct Config`, after `std::string inputMeasurementSimHitsMap;`:

```cpp
    /// Input cluster collection (cCKF instrumentation, Patch C).
    ///
    /// Optional: leave empty to disable the clus_* branches, which is what
    /// non-cCKF users and smearing-only digitization want. When set, it must
    /// name the DigitizationAlgorithm's outputClusters collection (default
    /// "clusters"), whose entries are one-to-one with measurement indices.
    std::string inputClusters;
```

Add the include near the other EventData includes:

```cpp
#include "ActsExamples/EventData/Cluster.hpp"
```

Add the read handle after `m_inputMeasurementSimHitsMap`:

```cpp
  ReadDataHandle<ClusterContainer> m_inputClusters{this, "InputClusters"};
```

Add the branch members after the Patch B member:

```cpp
  // --- cCKF instrumentation (Patch C) -------------------------------------
  // Cluster shape and charge, joined by measurement index. All -1 / NaN on
  // states with no measurement, or when inputClusters is not configured.
  //
  /// Cluster size along local u, in channels. -1 if unavailable.
  std::vector<int> m_clus_size_u;
  /// Cluster size along local v, in channels. -1 if unavailable.
  std::vector<int> m_clus_size_v;
  /// Total cluster charge, sum of channel activations. NaN if unavailable.
  std::vector<float> m_clus_qtot;
  /// Charge-weighted second central moment along u. NaN if unavailable.
  std::vector<float> m_clus_sigma_uu;
  /// Charge-weighted mixed second central moment. NaN if unavailable.
  std::vector<float> m_clus_sigma_uv;
  /// Charge-weighted second central moment along v. NaN if unavailable.
  std::vector<float> m_clus_sigma_vv;
  /// Track incidence angle atan2(d.u_hat, d.n_hat), radians. NaN if the state
  /// has no predicted parameters.
  std::vector<float> m_alpha_u;
  /// Track incidence angle atan2(d.v_hat, d.n_hat), radians. NaN as above.
  std::vector<float> m_alpha_v;
```

- [ ] **Step 3: Initialise the read handle in the constructor**

In `RootTrackStatesWriter.cpp`, in the constructor body where the other handles are initialised (search for `m_inputMeasurementSimHitsMap.initialize(`), add:

```cpp
  // cCKF instrumentation (Patch C): optional -- the clus_* branches are
  // filled with sentinels when no cluster collection is configured.
  if (!m_cfg.inputClusters.empty()) {
    m_inputClusters.initialize(m_cfg.inputClusters);
  }
```

Add these includes to the .cpp:

```cpp
#include "ActsExamples/EventData/ClusterFeatures.hpp"
```

- [ ] **Step 4: Register the branches**

After the Patch B `Branch` call:

```cpp
  // cCKF instrumentation (Patch C): cluster shape/charge and incidence angles.
  m_outputTree->Branch("clus_size_u", &m_clus_size_u);
  m_outputTree->Branch("clus_size_v", &m_clus_size_v);
  m_outputTree->Branch("clus_qtot", &m_clus_qtot);
  m_outputTree->Branch("clus_sigma_uu", &m_clus_sigma_uu);
  m_outputTree->Branch("clus_sigma_uv", &m_clus_sigma_uv);
  m_outputTree->Branch("clus_sigma_vv", &m_clus_sigma_vv);
  m_outputTree->Branch("alpha_u", &m_alpha_u);
  m_outputTree->Branch("alpha_v", &m_alpha_v);
```

- [ ] **Step 5: Read the cluster collection at the top of `writeT`**

Immediately after the existing input reads (search for `const auto& hitSimHitsMap = m_inputMeasurementSimHitsMap(ctx);`):

```cpp
  // cCKF instrumentation (Patch C): optional cluster collection, indexed by
  // measurement index (ClusterContainer is documented one-to-one with
  // measurements). Null when the writer was configured without it.
  const ClusterContainer* clusters =
      m_cfg.inputClusters.empty() ? nullptr : &m_inputClusters(ctx);
```

- [ ] **Step 6: Fill in the outer state loop**

Immediately after the Patch B block from Task 3 Step 11:

```cpp
      // cCKF instrumentation (Patch C): incidence angles and cluster features.
      // Pushed once per state, like Patches A and B.
      {
        float alphaU = nan;
        float alphaV = nan;

        if (state.hasPredicted()) {
          const Acts::FreeVector freePredicted =
              Acts::transformBoundToFreeParameters(surface, gctx,
                                                   state.predicted());
          const Acts::Vector3 direction =
              freePredicted.segment<3>(Acts::eFreeDir0);
          const Acts::Vector3 position =
              freePredicted.segment<3>(Acts::eFreePos0);
          // Columns of the reference frame are (u_hat, v_hat, n_hat).
          const auto angles = incidenceAngles(
              surface.referenceFrame(gctx, position, direction), direction);
          alphaU = Acts::clampValue<float>(angles.alphaU);
          alphaV = Acts::clampValue<float>(angles.alphaV);
        }

        m_alpha_u.push_back(alphaU);
        m_alpha_v.push_back(alphaV);

        int sizeU = -1;
        int sizeV = -1;
        float qTot = nan;
        float sigmaUU = nan;
        float sigmaUV = nan;
        float sigmaVV = nan;

        if (clusters != nullptr && state.hasUncalibratedSourceLink()) {
          const auto* isl = state.getUncalibratedSourceLink()
                                .template getPtr<IndexSourceLink>();
          // Join on the measurement index: ClusterContainer is one-to-one
          // with measurements. Guard the index -- a track container built
          // from a different measurement collection would index out of range.
          if (isl != nullptr && isl->index() < clusters->size()) {
            const Cluster& cluster = (*clusters)[isl->index()];
            const ClusterMoments moments = clusterChargeMoments(cluster);

            sizeU = static_cast<int>(cluster.sizeLoc0);
            sizeV = static_cast<int>(cluster.sizeLoc1);
            qTot = Acts::clampValue<float>(cluster.sumActivations());
            sigmaUU = Acts::clampValue<float>(moments.sigmaUU);
            sigmaUV = Acts::clampValue<float>(moments.sigmaUV);
            sigmaVV = Acts::clampValue<float>(moments.sigmaVV);
          }
        }

        m_clus_size_u.push_back(sizeU);
        m_clus_size_v.push_back(sizeV);
        m_clus_qtot.push_back(qTot);
        m_clus_sigma_uu.push_back(sigmaUU);
        m_clus_sigma_uv.push_back(sigmaUV);
        m_clus_sigma_vv.push_back(sigmaVV);
      }
```

- [ ] **Step 7: Clear them**

After the Patch B clear:

```cpp
    // cCKF instrumentation (Patch C)
    m_clus_size_u.clear();
    m_clus_size_v.clear();
    m_clus_qtot.clear();
    m_clus_sigma_uu.clear();
    m_clus_sigma_uv.clear();
    m_clus_sigma_vv.clear();
    m_alpha_u.clear();
    m_alpha_v.clear();
```

- [ ] **Step 8: Expose `inputClusters` to Python**

In `Python/Examples/src/plugins/Root.cpp`, find the `ACTS_PYTHON_DECLARE_WRITER` for `RootTrackStatesWriter` and add `inputClusters` to the field list:

```cpp
    ACTS_PYTHON_DECLARE_WRITER(
        RootTrackStatesWriter, root, "RootTrackStatesWriter", inputTracks,
        inputParticles, inputTrackParticleMatching, inputSimHits,
        inputMeasurementSimHitsMap, inputClusters, filePath, treeName,
        fileMode);
```

- [ ] **Step 9: Pass it from the reconstruction helper**

In `Python/Examples/python/reconstruction.py`, find the `RootTrackStatesWriter(` construction and add:

```python
                inputClusters="clusters",
```

immediately after the `inputMeasurementSimHitsMap="measurement_simhits_map",` line.

`"clusters"` is `DigitizationAlgorithm`'s default `outputClusters` key. If the cCKF pipeline renames it, use that name instead — Task 1 Step 4 recorded which digi config is in use; check `digi_and_reco.py` for an `outputClusters=` override.

- [ ] **Step 10: Build and run one event**

```bash
cd "$ACTS_BUILD" && cmake --build . -j"$(nproc 2>/dev/null || sysctl -n hw.ncpu)"
cd ~/SURP/cCKF
python digi_and_reco.py --events 1 --output-dir "$PWD/output/instr_patchC" \
    --write-track-states
python scripts/instrumentation/check_trackstate_branches.py \
    output/instr_patchC/trackstates_ckf.root
```

Expected: `PASS  all checks for patches ['A', 'B', 'C'] in ...`.

If `alpha_u` fails the `finite_where="all"` check, some states lack predicted parameters — that is legitimate for the very first state on some seeds. Change that spec to `"any"` and note it in the Task 6 summary rather than forcing it.

- [ ] **Step 11: Verify cluster values are physical, not sentinels**

```bash
cd ~/SURP/cCKF
python3 - <<'PY'
import awkward as ak, numpy as np, uproot
t = uproot.open("output/instr_patchC/trackstates_ckf.root:trackstates")
names = ["clus_size_u", "clus_size_v", "clus_qtot",
         "clus_sigma_uu", "clus_sigma_uv", "clus_sigma_vv",
         "alpha_u", "alpha_v"]
a = t.arrays(names)

su = np.asarray(ak.flatten(a["clus_size_u"]))
sv = np.asarray(ak.flatten(a["clus_size_v"]))
real = su > 0
print(f"states={su.size}  with cluster={int(real.sum())} "
      f"({real.mean():.1%})")
assert real.sum() > 0, "no state joined to a cluster -- check inputClusters wiring"
print(f"size_u: min={su[real].min()} median={np.median(su[real]):.1f} max={su[real].max()}")
print(f"size_v: min={sv[real].min()} median={np.median(sv[real]):.1f} max={sv[real].max()}")
assert su[real].min() >= 1 and sv[real].min() >= 1, "cluster size below 1 channel"

q = np.asarray(ak.flatten(a["clus_qtot"]))[real]
print(f"qtot: min={q.min():.4g} median={np.median(q):.4g} max={q.max():.4g}")
assert (q > 0).all(), "non-positive total charge"

uu = np.asarray(ak.flatten(a["clus_sigma_uu"]))[real]
vv = np.asarray(ak.flatten(a["clus_sigma_vv"]))[real]
uv = np.asarray(ak.flatten(a["clus_sigma_uv"]))[real]
assert (uu >= 0).all() and (vv >= 0).all(), "negative diagonal second moment"
# Cauchy-Schwarz on the charge-weighted moment matrix.
bad = int((uv**2 > uu * vv + 1e-12).sum())
print(f"sigma_uu median={np.median(uu):.4g}  sigma_vv median={np.median(vv):.4g}")
print(f"Cauchy-Schwarz violations: {bad}")
assert bad == 0, "sigma_uv^2 > sigma_uu*sigma_vv -- moment computation is wrong"
# Multi-channel clusters must have non-zero spread.
multi = real & ((su > 1) | (sv > 1))
if multi.sum():
    m_uu = np.asarray(ak.flatten(a["clus_sigma_uu"]))[multi]
    m_vv = np.asarray(ak.flatten(a["clus_sigma_vv"]))[multi]
    frac = float(((m_uu == 0) & (m_vv == 0)).mean())
    print(f"multi-channel clusters={int(multi.sum())} all-zero-moment frac={frac:.2%}")
    assert frac < 0.05, "multi-channel clusters with zero spread -- path2D is empty"

au = np.asarray(ak.flatten(a["alpha_u"]))
au = au[np.isfinite(au)]
print(f"alpha_u: min={au.min():.3f} median={np.median(au):.3f} max={au.max():.3f} rad")
assert np.all(np.abs(au) <= np.pi + 1e-6), "alpha outside (-pi, pi]"
assert au.std() > 1e-3, "alpha_u has no spread -- reference frame may be wrong"
print("OK")
PY
```

Expected: `OK`. The `path2D is empty` assertion is the important one — it catches the case where `Cluster::Cell::path2D` is not populated in the fork's digitization path, which would make every moment exactly zero while still passing the finiteness check.

**If that assertion fires**, `path2D` is unavailable and the moments must instead be computed from bin indices via the module `BinUtility`. That is a materially different implementation — stop, report it, and ask Matthew before proceeding, because it changes the units of `sigma_*` from mm² to channels².

- [ ] **Step 12: Format and commit**

```bash
cd "$ACTS_FORK"
clang-format -i \
  Examples/Io/Root/include/ActsExamples/Io/Root/RootTrackStatesWriter.hpp \
  Examples/Io/Root/src/RootTrackStatesWriter.cpp \
  Python/Examples/src/plugins/Root.cpp
git add Examples/Io/Root/include/ActsExamples/Io/Root/RootTrackStatesWriter.hpp \
        Examples/Io/Root/src/RootTrackStatesWriter.cpp \
        Python/Examples/src/plugins/Root.cpp \
        Python/Examples/python/reconstruction.py
git commit -m "feat(cckf): write cluster shape/charge and track incidence angles

Join the DigitizationAlgorithm cluster collection to track states by
measurement index -- ClusterContainer is one-to-one with measurements and the
writer already extracts the IndexSourceLink index -- so no offline CSV join is
needed. Cluster sizes and total charge come straight off Cluster; the three
charge second moments are computed from the per-channel deposits.

Incidence angles come from Surface::referenceFrame and the predicted
direction, both already available at write time.

inputClusters is optional; leaving it empty keeps the branches present but
filled with sentinels, so non-cCKF users and smearing-only digitization are
unaffected."
```

```bash
cd ~/SURP/cCKF
git add scripts/instrumentation/check_trackstate_branches.py
git commit -m "test: check cluster feature and incidence angle branches"
```

---

## Task 6: Integration verification and summary

Confirm all three patches together on a single event, prove the patches are a numerical no-op for track finding, and write the closing summary.

**Files:**
- Create: `docs/instrumentation/trackstate_branch_reference.md` (cCKF repo)
- Modify: `scripts/instrumentation/check_trackstate_branches.py` (add the no-op comparison mode)

**Interfaces:**
- Consumes: everything from Tasks 2–5.
- Produces: the final deliverable documents. Nothing downstream.

- [ ] **Step 1: Confirm the `--write-track-states` flag is properly plumbed**

`digi_and_reco.py` has `writeTrackStates=False` at four call sites (upstream cCKF lines ~584, 653, 670, 694). Confirm the argparse flag added in Task 2 Step 2 reaches the CKF one and not, say, only a fitter path:

```bash
cd ~/SURP/cCKF
grep -n "writeTrackStates" digi_and_reco.py
```

Every occurrence on the CKF reconstruction path should now read the flag. Leave the fitter paths at `False` — we do not need their track states, and the `pathInX0_interval` column does not exist there (it will correctly read back as NaN).

- [ ] **Step 2: Run one event with all three patches**

```bash
cd ~/SURP/cCKF
python digi_and_reco.py --events 1 --output-dir "$PWD/output/instr_all" \
    --write-track-states 2>&1 | tee output/instr_all/run.log
python scripts/instrumentation/check_trackstate_branches.py \
    output/instr_all/trackstates_ckf.root
```

Expected: `PASS  all checks for patches ['A', 'B', 'C']`.

- [ ] **Step 3: Add the numerical no-op check to the script**

Append to `scripts/instrumentation/check_trackstate_branches.py`, before `main()`:

```python
def compare_baseline(
    patched: str,
    baseline: str,
    tree: str = "trackstates",
) -> list[str]:
    """Assert the patches did not change track finding.

    Compares track count, per-track state count, and the chi2 distribution
    between a patched and an unpatched run of the same event. The
    instrumentation is purely additive, so all three must match exactly.

    Parameters
    ----------
    patched
        Path to the trackstates file from the patched build.
    baseline
        Path to the trackstates file from the unpatched build.
    tree
        TTree name.

    Returns
    -------
    list of str
        Failure messages; empty means the patches are a numerical no-op.
    """
    failures: list[str] = []

    with uproot.open(f"{patched}:{tree}") as tp, uproot.open(
        f"{baseline}:{tree}"
    ) as tb:
        cols = ["volume_id", "chi2"]
        ap = tp.arrays(cols)
        ab = tb.arrays(cols)

        if len(ap["chi2"]) != len(ab["chi2"]):
            failures.append(
                f"track count changed: {len(ab['chi2'])} -> {len(ap['chi2'])}"
            )
            return failures

        np_states = np.asarray(ak.num(ap["volume_id"], axis=1))
        nb_states = np.asarray(ak.num(ab["volume_id"], axis=1))
        if not np.array_equal(np_states, nb_states):
            failures.append(
                f"per-track state counts changed on "
                f"{int((np_states != nb_states).sum())} tracks"
            )

        cp = np.asarray(ak.flatten(ap["chi2"]))
        cb = np.asarray(ak.flatten(ab["chi2"]))
        if cp.shape != cb.shape:
            failures.append(f"chi2 length changed: {cb.shape} -> {cp.shape}")
        elif not np.allclose(cp, cb, rtol=0, atol=0, equal_nan=True):
            n = int((cp != cb).sum())
            failures.append(f"chi2 changed on {n} of {cp.size} states")

    return failures
```

And wire it into `main()` by adding the argument and the call:

```python
    parser.add_argument(
        "--baseline",
        default=None,
        help="unpatched trackstates file; if given, assert the patches "
        "did not change track finding",
    )
```

```python
    failures = check_file(args.path, args.tree, tuple(args.patches))
    if args.baseline:
        failures += compare_baseline(args.path, args.baseline, args.tree)
```

- [ ] **Step 4: Run the no-op comparison**

```bash
cd ~/SURP/cCKF
python scripts/instrumentation/check_trackstate_branches.py \
    output/instr_all/trackstates_ckf.root \
    --baseline output/instr_baseline/trackstates_ckf.root
```

Expected: `PASS`. If χ² changed, Patch B altered the propagation — most likely the `const` removal in Task 3 Step 2 accidentally changed the order of a state update. Re-read that diff; only `const` should have been dropped.

If the baseline file does not exist (Task 2 Step 2 was run after patching), regenerate it from a clean build:

```bash
cd "$ACTS_FORK" && git stash && cd "$ACTS_BUILD" && cmake --build . -j8
cd ~/SURP/cCKF && python digi_and_reco.py --events 1 \
    --output-dir "$PWD/output/instr_baseline" --write-track-states
cd "$ACTS_FORK" && git stash pop && cd "$ACTS_BUILD" && cmake --build . -j8
```

- [ ] **Step 5: Confirm the ACTS test suite still passes**

```bash
cd "$ACTS_BUILD"
ctest -R "Propagator|TrackFinding|ClusterFeatures|Examples" --output-on-failure
```

Expected: all pass. `PointwiseMaterialEffects` gained a defaulted field, which is ABI-additive and should break nothing; if a test constructs it with aggregate initialisation listing exactly four members, that still compiles because the fifth is defaulted.

- [ ] **Step 6: Write the summary document**

Create `docs/instrumentation/trackstate_branch_reference.md` in the cCKF repo:

```markdown
# cCKF TrackState Branch Reference

**ACTS fork revision:** [HEAD hash after all three commits]
**Verified on:** 1 event, ODD ttbar mu=200, geometric digitization
**Verifier:** `scripts/instrumentation/check_trackstate_branches.py`

## Branches added

All branches below are **state-aligned**: `len(branch[i]) == nStates[i]` for
every track `i`, so they are index-parallel with `volume_id`, `layer_id`,
`module_id`, `stateType`, `chi2` and `pathLength`.

| Branch | Type | Patch | Meaning | Sentinel |
|---|---|---|---|---|
| `S00_prt` | float | A | Innovation covariance S(0,0) at the predicted state | NaN |
| `S01_prt` | float | A | S(0,1) | NaN; also NaN for 1D measurements |
| `S11_prt` | float | A | S(1,1) | NaN; also NaN for 1D measurements |
| `pathInX0_interval` | float | B | Material X/X0 since the previous measurement surface | NaN if the CKF column is absent |
| `clus_size_u` | int | C | Cluster size along local u, channels | -1 |
| `clus_size_v` | int | C | Cluster size along local v, channels | -1 |
| `clus_qtot` | float | C | Total cluster charge | NaN |
| `clus_sigma_uu` | float | C | Charge-weighted second central moment in u, mm^2 | NaN |
| `clus_sigma_uv` | float | C | Charge-weighted mixed second moment, mm^2, signed | NaN |
| `clus_sigma_vv` | float | C | Charge-weighted second central moment in v, mm^2 | NaN |
| `alpha_u` | float | C | atan2(d.u_hat, d.n_hat), radians | NaN |
| `alpha_v` | float | C | atan2(d.v_hat, d.n_hat), radians | NaN |

## Fields that must be joined offline

**None.** Every quantity in the Phase 1 request is written directly.

The cluster join that the task statement anticipated might be needed offline
happens in-memory in the writer instead: `ClusterContainer` is one-to-one with
measurements, and the writer already has the measurement index from the
`IndexSourceLink`, so `clusters[sl.index()]` resolves at write time.

## Known caveats

1. **Incidence angles are stored as angles, not ratios.** The spec asked for
   `d.u_hat / d.n_hat`. That diverges at grazing incidence. We store
   `atan2(d.u_hat, d.n_hat)`; recover the ratio with `np.tan(alpha_u)`.

2. **`sigma_*` are population moments in mm^2.** A single-channel cluster
   gives exactly 0, not NaN. Downstream code should not treat 0 as missing --
   use `clus_size_u < 0` to test for "no cluster".

3. **The legacy `*_hit` branches are NOT state-aligned.** `dim_hit`,
   `res_x_hit`, `err_x_hit`, `pull_x_hit` and their `_y` counterparts are
   shorter than `nStates` because the writer's parameter loop `continue`s past
   states without a source link. Do not zip them against the branches above.
   Use `S00_prt` / `S11_prt` for the innovation covariance instead of
   `err_*_hit`.

4. **`pathInX0_interval` is only populated by the CKF.** Fitter output (KF,
   GSF, GX2F) does not register the dynamic column, so it reads back NaN there.

5. **Patch C requires geometric digitization.** Smearing-only digi produces no
   clusters; the `clus_*` branches would be all-sentinel.

## Numerical no-op

Verified: track count, per-track state count, and every per-state chi2 are
bit-identical between the patched and unpatched builds on the same event.
Reproduce with:

    python scripts/instrumentation/check_trackstate_branches.py \
        output/instr_all/trackstates_ckf.root \
        --baseline output/instr_baseline/trackstates_ckf.root

## Commits

- Patch A: [hash] `feat(cckf): write full innovation covariance ...`
- Patch B: [hash] `feat(cckf): record accumulated material X/X0 ...`
- Patch C1: [hash] `feat(cckf): add cluster charge-moment and incidence-angle helpers`
- Patch C2: [hash] `feat(cckf): write cluster shape/charge and track incidence angles`
```

- [ ] **Step 7: Log the work and commit**

Append to `experiments/LOG.md` (per CLAUDE.md experiment rule 3):

```markdown
## 2026-08-08 — ACTS instrumentation for cCKF gate features

Three additive patches to the cmuchancel ACTS fork so `trackstates.root`
carries the full gate feature vector (spec 8.2):

- **A** — full innovation covariance `S00_prt`/`S01_prt`/`S11_prt`
- **B** — `pathInX0_interval`, material X/X0 between measurement surfaces
- **C** — cluster shape/charge (`clus_*`) and incidence angles (`alpha_u/v`)

Verified numerically inert (track count, state counts and per-state chi2
bit-identical vs. the unpatched build on 1 event). No data collection run.
No offline join needed — clusters join in-memory by measurement index.

See `docs/instrumentation/trackstate_branch_reference.md`.
```

```bash
cd ~/SURP/cCKF
git add docs/instrumentation/trackstate_branch_reference.md \
        scripts/instrumentation/check_trackstate_branches.py \
        experiments/LOG.md
git commit -m "docs: trackstate instrumentation branch reference and no-op verification"
```

- [ ] **Step 8: Final confirmation**

```bash
cd "$ACTS_FORK" && git log --oneline -4
cd ~/SURP/cCKF && git log --oneline -5
cd ~/SURP/cCKF && python scripts/instrumentation/check_trackstate_branches.py \
    output/instr_all/trackstates_ckf.root \
    --baseline output/instr_baseline/trackstates_ckf.root
```

Expected: four ACTS commits (A, B, C1, C2), the cCKF report/script/reference commits, and a final `PASS`.

---

## Deliverables Checklist

| Deliverable | Task | Artifact |
|---|---|---|
| 1. Phase 1 report on cluster feature availability | 1 | `docs/instrumentation/phase1_cluster_feature_availability.md` |
| 2. Three patches as separate commits | 2, 3, 4+5 | 4 ACTS commits (C is split into pure-helper and wiring commits) |
| 3. Single-event test output confirming branches populated | 2, 3, 5, 6 | `output/instr_all/trackstates_ckf.root` + `PASS` from the verifier |
| 4. Summary of fields requiring offline join | 6 | `docs/instrumentation/trackstate_branch_reference.md` — answer: none |
