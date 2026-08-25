# Tier-3 value targets — truth-greedy rollout design

**Date:** 2026-08-25
**Status:** Approved in discussion. C++ executor buildable now; Python
walker carries Tier-1 `TODO(human)` blocks.

## What tier 3 adds over tier 2

The value target is V^{π†}(s_k): from branch state s_k, follow the
truth-greedy policy π† (take the branch majority particle's hit at every
remaining surface, hole where it has none) and score the completed branch
min(completeness, purity). Tier 2 substitutes the branch's own logged future
for the rollout, which is only correct where the CKF's choices coincide with
π†. Tier 3 computes the true continuation, which requires the Kalman
propagator: mid-detector starts with possibly-contaminated history exist
nowhere in the logs.

## Architecture: offline, worklist-driven, backward-collapsed

```
expanded parquet + trackstates ROOT
        │
        ▼
[Python] walker: mark every (branch, state) as
         collapse / divergence / tip          ← Tier 1 (π† pick rule)
        │  emits worklist: only divergence + tip states
        ▼
[C++]   TruthRolloutAlgorithm: for each worklist row,
         findTracks(bound params) with TruthMeasurementSelector
         → hit sequence + visited track states  ← Tier 2
        │
        ▼
[Python] backward stitch: counts flow tip → seed; every
         (branch, layer) receives V^{π†}        ← Tier 1 (V composition)
```

### Why offline (not in-run) is acceptable

The ROOT log stores filtered parameters and the covariance **diagonal**
(`err_*_flt`); the 15 off-diagonals are lost. Consequences, split by
product:

- **Hit sequence / target counts: mildly perturbed.** Navigation follows the
  propagated mean (covariance-free); the truth selector picks by particle
  identity on the reached surface, not by any window; the only leak is the
  misweighted Kalman gain drifting the filtered mean, tens of microns
  against centimeter module acceptances, damped as the filter forgets its
  seed within a few updates. Failure mode is one-sided: a spuriously missed
  surface records a hole π† would not have → target biased LOW, never high.
- **Rollout's own track states as gate training data: contaminated near the
  start** (S and χ² features wrong until correlations re-converge). This is
  the optional DAgger-supplement ablation, NOT the primary DAgger source —
  that is the sweep runs' logged states. Log rollout states anyway, tagged
  `steps_since_start`, so the ablation can burn in.

### The free bias measurement (run before trusting targets)

For every state whose branch follows truth to the tip, the logged future IS
the π† rollout, so tier 2 already gives the exact target. Running the
diagonal-seeded rollout on a sample of those states and comparing counts
measures the covariance-seeding error directly, zero extra C++. Acceptance:
disagreement on < 1% of sampled states. If it fails: add the 15
off-diagonals to instrumentation.patch (~1.6 GB/event) and re-run Stage 1
(~9 node-hours). In-run rollout stays off the table.

## Backward collapse (cost: one propagation per divergence + tip)

Reuse is licensed by **state equality**, certified by **action equality**:
same parent state + same hit → bit-identical filtered child (deterministic
KF update). Per branch, walk tip → seed:

- **tip**: CKF stopped, π† continues → always one fresh propagation
  (unless already at the detector edge).
- **collapse**: branch's next selected hit == π†'s pick from here →
  V(s_k) = one step + child's rollout. No propagation.
- **divergence**: branch took a wrong hit / holed where truth has a hit /
  took truth hit A where π† picks truth hit B → fresh propagation.

Holes agree when `majority_true_hit_on_surface == 0` (hole adds no update,
states stay equal). Hit-ID equality, not is-truth equality, is the test:
two truth hits on a surface (module overlaps, shared clusters) void the
certificate — rare, flagged and counted rather than ignored.

Cost anticorrelation: contamination-heavy branches diverge often but die
young (short rollouts); long branches are truth-dominated (few
divergences). Event-1 scale: 4.6M selected states / ~800K seeds → mean
branch length ~6. Executor logs propagation-step counters; the subsample
knob (uniform per-state, the only unbiased variant) exists but is expected
unnecessary.

Both sides of the divergence test are Parquet columns: `is_ckf_selected`
(branch's choice) and the next state's candidate list with truth labels
(π†'s pick — computable from the branch's own log because the predicted
state is shared).

## Components

### C++ (Tier 2) — `acts_patches/`

1. `cckf/TruthMeasurementSelector.hpp` — selector delegate: given surface
   candidates, return the one whose measurement maps to the rollout's
   majority PID (map loaded from measurement-simhit-map CSV, SensorLookup
   pattern); empty → hole. No MLP, no window, no χ².
2. `cckf/TruthRolloutStopper.hpp` — branch stopper that never stops.
3. `ActsExamples/TrackFinding/TruthRolloutAlgorithm.{hpp,cpp}` — mirrors
   CckfTrackFindingAlgorithm. Config: worklist CSV path, simhit-map CSV
   path, output paths. For each worklist row: build BoundTrackParameters on
   the row's surface (params + diagonal covariance), findTracks, write
   (rollout_id, step, geometry_id, cand_hit_id | -1) rows +
   RootTrackStatesWriter states tagged rollout_id/steps_since_start.
   Counters: n_rollouts, n_propagation_steps, n_holes.
4. pybind registration via apply_cckf_integration.sh; incremental build via
   build_cckf_nersc.sh rsync fast path.

### Worklist CSV (Python → C++)

```
rollout_id, event_id, seed_id, branch_id, step_k,
geometry_id, loc0, loc1, phi, theta, qop, t,
var_loc0, var_loc1, var_phi, var_theta, var_qop, var_t,
majority_pid_pv, majority_pid_sv, majority_pid_part,
majority_pid_gen, majority_pid_sub
```

Filtered params + err diagonals come from the trackstates ROOT
(`eLOC*_flt`, `err_*_flt`), joined on (seed_id, state_idx) — the exact
all-state key from the is_ckf_selected work.

### Python (Tier 1 — Matthew)

- `cckf/tier3_walker.py`: per-branch tip→seed classification, worklist
  emission. `TODO(human)`: the π† tie-break when >1 truth candidate (part
  of the π† definition, spec §11.1).
- `cckf/tier3_stitch.py`: merge C++ hit sequences with logged pasts,
  backward count flow, V^{π†} per (branch, layer). `TODO(human)`: the
  min(completeness, purity) composition (spec §11.1) and the truth-suffix
  bias check.

## Validation gates

1. Executor smoke: 100-row worklist on event 1, hit sequences land, states
   written, counters sane.
2. Truth-suffix bias check < 1% disagreement (see above).
3. Collapse audit: for a sample of collapse-marked states, run the rollout
   anyway and confirm identical counts.
4. Tier-3 vs tier-2 target distributions: differ exactly where branches
   diverge from truth, agree elsewhere.

## Out of scope

- In-run rollout (rejected: couples into the CKF job for a product the
  sweep already provides).
- Full-covariance instrumentation (fallback only, gated on validation 2).
- DAgger training itself.
