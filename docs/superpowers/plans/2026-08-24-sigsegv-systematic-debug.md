# cCKF SIGSEGV — systematic debugging plan

**Date:** 2026-08-24
**Status:** Phase 1 (evidence gathering). No fixes until root cause is identified.
**Method:** superpowers:systematic-debugging

---

## Why this plan exists

Seven debug cycles (Aug 20-21) produced three fixes, none of which resolved the
crash:

1. `.eval()` in `CckfMeasurementSelector.hpp` → still SIGSEGV
2. `.eval()` in `CckfBranchStopper.hpp` → still SIGSEGV
3. `.eval()` in `CckfTrackFindingAlgorithm.cpp:173` → still SIGSEGV

The debugging skill is explicit: **three or more failed fixes means the
approach is wrong, not the hypothesis.** Each cycle cost a ~20 minute rebuild
and ended with a new guess rather than new evidence.

The pattern to break: every cycle so far has inferred the corruption's *source*
from where it *surfaces*. That is guesswork. A memory checker reports the
corrupting write directly.

---

## What is actually established

Facts, from `cCKF/CLAUDE.md` cycle 6 and independent code reading:

| claim | status |
|-------|--------|
| Crash inside `CckfBranchStopperWrapper`, at `track.nMeasurements()` | Established. Partial stderr line `DIAG branchStopper: nMeas=` flushed, value never printed. |
| Survives 52 seeds at `threads: 1`, immediate at `threads: 8` | Established. |
| Threading is not the sole cause | Established (crash persists single-threaded). |
| Proxy invalidation is not the cause | Established. ACTS `TrackProxy` is (container, index) addressed; reallocation is safe. |
| Progressive heap corruption | **Hypothesis only.** Consistent with delayed onset, not demonstrated. |
| Value MLP dimension mismatch overflows a buffer | **Contradicted.** `MlpInference` sizes `m_buf_a`/`m_buf_b` from `blob.n_hidden` and `m_input_buf` from `blob.n_features`; every loop is bounded by those same values. No write exceeds a buffer. A wrong-sized blob over-*reads* the caller's `float features[11]` on the stack; it does not corrupt the heap. |

Delayed onset is equally consistent with a **stack** over-read/write, an
uninitialised value, or a rare-branch bug that only executes on some seeds. The
evidence does not yet distinguish these.

---

## Phase 1 — evidence, cheapest first

### Step 0 — component isolation (config only, no rebuild)

Three runs, one variable each, on the same event and seed set.

| run | `cckf_gate_weights` | `cckf_value_weights` | if it crashes | if it survives |
|-----|--------------------|---------------------|---------------|----------------|
| A | real | `""` (off) | not the value MLP | value MLP implicated |
| B | `""` (off) | real | not the gate | gate implicated |
| C | `""` | `""` | build/ACTS itself is broken | our code is implicated |

Run C is the control and must pass. If it does not, nothing downstream means
anything.

Cost: ~3 short runs, no rebuild. Narrows to a component before spending a build.

### Step 1 — make the allocator fail loudly (env only, no rebuild)

Re-run the crashing configuration with:

```bash
export MALLOC_CHECK_=3      # abort on detected heap inconsistency
export MALLOC_PERTURB_=42   # poison freed memory; surfaces use-after-free
ulimit -c unlimited         # core dumps are currently disabled (ulimit -c = 0)
```

`MALLOC_CHECK_=3` makes glibc abort at the *first* detected inconsistency
rather than whenever the damage happens to be fatal. If the crash moves earlier
than seed 52, heap corruption is confirmed and we have a much tighter window.
If it does not move at all, heap corruption becomes less likely and the stack
hypotheses rise.

Cost: one run. No rebuild.

### Step 2 — AddressSanitizer build (the decisive step)

`valgrind` and `gdb` are **absent** from the container. `libasan.so` is
**present** (gcc 13.3.0), so ASan is the available instrument.

Build into a separate tree so the normal build is untouched:

```bash
CCKF_ACTS_ROOT=$SCRATCH/cckf/acts-asan \
CMAKE_EXTRA="-DCMAKE_CXX_FLAGS=-fsanitize=address -fno-omit-frame-pointer -g -O1" \
  ./scripts/build_cckf_nersc.sh --bootstrap --install
```

Then run 2 events with:

```bash
export ASAN_OPTIONS=detect_leaks=0:abort_on_error=1:print_stacktrace=1:halt_on_error=1
```

ASan intercepts every allocation, puts redzones around it, and instruments
loads and stores. A heap-buffer-overflow, use-after-free, or stack-buffer-
overflow is reported **at the instruction that does it**, with a stack trace,
plus the allocation site. That is the information seven cycles have been trying
to infer.

Notes:
- `detect_leaks=0` because ACTS leaks by design at exit and the noise is useless.
- ASan runs 2-3x slower and uses ~3x memory. Fine for 2 events.
- Partial instrumentation is acceptable: if our code performs the bad write,
  instrumenting our translation units catches it even where ACTS is not
  instrumented. Instrumenting everything is stronger and is what the command
  above does.

Cost: one ~20 minute bootstrap plus a short run. This is the step that should
end the investigation.

### Step 3 — only if ASan is clean

If ASan reports nothing, memory corruption is disproven and the hypothesis was
wrong for seven cycles. Pivot to:

- Uninitialised reads: rebuild with `-fsanitize=memory` (needs clang, likely
  unavailable) or audit `BranchContext` / `SensorProps` / `Innovation` for
  fields not set on every path.
- Rare-branch bug: log `seed_index`, `step_k`, `n_candidates`, `calibratedSize`
  for every branch-stopper call. Seed 52 differs from seeds 0-51 in some way;
  find what.
- ACTS-side contract violation: compare our delegate wiring line by line
  against `TrackFindingAlgorithm.cpp`, which is known to work with the same
  extension points.

---

## Phase 2 — pattern analysis (do alongside Phase 1)

The working reference is `TrackFindingAlgorithm.cpp`. It uses the same
`measurementSelector` and `branchStopper` delegates and does not crash.

Enumerate every difference between it and `CckfTrackFindingAlgorithm.cpp`:
delegate lifetime, what is captured by reference vs value, whether any
captured object outlives its owner, dynamic column registration, and the order
of container construction relative to delegate connection.

A delegate capturing a pointer to something destroyed early would present
exactly as "works for a while, then crashes."

---

## Phase 3 — hypothesis and test

Only after Phase 1 yields evidence. One hypothesis, stated explicitly, one
minimal change, verify before moving on. No bundled fixes.

---

## Wargame — what else is likely to bite

Ordered by probability x cost.

### Build

| risk | detection | mitigation |
|------|-----------|------------|
| `ActsPythonBindings` target name is a guess | `make` errors "no rule to make target" | `make help \| grep -i python` in the build tree; the script takes `--targets` |
| pybind block anchors drift from the pinned ACTS commit | `apply_cckf_integration.sh` raises "anchor not found" | Anchors are exact-match and fail loudly by design. Good. |
| ccache cold on first ASan build | slow first build only | Expected; second is fast |

### Runtime correctness

| risk | detection | mitigation |
|------|-----------|------------|
| Wrong digi config silently selects smearing digitisation → empty clusters → zero cluster features | `gate_diag.n_cluster_ok` near zero | This already happened once (fixed in fce31a3). Assert `n_cluster_ok > 0` early in every run. |
| ODD v5 volume IDs vs `SensorLookup` keys | `pitch_u` non-finite; the 32-event sweep saw 0.9% already | Log the fraction of candidates with a sensor-lookup miss per run |
| `pathInX0_interval` column not registered → material silently zero | feature 19 all zeros in the trace CSV | Also already happened once (fce31a3). Add a startup assertion that the column exists. |
| Box change alters hole accounting | track counts differ from previous runs | **Expected, not a regression.** The three new hole counters exist to explain it. Compare `hole_gate_failure` against `hole_window_failure`. |

### Performance and budget

| risk | detection | mitigation |
|------|-----------|------------|
| cCKF does not thread well; Pareto sweep unaffordable | time one event on one full node | **Measure before designing the sweep grid.** 194 node-hours total. |
| An idle `salloc` burns allocation | — | Short interactive windows, or batch jobs |

### Data and infrastructure

| risk | detection | mitigation |
|------|-----------|------------|
| Modal volume deleted before mirror verified | irreversible | Verify the mirror the same way the Parquets were verified (per-file counts) before any deletion |
| SCRATCH purge removes the mirror | weeks away | Move anything to keep onto CFS |
| Training on Group A events (0-3) skews calibration | documented in LOG 2026-08-24 | Drop events 0-3 from the train split before retraining |

---

## Decision rule

If Phase 1 Step 2 (ASan) does not identify the root cause, **stop and discuss
architecture** rather than attempting fix number four. Per the debugging skill,
repeated failure across different locations indicates the wiring pattern is
wrong, not that the latest guess was.
