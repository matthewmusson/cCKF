# CLAUDE.md — cCKF Subrepo

This is the **cCKF** (calibrated CKF) implementation repo, a git submodule under `/Users/matthewm/SURP/`. The parent repo's `CLAUDE.md` has the full project context, research plan, and physics background. This file covers the repo structure and build system.

## Repo Structure

```
cCKF/
├── acts_patches/                    # C++ files copied into ACTS source tree at build time
│   ├── cckf/                        # Header-only cCKF library (7 headers)
│   │   ├── MlpInference.hpp         # Hand-written MLP forward pass (no ONNX/LibTorch)
│   │   ├── WeightBlob.hpp           # Binary blob loader (magic "CCKF", weights + stats + Platt)
│   │   ├── CckfMeasurementSelector.hpp  # Gate g_ψ — replaces χ² cut via measurementSelector delegate
│   │   ├── CckfBranchStopper.hpp    # Value V_φ — replaces hole/branch caps via branchStopper delegate
│   │   ├── CckfFeatures.hpp         # Feature vector construction (26-dim gate, 11-dim value)
│   │   ├── CckfTimers.hpp           # Per-event timing counters
│   │   └── SensorLookup.hpp         # JSON digi-config → sensor pitch/thickness lookup
│   └── ActsExamples/TrackFinding/
│       ├── CckfTrackFindingAlgorithm.hpp  # ACTS Algorithm class (mirrors TrackFindingAlgorithm)
│       └── CckfTrackFindingAlgorithm.cpp  # Wires gate + value + timing into CKF
│
├── cckf/                            # Python package
│   ├── models.py                    # PyTorch model definitions (GateMLP, ValueMLP)
│   └── features.py                  # Feature construction for training (build_gate_features)
│
├── scripts/
│   ├── build_patched_acts.sh        # Build script: clone ACTS → patch → copy cCKF → cmake → make
│   ├── export_weights.py            # PyTorch → CCKF binary blob converter
│   ├── patch_is_selected.py         # Expand Parquet with is_selected column
│   ├── pareto_sweep.py              # Threshold grid sweep (τ_g × τ_v)
│   └── smoke_test_cckf.sh           # End-to-end: build → dummy weights → run 2 events
│
├── modal_build_acts.py              # Modal app: build_acts, generate_dummy_weights
├── modal_acts.py                    # Modal app: run_cckf, run_baseline_ckf
├── modal_train.py                   # Modal app: gate/value training
├── digi_and_reco.py                 # Python ACTS pipeline (addCKFTracks / addCckfTracks)
├── expansion.py                     # Parquet expansion (truth-matching, track states)
├── instrumentation.patch            # Git patch adding S_k, X/X₀, cluster features to ACTS track states
│
├── tests/                           # Unit tests
└── output/                          # Training outputs (weights, metrics)
```

## Build System

The cCKF integrates into ACTS via source-level patching at build time. **No fork of ACTS is maintained** — we patch a pinned commit.

### Build Pipeline (`scripts/build_patched_acts.sh`)

1. Clone ACTS at pinned commit `4de1dcbbb2b8d8b6f14ec2c974d9b3a622028c01`
2. `git apply instrumentation.patch` — adds S_k, X/X₀, cluster features to track states
3. Copy `acts_patches/cckf/*.hpp` → `Examples/Framework/include/ActsExamples/cckf/`
4. Copy `acts_patches/ActsExamples/TrackFinding/*` → `Examples/Algorithms/TrackFinding/{include,src}/`
5. Patch `Examples/Algorithms/TrackFinding/CMakeLists.txt`:
   - Add `CckfTrackFindingAlgorithm.cpp` to source list
   - Add PUBLIC include directories for bare `cckf/` header resolution
   - Link `nlohmann_json::nlohmann_json` (already in ACTS top-level CMake, just needs linking)
6. Patch `Python/Examples/src/TrackFinding.cpp`:
   - Add pybind11 bindings for `CckfTrackFindingAlgorithm` + all config fields
7. `cmake` with full ACTS options → `make -j$(nproc)` → `make install`

### Key Build Decisions

- **Include dirs are PUBLIC** (not PRIVATE) because `CckfTrackFindingAlgorithm.hpp` transitively includes `cckf/SensorLookup.hpp`, and the Python bindings target needs to resolve this.
- **nlohmann_json is PUBLIC** for the same reason (SensorLookup.hpp includes `<nlohmann/json.hpp>`).
- **No new dependencies** — nlohmann_json is already in ACTS's dependency tree.
- Build runs on Modal via `modal run modal_build_acts.py::build_acts --force`.

## ACTS Extension Points Used

The CKF has two delegate-based extension points we hook into:

1. **`TrackStateCreator.measurementSelector`** — Connected to `CckfMeasurementSelectorAdapter` which wraps `CckfMeasurementSelector` (the gate MLP). Replaces the chi-squared cut.

2. **`CombinatorialKalmanFilterExtensions.branchStopper`** — Connected to `CckfBranchStopperWrapper` which wraps `CckfBranchStopper` (the value function MLP). Replaces hole/branch caps.

Both delegates are wired in `CckfTrackFindingAlgorithm::execute()`. The Kalman filter engine, propagator, and track containers are identical to the standard `TrackFindingAlgorithm`.

## Weight Blob Format

Binary format (`.cckf` extension):
```
Magic "CCKF" (4 bytes)
Version uint32 (currently 1)
input_dim uint32, output_dim uint32
n_hidden uint32
hidden_dims[n_hidden] uint32[]
mean[input_dim] float32[]     # per-feature standardization
std[input_dim] float32[]
platt_params[4] float32[]     # a₀, a₁, b₀, b₁ (gate) or zeros (value)
For each layer:
  weight[out × in] float32[]  # row-major
  bias[out] float32[]
```

## Common Build Errors and Fixes

These are the errors encountered during integration (Aug 19, 2026) and their fixes. Future agents modifying the C++ code should watch for these patterns:

| Error | Root Cause | Fix | Commit |
|-------|-----------|-----|--------|
| `GainMatrixUpdater` not found | Missing include | Add `#include "Acts/TrackFitting/GainMatrixUpdater.hpp"` | 726a7a5 |
| `no match for call to (unique_ptr<Logger>)()` | Constructor param `logger` shadows `IAlgorithm::logger()` | Rename param to `lgr` | 726a7a5 |
| `GeometryContext{}` private constructor | Default constructor is private in pinned ACTS | Thread real `GeometryContext` from `AlgorithmContext` through adapter | 726a7a5 |
| `invalid operands to binary operator<` on `.segment<3>()` | Template method context needs `template` keyword | Use `.template segment<3>()` | e5b3f8b |
| `sizeof incomplete type SensorLookup` | Forward declaration insufficient for pybind11 destructor | Include full `SensorLookup.hpp` in header | 7fc233a |
| Python bindings can't find `cckf/SensorLookup.hpp` | Include dirs were PRIVATE | Change to PUBLIC in CMakeLists patch | c39d150 |
| SIGSEGV on first CKF event (3 occurrences) | Eigen expression template UB: `const auto x = proxy.predicted()` stores an expression template referencing destroyed temporaries from TrackStateProxy accessors, not a concrete matrix. Stack layout changes surfaced latent dangling pointers. | Add `.eval()` to ALL Eigen proxy accessor results across 3 files — see below | uncommitted |
| Distribution shift: gate sees chi2 up to 250K | TrackStateCreator passes ALL hits on a surface to measurementSelector, no spatial pre-filtering | Added nSigma pre-filter (chi2 < nSigma²) before gate scoring | uncommitted |

### Eigen Expression Template UB — Full Details (Aug 20-21, 2026)

ACTS TrackStateProxy accessor methods (`.predicted()`, `.parameters()`, `.covariance()`, `.effectiveCalibrated()`, etc.) return Eigen `Map` or `Block` types — lightweight views into the track state's backing storage. When captured with `const auto`, C++ deduces the expression template type rather than a concrete `Eigen::VectorXd`. If the proxy or its temporaries are destroyed before the variable is used, the stored expression template holds dangling pointers.

This UB was always latent in the cCKF code but only manifested as SIGSEGV after the nSigma pre-filter was added (changing the stack layout of `select()` enough to surface the corruption).

**Files fixed (3 separate rebuild cycles to find all sites):**

1. `CckfMeasurementSelector.hpp` — `computeChi2()` and `buildFeatures()`:
   - `ts.predicted()`, `ts.effectiveCalibratedCovariance()`, `ts.effectiveCalibrated()`, `ts.projectorSubspaceHelper().fullProjector().topLeftCorner(...)`, and compound expressions involving `ts.predictedCovariance()`

2. `CckfBranchStopper.hpp` — `buildFeatures()`:
   - `trackState.parameters()`, `trackState.covariance()`

3. `CckfTrackFindingAlgorithm.cpp` — `CckfMeasurementSelectorAdapter::select()`:
   - `candidates[0].predicted()` (line 173) — **this was the actual crash site**, the first Eigen proxy accessor called when CKF processes its first surface

**Rule:** In any ACTS code using TrackStateProxy, ALWAYS call `.eval()` when storing the result of a proxy accessor in a local variable. Inline use (e.g., `H * ts.predictedCovariance() * H.transpose()` consumed in the same expression) is safe.

### nSigma Spatial Pre-Filter (Aug 20, 2026)

During training, gate candidates were collected within a chi2 window (max ~200). At inference, TrackStateCreator passes ALL hits on a surface — chi2 values up to 250,000. The nSigma pre-filter rejects candidates with chi2 > nSigma² before gate scoring, matching the training distribution.

Config: `cckf_gate_window_nsigma` (default 0.0 = disabled). Set to 10.0 in `cckf_tight.yaml` and `cckf_envelope.yaml` (chi2 < 100).

Files touched: `CckfMeasurementSelector.hpp` (pre-filter logic + `n_window_prefiltered` diagnostic), `CckfTimers.hpp`, `CckfTrackFindingAlgorithm.{hpp,cpp}`, `digi_and_reco.py`, `build_patched_acts.sh` (pybind).

## Running

```bash
# Build patched ACTS on Modal
modal run modal_build_acts.py::build_acts --force

# Generate dummy weights for smoke testing
modal run modal_build_acts.py::generate_dummy_weights

# Run cCKF on 2 test events
modal run modal_acts.py::run_cckf --events 2

# Run baseline CKF for comparison
modal run modal_acts.py::run_baseline_ckf --events 2
```

## Current Status (Aug 21, 2026)

- All 10 SDD implementation tasks complete and committed
- Build fixes committed (6 commits post-SDD) + Eigen UB fixes (uncommitted, 3 files)
- Trained gate g_ψ and value V_φ weights exported to binary blobs on Modal
- **nSigma=10 spatial pre-filter added** to match training distribution (chi2 < 100)
- **Eigen expression template UB fixed** across 3 files (3 rebuild cycles to find all sites)

### SIGSEGV Debugging — 7 Cycles (Aug 20–21)

**Cycle 6 result (threads: 1 + stderr diagnostics):**
With `threads: 1` and `std::cerr` diagnostics at gate-select entry, branch-stopper entry, and before/after findTracks, the CKF **ran successfully through 52 seeds** before crashing. Previous runs with `threads: 8` crashed immediately (0 seeds). Diagnostic output at crash:
```
DIAG gate-select: 30 cands
DIAG branchStopper: nMeas=3
Runner segmentation fault (SIGSEGV), exit code: 139.
DIAG branchStopper: nMeas=
```

**Key findings from Cycle 6:**
1. **Not a threading issue (alone)** — crash still occurs with `threads: 1`, just takes longer to manifest (~52 seeds vs immediate). Threading exacerbates but isn't the root cause.
2. **Crash is in the branch stopper** — the partial line `DIAG branchStopper: nMeas=` shows the crash happens during `track.nMeasurements()` call in the wrapper's diagnostic line. The string `"nMeas="` was output but the function call to get the value crashed.
3. **Not proxy invalidation** — ACTS `TrackProxy` uses `(container_pointer, integer_index)` addressing, not raw pointers. `VectorTrackContainer::addTrack_impl()` can reallocate internal vectors but proxies recompute addresses from index at every access. The standard `TrackFindingAlgorithm` uses the identical delegate wiring pattern.
4. **Heap corruption** — the delayed onset (works for 51 seeds, crashes on 52nd) is the classic signature of progressive heap corruption. Something earlier in execution corrupts memory, and by seed ~52 it has reached a critical allocation. Possible sources: value MLP dimension mismatch causing buffer overflow, or a remaining `.eval()` site causing use-after-free.

**Cycle 7 diagnostics (added, awaiting rebuild):**
- Value blob dimension logging (`n_features`, `n_hidden`, `n_layers`) — verify it matches expected 11 features
- `std::cerr` before/after `m_valueInference->forward()` in `CckfBranchStopper::operator()` — isolate whether crash is in MLP or surrounding code
- NaN/inf check on all 11 value features before MLP call
- Full feature dump for first 3 value MLP calls

**Isolation test (config-only, no rebuild):**
- Run with `cckf_value_weights: ""` to disable value function. If crash disappears → value MLP is the corruption source. If crash persists → gate MLP or other code.

### Build/Run History

| Cycle | Build App | Inference App | Change | Result |
|-------|-----------|---------------|--------|--------|
| 1 | (prev session) | (prev session) | `.eval()` in CckfMeasurementSelector.hpp | SIGSEGV (immediate) |
| 2 | ap-yxWK1atldEddnwk0C5HI40 | ap-ZFtGWXYoBTeUeTDPqRdcpb | `.eval()` in CckfBranchStopper.hpp | SIGSEGV (immediate) |
| 3 | ap-FmO09RnP5LOYYg8i96ZVZD | ap-LOZjfQZ9qRviixZXwOc2jx | `.eval()` in CckfTrackFindingAlgorithm.cpp:173 | SIGSEGV (immediate) |
| 4 | ap-UQMdtm62z6JjFIztMT48D9 | ap-32TA0L6vkBgWEkjdU35JVh | Diagnostic ACTS_INFO after value load + before seed loop | SIGSEGV — crash confirmed INSIDE seed loop |
| 5 | — | — | Exhaustive code review of all delegates | No new Eigen/null/bounds issues found |
| 6 | ap-SjYgr9e6d6zjM7HqKF9zjh | (from build_acts::run_cckf) | stderr diagnostics + threads: 1 | SIGSEGV at seed 52 in branchStopper — heap corruption |
| 7 | — | — | Value blob dims + value MLP before/after + NaN check | Awaiting rebuild (out of Modal credits) |

- Once crash is resolved: verify nSigma pre-filter diagnostics, analyze tracking performance
- Next: Pareto sweep across (τ_g, τ_v) threshold grid, calibration plots for poster
