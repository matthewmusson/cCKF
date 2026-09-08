# Spack-Compatible ACTS Rebuild

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebuild patched ACTS from source with EDM4HEP and Python bindings, ABI-compatible with the spack dependency tree, so the full CKF pipeline runs without the broken LD_PRELOAD hack.

**Architecture:** The current Docker image (`ghcr.io/opendatadetector/sw:0.2.2`) ships a pre-built spack ACTS with all dependencies (ROOT, DD4hep, podio 1.4.1, edm4hep, pybind11, gcc 13.3.0). Our existing cmake build script already builds against these spack dependencies via `CMAKE_PREFIX_PATH` but has two disabled flags: EDM4HEP (failed due to podio C++20 code generation) and Python bindings. The fix is to (1) diagnose the exact cmake configuration spack used, (2) enable C++20 standard to fix podio compatibility, (3) enable EDM4HEP and Python bindings, and (4) use the resulting self-contained ACTS install directly — no spack library mixing, no LD_PRELOAD.

**Tech Stack:** cmake, gcc 13.3.0, Modal, spack dependency tree, podio 1.4.1, edm4hep, pybind11, ROOT

## Global Constraints

- Events [0,32) for train/val/cal. Events [32,64) are SEALED — never touch.
- Base ACTS commit: `4de1dcbbb2b8d8b6f14ec2c974d9b3a622028c01`
- Instrumentation patch: 4 commits, 12 new ROOT branches
- Build target: Modal container from `ghcr.io/opendatadetector/sw:0.2.2`
- Build cached on Modal volume `surp-acts-build` at `/build`
- Tier 3 task — full delegation, infrastructure

---

## Background: Why the Previous Approaches Failed

1. **cmake build without EDM4HEP:** Built successfully, but couldn't read edm4hep.root ColliderML input. Required LD_PRELOAD hack to mix our patched writer library with spack's full ACTS — ABI mismatch on `_ZTIN12ActsExamples20RootSpacepointWriterE` and similar symbols because the two ACTS builds are from different commits.

2. **cmake build with EDM4HEP:** Failed at compile time. Podio 1.4.1's code generator produces C++20 code (concepts, `requires` clauses). Our cmake config used C++17 (ACTS default). The spack ACTS was built with the **same** podio 1.4.1 and succeeded — meaning spack either set C++20 explicitly or ACTS's cmake detects it.

3. **LD_PRELOAD approach:** Surgically replacing `libActsExamplesIoRoot.so` while using spack for everything else. Failed because spack's Python bindings (`.cpython-313-*.so`) reference symbols in spack's ACTS libraries, and our replacement library has different symbols.

**The fix:** Build ACTS from source with C++20 enabled + EDM4HEP + Python bindings. The result is a self-contained ACTS install linked against spack's dependencies but with no spack ACTS involvement at runtime.

---

### Task 1: Diagnose Spack ACTS Build Configuration

Extract the exact cmake flags, C++ standard, and available dependencies from the spack ACTS install. This determines what flags we need.

**Files:**
- Modify: `modal_build_acts.py` (add `diagnose_spack` function, ~40 lines)

**Interfaces:**
- Produces: Diagnostic output confirming: C++ standard used, pybind11 path, cmake flags, ACTS version

- [ ] **Step 1: Add diagnostic function to `modal_build_acts.py`**

Add after the existing imports and before the `_setup_acts_env` function (around line 54):

```python
@app.function(
    image=image,
    volumes={BUILD_PATH: build_vol, DATA_PATH: data_vol},
    cpu=2,
    memory=4096,
    timeout=300,
)
def diagnose_spack():
    """Extract spack ACTS build config to determine correct cmake flags."""
    import glob, os, re

    acts_dir = glob.glob("/spack/opt/spack/linux-x86_64/acts-main-*/")[0]
    print(f"ACTS dir: {acts_dir}")

    # 1. Check ActsConfig.cmake for build flags
    config = glob.glob(f"{acts_dir}lib/cmake/Acts/ActsConfig.cmake")
    if config:
        with open(config[0]) as f:
            print("\n=== ActsConfig.cmake (key lines) ===")
            for line in f:
                if any(k in line for k in [
                    "CXX_STANDARD", "COMPONENTS", "PLUGIN", "EDM4HEP",
                    "PYTHON", "ROOT", "JSON", "DD4HEP", "GEANT4",
                    "FATRAS", "version", "VERSION"
                ]):
                    print(line.rstrip())

    # 2. Check installed cmake targets for compile features
    targets = glob.glob(f"{acts_dir}lib/cmake/Acts/ActsTargets*.cmake")
    for t in targets:
        with open(t) as f:
            content = f.read()
            cxx_refs = [l.strip() for l in content.split('\n')
                       if 'cxx_std' in l.lower() or 'CXX_STANDARD' in l]
            if cxx_refs:
                print(f"\n=== {os.path.basename(t)} C++ refs ===")
                for ref in cxx_refs:
                    print(ref)

    # 3. Check pybind11
    pybind = glob.glob("/spack/opt/spack/linux-x86_64/py-pybind11-*/")
    print(f"\n=== pybind11 ===")
    print(f"Available: {len(pybind) > 0}")
    for p in pybind:
        print(f"  {p}")
        cmake_dir = glob.glob(f"{p}lib/cmake/pybind11/")
        print(f"  cmake dir: {cmake_dir}")

    # 4. Check ACTS version
    ver = glob.glob(f"{acts_dir}include/Acts/ActsVersion.hpp")
    if ver:
        with open(ver[0]) as f:
            print("\n=== ActsVersion.hpp ===")
            for line in f:
                if 'VERSION' in line or 'COMMIT' in line:
                    print(line.rstrip())

    # 5. List all spack packages (to confirm deps are present)
    pkgs = sorted(os.path.basename(p.rstrip('/'))
                  for p in glob.glob("/spack/opt/spack/linux-x86_64/*/"))
    print(f"\n=== Spack packages ({len(pkgs)}) ===")
    for p in pkgs:
        print(f"  {p}")

    # 6. Check if spack CLI is available
    spack_bin = "/spack/bin/spack"
    print(f"\n=== spack CLI ===")
    print(f"Exists: {os.path.exists(spack_bin)}")
```

- [ ] **Step 2: Run diagnostic**

```bash
cd /Users/matthewm/SURP/cCKF && modal run modal_build_acts.py::diagnose_spack
```

Expected: Output showing C++ standard (likely 20), pybind11 location, ACTS version, and available components.

- [ ] **Step 3: Record findings**

Note the following from the output — these drive Task 2:
- `CMAKE_CXX_STANDARD` value (expected: 20)
- pybind11 cmake directory path
- Any ACTS cmake flags not currently in our build script
- Any missing dependencies

---

### Task 2: Update Build Script with Correct Flags

Based on diagnostic output, update `build_patched_acts.sh` to produce a fully self-contained ACTS build.

**Files:**
- Modify: `scripts/build_patched_acts.sh` (lines 72-92, cmake flags section)

**Interfaces:**
- Consumes: Diagnostic output from Task 1
- Produces: Updated build script that enables C++20, EDM4HEP, and Python bindings

- [ ] **Step 1: Add C++20 standard to cmake flags**

In `scripts/build_patched_acts.sh`, add `-DCMAKE_CXX_STANDARD=20` to the cmake invocation (line 72). This is the critical fix for podio 1.4.1 code generation compatibility.

```bash
cmake "$ACTS_SOURCE" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="$ACTS_INSTALL" \
    -DCMAKE_PREFIX_PATH="$CMAKE_PREFIX_PATH" \
    -DCMAKE_CXX_STANDARD=20 \
    -DACTS_BUILD_UNITTESTS=ON \
    -DACTS_BUILD_EXAMPLES=ON \
    -DACTS_BUILD_EXAMPLES_DD4HEP=ON \
    -DACTS_BUILD_EXAMPLES_EDM4HEP=ON \
    -DACTS_BUILD_EXAMPLES_GEANT4=ON \
    -DACTS_BUILD_PLUGIN_DD4HEP=ON \
    -DACTS_BUILD_PLUGIN_EDM4HEP=ON \
    -DACTS_BUILD_PLUGIN_ROOT=ON \
    -DACTS_BUILD_EXAMPLES_ROOT=ON \
    -DACTS_BUILD_PLUGIN_JSON=ON \
    -DACTS_BUILD_PLUGIN_GEANT4=ON \
    -DACTS_BUILD_PYTHON_BINDINGS=ON \
    -DACTS_BUILD_ODD=OFF \
    -DACTS_BUILD_FATRAS=ON \
    -DACTS_BUILD_FATRAS_GEANT4=ON \
    -DACTS_BUILD_ALIGNMENT=OFF \
    -DACTS_BUILD_BENCHMARKS=OFF
```

Key changes from current script:
- Added: `-DCMAKE_CXX_STANDARD=20`
- Changed: `ACTS_BUILD_EXAMPLES_EDM4HEP` from `${ENABLE_EDM4HEP:-OFF}` to `ON`
- Changed: `ACTS_BUILD_PLUGIN_EDM4HEP` from `${ENABLE_EDM4HEP:-OFF}` to `ON`
- Changed: `ACTS_BUILD_PYTHON_BINDINGS` from `${ENABLE_PYTHON:-OFF}` to `ON`

Note: If Task 1 reveals additional flags spack used, add them here.

- [ ] **Step 2: Remove the ENABLE_EDM4HEP / ENABLE_PYTHON env vars**

They're no longer needed since we always enable both. Remove the references to `${ENABLE_EDM4HEP:-OFF}` and `${ENABLE_PYTHON:-OFF}` throughout the script.

- [ ] **Step 3: Verify the PYTHONPATH/jinja2 workaround is still present**

The existing build script has a workaround (lines 18-47) to make pyyaml and jinja2 available to podio's code generator during the build. With EDM4HEP now ON, this workaround is load-bearing. Confirm these lines are intact:

```bash
# Capture Modal Python's pyyaml location BEFORE spack overrides PATH.
YAML_SITE=$(python3 -c "import yaml, os; print(os.path.dirname(os.path.dirname(yaml.__file__)))" 2>/dev/null || true)
if [ -n "$YAML_SITE" ]; then
    export PYTHONPATH="${YAML_SITE}:${PYTHONPATH:-}"
fi
```

And after spack env sourcing:
```bash
if [ -n "${YAML_SITE:-}" ]; then
    export PYTHONPATH="$YAML_SITE"
else
    unset PYTHONPATH 2>/dev/null || true
fi
```

Also confirm jinja2 is in the Modal image pip_install (it is: line 39 of `modal_build_acts.py`).

---

### Task 3: Update Modal Build Function

Force a clean rebuild with the new flags and handle the new build artifacts (Python bindings, EDM4HEP libraries).

**Files:**
- Modify: `modal_build_acts.py` — `build_acts` function and `_setup_acts_env`

**Interfaces:**
- Consumes: Updated `build_patched_acts.sh` from Task 2
- Produces: Complete ACTS install on volume at `/build/acts-install/` with Python bindings and EDM4HEP

- [ ] **Step 1: Remove the `--force` flag guard in `build_acts`**

The current `build_acts` function checks for a `.build_complete` marker and skips if present. For this rebuild, we need to force. Read the `build_acts` function to understand the current logic, then run with `--force`:

```bash
cd /Users/matthewm/SURP/cCKF && modal run modal_build_acts.py::build_acts --force
```

Wait for this to complete (~20-40 min on 16 cores). Watch for:

**Success indicators:**
- `Build complete` message
- `Installed to /build/acts-install`
- No compilation errors about C++20 concepts/requires

**Likely failure: podio code generation still fails despite C++20**

If the build fails with errors in podio-generated code, check:
1. Is the error about C++20 features or something else?
2. Does the generated code use `std::ranges`, `std::format`, or other C++20 stdlib features not available in gcc 13.3's libstdc++?
3. If so, try adding `-DCMAKE_CXX_FLAGS="-fconcepts"` as a more targeted fix

**Likely failure: pybind11 not found**

If cmake can't find pybind11:
1. Check Task 1 output for pybind11 path
2. Add it explicitly: `-Dpybind11_DIR=/spack/opt/spack/linux-x86_64/py-pybind11-.../share/cmake/pybind11/`
3. Or install via pip in the Modal image: `.pip_install("pybind11")` and set `-DPYBIND11_FINDPYTHON=ON`

- [ ] **Step 2: Verify build artifacts**

After successful build, check that all required outputs exist:

```bash
# In a Modal function or via build_acts output:
ls /build/acts-install/lib/libActsExamplesIoRoot.so
ls /build/acts-install/lib/libActsPluginEDM4hep.so      # NEW - EDM4HEP plugin
ls /build/acts-install/python/acts/__init__.py            # NEW - Python bindings
ls /build/acts-install/python/acts/examples/              # NEW - Example bindings
```

- [ ] **Step 3: Run unit tests**

```bash
cd /Users/matthewm/SURP/cCKF && modal run modal_build_acts.py::run_unit_tests
```

Expected: All existing tests pass, plus the new `ClusterFeaturesTests` from our patch.

---

### Task 4: Update `_setup_acts_env` for Self-Contained Patched ACTS

Replace the LD_PRELOAD strategy with direct use of the fully-built patched ACTS.

**Files:**
- Modify: `modal_build_acts.py` — `_setup_acts_env` function (lines 55-181) and `verify_root_branches` function (lines 334-end)

**Interfaces:**
- Consumes: Complete ACTS install from Task 3 at `/build/acts-install/`
- Produces: Working `_setup_acts_env(use_patched=True)` that loads patched ACTS with EDM4HEP + Python bindings

- [ ] **Step 1: Update `_setup_acts_env(use_patched=True)` Python path handling**

The current code at line 126-135 checks for Python bindings at two locations. After Task 3, bindings will be at `/build/acts-install/python/acts/`. Verify this path is handled:

```python
if use_patched:
    install_lib = f"{ACTS_INSTALL}/lib"
    build_lib = f"{ACTS_BUILD}/lib"

    # Python path: our install has python/acts/ directly
    acts_python = f"{ACTS_INSTALL}/python"
    if os.path.isdir(acts_python):
        # Create acts_pypath-style symlink for import compatibility
        import tempfile
        pypath_dir = tempfile.mkdtemp(prefix="acts_pypath_")
        os.symlink(acts_python, os.path.join(pypath_dir, "acts"))
        sys.path.insert(0, pypath_dir)
    
    # Also check lib/pythonX.Y/site-packages (older ACTS layout)
    py_dirs = sorted(
        glob.glob(f"{ACTS_INSTALL}/lib/python*/site-packages"),
        reverse=True,
    )
    for pydir in py_dirs:
        if pydir not in sys.path:
            sys.path.insert(0, pydir)
```

Wait — check what the actual install layout is. ACTS may install Python bindings to `{prefix}/python/` (flat, like spack) or `{prefix}/lib/python3.13/site-packages/acts/` (standard). The correct handling depends on the actual layout. Read the ACTS cmake Python install rules to determine this, or just check after the build completes.

If the layout is `{prefix}/python/__init__.py` (flat, same as spack), we need the symlink trick. If it's `{prefix}/lib/python3.13/site-packages/acts/`, we just add the site-packages to sys.path.

- [ ] **Step 2: Rewrite `verify_root_branches` — remove LD_PRELOAD, use patched ACTS directly**

The entire subprocess + LD_PRELOAD approach is no longer needed. Since we now have a complete patched ACTS with EDM4HEP and Python bindings, we can run the CKF directly in-process.

Replace the current verify function body (everything after the material map check) with:

```python
def verify_root_branches(event_id: int = 0):
    """Run 1-event CKF with patched ACTS, check new branches."""
    import os, time, glob
    from pathlib import Path

    _setup_acts_env(use_patched=True)

    # Material maps
    material_map = glob.glob(f"{DATA_PATH}/odd-material-maps.root")
    if not material_map:
        material_map = glob.glob(f"{DATA_PATH}/material/odd-material-maps.root")
    if not material_map:
        raise FileNotFoundError("Material maps not found. Run seed_material_maps first.")
    material_map = material_map[0]
    odd_data_dir = f"{ODD_INSTALL}/data"
    os.makedirs(odd_data_dir, exist_ok=True)
    map_link = f"{odd_data_dir}/odd-material-maps.root"
    if not os.path.exists(map_link):
        os.symlink(material_map, map_link)

    output_dir = Path(f"{DATA_PATH}/results/verify_branches_{int(time.time())}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load config
    import yaml, types
    with open("/app/configs/envelope.yaml") as f:
        cfg_dict = yaml.safe_load(f)
    cfg_dict.update({
        "events": 1,
        "skip": event_id,
        "threads": 8,
        "write_track_states": True,
        "write_predicted_cov": True,
        "ckf": True,
        "ambi": False,
        "ckf_finding_performance": False,
        "ambi_finding_performance": False,
        "output_digi_csv": True,
        "output_simhits_csv": True,
        "output_seeds_csv": True,
    })
    cfg = types.SimpleNamespace(**cfg_dict)
    cfg.seed = 42

    # Run CKF directly (no subprocess needed)
    import acts, acts.examples
    print(f"acts module: {acts.__file__}")
    from digi_and_reco import setup_acts_reconstruction
    rnd = acts.examples.RandomNumbers(seed=cfg.seed)

    t0 = time.time()
    s = setup_acts_reconstruction(
        Path(f"{DATA_PATH}/events/edm4hep.root"),
        output_dir,
        cfg, rnd,
    )
    s.run()
    print(f"CKF completed in {time.time() - t0:.1f}s")

    # ... (keep existing uproot branch verification code)
```

The key change: no subprocess, no LD_PRELOAD, no runner script. Just `_setup_acts_env(use_patched=True)` then direct Python calls.

- [ ] **Step 3: Keep the uproot verification logic**

The existing code at the end of `verify_root_branches` that uses uproot to check for the 12 expected branches should be preserved. It checks:

```python
EXPECTED_BRANCHES = [
    "S00_prt", "S01_prt", "S11_prt",
    "pathInX0_interval",
    "clus_size_u", "clus_size_v", "clus_qtot",
    "clus_sigma_uu", "clus_sigma_uv", "clus_sigma_vv",
    "alpha_u", "alpha_v",
]
```

And verifies each branch exists and contains finite, nonzero values.

---

### Task 5: Run Full Verification

**Files:**
- No file changes — just running the updated code

**Interfaces:**
- Consumes: Everything from Tasks 1-4
- Produces: Confirmation that 12 new ROOT branches are present with valid data

- [ ] **Step 1: Run the verification**

```bash
cd /Users/matthewm/SURP/cCKF && modal run modal_build_acts.py::verify_root_branches
```

Expected output:
- `acts module: /build/acts-install/python/acts/__init__.py` (or similar, showing patched ACTS)
- `CKF completed in X.Xs`
- All 12 branches found with finite, nonzero values
- `PASS: All 12 expected branches verified`

- [ ] **Step 2: If verification fails, debug**

Common failure modes:

**`ModuleNotFoundError: No module named 'acts'`**
→ Python path not set up correctly in `_setup_acts_env`. Check the install layout and symlink.

**`ImportError: undefined symbol ...`**
→ ABI mismatch. But this shouldn't happen since we built everything from source. Check that `LD_LIBRARY_PATH` points to `/build/acts-install/lib` FIRST (before spack's ACTS lib). Also check that ctypes preloading loads from our install, not spack's.

**`ImportError: libedm4hep*.so not found`**
→ EDM4HEP shared libraries not on `LD_LIBRARY_PATH`. The edm4hep libs are in spack's tree and should be found if `_setup_acts_env` sources `/etc/acts_env.sh`.

**CKF runs but branches are missing**
→ The `inputClusters` wiring in our patch (Patch D, `Python/Examples/python/reconstruction.py`) might not be executing. Check that the patched `reconstruction.py` is what gets imported — it should be at `/build/acts-install/python/acts/examples/reconstruction.py`, and our install should be first on sys.path.

- [ ] **Step 3: Confirm data constraint**

Verify the run used event 0 (which is in the [0,32) train split). The default `event_id=0` is correct.

---

## Risk Register

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| C++20 doesn't fix podio code gen | Low | High | Check exact error; try `-fconcepts` flag or pin older podio headers |
| pybind11 not in spack tree | Low | Medium | Install via pip; add to Modal image |
| ACTS code doesn't compile with C++20 | Low | Medium | Spack's ACTS (same era) compiles fine; fix any deprecation warnings |
| Build takes >2 hours (timeout) | Medium | Low | Increase Modal timeout; build with `-j$(nproc)` already set |
| Python bindings layout differs from spack | Medium | Low | Check actual layout post-build; adapt symlink accordingly |
| Our ACTS commit incompatible with spack's edm4hep | Low | High | Our commit is close to spack's; edm4hep API is stable |

## Fallback: Full Spack Rebuild

If the cmake approach fails for unexpected reasons, the nuclear option is to rebuild through spack directly:

1. Modify `Dockerfile.modal` stage 1 to clone ACTS, apply patch, use `spack develop` + `spack install`
2. This guarantees ABI compatibility by construction
3. Downside: requires understanding spack's spec for ACTS (variants, dependencies)
4. Only try this if cmake + C++20 fails
