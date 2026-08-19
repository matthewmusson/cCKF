"""
Build patched ACTS from source on Modal and run verification.

Uses the same Dockerfile.modal base image (clean Ubuntu + spack tree) that
the existing pipeline uses, extended with cmake for building from source.

Usage:
    # Full build with Python bindings + unit tests
    modal run modal_build_acts.py

    # Just build (skip tests)
    modal run modal_build_acts.py::build_acts

    # Run unit tests on cached build
    modal run modal_build_acts.py::run_unit_tests

    # Run 1-event CKF to verify new ROOT branches
    modal run modal_build_acts.py::verify_root_branches
"""

import json
import modal

app = modal.App("surp-acts-build")

build_vol = modal.Volume.from_name("surp-acts-build", create_if_missing=True)
data_vol = modal.Volume.from_name("surp-acts-data", create_if_missing=True)
BUILD_PATH = "/build"
DATA_PATH = "/data"

ACTS_SOURCE = f"{BUILD_PATH}/acts-src"
ACTS_BUILD = f"{BUILD_PATH}/acts-build"
ACTS_INSTALL = f"{BUILD_PATH}/acts-install"
ODD_INSTALL = "/opt/ODD_v5/install/share/OpenDataDetector"

image = (
    modal.Image.from_dockerfile("Dockerfile.modal", add_python="3.13")
    .apt_install("cmake", "make", "git")
    .pip_install("uproot", "awkward", "pyyaml", "pandas", "pyarrow", "numpy", "jinja2", "matplotlib", "scipy")
    .add_local_file(
        "instrumentation.patch",
        remote_path="/app/instrumentation.patch",
    )
    .add_local_file(
        "scripts/build_patched_acts.sh",
        remote_path="/app/build_patched_acts.sh",
    )
    .add_local_dir("configs", remote_path="/app/configs")
    .add_local_file("digi_and_reco.py", remote_path="/app/digi_and_reco.py")
    .add_local_file("window_failure.py", remote_path="/app/window_failure.py")
    .add_local_file("expansion.py", remote_path="/app/expansion.py")
    .add_local_dir("utils", remote_path="/app/utils")
    .add_local_dir("scripts", remote_path="/app/scripts")
    .add_local_dir("acts_patches", remote_path="/app/acts_patches")
)


@app.function(
    image=image,
    volumes={BUILD_PATH: build_vol, DATA_PATH: data_vol},
    cpu=2,
    memory=4096,
    timeout=300,
)
def diagnose_spack():
    """Extract spack ACTS build config to determine correct cmake flags."""
    import glob
    import os

    acts_dir = glob.glob("/spack/opt/spack/linux-x86_64/acts-main-*/")[0]
    print(f"ACTS dir: {acts_dir}")

    # 1. ActsConfig.cmake — build flags
    config = glob.glob(f"{acts_dir}lib/cmake/Acts/ActsConfig.cmake")
    if config:
        with open(config[0]) as f:
            print("\n=== ActsConfig.cmake (key lines) ===")
            for line in f:
                if any(k in line for k in [
                    "CXX_STANDARD", "COMPONENTS", "PLUGIN", "EDM4HEP",
                    "PYTHON", "ROOT", "JSON", "DD4HEP", "GEANT4",
                    "FATRAS", "version", "VERSION",
                ]):
                    print(line.rstrip())

    # 2. Cmake targets — compile features / C++ standard
    targets = glob.glob(f"{acts_dir}lib/cmake/Acts/ActsTargets*.cmake")
    for t in targets:
        with open(t) as f:
            for line in f:
                if "cxx_std" in line.lower() or "CXX_STANDARD" in line:
                    print(f"\n[{os.path.basename(t)}] {line.strip()}")

    # 3. pybind11
    pybind = glob.glob("/spack/opt/spack/linux-x86_64/py-pybind11-*/")
    print(f"\n=== pybind11 ===")
    print(f"Available: {len(pybind) > 0}")
    for p in pybind:
        print(f"  {p}")

    # 4. ACTS version
    ver = glob.glob(f"{acts_dir}include/Acts/ActsVersion.hpp")
    if ver:
        with open(ver[0]) as f:
            print("\n=== ActsVersion.hpp ===")
            for line in f:
                if "VERSION" in line or "COMMIT" in line:
                    print(line.rstrip())

    # 5. Test if spack CLI actually works
    import subprocess
    spack_bin = "/spack/bin/spack"
    print(f"\n=== Spack CLI ===")
    print(f"Exists: {os.path.exists(spack_bin)}")

    if os.path.exists(spack_bin):
        for cmd_label, cmd in [
            ("spack --version", [spack_bin, "--version"]),
            ("spack find --long acts-main", [spack_bin, "find", "--long", "acts-main"]),
            ("spack spec acts-main", [spack_bin, "spec", "acts-main"]),
            ("spack find --variants acts-main", [spack_bin, "find", "--variants", "acts-main"]),
        ]:
            print(f"\n--- {cmd_label} ---")
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            print(r.stdout[:2000] if r.stdout else "(no stdout)")
            if r.returncode != 0:
                print(f"STDERR: {r.stderr[:1000]}")
                print(f"Exit code: {r.returncode}")

        # Check if spack store is writable
        print("\n--- spack store writable? ---")
        test_dir = "/spack/opt/spack/.write_test"
        try:
            os.makedirs(test_dir, exist_ok=True)
            os.rmdir(test_dir)
            print("YES — /spack/opt/spack is writable")
        except OSError as e:
            print(f"NO — {e}")


def _setup_acts_env(use_patched: bool = False):
    """Set up environment for ACTS (cmake deps, LD_LIBRARY_PATH, sys.path).

    use_patched=True prepends the source-built ACTS install to PATH and
    LD_LIBRARY_PATH so `import acts` loads our patched binaries.
    """
    import glob
    import os
    import subprocess
    import sys
    from pathlib import Path

    spack_paths_file = "/etc/spack_site_packages"
    if os.path.exists(spack_paths_file):
        with open(spack_paths_file) as f:
            for line in f:
                p = line.strip()
                if p and p not in sys.path:
                    sys.path.append(p)

    env_file = "/etc/acts_env.sh"
    if os.path.exists(env_file):
        result = subprocess.run(
            ["bash", "-c", f"source {env_file} && env"],
            capture_output=True,
            text=True,
        )
        for line in result.stdout.splitlines():
            if "=" in line:
                key, _, val = line.partition("=")
                os.environ[key] = val

    # ODD factory libs
    odd_lib = "/opt/ODD_v5/install/lib"
    ld = os.environ.get("LD_LIBRARY_PATH", "")
    if odd_lib not in ld.split(":"):
        os.environ["LD_LIBRARY_PATH"] = f"{odd_lib}:{ld}" if ld else odd_lib

    os.environ["ODD_PATH"] = ODD_INSTALL

    # edm4hep dictionaries
    spack = Path("/spack/opt/spack/linux-x86_64")
    include_dirs = [
        str(p)
        for pattern in ("edm4hep-*/include", "podio-*/include", "hepmc3-*/include")
        for p in spack.glob(pattern)
    ]
    if include_dirs:
        rip = os.environ.get("ROOT_INCLUDE_PATH", "")
        os.environ["ROOT_INCLUDE_PATH"] = ":".join(
            include_dirs + ([rip] if rip else [])
        )

    # PyROOT
    root_dir = os.environ.get("ROOT_DIR") or os.environ.get("ROOTSYS")
    root_candidates = []
    if root_dir:
        root_candidates.append(Path(root_dir) / "lib")
        root_candidates.append(Path(root_dir) / "lib" / "root")
    root_candidates.extend(spack.glob("root-*/lib"))
    root_candidates.extend(spack.glob("root-*/lib/root"))
    for candidate in root_candidates:
        if candidate.is_dir() and str(candidate) not in sys.path:
            sys.path.append(str(candidate))

    if use_patched:
        # Prepend our source-built install so `import acts` loads patched code
        install_lib = f"{ACTS_INSTALL}/lib"
        build_lib = f"{ACTS_BUILD}/lib"

        # Python path: check both lib/pythonX.Y/site-packages and python/
        py_dirs = sorted(
            glob.glob(f"{ACTS_INSTALL}/lib/python*/site-packages"),
            reverse=True,
        )
        acts_python = f"{ACTS_INSTALL}/python"
        if os.path.isdir(acts_python):
            py_dirs.insert(0, acts_python)
        for pydir in py_dirs:
            if pydir not in sys.path:
                sys.path.insert(0, pydir)

        # LD_LIBRARY_PATH: our libs first (for child processes)
        ld = os.environ.get("LD_LIBRARY_PATH", "")
        parts = ld.split(":") if ld else []
        for lib_dir in [build_lib, install_lib]:
            if lib_dir not in parts:
                parts.insert(0, lib_dir)
        os.environ["LD_LIBRARY_PATH"] = ":".join(parts)

        # Preload patched ACTS shared libs so dlopen() can find them
        # (LD_LIBRARY_PATH changes don't affect the current process's linker)
        import ctypes
        acts_libs_loaded = 0
        for lib_dir in [install_lib, build_lib]:
            for lib in sorted(glob.glob(f"{lib_dir}/libActs*.so")):
                if ".so." in Path(lib).name:
                    continue
                try:
                    ctypes.CDLL(lib, mode=ctypes.RTLD_GLOBAL)
                    acts_libs_loaded += 1
                except OSError:
                    pass
        print(f"Preloaded {acts_libs_loaded} patched ACTS shared libraries")

    else:
        acts_pypath = "/opt/acts_pypath"
        if acts_pypath not in sys.path:
            sys.path.insert(0, acts_pypath)

    # Preload edm4hep/podio shared libs
    import ctypes
    loaded_libs = []
    for pattern in (
        "/spack/opt/spack/linux-x86_64/podio-*/lib/libpodio*.so",
        "/spack/opt/spack/linux-x86_64/edm4hep-*/lib/libedm4hep*.so",
    ):
        for lib in sorted(glob.glob(pattern)):
            if ".so." in Path(lib).name:
                continue
            try:
                ctypes.CDLL(lib, mode=ctypes.RTLD_GLOBAL)
                loaded_libs.append(lib)
            except OSError:
                pass
    print(f"Preloaded {len(loaded_libs)} edm4hep/podio shared libraries")


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

@app.function(
    image=image,
    volumes={BUILD_PATH: build_vol},
    cpu=16,
    memory=32768,
    timeout=7200,
)
def build_acts(force: bool = False, target: str = "all", python: bool = True):
    """Build patched ACTS from source. Results cached on the build volume."""
    import os
    import subprocess

    _setup_acts_env()

    marker_name = ".build_complete_full" if target == "all" else ".build_complete"
    marker = f"{ACTS_BUILD}/{marker_name}"
    if os.path.exists(marker) and not force:
        print(f"Build already cached on volume ({marker_name}).")
        install_dir = f"{ACTS_INSTALL}/lib"
        print(f"Install lib exists: {os.path.isdir(install_dir)}")
        return {"status": "cached"}

    for d in [ACTS_SOURCE, ACTS_BUILD, ACTS_INSTALL]:
        subprocess.run(["rm", "-rf", d])

    env = os.environ.copy()
    env["BUILD_TARGET"] = target
    env["ACTS_SOURCE"] = ACTS_SOURCE
    env["ACTS_BUILD"] = ACTS_BUILD
    env["ACTS_INSTALL"] = ACTS_INSTALL
    env["ENABLE_PYTHON"] = "ON" if python else "OFF"

    print(f"=== Starting ACTS build (16 cores, target={target}, python={python}) ===")
    result = subprocess.run(
        ["bash", "/app/build_patched_acts.sh"],
        env=env,
        timeout=6000,
    )

    if result.returncode != 0:
        raise RuntimeError(f"Build failed with exit code {result.returncode}")

    with open(marker, "w") as f:
        f.write("ok\n")

    build_vol.commit()
    print("Build artifacts saved to volume.")

    # Report what we installed
    import glob
    py_sites = glob.glob(f"{ACTS_INSTALL}/lib/python*/site-packages/acts")
    print(f"Python acts module installed: {py_sites}")
    libs = glob.glob(f"{ACTS_INSTALL}/lib/libActs*.so")
    print(f"Shared libraries installed: {len(libs)}")

    return {"status": "built", "python_sites": py_sites}


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

@app.function(
    image=image,
    volumes={BUILD_PATH: build_vol},
    cpu=4,
    memory=8192,
    timeout=600,
)
def run_unit_tests():
    """Run the ClusterFeatures unit test from the cached build."""
    import glob
    import os
    import subprocess

    _setup_acts_env()
    build_vol.reload()

    test_binary = f"{ACTS_BUILD}/bin/ActsUnitTestClusterFeatures"
    if not os.path.exists(test_binary):
        raise FileNotFoundError(
            f"Test binary not found at {test_binary}. Run build_acts first."
        )

    print(f"=== Running ActsUnitTestClusterFeatures ===")

    env = os.environ.copy()
    ld_paths = []
    for d in [f"{ACTS_BUILD}/lib", f"{ACTS_INSTALL}/lib"]:
        if os.path.isdir(d):
            ld_paths.append(d)
    for lib_dir in sorted(glob.glob("/spack/opt/spack/linux-x86_64/*/lib")):
        ld_paths.append(lib_dir)
    existing = env.get("LD_LIBRARY_PATH", "")
    if existing:
        ld_paths.append(existing)
    env["LD_LIBRARY_PATH"] = ":".join(ld_paths)

    result = subprocess.run(
        [test_binary, "--log_level=all", "--report_level=detailed"],
        capture_output=True,
        text=True,
        timeout=120,
        env=env,
    )

    print("--- STDOUT ---")
    print(result.stdout)
    if result.stderr:
        print("--- STDERR ---")
        print(result.stderr)
    print(f"\nReturn code: {result.returncode}")

    if result.returncode != 0:
        raise RuntimeError(f"Unit tests FAILED (exit code {result.returncode})")

    print("\nAll ClusterFeatures tests PASSED.")
    return {"status": "passed", "returncode": result.returncode}


# ---------------------------------------------------------------------------
# Verify patched ROOT branches on 1 event
# ---------------------------------------------------------------------------

EXPECTED_BRANCHES = [
    "S00_prt",
    "S01_prt",
    "S11_prt",
    "pathInX0_interval",
    "clus_size_u",
    "clus_size_v",
    "clus_qtot",
    "clus_sigma_uu",
    "clus_sigma_uv",
    "clus_sigma_vv",
    "alpha_u",
    "alpha_v",
]


@app.function(
    image=image,
    volumes={BUILD_PATH: build_vol, DATA_PATH: data_vol},
    cpu=8,
    memory=65536,
    timeout=3600,
)
def verify_root_branches(event_id: int = 0):
    """Run 1-event CKF with patched ACTS (full build), check new branches."""
    import os
    import time
    import types
    from pathlib import Path

    build_vol.reload()
    _setup_acts_env(use_patched=True)

    # Symlink material maps
    material_map = f"{DATA_PATH}/odd-material-maps.root"
    if not os.path.exists(material_map):
        raise FileNotFoundError("Material maps not found. Run seed_material_maps first.")
    odd_data_dir = f"{ODD_INSTALL}/data"
    os.makedirs(odd_data_dir, exist_ok=True)
    map_link = f"{odd_data_dir}/odd-material-maps.root"
    if not os.path.exists(map_link):
        os.symlink(material_map, map_link)

    output_dir = Path(f"{DATA_PATH}/results/verify_branches_{int(time.time())}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load config
    import yaml
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

    # Run CKF directly — no subprocess, no LD_PRELOAD
    import acts, acts.examples
    print(f"acts module: {acts.__file__}")

    from digi_and_reco import setup_acts_reconstruction
    rnd = acts.examples.RandomNumbers(seed=cfg.seed)

    print(f"=== Running 1-event CKF (event {event_id}) ===")
    t0 = time.time()
    s = setup_acts_reconstruction(
        Path(f"{DATA_PATH}/events/edm4hep.root"),
        output_dir,
        cfg, rnd,
    )
    s.run()
    wall = time.time() - t0
    print(f"Pipeline completed in {wall:.1f}s")

    # Check output files
    print("\n=== Output files ===")
    for f in sorted(output_dir.iterdir()):
        if f.name.startswith("_"):
            continue
        size_mb = f.stat().st_size / 1e6
        print(f"  {f.name}: {size_mb:.2f} MB")

    # Inspect ROOT trackstates for new branches
    print("\n=== Checking ROOT branches ===")
    import uproot
    import numpy as np

    ts_path = output_dir / "trackstates_ckf.root"
    if not ts_path.exists():
        raise FileNotFoundError(f"trackstates not produced: {ts_path}")

    with uproot.open(ts_path) as root_file:
        keys = list(root_file.keys())
        print(f"Trees in file: {keys}")
        tree_key = next(
            (k for k in keys if "trackstate" in k.lower() or "states" in k.lower()),
            keys[0],
        )
        tree = root_file[tree_key]
        all_branches = sorted(tree.keys())
        n_entries = tree.num_entries
        print(f"Tree: {tree_key}")
        print(f"Entries (tracks): {n_entries}")
        print(f"Total branches: {len(all_branches)}")

        results = {}
        for branch in EXPECTED_BRANCHES:
            present = branch in all_branches
            results[branch] = {"present": present}
            if present:
                arr = tree[branch].array(entry_start=0, entry_stop=min(5, n_entries))
                flat = []
                for entry in arr:
                    try:
                        flat.extend(entry.tolist())
                    except AttributeError:
                        flat.append(entry)
                flat = [x for x in flat if x is not None]
                n_finite = sum(1 for x in flat if np.isfinite(x))
                n_nonzero = sum(1 for x in flat if x != 0.0)
                results[branch]["sample_size"] = len(flat)
                results[branch]["n_finite"] = n_finite
                results[branch]["n_nonzero"] = n_nonzero
                if flat:
                    results[branch]["min"] = float(min(flat))
                    results[branch]["max"] = float(max(flat))

        print("\n=== Branch verification ===")
        all_pass = True
        for branch in EXPECTED_BRANCHES:
            info = results[branch]
            if not info["present"]:
                print(f"  FAIL  {branch}: NOT FOUND")
                all_pass = False
            elif info["n_finite"] == 0:
                print(f"  WARN  {branch}: present but all NaN/inf")
            elif info["n_nonzero"] == 0:
                print(f"  WARN  {branch}: present but all zero")
            else:
                print(
                    f"  PASS  {branch}: "
                    f"finite={info['n_finite']}/{info['sample_size']} "
                    f"nonzero={info['n_nonzero']} "
                    f"range=[{info['min']:.4g}, {info['max']:.4g}]"
                )

        print(f"\n=== All {len(all_branches)} branches ===")
        for b in all_branches:
            marker = " <-- NEW" if b in EXPECTED_BRANCHES else ""
            print(f"  {b}{marker}")

    data_vol.commit()

    summary = {
        "status": "PASS" if all_pass else "FAIL",
        "event_id": event_id,
        "n_tracks": n_entries,
        "wall_seconds": round(wall, 1),
        "output_dir": str(output_dir),
        "branches": results,
    }
    print(f"\n=== Summary: {summary['status']} ===")
    print(json.dumps(summary, indent=2, default=str))
    return summary


@app.function(
    image=image,
    volumes={DATA_PATH: data_vol},
    cpu=2,
    memory=8192,
    timeout=300,
)
def dump_track_state(track_idx: int = 0, state_idx: int = 0):
    """Dump all branch values for one track state from the verified ROOT file."""
    import glob

    import numpy as np
    import uproot

    data_vol.reload()
    root_files = sorted(glob.glob(f"{DATA_PATH}/results/verify_branches_*/trackstates_ckf.root"))
    if not root_files:
        raise FileNotFoundError("No trackstates_ckf.root found in results dirs")
    root_path = root_files[-1]
    print(f"Reading: {root_path}")

    f = uproot.open(root_path)
    tree = f[f.keys()[0]]
    branches = sorted(tree.keys())

    # Read one track
    arrays = tree.arrays(branches, entry_start=track_idx, entry_stop=track_idx + 1)

    print(f"\n=== Track {track_idx}, state {state_idx} ===")
    print(f"Total branches: {len(branches)}")

    n_states = None
    for b in branches:
        vec = arrays[b][0]
        try:
            vals = vec.tolist()
            if not isinstance(vals, list):
                vals = [vals]
        except AttributeError:
            vals = [vec]
        if n_states is None and len(vals) > 1:
            n_states = len(vals)
            print(f"States in this track: {n_states}\n")
        if len(vals) == 1:
            v = vals[0]
            label = "(per-track)"
        elif state_idx < len(vals):
            v = vals[state_idx]
            label = ""
        else:
            v = "[idx OOB]"
            label = ""
        if isinstance(v, float) and np.isnan(v):
            print(f"  {b:40s} = NaN")
        else:
            print(f"  {b:40s} = {v}  {label}")

    return {"track_idx": track_idx, "state_idx": state_idx, "n_states": n_states, "n_branches": len(branches)}


@app.function(
    image=image,
    volumes={BUILD_PATH: build_vol, DATA_PATH: data_vol},
    cpu=8,
    memory=65536,
    timeout=1800,
)
def check_dm_track_recovery(event_id: int = 0):
    """Pilot Check 2b: Compare DM-matched particles between tight and envelope CKF.

    Runs tight config with full CKF on one event, then compares the set of
    DM-matched truth particles against the existing envelope CKF output.
    The envelope should recover all particles that tight finds.
    """
    import glob
    import os
    import time
    import types
    from pathlib import Path

    import numpy as np
    import pandas as pd

    build_vol.reload()
    data_vol.reload()
    _setup_acts_env(use_patched=True)

    material_map = f"{DATA_PATH}/odd-material-maps.root"
    odd_data_dir = f"{ODD_INSTALL}/data"
    os.makedirs(odd_data_dir, exist_ok=True)
    map_link = f"{odd_data_dir}/odd-material-maps.root"
    if not os.path.exists(map_link):
        os.symlink(material_map, map_link)

    # --- Step 1: Find envelope CKF output from prior run ---
    env_dirs = sorted(glob.glob(f"{DATA_PATH}/results/verify_branches_*"))
    if not env_dirs:
        raise FileNotFoundError("No envelope run found")
    env_dir = env_dirs[-1]
    print(f"Envelope dir: {env_dir}")

    # --- Step 2: Run tight CKF (no ambi, matching envelope's setup) ---
    import yaml
    with open("/app/configs/tight_t79.yaml") as f:
        tight_dict = yaml.safe_load(f)

    tight_dict.update({
        "events": 1, "skip": event_id, "threads": 8,
        "ckf": True, "ambi": False,
        "output_seeds_csv": False, "output_digi_csv": False,
        "output_simhits_csv": False,
        "write_track_states": True, "write_predicted_cov": False,
        "ckf_finding_performance": False, "ambi_finding_performance": False,
    })
    tight_cfg = types.SimpleNamespace(**tight_dict)
    tight_cfg.seed = 42

    tight_dir = Path(f"{DATA_PATH}/results/tight_full_{int(time.time())}")
    tight_dir.mkdir(parents=True, exist_ok=True)

    import acts, acts.examples
    from digi_and_reco import setup_acts_reconstruction
    rnd = acts.examples.RandomNumbers(seed=tight_cfg.seed)

    print(f"\n=== Running tight CKF (event {event_id}) ===")
    t0 = time.time()
    s = setup_acts_reconstruction(
        Path(f"{DATA_PATH}/events/edm4hep.root"),
        tight_dir, tight_cfg, rnd,
    )
    s.run()
    wall = time.time() - t0
    print(f"Tight pipeline completed in {wall:.1f}s")

    # --- Step 3: DM matching on both track sets ---
    import uproot
    import numpy as np

    env_root = f"{env_dir}/trackstates_ckf.root"
    tight_root = f"{tight_dir}/trackstates_ckf.root"
    print(f"\nEnvelope ROOT: {env_root} (exists: {os.path.exists(env_root)})")
    print(f"Tight ROOT:    {tight_root} (exists: {os.path.exists(tight_root)})")

    def _dm_matched_particles_UNUSED(root_path, label):
        """DEPRECATED — replaced by load_flat_states + dm_match_with_shared_denom.

        DM matching: for each track, the majority particle must have
        purity >= 0.5 AND completeness >= 0.5.

        Completeness denominator = number of UNIQUE detector modules hit by each
        particle, deduped by (volume_id, layer_id, module_id) across all CKF
        candidates. This prevents inflated denominators from the same measurement
        appearing on multiple candidate tracks.
        """
        f = uproot.open(root_path)
        tree = f["trackstates"]
        all_branches = tree.keys()

        # Tree structure: one entry per TRACK, with jagged arrays per state.
        # particle_ids are doubly jagged: track -> states -> contributing particles.
        # Take the primary (first) particle per state for DM matching.
        import awkward as ak

        pid_cols = [
            "particle_ids_vertex_primary",
            "particle_ids_vertex_secondary",
            "particle_ids_particle",
            "particle_ids_generation",
            "particle_ids_sub_particle",
        ]
        needed = ["track_nr", "nMeasurements", "volume_id", "layer_id", "module_id"] + pid_cols
        print(f"[{label}] Reading branches...")

        data = tree.arrays(needed, library="ak")
        n_tracks = len(data)
        print(f"\n[{label}] Total tracks (tree entries): {n_tracks}")

        # For each state, take the primary particle (first in list; ACTS stores
        # highest-weight first). ak.firsts gives None for empty lists (holes).
        primary_pids = {}
        for c in pid_cols:
            primary_pids[c] = ak.firsts(data[c], axis=2)

        # Encode the 5 barcode components into a single int64 for fast hashing.
        # Barcode layout: vertex_primary(16b) | vertex_secondary(16b) | particle(16b) | generation(8b) | sub(8b)
        pid_encoded = (
            ak.values_astype(primary_pids["particle_ids_vertex_primary"], np.int64) * (1 << 48)
            + ak.values_astype(primary_pids["particle_ids_vertex_secondary"], np.int64) * (1 << 32)
            + ak.values_astype(primary_pids["particle_ids_particle"], np.int64) * (1 << 16)
            + ak.values_astype(primary_pids["particle_ids_generation"], np.int64) * (1 << 8)
            + ak.values_astype(primary_pids["particle_ids_sub_particle"], np.int64)
        )  # jagged: track -> states, with None for holes

        # Flatten to a flat table: (track_nr, pid, volume_id, layer_id, module_id)
        # Broadcast scalar track_nr to match per-state jagged arrays
        track_nr_broad, pid_flat = ak.broadcast_arrays(data["track_nr"], pid_encoded)
        _, vol_flat = ak.broadcast_arrays(data["track_nr"], data["volume_id"])
        _, lay_flat = ak.broadcast_arrays(data["track_nr"], data["layer_id"])
        _, mod_flat = ak.broadcast_arrays(data["track_nr"], data["module_id"])

        df = pd.DataFrame({
            "track_nr": ak.to_numpy(ak.flatten(track_nr_broad)),
            "pid": ak.to_numpy(ak.flatten(pid_flat)),
            "volume_id": ak.to_numpy(ak.flatten(vol_flat)),
            "layer_id": ak.to_numpy(ak.flatten(lay_flat)),
            "module_id": ak.to_numpy(ak.flatten(mod_flat)),
        })
        # Replace NaN pids (from holes where ak.firsts returned None)
        df["pid"] = df["pid"].fillna(0).astype(np.int64)
        null_pid = 0

        print(f"[{label}] Total states (flat): {len(df)}")

        # Filter to states with a valid truth particle (not null barcode)
        valid = df[df["pid"] != null_pid].copy()
        print(f"[{label}] Valid (non-null particle) states: {len(valid)}")

        # Unique measurements per particle (dedup by detector module)
        dedup = valid.drop_duplicates(subset=["pid", "volume_id", "layer_id", "module_id"])
        particle_total = dedup.groupby("pid").size()
        print(f"[{label}] Unique truth particles with hits: {len(particle_total)}")

        # Per-track: count measurement states and majority particle hits
        n_meas_per_track = valid.groupby("track_nr").size()

        # For each (track, particle) pair, count hits
        track_particle_counts = valid.groupby(["track_nr", "pid"]).size().reset_index(name="count")
        # Keep only the majority particle per track
        idx_max = track_particle_counts.groupby("track_nr")["count"].idxmax()
        majority = track_particle_counts.loc[idx_max].copy()
        majority = majority.rename(columns={"count": "majority_count", "pid": "majority_pid"})
        majority["n_meas"] = majority["track_nr"].map(n_meas_per_track)
        majority["purity"] = majority["majority_count"] / majority["n_meas"]
        majority["total_hits"] = majority["majority_pid"].map(particle_total)
        majority["completeness"] = majority["majority_count"] / majority["total_hits"]

        dm = majority[(majority["purity"] >= 0.5) & (majority["completeness"] >= 0.5)]

        # Tracks with no valid measurement states
        all_tracks = df["track_nr"].unique()
        tracks_with_meas = valid["track_nr"].unique()
        n_no_majority = len(all_tracks) - len(tracks_with_meas)

        track_stats = {
            "total": len(all_tracks),
            "dm_matched": len(dm),
            "no_majority": int(n_no_majority),
            "low_purity": int((majority["purity"] < 0.5).sum()),
            "low_completeness": int(
                ((majority["purity"] >= 0.5) & (majority["completeness"] < 0.5)).sum()
            ),
        }
        matched_particles = set(dm["majority_pid"].unique())

        print(f"[{label}] Track stats: {track_stats}")
        print(f"[{label}] DM-matched unique particles: {len(matched_particles)}")

        # Summary statistics
        print(f"[{label}] Purity  — median: {majority['purity'].median():.3f}, "
              f"mean: {majority['purity'].mean():.3f}")
        print(f"[{label}] Completeness — median: {majority['completeness'].median():.3f}, "
              f"mean: {majority['completeness'].mean():.3f}")

        return matched_particles, track_stats

    # --- DM matching with a SHARED completeness denominator ---
    # The completeness denominator (total unique measurements per particle) must
    # be consistent across both runs. Using each run's own denominator biases
    # against the envelope (which explores more surfaces, inflating the count).
    # Fix: compute the union of both runs' particle-module counts.
    import awkward as ak

    def load_flat_states(root_path, label):
        """Load trackstates and return a flat DataFrame of (track_nr, pid, vol, lay, mod)."""
        f = uproot.open(root_path)
        tree = f["trackstates"]
        pid_cols = [
            "particle_ids_vertex_primary",
            "particle_ids_vertex_secondary",
            "particle_ids_particle",
            "particle_ids_generation",
            "particle_ids_sub_particle",
        ]
        needed = ["track_nr", "volume_id", "layer_id", "module_id"] + pid_cols
        data = tree.arrays(needed, library="ak")
        n_tracks = len(data)

        # Primary particle per state (first in list = highest-weight contributor)
        primary_pids = {c: ak.firsts(data[c], axis=2) for c in pid_cols}
        pid_encoded = (
            ak.values_astype(primary_pids["particle_ids_vertex_primary"], np.int64) * (1 << 48)
            + ak.values_astype(primary_pids["particle_ids_vertex_secondary"], np.int64) * (1 << 32)
            + ak.values_astype(primary_pids["particle_ids_particle"], np.int64) * (1 << 16)
            + ak.values_astype(primary_pids["particle_ids_generation"], np.int64) * (1 << 8)
            + ak.values_astype(primary_pids["particle_ids_sub_particle"], np.int64)
        )
        track_nr_broad, pid_flat = ak.broadcast_arrays(data["track_nr"], pid_encoded)
        _, vol_flat = ak.broadcast_arrays(data["track_nr"], data["volume_id"])
        _, lay_flat = ak.broadcast_arrays(data["track_nr"], data["layer_id"])
        _, mod_flat = ak.broadcast_arrays(data["track_nr"], data["module_id"])

        df = pd.DataFrame({
            "track_nr": ak.to_numpy(ak.flatten(track_nr_broad)),
            "pid": ak.to_numpy(ak.flatten(pid_flat)),
            "volume_id": ak.to_numpy(ak.flatten(vol_flat)),
            "layer_id": ak.to_numpy(ak.flatten(lay_flat)),
            "module_id": ak.to_numpy(ak.flatten(mod_flat)),
        })
        df["pid"] = df["pid"].fillna(0).astype(np.int64)
        print(f"[{label}] {n_tracks} tracks, {len(df)} flat states, "
              f"{(df['pid'] != 0).sum()} valid")
        return df

    def dm_match_with_shared_denom(df, particle_total, label):
        """Compute DM-matched particles using a shared completeness denominator."""
        valid = df[df["pid"] != 0].copy()
        n_meas_per_track = valid.groupby("track_nr").size()
        track_particle_counts = valid.groupby(["track_nr", "pid"]).size().reset_index(name="count")
        idx_max = track_particle_counts.groupby("track_nr")["count"].idxmax()
        majority = track_particle_counts.loc[idx_max].copy()
        majority = majority.rename(columns={"count": "majority_count", "pid": "majority_pid"})
        majority["n_meas"] = majority["track_nr"].map(n_meas_per_track)
        majority["purity"] = majority["majority_count"] / majority["n_meas"]
        majority["total_hits"] = majority["majority_pid"].map(particle_total)
        majority["completeness"] = majority["majority_count"] / majority["total_hits"]

        dm = majority[(majority["purity"] >= 0.5) & (majority["completeness"] >= 0.5)]
        matched = set(dm["majority_pid"].unique())

        all_tracks = df["track_nr"].nunique()
        stats = {
            "total": all_tracks,
            "dm_matched": len(dm),
            "low_purity": int((majority["purity"] < 0.5).sum()),
            "low_completeness": int(
                ((majority["purity"] >= 0.5) & (majority["completeness"] < 0.5)).sum()
            ),
        }
        print(f"[{label}] {stats}")
        print(f"[{label}] DM-matched unique particles: {len(matched)}")
        if len(majority) > 0:
            print(f"[{label}] Purity  — median: {majority['purity'].median():.3f}, "
                  f"mean: {majority['purity'].mean():.3f}")
            print(f"[{label}] Completeness — median: {majority['completeness'].median():.3f}, "
                  f"mean: {majority['completeness'].mean():.3f}")
        return matched, stats

    print("\n=== Loading track states ===")
    env_df = load_flat_states(env_root, "Envelope")
    tight_df = load_flat_states(tight_root, "Tight")

    # Shared completeness denominator: union of BOTH runs' unique (pid, vol, lay, mod)
    combined = pd.concat([
        env_df[env_df["pid"] != 0][["pid", "volume_id", "layer_id", "module_id"]],
        tight_df[tight_df["pid"] != 0][["pid", "volume_id", "layer_id", "module_id"]],
    ]).drop_duplicates()
    particle_total = combined.groupby("pid").size()
    print(f"\nShared particle_total: {len(particle_total)} unique particles")

    print("\n=== DM matching (shared denominator) ===")
    env_particles, env_stats = dm_match_with_shared_denom(env_df, particle_total, "Envelope")
    tight_particles, tight_stats = dm_match_with_shared_denom(tight_df, particle_total, "Tight")

    # Compare
    shared = tight_particles & env_particles
    tight_only = tight_particles - env_particles
    env_only = env_particles - tight_particles

    recovery_rate = len(shared) / len(tight_particles) * 100 if tight_particles else 0

    print(f"\n=== CHECK 2b: DM Track Recovery ===")
    print(f"Tight DM-matched particles:    {len(tight_particles)}")
    print(f"Envelope DM-matched particles: {len(env_particles)}")
    print(f"Shared (both match):           {len(shared)}")
    print(f"Tight-only (missing in env):   {len(tight_only)}")
    print(f"Envelope-only (extra in env):  {len(env_only)}")
    print(f"\nRecovery: {len(shared)}/{len(tight_particles)} = {recovery_rate:.2f}%")
    print(f"Pass criterion: 100% (envelope must recover all tight particles)")

    status = "PASS" if recovery_rate >= 100.0 else "FAIL"
    print(f"Result: {status}")

    if tight_only:
        print(f"\n=== Particles found by Tight but NOT Envelope ===")
        for pid in sorted(tight_only)[:10]:
            t_rows = tight_df[tight_df["pid"] == pid]
            e_rows = env_df[env_df["pid"] == pid]
            total = particle_total.get(pid, 0)
            t_best = t_rows.groupby("track_nr").size().max() if len(t_rows) > 0 else 0
            e_best = e_rows.groupby("track_nr").size().max() if len(e_rows) > 0 else 0
            print(f"  PID {pid}: total_hits={total}, tight_best={t_best}, "
                  f"env_best={e_best}, env_tracks_with_pid={e_rows['track_nr'].nunique()}")

    data_vol.commit()

    return {
        "status": status,
        "event_id": event_id,
        "tight_dm_particles": len(tight_particles),
        "envelope_dm_particles": len(env_particles),
        "shared": len(shared),
        "tight_only": len(tight_only),
        "envelope_only": len(env_only),
        "recovery_pct": round(recovery_rate, 2),
        "tight_tracks": tight_stats["total"],
        "tight_dm_tracks": tight_stats["dm_matched"],
        "envelope_tracks": env_stats["total"],
        "envelope_dm_tracks": env_stats["dm_matched"],
        "tight_wall_s": round(wall, 1),
    }


@app.function(
    image=image,
    volumes={BUILD_PATH: build_vol, DATA_PATH: data_vol},
    cpu=4,
    memory=65536,
    timeout=1200,
)
def analyze_seed_particle_conversion():
    """How often does a track's majority particle differ from its seed's majority?

    For each track, compares the seed majority particle (first 3 measurement
    states) to the full-track majority particle. Stratified by seed purity
    (3/3, 2/3, no majority). Reports conversion rates and DM match rates.
    """
    import glob
    import os

    import awkward as ak
    import numpy as np
    import pandas as pd
    import uproot

    data_vol.reload()

    # Use the envelope run (most representative of training data)
    env_dirs = sorted(glob.glob(f"{DATA_PATH}/results/verify_branches_*"))
    if not env_dirs:
        raise FileNotFoundError("No envelope run found")
    env_dir = env_dirs[-1]
    root_path = f"{env_dir}/trackstates_ckf.root"
    print(f"Analyzing: {root_path}")

    f = uproot.open(root_path)
    tree = f["trackstates"]

    pid_cols = [
        "particle_ids_vertex_primary",
        "particle_ids_vertex_secondary",
        "particle_ids_particle",
        "particle_ids_generation",
        "particle_ids_sub_particle",
    ]
    needed = ["track_nr", "volume_id", "layer_id", "module_id"] + pid_cols
    data = tree.arrays(needed, library="ak")
    n_tracks = len(data)
    print(f"Total tracks: {n_tracks}")

    # Encode primary particle per state as int64 barcode
    primary_pids = {c: ak.firsts(data[c], axis=2) for c in pid_cols}
    pid_encoded = (
        ak.values_astype(primary_pids["particle_ids_vertex_primary"], np.int64) * (1 << 48)
        + ak.values_astype(primary_pids["particle_ids_vertex_secondary"], np.int64) * (1 << 32)
        + ak.values_astype(primary_pids["particle_ids_particle"], np.int64) * (1 << 16)
        + ak.values_astype(primary_pids["particle_ids_generation"], np.int64) * (1 << 8)
        + ak.values_astype(primary_pids["particle_ids_sub_particle"], np.int64)
    )
    # Replace None (holes) with 0
    pid_encoded = ak.fill_none(pid_encoded, 0)

    # Flatten to a flat table for vectorized processing (include geometry for completeness)
    track_nr_broad, pid_flat = ak.broadcast_arrays(data["track_nr"], pid_encoded)
    vol_broad = ak.flatten(data["volume_id"])
    lay_broad = ak.flatten(data["layer_id"])
    mod_broad = ak.flatten(data["module_id"])
    flat_df = pd.DataFrame({
        "track_nr": ak.to_numpy(ak.flatten(track_nr_broad)),
        "pid": ak.to_numpy(ak.flatten(pid_flat)),
        "volume_id": ak.to_numpy(vol_broad),
        "layer_id": ak.to_numpy(lay_broad),
        "module_id": ak.to_numpy(mod_broad),
    })
    print(f"Flat states: {len(flat_df)}")

    # Keep only measurement states (pid != 0)
    meas = flat_df[flat_df["pid"] != 0].copy()
    print(f"Measurement states: {len(meas)}")

    # Assign a within-track measurement index (0, 1, 2, ...) for seed identification
    meas["meas_idx"] = meas.groupby("track_nr").cumcount()

    # --- Full-track majority particle ---
    track_particle_counts = meas.groupby(["track_nr", "pid"]).size().reset_index(name="count")
    idx_max = track_particle_counts.groupby("track_nr")["count"].idxmax()
    full_maj = track_particle_counts.loc[idx_max][["track_nr", "pid", "count"]].copy()
    full_maj = full_maj.rename(columns={"pid": "full_maj_pid", "count": "full_maj_count"})
    full_maj["n_meas"] = full_maj["track_nr"].map(meas.groupby("track_nr").size())
    full_maj["purity"] = full_maj["full_maj_count"] / full_maj["n_meas"]

    # --- Seed majority particle (first 3 measurement states) ---
    seed_states = meas[meas["meas_idx"] < 3].copy()
    seed_particle_counts = seed_states.groupby(["track_nr", "pid"]).size().reset_index(name="count")
    seed_idx_max = seed_particle_counts.groupby("track_nr")["count"].idxmax()
    seed_maj = seed_particle_counts.loc[seed_idx_max][["track_nr", "pid", "count"]].copy()
    seed_maj = seed_maj.rename(columns={"pid": "seed_maj_pid", "count": "seed_maj_count"})

    # --- Completeness: unique measurements per particle across all tracks ---
    # particle_total[pid] = number of unique (volume, layer, module) tuples for that pid
    unique_meas = meas[["pid", "volume_id", "layer_id", "module_id"]].drop_duplicates()
    particle_total = unique_meas.groupby("pid").size()
    print(f"Unique particles with measurements: {len(particle_total)}")

    # Per-track unique measurements for the majority particle
    # For each track, count unique (volume, layer, module) tuples matching the majority pid
    track_maj_meas = meas.merge(
        full_maj[["track_nr", "full_maj_pid"]],
        on="track_nr",
    )
    track_maj_meas = track_maj_meas[track_maj_meas["pid"] == track_maj_meas["full_maj_pid"]]
    track_unique = (
        track_maj_meas[["track_nr", "volume_id", "layer_id", "module_id"]]
        .drop_duplicates()
        .groupby("track_nr")
        .size()
        .rename("unique_maj_meas")
    )

    # Join seed and full-track info
    df = full_maj.merge(seed_maj, on="track_nr", how="inner")
    df = df.merge(track_unique, on="track_nr", how="left")
    # Only keep tracks with ≥ 3 measurements
    df = df[df["n_meas"] >= 3].copy()
    df["converted"] = df["seed_maj_pid"] != df["full_maj_pid"]
    df["completeness"] = df["unique_maj_meas"] / df["full_maj_pid"].map(particle_total)
    df["dm_matched"] = (df["purity"] >= 0.5) & (df["completeness"] >= 0.5)
    print(f"\nTracks with ≥3 measurements: {len(df)}")
    print(f"DM matched tracks: {df['dm_matched'].sum()}")

    # Stratify by seed purity
    df["seed_category"] = df["seed_maj_count"].map({
        3: "3/3 (pure seed)",
        2: "2/3 (one outlier)",
        1: "1/3 (no majority)",
    })

    print("\n" + "=" * 70)
    print("SEED-TO-TRACK PARTICLE CONVERSION ANALYSIS")
    print("=" * 70)

    for cat in ["3/3 (pure seed)", "2/3 (one outlier)", "1/3 (no majority)"]:
        subset = df[df["seed_category"] == cat]
        n = len(subset)
        if n == 0:
            continue

        n_converted = subset["converted"].sum()
        n_pure_track = (subset["purity"] >= 0.5).sum()

        # Of those with purity >= 0.5, how many converted?
        pure_subset = subset[subset["purity"] >= 0.5]
        n_pure_converted = pure_subset["converted"].sum() if len(pure_subset) > 0 else 0

        # DM matched
        n_dm = subset["dm_matched"].sum()
        dm_subset = subset[subset["dm_matched"]]
        n_dm_converted = dm_subset["converted"].sum() if len(dm_subset) > 0 else 0

        # Among converted tracks: how many have purity >= 0.5, and of those, how many DM?
        conv_subset = subset[subset["converted"]]
        n_conv_pure = (conv_subset["purity"] >= 0.5).sum() if len(conv_subset) > 0 else 0
        conv_pure_subset = conv_subset[conv_subset["purity"] >= 0.5]
        n_conv_pure_dm = conv_pure_subset["dm_matched"].sum() if len(conv_pure_subset) > 0 else 0

        print(f"\n--- {cat}: {n} tracks, {n_converted} converted ---")
        print(f"  Converted with purity ≥ 0.5: "
              f"{n_conv_pure}/{n_converted} ({100*n_conv_pure/n_converted:.2f}%)" if n_converted > 0 else "  No conversions")
        if n_conv_pure > 0:
            print(f"  Of those, DM matched: "
                  f"{n_conv_pure_dm}/{n_conv_pure} ({100*n_conv_pure_dm/n_conv_pure:.2f}%)")

    # Overall summary
    print(f"\n--- OVERALL ---")
    n_converted_total = df["converted"].sum()
    print(f"Total tracks (≥3 meas): {len(df)}")
    print(f"Total converted: {n_converted_total} ({100*n_converted_total/len(df):.2f}%)")

    conv_all = df[df["converted"]]
    n_conv_all = len(conv_all)
    n_conv_pure_all = (conv_all["purity"] >= 0.5).sum()
    conv_pure_all = conv_all[conv_all["purity"] >= 0.5]
    n_conv_pure_dm_all = conv_pure_all["dm_matched"].sum()
    print(f"\n--- OVERALL ---")
    print(f"Total converted: {n_conv_all}")
    print(f"Converted with purity ≥ 0.5: {n_conv_pure_all}/{n_conv_all} ({100*n_conv_pure_all/n_conv_all:.2f}%)")
    print(f"Of those, DM matched: {n_conv_pure_dm_all}/{n_conv_pure_all} ({100*n_conv_pure_dm_all/n_conv_pure_all:.2f}%)" if n_conv_pure_all > 0 else "None with purity >= 0.5")

    dm_df = df[df["dm_matched"]]
    return {
        "total_tracks_ge3": len(df),
        "seed_3_3": int((df["seed_maj_count"] == 3).sum()),
        "seed_2_3": int((df["seed_maj_count"] == 2).sum()),
        "seed_1_3": int((df["seed_maj_count"] == 1).sum()),
        "converted_total": int(n_conv_all),
        "converted_pct": round(100 * n_conv_all / len(df), 2),
        "dm_tracks": int(len(dm_df)),
        "dm_converted": int(dm_df["converted"].sum()),
    }


@app.function(
    image=image,
    volumes={BUILD_PATH: build_vol, DATA_PATH: data_vol},
    cpu=4,
    memory=65536,
    timeout=1200,
)
def window_failure_rate():
    """Check 4: For measurement states where the majority particle contributed,
    what fraction have truth position outside W_k(n)?

    Produces data for two plots:
    1. Overall failure rate vs n (n = 1..15)
    2. Failure rate vs eta at fixed n (for n = 2,3,4,5,6,7,8,9,10)
    """
    import glob
    import json
    import os

    import awkward as ak
    import numpy as np
    import pandas as pd
    import uproot

    data_vol.reload()

    env_dirs = sorted(glob.glob(f"{DATA_PATH}/results/verify_branches_*"))
    if not env_dirs:
        raise FileNotFoundError("No envelope run found")
    env_dir = env_dirs[-1]
    root_path = f"{env_dir}/trackstates_ckf.root"
    print(f"Analyzing: {root_path}")

    f = uproot.open(root_path)
    tree = f["trackstates"]

    # Load all needed branches
    pid_cols = [
        "particle_ids_vertex_primary", "particle_ids_vertex_secondary",
        "particle_ids_particle", "particle_ids_generation",
        "particle_ids_sub_particle",
    ]
    branches = [
        "track_nr", "volume_id", "layer_id", "module_id",
        "eLOC0_prt", "eLOC1_prt",
        "t_eLOC0", "t_eLOC1",
        "eTHETA_prt",
    ] + pid_cols

    # Check for innovation covariance branches
    all_keys = tree.keys()
    has_S = "S00" in all_keys
    if has_S:
        branches += ["S00", "S01", "S11"]
        print("Innovation covariance S00/S01/S11 found")
    else:
        # Fall back to predicted covariance diagonal
        branches += ["err_eLOC0_prt", "err_eLOC1_prt"]
        print("WARNING: S00/S01/S11 not found, using predicted errors as proxy")

    has_pathInX0 = "pathInX0" in all_keys
    if has_pathInX0:
        branches.append("pathInX0")
        print("pathInX0 found")

    data = tree.arrays(branches, library="ak")
    n_tracks = len(data)
    print(f"Total tracks: {n_tracks}")

    # Encode primary particle per state
    primary_pids = {c: ak.firsts(data[c], axis=2) for c in pid_cols}
    pid_encoded = (
        ak.values_astype(primary_pids["particle_ids_vertex_primary"], np.int64) * (1 << 48)
        + ak.values_astype(primary_pids["particle_ids_vertex_secondary"], np.int64) * (1 << 32)
        + ak.values_astype(primary_pids["particle_ids_particle"], np.int64) * (1 << 16)
        + ak.values_astype(primary_pids["particle_ids_generation"], np.int64) * (1 << 8)
        + ak.values_astype(primary_pids["particle_ids_sub_particle"], np.int64)
    )
    pid_encoded = ak.fill_none(pid_encoded, 0)

    # Flatten everything to a flat table
    track_nr_b, pid_flat = ak.broadcast_arrays(data["track_nr"], pid_encoded)
    cols = {
        "track_nr": ak.to_numpy(ak.flatten(track_nr_b)),
        "pid": ak.to_numpy(ak.flatten(pid_flat)),
        "pred_l0": ak.to_numpy(ak.flatten(data["eLOC0_prt"])),
        "pred_l1": ak.to_numpy(ak.flatten(data["eLOC1_prt"])),
        "truth_l0": ak.to_numpy(ak.flatten(data["t_eLOC0"])),
        "truth_l1": ak.to_numpy(ak.flatten(data["t_eLOC1"])),
        "theta_prt": ak.to_numpy(ak.flatten(data["eTHETA_prt"])),
        "volume_id": ak.to_numpy(ak.flatten(data["volume_id"])),
        "layer_id": ak.to_numpy(ak.flatten(data["layer_id"])),
    }
    if has_S:
        cols["S00"] = ak.to_numpy(ak.flatten(data["S00"]))
        cols["S01"] = ak.to_numpy(ak.flatten(data["S01"]))
        cols["S11"] = ak.to_numpy(ak.flatten(data["S11"]))
    else:
        cols["S00"] = ak.to_numpy(ak.flatten(data["err_eLOC0_prt"])) ** 2
        cols["S11"] = ak.to_numpy(ak.flatten(data["err_eLOC1_prt"])) ** 2
    if has_pathInX0:
        cols["pathInX0"] = ak.to_numpy(ak.flatten(data["pathInX0"]))

    df = pd.DataFrame(cols)
    print(f"Flat states: {len(df)}")

    # Keep only measurement states (pid != 0, truth not NaN)
    meas = df[(df["pid"] != 0) & df["truth_l0"].notna() & df["truth_l1"].notna()].copy()
    print(f"Measurement states with truth: {len(meas)}")

    # Compute branch majority particle
    meas["meas_idx"] = meas.groupby("track_nr").cumcount()
    tpc = meas.groupby(["track_nr", "pid"]).size().reset_index(name="count")
    idx_max = tpc.groupby("track_nr")["count"].idxmax()
    maj_map = tpc.loc[idx_max].set_index("track_nr")["pid"]
    meas["maj_pid"] = meas["track_nr"].map(maj_map)

    # Filter to states where the selected hit IS from the majority particle
    maj_states = meas[meas["pid"] == meas["maj_pid"]].copy()
    print(f"States where selected hit = majority particle: {len(maj_states)}")

    # Compute residuals (truth - predicted)
    maj_states["r0"] = maj_states["truth_l0"] - maj_states["pred_l0"]
    maj_states["r1"] = maj_states["truth_l1"] - maj_states["pred_l1"]

    # Window half-widths
    maj_states["sqrt_S00"] = np.sqrt(np.maximum(maj_states["S00"], 0))
    maj_states["sqrt_S11"] = np.sqrt(np.maximum(maj_states["S11"], 0))

    # Compute eta from theta
    theta = maj_states["theta_prt"].values
    eta = -np.log(np.tan(theta / 2.0))
    maj_states["eta"] = eta

    # --- Plot 1: Overall failure rate vs n ---
    n_values = np.arange(1, 16, dtype=float)
    failure_rates = []
    total_states = len(maj_states)

    for n in n_values:
        outside = (
            (np.abs(maj_states["r0"]) > n * maj_states["sqrt_S00"]) |
            (np.abs(maj_states["r1"]) > n * maj_states["sqrt_S11"])
        )
        rate = outside.sum() / total_states
        failure_rates.append(rate)
        print(f"  n={n:5.1f}: {outside.sum():8d} / {total_states} = {100*rate:.4f}%")

    # --- Plot 2: Failure rate vs eta at fixed n ---
    eta_bins = np.linspace(-4, 4, 41)  # 0.2-wide bins
    eta_centers = 0.5 * (eta_bins[:-1] + eta_bins[1:])
    n_curves = [2, 3, 4, 5, 6, 7, 8, 9, 10]

    eta_failure = {}
    for n in n_curves:
        outside = (
            (np.abs(maj_states["r0"]) > n * maj_states["sqrt_S00"]) |
            (np.abs(maj_states["r1"]) > n * maj_states["sqrt_S11"])
        )
        # Bin by eta
        bin_idx = np.digitize(maj_states["eta"].values, eta_bins) - 1
        rates = []
        for i in range(len(eta_centers)):
            mask = bin_idx == i
            n_in_bin = mask.sum()
            if n_in_bin > 0:
                rates.append(float(outside[mask].sum()) / n_in_bin)
            else:
                rates.append(np.nan)
        eta_failure[f"n={n}"] = rates

    # --- Also stratify by layer ---
    print("\n=== Failure rate at n=10, by volume ===")
    n = 10.0
    outside_10 = (
        (np.abs(maj_states["r0"]) > n * maj_states["sqrt_S00"]) |
        (np.abs(maj_states["r1"]) > n * maj_states["sqrt_S11"])
    )
    for vid in sorted(maj_states["volume_id"].unique()):
        vmask = maj_states["volume_id"] == vid
        n_vol = vmask.sum()
        n_fail = (outside_10 & vmask).sum()
        print(f"  Volume {vid}: {n_fail}/{n_vol} = {100*n_fail/n_vol:.3f}%")

    # Save results as JSON for plotting
    # List all files in the output directory
    print("\n=== Files in output directory ===")
    for fname in sorted(os.listdir(env_dir)):
        fpath = os.path.join(env_dir, fname)
        sz = os.path.getsize(fpath)
        print(f"  {fname}: {sz/1024:.0f} KB")

    results = {
        "n_values": n_values.tolist(),
        "failure_rates": failure_rates,
        "eta_centers": eta_centers.tolist(),
        "eta_failure": eta_failure,
        "n_curves": n_curves,
        "total_maj_states": total_states,
    }
    return results


@app.function(
    image=image,
    volumes={BUILD_PATH: build_vol, DATA_PATH: data_vol},
    cpu=8,
    memory=131072,
    timeout=3600,
)
def run_pilot_stage1(events: int = 2, skip: int = 0):
    """Stage 1: Run envelope CKF with full CSV+ROOT outputs for pilot data.

    Produces: trackstates_ckf.root, tracksummary_ckf.root,
    measurements CSV, cells CSV, measurement-simhit-map CSV,
    simhits CSV, seeds CSV, detectors.csv (tracking geometry).
    """
    import os
    import time
    import types
    from pathlib import Path

    import yaml

    for vol in (build_vol, data_vol):
        try:
            vol.reload()
        except RuntimeError:
            pass
    _setup_acts_env(use_patched=True)

    material_map = f"{DATA_PATH}/odd-material-maps.root"
    odd_data_dir = f"{ODD_INSTALL}/data"
    os.makedirs(odd_data_dir, exist_ok=True)
    map_link = f"{odd_data_dir}/odd-material-maps.root"
    if not os.path.exists(map_link):
        os.symlink(material_map, map_link)

    with open("/app/configs/envelope.yaml") as yf:
        cfg_dict = yaml.safe_load(yf)

    cfg_dict.update({
        "events": events,
        "skip": skip,
        "threads": 8,
        "ckf": True,
        "ambi": False,
        "output_digi_csv": True,
        "output_simhits_csv": True,
        "output_seeds_csv": True,
        "output_detectors_csv": True,
        "write_track_states": True,
        "write_predicted_cov": True,
        "write_track_summary": True,
        "ckf_finding_performance": False,
        "ambi_finding_performance": False,
    })
    cfg = types.SimpleNamespace(**cfg_dict)
    cfg.seed = 42

    out_dir = Path(f"{DATA_PATH}/results/pilot_{int(time.time())}")
    out_dir.mkdir(parents=True, exist_ok=True)

    import acts, acts.examples
    from digi_and_reco import setup_acts_reconstruction
    rnd = acts.examples.RandomNumbers(seed=cfg.seed)

    print(f"=== Pilot Stage 1: {events} events, skip={skip} ===")
    print(f"Output: {out_dir}")
    t0 = time.time()
    s = setup_acts_reconstruction(
        Path(f"{DATA_PATH}/events/edm4hep.root"),
        out_dir, cfg, rnd,
    )
    s.run()
    wall = time.time() - t0
    print(f"Pipeline completed in {wall:.1f}s")

    print("\n=== Output files ===")
    for fname in sorted(os.listdir(out_dir)):
        fpath = os.path.join(out_dir, fname)
        sz = os.path.getsize(fpath)
        print(f"  {fname}: {sz / 1024:.1f} KB")

    import uproot
    ts_path = str(out_dir / "trackstates_ckf.root")
    if os.path.exists(ts_path):
        with uproot.open(ts_path) as rf:
            tree = rf["trackstates"]
            all_keys = sorted(tree.keys())
            print(f"\nTrackstates: {tree.num_entries} tracks, {len(all_keys)} branches")
            expected = [
                "S00_prt", "S01_prt", "S11_prt", "pathInX0_interval",
                "clus_size_u", "clus_size_v", "clus_qtot",
                "clus_sigma_uu", "clus_sigma_uv", "clus_sigma_vv",
                "alpha_u", "alpha_v",
            ]
            for b in expected:
                status = "PASS" if b in all_keys else "FAIL"
                print(f"  {status}  {b}")

    csv_checks = {
        "measurements": "event*-measurements.csv",
        "cells": "event*-cells.csv",
        "meas-simhit-map": "event*-measurement-simhit-map.csv",
        "simhits": "event*-simhits.csv",
        "detectors": "detectors.csv",
    }
    for label, pattern in csv_checks.items():
        found = list(out_dir.glob(pattern))
        if found:
            print(f"  PASS  {label}: {len(found)} file(s)")
        else:
            print(f"  FAIL  {label}: not found (pattern: {pattern})")

    data_vol.commit()
    print(f"\nPilot stage 1 complete. Output at {out_dir}")
    return {"output_dir": str(out_dir), "wall_time": wall}


@app.function(
    image=image,
    volumes={BUILD_PATH: build_vol, DATA_PATH: data_vol},
    cpu=8,
    memory=65536,
    timeout=1800,
)
def tight_ambi_conversion():
    """Run tight CKF WITH ambi resolution, then check which converted
    DM-matched tracks survive ambiguity resolution."""
    import glob
    import os
    import time
    import types
    from pathlib import Path

    import awkward as ak
    import numpy as np
    import pandas as pd
    import uproot
    import yaml

    build_vol.reload()
    data_vol.reload()
    _setup_acts_env(use_patched=True)

    material_map = f"{DATA_PATH}/odd-material-maps.root"
    odd_data_dir = f"{ODD_INSTALL}/data"
    os.makedirs(odd_data_dir, exist_ok=True)
    map_link = f"{odd_data_dir}/odd-material-maps.root"
    if not os.path.exists(map_link):
        os.symlink(material_map, map_link)

    # --- Run tight CKF with ambi enabled ---
    with open("/app/configs/tight_t79.yaml") as yf:
        tight_dict = yaml.safe_load(yf)

    tight_dict.update({
        "events": 1, "skip": 0, "threads": 8,
        "ckf": True, "ambi": True, "ambi_solver": "greedy",
        "output_seeds_csv": False, "output_digi_csv": False,
        "output_simhits_csv": False,
        "write_track_states": True, "write_predicted_cov": False,
        "ckf_finding_performance": False,
        "ambi_finding_performance": False,
        "ambi_root_output": True,
        "write_track_summary": True,
    })
    cfg = types.SimpleNamespace(**tight_dict)
    cfg.seed = 42

    out_dir = Path(f"{DATA_PATH}/results/tight_ambi_{int(time.time())}")
    out_dir.mkdir(parents=True, exist_ok=True)

    import acts, acts.examples
    from digi_and_reco import setup_acts_reconstruction
    rnd = acts.examples.RandomNumbers(seed=cfg.seed)

    print("=== Running tight CKF + ambi resolution ===")
    t0 = time.time()
    s = setup_acts_reconstruction(
        Path(f"{DATA_PATH}/events/edm4hep.root"),
        out_dir, cfg, rnd,
    )
    s.run()
    print(f"Pipeline completed in {time.time() - t0:.1f}s")

    # List output files
    for f_name in sorted(os.listdir(out_dir)):
        fpath = out_dir / f_name
        sz = os.path.getsize(fpath)
        print(f"  {f_name}: {sz/1024:.0f} KB")

    # --- Load pre-ambi trackstates (CKF candidates) ---
    pre_ambi_path = str(out_dir / "trackstates_ckf.root")
    print(f"\nPre-ambi: {pre_ambi_path}")

    f = uproot.open(pre_ambi_path)
    tree = f["trackstates"]
    pid_cols = [
        "particle_ids_vertex_primary", "particle_ids_vertex_secondary",
        "particle_ids_particle", "particle_ids_generation",
        "particle_ids_sub_particle",
    ]
    needed = ["track_nr", "volume_id", "layer_id", "module_id"] + pid_cols
    data = tree.arrays(needed, library="ak")
    print(f"Pre-ambi tracks: {len(data)}")

    # Encode primary particle per state
    primary_pids = {c: ak.firsts(data[c], axis=2) for c in pid_cols}
    pid_encoded = (
        ak.values_astype(primary_pids["particle_ids_vertex_primary"], np.int64) * (1 << 48)
        + ak.values_astype(primary_pids["particle_ids_vertex_secondary"], np.int64) * (1 << 32)
        + ak.values_astype(primary_pids["particle_ids_particle"], np.int64) * (1 << 16)
        + ak.values_astype(primary_pids["particle_ids_generation"], np.int64) * (1 << 8)
        + ak.values_astype(primary_pids["particle_ids_sub_particle"], np.int64)
    )
    pid_encoded = ak.fill_none(pid_encoded, 0)

    # Flatten
    track_nr_broad, pid_flat = ak.broadcast_arrays(data["track_nr"], pid_encoded)
    flat_df = pd.DataFrame({
        "track_nr": ak.to_numpy(ak.flatten(track_nr_broad)),
        "pid": ak.to_numpy(ak.flatten(pid_flat)),
        "volume_id": ak.to_numpy(ak.flatten(data["volume_id"])),
        "layer_id": ak.to_numpy(ak.flatten(data["layer_id"])),
        "module_id": ak.to_numpy(ak.flatten(data["module_id"])),
    })

    meas = flat_df[flat_df["pid"] != 0].copy()
    meas["meas_idx"] = meas.groupby("track_nr").cumcount()

    # Full-track majority
    tpc = meas.groupby(["track_nr", "pid"]).size().reset_index(name="count")
    idx_max = tpc.groupby("track_nr")["count"].idxmax()
    full_maj = tpc.loc[idx_max][["track_nr", "pid", "count"]].copy()
    full_maj = full_maj.rename(columns={"pid": "full_maj_pid", "count": "full_maj_count"})
    full_maj["n_meas"] = full_maj["track_nr"].map(meas.groupby("track_nr").size())
    full_maj["purity"] = full_maj["full_maj_count"] / full_maj["n_meas"]

    # Seed majority (first 3 measurement states)
    seed_states = meas[meas["meas_idx"] < 3]
    spc = seed_states.groupby(["track_nr", "pid"]).size().reset_index(name="count")
    seed_idx_max = spc.groupby("track_nr")["count"].idxmax()
    seed_maj = spc.loc[seed_idx_max][["track_nr", "pid", "count"]].copy()
    seed_maj = seed_maj.rename(columns={"pid": "seed_maj_pid", "count": "seed_maj_count"})

    # Completeness
    unique_meas = meas[["pid", "volume_id", "layer_id", "module_id"]].drop_duplicates()
    particle_total = unique_meas.groupby("pid").size()

    track_maj_meas = meas.merge(full_maj[["track_nr", "full_maj_pid"]], on="track_nr")
    track_maj_meas = track_maj_meas[track_maj_meas["pid"] == track_maj_meas["full_maj_pid"]]
    track_unique = (
        track_maj_meas[["track_nr", "volume_id", "layer_id", "module_id"]]
        .drop_duplicates()
        .groupby("track_nr").size()
        .rename("unique_maj_meas")
    )

    df = full_maj.merge(seed_maj, on="track_nr", how="inner")
    df = df.merge(track_unique, on="track_nr", how="left")
    df = df[df["n_meas"] >= 3].copy()
    df["converted"] = df["seed_maj_pid"] != df["full_maj_pid"]
    df["completeness"] = df["unique_maj_meas"] / df["full_maj_pid"].map(particle_total)
    df["dm_matched"] = (df["purity"] >= 0.5) & (df["completeness"] >= 0.5)

    # --- Load post-ambi track list ---
    # Find the ambi tracksummary or trackstates file
    ambi_files = [f_name for f_name in os.listdir(out_dir) if "ambi" in f_name and f_name.endswith(".root")]
    print(f"\nAmbi output files: {ambi_files}")

    # The tracksummary has one entry per surviving track with track_nr
    summary_path = None
    for candidate in ["tracksummary_ambi.root", "tracksummary_ambi_scorebased.root"]:
        p = out_dir / candidate
        if p.exists():
            summary_path = str(p)
            break

    if summary_path is None:
        # Try any ambi root file that has a tracksummary tree
        for af in ambi_files:
            try:
                tf = uproot.open(str(out_dir / af))
                if "tracksummary" in tf:
                    summary_path = str(out_dir / af)
                    break
            except Exception:
                continue

    if summary_path is None:
        print("ERROR: No post-ambi tracksummary found!")
        print("Available files:", os.listdir(out_dir))
        return {"error": "no ambi output"}

    print(f"Post-ambi summary: {summary_path}")
    ambi_f = uproot.open(summary_path)
    print(f"  Trees: {list(ambi_f.keys())}")
    ambi_tree = ambi_f["tracksummary"]
    print(f"  Branches: {ambi_tree.keys()[:10]}...")
    ambi_arr = ambi_tree["track_nr"].array(library="ak")
    print(f"  track_nr type: {ambi_arr.type}")
    ambi_flat = ak.to_numpy(ak.flatten(ambi_arr)) if ambi_arr.ndim > 1 else ak.to_numpy(ambi_arr)
    ambi_track_nrs = set(ambi_flat.tolist())
    print(f"Post-ambi tracks: {len(ambi_track_nrs)}")

    df["survived_ambi"] = df["track_nr"].isin(ambi_track_nrs)

    # --- Report ---
    print("\n" + "=" * 70)
    print("CONVERTED DM-MATCHED TRACKS vs AMBIGUITY RESOLUTION (tight config)")
    print("=" * 70)

    for cat in ["3/3 (pure seed)", "2/3 (one outlier)", "1/3 (no majority)"]:
        df["seed_category"] = df["seed_maj_count"].map({
            3: "3/3 (pure seed)", 2: "2/3 (one outlier)", 1: "1/3 (no majority)",
        })
        subset = df[df["seed_category"] == cat]
        n = len(subset)
        if n == 0:
            continue

        conv = subset[subset["converted"]]
        conv_pure = conv[conv["purity"] >= 0.5]
        conv_dm = conv[conv["dm_matched"]]
        conv_dm_surv = conv_dm[conv_dm["survived_ambi"]]

        print(f"\n--- {cat}: {n} tracks ---")
        print(f"  Converted: {len(conv)}/{n}")
        print(f"  Converted + purity ≥ 0.5: {len(conv_pure)}/{len(conv)}")
        print(f"  Converted + DM matched: {len(conv_dm)}/{len(conv)}")
        print(f"  Converted + DM matched + survived ambi: {len(conv_dm_surv)}/{len(conv_dm)}")

    # Overall
    conv_all = df[df["converted"]]
    conv_dm_all = conv_all[conv_all["dm_matched"]]
    conv_dm_surv_all = conv_dm_all[conv_dm_all["survived_ambi"]]
    print(f"\n--- OVERALL ---")
    print(f"  Total tracks: {len(df)}")
    print(f"  DM matched: {df['dm_matched'].sum()}")
    print(f"  DM matched + survived ambi: {df[df['dm_matched'] & df['survived_ambi']].shape[0]}")
    print(f"  Converted + DM matched: {len(conv_dm_all)}")
    print(f"  Converted + DM matched + survived ambi: {len(conv_dm_surv_all)}")

    return {
        "pre_ambi_tracks": len(df),
        "post_ambi_tracks": len(ambi_track_nrs),
        "dm_matched": int(df["dm_matched"].sum()),
        "converted_dm": int(len(conv_dm_all)),
        "converted_dm_survived": int(len(conv_dm_surv_all)),
    }


@app.function(
    image=image,
    volumes={BUILD_PATH: build_vol, DATA_PATH: data_vol},
    cpu=2,
    memory=8192,
    timeout=600,
)
def check_normal_conventions():
    """Check whether ODD module normals point radially outward."""
    import os

    import numpy as np

    build_vol.reload()
    _setup_acts_env(use_patched=True)

    material_map = f"{DATA_PATH}/odd-material-maps.root"
    odd_data_dir = f"{ODD_INSTALL}/data"
    os.makedirs(odd_data_dir, exist_ok=True)
    map_link = f"{odd_data_dir}/odd-material-maps.root"
    if not os.path.exists(map_link):
        os.symlink(material_map, map_link)

    import acts
    from acts.examples.odd import getOpenDataDetector, getOpenDataDetectorDirectory

    geoDir = getOpenDataDetectorDirectory()
    oddMaterialMap = f"{ODD_INSTALL}/data/odd-material-maps.root"
    oddMaterialDeco = acts.IMaterialDecorator.fromFile(oddMaterialMap)
    detector = getOpenDataDetector(odd_dir=geoDir, materialDecorator=oddMaterialDeco)
    trackingGeo = detector.trackingGeometry()
    gctx = acts.GeometryContext.dangerouslyDefaultConstruct()

    # ODD volume IDs:
    # Pixel barrel: vol 16, Pixel endcap neg: vol 14, Pixel endcap pos: vol 18
    # Short strip barrel: vol 23, Long strip barrel: vol 25
    # Short strip endcap neg: vol 22, pos: vol 24

    volume_names = {
        16: "Pixel Barrel",
        14: "Pixel Endcap Neg",
        18: "Pixel Endcap Pos",
        23: "Short Strip Barrel",
        22: "Short Strip EC Neg",
        24: "Short Strip EC Pos",
    }

    results = {}
    for vol_id, vol_name in volume_names.items():
        normals_info = []
        # Walk tracking geometry to find surfaces
        def visitor(surface):
            geo_id = surface.geometryId
            if geo_id.volume != vol_id:
                return
            center = surface.center(gctx)
            normal = surface.normal(gctx, center, acts.Vector3(0, 0, 1))
            # Radial direction from beam axis
            r_vec = np.array([center[0], center[1], 0.0])
            r_mag = np.linalg.norm(r_vec)
            if r_mag < 1e-6:
                return
            r_hat = r_vec / r_mag
            n_vec = np.array([normal[0], normal[1], normal[2]])
            # Dot product: positive = outward, negative = inward
            radial_dot = np.dot(n_vec, r_hat)
            # Z component of normal (relevant for endcaps)
            z_center = center[2]
            normals_info.append({
                "layer": geo_id.layer,
                "module": geo_id.sensitive,
                "center": (round(center[0], 1), round(center[1], 1), round(center[2], 1)),
                "r": round(r_mag, 1),
                "normal": (round(n_vec[0], 4), round(n_vec[1], 4), round(n_vec[2], 4)),
                "n_dot_r_hat": round(radial_dot, 4),
                "n_z": round(n_vec[2], 4),
                "z_center": round(z_center, 1),
            })

        trackingGeo.visitSurfaces(visitor)

        if normals_info:
            dots = [x["n_dot_r_hat"] for x in normals_info]
            n_zs = [x["n_z"] for x in normals_info]
            results[vol_name] = {
                "n_surfaces": len(normals_info),
                "n_dot_r_hat": {
                    "min": round(min(dots), 4),
                    "max": round(max(dots), 4),
                    "mean": round(np.mean(dots), 4),
                    "all_positive": all(d > 0 for d in dots),
                    "all_negative": all(d < 0 for d in dots),
                },
                "n_z": {
                    "min": round(min(n_zs), 4),
                    "max": round(max(n_zs), 4),
                    "mean": round(np.mean(n_zs), 4),
                },
                "examples": normals_info[:3],
            }

    print("\n=== Normal Convention Check ===\n")
    for name, info in results.items():
        dot_info = info["n_dot_r_hat"]
        nz_info = info["n_z"]
        print(f"{name} ({info['n_surfaces']} surfaces):")
        print(f"  n . r_hat:  min={dot_info['min']:+.4f}  max={dot_info['max']:+.4f}  mean={dot_info['mean']:+.4f}  all_outward={dot_info['all_positive']}")
        print(f"  n_z:        min={nz_info['min']:+.4f}  max={nz_info['max']:+.4f}  mean={nz_info['mean']:+.4f}")
        for ex in info["examples"]:
            print(f"    L{ex['layer']}/M{ex['module']}  center={ex['center']}  r={ex['r']}  normal={ex['normal']}  n.r={ex['n_dot_r_hat']:+.4f}")
        print()

    return results


@app.function(
    image=image,
    volumes={DATA_PATH: data_vol},
    cpu=2,
    memory=16384,
    timeout=600,
)
def dump_schema_samples():
    """Extract sample rows from each pilot output file for schema documentation."""
    import json
    import os

    import numpy as np
    import pandas as pd
    import uproot
    import awkward as ak

    data_vol.reload()
    pilot_dir = f"{DATA_PATH}/results/pilot_1786472672"

    samples = {}

    # 1. trackstates_ckf.root — sample 1 track with ~10 states
    ts_path = f"{pilot_dir}/trackstates_ckf.root"
    with uproot.open(ts_path) as f:
        tree = f["trackstates"]
        all_keys = sorted(tree.keys())
        samples["trackstates_branches"] = all_keys
        samples["trackstates_num_entries"] = tree.num_entries

        # Read a few tracks to find one with ~10 states
        data = tree.arrays(all_keys, entry_start=100, entry_stop=105, library="ak")
        n_states = ak.num(data["volume_id"], axis=1)

        # Pick track with most states
        best = ak.argmax(n_states)
        best_idx = int(best)

        track_sample = {}
        for key in all_keys:
            arr = data[key][best_idx]
            try:
                vals = ak.to_list(arr)
            except Exception:
                vals = arr
            # For jagged arrays, take first 3 states
            if isinstance(vals, list):
                if len(vals) > 0 and isinstance(vals[0], list):
                    # doubly jagged
                    track_sample[key] = vals[:3]
                else:
                    track_sample[key] = vals[:3]
            else:
                track_sample[key] = vals
        samples["trackstates_sample"] = track_sample

    # 2. measurements.csv
    meas_path = f"{pilot_dir}/event000000000-measurements.csv"
    meas = pd.read_csv(meas_path, nrows=5)
    samples["measurements_columns"] = list(meas.columns)
    samples["measurements_sample"] = meas.to_dict(orient="records")
    samples["measurements_total_rows"] = sum(1 for _ in open(meas_path)) - 1

    # 3. cells.csv
    cells_path = f"{pilot_dir}/event000000000-cells.csv"
    cells = pd.read_csv(cells_path, nrows=10)
    samples["cells_columns"] = list(cells.columns)
    samples["cells_sample"] = cells.to_dict(orient="records")
    samples["cells_total_rows"] = sum(1 for _ in open(cells_path)) - 1

    # 4. measurement-simhit-map.csv
    map_path = f"{pilot_dir}/event000000000-measurement-simhit-map.csv"
    msm = pd.read_csv(map_path, nrows=5)
    samples["meas_simhit_map_columns"] = list(msm.columns)
    samples["meas_simhit_map_sample"] = msm.to_dict(orient="records")
    samples["meas_simhit_map_total_rows"] = sum(1 for _ in open(map_path)) - 1

    # 5. simhits.csv
    sh_path = f"{pilot_dir}/event000000000-simhits.csv"
    sh = pd.read_csv(sh_path, nrows=5)
    samples["simhits_columns"] = list(sh.columns)
    samples["simhits_sample"] = sh.to_dict(orient="records")
    samples["simhits_total_rows"] = sum(1 for _ in open(sh_path)) - 1

    # 6. detectors.csv
    det_path = f"{pilot_dir}/detectors.csv"
    det = pd.read_csv(det_path, nrows=5)
    samples["detectors_columns"] = list(det.columns)
    samples["detectors_sample"] = det.to_dict(orient="records")
    samples["detectors_total_rows"] = sum(1 for _ in open(det_path)) - 1

    # Print as JSON for extraction
    class NpEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return round(float(obj), 6)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, float) and np.isnan(obj):
                return "NaN"
            return super().default(obj)

    print(json.dumps(samples, cls=NpEncoder, indent=2))
    return samples


@app.function(
    image=image,
    volumes={BUILD_PATH: build_vol, DATA_PATH: data_vol},
    timeout=600,
)
def run_window_failure_analysis():
    """Re-run analysis with material columns, produce material plots."""
    import os
    import sys

    sys.path.insert(0, "/app")
    import numpy as np
    import pandas as pd
    from window_failure import plot_failure_vs_eta, run_analysis

    data_vol.reload()
    pilot_dir = f"{DATA_PATH}/results/pilot_1786472672"
    root_path = f"{pilot_dir}/trackstates_ckf.root"
    csv_dir = pilot_dir
    plot_dir = f"{pilot_dir}/plots"
    os.makedirs(plot_dir, exist_ok=True)

    for label, min_ag, suffix in [("majority", 2, ""), ("pure", 3, "_pure")]:
        dist = run_analysis(root_path, csv_dir, event_ids=[0, 1],
                            min_agreement=min_ag)
        dist.to_parquet(f"{plot_dir}/window_distances{suffix}.parquet",
                        index=False)

        plot_failure_vs_eta(
            dist, f"{plot_dir}/failure_rate_vs_pathInX0{suffix}.png",
            eta_col="pathInX0",
            x_range=(0, 0.15),
            n_bins=30,
            x_label=r"$X/X_0$ (interval)",
            subtitle=f"{label} seeds",
            y_max=0.8,
            min_count=30,
        )

        plot_failure_vs_eta(
            dist, f"{plot_dir}/failure_rate_vs_cumX0{suffix}.png",
            eta_col="cumX0",
            x_range=(0, 1.5),
            n_bins=30,
            x_label=r"Accumulated $X/X_0$",
            subtitle=f"{label} seeds",
            y_max=0.8,
            min_count=30,
        )

        print(f"{label}: {len(dist)} states, "
              f"median pathInX0={dist['pathInX0'].median():.4f}, "
              f"median cumX0={dist['cumX0'].median():.4f}")

    data_vol.commit()


@app.function(
    image=image,
    volumes={BUILD_PATH: build_vol, DATA_PATH: data_vol},
    cpu=8,
    memory=65536,
    timeout=1800,
)
def run_pilot_expansion(pilot_dir: str = f"{DATA_PATH}/results/pilot_1786472672"):
    """Stage 2: Run the offline expansion pipeline on pilot data.

    Reads the trackstates ROOT + CSVs produced by run_pilot_stage1,
    runs expansion.run_expansion for each event, writes Parquet.
    """
    import os
    import sys
    import time

    sys.path.insert(0, "/app")
    data_vol.reload()

    from expansion import run_expansion

    root_path = f"{pilot_dir}/trackstates_ckf.root"
    csv_dir = pilot_dir
    out_dir = f"{pilot_dir}/expanded"
    os.makedirs(out_dir, exist_ok=True)

    import uproot
    with uproot.open(root_path) as f:
        tree = f["trackstates"]
        event_ids = sorted(set(tree["event_nr"].array(library="np")))
    print(f"Found {len(event_ids)} events: {event_ids}")

    for eid in event_ids:
        out_path = f"{out_dir}/expanded_event{eid:09d}.parquet"
        print(f"\n=== Expanding event {eid} ===")
        t0 = time.time()
        df = run_expansion(root_path, csv_dir, eid, out_path)
        wall = time.time() - t0
        print(f"  {len(df)} rows, {df.shape[1]} cols, {wall:.1f}s")
        print(f"  Columns: {sorted(df.columns.tolist())}")

        n_branches = df[["event_id", "track_nr"]].drop_duplicates().shape[0]
        n_surfaces = df[["event_id", "track_nr", "state_idx"]].drop_duplicates().shape[0]
        n_cands = (df["cand_hit_id"] >= 0).sum()
        n_holes = (df["cand_hit_id"] < 0).sum()
        print(f"  Branches: {n_branches}, surfaces: {n_surfaces}")
        print(f"  Candidates: {n_cands}, holes: {n_holes}")

        if "branch_majority_pid" in df.columns:
            has_maj = (~df["majority_undefined"]).any()
            print(f"  Majority PID defined: {has_maj}")

    data_vol.commit()
    print(f"\nExpansion complete. Output at {out_dir}")


@app.function(
    image=image,
    volumes={DATA_PATH: data_vol},
    timeout=300,
)
def inspect_expanded_parquet(
    path: str = f"{DATA_PATH}/results/pilot_1786472672/expanded/expanded_event000000000.parquet",
):
    """Inspect an expanded Parquet file: schema, nulls, distributions, sample rows."""
    import pyarrow.parquet as pq
    import pandas as pd
    import numpy as np

    data_vol.reload()
    meta = pq.read_metadata(path)
    schema = pq.read_schema(path)
    print(f"=== {path} ===")
    print(f"Rows: {meta.num_rows:,}  Row groups: {meta.num_row_groups}  Columns: {meta.num_columns}")
    print(f"Serialized size: {meta.serialized_size:,} bytes")
    print(f"\n--- Schema ({schema.names.__len__()} cols) ---")
    for i, field in enumerate(schema):
        print(f"  {field.name}: {field.type}")

    df = pd.read_parquet(path)
    print(f"\n--- Null counts ---")
    nulls = df.isnull().sum()
    for col in nulls[nulls > 0].index:
        print(f"  {col}: {nulls[col]:,} ({100*nulls[col]/len(df):.1f}%)")
    if nulls.sum() == 0:
        print("  (none)")

    print(f"\n--- Numeric column stats ---")
    numeric_cols = [
        "S00", "S01", "S11", "chi2_inc", "residual_l0", "residual_l1",
        "pred_l0", "pred_l1", "eta", "n_window", "geometric_density",
        "pathInX0_interval", "n_hits", "n_holes", "n_seq_holes",
    ]
    for col in numeric_cols:
        if col in df.columns:
            s = df[col].dropna()
            print(f"  {col}: min={s.min():.4g}  median={s.median():.4g}  "
                  f"max={s.max():.4g}  mean={s.mean():.4g}  std={s.std():.4g}")

    print(f"\n--- Categorical distributions ---")
    for col in ["volume_id", "is_pixel", "is_barrel", "action_taken",
                "majority_undefined", "majority_true_hit_on_surface"]:
        if col in df.columns:
            vc = df[col].value_counts().head(8)
            print(f"  {col}: {dict(vc)}")

    cand_pos = (df["cand_hit_id"] >= 0).sum()
    cand_neg = (df["cand_hit_id"] < 0).sum()
    print(f"\n  cand_hit_id: {cand_pos:,} candidates, {cand_neg:,} holes "
          f"({100*cand_neg/(cand_pos+cand_neg):.1f}% hole rate)")

    print(f"\n--- Sample rows (first 5) ---")
    sample_cols = [c for c in ["event_id", "seed_id", "branch_id", "step_k",
                   "surface_id", "cand_hit_id", "residual_l0", "residual_l1",
                   "chi2_inc", "n_window", "branch_majority_pid",
                   "majority_true_hit_on_surface"] if c in df.columns]
    print(df[sample_cols].head().to_string())

    print(f"\n--- Sample candidate rows (chi2_inc present) ---")
    cands = df[df["cand_hit_id"] >= 0].head(5)
    print(cands[sample_cols].to_string())


@app.function(
    image=image,
    volumes={DATA_PATH: data_vol},
    cpu=2,
    memory=8192,
    timeout=120,
)
def sample_parquet(path: str, n_rows: int = 5):
    """Read schema + first N rows from a parquet file without loading the whole thing."""
    import pyarrow.parquet as pq
    import pandas as pd
    data_vol.reload()
    meta = pq.read_metadata(path)
    schema = pq.read_schema(path)
    pf = pq.ParquetFile(path)
    rows = []
    hit_rows = []
    for batch in pf.iter_batches(batch_size=1000):
        df_batch = batch.to_pandas()
        rows.append(df_batch)
        hits = df_batch[df_batch["cand_hit_id"] >= 0]
        if len(hits) > 0:
            hit_rows.append(hits)
        if len(pd.concat(hit_rows)) >= n_rows if hit_rows else False:
            break
        if len(pd.concat(rows)) >= 50000:
            break
    df_all = pd.concat(rows)
    df_hits = pd.concat(hit_rows) if hit_rows else pd.DataFrame()
    return {
        "num_rows": meta.num_rows,
        "num_columns": meta.num_columns,
        "schema": [(f.name, str(f.type)) for f in schema],
        "sample_hits": df_hits.head(n_rows).to_dict(orient="records"),
        "sample_holes": df_all[df_all["cand_hit_id"] < 0].head(2).to_dict(orient="records"),
    }


@app.function(
    image=image,
    volumes={DATA_PATH: data_vol},
    cpu=2,
    memory=4096,
    timeout=120,
)
def peek_csv(path: str, n_rows: int = 5):
    """Read header + first N rows from a CSV on the volume."""
    import pandas as pd
    data_vol.reload()
    df = pd.read_csv(path, nrows=n_rows)
    return {
        "columns": list(df.columns),
        "n_columns": len(df.columns),
        "rows": df.to_dict(orient="records"),
    }


@app.function(
    image=image,
    volumes={DATA_PATH: data_vol},
    cpu=2,
    memory=4096,
    timeout=120,
)
def peek_detectors(path: str):
    """Summarize detectors.csv: which volumes have pitch/thickness."""
    import pandas as pd
    data_vol.reload()
    df = pd.read_csv(path)
    has_pitch = df[(df["pitch_u"] > 0) & (df["pitch_v"] > 0)]
    return {
        "total_rows": len(df),
        "n_with_pitch": len(has_pitch),
        "volumes_with_pitch": sorted(has_pitch["volume_id"].unique().tolist()),
        "sample_with_pitch": has_pitch.head(5)[
            ["volume_id", "layer_id", "module_id", "pitch_u", "pitch_v", "module_t"]
        ].to_dict(orient="records"),
        "columns": list(df.columns),
    }


@app.function(
    image=image,
    volumes={DATA_PATH: data_vol},
    cpu=2,
    memory=4096,
    timeout=120,
)
def list_pilot_files(pilot_dir: str):
    """List files in a pilot directory on the volume."""
    import os
    data_vol.reload()
    if not os.path.isdir(pilot_dir):
        return {"pilot_dir": pilot_dir, "exists": False, "files": []}
    files = []
    for root, dirs, filenames in os.walk(pilot_dir):
        for fn in sorted(filenames):
            fp = os.path.join(root, fn)
            rel = os.path.relpath(fp, pilot_dir)
            sz = os.path.getsize(fp)
            files.append({"name": rel, "size_mb": round(sz / 1024 / 1024, 2)})
    return {"pilot_dir": pilot_dir, "exists": True, "files": files}


@app.function(
    image=image,
    volumes={DATA_PATH: data_vol},
    cpu=2,
    memory=4096,
    timeout=120,
)
def inspect_root_file(root_path: str):
    """Inspect a ROOT file: size, keys, tree names, entries."""
    import os
    import uproot
    data_vol.reload()
    result = {"path": root_path}
    if not os.path.exists(root_path):
        result["exists"] = False
        return result
    result["exists"] = True
    result["size_mb"] = round(os.path.getsize(root_path) / 1024 / 1024, 2)
    try:
        with uproot.open(root_path) as f:
            all_keys = list(f.keys())
            result["keys"] = all_keys
            result["classnames"] = {k: type(v).__name__ for k, v in f.items()}
            for k in all_keys:
                obj = f[k]
                if hasattr(obj, "num_entries"):
                    result[f"entries_{k}"] = obj.num_entries
    except Exception as e:
        result["error"] = str(e)
    return result


@app.function(
    image=image,
    volumes={BUILD_PATH: build_vol, DATA_PATH: data_vol},
    cpu=8,
    memory=262144,
    timeout=3600,
)
def expand_single_event(
    root_path: str,
    csv_dir: str,
    event_id: int,
    out_path: str,
):
    """Expand a single event. Called via .map() for parallelism."""
    import os
    import sys
    import time

    sys.path.insert(0, "/app")
    for vol in (data_vol, build_vol):
        try:
            vol.reload()
        except RuntimeError:
            pass

    if not os.path.exists(root_path):
        raise FileNotFoundError(f"ROOT file not found: {root_path}")

    from expansion import run_expansion

    t0 = time.time()
    df = run_expansion(
        root_path, csv_dir, event_id, out_path,
        digi_config_path="/app/configs/odd-digi-geometric-config.json",
    )
    wall = time.time() - t0

    n_branches = df[["event_id", "track_nr"]].drop_duplicates().shape[0]
    n_cands = int((df["cand_hit_id"] >= 0).sum())
    n_holes = int((df["cand_hit_id"] < 0).sum())

    data_vol.commit()
    return {
        "event_id": event_id,
        "rows": len(df),
        "cols": df.shape[1],
        "branches": n_branches,
        "candidates": n_cands,
        "holes": n_holes,
        "wall_time": round(wall, 1),
    }


@app.function(
    image=image,
    volumes={BUILD_PATH: build_vol, DATA_PATH: data_vol},
    cpu=4,
    memory=131072,
    timeout=3600,
)
def patch_single_event(
    parquet_path: str,
    root_path: str,
    event_id: int,
):
    """Patch one expanded parquet file with S00/S01/S11, volume_id, and sensor props."""
    import time
    import numpy as np
    import pandas as pd
    import pyarrow.parquet as pq
    import pyarrow as pa
    import uproot
    import awkward as ak

    for vol in [build_vol, data_vol]:
        try:
            vol.reload()
        except RuntimeError:
            pass

    # Load digi config
    digi_config_path = "/app/configs/odd-digi-geometric-config.json"
    with open(digi_config_path) as f:
        cfg = json.load(f)
    digi_table = {}
    for entry in cfg.get("entries", []):
        volume = entry.get("volume")
        if volume is None:
            continue
        geo = entry.get("value", {}).get("geometric")
        if geo is None:
            continue
        bins = geo.get("segmentation", {}).get("binningdata", [])
        pitch_u = pitch_v = np.nan
        if len(bins) > 0:
            b0 = bins[0]
            if b0.get("bins", 0) > 0:
                pitch_u = (b0["max"] - b0["min"]) / b0["bins"]
        if len(bins) > 1:
            b1 = bins[1]
            if b1.get("bins", 0) > 0:
                pitch_v = (b1["max"] - b1["min"]) / b1["bins"]
        digi_table[int(volume)] = {
            "pitch_u": pitch_u, "pitch_v": pitch_v,
            "thickness": float(geo.get("thickness", np.nan)),
        }

    BARREL_VOLUMES = {16, 23, 28}

    # Load S00/S01/S11 and volume_id from ROOT
    with uproot.open(root_path) as f:
        tree_names = [k for k in f.keys() if "trackstate" in k.lower()]
        if not tree_names:
            tree_names = list(f.keys())
        tree = f[tree_names[0]]
        available = set(tree.keys())
        read_fields = [b for b in ["volume_id", "S00_prt", "S01_prt", "S11_prt"] if b in available]
        if "event_nr" in available:
            read_fields.append("event_nr")
        arrays = tree.arrays(read_fields, library="ak")
        if "event_nr" in available:
            mask = ak.to_numpy(arrays["event_nr"]) == int(event_id)
            arrays = arrays[mask]

    n_tracks = len(arrays)
    n_states = ak.to_numpy(ak.num(arrays["volume_id"], axis=1))
    track_nr = np.repeat(np.arange(n_tracks, dtype=np.int64), n_states)
    state_idx = ak.to_numpy(ak.flatten(ak.local_index(arrays["volume_id"], axis=1))).astype(np.int64)
    root_df = pd.DataFrame({
        "track_nr": track_nr,
        "state_idx": state_idx,
        "volume_id": ak.to_numpy(ak.flatten(arrays["volume_id"])).astype(np.int64),
        "S00": ak.to_numpy(ak.flatten(arrays["S00_prt"])).astype(np.float64) if "S00_prt" in available else np.full(len(track_nr), np.nan),
        "S01": ak.to_numpy(ak.flatten(arrays["S01_prt"])).astype(np.float64) if "S01_prt" in available else np.full(len(track_nr), np.nan),
        "S11": ak.to_numpy(ak.flatten(arrays["S11_prt"])).astype(np.float64) if "S11_prt" in available else np.full(len(track_nr), np.nan),
    })

    t0 = time.time()

    # Read parquet and merge
    df = pd.read_parquet(parquet_path)
    n_rows = len(df)
    original_cols = list(df.columns)

    df = df.merge(root_df, left_on=["seed_id", "step_k"], right_on=["track_nr", "state_idx"],
                  how="left", suffixes=("", "_root"))
    df.drop(columns=["track_nr", "state_idx"], inplace=True, errors="ignore")
    for col in ("S00", "S01", "S11", "volume_id"):
        root_col = f"{col}_root"
        if root_col in df.columns:
            df[col] = df[root_col]
            df.drop(columns=[root_col], inplace=True)

    # Fill sensor properties
    vol = df["volume_id"].to_numpy().astype(np.int64) if "volume_id" in df.columns else np.full(n_rows, -1, dtype=np.int64)
    pitch_u = np.full(n_rows, np.nan)
    pitch_v = np.full(n_rows, np.nan)
    thickness = np.full(n_rows, np.nan)
    is_barrel = np.full(n_rows, np.nan)
    for v_int, props in digi_table.items():
        mask = vol == v_int
        pitch_u[mask] = props["pitch_u"]
        pitch_v[mask] = props["pitch_v"]
        thickness[mask] = props["thickness"]
    known_vol_mask = vol > 0
    barrel_mask = np.isin(vol, list(BARREL_VOLUMES))
    is_barrel[known_vol_mask] = 0.0
    is_barrel[barrel_mask] = 1.0
    df["pitch_u"] = pitch_u
    df["pitch_v"] = pitch_v
    df["thickness"] = thickness
    df["is_barrel"] = is_barrel

    # Reorder columns
    insert_after_chi2 = ["S00", "S01", "S11"]
    final_cols = []
    for c in original_cols:
        final_cols.append(c)
        if c == "chi2_inc":
            final_cols.extend(insert_after_chi2)
        elif c == "surface_id":
            final_cols.append("volume_id")
    for c in df.columns:
        if c not in final_cols:
            final_cols.append(c)
    final_cols = [c for c in final_cols if c in df.columns]
    df = df[final_cols]

    # Write back
    table = pa.Table.from_pandas(df, preserve_index=False)
    for col_name, pa_type in [("contrib_pids", pa.list_(pa.int64())), ("contrib_charge_frac", pa.list_(pa.float32()))]:
        if col_name in table.column_names:
            idx = table.column_names.index(col_name)
            try:
                col = table.column(col_name).cast(pa_type)
                table = table.set_column(idx, col_name, col)
            except (pa.ArrowInvalid, pa.ArrowNotImplementedError):
                pass
    pq.write_table(table, parquet_path, compression="zstd")

    wall = time.time() - t0
    s00_fill = float((~np.isnan(df["S00"].to_numpy())).mean()) if "S00" in df.columns else 0
    pitch_fill = float((~np.isnan(df["pitch_u"].to_numpy())).mean()) if "pitch_u" in df.columns else 0
    vol_fill = float((df["volume_id"] > 0).mean()) if "volume_id" in df.columns else 0

    data_vol.commit()
    return {
        "event_id": event_id,
        "rows": n_rows,
        "S00_fill_pct": round(100 * s00_fill, 1),
        "pitch_fill_pct": round(100 * pitch_fill, 1),
        "volume_id_fill_pct": round(100 * vol_fill, 1),
        "wall_time": round(wall, 1),
    }


@app.local_entrypoint()
def patch_all_events():
    """Patch all 32 expanded parquet files with S00/S01/S11 and sensor props.

    Runs sequentially to avoid concurrent volume commits corrupting files.
    """
    pilot_dirs = [
        ("/data/results/pilot_1786524194", 0),
        ("/data/results/pilot_1786524971", 2),
        ("/data/results/pilot_1786525888", 4),
        ("/data/results/pilot_1786527639", 6),
        ("/data/results/pilot_1786529841", 8),
        ("/data/results/pilot_1786531084", 10),
        ("/data/results/pilot_1786532999", 12),
        ("/data/results/pilot_1786534329", 14),
        ("/data/results/pilot_1786536577", 16),
        ("/data/results/pilot_1786538022", 18),
        ("/data/results/pilot_1786539360", 20),
        ("/data/results/pilot_1786540365", 22),
        ("/data/results/pilot_1786541815", 24),
        ("/data/results/pilot_1786542563", 26),
        ("/data/results/pilot_1786543789", 28),
        ("/data/results/pilot_1786547065", 30),
    ]
    expanded_dir = f"{DATA_PATH}/results/train32/expanded"

    patch_args = []
    for pilot_dir, global_start in pilot_dirs:
        root_path = f"{pilot_dir}/trackstates_ckf.root"
        for offset in range(2):
            global_eid = global_start + offset
            parquet_path = f"{expanded_dir}/expanded_event{global_eid:09d}.parquet"
            patch_args.append((parquet_path, root_path, global_eid))

    print(f"=== Patching {len(patch_args)} expanded parquet files (sequential) ===")
    for args in patch_args:
        parquet_path, root_path, eid = args
        print(f"  Starting event {eid}...", flush=True)
        try:
            result = patch_single_event.remote(parquet_path, root_path, eid)
            print(f"  Event {result['event_id']:2d}: {result['rows']:>12,} rows  "
                  f"S00={result['S00_fill_pct']:5.1f}%  "
                  f"pitch={result['pitch_fill_pct']:5.1f}%  "
                  f"vol_id={result['volume_id_fill_pct']:5.1f}%  "
                  f"{result['wall_time']:.0f}s")
        except Exception as e:
            print(f"  Event {eid}: FAILED — {e}")
    print("\n=== Done ===")


@app.local_entrypoint()
def repair_and_patch():
    """Re-expand corrupt events, then patch all remaining (corrupt + unpatched).

    Runs everything sequentially to avoid concurrent volume corruption.
    """
    pilot_dirs = [
        ("/data/results/pilot_1786524194", 0),
        ("/data/results/pilot_1786524971", 2),
        ("/data/results/pilot_1786525888", 4),
        ("/data/results/pilot_1786527639", 6),
        ("/data/results/pilot_1786529841", 8),
        ("/data/results/pilot_1786531084", 10),
        ("/data/results/pilot_1786532999", 12),
        ("/data/results/pilot_1786534329", 14),
        ("/data/results/pilot_1786536577", 16),
        ("/data/results/pilot_1786538022", 18),
        ("/data/results/pilot_1786539360", 20),
        ("/data/results/pilot_1786540365", 22),
        ("/data/results/pilot_1786541815", 24),
        ("/data/results/pilot_1786542563", 26),
        ("/data/results/pilot_1786543789", 28),
        ("/data/results/pilot_1786547065", 30),
    ]
    expanded_dir = f"{DATA_PATH}/results/train32/expanded"

    corrupt_events = {5, 6, 10, 14, 15, 17, 18, 22, 28, 29, 30}
    unpatched_events = set()
    needs_work = corrupt_events | unpatched_events

    # Build event_id → pilot_dir mapping
    eid_to_pilot = {}
    for pilot_dir, global_start in pilot_dirs:
        for offset in range(2):
            eid_to_pilot[global_start + offset] = pilot_dir

    # Phase 1: Re-expand corrupt events
    print(f"=== Phase 1: Re-expanding {len(corrupt_events)} corrupt events ===")
    for eid in sorted(corrupt_events):
        pilot_dir = eid_to_pilot[eid]
        root_path = f"{pilot_dir}/trackstates_ckf.root"
        csv_dir = pilot_dir
        out_path = f"{expanded_dir}/expanded_event{eid:09d}.parquet"
        print(f"  Re-expanding event {eid}...", flush=True)
        try:
            result = expand_single_event.remote(root_path, csv_dir, eid, out_path)
            print(f"  Event {eid:2d}: {result['rows']:>12,} rows, {result['cols']} cols, {result['wall_time']:.0f}s")
        except Exception as e:
            print(f"  Event {eid}: EXPAND FAILED — {e}")

    # Phase 2: Patch all that need it (re-expanded + unpatched)
    print(f"\n=== Phase 2: Patching {len(needs_work)} events ===")
    for eid in sorted(needs_work):
        pilot_dir = eid_to_pilot[eid]
        root_path = f"{pilot_dir}/trackstates_ckf.root"
        parquet_path = f"{expanded_dir}/expanded_event{eid:09d}.parquet"
        print(f"  Patching event {eid}...", flush=True)
        try:
            result = patch_single_event.remote(parquet_path, root_path, eid)
            print(f"  Event {result['event_id']:2d}: {result['rows']:>12,} rows  "
                  f"S00={result['S00_fill_pct']:5.1f}%  "
                  f"pitch={result['pitch_fill_pct']:5.1f}%  "
                  f"vol_id={result['volume_id_fill_pct']:5.1f}%  "
                  f"{result['wall_time']:.0f}s")
        except Exception as e:
            print(f"  Event {eid}: PATCH FAILED — {e}")
    print("\n=== Done ===")


@app.function(
    image=image,
    volumes={DATA_PATH: data_vol},
    cpu=1,
    memory=4096,
    timeout=120,
)
def check_parquet_health(parquet_path: str, event_id: int):
    """Check if a parquet file is readable and whether it's already patched."""
    import pyarrow.parquet as pq

    data_vol.reload()
    try:
        meta = pq.read_metadata(parquet_path)
        schema = pq.read_schema(parquet_path)
        cols = schema.names
        has_s00 = "S00" in cols
        has_pitch = "pitch_u" in cols
        has_vol = "volume_id" in cols
        return {
            "event_id": event_id,
            "status": "ok",
            "rows": meta.num_rows,
            "cols": len(cols),
            "has_S00": has_s00,
            "has_pitch": has_pitch,
            "has_volume_id": has_vol,
            "patched": has_s00 and has_pitch and has_vol,
        }
    except Exception as e:
        return {"event_id": event_id, "status": "CORRUPT", "error": str(e)[:200]}


@app.local_entrypoint()
def check_pilot_csvs():
    """Check which pilot directories have csv/ subdirs with measurements CSVs."""
    pilot_dirs = [
        ("/data/results/pilot_1786525888", 4),    # events 4-5
        ("/data/results/pilot_1786527639", 6),    # events 6-7
        ("/data/results/pilot_1786531084", 10),   # events 10-11
        ("/data/results/pilot_1786534329", 14),   # events 14-15
        ("/data/results/pilot_1786536577", 16),   # events 16-17
        ("/data/results/pilot_1786538022", 18),   # events 18-19
        ("/data/results/pilot_1786540365", 22),   # events 22-23
        ("/data/results/pilot_1786543789", 28),   # events 28-29
        ("/data/results/pilot_1786547065", 30),   # events 30-31
    ]
    for result in list_pilot_files.map([d for d, _ in pilot_dirs]):
        print(f"\n{result['pilot_dir']}:")
        if not result['exists']:
            print("  DOES NOT EXIST")
            continue
        for f in result['files']:
            print(f"  {f['name']:60s} {f['size_mb']:8.2f} MB")


@app.local_entrypoint()
def check_all_parquets():
    """Check health of all 32 expanded parquet files."""
    expanded_dir = f"{DATA_PATH}/results/train32/expanded"
    args = []
    for eid in range(32):
        path = f"{expanded_dir}/expanded_event{eid:09d}.parquet"
        args.append((path, eid))

    print("=== Parquet Health Check ===")
    corrupt = []
    patched = []
    unpatched = []
    for result in check_parquet_health.starmap(args, order_outputs=False):
        eid = result["event_id"]
        if result["status"] == "CORRUPT":
            print(f"  Event {eid:2d}: CORRUPT — {result['error'][:80]}")
            corrupt.append(eid)
        elif result.get("patched"):
            print(f"  Event {eid:2d}: OK (patched) — {result['rows']:>12,} rows, {result['cols']} cols")
            patched.append(eid)
        else:
            print(f"  Event {eid:2d}: OK (unpatched) — {result['rows']:>12,} rows, {result['cols']} cols  "
                  f"S00={result['has_S00']} pitch={result['has_pitch']} vol={result['has_volume_id']}")
            unpatched.append(eid)
    print(f"\nSummary: {len(patched)} patched, {len(unpatched)} unpatched, {len(corrupt)} corrupt")
    if corrupt:
        print(f"  Corrupt events: {sorted(corrupt)}")
    if unpatched:
        print(f"  Unpatched events: {sorted(unpatched)}")


@app.local_entrypoint()
def run_full_pipeline(events: int = 32, batch: int = 2):
    """Run CKF in batches, then expand each event in parallel.

    Stage 1 OOMs at 64GB with >4 events (envelope CKF, no branch caps).
    Process in batches of `batch` events, each producing its own output dir.
    Then expand all events in parallel across all batches.
    """
    import time as _time

    base_dir = f"{DATA_PATH}/results/train32"
    expand_args = []

    for start in range(0, events, batch):
        n = min(batch, events - start)
        print(f"\n=== Stage 1: CKF batch events [{start}, {start + n}) ===")
        for attempt in range(3):
            try:
                result = run_pilot_stage1.remote(events=n, skip=start)
                break
            except Exception as e:
                if attempt < 2 and "open files" in str(e):
                    print(f"  Volume conflict, retrying in 15s (attempt {attempt + 1})...")
                    _time.sleep(15)
                else:
                    raise
        pilot_dir = result["output_dir"]
        print(f"  Batch done in {result['wall_time']:.0f}s -> {pilot_dir}")

        root_path = f"{pilot_dir}/trackstates_ckf.root"
        csv_dir = pilot_dir
        out_dir = f"{base_dir}/expanded"
        for local_eid in range(n):
            global_eid = start + local_eid
            expand_args.append((
                root_path, csv_dir, local_eid,
                f"{out_dir}/expanded_event{global_eid:09d}.parquet",
            ))
        _time.sleep(5)

    print(f"\n=== Stage 2: Expanding {len(expand_args)} events in parallel ===")
    total_rows = 0
    results = []
    for result in expand_single_event.starmap(expand_args, order_outputs=False):
        eid = result["event_id"]
        total_rows += result["rows"]
        results.append(result)
        print(f"  Event {eid}: {result['rows']:,} rows, "
              f"{result['branches']:,} branches, "
              f"{result['candidates']:,} cands, "
              f"{result['holes']:,} holes, "
              f"{result['wall_time']}s")

    print(f"\n=== Done: {total_rows:,} total rows across {len(results)} events ===")
    print(f"Output: {base_dir}/expanded")


@app.local_entrypoint()
def check_pilot_dirs():
    """Verify all pilot directories and inspect ROOT files."""
    pilot_dirs = [
        "/data/results/pilot_1786524194",   # events 0-1
        "/data/results/pilot_1786524971",   # events 2-3
        "/data/results/pilot_1786525888",   # events 4-5
        "/data/results/pilot_1786527639",   # events 6-7
        "/data/results/pilot_1786529841",   # events 8-9
        "/data/results/pilot_1786531084",   # events 10-11
        "/data/results/pilot_1786532999",   # events 12-13
        "/data/results/pilot_1786534329",   # events 14-15
        "/data/results/pilot_1786536577",   # events 16-17
        "/data/results/pilot_1786538022",   # events 18-19
        "/data/results/pilot_1786539360",   # events 20-21
        "/data/results/pilot_1786540365",   # events 22-23
        "/data/results/pilot_1786541815",   # events 24-25
        "/data/results/pilot_1786542563",   # events 26-27
        "/data/results/pilot_1786543789",   # events 28-29
        "/data/results/pilot_1786546373",   # events 30-31
    ]
    root_paths = [f"{d}/trackstates_ckf.root" for d in pilot_dirs]
    print("=== ROOT file inspection ===")
    for info in inspect_root_file.map(root_paths, order_outputs=False):
        p = info["path"].replace("/data/results/", "")
        if not info.get("exists"):
            print(f"  MISSING: {p}")
        elif "error" in info:
            print(f"  ERROR: {p} ({info['size_mb']} MB) — {info['error']}")
        else:
            keys = info.get("keys", [])
            classes = info.get("classnames", {})
            print(f"  {p}: {info['size_mb']} MB, keys={keys}, classes={classes}")


@app.local_entrypoint()
def expand_all_events():
    """Expand all 32 training events from pre-computed stage 1 batches."""
    pilot_dirs = [
        ("/data/results/pilot_1786524194", 0),    # events 0-1
        ("/data/results/pilot_1786524971", 2),    # events 2-3
        ("/data/results/pilot_1786525888", 4),    # events 4-5
        ("/data/results/pilot_1786527639", 6),    # events 6-7
        ("/data/results/pilot_1786529841", 8),    # events 8-9
        ("/data/results/pilot_1786531084", 10),   # events 10-11
        ("/data/results/pilot_1786532999", 12),   # events 12-13
        ("/data/results/pilot_1786534329", 14),   # events 14-15
        ("/data/results/pilot_1786536577", 16),   # events 16-17
        ("/data/results/pilot_1786538022", 18),   # events 18-19
        ("/data/results/pilot_1786539360", 20),   # events 20-21
        ("/data/results/pilot_1786540365", 22),   # events 22-23
        ("/data/results/pilot_1786541815", 24),   # events 24-25
        ("/data/results/pilot_1786542563", 26),   # events 26-27
        ("/data/results/pilot_1786543789", 28),   # events 28-29
        ("/data/results/pilot_1786547065", 30),   # events 30-31
    ]

    already_done = {
        0, 1, 2, 3, 4, 6, 8, 9, 10, 11, 12, 13, 14, 15,
        16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
    }

    out_dir = f"{DATA_PATH}/results/train32/expanded"
    expand_args = []
    for pilot_dir, global_start in pilot_dirs:
        root_path = f"{pilot_dir}/trackstates_ckf.root"
        csv_dir = pilot_dir
        for local_eid in range(2):
            global_eid = global_start + local_eid
            if global_eid in already_done:
                print(f"  Skipping event {global_eid} (already expanded)")
                continue
            expand_args.append((
                root_path, csv_dir, global_eid,
                f"{out_dir}/expanded_event{global_eid:09d}.parquet",
            ))

    print(f"=== Expanding {len(expand_args)} remaining events in parallel (256 GB) ===")
    total_rows = 0
    results = []
    for result in expand_single_event.starmap(expand_args, order_outputs=False):
        eid = result["event_id"]
        total_rows += result["rows"]
        results.append(result)
        print(f"  Event {eid}: {result['rows']:,} rows, "
              f"{result['branches']:,} branches, "
              f"{result['candidates']:,} cands, "
              f"{result['holes']:,} holes, "
              f"{result['wall_time']}s")

    print(f"\n=== Done: {total_rows:,} total rows across {len(results)} events ===")
    print(f"Output: {out_dir}")


# ---------------------------------------------------------------------------
# Task 9: end-to-end cCKF smoke test
# ---------------------------------------------------------------------------


def _write_dummy_weight_blob(path, n_features, n_hidden, n_layers, rng):
    """Write a randomly-initialized CCKF weight blob.

    Matches the binary format read by ``WeightBlob::load``
    (``acts_patches/cckf/WeightBlob.hpp``) and written by
    ``scripts/export_weights.py``: magic ``"CCKF"``, version 1, then
    ``n_features``/``n_hidden``/``n_layers`` as little-endian uint32,
    standardization mean/std (``n_features`` float32 each), 4 Platt
    params, then ``n_layers`` hidden (n_hidden, in_dim) weight/bias pairs
    followed by one (1, n_hidden) head weight/bias pair.

    This is intentionally dependency-free (no torch, no ``cckf.models``) so
    the Modal image for this file doesn't need a torch install just to
    produce smoke-test fixtures — it writes the same bytes
    ``export_weights.export()`` would for a randomly-initialized
    ``GateMLP``/``ValueMLP``, using identity standardization
    (mean=0, std=1) and an identity Platt fit (a=1, b=0) so the blob is
    self-consistent even though the weights themselves are meaningless.
    """
    import struct

    import numpy as np

    layer_dims = [n_features] + [n_hidden] * n_layers + [1]
    with open(path, "wb") as f:
        f.write(b"CCKF")
        f.write(struct.pack("<I", 1))  # version
        f.write(struct.pack("<III", n_features, n_hidden, n_layers))
        f.write(np.zeros(n_features, dtype=np.float32).tobytes())  # mean
        f.write(np.ones(n_features, dtype=np.float32).tobytes())  # std
        f.write(struct.pack("<ffff", 1.0, 0.0, 0.0, 0.0))  # identity Platt
        for i in range(n_layers + 1):
            in_dim, out_dim = layer_dims[i], layer_dims[i + 1]
            scale = np.sqrt(2.0 / in_dim).astype(np.float32)
            w = rng.standard_normal((out_dim, in_dim)).astype(np.float32) * scale
            b = np.zeros(out_dim, dtype=np.float32)
            f.write(w.tobytes())
            f.write(b.tobytes())


@app.function(
    image=image,
    volumes={DATA_PATH: data_vol},
    cpu=2,
    memory=4096,
    timeout=300,
)
def generate_dummy_weights(seed: int = 0):
    """Write random gate.bin/value.bin CCKF blobs to the data volume.

    For the Task 9 smoke test only: verifies the pipeline compiles and runs
    end-to-end with the correct binary format, not that the resulting
    tracking is any good. Real weights come from a trained checkpoint run
    through ``scripts/export_weights.py``.

    Shapes match ``cckf_envelope.yaml``'s expected paths
    (``/data/weights/gate.bin``, ``/data/weights/value.bin``) and
    ``cckf/models.py``: gate is 26->128x3->1 (``GateMLP`` defaults, matching
    the ``n_layers == 3`` assertion in ``export_weights.export``), value is
    11->128x2->1 (``ValueMLP`` defaults, ``n_layers == 2``).
    """
    import os

    import numpy as np

    data_vol.reload()
    weights_dir = f"{DATA_PATH}/weights"
    os.makedirs(weights_dir, exist_ok=True)

    rng = np.random.default_rng(seed)
    gate_path = f"{weights_dir}/gate.bin"
    value_path = f"{weights_dir}/value.bin"
    _write_dummy_weight_blob(gate_path, n_features=26, n_hidden=128, n_layers=3, rng=rng)
    _write_dummy_weight_blob(value_path, n_features=11, n_hidden=128, n_layers=2, rng=rng)

    data_vol.commit()

    sizes = {p: os.path.getsize(p) for p in (gate_path, value_path)}
    print(f"Wrote dummy weight blobs: {sizes}")
    return {"gate_path": gate_path, "value_path": value_path, "sizes": sizes}


@app.function(
    image=image,
    volumes={BUILD_PATH: build_vol, DATA_PATH: data_vol},
    cpu=8,
    memory=65536,
    timeout=3600,
)
def run_cckf(
    config: str = "cckf_envelope",
    events: int = 2,
    gate_threshold: float = 0.5,
    value_threshold: float = 0.1,
):
    """End-to-end cCKF smoke test: patched CckfTrackFindingAlgorithm on real events.

    Loads ``configs/<config>.yaml`` (default ``cckf_envelope.yaml``, which
    sets ``cckf: true`` so ``digi_and_reco.py``'s ``setup_acts_reconstruction``
    routes the CKF stage through ``addCckfTracks()`` /
    ``CckfTrackFindingAlgorithm`` instead of the stock ``addCKFTracks()``),
    overrides the event count and gate/value thresholds, runs the pipeline,
    and parses the resulting ROOT/CSV output into a metrics dict via
    ``scripts/extract_metrics.py``.

    Preconditions (not run here — see ``scripts/smoke_test_cckf.sh``):
    ``build_acts(force=True)`` must have produced a patched ACTS install on
    ``build_vol``, and ``generate_dummy_weights()`` (or a real
    ``export_weights.py`` run) must have written
    ``/data/weights/gate.bin`` and ``/data/weights/value.bin`` to
    ``data_vol``.
    """
    import json
    import os
    import sys
    import time
    import types
    from pathlib import Path

    import yaml

    build_vol.reload()
    data_vol.reload()
    _setup_acts_env(use_patched=True)

    # Symlink material maps (same pattern as verify_root_branches)
    material_map = f"{DATA_PATH}/odd-material-maps.root"
    if not os.path.exists(material_map):
        raise FileNotFoundError("Material maps not found. Run seed_material_maps first.")
    odd_data_dir = f"{ODD_INSTALL}/data"
    os.makedirs(odd_data_dir, exist_ok=True)
    map_link = f"{odd_data_dir}/odd-material-maps.root"
    if not os.path.exists(map_link):
        os.symlink(material_map, map_link)

    # Weight blobs must already be on the data volume (generate_dummy_weights
    # or a real export_weights.py run).
    gate_weights = f"{DATA_PATH}/weights/gate.bin"
    value_weights = f"{DATA_PATH}/weights/value.bin"
    for blob_path, label in [(gate_weights, "gate"), (value_weights, "value")]:
        if not os.path.exists(blob_path):
            raise FileNotFoundError(
                f"{label} weight blob not found at {blob_path}. Run "
                "generate_dummy_weights (or export_weights.py for trained "
                "weights) first."
            )

    # Load config — cckf_envelope.yaml already sets cckf: true, skip: 32
    # (test set [32, 64), spec §6.1), and the weight/threshold defaults;
    # here we override event count and thresholds per the function args.
    config_name = config if config.endswith(".yaml") else f"{config}.yaml"
    config_path = Path("/app/configs") / config_name
    if not config_path.is_file():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with open(config_path) as f:
        cfg_dict = yaml.safe_load(f)

    cfg_dict.update({
        "events": events,
        "threads": 8,
        "cckf_gate_weights": gate_weights,
        "cckf_value_weights": value_weights,
        "cckf_gate_threshold": gate_threshold,
        "cckf_value_threshold": value_threshold,
    })
    cfg = types.SimpleNamespace(**cfg_dict)
    cfg.seed = 42

    output_dir = Path(f"{DATA_PATH}/results/run_cckf_{int(time.time())}")
    output_dir.mkdir(parents=True, exist_ok=True)

    import acts, acts.examples
    print(f"acts module: {acts.__file__}")

    from digi_and_reco import setup_acts_reconstruction
    rnd = acts.examples.RandomNumbers(seed=cfg.seed)

    print(
        f"=== Running cCKF ({events} events, skip={getattr(cfg, 'skip', 0)}, "
        f"gate_threshold={gate_threshold}, value_threshold={value_threshold}) ==="
    )
    t0 = time.time()
    s = setup_acts_reconstruction(
        Path(f"{DATA_PATH}/events/edm4hep.root"),
        output_dir,
        cfg, rnd,
    )
    s.run()
    wall = time.time() - t0
    print(f"Pipeline completed in {wall:.1f}s")

    print("\n=== Output files ===")
    for f_out in sorted(output_dir.iterdir()):
        if f_out.name.startswith("_"):
            continue
        print(f"  {f_out.name}: {f_out.stat().st_size / 1e3:.1f} KB")

    data_vol.commit()

    # Parse metrics via scripts/extract_metrics.py's build_metrics()
    sys.path.insert(0, "/app")
    from scripts.extract_metrics import build_metrics

    metrics = build_metrics(
        output_dir,
        cckf_timing_name=getattr(cfg, "cckf_timing_output", "timing.csv"),
        acts_timing_name="timing.tsv",
        pre_ambi_name="performance_finding_ckf.root",
        post_ambi_name="performance_finding_ambi.root",
    )
    metrics["wall_seconds"] = round(wall, 1)
    metrics["output_dir"] = str(output_dir)
    metrics["config"] = config_name
    metrics["gate_threshold"] = gate_threshold
    metrics["value_threshold"] = value_threshold

    print("\n=== Metrics ===")
    print(json.dumps(metrics, indent=2, default=str))

    return metrics


# ---------------------------------------------------------------------------
# Task 10: Pareto sweep over (tau_g, tau_v)
# ---------------------------------------------------------------------------

def _extract_sweep_row(tau_g: float, tau_v: float, metrics: dict) -> dict:
    """Flatten a ``run_cckf`` metrics dict into one Pareto-sweep CSV row.

    Efficiency/fake rate are read post-ambi (the final, deployed numbers —
    same convention as ``run_joint_motpe_optimizer``'s objectives). Duplicate
    rate is reported both pre- and post-ambi per Task 8's per-event timing +
    duplicate-rate reporting. Per CLAUDE.md: efficiency = ``eff_particles``,
    fake/duplicate rate = ``fakeratio_tracks`` / ``duplicateratio_tracks``
    (DM matching, track-level fake/duplicate — NOT the particle-level ratios).
    """
    pre_ambi = metrics.get("summary", {}).get("pre_ambi") or {}
    post_ambi = metrics.get("summary", {}).get("post_ambi") or {}

    def _pct(block: dict, key: str) -> float | None:
        v = block.get(key)
        return None if v is None else round(float(v) * 100.0, 4)

    timing_rows = metrics.get("cckf_timing_per_event") or []
    gate_calls = sum(int(r.get("gate_inference_calls", 0) or 0) for r in timing_rows)
    value_calls = sum(int(r.get("value_inference_calls", 0) or 0) for r in timing_rows)

    wall_seconds = metrics.get("wall_seconds")
    events = metrics.get("events") or len(timing_rows) or None
    runtime_per_event_s = (
        round(wall_seconds / events, 4) if wall_seconds is not None and events else None
    )

    return {
        "tau_g": tau_g,
        "tau_v": tau_v,
        "efficiency": _pct(post_ambi, "eff_particles"),
        "fake_rate": _pct(post_ambi, "fakeratio_tracks"),
        "duplicate_rate_pre_ambi": _pct(pre_ambi, "duplicateratio_tracks"),
        "duplicate_rate_post_ambi": _pct(post_ambi, "duplicateratio_tracks"),
        "runtime_per_event_s": runtime_per_event_s,
        "gate_calls": gate_calls,
        "value_calls": value_calls,
        "wall_seconds": wall_seconds,
    }


def _is_dominated(row: dict, others: list[dict]) -> bool:
    """Non-domination test on (efficiency MAX, fake_rate MIN, wall_seconds MIN).

    A row is dominated if some other row is at least as good on all three
    axes and strictly better on at least one. Rows with a missing (None)
    objective value (a failed/partial run) can never dominate and are never
    reported as part of the front.
    """
    e, f, w = row["efficiency"], row["fake_rate"], row["wall_seconds"]
    if e is None or f is None or w is None:
        return True
    for other in others:
        if other is row:
            continue
        oe, of, ow = other["efficiency"], other["fake_rate"], other["wall_seconds"]
        if oe is None or of is None or ow is None:
            continue
        if oe >= e and of <= f and ow <= w and (oe > e or of < f or ow < w):
            return True
    return False


SWEEP_FIELDNAMES = [
    "tau_g",
    "tau_v",
    "efficiency",
    "fake_rate",
    "duplicate_rate_pre_ambi",
    "duplicate_rate_post_ambi",
    "runtime_per_event_s",
    "gate_calls",
    "value_calls",
    "wall_seconds",
]


@app.function(
    image=image,
    volumes={BUILD_PATH: build_vol, DATA_PATH: data_vol},
    cpu=2,
    memory=8192,
    timeout=86400,
)
def pareto_sweep(
    gate_thresholds: str = "0.1,0.3,0.5,0.7,0.9",
    value_thresholds: str = "0.01,0.05,0.1,0.2,0.5",
    events: int = 32,
    max_parallel: int = 8,
    config: str = "cckf_envelope",
):
    """Sweep the (tau_g, tau_v) grid with ``run_cckf`` and collect Pareto data.

    Fans out ``run_cckf.remote()`` calls (same app as this function — Task 9
    put ``run_cckf`` here because it's the only app with the patched ACTS
    build; cross-app ``.remote()`` calls aren't supported) over the full
    threshold grid, ``max_parallel`` at a time via ``ThreadPoolExecutor``
    (same pattern as ``validate_joint_motpe_eval`` in modal_acts.py).
    Resumes from an existing ``results/pareto_sweep.csv`` on the data volume
    if present, so a killed/timed-out sweep can be restarted without
    re-running already-completed grid points.

    Usage:
        modal run --detach modal_build_acts.py::pareto_sweep \\
            --gate-thresholds 0.1,0.3,0.5,0.7,0.9 \\
            --value-thresholds 0.01,0.05,0.1,0.2,0.5 \\
            --events 32 --max-parallel 8
    """
    import csv
    import itertools
    import os
    import time
    from concurrent.futures import ThreadPoolExecutor, as_completed

    data_vol.reload()

    tau_gs = [float(x) for x in gate_thresholds.split(",") if x.strip() != ""]
    tau_vs = [float(x) for x in value_thresholds.split(",") if x.strip() != ""]
    grid = list(itertools.product(tau_gs, tau_vs))

    results_dir = f"{DATA_PATH}/results"
    os.makedirs(results_dir, exist_ok=True)
    out_csv = f"{results_dir}/pareto_sweep.csv"

    print("=== Pareto sweep: (tau_g, tau_v) grid ===")
    print(f"tau_g: {tau_gs}")
    print(f"tau_v: {tau_vs}")
    print(f"Grid points: {len(grid)}  events/point: {events}  "
          f"total event-runs: {len(grid) * events}")
    print(f"config={config}  max_parallel={max_parallel}")
    print(f"Output: {out_csv}")

    # Resume: skip (tau_g, tau_v) pairs already present in an existing CSV.
    done: dict[tuple[float, float], dict] = {}
    if os.path.exists(out_csv):
        with open(out_csv, newline="") as f:
            for row in csv.DictReader(f):
                key = (round(float(row["tau_g"]), 6), round(float(row["tau_v"]), 6))
                done[key] = {
                    k: (float(v) if v not in ("", None) else None)
                    for k, v in row.items()
                }
                done[key]["tau_g"] = float(row["tau_g"])
                done[key]["tau_v"] = float(row["tau_v"])
        print(f"Resuming: {len(done)} grid point(s) already in {out_csv}")

    pending = [
        (tg, tv) for (tg, tv) in grid
        if (round(tg, 6), round(tv, 6)) not in done
    ]
    print(f"Remaining: {len(pending)}")

    def _run_one(tau_g: float, tau_v: float):
        t0 = time.time()
        try:
            metrics = run_cckf.remote(
                config=config,
                events=events,
                gate_threshold=tau_g,
                value_threshold=tau_v,
            )
            metrics.setdefault("events", events)
            row = _extract_sweep_row(tau_g, tau_v, metrics)
            return tau_g, tau_v, row, None
        except Exception as e:  # keep the sweep alive on a single bad point
            wall = time.time() - t0
            print(f"  [tau_g={tau_g} tau_v={tau_v}] FAILED after {wall:.0f}s: "
                  f"{type(e).__name__}: {e}")
            return tau_g, tau_v, None, str(e)

    rows_out = list(done.values())
    n_finished = 0

    with ThreadPoolExecutor(max_workers=max_parallel) as ex:
        futs = {ex.submit(_run_one, tg, tv): (tg, tv) for (tg, tv) in pending}
        for fut in as_completed(futs):
            tau_g, tau_v, row, err = fut.result()
            n_finished += 1
            if row is not None:
                rows_out.append(row)
                print(
                    f"  [{n_finished}/{len(pending)}] tau_g={tau_g} tau_v={tau_v}  "
                    f"eff={row['efficiency']}  fake={row['fake_rate']}  "
                    f"dup(pre/post)={row['duplicate_rate_pre_ambi']}/"
                    f"{row['duplicate_rate_post_ambi']}  "
                    f"t/evt={row['runtime_per_event_s']}s"
                )
                # Incremental save + volume commit for crash resilience.
                rows_out.sort(key=lambda r: (r["tau_g"], r["tau_v"]))
                with open(out_csv, "w", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=SWEEP_FIELDNAMES)
                    w.writeheader()
                    w.writerows(rows_out)
                data_vol.commit()

    rows_out.sort(key=lambda r: (r["tau_g"], r["tau_v"]))
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SWEEP_FIELDNAMES)
        w.writeheader()
        w.writerows(rows_out)
    data_vol.commit()

    # Pareto front: non-dominated on (efficiency MAX, fake_rate MIN, wall_seconds MIN).
    complete_rows = [
        r for r in rows_out
        if r["efficiency"] is not None and r["fake_rate"] is not None
        and r["wall_seconds"] is not None
    ]
    pareto = [r for r in complete_rows if not _is_dominated(r, complete_rows)]
    pareto.sort(key=lambda r: -r["efficiency"])

    print(f"\n=== Pareto sweep complete ===")
    print(f"Grid points: {len(grid)}  completed: {len(rows_out)}  "
          f"failed: {len(grid) - len(rows_out)}")
    print(f"Pareto front size: {len(pareto)}")
    print(f"\nPareto front (eff↑, fake↓, wall↓):")
    for r in pareto:
        print(
            f"  tau_g={r['tau_g']:.3f}  tau_v={r['tau_v']:.3f}  "
            f"eff={r['efficiency']:.2f}%  fake={r['fake_rate']:.3f}%  "
            f"dup_post={r['duplicate_rate_post_ambi']:.3f}%  "
            f"t/evt={r['runtime_per_event_s']:.2f}s  wall={r['wall_seconds']:.1f}s"
        )

    return {
        "grid_points": len(grid),
        "completed": len(rows_out),
        "failed": len(grid) - len(rows_out),
        "n_pareto": len(pareto),
        "csv_path": out_csv,
        "pareto": pareto,
    }


@app.local_entrypoint()
def show_sample():
    """Print schema and sample rows from one expanded parquet file."""
    import json
    path = f"{DATA_PATH}/results/train32/expanded/expanded_event000000000.parquet"
    pilot = "/data/results/pilot_1786524194"

    print("=== layer-volumes.csv ===")
    lv = peek_csv.remote(f"{pilot}/layer-volumes.csv", 30)
    print(f"Columns: {lv['columns']}")
    for r in lv["rows"]:
        print(f"  vol={r.get('volume_id')} layer={r.get('layer_id')} {r}")


@app.local_entrypoint()
def main():
    print("Step 1: Building patched ACTS (full, with Python bindings)...")
    build_result = build_acts.remote(target="all", python=True)
    print(f"Build result: {build_result}")

    print("\nStep 2: Running unit tests...")
    test_result = run_unit_tests.remote()
    print(f"Test result: {test_result}")

    print("\nStep 3: Verifying ROOT branches on 1 event...")
    verify_result = verify_root_branches.remote(event_id=0)
    print(f"\nVerification result: {verify_result}")
