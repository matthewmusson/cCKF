"""
Modal app for running the ACTS CKF pipeline on ColliderML data.

Provides the same functionality as the NERSC/shifter-based pipeline,
but runs entirely on Modal infrastructure. Uses the same ghcr.io container
image with ODD v5.0.0 geometry built from source.

Usage:
    # One-time: seed official ODD v5 material maps from GitLab LFS (~14 MB)
    # Prefer this over generate_material_maps — ColliderML used these maps.
    modal run modal_acts.py::seed_material_maps

    # Alternative: generate material maps from scratch (~30 min, 100k events)
    modal run modal_acts.py::generate_material_maps --n-events 100000

    # Smoke test without edm4hep (particle gun + Fatras + digi + CKF)
    modal run modal_acts.py::run_full_chain --n-events 8

    # Run CKF benchmark on edm4hep files (needs input data on the volume)
    modal run modal_acts.py::run_ckf --config baseline --events 128

    # Stage 1: Seeding optimizer (xopt EI, already completed)
    modal run modal_acts.py::run_optimizer \\
        --opt-vars seeding --events-per-trial 8 --threads-per-trial 8 \\
        --max-parallel-trials 4 --n-seed-trials 8 --n-guided-trials 16

    # Stage 3 (legacy): NSGA-II CKF per seeding point
    modal run --detach modal_acts.py::app.launch_ckf_studies

    # Stage 4: Joint MO-TPE seeding+CKF (10D, 500 trials, 32/32 split)
    # (Optuna 4: TPESampler replaces removed MOTPESampler)
    modal run --detach modal_acts.py::run_joint_motpe_optimizer \\
        --n-trials 500 --events-per-trial 32 --threads-per-trial 8 \\
        --max-parallel-trials 4

    # Stage 4b: Re-evaluate Pareto configs on held-out [32, 64)
    modal run --detach modal_acts.py::validate_joint_motpe_eval \\
        --front 4d --max-parallel 8

    # Generate Pareto front plots (after optimization completes)
    modal run modal_acts.py::plot_pareto_fronts

    # Download results
    modal run modal_acts.py::app.download --output-dir ./results
"""

import modal

app = modal.App("surp-acts")

# Persistent volumes for data and results
data_vol = modal.Volume.from_name("surp-acts-data", create_if_missing=True)
DATA_PATH = "/data"

# Container paths (fixed inside the image)
ACTS_DIR = "/spack/opt/spack/linux-x86_64/acts-main-udwtnx3aoh5lh6s76slc2fzc5szhwe7y"
SPACK_PYTHON = "/spack/opt/spack/linux-x86_64/python-3.13.11-awxtqzerpdzhatylv3uagd35ebciqs3o"
ODD_INSTALL = "/opt/ODD_v5/install/share/OpenDataDetector"

image = (
    modal.Image.from_dockerfile("Dockerfile.modal", add_python="3.13")
    .run_commands(
        "pip install torch --index-url https://download.pytorch.org/whl/cpu",
    )
    .pip_install(
        "xopt",
        "uproot",
        "awkward",
        "pyyaml",
        "pandas",
        "optuna",
        "optuna-integration[wandb]",
        "matplotlib",
        "wandb",
        "pyarrow",
        "numpy",
    )
    .add_local_dir("configs", remote_path="/app/configs")
    .add_local_file("digi_and_reco.py", remote_path="/app/digi_and_reco.py")
    .add_local_dir("utils", remote_path="/app/utils")
    .add_local_dir("scripts", remote_path="/app/scripts")
    .add_local_dir("acts_patches", remote_path="/app/acts_patches")
)


def _setup_acts_env():
    """Set up ACTS environment variables at runtime.

    Must be called at the start of every Modal function that uses ACTS.
    Adds /opt/acts_pypath to sys.path (for `import acts`), adds spack
    site-packages (for dd4hep, podio, ROOT bindings), and sources
    LD_LIBRARY_PATH + Geant4 data paths from /etc/acts_env.sh.
    """
    import glob
    import os
    import subprocess
    import sys
    from pathlib import Path

    # Add acts symlink dir to sys.path so `import acts` works.
    acts_pypath = "/opt/acts_pypath"
    if acts_pypath not in sys.path:
        sys.path.insert(0, acts_pypath)

    # Add spack site-packages dirs (dd4hep, podio, ROOT, numpy, etc.)
    spack_paths_file = "/etc/spack_site_packages"
    if os.path.exists(spack_paths_file):
        with open(spack_paths_file) as f:
            for line in f:
                p = line.strip()
                if p and p not in sys.path:
                    sys.path.append(p)

    # Source LD_LIBRARY_PATH, Geant4 data paths, DD4hep vars
    env_file = "/etc/acts_env.sh"
    if os.path.exists(env_file):
        result = subprocess.run(
            ["bash", "-c", f"source {env_file} && env"],
            capture_output=True, text=True,
        )
        for line in result.stdout.splitlines():
            if "=" in line:
                key, _, val = line.partition("=")
                os.environ[key] = val

    # acts_env.sh overwrites LD_LIBRARY_PATH with spack paths — keep ODD
    # factory libs first so getOpenDataDetector can find libOpenDataDetector.so
    odd_lib = "/opt/ODD_v5/install/lib"
    ld = os.environ.get("LD_LIBRARY_PATH", "")
    if odd_lib not in ld.split(":"):
        os.environ["LD_LIBRARY_PATH"] = (
            f"{odd_lib}:{ld}" if ld else odd_lib
        )

    os.environ["ODD_PATH"] = ODD_INSTALL

    # edm4hep Frame reads segfault without dictionaries + include path.
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

    # PyROOT lives under $ROOTSYS/lib, not site-packages. Modal's Python needs
    # that path explicitly (spack python gets it via this_acts_withdeps).
    root_candidates = []
    root_dir = os.environ.get("ROOT_DIR") or os.environ.get("ROOTSYS")
    if root_dir:
        root_candidates.append(Path(root_dir) / "lib")
        root_candidates.append(Path(root_dir) / "lib" / "root")
    root_candidates.extend(spack.glob("root-*/lib"))
    root_candidates.extend(spack.glob("root-*/lib/root"))
    for candidate in root_candidates:
        if candidate.is_dir() and str(candidate) not in sys.path:
            sys.path.append(str(candidate))

    # Import acts first — initializes cppyy/ROOT bindings used by the stack.
    try:
        import acts  # noqa: F401
    except Exception as exc:
        print(f"WARNING: acts import during env setup failed: {exc}")

    # Load edm4hep/podio shared libs into the process (dictionaries + .rootmap).
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
            except OSError as exc:
                print(f"WARNING: failed to load {lib}: {exc}")
    print(f"Preloaded {len(loaded_libs)} edm4hep/podio shared libraries")

    # Prefer gSystem.Load so ROOT registers dictionaries properly.
    try:
        import ROOT

        for lib in loaded_libs:
            ROOT.gSystem.Load(lib)
        import edm4hep  # noqa: F401

        print("edm4hep dictionaries: loaded via ROOT")
    except Exception as exc:
        print(f"WARNING: ROOT/edm4hep python preload failed: {exc}")
        # Last resort: try edm4hep module alone (may still register dicts).
        try:
            import edm4hep  # noqa: F401

            print("edm4hep module imported without ROOT")
        except Exception as exc2:
            print(f"WARNING: edm4hep import failed: {exc2}")


# ─── Material Maps ────────────────────────────────────────────────────

# Official ODD v5.0.0 LFS object (GitHub migration left v5 LFS on GitLab only).
# Source: https://gitlab.cern.ch/acts/OpenDataDetector/-/tree/v5.0.0/data
_ODD_V5_MATERIAL_MAP_OID = (
    "aa8c168f8046c1b252e41af030b53787c7cf59b86cfb3ab49ada656a97ec883a"
)
_ODD_V5_MATERIAL_MAP_SIZE = 13984935
_ODD_V5_LFS_BATCH = (
    "https://gitlab.cern.ch/acts/OpenDataDetector.git/info/lfs/objects/batch"
)


@app.function(
    image=image,
    volumes={DATA_PATH: data_vol},
    cpu=1,
    memory=2048,
    timeout=600,
)
def seed_material_maps():
    """Download official odd-material-maps.root for ODD v5.0.0 from GitLab LFS.

    Prefer this over generate_material_maps(): ColliderML was produced with the
    published v5 maps. Regenerating from Geant4 can differ slightly in binning
    / statistics and is much slower (~30 min vs seconds).
    """
    import json
    import os
    import urllib.request

    dest = f"{DATA_PATH}/odd-material-maps.root"
    if os.path.exists(dest) and os.path.getsize(dest) == _ODD_V5_MATERIAL_MAP_SIZE:
        print(f"Already present: {dest} ({os.path.getsize(dest)} bytes)")
        return dest

    body = json.dumps(
        {
            "operation": "download",
            "transfer": ["basic"],
            "objects": [
                {
                    "oid": _ODD_V5_MATERIAL_MAP_OID,
                    "size": _ODD_V5_MATERIAL_MAP_SIZE,
                }
            ],
        }
    ).encode()
    req = urllib.request.Request(
        _ODD_V5_LFS_BATCH,
        data=body,
        headers={
            "Accept": "application/vnd.git-lfs+json",
            "Content-Type": "application/vnd.git-lfs+json",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        payload = json.load(resp)

    href = payload["objects"][0]["actions"]["download"]["href"]
    print(f"Downloading ODD v5 material maps ({_ODD_V5_MATERIAL_MAP_SIZE} bytes)...")
    urllib.request.urlretrieve(href, dest)

    size = os.path.getsize(dest)
    with open(dest, "rb") as f:
        magic = f.read(4)
    if magic != b"root":
        raise RuntimeError(f"Downloaded file is not a ROOT file (magic={magic!r})")
    if size != _ODD_V5_MATERIAL_MAP_SIZE:
        raise RuntimeError(
            f"Size mismatch: got {size}, expected {_ODD_V5_MATERIAL_MAP_SIZE}"
        )

    data_vol.commit()
    print(f"Saved {dest} ({size} bytes)")
    return dest


@app.function(
    image=image,
    volumes={DATA_PATH: data_vol},
    cpu=8,
    memory=16384,
    timeout=3600,
)
def generate_material_maps(n_events: int = 100000):
    """Generate odd-material-maps.root from scratch using Geant4 + ACTS.

    Prefer seed_material_maps() for ColliderML benchmarking. Use this only if
    you need to regenerate maps (e.g. custom geometry).

    Step 1: Material recording — shoots particles through ODD geometry,
            records material interactions.
    Step 2: Material mapping — maps recorded material onto tracking surfaces
            using the binning config from ODD.
    """
    import os
    import time

    _setup_acts_env()

    import acts
    import acts.examples
    from acts.examples.odd import getOpenDataDetector, getOpenDataDetectorDirectory
    from acts.examples import Sequencer

    u = acts.UnitConstants
    geoDir = getOpenDataDetectorDirectory()
    output_dir = f"{DATA_PATH}/material_generation"
    os.makedirs(output_dir, exist_ok=True)

    print(f"=== Step 1: Material Recording ({n_events} events) ===")
    t0 = time.time()

    detector = getOpenDataDetector(odd_dir=geoDir)
    trackingGeometry = detector.trackingGeometry()

    s = Sequencer(
        events=n_events,
        numThreads=1,
        logLevel=acts.logging.INFO,
    )

    rnd = acts.examples.RandomNumbers(seed=42)

    # Particle gun: uniform in eta, 1-10 GeV, 10 particles per event
    vtxGen = acts.examples.GaussianVertexGenerator(
        stddev=acts.Vector4(0, 0, 0, 0),
        mean=acts.Vector4(0, 0, 0, 0),
    )
    pGen = acts.examples.ParametricParticleGenerator(
        p=(1 * u.GeV, 10 * u.GeV),
        eta=(-4.0, 4.0),
        numParticles=10,
        etaUniform=True,
    )
    evGen = acts.examples.EventGenerator(
        level=acts.logging.INFO,
        generators=[
            acts.examples.EventGenerator.Generator(
                multiplicity=acts.examples.FixedMultiplicityGenerator(n=1),
                vertex=vtxGen,
                particles=pGen,
            )
        ],
        outputParticles="particles_input",
        randomNumbers=rnd,
    )
    s.addReader(evGen)

    # Geant4 material recording
    from acts.examples.geant4 import Geant4MaterialRecording

    g4MatRec = Geant4MaterialRecording(
        level=acts.logging.INFO,
        detector=detector,
        inputParticles="particles_input",
        outputMaterialTracks="material-tracks",
        randomNumbers=rnd,
    )
    s.addAlgorithm(g4MatRec)

    # Write material tracks
    s.addWriter(
        acts.examples.root.RootMaterialTrackWriter(
            level=acts.logging.INFO,
            inputMaterialTracks="material-tracks",
            filePath=f"{output_dir}/geant4_material_tracks.root",
            prePostStep=True,
            recalculateTotals=True,
        )
    )

    s.run()
    print(f"Material recording done in {time.time() - t0:.1f}s")

    print("=== Step 2: Material Mapping ===")
    t1 = time.time()

    # Reload geometry with the mapping config as material decorator
    mapping_config = geoDir / "config/odd-material-mapping-config.json"
    matDeco = acts.IMaterialDecorator.fromFile(str(mapping_config))
    detector2 = getOpenDataDetector(odd_dir=geoDir, materialDecorator=matDeco)
    trackingGeometry2 = detector2.trackingGeometry()

    s2 = Sequencer(
        events=n_events,
        numThreads=1,
        logLevel=acts.logging.INFO,
    )

    # Read recorded material tracks
    s2.addReader(
        acts.examples.root.RootMaterialTrackReader(
            level=acts.logging.INFO,
            fileList=[f"{output_dir}/geant4_material_tracks.root"],
            outputMaterialTracks="material-tracks",
        )
    )

    # Surface material mapper (uses Navigator + Propagator to project
    # material steps onto tracking surfaces)
    stepper = acts.StraightLineStepper()
    navigator = acts.Navigator(
        trackingGeometry=trackingGeometry2,
    )
    propagator = acts.Propagator(stepper, navigator)

    mapper = acts.SurfaceMaterialMapper(
        config=acts.SurfaceMaterialMapper.Config(propagator=propagator),
        level=acts.logging.INFO,
    )

    mapName = "odd-material-maps"
    materialMapping = acts.examples.MaterialMapping(
        level=acts.logging.INFO,
        inputMaterialTracks="material-tracks",
        mappingMaterialCollection="material-map",
        materialSurfaceMapper=mapper,
        materialWriters=[
            acts.examples.root.RootMaterialWriter(
                level=acts.logging.INFO,
                filePath=f"{output_dir}/{mapName}_map.root",
            ),
        ],
    )
    s2.addAlgorithm(materialMapping)

    s2.run()
    print(f"Material mapping done in {time.time() - t1:.1f}s")

    # Copy the material map to the canonical location
    import shutil
    final_path = f"{DATA_PATH}/odd-material-maps.root"
    shutil.copy2(f"{output_dir}/{mapName}_map.root", final_path)
    data_vol.commit()

    print(f"Material maps saved to {final_path}")
    print(f"Total time: {time.time() - t0:.1f}s")


# ─── Full Chain (Sim + Digi + Reco) ───────────────────────────────────


@app.function(
    image=image,
    volumes={DATA_PATH: data_vol},
    cpu=8,
    memory=32768,
    timeout=7200,
)
def run_full_chain(
    n_events: int = 128,
    n_particles: int = 2,
    pt_min: float = 1.0,
    pt_max: float = 10.0,
    config: str = "baseline",
    config_overrides: str = None,
):
    """Run the full chain: particle gun → Fatras sim → digi → CKF → ambiguity.

    This bypasses the EDM4hep reader/converter entirely — Fatras puts sim hits
    directly on the whiteboard, then digi+reco proceeds as normal. Use this
    when you don't have pre-generated edm4hep.root files.

    For running on pre-generated ColliderML edm4hep files, use run_ckf() instead.
    """
    import os
    import sys
    import json
    import time
    import types
    import yaml

    _setup_acts_env()
    sys.path.insert(0, "/app")

    import acts
    import acts.examples
    from acts.examples.odd import getOpenDataDetector, getOpenDataDetectorDirectory
    from acts.examples import Sequencer
    from acts.examples.simulation import (
        addParticleGun, addFatras, addDigitization,
        addSimParticleSelection, addDigiParticleSelection,
        MomentumConfig, EtaConfig, ParticleConfig, ParticleSelectorConfig,
    )
    from acts.examples.reconstruction import (
        addSeeding, addCKFTracks, addAmbiguityResolution,
        addTrackWriters, SeedingAlgorithm, SeedFinderConfigArg,
        TrackSelectorConfig, CkfConfig, AmbiguityResolutionConfig,
    )
    from utils.app_logging import TimingRecorder

    u = acts.UnitConstants
    geoDir = getOpenDataDetectorDirectory()

    material_map = f"{DATA_PATH}/odd-material-maps.root"
    if not os.path.exists(material_map):
        raise FileNotFoundError(
            "Material maps not found. Run generate_material_maps first."
        )

    # Symlink material maps into ODD data dir
    odd_data_dir = f"{ODD_INSTALL}/data"
    os.makedirs(odd_data_dir, exist_ok=True)
    map_link = f"{odd_data_dir}/odd-material-maps.root"
    if not os.path.exists(map_link):
        os.symlink(material_map, map_link)

    oddMaterialDeco = acts.IMaterialDecorator.fromFile(material_map)
    detector = getOpenDataDetector(odd_dir=geoDir, materialDecorator=oddMaterialDeco)
    trackingGeometry = detector.trackingGeometry()
    field = detector.field

    # Load CKF config
    config_path = f"/app/configs/{config}.yaml"
    with open(config_path) as f:
        cfg_dict = yaml.safe_load(f)
    if config_overrides:
        cfg_dict.update(json.loads(config_overrides))
    cfg_dict["events"] = n_events
    cfg = types.SimpleNamespace(**cfg_dict)

    run_id = f"fullchain_{config}_{int(time.time())}"
    output_dir = f"{DATA_PATH}/results/{run_id}"
    os.makedirs(output_dir, exist_ok=True)
    from pathlib import Path
    output_path = Path(output_dir)

    print(f"=== Full Chain: {n_events} events, {n_particles} particles, "
          f"pT {pt_min}-{pt_max} GeV ===")
    t0 = time.time()

    oddDigiConfig = geoDir / "config/odd-digi-smearing-config.json"
    oddSeedingSel = geoDir / "config/odd-seeding-config.json"

    s = Sequencer(
        events=n_events,
        numThreads=1,
        logLevel=acts.logging.INFO,
        outputDir=str(output_dir),
    )
    rnd = acts.examples.RandomNumbers(seed=getattr(cfg, "seed", 42))

    # 1. Particle gun
    addParticleGun(
        s,
        MomentumConfig(pt_min * u.GeV, pt_max * u.GeV, transverse=True),
        EtaConfig(-3.0, 3.0, uniform=True),
        ParticleConfig(n_particles, acts.PdgParticle.eMuon, randomizeCharge=True),
        rnd=rnd,
    )

    # 2. Fatras simulation
    addFatras(
        s,
        trackingGeometry,
        field,
        enableInteractions=True,
        rnd=rnd,
    )

    # 3. Particle selection
    addSimParticleSelection(
        s,
        ParticleSelectorConfig(
            rho=(0.0, 1080 * u.mm),
            absZ=(0.0, 3.03 * u.m),
            pt=(150 * u.MeV, None),
        ),
    )

    # 4. Digitization
    addDigitization(
        s,
        trackingGeometry,
        field,
        digiConfigFile=oddDigiConfig,
        rnd=rnd,
        logLevel=acts.logging.INFO,
    )

    # 5. Digi particle selection
    def make_geoid(vol):
        geoid = acts.GeometryIdentifier()
        geoid.volume = vol
        return geoid

    measurementCounter = acts.examples.ParticleSelector.MeasurementCounter()
    measurementCounter.addCounter(
        [make_geoid(16), make_geoid(17), make_geoid(18)],
        3,
        2**31 - 1,
    )
    addDigiParticleSelection(
        s,
        ParticleSelectorConfig(
            rho=(0.0, 24 * u.mm),
            absZ=(0.0, 1.0 * u.m),
            eta=(-3.0, 3.0),
            pt=(0.999 * u.GeV, None),
            measurements=(6, None),
            removeNeutral=True,
            removeSecondaries=False,
            nMeasurementsGroupMin=measurementCounter,
        ),
    )

    # 6. Seeding
    seed_minPt = getattr(cfg, "seed_minPt", 0.5)
    seed_impactMax = getattr(cfg, "seed_impactMax", 3.0)
    num_seeds_per_spm = getattr(cfg, "num_seeds_per_spm", 40)

    addSeeding(
        s,
        trackingGeometry,
        field,
        seedingAlgorithm=SeedingAlgorithm.Default,
        seedFinderConfigArg=SeedFinderConfigArg(
            minPt=seed_minPt * u.GeV,
            impactMax=seed_impactMax * u.mm,
            maxSeedsPerSpM=num_seeds_per_spm,
        ),
        geoSelectionConfigFile=oddSeedingSel,
        logLevel=acts.logging.INFO,
    )

    # 7. CKF track finding
    addCKFTracks(
        s,
        trackingGeometry,
        field,
        trackSelectorConfig=TrackSelectorConfig(
            pt=(getattr(cfg, "ckf_ptMin", 0.7) * u.GeV, None),
            loc0=(-4.0 * u.mm, 4.0 * u.mm),
            nMeasurementsMin=getattr(cfg, "ckf_nMeasurementsMin", 6),
            maxHolesAndOutliers=getattr(cfg, "ckf_maxHolesAndOutliers", 3),
        ),
        ckfConfig=CkfConfig(
            chi2CutOffMeasurement=getattr(cfg, "ckf_chi2CutOffMeasurement", 15.0),
            chi2CutOffOutlier=getattr(cfg, "ckf_chi2CutOffOutlier", 25.0),
            numMeasurementsCutOff=getattr(cfg, "ckf_numMeasurementsCutOff", 1),
            seedDeduplication=True,
            stayOnSeed=True,
        ),
        twoWay=True,
        outputDirRoot=None,
        writeTrackSummary=False,
        writeTrackStates=False,
        writePerformance=False,
        writeCovMat=False,
        logLevel=acts.logging.INFO,
    )

    # CKF performance writers
    addTrackWriters(
        s,
        name="ckf",
        tracks="tracks",
        outputDirCsv=None,
        outputDirRoot=output_dir,
        writeSummary=False,
        writeStates=False,
        writeFitterPerformance=False,
        writeFinderPerformance=True,
        logLevel=acts.logging.INFO,
    )

    # 8. Ambiguity resolution
    addAmbiguityResolution(
        s,
        config=AmbiguityResolutionConfig(
            maximumSharedHits=3,
            maximumIterations=1000000,
            nMeasurementsMin=6,
        ),
        outputDirRoot=None,
        writeTrackSummary=False,
        writeTrackStates=False,
        writePerformance=False,
        writeCovMat=False,
    )

    addTrackWriters(
        s,
        name="ambi",
        tracks="tracks",
        outputDirCsv=None,
        outputDirRoot=output_dir,
        writeSummary=False,
        writeStates=False,
        writeFitterPerformance=False,
        writeFinderPerformance=True,
        logLevel=acts.logging.INFO,
    )

    s.run()
    wall_time = time.time() - t0

    # Parse metrics
    metrics = {"config": config, "events": n_events, "wall_time": wall_time}
    try:
        import uproot
        perf_file = f"{output_dir}/performance_finding_ckf.root"
        if os.path.exists(perf_file):
            with uproot.open(perf_file) as rh:
                metrics["efficiency"] = float(
                    rh["eff_particles"].member("fElements")[0] * 100
                )
                metrics["fakerate"] = float(
                    rh["fakeratio_tracks"].member("fElements")[0] * 100
                )
                metrics["duplicaterate"] = float(
                    rh["duplicateratio_tracks"].member("fElements")[0] * 100
                )
    except Exception as e:
        print(f"Warning: could not parse metrics: {e}")

    data_vol.commit()

    print(f"\n=== Results ({wall_time:.1f}s) ===")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    return metrics


# ─── CKF Reconstruction ───────────────────────────────────────────────


@app.function(
    image=image,
    volumes={DATA_PATH: data_vol},
    cpu=1,
    memory=4096,
    timeout=300,
)
def check_arrow():
    """Probe whether ACTS Arrow/Parquet Python bindings are available."""
    import json
    import sys

    _setup_acts_env()
    report = {"python": sys.version}

    for name in (
        "acts.examples.arrow",
        "acts.examples.parquet",
        "acts.arrow",
        "acts.plugins.arrow",
    ):
        try:
            mod = __import__(name, fromlist=["*"])
            report[name] = {
                "ok": True,
                "attrs_sample": [a for a in dir(mod) if not a.startswith("_")][:40],
            }
        except Exception as e:
            report[name] = {"ok": False, "error": f"{type(e).__name__}: {e}"}

    for path, symbol in (
        ("acts.examples.arrow", "ParquetReader"),
        ("acts.examples.arrow", "ColliderMLRelease1InputConverter"),
        ("acts.examples.arrow", "ParquetWriter"),
    ):
        key = f"{path}.{symbol}"
        try:
            mod = __import__(path, fromlist=[symbol])
            getattr(mod, symbol)
            report[key] = True
        except Exception as e:
            report[key] = f"{type(e).__name__}: {e}"

    try:
        import acts.examples as ex

        report["acts.examples_arrowish"] = [
            x
            for x in dir(ex)
            if "arrow" in x.lower() or "parquet" in x.lower()
        ]
    except Exception as e:
        report["acts.examples"] = str(e)

    print(json.dumps(report, indent=2))
    return report


@app.function(
    image=image,
    volumes={DATA_PATH: data_vol},
    cpu=8,
    memory=65536,  # pu200 ColliderML events are memory-heavy (~200k hits/event)
    timeout=7200,
)
def run_ckf(config: str = "baseline", events: int = 128,
            input_file: str = None, config_overrides: str = None):
    """Run the full digi+reco chain and return performance metrics.

    Args:
        config: Name of a YAML config in configs/ (without .yaml extension)
        events: Number of events to process
        input_file: Path to edm4hep.root (relative to data volume).
                    Default: events/edm4hep.root
        config_overrides: JSON string of config overrides, e.g.
                          '{"ckf_chi2CutOffMeasurement": 20.0}'

    Returns:
        Dict with efficiency, fakerate, duplicaterate, runtime
    """
    import os
    import sys
    import json
    import time
    import types
    import yaml

    _setup_acts_env()
    sys.path.insert(0, "/app")

    material_map = f"{DATA_PATH}/odd-material-maps.root"
    if not os.path.exists(material_map):
        raise FileNotFoundError(
            "Material maps not found. Run generate_material_maps first."
        )

    # Symlink material maps into ODD data dir so digi_and_reco.py finds them
    odd_data_dir = f"{ODD_INSTALL}/data"
    os.makedirs(odd_data_dir, exist_ok=True)
    map_link = f"{odd_data_dir}/odd-material-maps.root"
    if not os.path.exists(map_link):
        os.symlink(material_map, map_link)

    # Load config
    config_path = f"/app/configs/{config}.yaml"
    with open(config_path) as f:
        cfg_dict = yaml.safe_load(f)

    # Apply overrides
    if config_overrides:
        overrides = json.loads(config_overrides)
        cfg_dict.update(overrides)

    cfg_dict["events"] = events
    if "skip" not in cfg_dict:
        cfg_dict["skip"] = 0

    # Convert dict to namespace (what digi_and_reco expects)
    cfg = types.SimpleNamespace(**cfg_dict)
    if not hasattr(cfg, "seed"):
        cfg.seed = 42
    if not hasattr(cfg, "threads"):
        cfg.threads = 1

    # Resolve input file
    if input_file:
        edm4hep_path = f"{DATA_PATH}/{input_file}"
    else:
        edm4hep_path = f"{DATA_PATH}/events/edm4hep.root"

    if not os.path.exists(edm4hep_path):
        raise FileNotFoundError(
            f"Input file not found: {edm4hep_path}. "
            "Run generate_events first, or specify --input-file."
        )

    # Set up output directory
    run_id = f"ckf_{config}_{int(time.time())}"
    output_dir = f"{DATA_PATH}/results/{run_id}"
    os.makedirs(output_dir, exist_ok=True)

    skip = int(getattr(cfg, "skip", 0) or 0)
    print(f"=== Running CKF: config={config}, events={events}, skip={skip} ===")
    print(f"Input: {edm4hep_path}")
    print(f"Event range: [{skip}, {skip + events})")
    print(f"Output: {output_dir}")
    t0 = time.time()

    # Import and run the pipeline
    import acts
    from digi_and_reco import setup_acts_reconstruction
    from utils.app_logging import TimingRecorder

    rnd = acts.examples.RandomNumbers(seed=cfg.seed)
    timer = TimingRecorder(output_dir)

    from pathlib import Path

    with timer.record("Setup"):
        s = setup_acts_reconstruction(
            Path(edm4hep_path), Path(output_dir), cfg, rnd
        )

    with timer.record("Sequencer"):
        s.run()

    wall_time = time.time() - t0

    # Parse ACTS timing (tsv or csv depending on ACTS/Sequencer version)
    import pathlib

    acts_timing = pathlib.Path(output_dir) / "timing.tsv"
    if not acts_timing.exists():
        acts_timing = pathlib.Path(output_dir) / "timing.csv"
    if acts_timing.exists():
        timer.parse_acts_timing(acts_timing)
    timer.write_report()

    # Parse performance metrics (prefer post-ambi particle-level for opt objectives).
    # Use PyROOT: Modal's uproot often fails membership/extraction on these
    # TVectorT scalars even when the objects are present and valid.
    metrics = {
        "config": config,
        "events": events,
        "wall_time": wall_time,
        "wall_per_event": wall_time / max(events, 1),
        "threads": getattr(cfg, "threads", 1),
        "output_dir": output_dir,
    }

    def _scalar_pyroot(tfile, name: str) -> float:
        obj = tfile.Get(name)
        if not obj:
            raise KeyError(name)
        # TVectorT<float> with one element
        return float(obj[0])

    def _load_perf(path: str, prefix: str) -> None:
        import ROOT

        tfile = ROOT.TFile.Open(path)
        if not tfile or tfile.IsZombie():
            raise RuntimeError(f"cannot open {path}")
        try:
            metrics[f"{prefix}_efficiency"] = (
                _scalar_pyroot(tfile, "eff_particles") * 100
            )
            metrics[f"{prefix}_fakerate"] = (
                _scalar_pyroot(tfile, "fakeratio_particles") * 100
            )
            metrics[f"{prefix}_duplicaterate"] = (
                _scalar_pyroot(tfile, "duplicateratio_particles") * 100
            )
            metrics[f"{prefix}_track_efficiency"] = (
                _scalar_pyroot(tfile, "eff_tracks") * 100
            )
            metrics[f"{prefix}_track_fakerate"] = (
                _scalar_pyroot(tfile, "fakeratio_tracks") * 100
            )
            metrics[f"{prefix}_track_duplicaterate"] = (
                _scalar_pyroot(tfile, "duplicateratio_tracks") * 100
            )
        finally:
            tfile.Close()

    try:
        ckf_perf = f"{output_dir}/performance_finding_ckf.root"
        ambi_perf = f"{output_dir}/performance_finding_ambi.root"
        if os.path.exists(ckf_perf):
            _load_perf(ckf_perf, "ckf")
        if os.path.exists(ambi_perf):
            _load_perf(ambi_perf, "ambi")

        # Primary metrics: particle-level efficiency (ε_DM) + track-level fake rate (f_DM).
        # This matches the standard definitions in HEPTv2, ACORN, etc:
        #   ε_DM = |{particles with ≥1 DM-matched track}| / |{reconstructable particles}|
        #   f_DM = |{reco tracks not DM-matched}| / |{all reco tracks}|
        if "ambi_efficiency" in metrics:
            metrics["efficiency"] = metrics["ambi_efficiency"]
            metrics["fakerate"] = metrics["ambi_track_fakerate"]
            metrics["duplicaterate"] = metrics["ambi_track_duplicaterate"]
        elif "ckf_efficiency" in metrics:
            metrics["efficiency"] = metrics["ckf_efficiency"]
            metrics["fakerate"] = metrics["ckf_track_fakerate"]
            metrics["duplicaterate"] = metrics["ckf_track_duplicaterate"]
        else:
            print(
                f"Warning: no efficiency keys parsed "
                f"(ckf exists={os.path.exists(ckf_perf)}, "
                f"ambi exists={os.path.exists(ambi_perf)})"
            )

        # ACTS per-algorithm timing + seed→tracks sum (excludes digi/IO/truth/writers)
        timing_breakdown = _load_acts_timing_breakdown(acts_timing)
        metrics["acts_timing"] = timing_breakdown
        metrics["t_seed_to_tracks"] = _sum_seed_to_tracks(timing_breakdown)
        metrics["runtime"] = float(
            timing_breakdown.get("TrackFindingAlgorithm", wall_time / max(events, 1))
        )
    except Exception as e:
        print(f"Warning: could not parse metrics: {e}")
        metrics.setdefault("runtime", wall_time)
        metrics.setdefault("t_seed_to_tracks", float("nan"))

    data_vol.commit()

    print(f"\n=== Results ===")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    return metrics


# ACTS algorithms counted in "seeding → final tracks out" (per-event seconds).
# Excludes digi/IO, truth matching, and ROOT performance writers.
SEED_TO_TRACKS_ALGOS = (
    "SpacePointMaker",
    "SeedingAlgorithm",
    "TrackParamsEstimationAlgorithm",
    "SeedsToPrototracks",
    "PrototracksToTracks",
    "TrackFindingAlgorithm",
    "GreedyAmbiguityResolutionAlgorithm",
)


def _load_acts_timing_breakdown(path) -> dict[str, float]:
    """Parse ACTS Sequencer timing.tsv/.csv → {short_name: time_perevent_s}."""
    import csv
    from pathlib import Path

    path = Path(path)
    if not path.exists():
        return {}
    text = path.read_text()
    delim = "\t" if "\t" in text.splitlines()[0] else ","
    rows = list(csv.DictReader(text.splitlines(), delimiter=delim))
    if not rows:
        return {}
    name_col = "identifier" if "identifier" in rows[0] else "name"
    time_col = "time_perevent_s"
    if time_col not in rows[0]:
        time_col = "time_total_s" if "time_total_s" in rows[0] else None
    if time_col is None:
        return {}
    out: dict[str, float] = {}
    for row in rows:
        raw = row.get(name_col, "")
        # "Algorithm:SeedingAlgorithm" → "SeedingAlgorithm"
        short = raw.split(":")[-1] if raw else ""
        if not short:
            continue
        # Sum duplicate names (e.g. multiple TrackTruthMatcher rows) separately
        # by keeping the first occurrence of reco algs; for seed→tracks we only
        # match the unique algorithm names listed in SEED_TO_TRACKS_ALGOS.
        if short not in out:
            out[short] = float(row[time_col])
        else:
            # Rare: same short name twice — accumulate
            out[short] += float(row[time_col])
    return out


def _sum_seed_to_tracks(breakdown: dict[str, float]) -> float:
    return float(sum(breakdown.get(name, 0.0) for name in SEED_TO_TRACKS_ALGOS))


# ─── Bayesian Optimizer ────────────────────────────────────────────────


DEFAULT_PARAMS = {
    "num_seeds_per_spm": 40,
    "seed_minPt": 0.5,
    "seed_impactMax": 3.0,
    "seed_sigmaScattering": 5,
    "ckf_chi2CutOffMeasurement": 15.0,
    "ckf_chi2CutOffOutlier": 25.0,
    "ckf_numMeasurementsCutOff": 1,
    "ckf_nMeasurementsMin": 6,
    "ckf_maxHolesAndOutliers": 3,
    "ckf_ptMin": 0.7,
    "threads": 8,
    "ckf_finding_performance": True,
    "ambi_finding_performance": True,
}

# xopt VOCS requires list[2] bounds (not tuples)
VARIABLE_BOUNDS = {
    "num_seeds_per_spm": [1, 80],
    "seed_minPt": [0.3, 2.0],
    "seed_impactMax": [1.0, 10.0],
    "seed_sigmaScattering": [2.0, 10.0],
    "ckf_chi2CutOffMeasurement": [5.0, 30.0],
    "ckf_chi2CutOffOutlier": [10.0, 50.0],
    "ckf_numMeasurementsCutOff": [1, 5],
    "ckf_nMeasurementsMin": [4, 9],
    "ckf_maxHolesAndOutliers": [1, 6],
    "ckf_ptMin": [0.3, 1.2],
}

INT_PARAMS = {
    "num_seeds_per_spm",
    "ckf_numMeasurementsCutOff",
    "ckf_nMeasurementsMin",
    "ckf_maxHolesAndOutliers",
    "threads",
}


@app.function(
    image=image,
    volumes={DATA_PATH: data_vol},
    cpu=2,
    memory=8192,
    timeout=86400,
)
def run_optimizer(
    n_seed_trials: int = 8,
    n_guided_trials: int = 16,
    events_per_trial: int = 8,
    opt_vars: str = "seeding",
    fake_weight: float = 1.0,
    max_parallel_trials: int = 4,
    threads_per_trial: int = 8,
    input_file: str = None,
    stage_name: str = "seeding",
):
    """Bayesian optimization over ACTS parameters (Modal).

    Stage 1 (default): optimize seeding only; CKF knobs frozen at baseline
    (including numMeasurementsCutOff=1). Objectives are post-ambi particle
    efficiency / fake rate:

        score = -(efficiency - fake_weight * fakerate)

    so minimizing score traces a soft eff–fake tradeoff (Pareto proxy).
    Duplicate rate is logged as a sanity check (expect ~0 after ambi).

    Parallelism:
      - max_parallel_trials: concurrent Modal run_ckf containers
      - threads_per_trial: ACTS Sequencer event-level threads inside each trial
    """
    import json
    import math
    import os

    import pandas as pd
    import xopt
    from xopt.generators.bayesian import ExpectedImprovementGenerator
    from xopt.vocs import VOCS

    all_vars = list(VARIABLE_BOUNDS.keys())
    seeding_vars = ["num_seeds_per_spm", "seed_minPt", "seed_impactMax"]
    ckf_vars = [v for v in all_vars if v not in seeding_vars]

    if opt_vars == "all":
        active_vars = all_vars
    elif opt_vars == "seeding":
        active_vars = seeding_vars
    elif opt_vars == "ckf":
        active_vars = ckf_vars
    else:
        active_vars = [v.strip() for v in opt_vars.split(",") if v.strip()]

    # No hard fail constraint: a previous run marked all trials failed when
    # metrics parsing broke, which made ExpectedImprovement abort. Log fail
    # as an observable instead.
    vocs = VOCS(
        variables={k: VARIABLE_BOUNDS[k] for k in active_vars},
        objectives={"score": "MINIMIZE"},
        observables=["fail", "efficiency", "fakerate", "duplicaterate", "runtime"],
    )

    results_dir = f"{DATA_PATH}/optimizer/{stage_name}"
    os.makedirs(results_dir, exist_ok=True)
    csv_path = f"{results_dir}/optimization_history.csv"

    def evaluate_trial(params):
        full_params = DEFAULT_PARAMS.copy()
        full_params.update(params)
        full_params["threads"] = int(threads_per_trial)

        for p in INT_PARAMS:
            if p in full_params:
                full_params[p] = int(round(full_params[p]))

        # Freeze CKF branching for seeding stage
        if opt_vars == "seeding":
            full_params["ckf_numMeasurementsCutOff"] = 1

        overrides = json.dumps(full_params)
        print(f"  trial params: {full_params}")
        try:
            metrics = run_ckf.remote(
                config="baseline",
                events=events_per_trial,
                input_file=input_file,
                config_overrides=overrides,
            )
        except Exception as e:
            print(f"Trial failed: {e}")
            return {"score": 1.0, "fail": 1.0}

        if metrics is None or "efficiency" not in metrics:
            return {"score": 1.0, "fail": 1.0}

        eff = float(metrics["efficiency"])
        fake = float(metrics["fakerate"])
        dup = float(metrics.get("duplicaterate", 0.0))
        rt = float(metrics.get("runtime", metrics.get("wall_time", 0.0)))

        if any(math.isnan(v) or math.isinf(v) for v in [eff, fake, dup, rt]):
            return {"score": 1.0, "fail": 1.0}

        # Soft Pareto proxy: maximize eff - w*fake (no runtime term)
        # eff = particle-level ε_DM, fake = track-level f_DM (post-ambi)
        score = -(eff - fake_weight * fake)
        return {
            "score": score,
            "fail": 0.0,
            "efficiency": eff,
            "fakerate": fake,
            "duplicaterate": dup,
            "runtime": rt,
            "wall_time": float(metrics.get("wall_time", rt)),
        }

    from concurrent.futures import ThreadPoolExecutor

    # Thread pool (not process pool): nested evaluate_trial isn't picklable,
    # and run_ckf.remote is I/O-bound wait on Modal containers.
    executor = ThreadPoolExecutor(max_workers=max_parallel_trials)
    evaluator = xopt.Evaluator(
        function=evaluate_trial,
        executor=executor,
        max_workers=max_parallel_trials,
    )
    # Propose up to max_parallel_trials candidates per guided step so
    # Evaluator can fan out concurrent run_ckf.remote calls.
    generator = ExpectedImprovementGenerator(
        vocs=vocs, n_candidates=max_parallel_trials
    )
    X = xopt.Xopt(evaluator=evaluator, generator=generator, vocs=vocs)

    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        print(f"Resuming from {len(df)} previous trials at {csv_path}")
        X.data = df.copy()
        X.generator.add_data(df)

    print("=== Bayesian Optimization (Modal) ===")
    print(f"Stage: {stage_name}")
    print(f"Variables: {active_vars}")
    print(f"Events/trial: {events_per_trial}, threads/trial: {threads_per_trial}")
    print(f"Parallel trials: {max_parallel_trials}")
    print(f"Score: -(eff - {fake_weight}*fake)  [post-ambi particle metrics]")
    print(f"Trials — random: {n_seed_trials}, guided: {n_guided_trials}")

    # Batch random seed trials so max_workers can parallelize
    remaining = n_seed_trials
    batch_idx = 0
    while remaining > 0:
        batch = min(max_parallel_trials, remaining)
        batch_idx += 1
        print(
            f"\n--- Random batch {batch_idx} "
            f"({batch} trials, {remaining - batch} left) ---"
        )
        X.random_evaluate(batch)
        remaining -= batch
        df = X.data if isinstance(X.data, pd.DataFrame) else pd.DataFrame(X.data)
        df.to_csv(csv_path, index=False)
        data_vol.commit()

    # Guided steps: each step proposes n_candidates in parallel
    n_guided_steps = math.ceil(n_guided_trials / max_parallel_trials)
    for i in range(n_guided_steps):
        print(
            f"\n--- Guided step {i + 1}/{n_guided_steps} "
            f"(up to {max_parallel_trials} candidates) ---"
        )
        X.step()
        df = X.data if isinstance(X.data, pd.DataFrame) else pd.DataFrame(X.data)
        df.to_csv(csv_path, index=False)
        data_vol.commit()
        if len(df) >= n_seed_trials + n_guided_trials:
            break

    df = X.data if isinstance(X.data, pd.DataFrame) else pd.DataFrame(X.data)
    df["inv_score"] = -df["score"]
    df.to_csv(csv_path, index=False)

    with open(f"{results_dir}/xopt_state.json", "w") as f:
        f.write(X.model_dump_json(indent=2))
    data_vol.commit()

    # Best by score, plus report Pareto-ish top by efficiency among low-fake
    best = df.loc[df["inv_score"].idxmax()]
    print("\n=== Optimization Complete ===")
    print(f"Total trials: {len(df)}")
    print(f"Best by score:\n{best.to_dict()}")
    if "efficiency" in df.columns and "fakerate" in df.columns:
        print("\nAll trials (eff, fake, dup, runtime):")
        cols = [
            c
            for c in [
                "num_seeds_per_spm",
                "seed_minPt",
                "seed_impactMax",
                "efficiency",
                "fakerate",
                "duplicaterate",
                "runtime",
                "score",
            ]
            if c in df.columns
        ]
        print(df[cols].sort_values("efficiency", ascending=False).to_string())

    return best.to_dict()


# ─── Optuna NSGA-II Seeding Optimizer ──────────────────────────────────

SEEDING_PARAM_SPACE = {
    "num_seeds_per_spm": {"type": "int", "low": 1, "high": 80},
    "seed_minPt": {"type": "float", "low": 0.3, "high": 2.0},
    "seed_impactMax": {"type": "float", "low": 1.0, "high": 10.0},
    "seed_sigmaScattering": {"type": "float", "low": 2.0, "high": 10.0},
}

CKF_DEFAULTS_FOR_SEEDING = {
    "ckf_chi2CutOffMeasurement": 15.0,
    "ckf_chi2CutOffOutlier": 25.0,
    "ckf_numMeasurementsCutOff": 1,
    "ckf_nMeasurementsMin": 6,
    "ckf_maxHolesAndOutliers": 3,
    "ckf_ptMin": 0.7,
}


def _seeding_objective(trial, events_per_trial: int,
                       threads_per_trial: int, input_file: str | None):
    """Optuna objective for seeding optimization. Returns (eff, fake, dup, runtime)."""
    import json
    import math

    params = {}
    for name, spec in SEEDING_PARAM_SPACE.items():
        if spec["type"] == "float":
            params[name] = trial.suggest_float(name, spec["low"], spec["high"])
        else:
            params[name] = trial.suggest_int(name, spec["low"], spec["high"])

    full_params = DEFAULT_PARAMS.copy()
    full_params.update(CKF_DEFAULTS_FOR_SEEDING)
    full_params.update(params)
    full_params["threads"] = threads_per_trial

    for p in INT_PARAMS:
        if p in full_params:
            full_params[p] = int(round(full_params[p]))

    overrides = json.dumps(full_params)
    print(f"  [trial {trial.number}] params: {params}")

    try:
        metrics = run_ckf.remote(
            config="baseline",
            events=events_per_trial,
            input_file=input_file,
            config_overrides=overrides,
        )
    except Exception as e:
        print(f"  [trial {trial.number}] FAILED: {e}")
        return (0.0, 100.0, 100.0, 9999.0)

    if metrics is None or "efficiency" not in metrics:
        return (0.0, 100.0, 100.0, 9999.0)

    eff = float(metrics["efficiency"])
    fake = float(metrics["fakerate"])
    dup = float(metrics.get("duplicaterate", 0.0))
    rt = float(metrics.get("runtime", metrics.get("wall_time", 0.0)))

    if any(math.isnan(v) or math.isinf(v) for v in [eff, fake, dup, rt]):
        return (0.0, 100.0, 100.0, 9999.0)

    print(f"  [trial {trial.number}] eff={eff:.2f}% fake={fake:.3f}% "
          f"dup={dup:.3f}% rt={rt:.0f}s")
    return (eff, fake, dup, rt)


@app.function(
    image=image,
    volumes={DATA_PATH: data_vol},
    cpu=2,
    memory=8192,
    timeout=86400,
)
def run_seeding_optimizer(
    n_trials: int = 200,
    events_per_trial: int = 8,
    threads_per_trial: int = 8,
    max_parallel_trials: int = 8,
    input_file: str = None,
):
    """Multi-objective seeding optimization with Optuna NSGA-II.

    Sweeps 4 seeding parameters (CKF frozen at greedy defaults):
        num_seeds_per_spm [1, 80]
        seed_minPt [0.3, 2.0] GeV
        seed_impactMax [1.0, 10.0] mm
        seed_sigmaScattering [2, 10]

    Objectives (post-ambi, standard DM matching):
        1. ε_DM  (particle efficiency)  — MAXIMIZE
        2. f_DM  (track fake rate)      — MINIMIZE
        3. d_DM  (track duplicate rate) — MINIMIZE
        4. wall time                    — MINIMIZE

    Usage:
        modal run --detach modal_acts.py::run_seeding_optimizer \\
            --n-trials 200 --max-parallel-trials 8
    """
    import os
    from functools import partial

    import optuna
    import pandas as pd

    results_dir = f"{DATA_PATH}/optimizer/seeding_v2"
    os.makedirs(results_dir, exist_ok=True)
    db_path = f"sqlite:///{results_dir}/optuna.db"
    csv_path = f"{results_dir}/trials.csv"

    study = optuna.create_study(
        study_name="seeding_v2",
        directions=["maximize", "minimize", "minimize", "minimize"],
        sampler=optuna.samplers.NSGAIISampler(seed=42),
        storage=db_path,
        load_if_exists=True,
    )

    existing = len(study.trials)
    remaining = max(0, n_trials - existing)
    if existing > 0:
        print(f"Resuming: {existing} existing trials, {remaining} to go")

    objective = partial(
        _seeding_objective,
        events_per_trial=events_per_trial,
        threads_per_trial=threads_per_trial,
        input_file=input_file,
    )

    print(f"=== Optuna NSGA-II Seeding Optimization ===")
    print(f"Params: {list(SEEDING_PARAM_SPACE.keys())}")
    print(f"CKF frozen at: {CKF_DEFAULTS_FOR_SEEDING}")
    print(f"Objectives: ε_DM↑, f_DM↓, d_DM↓, runtime↓")
    print(f"Events/trial: {events_per_trial}, threads: {threads_per_trial}")
    print(f"Parallel trials: {max_parallel_trials}")
    print(f"Trials: {n_trials} ({remaining} remaining)")

    if remaining > 0:
        study.optimize(objective, n_trials=remaining, n_jobs=max_parallel_trials)

    df = study.trials_dataframe()
    df.to_csv(csv_path, index=False)
    data_vol.commit()

    pareto_trials = study.best_trials
    print(f"\n=== Seeding Optimization Complete ===")
    print(f"Total trials: {len(study.trials)}")
    print(f"Pareto front size: {len(pareto_trials)}")
    print(f"\nPareto front (ε_DM↑, f_DM↓):")
    for t in sorted(pareto_trials, key=lambda t: -t.values[0]):
        eff, fake, dup, rt = t.values
        p = t.params
        print(f"  eff={eff:.2f}%  fake={fake:.3f}%  dup={dup:.3f}%  "
              f"rt={rt:.0f}s  seeds={p.get('num_seeds_per_spm', '?')}  "
              f"minPt={p.get('seed_minPt', '?'):.2f}  "
              f"impact={p.get('seed_impactMax', '?'):.1f}  "
              f"σscat={p.get('seed_sigmaScattering', '?')}")

    # Select 3 operating points via tercile method on num_seeds_per_spm
    viable = [t for t in study.trials if t.values[0] > 80]
    if len(viable) >= 6:
        viable.sort(key=lambda t: t.params["num_seeds_per_spm"])
        n = len(viable)
        terciles = [viable[:n//3], viable[n//3:2*n//3], viable[2*n//3:]]
        labels = ["tight", "medium", "loose"]
        print(f"\n=== Suggested operating points (tercile on seeds/spm, lowest f_DM) ===")
        for label, group in zip(labels, terciles):
            best = min(group, key=lambda t: t.values[1])
            p = best.params
            print(f"  {label:7s}: seeds={p['num_seeds_per_spm']:3d}  "
                  f"minPt={p['seed_minPt']:.3f}  "
                  f"impact={p['seed_impactMax']:.1f}  "
                  f"σscat={p['seed_sigmaScattering']}  "
                  f"eff={best.values[0]:.2f}%  fake={best.values[1]:.3f}%")

    return {
        "n_trials": len(study.trials),
        "n_pareto": len(pareto_trials),
        "csv_path": csv_path,
    }


# ─── Optuna NSGA-II CKF Optimizer ──────────────────────────────────────

# Populated after seeding optimization; update before launching CKF stage
SEEDING_POINTS = {
    "tight": {
        "num_seeds_per_spm": 5,
        "seed_minPt": 0.698,
        "seed_impactMax": 6.1,
        "seed_sigmaScattering": 4.2,
    },
    "medium": {
        "num_seeds_per_spm": 18,
        "seed_minPt": 0.328,
        "seed_impactMax": 1.6,
        "seed_sigmaScattering": 4.1,
    },
    "loose": {
        "num_seeds_per_spm": 62,
        "seed_minPt": 0.662,
        "seed_impactMax": 1.6,
        "seed_sigmaScattering": 2.1,
    },
}

CKF_PARAM_SPACE = {
    "ckf_chi2CutOffMeasurement": {"type": "float", "low": 5.0, "high": 30.0},
    # chi2CutOffOutlier sampled conditionally: always > chi2CutOffMeasurement
    "ckf_chi2CutOffOutlier_margin": {"type": "float", "low": 0.0, "high": 30.0},
    "ckf_numMeasurementsCutOff": {"type": "int", "low": 1, "high": 5},
    "ckf_nMeasurementsMin": {"type": "int", "low": 4, "high": 9},
    "ckf_maxHolesAndOutliers": {"type": "int", "low": 1, "high": 6},
    "ckf_ptMin": {"type": "float", "low": 0.3, "high": 1.2},
}


def _ckf_objective(trial, seeding_point: str, events_per_trial: int,
                   threads_per_trial: int, input_file: str | None):
    """Optuna objective for joint CKF optimization. Returns (eff, fake, dup, runtime)."""
    import json
    import math

    params = {}
    for name, spec in CKF_PARAM_SPACE.items():
        if spec["type"] == "float":
            params[name] = trial.suggest_float(name, spec["low"], spec["high"])
        else:
            params[name] = trial.suggest_int(name, spec["low"], spec["high"])

    # Enforce chi2CutOffOutlier > chi2CutOffMeasurement
    chi2_meas = params["ckf_chi2CutOffMeasurement"]
    chi2_margin = params.pop("ckf_chi2CutOffOutlier_margin")
    params["ckf_chi2CutOffOutlier"] = chi2_meas + chi2_margin

    # Enforce nMeasurementsMin + maxHolesAndOutliers < 12 (ODD layer count)
    n_meas_min = params["ckf_nMeasurementsMin"]
    max_holes = params["ckf_maxHolesAndOutliers"]
    if n_meas_min + max_holes >= 12:
        params["ckf_maxHolesAndOutliers"] = 12 - n_meas_min - 1

    full_params = DEFAULT_PARAMS.copy()
    full_params.update(SEEDING_POINTS[seeding_point])
    full_params.update(params)
    full_params["threads"] = threads_per_trial

    for p in INT_PARAMS:
        if p in full_params:
            full_params[p] = int(round(full_params[p]))

    overrides = json.dumps(full_params)
    print(f"  [trial {trial.number}] params: {params}")

    try:
        metrics = run_ckf.remote(
            config="baseline",
            events=events_per_trial,
            input_file=input_file,
            config_overrides=overrides,
        )
    except Exception as e:
        print(f"  [trial {trial.number}] FAILED: {e}")
        return (0.0, 100.0, 100.0, 9999.0)

    if metrics is None or "efficiency" not in metrics:
        return (0.0, 100.0, 100.0, 9999.0)

    eff = float(metrics["efficiency"])
    fake = float(metrics["fakerate"])
    dup = float(metrics.get("duplicaterate", 0.0))
    rt = float(metrics.get("runtime", metrics.get("wall_time", 0.0)))

    if any(math.isnan(v) or math.isinf(v) for v in [eff, fake, dup, rt]):
        return (0.0, 100.0, 100.0, 9999.0)

    print(f"  [trial {trial.number}] eff={eff:.2f}% fake={fake:.3f}% "
          f"dup={dup:.3f}% rt={rt:.0f}s")
    return (eff, fake, dup, rt)


@app.function(
    image=image,
    volumes={DATA_PATH: data_vol},
    cpu=2,
    memory=8192,
    timeout=86400,
)
def run_ckf_optimizer(
    seeding_point: str = "medium",
    n_trials: int = 100,
    events_per_trial: int = 8,
    threads_per_trial: int = 8,
    max_parallel_trials: int = 4,
    input_file: str = None,
):
    """Multi-objective CKF optimization with Optuna NSGA-II.

    Jointly optimizes chi2CutOff, numMeasurementsCutOff, holes, outliers,
    nMeasurementsMin, ptMin — seeding frozen at one of 3 operating points.

    Objectives (post-ambi, standard DM matching definitions):
        1. ε_DM  (particle-level efficiency) — MAXIMIZE
        2. f_DM  (track-level fake rate)     — MINIMIZE
        3. d_DM  (track-level duplicate rate) — MINIMIZE
        4. wall time                          — MINIMIZE

    Usage:
        modal run modal_acts.py::run_ckf_optimizer \\
            --seeding-point medium --n-trials 100 \\
            --events-per-trial 8 --threads-per-trial 8 \\
            --max-parallel-trials 4
    """
    import os
    from concurrent.futures import ThreadPoolExecutor
    from functools import partial

    import optuna
    import pandas as pd

    if seeding_point not in SEEDING_POINTS:
        raise ValueError(
            f"Unknown seeding point '{seeding_point}'. "
            f"Choose from: {list(SEEDING_POINTS.keys())}"
        )

    results_dir = f"{DATA_PATH}/optimizer/ckf_{seeding_point}"
    os.makedirs(results_dir, exist_ok=True)
    db_path = f"sqlite:///{results_dir}/optuna.db"
    csv_path = f"{results_dir}/trials.csv"

    study = optuna.create_study(
        study_name=f"ckf_{seeding_point}",
        directions=["maximize", "minimize", "minimize", "minimize"],
        sampler=optuna.samplers.NSGAIISampler(seed=42),
        storage=db_path,
        load_if_exists=True,
    )

    existing = len(study.trials)
    remaining = max(0, n_trials - existing)
    if existing > 0:
        print(f"Resuming: {existing} existing trials, {remaining} to go")

    objective = partial(
        _ckf_objective,
        seeding_point=seeding_point,
        events_per_trial=events_per_trial,
        threads_per_trial=threads_per_trial,
        input_file=input_file,
    )

    print(f"=== Optuna NSGA-II CKF Optimization ===")
    print(f"Seeding point: {seeding_point} -> {SEEDING_POINTS[seeding_point]}")
    print(f"CKF params: {list(CKF_PARAM_SPACE.keys())}")
    print(f"Objectives: efficiency↑, fake↓, dup↓, runtime↓")
    print(f"Events/trial: {events_per_trial}, threads: {threads_per_trial}")
    print(f"Parallel trials: {max_parallel_trials}")
    print(f"Trials: {n_trials} ({remaining} remaining)")

    if remaining > 0:
        study.optimize(objective, n_trials=remaining, n_jobs=max_parallel_trials)

    # Save results
    df = study.trials_dataframe()
    df.to_csv(csv_path, index=False)
    data_vol.commit()

    # Report Pareto front
    pareto_trials = study.best_trials
    print(f"\n=== Optimization Complete ===")
    print(f"Total trials: {len(study.trials)}")
    print(f"Pareto front size: {len(pareto_trials)}")
    print(f"\nPareto front (eff↑, fake↓, dup↓, runtime↓):")
    for t in sorted(pareto_trials, key=lambda t: -t.values[0]):
        eff, fake, dup, rt = t.values
        p = {k: v for k, v in t.params.items()}
        print(f"  eff={eff:.2f}%  fake={fake:.3f}%  dup={dup:.3f}%  "
              f"rt={rt:.0f}s  branch={p.get('ckf_numMeasurementsCutOff', '?')}  "
              f"chi2={p.get('ckf_chi2CutOffMeasurement', '?'):.1f}")

    return {
        "seeding_point": seeding_point,
        "n_trials": len(study.trials),
        "n_pareto": len(pareto_trials),
        "csv_path": csv_path,
        "db_path": db_path,
    }


@app.function(
    image=image,
    volumes={DATA_PATH: data_vol},
    cpu=2,
    memory=4096,
    timeout=3600,
)
def plot_pareto_fronts():
    """Generate Pareto front visualizations from completed Optuna studies.

    Produces:
      1. Seeding Pareto front with 3 operating points marked
      2. Per-seeding-point CKF Pareto fronts (eff vs fake, colored by branch cap)
      3. System-level Pareto envelope overlaying all 3
      4. Branch cap regime analysis (eff vs fake, faceted by numMeasurementsCutOff)

    Saved to /data/optimizer/plots/ on the Modal volume.
    """
    import os

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    plot_dir = f"{DATA_PATH}/optimizer/plots"
    os.makedirs(plot_dir, exist_ok=True)

    colors = {"tight": "#2196F3", "medium": "#4CAF50", "loose": "#FF9800"}
    markers = {"tight": "s", "medium": "o", "loose": "D"}

    # ── Plot 1: Seeding Pareto front ──
    # Prefer v2 Optuna results; fall back to v1 xopt CSV
    seeding_v2_csv = f"{DATA_PATH}/optimizer/seeding_v2/trials.csv"
    seeding_v1_csv = f"{DATA_PATH}/optimizer/seeding/optimization_history.csv"

    if os.path.exists(seeding_v2_csv):
        sd_raw = pd.read_csv(seeding_v2_csv)
        # Optuna dataframe: values_0=eff, values_1=fake, values_2=dup, values_3=rt
        eff_s = [c for c in sd_raw.columns if "values_0" in c][0]
        fake_s = [c for c in sd_raw.columns if "values_1" in c][0]
        seeds_s = [c for c in sd_raw.columns if "num_seeds_per_spm" in c][0]
        minpt_s = [c for c in sd_raw.columns if "seed_minPt" in c][0]
        impact_s = [c for c in sd_raw.columns if "seed_impactMax" in c][0]
        sigma_s = [c for c in sd_raw.columns if "seed_sigmaScattering" in c][0]
        sd = pd.DataFrame({
            "efficiency": sd_raw[eff_s],
            "fakerate": sd_raw[fake_s],
            "num_seeds_per_spm": sd_raw[seeds_s],
            "seed_minPt": sd_raw[minpt_s],
            "seed_impactMax": sd_raw[impact_s],
            "seed_sigmaScattering": sd_raw[sigma_s],
        })
        sd = sd[(sd["efficiency"] > 0) & (sd["fakerate"] < 100)]
        seeding_version = "v2 (Optuna NSGA-II, 4-param)"
    elif os.path.exists(seeding_v1_csv):
        sd = pd.read_csv(seeding_v1_csv)
        sd["seed_sigmaScattering"] = 5  # fixed in v1
        seeding_version = "v1 (xopt, 3-param)"
    else:
        sd = None
        seeding_version = None

    if sd is not None and len(sd) > 0:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(sd["efficiency"], sd["fakerate"], c="#90CAF9", s=40,
                   alpha=0.5, label="All trials", zorder=2)

        # Mark Pareto front
        pts = sd[["efficiency", "fakerate"]].values
        mask = np.ones(len(pts), dtype=bool)
        for i in range(len(pts)):
            for j in range(len(pts)):
                if i == j:
                    continue
                if pts[j, 0] >= pts[i, 0] and pts[j, 1] <= pts[i, 1] and \
                   (pts[j, 0] > pts[i, 0] or pts[j, 1] < pts[i, 1]):
                    mask[i] = False
                    break
        pareto = sd[mask].sort_values("efficiency")
        ax.plot(pareto["efficiency"], pareto["fakerate"], "k-", alpha=0.4,
                linewidth=1, zorder=3)
        ax.scatter(pareto["efficiency"], pareto["fakerate"], c="#1565C0",
                   s=60, zorder=4, label="Pareto front")

        # Mark the 3 operating points
        for name, sp in SEEDING_POINTS.items():
            dists = ((sd["num_seeds_per_spm"] - sp["num_seeds_per_spm"])**2 +
                     (sd["seed_minPt"] - sp["seed_minPt"])**2 * 1000 +
                     (sd["seed_impactMax"] - sp["seed_impactMax"])**2 +
                     (sd["seed_sigmaScattering"] - sp.get("seed_sigmaScattering", 5))**2)
            idx = dists.idxmin()
            row = sd.loc[idx]
            ax.scatter(row["efficiency"], row["fakerate"], c=colors[name],
                       marker=markers[name], s=200, zorder=5, edgecolors="black",
                       linewidths=1.5, label=f"{name.capitalize()}")
            ax.annotate(f" {name}", (row["efficiency"], row["fakerate"]),
                        fontsize=9, fontweight="bold", color=colors[name])

        ax.set_xlabel("Particle Efficiency (%)", fontsize=12)
        ax.set_ylabel("Track Fake Rate (%)", fontsize=12)
        ax.set_title(f"Seeding Optimization — Pareto Front ({seeding_version})\n"
                     "(ColliderML ttbar μ=200, post-ambi DM matching)",
                     fontsize=11)
        ax.legend(loc="upper left", fontsize=9)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(f"{plot_dir}/seeding_pareto.png", dpi=150)
        plt.close(fig)
        print(f"Saved seeding_pareto.png")

    # ── Plot 2 & 3: CKF Pareto fronts per seeding point + overlay ──
    fig_overlay, ax_overlay = plt.subplots(figsize=(10, 7))
    all_pareto_pts = []

    for sp_name in ["tight", "medium", "loose"]:
        csv_path = f"{DATA_PATH}/optimizer/ckf_{sp_name}/trials.csv"
        if not os.path.exists(csv_path):
            print(f"No CKF results for {sp_name}, skipping")
            continue

        df = pd.read_csv(csv_path)
        # Optuna dataframe columns: values_0=eff, values_1=fake, values_2=dup, values_3=rt
        eff_col = [c for c in df.columns if "values_0" in c][0]
        fake_col = [c for c in df.columns if "values_1" in c][0]
        dup_col = [c for c in df.columns if "values_2" in c][0]
        rt_col = [c for c in df.columns if "values_3" in c][0]
        branch_col = [c for c in df.columns
                      if "ckf_numMeasurementsCutOff" in c][0]

        valid = df[(df[eff_col] > 0) & (df[fake_col] < 100)].copy()
        if valid.empty:
            continue

        # Compute 2D Pareto (eff↑, fake↓)
        pts = valid[[eff_col, fake_col]].values
        mask = np.ones(len(pts), dtype=bool)
        for i in range(len(pts)):
            for j in range(len(pts)):
                if i == j:
                    continue
                if pts[j, 0] >= pts[i, 0] and pts[j, 1] <= pts[i, 1] and \
                   (pts[j, 0] > pts[i, 0] or pts[j, 1] < pts[i, 1]):
                    mask[i] = False
                    break
        pareto_df = valid[mask].sort_values(eff_col)

        # Per-point figure: color by branch cap
        fig_sp, ax_sp = plt.subplots(figsize=(8, 6))
        sc = ax_sp.scatter(valid[eff_col], valid[fake_col],
                           c=valid[branch_col], cmap="viridis",
                           s=40, alpha=0.6, vmin=1, vmax=10)
        ax_sp.plot(pareto_df[eff_col], pareto_df[fake_col],
                   "k-", alpha=0.5, linewidth=1)
        ax_sp.scatter(pareto_df[eff_col], pareto_df[fake_col],
                      c=pareto_df[branch_col], cmap="viridis",
                      s=100, edgecolors="black", linewidths=1.2,
                      vmin=1, vmax=10, zorder=5)
        plt.colorbar(sc, ax=ax_sp, label="numMeasurementsCutOff (branch cap)")
        ax_sp.set_xlabel("Particle Efficiency (%)", fontsize=12)
        ax_sp.set_ylabel("Track Fake Rate (%)", fontsize=12)
        ax_sp.set_title(f"CKF Optimization — {sp_name.capitalize()} Seeding\n"
                        f"(color = branch cap, post-ambi DM matching)",
                        fontsize=11)
        ax_sp.grid(True, alpha=0.3)
        fig_sp.tight_layout()
        fig_sp.savefig(f"{plot_dir}/ckf_pareto_{sp_name}.png", dpi=150)
        plt.close(fig_sp)
        print(f"Saved ckf_pareto_{sp_name}.png")

        # Add to overlay
        ax_overlay.scatter(valid[eff_col], valid[fake_col],
                           c=colors[sp_name], s=20, alpha=0.2)
        ax_overlay.plot(pareto_df[eff_col], pareto_df[fake_col],
                        color=colors[sp_name], linewidth=2, alpha=0.8,
                        label=f"{sp_name.capitalize()} seeding")
        ax_overlay.scatter(pareto_df[eff_col], pareto_df[fake_col],
                           c=colors[sp_name], s=60, edgecolors="black",
                           linewidths=0.8, zorder=5)

        for _, row in pareto_df.iterrows():
            all_pareto_pts.append({
                "eff": row[eff_col], "fake": row[fake_col],
                "seeding": sp_name,
            })

    # System envelope
    if all_pareto_pts:
        env = pd.DataFrame(all_pareto_pts)
        pts = env[["eff", "fake"]].values
        mask = np.ones(len(pts), dtype=bool)
        for i in range(len(pts)):
            for j in range(len(pts)):
                if i == j:
                    continue
                if pts[j, 0] >= pts[i, 0] and pts[j, 1] <= pts[i, 1] and \
                   (pts[j, 0] > pts[i, 0] or pts[j, 1] < pts[i, 1]):
                    mask[i] = False
                    break
        envelope = env[mask].sort_values("eff")
        ax_overlay.plot(envelope["eff"], envelope["fake"], "k--",
                        linewidth=2.5, alpha=0.7, label="System envelope",
                        zorder=10)

    ax_overlay.set_xlabel("Particle Efficiency (%)", fontsize=12)
    ax_overlay.set_ylabel("Track Fake Rate (%)", fontsize=12)
    ax_overlay.set_title("System Pareto Envelope — All Seeding Regimes\n"
                         "(ColliderML ttbar μ=200, post-ambi DM matching)",
                         fontsize=11)
    ax_overlay.legend(fontsize=10)
    ax_overlay.grid(True, alpha=0.3)
    fig_overlay.tight_layout()
    fig_overlay.savefig(f"{plot_dir}/system_pareto_envelope.png", dpi=150)
    plt.close(fig_overlay)
    print(f"Saved system_pareto_envelope.png")

    data_vol.commit()
    return {"plot_dir": plot_dir}


# ─── Optuna MOTPE Joint Seeding+CKF Optimizer ──────────────────────────
#
# 10D multi-objective study. Every trial runs on the optimization split
# [0, 32). The evaluation split [32, 64) is held out for post-hoc validation
# only (same events as the ACTS defaults baseline).

JOINT_PARAM_SPACE = {
    # Seeding
    "num_seeds_per_spm": {"type": "int", "low": 5, "high": 80},
    "seed_minPt": {"type": "float", "low": 0.3, "high": 1.2},
    "seed_impactMax": {"type": "float", "low": 0.5, "high": 5.0},
    "seed_sigmaScattering": {"type": "float", "low": 2.0, "high": 10.0},
    # CKF
    "ckf_chi2CutOffMeasurement": {"type": "float", "low": 5.0, "high": 30.0},
    # Outlier chi2 sampled as measurement + margin so outlier > measurement
    "ckf_chi2CutOffOutlier_margin": {"type": "float", "low": 0.0, "high": 30.0},
    "ckf_numMeasurementsCutOff": {"type": "int", "low": 1, "high": 5},
    "ckf_nMeasurementsMin": {"type": "int", "low": 4, "high": 9},
    "ckf_maxHolesAndOutliers": {"type": "int", "low": 1, "high": 6},
    "ckf_ptMin": {"type": "float", "low": 0.3, "high": 1.2},
}

# Optimization split: events [0, 32). Never change without updating LOG.md.
OPT_SKIP = 0
OPT_EVENTS = 32
# Evaluation split (held out): events [32, 64)
EVAL_SKIP = 32
EVAL_EVENTS = 32

# Hard constraints
MIN_EFF_PCT = 50.0  # prune if ε_DM < 50%
MAX_WALL_S = 20 * 60  # prune if wall > 20 min for the 32-event batch


def _joint_objective(
    trial,
    events_per_trial: int,
    threads_per_trial: int,
    input_file: str | None,
    skip: int,
):
    """MOTPE objective. Returns (ε_DM, f_DM, d_DM, wall_s_per_event)."""
    import json
    import math
    import time

    import optuna

    params = {}
    for name, spec in JOINT_PARAM_SPACE.items():
        if spec["type"] == "float":
            params[name] = trial.suggest_float(name, spec["low"], spec["high"])
        else:
            params[name] = trial.suggest_int(name, spec["low"], spec["high"])

    chi2_meas = params["ckf_chi2CutOffMeasurement"]
    chi2_margin = params.pop("ckf_chi2CutOffOutlier_margin")
    params["ckf_chi2CutOffOutlier"] = chi2_meas + chi2_margin

    n_meas_min = params["ckf_nMeasurementsMin"]
    max_holes = params["ckf_maxHolesAndOutliers"]
    if n_meas_min + max_holes >= 12:
        params["ckf_maxHolesAndOutliers"] = 12 - n_meas_min - 1

    # Base knobs for joint study: no loc0 cut; ambi fixed; performance on.
    full_params = {
        "digi": True,
        "reco": True,
        "threads": threads_per_trial,
        "skip": skip,
        "ckf_finding_performance": True,
        "ambi_finding_performance": True,
        "ambi_solver": "greedy",
        "ambi_nMeasurementsMin": 6,
    }
    full_params.update(params)

    for p in INT_PARAMS:
        if p in full_params:
            full_params[p] = int(round(full_params[p]))

    overrides = json.dumps(full_params)
    print(f"  [trial {trial.number}] skip={skip} events={events_per_trial} "
          f"params={params}")

    t0 = time.time()
    try:
        metrics = run_ckf.remote(
            config="baseline",
            events=events_per_trial,
            input_file=input_file,
            config_overrides=overrides,
        )
    except Exception as e:
        print(f"  [trial {trial.number}] FAILED: {e}")
        raise optuna.TrialPruned()

    wall_batch = time.time() - t0
    if metrics is None or "efficiency" not in metrics:
        raise optuna.TrialPruned()

    eff = float(metrics["efficiency"])
    fake = float(metrics["fakerate"])
    dup = float(metrics.get("duplicaterate", 0.0))
    # Prefer true batch wall; fall back to ACTS-reported runtime.
    wall = float(metrics.get("wall_time", wall_batch))
    if wall <= 0:
        wall = wall_batch
    wall_per_event = wall / max(events_per_trial, 1)

    if any(math.isnan(v) or math.isinf(v) for v in [eff, fake, dup, wall_per_event]):
        raise optuna.TrialPruned()
    if eff < MIN_EFF_PCT:
        print(f"  [trial {trial.number}] pruned: eff={eff:.2f}% < {MIN_EFF_PCT}")
        raise optuna.TrialPruned()
    if wall > MAX_WALL_S:
        print(f"  [trial {trial.number}] pruned: wall={wall:.0f}s > {MAX_WALL_S}")
        raise optuna.TrialPruned()

    print(f"  [trial {trial.number}] eff={eff:.2f}% fake={fake:.3f}% "
          f"dup={dup:.3f}% wall/evt={wall_per_event:.2f}s "
          f"(batch={wall:.0f}s)")

    # Per-trial W&B log (works with n_jobs>1; callback is optional).
    try:
        import wandb

        if wandb.run is not None:
            wandb.log(
                {
                    "efficiency": eff,
                    "fakerate": fake,
                    "duplicaterate": dup,
                    "wall_per_event": wall_per_event,
                    "wall_batch": wall,
                    **{f"param/{k}": v for k, v in params.items()},
                },
                step=trial.number,
            )
    except Exception:
        pass

    return (eff, fake, dup, wall_per_event)


@app.function(
    image=image,
    volumes={DATA_PATH: data_vol},
    secrets=[modal.Secret.from_name("wandb")],
    cpu=2,
    memory=8192,
    timeout=86400,
)
def run_joint_motpe_optimizer(
    n_trials: int = 500,
    events_per_trial: int = OPT_EVENTS,
    threads_per_trial: int = 8,
    max_parallel_trials: int = 4,
    input_file: str = None,
    skip: int = OPT_SKIP,
):
    """Joint 10D seeding+CKF multi-objective optimization (MO-TPE).

    Uses Optuna ``TPESampler`` — the Optuna 4 successor to ``MOTPESampler``
    (removed in v4.0). Same algorithm family: surrogate updated after every
    trial; ``n_startup_trials=20`` random warmup then informed proposals.

    Objectives (post-ambi, track-level f_DM / d_DM):
        1. ε_DM  — MAXIMIZE
        2. f_DM  — MINIMIZE  (fakeratio_tracks, double-majority)
        3. d_DM  — MINIMIZE
        4. wall_time_per_event — MINIMIZE

    Default split: optimize on events [0, 32) (skip=0). Do NOT pass the
    evaluation split here — validate Pareto configs separately with skip=32.

    Usage:
        modal run --detach modal_acts.py::run_joint_motpe_optimizer \\
            --n-trials 500 --events-per-trial 32 --threads-per-trial 8 \\
            --max-parallel-trials 4
    """
    import os
    from functools import partial

    import optuna

    results_dir = f"{DATA_PATH}/optimizer/joint_motpe"
    os.makedirs(results_dir, exist_ok=True)
    db_path = f"sqlite:///{results_dir}/optuna.db"
    csv_path = f"{results_dir}/trials.csv"

    # MOTPESampler was removed in Optuna 4.0; TPESampler is the official
    # replacement and implements the same multi-objective TPE/MOTPE algorithm
    # (with the v4 MO-TPE speedups). Keep n_startup_trials=20 as specified.
    sampler = optuna.samplers.TPESampler(seed=42, n_startup_trials=20)
    study = optuna.create_study(
        study_name="joint_seeding_ckf_motpe",
        directions=["maximize", "minimize", "minimize", "minimize"],
        sampler=sampler,
        storage=db_path,
        load_if_exists=True,
    )

    existing = len([t for t in study.trials if t.state.is_finished()])
    remaining = max(0, n_trials - existing)
    if existing > 0:
        print(f"Resuming: {existing} finished trials, {remaining} to go")

    objective = partial(
        _joint_objective,
        events_per_trial=events_per_trial,
        threads_per_trial=threads_per_trial,
        input_file=input_file,
        skip=skip,
    )

    print("=== Optuna MOTPE Joint Seeding+CKF Optimization ===")
    print(f"Params: {list(JOINT_PARAM_SPACE.keys())}")
    print(f"Objectives: ε_DM↑, f_DM↓, d_DM↓, wall_s/event↓")
    print(f"Event range: [{skip}, {skip + events_per_trial})")
    print(f"Events/trial: {events_per_trial}, threads: {threads_per_trial}")
    print(f"Parallel trials: {max_parallel_trials}")
    print(f"Startup (random): 20, total target: {n_trials} ({remaining} remaining)")
    print(f"Constraints: ε≥{MIN_EFF_PCT}%, wall≤{MAX_WALL_S}s, no loc0 cut")
    print(f"Storage: {db_path}")
    try:
        import acts
        print(f"ACTS: {getattr(acts, '__version__', 'unknown')} "
              f"({getattr(acts, '__file__', '?')})")
    except Exception as e:
        print(f"ACTS version probe failed: {e}")

    callbacks = []
    if os.environ.get("WANDB_API_KEY"):
        try:
            import wandb

            wandb.init(
                project="cckf-baseline-optimization",
                name="joint_motpe",
                config={
                    "sampler": "TPESampler",
                    "n_startup_trials": 20,
                    "n_trials": n_trials,
                    "seed": 42,
                    "event_range": [skip, skip + events_per_trial],
                    "param_space": JOINT_PARAM_SPACE,
                },
            )
            print("W&B logging enabled (project=cckf-baseline-optimization)")
            try:
                from optuna.integration.wandb import WeightsAndBiasesCallback

                callbacks.append(
                    WeightsAndBiasesCallback(
                        metric_names=[
                            "efficiency",
                            "fakerate",
                            "duplicaterate",
                            "wall_per_event",
                        ],
                        as_multirun=False,
                    )
                )
            except Exception as e:
                print(f"Optuna W&B callback unavailable ({e}); "
                      "using per-trial wandb.log in objective")
        except Exception as e:
            print(f"W&B init failed ({e}); continuing without it")
    else:
        print("WANDB_API_KEY not set — trial metrics still saved to Optuna CSV/SQLite")

    if remaining > 0:
        study.optimize(
            objective,
            n_trials=remaining,
            n_jobs=max_parallel_trials,
            callbacks=callbacks,
        )

    df = study.trials_dataframe()
    df.to_csv(csv_path, index=False)
    data_vol.commit()

    pareto = study.best_trials
    print(f"\n=== MOTPE Joint Optimization Complete ===")
    print(f"Total trials: {len(study.trials)}")
    print(f"Pareto front size: {len(pareto)}")
    print(f"\nPareto front (ε↑, f↓, d↓, wall/evt↓):")
    for t in sorted(pareto, key=lambda t: -t.values[0]):
        eff, fake, dup, rt = t.values
        p = t.params
        print(
            f"  ε={eff:.2f}%  f={fake:.3f}%  d={dup:.3f}%  "
            f"t={rt:.2f}s/evt  seeds={p.get('num_seeds_per_spm', '?')}  "
            f"branch={p.get('ckf_numMeasurementsCutOff', '?')}  "
            f"χ²={p.get('ckf_chi2CutOffMeasurement', float('nan')):.1f}  "
            f"nMeas={p.get('ckf_nMeasurementsMin', '?')}"
        )

    return {
        "n_trials": len(study.trials),
        "n_pareto": len(pareto),
        "csv_path": csv_path,
        "db_path": db_path,
        "event_range": [skip, skip + events_per_trial],
    }


def _params_from_joint_trial(trial_params: dict) -> dict:
    """Map Optuna joint-study params → digi_and_reco overrides."""
    params = dict(trial_params)
    chi2_meas = float(params["ckf_chi2CutOffMeasurement"])
    margin = float(params.pop("ckf_chi2CutOffOutlier_margin"))
    params["ckf_chi2CutOffOutlier"] = chi2_meas + margin

    n_meas_min = int(params["ckf_nMeasurementsMin"])
    max_holes = int(params["ckf_maxHolesAndOutliers"])
    if n_meas_min + max_holes >= 12:
        params["ckf_maxHolesAndOutliers"] = 12 - n_meas_min - 1

    full = {
        "digi": True,
        "reco": True,
        "ckf_finding_performance": True,
        "ambi_finding_performance": True,
        "ambi_solver": "greedy",
        "ambi_nMeasurementsMin": 6,
    }
    full.update(params)
    for p in INT_PARAMS:
        if p in full:
            full[p] = int(round(full[p]))
    return full


def _pareto_trials_from_study(study, front: str = "4d"):
    """Return COMPLETE trials on the requested Pareto front.

    front:
      - ``4d``: Optuna multi-objective best_trials (ε↑,f↓,d↓,wall↓)
      - ``2d``: non-dominated on (ε↑, f↓) only — smaller, cheaper eval
    """
    if front == "4d":
        return list(study.best_trials)

    complete = [t for t in study.trials if t.state.name == "COMPLETE" and t.values]
    pareto = []
    for t in complete:
        dominated = False
        for u in complete:
            if u.number == t.number:
                continue
            # u dominates t on (max ε, min f)
            if (
                u.values[0] >= t.values[0]
                and u.values[1] <= t.values[1]
                and (u.values[0] > t.values[0] or u.values[1] < t.values[1])
            ):
                dominated = True
                break
        if not dominated:
            pareto.append(t)
    return pareto


@app.function(
    image=image,
    volumes={DATA_PATH: data_vol},
    cpu=2,
    memory=8192,
    timeout=86400,
)
def validate_joint_motpe_eval(
    front: str = "4d",
    events: int = EVAL_EVENTS,
    skip: int = EVAL_SKIP,
    threads: int = 8,
    max_parallel: int = 8,
    input_file: str = None,
):
    """Re-evaluate joint MOTPE Pareto configs on the held-out eval split.

    Default: events [32, 64) — same set as the ACTS defaults baseline.
    Reads the Optuna study from ``/data/optimizer/joint_motpe/optuna.db``.
    Resumes from an existing ``eval_pareto_{front}.csv`` if present.

    Usage:
        modal run --detach modal_acts.py::validate_joint_motpe_eval \\
            --front 4d --max-parallel 8
        # cheaper 2D (ε,f) front (~40 configs):
        modal run --detach modal_acts.py::validate_joint_motpe_eval --front 2d
    """
    import csv
    import json
    import os
    import time
    from concurrent.futures import ThreadPoolExecutor, as_completed

    import optuna

    results_dir = f"{DATA_PATH}/optimizer/joint_motpe"
    db_path = f"sqlite:///{results_dir}/optuna.db"
    out_csv = f"{results_dir}/eval_pareto_{front}.csv"

    study = optuna.load_study(
        study_name="joint_seeding_ckf_motpe",
        storage=db_path,
    )
    pareto = _pareto_trials_from_study(study, front=front)
    pareto = sorted(pareto, key=lambda t: -t.values[0])
    print(f"=== Joint MOTPE eval-set validation ===")
    print(f"Front: {front}  n_pareto={len(pareto)}")
    print(f"Eval events: [{skip}, {skip + events})")
    print(f"Parallel: {max_parallel}, threads/trial: {threads}")

    # Resume: skip trial numbers already evaluated.
    done = {}
    fieldnames = [
        "trial_number",
        "opt_efficiency",
        "opt_fakerate",
        "opt_duplicaterate",
        "opt_wall_per_event",
        "eval_efficiency",
        "eval_fakerate",
        "eval_duplicaterate",
        "eval_wall_per_event",
        "eval_wall_batch",
        "delta_efficiency",
        "delta_fakerate",
        # params
        "num_seeds_per_spm",
        "seed_minPt",
        "seed_impactMax",
        "seed_sigmaScattering",
        "ckf_chi2CutOffMeasurement",
        "ckf_chi2CutOffOutlier",
        "ckf_numMeasurementsCutOff",
        "ckf_nMeasurementsMin",
        "ckf_maxHolesAndOutliers",
        "ckf_ptMin",
    ]
    if os.path.exists(out_csv):
        with open(out_csv) as f:
            for row in csv.DictReader(f):
                done[int(row["trial_number"])] = row
        print(f"Resuming: {len(done)} configs already evaluated")

    pending = [t for t in pareto if t.number not in done]
    print(f"Remaining: {len(pending)}")

    def _run_one(trial):
        overrides = _params_from_joint_trial(trial.params)
        overrides["threads"] = threads
        overrides["skip"] = skip
        t0 = time.time()
        metrics = run_ckf.remote(
            config="baseline",
            events=events,
            input_file=input_file,
            config_overrides=json.dumps(overrides),
        )
        wall = time.time() - t0
        if metrics is None or "efficiency" not in metrics:
            return trial.number, None, wall, overrides
        return trial.number, metrics, float(metrics.get("wall_time", wall)), overrides

    rows_out = list(done.values())

    def _record(trial, metrics, wall, overrides):
        opt_e, opt_f, opt_d, opt_w = trial.values
        if metrics is None:
            print(f"  [eval t{trial.number}] FAILED")
            return None
        ev_e = float(metrics["efficiency"])
        ev_f = float(metrics["fakerate"])
        ev_d = float(metrics.get("duplicaterate", 0.0))
        ev_w = wall / max(events, 1)
        row = {
            "trial_number": trial.number,
            "opt_efficiency": opt_e,
            "opt_fakerate": opt_f,
            "opt_duplicaterate": opt_d,
            "opt_wall_per_event": opt_w,
            "eval_efficiency": ev_e,
            "eval_fakerate": ev_f,
            "eval_duplicaterate": ev_d,
            "eval_wall_per_event": ev_w,
            "eval_wall_batch": wall,
            "delta_efficiency": ev_e - opt_e,
            "delta_fakerate": ev_f - opt_f,
            "num_seeds_per_spm": overrides.get("num_seeds_per_spm"),
            "seed_minPt": overrides.get("seed_minPt"),
            "seed_impactMax": overrides.get("seed_impactMax"),
            "seed_sigmaScattering": overrides.get("seed_sigmaScattering"),
            "ckf_chi2CutOffMeasurement": overrides.get("ckf_chi2CutOffMeasurement"),
            "ckf_chi2CutOffOutlier": overrides.get("ckf_chi2CutOffOutlier"),
            "ckf_numMeasurementsCutOff": overrides.get("ckf_numMeasurementsCutOff"),
            "ckf_nMeasurementsMin": overrides.get("ckf_nMeasurementsMin"),
            "ckf_maxHolesAndOutliers": overrides.get("ckf_maxHolesAndOutliers"),
            "ckf_ptMin": overrides.get("ckf_ptMin"),
        }
        print(
            f"  [eval t{trial.number}] opt ε={opt_e:.2f}/f={opt_f:.3f} → "
            f"eval ε={ev_e:.2f}/f={ev_f:.3f}  "
            f"(Δε={ev_e - opt_e:+.2f}, Δf={ev_f - opt_f:+.3f}) "
            f"wall={wall:.0f}s"
        )
        return row

    # Map trial number → trial object for as_completed ordering.
    by_num = {t.number: t for t in pending}

    with ThreadPoolExecutor(max_workers=max_parallel) as ex:
        futs = {ex.submit(_run_one, t): t.number for t in pending}
        for fut in as_completed(futs):
            num, metrics, wall, overrides = fut.result()
            trial = by_num[num]
            row = _record(trial, metrics, wall, overrides)
            if row is not None:
                rows_out.append(row)
                # Incremental save + volume commit for crash resilience.
                rows_out.sort(key=lambda r: -float(r["opt_efficiency"]))
                with open(out_csv, "w", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=fieldnames)
                    w.writeheader()
                    w.writerows(rows_out)
                data_vol.commit()

    rows_out.sort(key=lambda r: -float(r["opt_efficiency"]))
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows_out)
    data_vol.commit()

    # Summary: overfit gap + 3 operating points on eval metrics.
    if rows_out:
        deltas_e = [float(r["delta_efficiency"]) for r in rows_out]
        deltas_f = [float(r["delta_fakerate"]) for r in rows_out]
        print(f"\n=== Eval complete: {len(rows_out)} configs ===")
        print(
            f"Δε (eval-opt): mean={sum(deltas_e)/len(deltas_e):+.2f} pp  "
            f"min={min(deltas_e):+.2f}  max={max(deltas_e):+.2f}"
        )
        print(
            f"Δf (eval-opt): mean={sum(deltas_f)/len(deltas_f):+.3f} pp  "
            f"min={min(deltas_f):+.3f}  max={max(deltas_f):+.3f}"
        )

        # Pick tight/medium/loose by eval efficiency among low-ish fake.
        ranked = sorted(rows_out, key=lambda r: float(r["eval_efficiency"]))
        picks = {
            "tight": ranked[len(ranked) // 6],
            "medium": ranked[len(ranked) // 2],
            "loose": ranked[-1],
        }
        # Prefer low-fake alternatives near those ε bands if available.
        def _best_near(target_eff, fake_cap=5.0):
            cand = [
                r
                for r in rows_out
                if abs(float(r["eval_efficiency"]) - target_eff) < 3.0
                and float(r["eval_fakerate"]) <= fake_cap
            ]
            if not cand:
                cand = sorted(
                    rows_out,
                    key=lambda r: abs(float(r["eval_efficiency"]) - target_eff),
                )[:5]
            return min(cand, key=lambda r: float(r["eval_fakerate"]))

        print("\nSuggested operating points (eval set):")
        for name, seed_row in picks.items():
            row = _best_near(float(seed_row["eval_efficiency"]))
            print(
                f"  {name}: t{row['trial_number']}  "
                f"ε={float(row['eval_efficiency']):.2f}%  "
                f"f={float(row['eval_fakerate']):.3f}%  "
                f"(opt ε={float(row['opt_efficiency']):.2f}% "
                f"f={float(row['opt_fakerate']):.3f}%)  "
                f"seeds={row['num_seeds_per_spm']} "
                f"branch={row['ckf_numMeasurementsCutOff']} "
                f"nMeas={row['ckf_nMeasurementsMin']}"
            )

    return {"n_evaluated": len(rows_out), "csv_path": out_csv, "front": front}


def _mean_std(xs: list[float]) -> tuple[float, float]:
    import statistics

    if not xs:
        return float("nan"), float("nan")
    if len(xs) == 1:
        return xs[0], 0.0
    return statistics.fmean(xs), statistics.stdev(xs)


def _per_event_ckf_summary(
    *,
    label: str,
    config_name: str,
    params: dict,
    skip0: int,
    n_events: int,
    max_parallel: int,
    out_path: str,
) -> dict:
    """Run digi+reco once per event; return mean±stdev of post-ambi metrics."""
    import json
    import time
    from concurrent.futures import ThreadPoolExecutor, as_completed

    base_overrides = {
        "digi": True,
        "reco": True,
        "threads": 8,
        "ckf_finding_performance": True,
        "ambi_finding_performance": True,
        "ambi_solver": "greedy",
        "ambi_nMeasurementsMin": 6,
        **params,
    }
    for p in INT_PARAMS:
        if p in base_overrides:
            base_overrides[p] = int(round(base_overrides[p]))

    def one(skip: int) -> dict:
        overrides = dict(base_overrides)
        overrides["skip"] = skip
        t0 = time.time()
        m = run_ckf.remote(
            config=config_name,
            events=1,
            config_overrides=json.dumps(overrides),
        )
        wall = time.time() - t0
        if m is None or "efficiency" not in m:
            print(f"  [{label}] event {skip}: FAILED")
            return {"skip": skip, "ok": False}
        row = {
            "skip": skip,
            "ok": True,
            "efficiency": float(m["efficiency"]),
            "fakerate": float(m["fakerate"]),
            "duplicaterate": float(m.get("duplicaterate", 0.0)),
            "wall": float(m.get("wall_time", wall)),
        }
        print(
            f"  [{label}] event {skip}: ε={row['efficiency']:.2f}% "
            f"f={row['fakerate']:.3f}% d={row['duplicaterate']:.3f}% "
            f"wall={row['wall']:.1f}s"
        )
        return row

    print(
        f"=== Per-event {label} ({config_name}) "
        f"events [{skip0}, {skip0 + n_events}) ==="
    )
    rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=max_parallel) as ex:
        futs = [ex.submit(one, skip0 + i) for i in range(n_events)]
        for fut in as_completed(futs):
            rows.append(fut.result())
    rows = sorted(rows, key=lambda r: r["skip"])
    ok = [r for r in rows if r.get("ok")]

    summary = {
        "label": label,
        "config": config_name,
        "event_range": [skip0, skip0 + n_events],
        "n_ok": len(ok),
        "n_fail": len(rows) - len(ok),
        "params": params,
        "mean_std": {
            "efficiency_pct": _mean_std([float(r["efficiency"]) for r in ok]),
            "fakerate_pct": _mean_std([float(r["fakerate"]) for r in ok]),
            "duplicaterate_pct": _mean_std([float(r["duplicaterate"]) for r in ok]),
            "wall_s_per_event": _mean_std([float(r["wall"]) for r in ok]),
        },
        "definition": (
            "Unweighted mean ± sample stdev (ddof=1) of per-event post-ambi "
            "metrics on events [skip0, skip0+n). Each event is a separate "
            "digi+reco run (events=1). ε_DM=particle efficiency; "
            "f_DM/d_DM=track fakeratio/duplicateratio (DM matching 0.5). "
            "Wall is per-event digi+reco wall time (includes per-run setup)."
        ),
        "per_event": ok,
    }
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    data_vol.commit()
    print(json.dumps({k: summary[k] for k in summary if k != "per_event"}, indent=2))
    return summary


# Joint-MOTPE operating points selected on the eval set (pooled metrics).
JOINT_OP_POINTS: dict[str, dict] = {
    "tight": {  # trial 79 — min f among ε≥90% on 2D (ε,f) Pareto
        "trial": 79,
        "num_seeds_per_spm": 16,
        "seed_minPt": 0.6876416997703818,
        "seed_impactMax": 2.5032955914538624,
        "seed_sigmaScattering": 2.3396240725684057,
        "ckf_chi2CutOffMeasurement": 16.255929655134203,
        "ckf_chi2CutOffOutlier": 20.657664491030744,
        "ckf_numMeasurementsCutOff": 3,
        "ckf_nMeasurementsMin": 9,
        "ckf_maxHolesAndOutliers": 1,
        "ckf_ptMin": 0.4597609399152028,
    },
    "medium": {  # trial 70 — max ε with f<1% on 2D (ε,f) Pareto
        "trial": 70,
        "num_seeds_per_spm": 46,
        "seed_minPt": 0.586901860986303,
        "seed_impactMax": 2.861631736090798,
        "seed_sigmaScattering": 2.9531425900152315,
        "ckf_chi2CutOffMeasurement": 12.044986040316438,
        "ckf_chi2CutOffOutlier": 35.75010517110412,
        "ckf_numMeasurementsCutOff": 2,
        "ckf_nMeasurementsMin": 7,
        "ckf_maxHolesAndOutliers": 1,
        "ckf_ptMin": 0.5942361575936116,
    },
    "fast": {  # trial 331 — min wall on 3D Pareto among ε≥93%, f≤0.5%
        "trial": 331,
        "num_seeds_per_spm": 25,
        "seed_minPt": 0.6706789067203247,
        "seed_impactMax": 0.5807011363616343,
        "seed_sigmaScattering": 2.1464450037640006,
        "ckf_chi2CutOffMeasurement": 15.401021709868644,
        "ckf_chi2CutOffOutlier": 31.873177306280834,
        "ckf_numMeasurementsCutOff": 5,
        "ckf_nMeasurementsMin": 8,
        "ckf_maxHolesAndOutliers": 1,
        "ckf_ptMin": 0.6215760132564532,
    },
    "loose": {  # trial 284 — max ε with f<5% (legacy)
        "trial": 284,
        "num_seeds_per_spm": 47,
        "seed_minPt": 0.5504192505343526,
        "seed_impactMax": 2.9288131499613517,
        "seed_sigmaScattering": 2.0151625112096774,
        "ckf_chi2CutOffMeasurement": 17.936400547956712,
        "ckf_chi2CutOffOutlier": 27.334654122770765,
        "ckf_numMeasurementsCutOff": 1,
        "ckf_nMeasurementsMin": 5,
        "ckf_maxHolesAndOutliers": 1,
        "ckf_ptMin": 0.6162670911627443,
    },
}


@app.function(
    image=image,
    volumes={DATA_PATH: data_vol},
    cpu=2,
    memory=8192,
    timeout=86400,
)
def run_acts_baseline_per_event(
    skip0: int = EVAL_SKIP,
    n_events: int = EVAL_EVENTS,
    max_parallel: int = 8,
    impact_max: float = 5.0,
):
    """Evaluate ACTS-defaults (impactMax clipped) once per event for mean±std.

    Usage:
        modal run --detach modal_acts.py::run_acts_baseline_per_event \\
            --impact-max 5 --max-parallel 8
    """
    config_name = (
        "acts_odd_defaults_impact5" if abs(impact_max - 5.0) < 1e-9 else "acts_odd_defaults"
    )
    params = {
        "seed_impactMax": float(impact_max),
    }
    # Full ACTS-default param vector for the table (impact clipped into search box).
    display_params = {
        "num_seeds_per_spm": 5,
        "seed_minPt": 0.4,
        "seed_impactMax": float(impact_max),
        "seed_sigmaScattering": 5.0,
        "ckf_chi2CutOffMeasurement": 15.0,
        "ckf_chi2CutOffOutlier": 25.0,
        "ckf_numMeasurementsCutOff": 2,
        "ckf_nMeasurementsMin": 7,
        "ckf_maxHolesAndOutliers": 2,  # ACTS uses separate holes/outliers=2/2
        "ckf_ptMin": 1.0,
    }
    out = f"{DATA_PATH}/optimizer/joint_motpe/acts_impact{impact_max:g}_per_event.json"
    summary = _per_event_ckf_summary(
        label=f"acts_impact{impact_max:g}",
        config_name=config_name,
        params=params,
        skip0=skip0,
        n_events=n_events,
        max_parallel=max_parallel,
        out_path=out,
    )
    summary["params"] = display_params
    with open(out, "w") as f:
        import json

        json.dump(summary, f, indent=2)
    data_vol.commit()
    return summary


@app.function(
    image=image,
    volumes={DATA_PATH: data_vol},
    cpu=2,
    memory=8192,
    timeout=86400,
)
def profile_op_points(
    points: str = "tight,medium,fast",
    skip: int = EVAL_SKIP,
    n_events: int = EVAL_EVENTS,
    threads: int = 8,
):
    """Run each operating point once and scrape ACTS seed→tracks profiler times.

    Usage:
        modal run modal_acts.py::profile_op_points --points tight,medium,fast
    """
    import json

    results = {}
    for name in [p.strip() for p in points.split(",") if p.strip()]:
        if name not in JOINT_OP_POINTS:
            raise ValueError(f"Unknown op point {name!r}; choose from {list(JOINT_OP_POINTS)}")
        raw = dict(JOINT_OP_POINTS[name])
        trial = raw.pop("trial")
        overrides = {
            "digi": True,
            "reco": True,
            "threads": threads,
            "skip": skip,
            "ckf_finding_performance": True,
            "ambi_finding_performance": True,
            "ambi_solver": "greedy",
            "ambi_nMeasurementsMin": 6,
            **raw,
        }
        for p in INT_PARAMS:
            if p in overrides:
                overrides[p] = int(round(overrides[p]))
        print(f"=== Profiling {name} (trial {trial}) ===")
        m = run_ckf.remote(
            config="baseline",
            events=n_events,
            config_overrides=json.dumps(overrides),
        )
        if m is None:
            results[name] = {"trial": trial, "ok": False}
            continue
        breakdown = m.get("acts_timing") or {}
        row = {
            "ok": True,
            "trial": trial,
            "event_range": [skip, skip + n_events],
            "threads": threads,
            "efficiency": m.get("efficiency"),
            "fakerate": m.get("fakerate"),
            "duplicaterate": m.get("duplicaterate"),
            "wall_per_event": m.get("wall_per_event"),
            "t_seed_to_tracks": m.get("t_seed_to_tracks"),
            "t_track_finding": breakdown.get("TrackFindingAlgorithm"),
            "t_seeding": breakdown.get("SeedingAlgorithm"),
            "t_ambi": breakdown.get("GreedyAmbiguityResolutionAlgorithm"),
            "acts_timing": breakdown,
            "seed_to_tracks_algos": list(SEED_TO_TRACKS_ALGOS),
            "definition": (
                "t_seed_to_tracks = sum of ACTS time_perevent_s over "
                "SpacePointMaker → Seeding → TrackParamsEstimation → "
                "SeedsToPrototracks → PrototracksToTracks → TrackFinding → "
                "GreedyAmbiguityResolution (excludes digi/IO/truth/writers)."
            ),
        }
        results[name] = row
        print(
            f"  {name}: ε={row['efficiency']:.2f}% f={row['fakerate']:.3f}% "
            f"wall/evt={row['wall_per_event']:.2f}s "
            f"seed→tracks={row['t_seed_to_tracks']:.2f}s/evt "
            f"(CKF={row['t_track_finding']:.2f}, seed={row['t_seeding']:.2f})"
        )

    out = f"{DATA_PATH}/optimizer/joint_motpe/op_points_timing.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    data_vol.commit()
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "acts_timing"}
                      for k, v in results.items()}, indent=2))
    return results


@app.function(
    image=image,
    volumes={DATA_PATH: data_vol},
    cpu=2,
    memory=8192,
    timeout=86400,
)
def run_op_points_matching_scan(
    points: str = "tight,medium,fast",
    skip: int = EVAL_SKIP,
    n_events: int = EVAL_EVENTS,
    threads: int = 8,
    truth_pt_min: float = 0.15,
):
    """Re-run op points with low truth pT floor + matching/particle/track dumps.

    Enables post-hoc ε_DM(f_DM) vs true-pT cutoff. Does NOT change Motpe
    study artifacts; writes under optimizer/joint_motpe/pt_scan/<point>/.

    Usage:
        modal run --detach modal_acts.py::run_op_points_matching_scan \\
            --points tight,medium,fast --truth-pt-min 0.15
    """
    import json
    import os
    import shutil

    results = {}
    for name in [p.strip() for p in points.split(",") if p.strip()]:
        if name not in JOINT_OP_POINTS:
            raise ValueError(f"Unknown op point {name!r}; choose from {list(JOINT_OP_POINTS)}")
        raw = dict(JOINT_OP_POINTS[name])
        trial = raw.pop("trial")
        overrides = {
            "digi": True,
            "reco": True,
            "threads": threads,
            "skip": skip,
            "ckf_finding_performance": True,
            "ambi_finding_performance": True,
            "ambi_solver": "greedy",
            "ambi_nMeasurementsMin": 6,
            "truth_pt_min": float(truth_pt_min),
            "write_matching_details": True,
            "write_track_summary": True,
            "write_particles_selected": True,
            **raw,
        }
        for p in INT_PARAMS:
            if p in overrides:
                overrides[p] = int(round(overrides[p]))
        print(f"=== Matching scan {name} (trial {trial}), truth_pt_min={truth_pt_min} ===")
        m = run_ckf.remote(
            config="baseline",
            events=n_events,
            config_overrides=json.dumps(overrides),
        )
        if m is None:
            results[name] = {"trial": trial, "ok": False}
            continue
        # run_ckf commits the volume in another container; refresh mount
        # so we see results/ before copying into pt_scan/.
        data_vol.reload()
        src = m.get("output_dir")
        dest = f"{DATA_PATH}/optimizer/joint_motpe/pt_scan/{name}_t{trial}"
        os.makedirs(dest, exist_ok=True)
        n_copied = 0
        if src and os.path.isdir(src):
            for fn in os.listdir(src):
                if fn.endswith((".root", ".csv", ".txt", ".tsv")):
                    shutil.copy2(os.path.join(src, fn), os.path.join(dest, fn))
                    n_copied += 1
        if n_copied == 0:
            print(f"  WARNING: no files copied from {src!r} → {dest}")
        row = {
            "ok": True,
            "trial": trial,
            "event_range": [skip, skip + n_events],
            "truth_pt_min": truth_pt_min,
            "efficiency_scalar_1gev_style": m.get("efficiency"),
            "fakerate_scalar": m.get("fakerate"),
            "output_dir": dest,
            "source_output_dir": src,
            "n_files_copied": n_copied,
            "note": (
                "Scalars still use ACTS built-in summary over all particles "
                "passing truth_pt_min; use plot_eff_fake_vs_pt.py for cutoff scans."
            ),
        }
        results[name] = row
        print(f"  {name}: wrote {dest} ({n_copied} files)")

    out = f"{DATA_PATH}/optimizer/joint_motpe/pt_scan/manifest.json"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    data_vol.commit()
    print(json.dumps(results, indent=2))
    return results


@app.function(
    image=image,
    volumes={DATA_PATH: data_vol},
    cpu=2,
    memory=8192,
    timeout=86400,
)
def run_op_points_per_event(
    points: str = "tight,medium,loose",
    skip0: int = EVAL_SKIP,
    n_events: int = EVAL_EVENTS,
    max_parallel: int = 8,
):
    """Per-event mean±std for joint-MOTPE operating points on the eval split.

    Usage:
        modal run --detach modal_acts.py::run_op_points_per_event \\
            --points tight,medium,loose --max-parallel 8
    """
    import json

    results = {}
    for name in [p.strip() for p in points.split(",") if p.strip()]:
        if name not in JOINT_OP_POINTS:
            raise ValueError(f"Unknown op point {name!r}; choose from {list(JOINT_OP_POINTS)}")
        raw = dict(JOINT_OP_POINTS[name])
        trial = raw.pop("trial")
        out = f"{DATA_PATH}/optimizer/joint_motpe/{name}_t{trial}_per_event.json"
        summary = _per_event_ckf_summary(
            label=f"{name}_t{trial}",
            config_name="baseline",
            params=raw,
            skip0=skip0,
            n_events=n_events,
            max_parallel=max_parallel,
            out_path=out,
        )
        summary["trial"] = trial
        with open(out, "w") as f:
            json.dump(summary, f, indent=2)
        data_vol.commit()
        results[name] = {k: summary[k] for k in summary if k != "per_event"}
    return results


# ─── Launch (use --detach to survive disconnection) ────────────────────


@app.local_entrypoint()
def launch_ckf_studies(
    seeding_points: str = "tight,medium,loose",
    n_trials: int = 100,
    events_per_trial: int = 8,
    threads_per_trial: int = 8,
    max_parallel_trials: int = 4,
    input_file: str = None,
):
    """Launch CKF optimizer runs for multiple seeding points.

    IMPORTANT: Use --detach so the runs survive local disconnection:

        modal run --detach modal_acts.py::app.launch_ckf_studies
        modal run --detach modal_acts.py::app.launch_ckf_studies \\
            --seeding-points medium --n-trials 50

    Without --detach, Ctrl-C or disconnect kills the remote jobs.

    Monitor progress at https://modal.com/apps
    """
    points = [p.strip() for p in seeding_points.split(",")]
    handles = []
    for sp in points:
        if sp not in SEEDING_POINTS:
            print(f"Skipping unknown seeding point: {sp}")
            continue
        handle = run_ckf_optimizer.spawn(
            seeding_point=sp,
            n_trials=n_trials,
            events_per_trial=events_per_trial,
            threads_per_trial=threads_per_trial,
            max_parallel_trials=max_parallel_trials,
            input_file=input_file,
        )
        print(f"Spawned CKF optimizer for '{sp}' -> {handle.object_id}")
        handles.append((sp, handle))

    print(f"\n{len(handles)} jobs spawned.")
    print("Monitor at: https://modal.com/apps")
    print("Download results: modal run modal_acts.py::app.download")

    # Block on all handles so --detach keeps the app alive until completion
    for sp, handle in handles:
        result = handle.get()
        print(f"\n=== {sp} complete: {result}")


# ─── Download Results ──────────────────────────────────────────────────


@app.local_entrypoint()
def download(output_dir: str = "./results"):
    """Download results from the Modal volume to local filesystem."""
    import os

    os.makedirs(output_dir, exist_ok=True)

    # List and download key result files
    prefixes = ["/results", "/optimizer"]
    for prefix in prefixes:
        try:
            for item in data_vol.listdir(prefix):
                if hasattr(item, "path"):
                    rel = item.path.lstrip("/")
                    local_path = os.path.join(output_dir, rel)
                    os.makedirs(os.path.dirname(local_path), exist_ok=True)
                    print(f"  {item.path} -> {local_path}")
                    with open(local_path, "wb") as f:
                        for chunk in data_vol.read_file(item.path):
                            f.write(chunk)
        except Exception as e:
            print(f"  Skipping {prefix}: {e}")

    print(f"Downloaded to {output_dir}/")


# ─── Gate pilot (§6.6 checks 1–2) ───────────────────────────────────────


def _jsonable(o):
    """Convert numpy scalars so json.dump does not choke."""
    import numpy as np

    if isinstance(o, dict):
        return {k: _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    return o


def _run_pipeline_to_dir(
    *,
    config_name: str,
    output_dir: str,
    events: int,
    skip: int,
    input_file: str | None,
    overrides: dict,
) -> str:
    """Run digi_and_reco into output_dir; return that path."""
    import json
    import os
    import sys
    import time
    import types
    from pathlib import Path

    import yaml

    _setup_acts_env()
    sys.path.insert(0, "/app")

    material_map = f"{DATA_PATH}/odd-material-maps.root"
    if not os.path.exists(material_map):
        raise FileNotFoundError(
            "Material maps not found. Run seed_material_maps first."
        )
    odd_data_dir = f"{ODD_INSTALL}/data"
    os.makedirs(odd_data_dir, exist_ok=True)
    map_link = f"{odd_data_dir}/odd-material-maps.root"
    if not os.path.exists(map_link):
        os.symlink(material_map, map_link)

    config_path = f"/app/configs/{config_name}.yaml"
    with open(config_path) as f:
        cfg_dict = yaml.safe_load(f)
    cfg_dict.update(overrides)
    cfg_dict["events"] = events
    cfg_dict["skip"] = skip

    # Phase 0 loc0 guard
    if cfg_dict.get("forbid_loc0_cut", True) and cfg_dict.get("ckf_loc0_max") is not None:
        raise ValueError("loc0 cut present under forbid_loc0_cut — abort")

    cfg = types.SimpleNamespace(**cfg_dict)
    if not hasattr(cfg, "seed"):
        cfg.seed = 42
    if not hasattr(cfg, "threads"):
        cfg.threads = 8

    edm4hep_path = (
        f"{DATA_PATH}/{input_file}"
        if input_file
        else f"{DATA_PATH}/events/edm4hep.root"
    )
    if not os.path.exists(edm4hep_path):
        raise FileNotFoundError(f"Input not found: {edm4hep_path}")

    os.makedirs(output_dir, exist_ok=True)
    print(
        f"=== Pilot pipeline config={config_name} events={events} skip={skip} "
        f"out={output_dir} overrides={json.dumps(overrides)} ==="
    )
    t0 = time.time()

    import acts
    from digi_and_reco import setup_acts_reconstruction

    rnd = acts.examples.RandomNumbers(seed=cfg.seed)
    s = setup_acts_reconstruction(Path(edm4hep_path), Path(output_dir), cfg, rnd)
    s.run()
    print(f"Pipeline wall: {time.time() - t0:.1f}s -> {output_dir}")
    data_vol.commit()
    return output_dir


@app.function(
    image=image,
    volumes={DATA_PATH: data_vol},
    cpu=8,
    memory=65536,
    timeout=7200,
)
def run_gate_pilot(
    events: int = 2,
    skip: int = 0,
    input_file: str = None,
    digi_variant: str = "geometric",
):
    """Phase-1 pilot: envelope digi+seeds, Tight seeds, checks 1–2, schema skeleton.

    Usage:
        modal run modal_acts.py::run_gate_pilot --events 2

    digi_variant: 'geometric' (default, required for cluster feats) or 'smearing'
    (control — expected to fail check 1).
    """
    import json
    import os
    import sys
    import time
    from pathlib import Path

    import yaml

    sys.path.insert(0, "/app")
    sys.path.insert(0, "/app/scripts")

    run_id = f"gate_pilot_{int(time.time())}"
    base = f"{DATA_PATH}/results/{run_id}"
    os.makedirs(base, exist_ok=True)

    digi_config = (
        "odd-digi-geometric-config.json"
        if digi_variant == "geometric"
        else "odd-digi-smearing-config.json"
    )

    # Seeding-only for checks 1–2 (CKF not required). Terminal cuts already
    # disabled in envelope.yaml; Tight keeps its cuts but ckf is skipped.
    common = {
        "digi_config": digi_config,
        "output_digi_csv": True,
        "output_seeds_csv": True,
        "ckf": False,
        "window_n": 10,
        "forbid_loc0_cut": True,
    }

    env_dir = _run_pipeline_to_dir(
        config_name="envelope",
        output_dir=f"{base}/envelope",
        events=events,
        skip=skip,
        input_file=input_file,
        overrides=common,
    )
    tight_dir = _run_pipeline_to_dir(
        config_name="tight_t79",
        output_dir=f"{base}/tight_t79",
        events=events,
        skip=skip,
        input_file=input_file,
        overrides=common,
    )

    from probe_cluster_features import probe_digi_dir
    from seed_recovery import probe_seed_recovery
    from utils.decision_log import assert_schema, write_skeleton_example

    event_ids = list(range(skip, skip + events))

    check1 = probe_digi_dir(Path(env_dir), event_ids)
    with open(f"/app/configs/tight_t79.yaml") as f:
        tight_cfg = yaml.safe_load(f)
    check2 = probe_seed_recovery(
        Path(tight_dir),
        Path(env_dir),
        event_ids,
        min_pt=float(tight_cfg["seed_minPt"]),
        max_seeds_per_spm=int(tight_cfg["num_seeds_per_spm"]),
    )

    with open(f"/app/configs/envelope.yaml") as f:
        env_cfg = yaml.safe_load(f)
    env_cfg.update(common)
    # Confirm loc0 absent
    loc0_absent = env_cfg.get("ckf_loc0_max") is None and env_cfg.get(
        "forbid_loc0_cut", False
    )
    skeleton_path = str(
        write_skeleton_example(f"{base}/decision_log_skeleton.parquet", env_cfg)
    )
    schema_cols = assert_schema(skeleton_path)

    report = {
        "run_id": run_id,
        "events": event_ids,
        "digi_variant": digi_variant,
        "digi_config": digi_config,
        "window_n": 10,
        "window_n_status": "PILOT ONLY — [OPEN] do not freeze into env_config_hash yet",
        "loc0_cut_absent": loc0_absent,
        "canonical_data_location": (
            "[OPEN] Project data lives at NERSC m4958; this pilot used Modal "
            "volume surp-acts-data. Declare which is authoritative before full "
            "collection."
        ),
        "check1_cluster_features": check1,
        "check2_seed_recovery": check2,
        "decision_log_skeleton": {
            "path": skeleton_path,
            "n_columns": len(schema_cols),
            "columns": schema_cols,
        },
        "envelope_dir": env_dir,
        "tight_dir": tight_dir,
        "schedule_note": (
            "Design-doc W2 deliverable was gate v0 + G3 (Jul 20–26). This pilot "
            "is Jul 28 — that milestone has slipped; poster hard deadline Aug 28. "
            "Do not invent Phase-2 wall time until pilot check 3."
        ),
        "phase5_decision_for_matthew_rocky": (
            "An undated 'deferred' bucket containing DAgger / V_phi / ILP is a de "
            "facto cut of never-cut Stage 0 and A1–A3. Scope-cut order is "
            "A8→A4c→A9→A6→Stage-A v1. Resolve explicitly; do not leave undated."
        ),
    }

    out_json = f"{base}/pilot_report.json"
    report = _jsonable(report)
    with open(out_json, "w") as f:
        json.dump(report, f, indent=2)
    data_vol.commit()

    print("\n========== GATE PILOT REPORT ==========")
    print(json.dumps(report, indent=2))
    print(f"Wrote {out_json}")
    print(
        f"CHECK 1 (cluster features): {'PASS' if check1.get('pass') else 'FAIL'}"
    )
    print(
        f"CHECK 2 (seed recovery ≥98%): {'PASS' if check2.get('pass') else 'FAIL'} "
        f"(mean={check2.get('mean_frac_recovered')})"
    )
    return report


@app.local_entrypoint()
def gate_pilot(events: int = 2, digi_variant: str = "geometric", detach: bool = False):
    """Local entrypoint wrapper for run_gate_pilot.

    Use ``--detach`` (or ``modal run --detach``) so a local heartbeat blip does
    not cancel the remote after digi/seeding finishes.
    """
    if detach:
        handle = run_gate_pilot.spawn(events=events, digi_variant=digi_variant)
        print(f"Spawned run_gate_pilot -> {handle.object_id}")
        print("Monitor: https://modal.com/apps")
        report = handle.get()
    else:
        report = run_gate_pilot.remote(events=events, digi_variant=digi_variant)
    print(
        "Done.",
        "check1=",
        report["check1_cluster_features"]["pass"],
        "check2=",
        report["check2_seed_recovery"]["pass"],
    )
    return report


@app.function(
    image=image,
    volumes={DATA_PATH: data_vol},
    cpu=2,
    memory=8192,
    timeout=600,
)
def analyze_gate_pilot(run_id: str, events: int = 2, skip: int = 0):
    """Re-run checks 1–2 on an existing gate_pilot_* directory (no ACTS)."""
    import json
    import sys
    from pathlib import Path

    import yaml

    sys.path.insert(0, "/app")
    sys.path.insert(0, "/app/scripts")
    from probe_cluster_features import probe_digi_dir
    from seed_recovery import probe_seed_recovery
    from utils.decision_log import assert_schema, write_skeleton_example

    base = f"{DATA_PATH}/results/{run_id}"
    env_dir = f"{base}/envelope"
    tight_dir = f"{base}/tight_t79"
    event_ids = list(range(skip, skip + events))

    check1 = probe_digi_dir(Path(env_dir), event_ids)
    with open("/app/configs/tight_t79.yaml") as f:
        tight_cfg = yaml.safe_load(f)
    check2 = probe_seed_recovery(
        Path(tight_dir),
        Path(env_dir),
        event_ids,
        min_pt=float(tight_cfg["seed_minPt"]),
        max_seeds_per_spm=int(tight_cfg["num_seeds_per_spm"]),
    )
    with open("/app/configs/envelope.yaml") as f:
        env_cfg = yaml.safe_load(f)
    skeleton_path = str(
        write_skeleton_example(f"{base}/decision_log_skeleton.parquet", env_cfg)
    )
    schema_cols = assert_schema(skeleton_path)
    report = {
        "run_id": run_id,
        "check1_cluster_features": check1,
        "check2_seed_recovery": check2,
        "decision_log_skeleton": {
            "path": skeleton_path,
            "n_columns": len(schema_cols),
            "columns": schema_cols,
        },
        "loc0_cut_absent": env_cfg.get("ckf_loc0_max") is None
        and bool(env_cfg.get("forbid_loc0_cut", False)),
    }
    out_json = f"{base}/pilot_report.json"
    report = _jsonable(report)
    with open(out_json, "w") as f:
        json.dump(report, f, indent=2)
    data_vol.commit()
    print(json.dumps(report, indent=2))
    print(
        f"CHECK 1: {'PASS' if report['check1_cluster_features']['pass'] else 'FAIL'}"
    )
    print(
        f"CHECK 2: {'PASS' if report['check2_seed_recovery']['pass'] else 'FAIL'} "
        f"(mean={report['check2_seed_recovery'].get('mean_frac_recovered')})"
    )
    return report


# ─── χ² gate calibration (model-free, Tier 3) ───────────────────────────


def _expand_chi2_event_dir(event_id: int, out_dir: str) -> dict:
    """Expand digi+trackstates in ``out_dir`` → slim Parquet (no digi/CKF)."""
    import glob
    import os
    import sys
    from pathlib import Path

    sys.path.insert(0, "/app")
    sys.path.insert(0, "/app/scripts")
    from expand_trackstates_to_chi2_rows import expand_event, write_parquet

    def _one(pattern: str) -> Path:
        hits = sorted(glob.glob(f"{out_dir}/{pattern}"))
        if not hits:
            raise FileNotFoundError(f"No files matching {out_dir}/{pattern}")
        return Path(hits[0])

    trackstates = Path(out_dir) / "trackstates_ckf.root"
    if not trackstates.exists():
        trackstates = _one("trackstates*.root")

    cov_hits = sorted(glob.glob(f"{out_dir}/event*-predicted-cov.csv"))
    predicted_cov = Path(cov_hits[0]) if cov_hits else None
    df = expand_event(
        event_id=event_id,
        trackstates_path=trackstates,
        measurements_path=_one("event*-measurements.csv"),
        map_path=_one("event*-measurement-simhit-map.csv"),
        simhits_path=_one("event*-simhits.csv"),
        predicted_cov_path=predicted_cov,
        window_n=10.0,
        r_geom_mm=5.0,
        # Envelope+ambi-off writes ~2e6 candidates/event; full expand ≫40 GB.
        max_tracks=25_000,
        sample_seed=42,
    )
    parquet_path = Path(out_dir) / "chi2_calib_rows.parquet"
    write_parquet(df, parquet_path)
    for pat in ("event*-cells.csv", "timing.csv"):
        for p in glob.glob(f"{out_dir}/{pat}"):
            try:
                os.remove(p)
            except OSError:
                pass
    size_mb = parquet_path.stat().st_size / 1e6
    print(
        f"event {event_id}: {len(df)} rows, {size_mb:.1f} MB -> {parquet_path}"
    )
    return {
        "event_id": event_id,
        "n_rows": int(len(df)),
        "size_mb": size_mb,
        "path": str(parquet_path),
    }


@app.function(
    image=image,
    volumes={DATA_PATH: data_vol},
    cpu=8,
    memory=65536,
    timeout=14400,
)
def expand_chi2_calib_event(event_id: int, run_id: str):
    """Re-expand existing digi+trackstates for one event (no digi/CKF)."""
    if not (0 <= event_id < 32):
        raise ValueError(
            f"event_id={event_id} outside train/cal range [0,32); "
            "held-out [32,64) is sealed"
        )
    out_dir = f"{DATA_PATH}/results/{run_id}/event_{event_id:03d}"
    result = _expand_chi2_event_dir(event_id, out_dir)
    data_vol.commit()
    return result


def _trackstates_usable(path: "Path") -> bool:
    """True if ROOT trackstates file exists and has ≥1 track entry."""
    try:
        import uproot

        if not path.exists() or path.stat().st_size < 1000:
            return False
        with uproot.open(path) as f:
            keys = list(f.keys())
            if not keys:
                return False
            tree_key = next(
                (
                    k
                    for k in keys
                    if "trackstate" in k.lower() or "states" in k.lower()
                ),
                keys[0],
            )
            return int(f[tree_key].num_entries) > 0
    except Exception as exc:
        print(f"trackstates unusable ({path}): {exc}")
        return False


@app.function(
    image=image,
    volumes={DATA_PATH: data_vol},
    cpu=8,
    memory=98304,
    timeout=14400,
)
def collect_chi2_calib_event(event_id: int, run_id: str):
    """Envelope digi+CKF for one event → slim chi²-calib Parquet.

    Events must be in [0, 32). Held-out [32, 64) is refused.
    Skips digi+CKF if trackstates already exist (expand-only retry).
    """
    import os
    from pathlib import Path

    if not (0 <= event_id < 32):
        raise ValueError(
            f"event_id={event_id} outside train/cal range [0,32); "
            "held-out [32,64) is sealed"
        )

    out_dir = f"{DATA_PATH}/results/{run_id}/event_{event_id:03d}"
    parquet_path = Path(out_dir) / "chi2_calib_rows.parquet"
    if parquet_path.exists() and parquet_path.stat().st_size > 1000:
        import pyarrow.parquet as pq

        n_rows = pq.read_metadata(parquet_path).num_rows
        size_mb = parquet_path.stat().st_size / 1e6
        print(
            f"event {event_id}: reusing existing parquet "
            f"({n_rows} rows, {size_mb:.1f} MB)"
        )
        return {
            "event_id": event_id,
            "n_rows": int(n_rows),
            "size_mb": size_mb,
            "path": str(parquet_path),
        }

    trackstates = Path(out_dir) / "trackstates_ckf.root"
    if _trackstates_usable(trackstates):
        print(f"event {event_id}: reusing existing {trackstates}")
    else:
        if trackstates.exists():
            print(f"event {event_id}: removing unusable {trackstates}")
            try:
                os.remove(trackstates)
            except OSError:
                pass
        overrides = {
            "digi_config": "odd-digi-geometric-config.json",
            "output_digi_csv": True,
            "output_simhits_csv": True,
            "output_seeds_csv": True,
            "write_track_states": True,
            "write_predicted_cov": True,
            "ckf": True,
            "ambi": False,
            "ckf_finding_performance": False,
            "ambi_finding_performance": False,
            "window_n": 10,
            "forbid_loc0_cut": True,
            # Tractability cap for this dump only (logged). Envelope YAML keeps
            # terminal cuts disabled for V_φ collection; χ²-gate calibration only
            # needs states visited before hole-out, so a generous hole/outlier
            # floor keeps the ACTS CKF finite under ~5e5 seeds/event.
            "ckf_maxHolesAndOutliers": 6,
            "events": 1,
            "skip": event_id,
            "threads": 8,
        }
        _run_pipeline_to_dir(
            config_name="envelope",
            output_dir=out_dir,
            events=1,
            skip=event_id,
            input_file=None,
            overrides=overrides,
        )

    result = _expand_chi2_event_dir(event_id, out_dir)
    data_vol.commit()
    return result


@app.function(
    image=image,
    volumes={DATA_PATH: data_vol},
    cpu=4,
    memory=32768,
    timeout=7200,
)
def analyze_chi2_calib(run_id: str, n_boot: int = 1000):
    """Bootstrap χ²-gate calibration analysis on collected slim Parquets."""
    import glob
    import json
    import sys
    from pathlib import Path

    sys.path.insert(0, "/app")
    sys.path.insert(0, "/app/scripts")

    from analyze_chi2_gate_calibration import analyze_strata, summarize

    base = Path(f"{DATA_PATH}/results/{run_id}")
    paths = sorted(base.glob("event_*/chi2_calib_rows.parquet"))
    if not paths:
        raise FileNotFoundError(f"No chi2_calib_rows.parquet under {base}")

    import pandas as pd
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    # Filter at read time — ~99.7% of slim rows are majority_undefined.
    frames = []
    for p in paths:
        table = pq.read_table(p)
        if "majority_undefined" in table.column_names:
            mask = pc.equal(table["majority_undefined"], pa.scalar(0))
            table = table.filter(mask)
        frames.append(table.to_pandas())
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    print(
        f"Loaded {len(paths)} events, {len(df)} rows after majority_undefined cut"
    )

    out_dir = base / "analysis"
    results = analyze_strata(df, out_dir, n_boot=n_boot)
    summary = summarize(results)
    print(summary)
    (out_dir / "SUMMARY.txt").write_text(summary + "\n")
    compact = {
        "run_id": run_id,
        "n_events": int(df["event_id"].nunique()),
        "n_rows": int(len(df)),
        "geometric_density_radius_mm": 5.0,
        "window_n": 10,
        "global_geom": results["global"]["geom"],
        "global_window_headline": results["global"]
        .get("window", {})
        .get("headline_logodds_sep_Q1_minus_Q5"),
        "layer_headlines": {
            k: v["headline_logodds_sep_Q1_minus_Q5"]
            for k, v in results["by_layer"].items()
        },
        "eta_headlines": {
            k: v["headline_logodds_sep_Q1_minus_Q5"]
            for k, v in results["by_eta"].items()
        },
        "summary": summary,
        "held_out_sealed": "[32,64) never opened",
    }
    (out_dir / "SUMMARY.json").write_text(json.dumps(compact, indent=2))
    data_vol.commit()
    return compact


@app.function(
    image=image,
    volumes={DATA_PATH: data_vol},
    cpu=1,
    memory=2048,
    timeout=86400,
)
def run_chi2_gate_calib_orchestrator(
    n_events: int = 32,
    max_parallel: int = 4,
    n_boot: int = 1000,
    resume_run_id: str = "",
):
    """Remote orchestrator so local client disconnects cannot cancel collection."""
    import time

    if n_events > 32:
        raise ValueError("n_events > 32 would enter the sealed held-out range")
    run_id = resume_run_id or f"chi2_gate_calib_{int(time.time())}"
    print(f"run_id={run_id} events=[0,{n_events}) max_parallel={max_parallel}")
    print("geometric_density radius = 5.0 mm; window_n = 10")
    print("tractability: ckf_maxHolesAndOutliers=6 for this dump only")
    if resume_run_id:
        print(f"resuming: reuse digi/trackstates under {resume_run_id} when present")

    results = []
    failures = []
    for wave_start in range(0, n_events, max_parallel):
        wave = list(range(wave_start, min(wave_start + max_parallel, n_events)))
        print(f"Spawning events {wave} ...")
        handles = [
            (i, collect_chi2_calib_event.spawn(event_id=i, run_id=run_id))
            for i in wave
        ]
        for i, h in handles:
            try:
                r = h.get()
                print(
                    f"  event {r['event_id']}: {r['n_rows']} rows "
                    f"({r['size_mb']:.1f} MB)"
                )
                results.append(r)
            except Exception as exc:
                print(f"  EVENT {i} FAILED: {exc}")
                failures.append({"event_id": i, "error": str(exc)})

    if failures:
        print(f"Retrying {len(failures)} failed events once ...")
        for item in list(failures):
            i = item["event_id"]
            try:
                r = collect_chi2_calib_event.remote(event_id=i, run_id=run_id)
                print(
                    f"  retry event {r['event_id']}: {r['n_rows']} rows "
                    f"({r['size_mb']:.1f} MB)"
                )
                results.append(r)
                failures.remove(item)
            except Exception as exc:
                print(f"  RETRY EVENT {i} FAILED: {exc}")
                item["error"] = str(exc)

    if failures:
        raise RuntimeError(
            f"{len(failures)} events still failed after retry: {failures}"
        )

    total_rows = sum(r["n_rows"] for r in results)
    total_mb = sum(r["size_mb"] for r in results)
    print(f"Collection done: {total_rows} rows, {total_mb:.1f} MB slim Parquet")

    report = analyze_chi2_calib.remote(run_id=run_id, n_boot=n_boot)
    report["run_id"] = run_id
    report["collection"] = {
        "n_events": n_events,
        "total_rows": total_rows,
        "total_mb": total_mb,
        "tractability_maxHolesAndOutliers": 6,
    }
    print(report.get("summary", report))
    print(f"Artifacts: /data/results/{run_id}/analysis/")
    return report


@app.local_entrypoint()
def chi2_gate_calib(
    n_events: int = 32,
    max_parallel: int = 4,
    n_boot: int = 1000,
    smoke: bool = False,
    resume_run_id: str = "",
):
    """Collect [0, n_events) under envelope + analyze χ²-gate calibration.

    Prefer: ``modal run --detach modal_acts.py::chi2_gate_calib --n-events 32``
    """
    if smoke:
        n_events = 2
    report = run_chi2_gate_calib_orchestrator.remote(
        n_events=n_events,
        max_parallel=max_parallel,
        n_boot=n_boot,
        resume_run_id=resume_run_id,
    )
    print(report.get("summary", report))
    return report


@app.function(
    image=image,
    volumes={DATA_PATH: data_vol},
    cpu=8,
    memory=65536,
    timeout=7200,
)
def collect_calib_medium_event(event_id: int, run_id: str):
    """Medium op-point CKF + expand to all-candidates slim Parquet.

    Same pipeline as collect_chi2_calib_event but uses medium_t70 config
    (spec §3.2: chi2_meas=12.04, nMeasMin=7, maxHolesAndOutliers=1).
    Events must be in [0, 32).
    """
    import os
    from pathlib import Path

    if not (0 <= event_id < 32):
        raise ValueError(
            f"event_id={event_id} outside train/cal range [0,32)"
        )

    out_dir = f"{DATA_PATH}/results/{run_id}/event_{event_id:03d}"
    parquet_path = Path(out_dir) / "chi2_calib_rows.parquet"
    if parquet_path.exists() and parquet_path.stat().st_size > 1000:
        import pyarrow.parquet as pq

        n_rows = pq.read_metadata(parquet_path).num_rows
        size_mb = parquet_path.stat().st_size / 1e6
        print(
            f"event {event_id}: reusing existing parquet "
            f"({n_rows} rows, {size_mb:.1f} MB)"
        )
        return {
            "event_id": event_id,
            "n_rows": int(n_rows),
            "size_mb": size_mb,
            "path": str(parquet_path),
        }

    trackstates = Path(out_dir) / "trackstates_ckf.root"
    if _trackstates_usable(trackstates):
        print(f"event {event_id}: reusing existing {trackstates}")
    else:
        if trackstates.exists():
            try:
                os.remove(trackstates)
            except OSError:
                pass
        overrides = {
            "write_track_states": True,
            "write_predicted_cov": True,
            "output_digi_csv": True,
            "output_simhits_csv": True,
            "ckf": True,
            "ambi": False,
            "ckf_finding_performance": False,
            "ambi_finding_performance": False,
            "forbid_loc0_cut": True,
            "events": 1,
            "skip": event_id,
            "threads": 8,
        }
        _run_pipeline_to_dir(
            config_name="medium_t70",
            output_dir=out_dir,
            events=1,
            skip=event_id,
            input_file=None,
            overrides=overrides,
        )

    # Expand: find all candidates in window, recompute chi2, assign labels
    result = _expand_chi2_event_dir(event_id, out_dir)
    data_vol.commit()
    return result


@app.function(
    image=image,
    volumes={DATA_PATH: data_vol},
    cpu=1,
    memory=2048,
    timeout=86400,
)
def run_calib_medium_orchestrator(
    n_events: int = 32,
    max_parallel: int = 4,
    resume_run_id: str = "",
):
    """Orchestrate Medium CKF + expansion on [0, n_events) for reliability diagrams."""
    import time

    if n_events > 32:
        raise ValueError("n_events > 32 would enter the sealed held-out range")
    run_id = resume_run_id or f"calib_medium_{int(time.time())}"
    print(f"run_id={run_id} events=[0,{n_events}) max_parallel={max_parallel}")
    print("config=medium_t70 (chi2_meas=12.04, nMeasMin=7, maxHO=1)")

    results = []
    failures = []
    for wave_start in range(0, n_events, max_parallel):
        wave = list(range(wave_start, min(wave_start + max_parallel, n_events)))
        print(f"Spawning events {wave} ...")
        handles = [
            (i, collect_calib_medium_event.spawn(event_id=i, run_id=run_id))
            for i in wave
        ]
        for i, h in handles:
            try:
                r = h.get()
                print(
                    f"  event {r['event_id']}: {r['n_rows']} rows "
                    f"({r['size_mb']:.1f} MB)"
                )
                results.append(r)
            except Exception as exc:
                print(f"  EVENT {i} FAILED: {exc}")
                failures.append({"event_id": i, "error": str(exc)})

    if failures:
        print(f"Retrying {len(failures)} failed events once ...")
        for item in list(failures):
            i = item["event_id"]
            try:
                r = collect_calib_medium_event.remote(event_id=i, run_id=run_id)
                results.append(r)
                failures.remove(item)
            except Exception as exc:
                item["error"] = str(exc)

    if failures:
        raise RuntimeError(f"{len(failures)} events still failed: {failures}")

    total_rows = sum(r["n_rows"] for r in results)
    total_mb = sum(r["size_mb"] for r in results)
    print(
        f"Collection done: {total_rows} rows, {total_mb:.1f} MB across "
        f"{len(results)} events"
    )
    return {
        "run_id": run_id,
        "n_events": len(results),
        "total_rows": total_rows,
        "total_mb": total_mb,
    }


@app.local_entrypoint()
def calib_medium(
    n_events: int = 32,
    max_parallel: int = 4,
    smoke: bool = False,
    resume_run_id: str = "",
):
    """Collect Medium CKF all-candidates data for reliability diagrams.

    Usage: modal run --detach modal_acts.py::calib_medium --n-events 32
    """
    if smoke:
        n_events = 2
    report = run_calib_medium_orchestrator.remote(
        n_events=n_events,
        max_parallel=max_parallel,
        resume_run_id=resume_run_id,
    )
    print(report)
    return report


@app.local_entrypoint()
def expand_and_analyze_chi2_calib(
    run_id: str,
    n_events: int = 2,
    max_parallel: int = 2,
    n_boot: int = 200,
):
    """Expand existing digi+trackstates under run_id, then analyze.

    Use after digi/CKF succeeded but expand failed (e.g. nested particle_ids).
    """
    if n_events > 32:
        raise ValueError("n_events > 32 would enter the sealed held-out range")
    print(f"expand+analyze run_id={run_id} events=[0,{n_events})")
    results = []
    for wave_start in range(0, n_events, max_parallel):
        wave = list(range(wave_start, min(wave_start + max_parallel, n_events)))
        handles = [
            expand_chi2_calib_event.spawn(event_id=i, run_id=run_id) for i in wave
        ]
        for h in handles:
            r = h.get()
            print(
                f"  event {r['event_id']}: {r['n_rows']} rows ({r['size_mb']:.1f} MB)"
            )
            results.append(r)
    report = analyze_chi2_calib.remote(run_id=run_id, n_boot=n_boot)
    report["run_id"] = run_id
    report["collection"] = {
        "n_events": n_events,
        "total_rows": sum(r["n_rows"] for r in results),
        "total_mb": sum(r["size_mb"] for r in results),
        "expand_only": True,
    }
    print(report.get("summary", report))
    return report


@app.function(
    image=image,
    volumes={DATA_PATH: data_vol},
    cpu=4,
    memory=32768,
    timeout=3600,
)
def diagnose_chi2_diag_approx(run_id: str):
    """Run diagonal-vs-full χ² diagnostic on slim Parquets under run_id."""
    import json
    import sys
    from pathlib import Path

    sys.path.insert(0, "/app")
    sys.path.insert(0, "/app/scripts")
    from diagnose_chi2_diag_approx import load_df, run

    base = Path(f"{DATA_PATH}/results/{run_id}")
    paths = sorted(base.glob("event_*/chi2_calib_rows.parquet"))
    if not paths:
        raise FileNotFoundError(f"No parquets under {base}")
    df = load_df(paths)
    if "majority_undefined" in df.columns:
        df = df[df["majority_undefined"].astype(int) == 0].copy()
    report = run(df)
    out_dir = base / "diag_chi2_approx"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "DIAGNOSTIC.json").write_text(json.dumps(report, indent=2))
    data_vol.commit()
    print(json.dumps(report, indent=2))
    return report


@app.function(
    image=image,
    volumes={DATA_PATH: data_vol},
    cpu=8,
    memory=65536,
    timeout=7200,
)
def diagnose_chi2_from_trackstates(
    run_id: str = "chi2_gate_calib_1785440155",
    event_ids: str = "0,1,2,3",
    max_tracks: int = 50000,
):
    """Confound check using ACTS state.chi2 vs diagonal pulls on trackstates.

    Works on existing dumps that lack S01 in the slim Parquet.
    """
    import json
    import sys
    from pathlib import Path

    sys.path.insert(0, "/app")
    sys.path.insert(0, "/app/scripts")
    from diagnose_chi2_from_trackstates import analyze, collect_event

    base = Path(f"{DATA_PATH}/results/{run_id}")
    ids = [int(x) for x in event_ids.split(",") if x.strip() != ""]
    frames = []
    for eid in ids:
        ed = base / f"event_{eid:03d}"
        ts = ed / "trackstates_ckf.root"
        if not ts.exists():
            raise FileNotFoundError(f"missing {ts}")
        meas = next(ed.glob("event*-measurements.csv"))
        print(f"collecting event {eid} from {ed}", flush=True)
        frames.append(
            collect_event(eid, ts, meas, max_tracks=max_tracks)
        )
    import pandas as pd

    df = pd.concat(frames, ignore_index=True)
    report = analyze(df)
    report["run_id"] = run_id
    report["event_ids"] = ids
    out_dir = base / "diag_chi2_approx"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "DIAGNOSTIC_trackstates.json").write_text(
        json.dumps(report, indent=2)
    )
    data_vol.commit()
    print(json.dumps(report, indent=2))
    return report


@app.local_entrypoint()
def chi2_cov_diagnostic(
    run_id: str = "chi2_gate_calib_1785440155",
    event_ids: str = "0,1,2,3",
    max_tracks: int = 50000,
):
    """Local entry: diagonal-approx confound check on existing trackstates."""
    report = diagnose_chi2_from_trackstates.remote(
        run_id=run_id, event_ids=event_ids, max_tracks=max_tracks
    )
    print("VERDICT:", report.get("verdict"))
    return report


@app.local_entrypoint()
def chi2_diag_approx_diagnostic(
    n_events: int = 2,
    max_parallel: int = 2,
    max_tracks: int = 5000,
):
    """Fresh short rollout with full S + diagonal-approx diagnostic.

    Does not touch held-out [32,64). Uses a new run_id.
    """
    import json
    import time

    if n_events > 32:
        raise ValueError("n_events > 32 would enter sealed range")
    run_id = f"chi2_diag_approx_{int(time.time())}"
    print(f"diagnostic run_id={run_id} events=[0,{n_events})")
    print("logging S00,S01,S11; chi2_inc = full rᵀS⁻¹r")

    # Temporarily lower max_tracks via env is not wired; use collect as-is
    # (25k) — diagnostic only needs defined-majority subset.
    results = []
    for wave_start in range(0, n_events, max_parallel):
        wave = list(range(wave_start, min(wave_start + max_parallel, n_events)))
        handles = [
            collect_chi2_calib_event.spawn(event_id=i, run_id=run_id) for i in wave
        ]
        for h in handles:
            r = h.get()
            print(
                f"  event {r['event_id']}: {r['n_rows']} rows ({r['size_mb']:.1f} MB)"
            )
            results.append(r)

    report = diagnose_chi2_diag_approx.remote(run_id=run_id)
    report["run_id"] = run_id
    report["collection"] = {
        "n_events": n_events,
        "total_rows": sum(r["n_rows"] for r in results),
        "total_mb": sum(r["size_mb"] for r in results),
    }
    print("DIAGNOSTIC COMPLETE")
    print(json.dumps(report.get("verdict", report), indent=2))
    return report
