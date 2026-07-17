import time
from pathlib import Path
import acts
import acts.examples
import acts.examples.edm4hep
from acts.examples import Sequencer
from acts.examples.odd import getOpenDataDetector, getOpenDataDetectorDirectory
from acts.examples.simulation import (
    addDigitization,
    addSimParticleSelection,
    addDigiParticleSelection,
    ParticleSelectorConfig,
)
from acts.examples.reconstruction import (
    addSeeding,
    SeedingAlgorithm,
    SeedFinderConfigArg,
    addCKFTracks,
    addVertexFitting,
    addAmbiguityResolution,
    addAmbiguityResolutionML,
    addScoreBasedAmbiguityResolution,
    addTrackWriters,
    VertexFinder,
    TrackSelectorConfig,
    CkfConfig,
    AmbiguityResolutionConfig,
    AmbiguityResolutionMLConfig,
    ScoreBasedAmbiguityResolutionConfig,
)
import traceback
from utils.app_logging import setup_logging, TimingRecorder
from utils.config import create_base_parser, load_config
from contextlib import contextmanager
import math
from acts.examples.edm4hep import EDM4hepSimInputConverter, PodioReader

u = acts.UnitConstants

# LOG_LEVEL = acts.logging.DEBUG
# LOG_LEVEL = acts.logging.INFO
# LOG_LEVEL = acts.logging.ERROR
LOG_LEVEL = acts.logging.FATAL

def parse_args():
    """Parse command line arguments"""
    parser = create_base_parser("Digitization and reconstruction for ACTS")
    parser.add_argument(
        "--input-file",
        help="Input EDM4hep file (default: {output_dir}/edm4hep.root)",
        type=Path,
        default=None
    )
    parser.add_argument(
        "--digi-config",
        help="Digitization configuration file",
        type=Path,
        default=None
    )
    parser.add_argument(
        "--material-config",
        help="Material map configuration file",
        type=Path,
        default=None
    )
    parser.add_argument(
        "--ambi-solver",
        help="Ambiguity solver to use",
        choices=["greedy", "scoring", "ML"],
        default=None
    )
    parser.add_argument(
        "--ambi-config",
        help="Score Based ambiguity resolution config",
        type=Path,
        default=None
    )
    parser.add_argument(
        "--output-root",
        help="Write ROOT output files (default: True)",
        action="store_true",
        default=True
    )
    parser.add_argument(
        "--digi",
        help="Run digitization",
        action="store_true",
        default=None
    )
    parser.add_argument(
        "--reco",
        help="Run reconstruction",
        action="store_true",
        default=None
    )
    
    parser.add_argument(
        "--vertexing",
        help="Run vertexing",
        action="store_true",
        default=None
    )

    parser.add_argument(
        "--threads",
        help="Number of threads to use",
        type=int,
        default=None
    )

    return parser.parse_args()

def setup_acts_reconstruction(input_path, output_dir, config, rnd, logger=None):
    """Configure ACTS reconstruction chain"""
    logger = logger or setup_logging("ACTSReco")

    # Create sequencer — outputDir enables ACTS's built-in timing.tsv
    s = Sequencer(
        numThreads=config.threads if config.threads is not None else 1,
        events=config.events,
        logLevel=LOG_LEVEL,
        trackFpes=False,
        outputDir=str(output_dir),
    )
    
    # Get detector and field
    geoDir = getOpenDataDetectorDirectory()
    
    # Granular control of ROOT output and performance writers
    output_particles_root = getattr(config, "output_particles_root", False)
    output_simhits_root = getattr(config, "output_simhits_root", False)
    output_measurements_root = getattr(config, "output_measurements_root", False)
    output_seeds_root = getattr(config, "output_seeds_root", False)
    output_spacepoints_root = getattr(config, "output_spacepoints_root", False)

    ckf_root_output = getattr(config, "ckf_root_output", False)
    ambi_root_output = getattr(config, "ambi_root_output", False)

    ckf_finding_performance = getattr(config, "ckf_finding_performance", False)
    ckf_fitting_performance = getattr(config, "ckf_fitting_performance", False)
    ambi_finding_performance = getattr(config, "ambi_finding_performance", False)
    ambi_fitting_performance = getattr(config, "ambi_fitting_performance", False)
    
    # Load material map
    material_config = getattr(config, 'material_config', None)
    oddMaterialMap = (
        geoDir / f"data/{material_config}"
        if material_config
        else geoDir / "data/odd-material-maps.root"
    )

    digi_config = getattr(config, 'digi_config', None)
    if digi_config:
        dc_path = Path(digi_config)
        oddDigiConfig = dc_path if dc_path.is_file() else (geoDir / f"config/{digi_config}")
    else:
        oddDigiConfig = geoDir / "config/odd-digi-smearing-config.json"

    oddSeedingSel = geoDir / "config/odd-seeding-config.json"
    oddMaterialDeco = acts.IMaterialDecorator.fromFile(oddMaterialMap)
    
    # Get detector
    detector = getOpenDataDetector(
        odd_dir=geoDir,
        materialDecorator=oddMaterialDeco
    )
    trackingGeometry = detector.trackingGeometry()
    field = detector.field
    
    # Configure EDM4hep reader and converter
    # Step 1: PodioReader to read the EDM4hep file
    podioReader = PodioReader(
        level=LOG_LEVEL,
        inputPath=str(input_path),
        outputFrame="events",
        category="events",
    )
    s.addReader(podioReader)
    
    # Step 2: EDM4hepSimInputConverter algorithm to convert EDM4hep data to ACTS format
    edm4hepConverter = EDM4hepSimInputConverter(
        level=LOG_LEVEL,
        inputFrame="events",
        inputSimHits=[
            "PixelBarrelReadout",
            "PixelEndcapReadout",
            "ShortStripBarrelReadout",
            "ShortStripEndcapReadout",
            "LongStripBarrelReadout",
            "LongStripEndcapReadout"
        ],
        outputParticlesGenerator="particles_input",
        outputParticlesSimulation="particles_simulated",
        outputSimHits="simhits",
        outputSimVertices="simvertices",
        dd4hepDetector=detector,
        trackingGeometry=trackingGeometry,
        sortSimHitsInTime=False,
        # particleRMax=1080 * u.mm,
        # particleZ=(-3030 * u.mm, 3030 * u.mm),
        # particlePtMin=150 * u.MeV,
        particleRMax=None,
        particleZ=(None, None),
        particlePtMin=None,
    )
    s.addAlgorithm(edm4hepConverter)
    s.addWhiteboardAlias("particles", edm4hepConverter.config.outputParticlesSimulation)
    
    # Add sim particle selection (filters particles from simulation)
    if not getattr(config, 'output_all_particles', False):
        particle_selection_config = ParticleSelectorConfig(
            rho=(0.0, 1080 * u.mm),
            absZ=(0.0, 3.03 * u.m),
            pt=(150 * u.MeV, None),
        )
    else:
        particle_selection_config = ParticleSelectorConfig()

    addSimParticleSelection(
        s,
        particle_selection_config,
    )
    
    # Add digitization if enabled
    digi_enabled = getattr(config, 'digi', True)  # Default True
    if digi_enabled:
        logger.info("Adding digitization")
        # ROOT output for digitized measurements (purely controlled by config flag)
        measurements_root_dir = output_dir if output_measurements_root else None

        addDigitization(
            s,
            trackingGeometry,
            field,
            digiConfigFile=oddDigiConfig,
            outputDirRoot=measurements_root_dir,
            outputDirCsv=None,
            rnd=rnd,
            logLevel=LOG_LEVEL,
        )

        def make_geoid(vol=None, lay=None):
            geoid = acts.GeometryIdentifier()
            if vol is not None:
                geoid.volume = vol
            if lay is not None:
                geoid.layer = lay
            return geoid

        measurementCounter = acts.examples.ParticleSelector.MeasurementCounter()
        # At least 3 hits in the pixels (min=3, max=unlimited)
        measurementCounter.addCounter(
            [
                make_geoid(16),
                make_geoid(17),
                make_geoid(18),
            ],
            3,
            2**31 - 1,
        )
        
        # Add digi particle selection (filters particles with sufficient measurements)
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
    
    # Add spacepoint creation if enabled
    spacepoints_enabled = getattr(config, 'spacepoints', False)  # Default False
    if spacepoints_enabled and digi_enabled:
        logger.info("Adding spacepoint creation and writing")
        
        # Get geometry selection file from config
        spacepoint_geo_config = getattr(config, 'spacepoint_geometry_selection', None)
        if not spacepoint_geo_config:
            logger.error("spacepoints enabled but 'spacepoint_geometry_selection' not specified in config")
            raise ValueError("spacepoint_geometry_selection must be specified in config when spacepoints=True")
        
        spGeometrySelection = Path(spacepoint_geo_config)
        if not spGeometrySelection.exists():
            logger.error(f"Spacepoint geometry selection file not found: {spGeometrySelection}")
            raise FileNotFoundError(f"Spacepoint geometry selection file not found: {spGeometrySelection}")
        
        # Add SpacePointMaker algorithm
        s.addAlgorithm(
            acts.examples.SpacePointMaker(
                level=LOG_LEVEL,
                trackingGeometry=trackingGeometry,
                inputMeasurements="measurements",
                outputSpacePoints="spacepoints",
                stripGeometrySelection=acts.examples.readJsonGeometryList(
                    str(spGeometrySelection)
                ),
            )
        )
        
        # Write spacepoints to ROOT if requested
        if output_spacepoints_root:
            s.addWriter(
                acts.examples.root.RootSpacepointWriter(
                    level=LOG_LEVEL,
                    inputSpacepoints="spacepoints",
                    inputMeasurementParticlesMap="measurement_particles_map",
                    filePath=str(output_dir / "spacepoints.root"),
                )
            )
    
    # Add reconstruction components if enabled
    reco_enabled = getattr(config, "reco", False)  # Default False
    if reco_enabled:
        logger.info("Adding reconstruction chain")
        # Add seeding
        # ROOT output for seeding performance (purely controlled by config flag)
        seeds_root_dir = output_dir if output_seeds_root else None

        seed_minPt = getattr(config, 'seed_minPt', 0.5) * u.GeV
        seed_impactMax = getattr(config, 'seed_impactMax', 3.0) * u.mm
        seed_maxSeedsPerSpM = getattr(config, 'num_seeds_per_spm', 5)

        addSeeding(
            s,
            trackingGeometry,
            field,
            seedingAlgorithm=SeedingAlgorithm.Default,
            particleHypothesis=acts.ParticleHypothesis.pion,
            seedFinderConfigArg=SeedFinderConfigArg(
                r=(33 * u.mm, 200 * u.mm),
                deltaR=(1 * u.mm, 300 * u.mm),
                collisionRegion=(-250 * u.mm, 250 * u.mm),
                z=(-2000 * u.mm, 2000 * u.mm),
                maxSeedsPerSpM=seed_maxSeedsPerSpM,
                sigmaScattering=5,
                radLengthPerSeed=0.1,
                minPt=seed_minPt,
                impactMax=seed_impactMax,
                zBinEdges=[-1600, -1000, -600, 0, 600, 1000, 1600],
            ),
            initialSigmas=[
                1 * u.mm,
                1 * u.mm,
                1 * u.degree,
                1 * u.degree,
                0.1 * u.e / u.GeV,
                1 * u.ns,
            ],
            initialSigmaQoverPt=0.1 * u.e / u.GeV,
            initialSigmaPtRel = 0.1,
            initialVarInflation = [1e0, 1e0, 1e0, 1e0, 1e0, 1e0],
            geoSelectionConfigFile=oddSeedingSel,
            outputDirRoot=seeds_root_dir,
        )

        ckf_chi2Measurement = getattr(config, 'ckf_chi2CutOffMeasurement', 15.0)
        ckf_chi2Outlier = getattr(config, 'ckf_chi2CutOffOutlier', 25.0)
        ckf_numMeasCutOff = getattr(config, 'ckf_numMeasurementsCutOff', 1)
        ckf_nMeasMin = getattr(config, 'ckf_nMeasurementsMin', 6)
        ckf_maxHolesAndOutliers = getattr(config, 'ckf_maxHolesAndOutliers', 3)
        ckf_ptMin = getattr(config, 'ckf_ptMin', 0.7) * u.GeV

        addCKFTracks(
            s,
            trackingGeometry,
            field,
            trackSelectorConfig=TrackSelectorConfig(
                pt=(ckf_ptMin, None),
                absEta=(None, 3.5),
                nMeasurementsMin=ckf_nMeasMin,
                maxHolesAndOutliers=ckf_maxHolesAndOutliers,
            ),
            ckfConfig=CkfConfig(
                chi2CutOffMeasurement=ckf_chi2Measurement,
                chi2CutOffOutlier=ckf_chi2Outlier,
                numMeasurementsCutOff=ckf_numMeasCutOff,
                seedDeduplication=True,
                stayOnSeed=True,
            ),
            twoWay=True,
            # Disable internal ROOT/CSV writers; we add them explicitly below
            outputDirRoot=None,
            writeCovMat=False,
            writeTrackStates=False,
            writeTrackSummary=False,
            writePerformance=False,
        )

        # Optional ROOT output & performance writers for CKF stage
        if ckf_root_output or ckf_finding_performance or ckf_fitting_performance:
            addTrackWriters(
                s,
                name="ckf",
                tracks="tracks",  # CKF alias
                outputDirCsv=None,
                outputDirRoot=output_dir,
                writeSummary=ckf_root_output,
                writeStates=False,
                writeFitterPerformance=ckf_fitting_performance,
                writeFinderPerformance=ckf_finding_performance,
                logLevel=LOG_LEVEL,
                writeCovMat=getattr(config, "performance_metrics", False),
            )
        
        # Add ambiguity resolution
        ambi_solver = getattr(config, "ambi_solver", "greedy")  # Default greedy
        ambi_config = getattr(config, "ambi_config", None)
        
        if ambi_solver == "ML":
            addAmbiguityResolutionML(
                s,
                config=AmbiguityResolutionMLConfig(
                    maximumSharedHits=3,
                    maximumIterations=1000000,
                    nMeasurementsMin=7,
                ),
                # Disable internal ROOT writers; handled explicitly below
                outputDirRoot=None,
                writeTrackSummary=False,
                writeTrackStates=False,
                writePerformance=False,
                writeCovMat=False,
                onnxModelFile=str(ambi_config),
            )
            ambi_name = "ambiML"
        elif ambi_solver == "scoring":
            addScoreBasedAmbiguityResolution(
                s,
                config=ScoreBasedAmbiguityResolutionConfig(
                    minScore=0,
                    maxShared=2,
                    maxSharedTracksPerMeasurement=2
                ),
                # Disable internal ROOT writers; handled explicitly below
                outputDirRoot=None,
                writeTrackSummary=False,
                writeTrackStates=False,
                writePerformance=False,
                writeCovMat=False,
                ambiVolumeFile=ambi_config,
            )
            ambi_name = "ambi_scorebased"
        else:
            addAmbiguityResolution(
                s,
                config=AmbiguityResolutionConfig(
                    maximumSharedHits=3,
                    maximumIterations=1000000,
                    nMeasurementsMin=6,
                ),
                # Disable internal ROOT writers; handled explicitly below
                outputDirRoot=None,
                writeTrackSummary=False,
                writeTrackStates=False,
                writePerformance=False,
                writeCovMat=False,
            )
            ambi_name = "ambi"

        # Optional ROOT output & performance writers for ambiguity-resolved tracks
        if ambi_root_output or ambi_finding_performance or ambi_fitting_performance:
            addTrackWriters(
                s,
                name=ambi_name,
                tracks="tracks",  # ambiguity-resolved alias
                outputDirCsv=None,
                outputDirRoot=output_dir,
                writeSummary=ambi_root_output,
                writeStates=False,
                writeFitterPerformance=ambi_fitting_performance,
                writeFinderPerformance=ambi_finding_performance,
                logLevel=LOG_LEVEL,
                writeCovMat=getattr(config, "performance_metrics", False),
            )
        
        # Add vertex fitting
        vertexing_enabled = getattr(config, 'vertexing', False)  # Default False
        if vertexing_enabled:
            addVertexFitting(
                s,
                field,
                vertexFinder=VertexFinder.AMVF,
                outputDirRoot=output_dir if getattr(config, 'output_root', True) else None,
            )
    
    # Add ROOT writers for particles/simhits if requested
    if output_particles_root or output_simhits_root:
        add_root_writers(s, output_dir, field, config)
    
    return s

def add_root_writers(s, output_dir, field, config=None):
    """Add ROOT output writers to the sequencer"""
    # Control via config flags only
    output_particles_root = getattr(config, "output_particles_root", False) if config else False
    output_simhits_root = getattr(config, "output_simhits_root", False) if config else False
    write_helix_parameters = getattr(config, "write_helix_parameters", True) if config else True

    # Write tracking hits (simhits) if requested
    if output_simhits_root:
        s.addWriter(
            acts.examples.root.RootSimHitWriter(
                config=acts.examples.root.RootSimHitWriter.Config(
                    filePath=str(output_dir / "simhits.root"),
                    inputSimHits="simhits",
                ),
                level=LOG_LEVEL,
            )
        )

    # Write simulated particles if requested
    if output_particles_root:
        s.addWriter(
            acts.examples.root.RootParticleWriter(
                config=acts.examples.root.RootParticleWriter.Config(
                    filePath=str(output_dir / "particles.root"),
                    inputParticles="particles",
                    referencePoint=acts.Vector3(0.0, 0.0, 0.0),
                    bField=field,
                    writeHelixParameters=write_helix_parameters,
                ),
                level=LOG_LEVEL,
            )
        )

def main():
    logger = setup_logging()
    try:
        # Parse arguments and load config
        args = parse_args()
        config = load_config(args)
        rnd = acts.examples.RandomNumbers(seed=config.seed)

        # Create output directory structure
        output_dir = Path(args.output)
        if args.output_subdir:
            output_dir = output_dir / args.output_subdir
        output_dir.mkdir(parents=True, exist_ok=True)

        # Set default input path if not specified
        input_path = args.input_file or output_dir / "edm4hep.root"

        # Initialize timing recorder
        timer = TimingRecorder(output_dir)

        # Setup (geometry, config, algorithm registration)
        with timer.record("Setup"):
            s = setup_acts_reconstruction(input_path, output_dir, config, rnd, logger)

        # Sequencer execution (ACTS also writes per-algorithm timing.tsv)
        with timer.record("Sequencer"):
            s.run()

        # Parse ACTS timing.tsv and merge into our report
        acts_timing = output_dir / "timing.tsv"
        if acts_timing.exists():
            timer.parse_acts_timing(acts_timing)

        # Write timing report
        timer.write_report()

        logger.info("ACTS reconstruction completed successfully")

    except Exception as e:
        logger.error(f"Fatal error in main: {str(e)}")
        logger.error(traceback.format_exc())
        raise

if __name__ == "__main__":
    main()