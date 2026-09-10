# 00 — The pipeline, from a simulated collision to a trained decision function

This is the first of a series of short documents that walk through the data
this project works with. They assume nothing: no ACTS, no ROOT, no detector
physics. Each document introduces one stage, shows the real files that stage
produces, and follows **one particle** through them so every table has one
row you can point at.

## The one particle we follow

In event 4 of our working file there is a negative pion (PDG code −211) made
in the proton-proton collision itself, with transverse momentum 2.36 GeV,
flying slightly backward along the beam (η = −1.30). Geant4 records it
crossing ten silicon sensors: six in the pixel barrel and four in the
short-strip barrel. We call it **particle 482** because that is its index in
the simulation file's particle list, and its "barcode" once ACTS has renamed
it is `(1, 0, 7, 0, 6)`. Every document in this series shows what this one
particle looks like at that stage.

## The map

```
 ColliderML simulation (Geant4, done for us)        file: edm4hep.root         doc 01
   one file per "run" of 128 collision events
   particle list + every energy deposit in every sensor
        │
        ▼  digi_and_reco.py  (ACTS, one YAML config in configs/)
 ┌──────────────────────────────────────────────────────────────────────────┐
 │ 1. read the event, rename particles and hits into ACTS' own structures     │
 │ 2. keep the particles we will grade ourselves against (truth selection)    │
 │ 3. DIGITISATION: turn each energy deposit into what the sensor would       │  doc 02
 │    actually read out (channels → cluster → measurement)                    │
 │        files: event*-measurements.csv, cells.csv, simhits.csv,             │
 │               measurement-simhit-map.csv, detectors.csv                    │
 │ 4. SPACE POINTS and SEEDING: guess where tracks start                      │  doc 03
 │        file: event*-seed.csv                                               │
 │ 5. the COMBINATORIAL KALMAN FILTER (CKF): grow each seed into a track,     │  doc 03
 │    one sensor layer at a time, deciding at every layer which hit to take   │
 │    and whether to keep going. This is where the gate and the value         │
 │    function plug in.                                                       │
 │        file: trackstates_ckf.root  (every decision, logged)                │  doc 05
 │        file: event*-predicted-cov.csv                                      │
 │ 6. AMBIGUITY RESOLUTION and SCORING: pick a non-overlapping set of tracks  │  doc 04
 │    and grade them against the truth                                        │
 │        files: performance_finding_*.root, tracksummary_*.root             │
 └──────────────────────────────────────────────────────────────────────────┘
        │
        ▼  expansion.py
 EXPANSION: for every logged decision, list every hit the CKF could have      doc 06
 chosen, with the truth of each   →   expanded_event*.parquet
        │
        ▼  scripts/build_*_cache.py, scripts/train_*.py
 feature caches → the gate g_ψ and the value function V_φ → weight blobs
        │
        ▼  digi_and_reco.py again, with cckf: true
 the CKF re-run with the learned decisions, graded the same way (step 6)
```

The left column is what happens; the right column says which document
explains it. Documents 03 to 06 are being written; 06 will replace the
older `data_schema.md`, which is the same material at reference density.

## Three words that come up everywhere

- **Hit.** Used loosely for two different things, and the difference matters.
  A *simhit* is a truth record: Geant4 says this particle deposited this
  much energy at this exact point. A *measurement* is what the detector
  reports: a position with an uncertainty, made from one or more channels
  firing, with no particle attached. Digitisation (doc 02) turns the first
  into the second, and `measurement-simhit-map.csv` remembers which simhits
  went into which measurement, which is how we know the truth later.
- **Surface.** A single silicon sensor module, flat, with its own local
  coordinate system (`local0`, `local1` in millimetres across its face).
  Every measurement and every track state lives on a surface. Surfaces are
  addressed by (volume, layer, module); doc 02 explains the numbers.
- **Track state.** What the Kalman filter knows about a track when it
  arrives at a surface: predicted position and direction, their
  uncertainties, and, after it has chosen a hit there, the updated values.
  One track has one state per surface it crossed. Doc 03 explains the
  filter; doc 05 shows the stored states.

## Where the files are

Everything shown in these documents is readable on NERSC without any job:

```
/global/cfs/cdirs/atlas/mussonm/cckf_handoff/examples/envelope_event4/
    stage1/          the Stage 1 output directory for events 4 and 5
    expanded_event000000004.parquet
/pscratch/sd/m/mussonm/cckf/modal_backup/events/edm4hep.root    (the input; a 64-event copy of
    /global/cfs/cdirs/m4958/data/ColliderML/simulation/hard_scatter/ttbar/v1/runs/0/edm4hep.root)
```

`NERSC.md` has the full layout. Each document ends with the exact Python that
produced its tables, so you can rerun it on any other particle or event.
