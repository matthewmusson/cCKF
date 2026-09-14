# 03 — Seeding and the Combinatorial Kalman Filter

Doc 02 left us with 258,989 measurements in event 4, ten of which belong to
particle 482, and no particle labels on any of them. Track finding has to
recover the particle from the measurements alone. It does this in two steps:
**seeding** proposes starting points, and the **Combinatorial Kalman Filter
(CKF)** grows each starting point outward one surface at a time, choosing at
every surface which measurement, if any, to add.

## Seeding

### Space points

A measurement is a position on a module's face. Using `detectors.csv` (the
module's centre and rotation) every measurement becomes a point in the
detector frame, a *space point*. ODD pixel and short-strip modules measure
both local coordinates, so this is a direct transform; nothing is lost.

### Triplets

A seed is three space points on three different radii that are consistent
with a helix from near the beam line. The seeder bins space points in r and
z, then for every middle point tries bottom and top partners and keeps the
triplets whose curvature and impact parameter pass its cuts. Our
configuration (`digi_and_reco.py`, `addSeeding`) searches
33 mm < r < 200 mm, which is the pixel barrel and pixel endcaps only; the
strips are never used for seeding. The knobs that were tuned in Phase 1 are:

| Config key | Meaning | envelope / tight value |
|---|---|---|
| `num_seeds_per_spm` | how many seeds one middle point may start | 46 / 16 |
| `seed_minPt` | lowest curvature accepted, GeV | 0.587 / 0.688 |
| `seed_impactMax` | largest transverse distance from the beam line, mm | 2.86 / 2.50 |
| `seed_sigmaScattering` | how much multiple scattering to tolerate, in σ | 2.95 / 2.34 |

The "envelope" column is `configs/cckf_envelope.yaml`, the loose setting used
to collect training data; "tight" is `configs/_regen_tight.yaml`, the best
classical operating point. Looser seeding finds more particles and makes many
more fakes.

### What a seed carries

From the three points the seeder estimates a full set of track parameters at
the innermost point (position on that module, direction, charge over
momentum) and gives them a deliberately wide uncertainty, the
`initialSigmas` in `digi_and_reco.py`: 1 mm in each local coordinate, 1° in
each angle, 0.1/GeV in q/p. That covariance is where the Kalman filter
starts, and it shows up as a real number in the next section.

### Our particle in `seed.csv`

`event000000004-seed.csv` has one row per seed. The columns: `seed_id`, the
truth particle that owns the triplet (`particleId`, as `vp=..|vs=..|p=..`),
the seed's own estimate of `pT`, `eta`, `phi`, the three space points
(`bX bY bZ`, `mX mY mZ`, `tX tY tZ`), a `good/duplicate/fake` verdict,
`vertexZ`, a `quality`, and `Hits_ID`, the three measurement ids.

71 seeds use at least one of particle 482's ten measurements. Most are fakes
that combine one of its hits with two from other particles. The ones built
entirely from its hits:

| seed_id | Hits_ID (measurement ids) | layers | seed pT (GeV) | verdict |
|---|---|---|---|---|
| 80486 | 53230, 67210, 78536 | pixel 4, 6, 8 | 1.82 | good |
| 80484 | 26778, 67210, 78536 | pixel 2, 6, 8 | 1.79 | duplicate |
| 80485 | 33791, 67210, 78536 | pixel 2, 6, 8 | 1.80 | duplicate |
| 80488 | 66940, 67210, 78536 | pixel 6, 6, 8 | 1.58 | duplicate |
| 77552 | 26778, 66940, 78536 | pixel 2, 6, 8 | 1.80 | duplicate |
| 77553 | 33791, 66940, 78536 | pixel 2, 6, 8 | 1.81 | duplicate |
| 77554 | 26778, 66940, 67210 | pixel 2, 6, 6 | 1.98 | duplicate |
| 77555 | 33791, 66940, 67210 | pixel 2, 6, 6 | 2.00 | duplicate |

Three things to read off this table. The seed pT estimates (1.6 to 2.0 GeV)
are rough compared with the true 2.36 GeV; three points give a poor curvature
measurement, which is why the initial covariance is wide. The two overlap
pairs from doc 02 (measurements 26778/33791 on layer 2, 66940/67210 on
layer 6) multiply the seeds: one particle, eight all-true triplets. And the
verdict column is ACTS' own truth bookkeeping, "good" for the first seed on a
particle and "duplicate" for the rest; it is there for diagnostics and the
CKF never sees it. A particle seeded eight times produces eight CKF tracks
that all look alike, which is the ambiguity resolution's problem (doc 04).

## The Combinatorial Kalman Filter

### The loop

The CKF takes one seed and repeats, surface after surface, moving outward:

1. **Predict.** Propagate the current state (position, direction, q/p) and
   its covariance through the magnetic field and the material to the next
   surface. The result is a predicted local position `(eLOC0_prt, eLOC1_prt)`
   with uncertainty `err_eLOC0_prt`, `err_eLOC1_prt`.
2. **Look.** Every measurement on that surface is a candidate. For each,
   compute the residual r = measured − predicted and the innovation
   covariance S = (prediction covariance projected onto the surface) +
   (measurement variance), and χ² = rᵀ S⁻¹ r. A candidate is compatible if
   χ² is below `ckf_chi2CutOffMeasurement` (16.26 in both configs above).
3. **Branch.** Each compatible candidate spawns its own continuation,
   updated with that measurement, up to `ckf_numMeasurementsCutOff` of them
   (5 envelope, 3 tight). If none is compatible the state is recorded as a
   hole and the branch continues without an update.
4. **Prune.** A branch is dropped when it exceeds the hole/outlier cap
   (`ckf_maxHolesAndOutliers`, 1 in the tight config, disabled in the
   envelope config).
5. **Stop** when the propagation leaves the tracker. Every surviving branch
   with at least `ckf_nMeasurementsMin` measurements (9 tight, disabled
   envelope) becomes a track candidate.

Steps 2 and 4 are the two decisions this project replaces: the χ² test with
the gate g_ψ, and the hole/branch caps with the value function V_φ. Step 3
is where a single seed turns into a tree of branches, and where the training
data comes from: at every surface the CKF saw a set of candidates and chose,
and doc 06 records every candidate it could have chosen.

The envelope config disables the terminal cuts on purpose. Training data
should contain the branches a tighter configuration would have killed, so
that the value function can learn what a doomed branch looks like.

### The seed state, concretely

Track 128909 is the cleanest CKF track on our particle. Its first state sits
on pixel layer 2, module 12, at measurement 26778. The stored uncertainty of
the predicted position there is `err_eLOC0_prt = 1.000` mm exactly, the seed
covariance from `initialSigmas`, and the accepted hit has χ² = 2 × 10⁻¹³: the
seed parameters were computed from this very point, so the residual is zero.
A 1 mm uncertainty with a 10σ search box means a 10 mm window, and 27
measurements fell inside it. This is the widest window on the whole track;
by pixel layer 4 the uncertainty has shrunk to 0.61 mm and by layer 6 to
0.007 mm.

### The track, surface by surface

All 23 states of track 128909 in the order the CKF created them (doc 05 shows
how they are stored). `pred` and `σ` are the predicted local0 and its
uncertainty, `hit` the accepted measurement's local0, χ² its increment.
"ours" marks measurements that belong to particle 482.

| step | volume/layer/module | path (mm) | pred (mm) | σ (mm) | hit (mm) | χ² | ours |
|---|---|---|---|---|---|---|---|
| 0 | 17/2/110 | −3.3 | 6.338 | 0.006 | 6.325 | 2.08 | yes |
| 1 | 17/2/12 | 0.0 | −6.572 | 1.000 | −6.572 | 2e−13 | yes |
| 2 | 17/2/0 | 11.2 | | | | | (no sensor) |
| 3 | 17/4/361 | 68.2 | 0.325 | 0.612 | 0.325 | 2e−5 | yes |
| 4 | 17/4/0 | 81.3 | | | | | (no sensor) |
| 5 | 17/6/654 | 158.1 | 6.525 | 0.254 | 6.525 | 0.02 | yes |
| 6 | 17/6/668 | 161.5 | −7.371 | 0.007 | −7.375 | 0.17 | yes |
| 7 | 17/6/0 | 171.9 | | | | | (no sensor) |
| 8 | 17/8/1045 | 270.7 | −2.407 | 0.094 | −2.475 | 0.66 | yes |
| 9 | 17/8/0 | 282.5 | | | | | (no sensor) |
| 10 | 20/2/0 | 335.0 | | | | | beam-pipe / support volume |
| 11 | 24/2/0 | 422.9 | | | | | (no sensor) |
| 12 | 24/2/712 | 447.5 | −1.205 | 0.156 | −0.975 | 2.16 | yes |
| 13 | 24/4/0 | 620.4 | | | | | (no sensor) |
| 14 | 24/4/1088 | 643.0 | −11.220 | 0.151 | −10.840 | 6.29 | yes |
| 15 | 24/6/0 | 897.8 | | | | | (no sensor) |
| 16 | 24/6/1590 | 919.2 | −17.752 | 0.228 | −17.720 | 0.02 | yes |
| 17 | 24/6/1569 | 932.0 | 23.727 | 0.023 | 23.720 | 0.05 | yes |
| 18 | 24/8/0 | 1216.0 | | | | | (no sensor) |
| 19 | 24/8/86 | 1238.1 | −15.870 | 0.240 | none | | **hole** |
| 20 | 30/0/0 | 1348.8 | | | | | volume boundary |
| 21 | 29/0/0 | 1455.1 | | | | | volume boundary |
| 22 | 28/12/1 | 1532.6 | −46.571 | 0.659 | −49.188 | 15.75 | **no** |

Reading it:

- **Module 0 states are not sensors.** ACTS logs a state whenever the
  propagation crosses a surface it navigates by, including layer
  approach surfaces (`module 0`) and the passive beam-pipe volume 20. They
  carry no measurement and are not holes in the physical sense. The file does
  not label them, which is a known nuisance for the hole counters (doc 06).
- **Step 0 has a negative path length.** The seed's parameters live on
  module 12 (path 0). The overlapping module 110 sits 3.3 mm behind that
  point along the trajectory, and the navigator visits it first. Both hits
  are real (doc 01).
- **The two overlap pairs are two consecutive states each** (steps 5/6 and
  16/17), on the same layer with different modules. The CKF handles them like
  any other surfaces.
- **Step 19 is a real hole.** Short-strip layer 8 module 86 was crossed and
  nothing compatible was there. The particle's Geant4 endpoint (doc 01) is at
  r = 636 mm, inside layer 6 and layer 8: the pion decayed or interacted
  before reaching layer 8. The hole is correct.
- **Step 22 is a wrong hit.** With the particle gone, the branch kept
  propagating (the envelope config has no hole cap) and found a hit on the
  long-strip disc 28/12 with χ² = 15.75, just under the 16.26 cut. That one
  measurement is why the track's purity is 10/11 rather than 1. A second
  candidate on the same disc (χ² = 11.86) spawned the sibling branch, track
  128908.

### One decision in full

At step 3, pixel layer 4 module 361, the 10σ window held 18 candidates.
Doc 06 shows the full table; the shape of it is what matters here. The true
measurement 53230 has χ² = 2 × 10⁻⁵. The next best, 53229, has χ² = 10.2,
and the rest run from 17 to 137. Under the 16.26 cut, two candidates pass,
so the CKF branched here as well. The branch that took 53229 is not among
the four candidate tracks that end up holding three or more of the
particle's hits, so it did not stay on the particle for long.

This is the situation the gate is trained on. A χ² of 10.2 is "compatible"
to the cut, but the residual is 2 mm along local0 and 0.1 mm along local1 on
a pixel sensor with 0.05 mm channels, the cluster is a different shape, and
the branch is on a track whose previous two hits point elsewhere. The gate
sees all of that; the cut sees one number.

## Which seed became track 128909

Track numbers in the CKF output are not seed ids from `seed.csv`; ACTS
numbers tracks as it produces them, and 128908/128909 are two branches of
one seed. That seed's innermost point is measurement 26778 (module 12, path
0 above), so it is one of 80484, 77552, 77554 or the fakes that start there.
`tracksummary_ckf.root` has no seed id column either; if you need the link,
match the track's first three measurement ids against `Hits_ID`.

## How to look at it yourself

`scripts/trail_csv.py` (doc 02) prints the seed rows above when
`event*-seed.csv` is present in the run directory (`output_seeds_csv: true`
in the config). The track table came from the track-states file, which is
doc 05's subject; the script there reproduces it.
