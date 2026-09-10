# 02 — Digitisation: from a truth deposit to what the detector reports

## The idea

Geant4 tells us exactly where particle 482 crossed each sensor. A real
detector never knows that. A silicon sensor is divided into channels
(pixels, or strips), a crossing particle deposits charge in one or a few of
them, and the readout reports which channels fired and how much. From that
the reconstruction estimates a position, with an uncertainty set by the
channel size. Digitisation is the simulation of that step: it turns each
truth deposit into fired channels, groups neighbouring channels into a
*cluster*, and reports the cluster's centre as a *measurement*.

Nothing downstream ever sees the truth deposit again. The track finder works
on measurements alone; the truth survives only in a side file that says
which deposits went into which measurement.

## The sensors

The ODD tracker has three technologies, each in a barrel (cylinders around
the beam) and two endcaps (discs at each end). ACTS numbers them by
**volume**:

| Volume | Sub-detector | Channel size (u × v) | Measures |
|---|---|---|---|
| 16 / **17** / 18 | pixel endcap − / **barrel** / endcap + | 0.05 × 0.05 mm | 2D position |
| 23 / **24** / 25 | short strip endcap − / **barrel** / endcap + | 0.08 × 0.5 mm | 2D position, coarse in v |
| 28 / **29** / 30 | long strip endcap − / **barrel** / endcap + | 0.125 mm × (whole strip) | 1D: only u |

Volume 20 is the beam pipe and support material: particles cross it but no
sensor lives there. Inside a volume, **layer** numbers the concentric
cylinders (barrel) or discs (endcap), even numbers being the sensitive ones,
and **module** numbers the individual sensor within the layer. Every
measurement, and every track state, is addressed by this triple.

The settings live in `configs/odd-digi-geometric-config.json`, one entry per
volume: the channel grid, the sensor thickness (0.125 mm pixels, 0.2 mm
short strips, 0.25 mm long strips), a charge threshold below which a channel
does not fire, and the resolutions used for the reported uncertainties.

## What digitisation writes

Stage 1 writes five CSV files per event into the run directory (plus one
geometry file for the whole run). Our example directory is
`cckf_handoff/examples/envelope_event4/stage1/`, files
`event000000004-*.csv`.

### `simhits.csv` — the truth deposits, in ACTS units

Same ten deposits as doc 01, now with the ACTS barcode and surface id. The
row number in this file (0-based) is the `hit_id` the other files refer to;
the `index` column is always −1 and means nothing.

| row (hit_id) | barcode (pv, sv, part, gen, sub) | volume | layer | module | tx, ty, tz (mm) | tt | ΔE (GeV) |
|---|---|---|---|---|---|---|---|
| 34631 | (1, 0, 7, 0, 6) | 17 | 2 | 110 | 31.39, −6.42, 51.13 | 69.8 | −6.2e−5 |
| 27241 | (1, 0, 7, 0, 6) | 17 | 2 | 12 | 33.00, −6.74, 48.33 | 73.0 | −8.1e−5 |
| 54910 | (1, 0, 7, 0, 6) | 17 | 4 | 361 | 66.88, −13.20, −10.52 | 141.3 | −6.7e−5 |
| 68976 | (1, 0, 7, 0, 6) | 17 | 6 | 654 | 111.64, −21.05, −88.06 | 231.2 | −5.5e−5 |
| 69256 | (1, 0, 7, 0, 6) | 17 | 6 | 668 | 113.35, −21.34, −91.03 | 234.6 | −7.4e−5 |
| 80852 | (1, 0, 7, 0, 6) | 17 | 8 | 1045 | 167.87, −30.00, −185.28 | 343.9 | −5.0e−5 |
| 150227 | (1, 0, 7, 0, 6) | 24 | 2 | 712 | 256.51, −41.59, −337.94 | 521.0 | −5.4e−4 |
| 167823 | (1, 0, 7, 0, 6) | 24 | 4 | 1088 | 354.93, −50.76, −506.84 | 716.8 | −1.1e−4 |
| 181588 | (1, 0, 7, 0, 6) | 24 | 6 | 1590 | 494.33, −57.50, −745.12 | 993.0 | −1.1e−4 |
| 181444 | (1, 0, 7, 0, 6) | 24 | 6 | 1569 | 500.79, −57.63, −756.14 | 1005.8 | −1.2e−4 |

The two overlap pairs from doc 01 are now visible as two *modules* on one
layer: modules 110 and 12 on pixel layer 2, modules 654 and 668 on pixel
layer 6, and modules 1590 and 1569 on short-strip layer 6.

`tt` is the time in ACTS units (mm; divide by 299.79 for ns). `ΔE` is
negative because it is the change in the particle's energy.

### `cells.csv` — the channels that fired

Each deposit spread charge over a few channels. Measurement 26778 (the
deposit at hit 27241, pixel layer 2, module 12) fired six pixels:

| channel0 (u) | channel1 (v) | value |
|---|---|---|
| 36 | 959 | 0.035 |
| 36 | 960 | 0.062 |
| 36 | 961 | 0.062 |
| 36 | 962 | 0.058 |
| 36 | 963 | 0.046 |
| 37 | 963 | 0.014 |

One column of five pixels along v plus one neighbour: the particle crossed
this module at a shallow angle in v (it is heading toward −z, so it travels
a long way along the strip direction while passing through 125 µm of
silicon). The pixel measurements have 5 or 6 cells each; the short-strip
ones have 2.

### `measurements.csv` — what the reconstruction uses

One row per cluster. Our ten:

| measurement_id | volume | layer | module | local0 (mm) | local1 (mm) | σ(local0) | σ(local1) |
|---|---|---|---|---|---|---|---|
| 26778 | 17 | 2 | 12 | −6.572 | 12.083 | 0.0063 | 0.0063 |
| 33791 | 17 | 2 | 110 | 6.325 | 14.890 | 0.0070 | 0.0070 |
| 53230 | 17 | 4 | 361 | 0.325 | 25.727 | 0.0066 | 0.0066 |
| 66940 | 17 | 6 | 654 | 6.525 | 20.687 | 0.0069 | 0.0069 |
| 67210 | 17 | 6 | 668 | −7.375 | 17.720 | 0.0066 | 0.0066 |
| 78536 | 17 | 8 | 1045 | −2.475 | −4.032 | 0.0066 | 0.0066 |
| 145510 | 24 | 2 | 712 | −0.975 | −12.408 | 0.017 | 0.109 |
| 162283 | 24 | 4 | 1088 | −10.840 | 35.722 | 0.022 | 0.136 |
| 175179 | 24 | 6 | 1569 | 23.720 | 3.279 | 0.022 | 0.136 |
| 175321 | 24 | 6 | 1590 | −17.720 | 14.322 | 0.020 | 0.125 |

`local0`, `local1` are positions on the module's face, in the module's own
frame; they are what the file stores as `local0/local1`, with the variances
`var_local0/var_local1` (the σ above are their square roots). Check the
first row against the cells: pixel module 12 spans −8.4 to +8.4 mm in u in
336 channels of 0.05 mm, so channel 36 sits at −8.4 + 36.5 × 0.05 = −6.575
mm, and the reported `local0` is −6.572. Along v, channels 959 to 963 centre
on −36 + 961.5 × 0.05 = 12.08 mm. The reported position is the
charge-weighted centre of the cluster. The strip measurements are ten times
less precise in u and twenty times less precise in v, as their channel size
says.

The file also carries `global_x/y/z` (the same position in the detector
frame, for convenience) and unused columns (`phi`, `theta`, `time` and their
variances) that are zero here.

### `measurement-simhit-map.csv` — the truth link

| measurement_id | hit_id |
|---|---|
| 26778 | 27241 |
| 33791 | 34631 |
| 53230 | 54910 |
| … | … |

Ten rows for our ten measurements, one deposit each. This is the file that
lets us grade decisions later: a measurement "belongs" to particle 482 if
any of its deposits carries barcode (1, 0, 7, 0, 6). In dense regions one
cluster can contain deposits from two particles (a merged cluster), in which
case the map has two rows for one `measurement_id` and the measurement
belongs to both. None of ours are merged; event 4 has 269,795 map rows for
258,989 measurements, so about 4% are.

### `detectors.csv` — the sensors themselves

One row per sensitive surface for the whole detector (18,918 of them), not
per event. For module 12 of pixel layer 2: centre at (32.2, −0.2, 36.25) mm,
a 3×3 rotation matrix from the module frame to the detector frame, the
bounds (a 16.8 × 72 mm rectangle), thickness 0.125 mm. This is how a
`local0/local1` on a module becomes an x, y, z in space.

## The surface id, and the one trap in it

Every CSV names a surface with a single packed 64-bit `geometry_id`:

```
 bits 63-56   volume     (e.g. 17)
 bits 55-48   boundary   (0 for sensors)
 bits 47-36   layer      (e.g. 2)
 bits 27-8    sensitive  = module (e.g. 12)
 bits  7-0    extra
```

So `1224979236083731456` decodes to volume 17, layer 2, module 12. For barrel
sensors the `extra` byte is 0. For **endcap** sensors the CSVs put the disc's
ring number (1 to 3) into `extra`, while the track-state file (doc 05) knows
only volume, layer and module. A join that compares the raw 64-bit ids
therefore matches every barrel surface and no endcap surface. This bit us
once (August 25: a whole re-expansion with no endcap hits). The rule now:
every loader clears the `extra` byte on read (`expansion.normalize_geometry_id`),
and all joins compare (volume, layer, module), which is unique across the
detector.

## How to look at the files yourself

Everything above came from `scripts/trail_csv.py <run_dir> <event> pv sv part gen sub`,
for example

```
python scripts/trail_csv.py cckf_handoff/examples/envelope_event4/stage1 4  1 0 7 0 6
```

which prints the particle's rows in every CSV, the cells behind its first
measurement, and its module's row in `detectors.csv`. The heart of it is
three lines:

```python
sh = pd.read_csv("event000000004-simhits.csv", comment="#")      # row number = hit_id
mp = pd.read_csv("event000000004-measurement-simhit-map.csv")    # hit_id -> measurement_id
me = pd.read_csv("event000000004-measurements.csv")              # measurement_id -> position
```

Next: doc 03, where the seeding proposes track starts from these
measurements and the Kalman filter grows them, and where the two
overlap-module pairs turn into a real decision.
