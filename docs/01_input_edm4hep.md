# 01 — The input: one simulated collision in `edm4hep.root`

## What the file is

ColliderML is a public dataset of simulated proton-proton collisions. For
each collision ("event") a generator produced the particles, and Geant4
tracked every one of them through the Open Data Detector (ODD), recording
every place a charged particle deposited energy in a sensor. The result is
written in the EDM4hep format, a ROOT file with one entry per event.

We work with top-quark-pair events at 200 pileup: on top of the interesting
collision, about 200 other soft proton-proton collisions happen in the same
bunch crossing, so an event holds hundreds of thousands of hits from tens of
thousands of particles. Our working file holds 64 events (a subset of the
128 in ColliderML run 0). Events 0 to 31 are the ones we train, tune, and
plot on; events 32 to 63 are sealed for a final evaluation and have never
been opened by this project.

## What is inside

Opening the file with `uproot` shows one tree, `events`, with 64 entries and
these collections (each is a list per event):

| Collection | One row per | We use it for |
|---|---|---|
| `MCParticles` | simulated particle | the truth: who was there, with what momentum, from where |
| `PixelBarrelReadout`, `PixelEndcapReadout` | energy deposit in a pixel sensor | the truth hits (simhits) in the pixel detector |
| `ShortStripBarrelReadout`, `ShortStripEndcapReadout` | deposit in a short-strip sensor | simhits in the short strips |
| `LongStripBarrelReadout`, `LongStripEndcapReadout` | deposit in a long-strip sensor | simhits in the long strips |
| `_<collection>_particle` | (relation) | which `MCParticles` row made each deposit |
| `ECal*`, `HCal*` | calorimeter deposits | not used; tracking only |
| `EventHeader` | event | event number, run number |

Event 4 has **244,232** particles in `MCParticles`, of which 16,494 are
charged and "final state" (`generatorStatus == 1`: the generator handed them
to Geant4 as real outgoing particles rather than intermediate ones).

## Our particle in `MCParticles`

Row 482 of event 4:

| Field | Value | Meaning |
|---|---|---|
| `PDG` | −211 | a π⁻ (the Particle Data Group code) |
| `generatorStatus` | 1 | final-state particle from the generator |
| `charge` | −1 | |
| `mass` | 0.13957 GeV | |
| `vertex.x, .y, .z` | 0.015, 0.014, 105.73 mm | where it was created: on the beam axis, 106 mm from the centre along z |
| `momentum.x, .y, .z` | 2.304, −0.487, −4.016 GeV | production momentum |
| derived pT, η | 2.355 GeV, −1.303 | transverse momentum; pseudorapidity (negative: it goes toward −z) |
| `endpoint.x, .y, .z` | 633, −57, −982 mm | where Geant4 stopped tracking it |
| `parents_begin/end`, `daughters_begin/end` | 1386–1410, 2715–2725 | index ranges into the relation lists |

A vertex at z = 106 mm is normal: the colliding bunches are long, so the
collision point is spread along the beam by about 50 mm rms.

## Our particle's deposits (simhits)

Six rows in `PixelBarrelReadout` and four in `ShortStripBarrelReadout` point
back to particle 482 through the `_..._particle` relation. In the order the
particle made them:

| Collection | Row | position (x, y, z) mm | r mm | eDep | time | momentum (x, y, z) GeV |
|---|---|---|---|---|---|---|
| PixelBarrel | 54855 | (31.39, −6.42, 51.13) | 32.0 | 62 keV | 0.233 ns | (2.310, −0.458, −4.015) |
| PixelBarrel | 54856 | (33.00, −6.74, 48.33) | 33.7 | 81 keV | 0.244 ns | (2.309, −0.458, −4.016) |
| PixelBarrel | 54857 | (66.88, −13.20, −10.52) | 68.2 | 67 keV | 0.471 ns | (2.314, −0.426, −4.014) |
| PixelBarrel | 54858 | (111.64, −21.05, −88.06) | 113.6 | 55 keV | 0.771 ns | (2.319, −0.386, −4.013) |
| PixelBarrel | 54859 | (113.35, −21.34, −91.03) | 115.3 | 74 keV | 0.783 ns | (2.316, −0.395, −4.014) |
| PixelBarrel | 54860 | (167.87, −30.00, −185.28) | 170.5 | 50 keV | 1.147 ns | (2.325, −0.343, −4.013) |
| ShortStripBarrel | 57228 | (256.51, −41.59, −337.94) | 259.9 | 542 keV | 1.738 ns | (2.332, −0.262, −4.008) |
| ShortStripBarrel | 57229 | (354.93, −50.76, −506.84) | 358.5 | 108 keV | 2.391 ns | (2.340, −0.174, −4.006) |
| ShortStripBarrel | 57230 | (494.33, −57.50, −745.12) | 497.7 | 109 keV | 3.312 ns | (2.347, −0.050, −4.004) |
| ShortStripBarrel | 57231 | (500.79, −57.63, −756.14) | 504.1 | 119 keV | 3.355 ns | (2.346, −0.045, −4.004) |

Three things to notice, because they come back in every later stage:

1. **r grows, z falls.** The particle spirals outward (r from 32 to 504 mm)
   while moving toward −z, as its negative η says. The pixel barrel has four
   layers at r ≈ 33, 68, 114, 170 mm; the short-strip barrel starts at
   r ≈ 260 mm.
2. **Two layers have two deposits each.** Rows 54855/54856 are both at
   r ≈ 33 mm, and 54858/54859 both at r ≈ 114 mm. A barrel layer is tiled
   with flat modules that overlap at their edges so there are no gaps; this
   particle crossed an overlap region twice and left a deposit in both
   modules. Both deposits are real and both become measurements. Later, the
   seeding and the track finder see them as two consecutive surfaces on the
   same layer.
3. **The momentum barely changes.** It loses about 1 MeV of 4.6 GeV across
   the whole tracker. The direction rotates slowly in the magnetic field
   (momentum.y drifts from −0.46 to −0.05 GeV): that curvature is what lets
   the tracker measure momentum.

`eDep` is the energy left in the sensor (tens of keV in 125 µm of silicon).
`time` is in nanoseconds here; the ACTS CSVs in doc 02 store the same times
in ACTS' native unit (millimetres of light travel, 1 ns = 299.79), which is
why the same hit shows `tt = 69.77` there.

## How ACTS reads it

`digi_and_reco.py` opens the file with ACTS' `PodioReader` and hands each
event to `EDM4hepSimInputConverter`, which produces two ACTS collections:

- **particles**, one per `MCParticles` row, each given an ACTS *barcode*
  made of five small integers: primary vertex number, secondary vertex
  number, particle number within the vertex, generation, sub-particle.
  Particle 482 becomes `(1, 0, 7, 0, 6)`. The index 482 is not part of the
  barcode, which is why the two look unrelated. The barcode is what every
  later file uses to name the particle, either as five columns or packed
  into one 64-bit integer (`281474977169414` for ours).
- **simhits**, one per deposit in the six tracker collections, each carrying
  the barcode of its particle and the surface it landed on.

Then a **truth selection** decides which particles we will grade ourselves
against. The reconstruction is scored only on particles that could
reasonably be found: charged, |η| < 3, transverse momentum above a threshold
(1 GeV for every number reported in the log; every plot states the value
it used), produced within 24 mm of the beam axis and within 1 m of the
centre along z, and leaving at least six measurements. Particle 482 passes
all of these, so a track finder that misses it is charged with an
inefficiency; one that finds it with the wrong hits is charged with a fake.

## How to look at the file yourself

```python
import uproot, awkward as ak
t = uproot.open("edm4hep.root")["events"]
ev = 4
mc = t.arrays(["MCParticles/MCParticles.PDG", "MCParticles/MCParticles.momentum.x"],
              entry_start=ev, entry_stop=ev + 1)[0]
hits = t.arrays(["PixelBarrelReadout/PixelBarrelReadout.position.x",
                 "_PixelBarrelReadout_particle/_PixelBarrelReadout_particle.index"],
                entry_start=ev, entry_stop=ev + 1)[0]
# hits whose relation index is 482 belong to our particle
```

The full script that printed the tables above is
`scripts/trail_edm4hep.py <edm4hep.root> <event> <px> <py> <pz>`; it finds
the final-state charged particle whose production momentum is closest to
the momentum you give it and lists its deposits.
