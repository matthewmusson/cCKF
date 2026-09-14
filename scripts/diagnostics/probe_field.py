"""Print the magnetic field the DD4hep detector description provides, at a few points.

Run with the pipeline environment:
    CCKF_ENTRY=scripts/diagnostics/probe_field.py ./scripts/run_cckf_nersc.sh probe
(the trailing word is a placeholder; shifter rejects an empty RUN_ARGS).
Used 2026-09-14 to show the ODD_v5 xml field is 2 T while ColliderML was simulated at 3 T.
"""
import os, sys
from pathlib import Path
import acts
from acts import UnitConstants as u
from acts.examples.odd import getOpenDataDetector
geoDir = Path(os.environ["ODD_PATH"])
deco = acts.IMaterialDecorator.fromFile(geoDir / "data/odd-material-maps.root")
det = getOpenDataDetector(odd_dir=geoDir, materialDecorator=deco)
field = det.field
print("field object:", type(field))
ctx = acts.MagneticFieldContext()
cache = field.makeCache(ctx)
for (x, y, z) in [(0, 0, 0), (100, 0, 0), (500, 0, 0), (0, 0, 500), (0, 0, 1500), (1000, 0, 0), (1200, 0, 0), (0, 0, 2500)]:
    b = field.getField(acts.Vector3(x * u.mm, y * u.mm, z * u.mm), cache)
    print(f"({x:5d},{y:3d},{z:5d}) mm  B = ({b[0]/u.T:+.4f}, {b[1]/u.T:+.4f}, {b[2]/u.T:+.4f}) T")
c = acts.ConstantBField(acts.Vector3(0, 0, 2 * u.T))
cc = c.makeCache(ctx)
print("constant 2T check:", c.getField(acts.Vector3(0, 0, 0), cc)[2] / u.T)
