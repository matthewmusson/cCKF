"""Follow one particle through the Stage-1 CSVs of one event.
Usage: python3 trail_csv.py <run_dir> <event> pv sv part gen sub
"""
import sys, numpy as np, pandas as pd
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
D, E = sys.argv[1], int(sys.argv[2]); pv, sv, part, gen, sub = map(int, sys.argv[3:8])
f = lambda stem: f"{D}/event{E:09d}-{stem}.csv"
sh = pd.read_csv(f("simhits"), comment="#")
sh["hit_id"] = np.arange(len(sh))
mine = sh[(sh.particle_id_pv == pv) & (sh.particle_id_sv == sv) & (sh.particle_id_part == part)
          & (sh.particle_id_gen == gen) & (sh.particle_id_subpart == sub)].copy()
g = mine.geometry_id.to_numpy().astype(np.uint64)
mine["vol"] = (g >> np.uint64(56)) & np.uint64(0xFF); mine["lay"] = (g >> np.uint64(36)) & np.uint64(0xFFF)
mine["mod"] = (g >> np.uint64(8)) & np.uint64(0xFFFFF); mine["extra"] = g & np.uint64(0xFF)
mine["r"] = np.hypot(mine.tx, mine.ty); mine["pt"] = np.hypot(mine.tpx, mine.tpy)
print(f"simhits.csv: {len(sh)} rows in event {E}; {len(mine)} belong to particle ({pv},{sv},{part},{gen},{sub})")
print(mine.sort_values("tt")[["hit_id", "geometry_id", "vol", "lay", "mod", "extra", "tx", "ty", "tz", "r", "tt", "pt", "te", "deltae", "index"]].to_string(index=False))
packed = (pv << 48) | (sv << 32) | (part << 16) | (gen << 8) | sub
print(f"packed particle_id (expansion.encode_particle_id): {packed}")

mp = pd.read_csv(f("measurement-simhit-map"), comment="#")
mp.columns = [c.strip() for c in mp.columns]
mm = mp[mp.hit_id.isin(mine.hit_id)]
print(f"\nmeasurement-simhit-map.csv: {len(mp)} rows; {len(mm)} rows point at our hits")
print(mm.to_string(index=False))
shared = mp[mp.measurement_id.isin(mm.measurement_id)]
extra = shared[~shared.hit_id.isin(mine.hit_id)]
print(f"other particles' hits merged into the same measurements: {len(extra)}")
if len(extra):
    print(sh.loc[extra.hit_id, ["hit_id", "particle_id_pv", "particle_id_sv", "particle_id_part", "particle_id_gen", "particle_id_subpart"]].to_string(index=False))

me = pd.read_csv(f("measurements"), comment="#"); me.columns = [c.strip() for c in me.columns]
mine_m = me[me.measurement_id.isin(mm.measurement_id)].copy()
gm = mine_m.geometry_id.to_numpy().astype(np.uint64)
mine_m["vol"] = (gm >> np.uint64(56)) & np.uint64(0xFF); mine_m["lay"] = (gm >> np.uint64(36)) & np.uint64(0xFFF)
mine_m["mod"] = (gm >> np.uint64(8)) & np.uint64(0xFFFFF); mine_m["extra"] = gm & np.uint64(0xFF)
print(f"\nmeasurements.csv: {len(me)} rows; {len(mine_m)} are ours   columns={list(me.columns)}")
print(mine_m.to_string(index=False))

ce = pd.read_csv(f("cells"), comment="#"); ce.columns = [c.strip() for c in ce.columns]
first = int(mine_m.sort_values("geometry_id").measurement_id.iloc[0])
sub_c = ce[ce.measurement_id.isin(mine_m.measurement_id)]
print(f"\ncells.csv: {len(ce)} rows; {len(sub_c)} behind our {len(mine_m)} measurements; cells per measurement:")
print(sub_c.groupby("measurement_id").size().to_string())
print(f"cells of measurement {first}:")
print(ce[ce.measurement_id == first].to_string(index=False))

det = pd.read_csv(f"{D}/detectors.csv"); det.columns = [c.strip() for c in det.columns]
gcol = [c for c in det.columns if "geometry" in c][0]
gd = det[gcol].to_numpy().astype(np.uint64)
det["vol"] = (gd >> np.uint64(56)) & np.uint64(0xFF); det["lay"] = (gd >> np.uint64(36)) & np.uint64(0xFFF); det["mod"] = (gd >> np.uint64(8)) & np.uint64(0xFFFFF)
row0 = mine_m.sort_values("geometry_id").iloc[0]
drow = det[(det.vol == row0.vol) & (det.lay == row0.lay) & (det["mod"] == row0["mod"])]
print(f"\ndetectors.csv: {len(det)} surfaces; the module of measurement {first}:")
print(drow.to_string(index=False))

sd = pd.read_csv(f("seed"), comment="#") if __import__("os").path.exists(f("seed")) else None
if sd is not None:
    sd.columns = [c.strip() for c in sd.columns]
    print(f"\nseed.csv columns: {list(sd.columns)}  rows: {len(sd)}")
    mcols = [c for c in sd.columns if "measurement" in c.lower()]
    if mcols:
        hit = sd[sd[mcols].isin(mine_m.measurement_id.tolist()).any(axis=1)]
        print(f"seeds containing at least one of our measurements: {len(hit)}")
        print(hit.head(6).to_string(index=False))
