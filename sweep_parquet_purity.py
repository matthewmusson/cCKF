"""Read-only purity sweep over all 32 expanded training Parquets.

Audits each event for the two defects found on 2026-08-21:
  1. ``S11`` missing on sensors that digitize 2D (volumes 16-18, 23-25).
     Those rows lost ``residual_l1`` and had ``chi2_inc`` recomputed as the
     1D form ``r0^2/S00``, so gate features 1, 3, 4 and 5 are wrong.
  2. Genuine 1D long-strip states (volumes 28/29/30) dropped entirely by
     ``expand_trackstates``'s ``S11.notna()`` filter.

Also reports the gate label base rate per event, since the two defects move
it by roughly an order of magnitude.

This job NEVER calls ``data_vol.commit()`` -- it only reads. It is safe to
run beside a training or inference job.

Usage:
    modal run sweep_parquet_purity.py
"""

import modal

app = modal.App("cckf-parquet-purity-sweep")
data_vol = modal.Volume.from_name("surp-acts-data", create_if_missing=False)
DATA_PATH = "/data"
PARQUET_DIR = f"{DATA_PATH}/results/train32/expanded"

image = modal.Image.debian_slim(python_version="3.12").pip_install(
    "pyarrow", "numpy"
)

# Volumes whose geometric digitisation emits a single index (0,) and so
# legitimately have no l1 coordinate. Everything else digitises (0, 1).
# Source: configs/odd-digi-geometric-config.json
TRUE_1D_VOLUMES = {28, 29, 30}

COLUMNS = [
    "cand_hit_id", "contrib_pids", "branch_majority_pid", "majority_undefined",
    "action_taken", "volume_id", "S00", "S11", "residual_l1", "chi2_inc",
    "n_window", "vstar_soft",
]


@app.function(
    image=image, volumes={DATA_PATH: data_vol}, cpu=4, memory=16384, timeout=3600
)
def sweep_event(event_id: int) -> dict:
    import numpy as np
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    path = f"{PARQUET_DIR}/expanded_event{event_id:09d}.parquet"
    try:
        pf = pq.ParquetFile(path)
    except Exception as exc:                      # unreadable / corrupt footer
        return {"event": event_id, "error": str(exc)[:200]}

    acc = dict.fromkeys(
        ["rows", "cand", "trainable", "pos", "clean", "clean_pos",
         "true1d", "true1d_pos", "corrupt", "corrupt_pos",
         "vstar_finite", "s00_missing", "box_violate"], 0
    )
    nwin_sum = 0.0
    corrupt_volumes: dict[int, int] = {}

    for rg in range(pf.metadata.num_row_groups):
        t = pf.read_row_group(rg, columns=COLUMNS)
        n = t.num_rows
        acc["rows"] += n

        cand_id = np.asarray(t["cand_hit_id"].to_numpy(zero_copy_only=False))
        undef = np.array([bool(x) for x in t["majority_undefined"].to_pylist()])
        keep = (~undef) & (cand_id != -1)          # cckf/labels.py gate_row_mask
        acc["cand"] += int((np.asarray(
            t["action_taken"].to_numpy(zero_copy_only=False)) == 0).sum())

        # y = 1[branch_majority_pid in contrib_pids]  (cckf/labels.py)
        pids = t.column("contrib_pids").combine_chunks()
        if isinstance(pids, pa.ChunkedArray):
            pids = (pids.chunk(0) if pids.num_chunks == 1
                    else pa.concat_arrays(pids.chunks))
        maj = np.asarray(t["branch_majority_pid"].to_numpy(zero_copy_only=False))
        flat = np.asarray(pc.list_flatten(pids))
        parent = np.asarray(pc.list_parent_indices(pids))
        if len(parent):
            y = np.bincount(parent[flat == maj[parent]], minlength=n) > 0
        else:
            y = np.zeros(n, dtype=bool)

        s00 = np.asarray(t["S00"].to_numpy(zero_copy_only=False), dtype=float)
        s11 = np.asarray(t["S11"].to_numpy(zero_copy_only=False), dtype=float)
        vol = np.asarray(t["volume_id"].to_numpy(zero_copy_only=False))
        vstar = np.asarray(t["vstar_soft"].to_numpy(zero_copy_only=False), float)

        lost = ~np.isfinite(s11)
        genuine_1d = np.isin(vol, list(TRUE_1D_VOLUMES))
        corrupt = lost & ~genuine_1d               # 2D sensor, S11 gone
        clean = ~lost

        acc["trainable"] += int(keep.sum())
        acc["pos"] += int((keep & y).sum())
        for name, mask in (("clean", clean), ("true1d", genuine_1d),
                           ("corrupt", corrupt)):
            acc[name] += int((keep & mask).sum())
            acc[name + "_pos"] += int((keep & mask & y).sum())

        acc["vstar_finite"] += int(np.isfinite(vstar).sum())
        acc["s00_missing"] += int((keep & ~np.isfinite(s00)).sum())

        # The n=10 box that produced these rows must hold on every candidate.
        r1 = np.asarray(t["residual_l1"].to_numpy(zero_copy_only=False), float)
        with np.errstate(invalid="ignore"):
            two_d = keep & clean
            acc["box_violate"] += int(
                (two_d & (np.abs(r1) > 10.0 * np.sqrt(np.abs(s11)))).sum())

        nwin = np.asarray(t["n_window"].to_numpy(zero_copy_only=False), float)
        nwin_sum += float(np.nansum(nwin[keep]))

        if corrupt.any():
            for v, c in zip(*np.unique(vol[keep & corrupt], return_counts=True)):
                corrupt_volumes[int(v)] = corrupt_volumes.get(int(v), 0) + int(c)

    acc["event"] = event_id
    acc["row_groups"] = pf.metadata.num_row_groups
    acc["nwin_mean"] = nwin_sum / acc["trainable"] if acc["trainable"] else 0.0
    acc["corrupt_volumes"] = corrupt_volumes
    return acc


@app.local_entrypoint()
def main(events: str = ""):
    ids = ([int(x) for x in events.split(",")] if events else list(range(32)))
    results = sorted(sweep_event.map(ids), key=lambda r: r["event"])

    hdr = (f"{'ev':>3} {'rows':>13} {'trainable':>11} {'pos%':>7} "
           f"{'S11lost%':>9} {'1D%':>6} {'clean%':>7} {'cleanPos%':>10} "
           f"{'nwin':>6} {'vstar':>7} {'box!':>5}")
    print(hdr)
    print("-" * len(hdr))
    groups = {"A": [], "B": [], "other": []}
    for r in results:
        if "error" in r:
            print(f"{r['event']:>3}  ERROR: {r['error']}")
            continue
        tr = r["trainable"] or 1
        lost_f, one_f = r["corrupt"] / tr, r["true1d"] / tr
        print(f"{r['event']:>3} {r['rows']:>13,} {r['trainable']:>11,} "
              f"{r['pos']/tr:>7.3%} {lost_f:>9.2%} {one_f:>6.2%} "
              f"{r['clean']/tr:>7.2%} {r['clean_pos']/max(r['clean'],1):>10.3%} "
              f"{r['nwin_mean']:>6.1f} {r['vstar_finite']:>7,} "
              f"{r['box_violate']:>5,}")
        key = "A" if lost_f > 0.05 else ("B" if one_f < 1e-9 else "other")
        groups[key].append(r["event"])

    print(f"\nGroup A (S11 lost on 2D sensors): {groups['A']}")
    print(f"Group B (no long-strip rows at all): {groups['B']}")
    print(f"Unclassified: {groups['other']}")

    ok = [r for r in results if "error" not in r]
    tot = sum(r["trainable"] for r in ok)
    print(f"\nTotals over {len(ok)} events")
    print(f"  trainable gate rows : {tot:,}")
    for k, lab in (("corrupt", "corrupted (2D sensor, S11 lost)"),
                   ("true1d", "genuine 1D long-strip"),
                   ("clean", "clean")):
        s = sum(r[k] for r in ok)
        sp = sum(r[k + "_pos"] for r in ok)
        print(f"  {lab:<32} {s:>13,}  {s/tot:6.2%}  pos={sp/max(s,1):7.3%}")
    print(f"  box violations (should be 0): {sum(r['box_violate'] for r in ok):,}")
    print(f"  finite vstar_soft cells     : {sum(r['vstar_finite'] for r in ok):,}")
