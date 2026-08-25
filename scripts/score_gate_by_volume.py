"""Score archived Parquet rows with the deployed gate blob, stratified by volume.

Reimplements WeightBlob.hpp + MlpInference.hpp in numpy so the diagnosis uses
the EXACT weights the C++ runs, not a PyTorch checkpoint that may differ.
Runs on val/cal events only.
"""
import struct, sys, glob
import numpy as np, pyarrow.parquet as pq, pyarrow as pa, pyarrow.compute as pc
sys.path.insert(0, "/global/cfs/cdirs/atlas/mussonm/cCKF")
import pandas as pd
from cckf.features import build_gate_features, GATE_SOURCE_COLUMNS

def load_blob(path):
    d = open(path, "rb").read(); o = 0
    assert d[0:4] == b"CCKF", "bad magic"; o = 4
    ver, nf, nh, nl = struct.unpack_from("<4I", d, o); o += 16
    assert ver == 1
    mean = np.frombuffer(d, "<f4", nf, o); o += 4*nf
    std  = np.frombuffer(d, "<f4", nf, o); o += 4*nf
    a0,a1,b0,b1 = struct.unpack_from("<4f", d, o); o += 16
    W, B, ind = [], [], nf
    for _ in range(nl):
        W.append(np.frombuffer(d,"<f4",nh*ind,o).reshape(nh,ind)); o += 4*nh*ind
        B.append(np.frombuffer(d,"<f4",nh,o)); o += 4*nh; ind = nh
    W.append(np.frombuffer(d,"<f4",nh,o).reshape(1,nh)); o += 4*nh
    B.append(np.frombuffer(d,"<f4",1,o)); o += 4
    return dict(nf=nf,nh=nh,nl=nl,mean=mean,std=std,platt=(a0,a1,b0,b1),W=W,B=B)

def forward(blob, X):
    s = np.where(blob["std"] > 1e-30, blob["std"], 1.0)
    z = np.where(blob["std"] > 1e-30, (X - blob["mean"]) / s, 0.0)
    for i in range(blob["nl"]):
        z = z @ blob["W"][i].T + blob["B"][i]
        z = z * (1.0/(1.0+np.exp(-z)))            # SiLU
    return (z @ blob["W"][-1].T + blob["B"][-1]).ravel()

def calibrate(blob, logit, log_nw):
    a0,a1,b0,b1 = blob["platt"]
    return 1.0/(1.0+np.exp(-((a0+a1*log_nw)*logit + (b0+b1*log_nw))))

blob = load_blob(sys.argv[1])
print(f"blob: n_features={blob['nf']} n_hidden={blob['nh']} n_layers={blob['nl']} platt={blob['platt']}")

TRUE_1D = {28,29,30}
NAMES = {16:"pixel",17:"pixel",18:"pixel",23:"sstrip",24:"sstrip",25:"sstrip",
         28:"lstrip",29:"lstrip",30:"lstrip",20:"other"}
COLS = list(dict.fromkeys(list(GATE_SOURCE_COLUMNS) +
        ["volume_id","cand_hit_id","contrib_pids","branch_majority_pid",
         "majority_undefined","n_window","action_taken"]))
agg = {}
for ev in [int(x) for x in sys.argv[2].split(",")]:
    f = pq.ParquetFile(f"{sys.argv[3]}/expanded_event{ev:09d}.parquet")
    for rg in range(0, f.metadata.num_row_groups, max(1, f.metadata.num_row_groups//4))[:4]:
        t = f.read_row_group(rg, columns=COLS); n = t.num_rows
        cand = np.asarray(t["cand_hit_id"].to_numpy(zero_copy_only=False))
        undef = np.array([bool(x) for x in t["majority_undefined"].to_pylist()])
        keep = (~undef) & (cand != -1)
        if not keep.any(): continue
        pids = t.column("contrib_pids").combine_chunks()
        if isinstance(pids, pa.ChunkedArray):
            pids = pids.chunk(0) if pids.num_chunks==1 else pa.concat_arrays(pids.chunks)
        maj = np.asarray(t["branch_majority_pid"].to_numpy(zero_copy_only=False))
        fl = np.asarray(pc.list_flatten(pids)); pr = np.asarray(pc.list_parent_indices(pids))
        y = (np.bincount(pr[fl==maj[pr]], minlength=n) > 0) if len(pr) else np.zeros(n,bool)
        df = t.to_pandas()
        X = build_gate_features(df)
        nw = np.maximum(np.asarray(df["n_window"], float), 1.0)
        p = calibrate(blob, forward(blob, X.astype(np.float64)), np.log(nw))
        vol = np.asarray(df["volume_id"])
        for v in np.unique(vol[keep]):
            m = keep & (vol == v)
            if m.sum() == 0: continue
            a = agg.setdefault(int(v), dict(n=0,pos=0,acc=0,acc_pos=0,tp=0,sum_p=0.0))
            a["n"] += int(m.sum()); a["pos"] += int(y[m].sum())
            hi = m & (p >= 0.5)
            a["acc"] += int(hi.sum()); a["acc_pos"] += int((hi & y).sum())
            a["tp"] += int((m & y & (p>=0.5)).sum()); a["sum_p"] += float(p[m].sum())

print(f"\n{'vol':>4}{'type':>8}{'rows':>10}{'base pos%':>11}{'gate acc%':>11}{'purity%':>10}{'recall%':>10}{'meanP':>8}")
for v in sorted(agg):
    a = agg[v]
    base = a["pos"]/a["n"]; accr = a["acc"]/a["n"]
    pur = a["acc_pos"]/a["acc"] if a["acc"] else 0.0
    rec = a["acc_pos"]/a["pos"] if a["pos"] else 0.0
    print(f"{v:>4}{NAMES.get(v,'?'):>8}{a['n']:>10,}{base:>10.3%}{accr:>11.3%}{pur:>10.2%}{rec:>10.2%}{a['sum_p']/a['n']:>8.3f}")
