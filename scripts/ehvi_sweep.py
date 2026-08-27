"""Adaptive (tau_g, tau_v) Pareto densification with qEHVI.

Optuna study over the two decision thresholds, two objectives:
maximize efficiency (eps_DM, post-ambiguity), minimize track fake rate.
The sampler is optuna-integration's BoTorchSampler, which for
multi-objective studies uses expected hypervolume improvement (qEHVI).
Warm-started from an existing grid-sweep CSV so the GP begins with the
coarse front instead of random exploration.

Each evaluation is one sbatch run of run_p1_input.sbatch on one event
(~4 min). Failed or timed-out runs are told (eff=0, fake=1): a runaway
low-threshold point is informative, and this steers the sampler away
without crashing the study.

Run inside shifter with the venv python plus PyROOT on PYTHONPATH
(see --help epilog for the exact incantation). Designed to run on a
login node under nohup; it only submits jobs and reads results.

Usage:
    python scripts/ehvi_sweep.py --pair maj \
        --weights $SCRATCH/cckf/weights_v3/maj \
        --warm-csv $SCRATCH/cckf/results/pareto_maj.csv \
        --runs-dir $SCRATCH/cckf/runs_maj_ehvi \
        --out $SCRATCH/cckf/results/pareto_maj_dense.csv \
        --rounds 5 --batch 8 --event 4
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import time
from pathlib import Path

import optuna
from optuna_integration import BoTorchSampler

REPO = Path("/global/cfs/cdirs/atlas/mussonm/cCKF")
SCRATCH = Path(os.environ["SCRATCH"]) / "cckf"
BASE_CONFIG = REPO / "configs" / "nersc_cckf_full_dm.yaml"
LO, HI = 0.05, 0.95


def sq(cmd: list[str]) -> str:
    return subprocess.run(cmd, capture_output=True, text=True).stdout


def read_run(run_dir: Path) -> dict | None:
    """Post-ambiguity efficiency and fake rate from one finished run."""
    import ROOT

    ROOT.gROOT.SetBatch(True)
    ROOT.gErrorIgnoreLevel = ROOT.kFatal
    ambi = run_dir / "performance_finding_ambi.root"
    if not ambi.exists():
        return None
    tf = ROOT.TFile.Open(str(ambi))
    try:
        eff_obj = tf.Get("trackeff_vs_eta")
        total = eff_obj.GetTotalHistogram().Integral()
        passed = eff_obj.GetPassedHistogram().Integral()
        fake = float(tf.Get("fakeratio_tracks")[0])
        if total <= 0:
            return None
        return {"efficiency": passed / total, "fake_rate": fake}
    except Exception:
        return None
    finally:
        tf.Close()


def launch(tag: str, g: float, v: float, weights: Path, event: int,
           runs_dir: Path) -> str:
    cfg = REPO / "configs" / f"_{runs_dir.name}_{tag}.yaml"
    text = BASE_CONFIG.read_text()
    import re

    text = re.sub(r"^cckf_gate_threshold: .*", f"cckf_gate_threshold: {g}",
                  text, flags=re.M)
    text = re.sub(r"^cckf_value_threshold: .*", f"cckf_value_threshold: {v}",
                  text, flags=re.M)
    text = re.sub(r"^cckf_gate_weights: .*",
                  f"cckf_gate_weights: {weights}/gate.bin", text, flags=re.M)
    text = re.sub(r"^cckf_value_weights: .*",
                  f"cckf_value_weights: {weights}/value.bin", text, flags=re.M)
    text = re.sub(r"^skip: .*", f"skip: {event}", text, flags=re.M)
    cfg.write_text(text)
    out = sq([
        "sbatch", "--parsable", "--qos=regular", "--time=01:00:00",
        f"--output={SCRATCH}/logs/{runs_dir.name}_{tag}_%j.out",
        str(SCRATCH / "run_p1_input.sbatch"), cfg.name, tag,
        str(SCRATCH / "modal_backup/events/edm4hep.root"), str(runs_dir),
    ])
    return out.strip()


def wait_all(job_ids: list[str], poll_s: int = 120) -> None:
    ids = ",".join(job_ids)
    while True:
        states = sq(["sacct", "-j", ids, "--format=State", "-X", "-n"]).split()
        if states and all(
            s.split("+")[0] in
            {"COMPLETED", "FAILED", "TIMEOUT", "CANCELLED", "OUT_OF_MEMORY"}
            for s in states
        ):
            return
        time.sleep(poll_s)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pair", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--warm-csv", required=True)
    ap.add_argument("--runs-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--event", type=int, default=4)
    args = ap.parse_args()

    runs_dir = Path(args.runs_dir)
    runs_dir.mkdir(parents=True, exist_ok=True)
    weights = Path(args.weights)

    # n_startup_trials below the warm count so the GP is active from round 1.
    sampler = BoTorchSampler(n_startup_trials=8, seed=17)
    study = optuna.create_study(
        directions=["maximize", "minimize"], sampler=sampler,
        study_name=f"ehvi_{args.pair}",
    )

    rows: list[dict] = []
    with open(args.warm_csv) as fh:
        for r in csv.DictReader(fh):
            g, v = float(r["tau_g"]), float(r["tau_v"])
            eff, fake = float(r["efficiency"]), float(r["fake_rate"])
            study.add_trial(optuna.trial.create_trial(
                params={"tau_g": g, "tau_v": v},
                distributions={
                    "tau_g": optuna.distributions.FloatDistribution(LO, HI),
                    "tau_v": optuna.distributions.FloatDistribution(LO, HI),
                },
                values=[eff, fake],
            ))
            rows.append({"tau_g": g, "tau_v": v, "efficiency": eff,
                         "fake_rate": fake, "source": "warm"})
    print(f"[{args.pair}] warm-started with {len(rows)} grid points",
          flush=True)

    for rnd in range(args.rounds):
        batch = []
        for _ in range(args.batch):
            t = study.ask({
                "tau_g": optuna.distributions.FloatDistribution(LO, HI),
                "tau_v": optuna.distributions.FloatDistribution(LO, HI),
            })
            g = round(t.params["tau_g"], 4)
            v = round(t.params["tau_v"], 4)
            tag = f"e{t.number:03d}_g{g}_v{v}".replace(".", "p")
            jid = launch(tag, g, v, weights, args.event, runs_dir)
            batch.append((t, g, v, tag, jid))
            print(f"[{args.pair}] round {rnd} trial {t.number}: "
                  f"g={g} v={v} job {jid}", flush=True)
        wait_all([jid for *_, jid in batch])
        for t, g, v, tag, jid in batch:
            m = read_run(runs_dir / tag)
            if m is None:
                study.tell(t, [0.0, 1.0])
                rows.append({"tau_g": g, "tau_v": v, "efficiency": 0.0,
                             "fake_rate": 1.0, "source": f"fail_j{jid}"})
            else:
                study.tell(t, [m["efficiency"], m["fake_rate"]])
                rows.append({"tau_g": g, "tau_v": v,
                             "efficiency": m["efficiency"],
                             "fake_rate": m["fake_rate"],
                             "source": f"ehvi_r{rnd}"})
        with open(args.out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        n_front = len(study.best_trials)
        print(f"[{args.pair}] round {rnd} done: {len(rows)} evals, "
              f"{n_front} on front", flush=True)

    print(f"[{args.pair}] DONE {args.out}", flush=True)


if __name__ == "__main__":
    main()
