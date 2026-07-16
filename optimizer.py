#!/usr/bin/env python3
"""
Bayesian optimization of ACTS CKF + seeding parameters on ColliderML data.

Adapted from cmuchancel/acts full_chain_odd_optimizer.py.
Key differences:
  - Adds CKF parameters (chi2 cut, branch cap, nMeasMin, holes) to the search space
  - Uses cached edm4hep.root files instead of generating events per trial
  - Runs digi_and_reco.py via shifter on NERSC
  - Generates per-trial YAML configs compatible with our config system

Usage (NERSC):
  python optimizer.py --n-seed-trials 10 --n-guided-trials 40 --verbose
  python optimizer.py --multi-objective --events-per-trial 32
  python optimizer.py --resume-from bayesian_ckf_results/
"""

import sys
import tempfile
import shutil
import argparse
import time
import json
import math
import subprocess
import pathlib
import os
from typing import Dict
from pathlib import Path
from datetime import datetime

import yaml
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import uproot
import xopt
from xopt.vocs import VOCS
from xopt.evaluator import Evaluator
from xopt.generators.bayesian import ExpectedImprovementGenerator
from xopt.generators.bayesian.mobo import MOBOGenerator

# ---------------------------------------------------------------------------
# Globals (set from CLI)
# ---------------------------------------------------------------------------
COMPLETED_TRIALS: int = 0
VERBOSE: bool = False
K_VALUE: float = 5.0
EVENTS_PER_TRIAL: int = 128
OPT_VARS: set = set()
EDM4HEP_FILE: str = ""
SHIFTER_IMAGE: str = ""
SCRIPT_DIR: Path = Path(__file__).parent.absolute()

# ---------------------------------------------------------------------------
# Parameter search space
# ---------------------------------------------------------------------------
VOCS_FULL = VOCS(
    variables={
        # --- Seeding parameters ---
        "num_seeds_per_spm": (1, 80),
        "seed_minPt": (0.3, 2.0),
        "seed_impactMax": (1.0, 10.0),
        # --- CKF parameters ---
        "ckf_chi2CutOffMeasurement": (5.0, 50.0),
        "ckf_chi2CutOffOutlier": (10.0, 100.0),
        "ckf_numMeasurementsCutOff": (1, 10),
        "ckf_nMeasurementsMin": (4, 9),
        "ckf_maxHolesAndOutliers": (1, 6),
        "ckf_ptMin": (0.3, 2.0),
    },
    objectives={
        "efficiency": "MAXIMIZE",
        "fakerate": "MINIMIZE",
        "duplicaterate": "MINIMIZE",
        "runtime": "MINIMIZE",
    },
)

VOCS_SINGLE = VOCS(
    variables=VOCS_FULL.variables,
    objectives={"score": "MINIMIZE"},
    constraints={"fail": ["LESS_THAN", 0.5]},
)

DEFAULT_PARAMS = {
    "num_seeds_per_spm": 40,
    "seed_minPt": 0.5,
    "seed_impactMax": 3.0,
    "ckf_chi2CutOffMeasurement": 15.0,
    "ckf_chi2CutOffOutlier": 25.0,
    "ckf_numMeasurementsCutOff": 1,
    "ckf_nMeasurementsMin": 6,
    "ckf_maxHolesAndOutliers": 3,
    "ckf_ptMin": 0.7,
}

ALL_VARS = list(VOCS_FULL.variables.keys())
SEEDING_VARS = ["num_seeds_per_spm", "seed_minPt", "seed_impactMax"]
CKF_VARS = [
    "ckf_chi2CutOffMeasurement",
    "ckf_chi2CutOffOutlier",
    "ckf_numMeasurementsCutOff",
    "ckf_nMeasurementsMin",
    "ckf_maxHolesAndOutliers",
    "ckf_ptMin",
]


def vprint(*args, **kwargs):
    if VERBOSE:
        print(*args, **kwargs)


# ---------------------------------------------------------------------------
# Config generation
# ---------------------------------------------------------------------------
def make_trial_config(params: Dict[str, float], output_dir: Path) -> Path:
    """Write a YAML config for one trial, compatible with digi_and_reco.py."""
    full = DEFAULT_PARAMS.copy()
    full.update(params)

    cfg = {
        "events": EVENTS_PER_TRIAL,
        "threads": 1,
        "digi": True,
        "reco": True,
        # seeding
        "num_seeds_per_spm": int(round(full["num_seeds_per_spm"])),
        "seed_minPt": float(full["seed_minPt"]),
        "seed_impactMax": float(full["seed_impactMax"]),
        # CKF
        "ckf_chi2CutOffMeasurement": float(full["ckf_chi2CutOffMeasurement"]),
        "ckf_chi2CutOffOutlier": float(full["ckf_chi2CutOffOutlier"]),
        "ckf_numMeasurementsCutOff": int(round(full["ckf_numMeasurementsCutOff"])),
        "ckf_nMeasurementsMin": int(round(full["ckf_nMeasurementsMin"])),
        "ckf_maxHolesAndOutliers": int(round(full["ckf_maxHolesAndOutliers"])),
        "ckf_ptMin": float(full["ckf_ptMin"]),
        # performance output
        "ckf_finding_performance": True,
        "ambi_finding_performance": True,
    }

    config_path = output_dir / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)
    return config_path


# ---------------------------------------------------------------------------
# Evaluation functions
# ---------------------------------------------------------------------------
def _run_trial(params: Dict[str, float]) -> Dict[str, float]:
    """Run digi_and_reco.py via shifter and parse metrics."""
    workdir = Path(tempfile.mkdtemp(prefix="cckf_trial_"))
    results_dir = workdir / "results"
    results_dir.mkdir()

    try:
        # Copy pipeline code
        shutil.copy2(SCRIPT_DIR / "digi_and_reco.py", workdir / "digi_and_reco.py")
        shutil.copytree(SCRIPT_DIR / "utils", workdir / "utils")

        # Write trial config
        config_path = make_trial_config(params, workdir)

        # Build shifter command using spack Python 3.13 + symlinked acts module
        acts_dir = os.environ.get(
            "CCKF_ACTS_DIR",
            "/spack/opt/spack/linux-x86_64/acts-main-udwtnx3aoh5lh6s76slc2fzc5szhwe7y",
        )
        spack_python = os.environ.get(
            "CCKF_PYTHON",
            "/spack/opt/spack/linux-x86_64/python-3.13.11-awxtqzerpdzhatylv3uagd35ebciqs3o/bin/python3",
        )
        setup_cmds = (
            f"rm -rf /tmp/acts_pypath && mkdir -p /tmp/acts_pypath && "
            f"ln -sf {acts_dir}/python /tmp/acts_pypath/acts && "
            f"export PYTHONPATH=/tmp/acts_pypath:$PYTHONPATH && "
            f"export LD_LIBRARY_PATH={acts_dir}/lib:$LD_LIBRARY_PATH"
        )
        run_cmd = (
            f"{spack_python} {workdir / 'digi_and_reco.py'} "
            f"--config {config_path} "
            f"--input-file {EDM4HEP_FILE} "
            f"--output {workdir} "
            f"--output-subdir results "
            f"--digi --reco"
        )
        cmd = [
            "shifter",
            f"--image={SHIFTER_IMAGE}",
            "--",
            "bash", "-c",
            f"{setup_cmds} && {run_cmd}",
        ]

        vprint(f"  CMD: {' '.join(cmd[:6])} ...")

        t0 = time.perf_counter()
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        wall_time = time.perf_counter() - t0

        if proc.returncode != 0:
            vprint(f"  FAILED (rc={proc.returncode})")
            vprint(f"  stderr: {proc.stderr[-500:]}")
            return None

        # Parse performance ROOT file
        perf_file = results_dir / "performance_finding_ckf.root"
        if not perf_file.exists():
            vprint("  performance_finding_ckf.root not found")
            return None

        with uproot.open(str(perf_file)) as rh:
            eff = rh["eff_particles"].member("fElements")[0] * 100
            fake = rh["fakeratio_tracks"].member("fElements")[0] * 100
            dup = rh["duplicateratio_tracks"].member("fElements")[0] * 100

        # Try to get per-algorithm timing from ACTS timing output
        timing_file = results_dir / "timing.tsv"
        if timing_file.exists():
            try:
                timing = pd.read_csv(timing_file, sep="\t")
                col = "identifier" if "identifier" in timing.columns else "name"
                ckf_row = timing[timing[col].str.contains("TrackFinding", na=False)]
                run_time = ckf_row["time_perevent_s"].values[0] if len(ckf_row) else wall_time
            except Exception:
                run_time = wall_time
        else:
            run_time = wall_time

        return {
            "efficiency": eff,
            "fakerate": fake,
            "duplicaterate": dup,
            "runtime": run_time,
        }

    except subprocess.TimeoutExpired:
        vprint("  TIMEOUT")
        return None
    except Exception as exc:
        vprint(f"  ERROR: {type(exc).__name__}: {exc}")
        return None
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def evaluate_single_objective(params: Dict[str, float]) -> Dict[str, float]:
    global COMPLETED_TRIALS
    trial_idx = COMPLETED_TRIALS + 1
    stamp = datetime.now().strftime("%H:%M:%S")
    vprint(f"{stamp}  START trial={trial_idx}  params={params}")

    full_params = DEFAULT_PARAMS.copy()
    full_params.update(params)
    metrics = _run_trial(full_params)

    COMPLETED_TRIALS += 1

    if metrics is None or any(
        math.isnan(v) or math.isinf(v) for v in metrics.values()
    ):
        vprint(f"{datetime.now().strftime('%H:%M:%S')}  FAILED trial={trial_idx}")
        return {
            "score": 1.0,
            "efficiency": 0.0,
            "fakerate": 100.0,
            "duplicaterate": 100.0,
            "runtime": 999.0,
            "fail": 1.0,
        }

    k = K_VALUE
    score = -(metrics["efficiency"] - (metrics["fakerate"] + metrics["duplicaterate"] / k + metrics["runtime"] / k))

    vprint(
        f"{datetime.now().strftime('%H:%M:%S')}  END   trial={trial_idx}  "
        f"eff={metrics['efficiency']:.2f}  fake={metrics['fakerate']:.2f}  "
        f"dup={metrics['duplicaterate']:.2f}  time={metrics['runtime']:.3f}s  "
        f"score={score:.3f}"
    )

    return {
        "score": score,
        "fail": 0.0,
        **metrics,
    }


def evaluate_multi_objective(params: Dict[str, float]) -> Dict[str, float]:
    global COMPLETED_TRIALS
    trial_idx = COMPLETED_TRIALS + 1
    stamp = datetime.now().strftime("%H:%M:%S")
    vprint(f"{stamp}  START trial={trial_idx}")

    full_params = DEFAULT_PARAMS.copy()
    full_params.update(params)
    metrics = _run_trial(full_params)

    COMPLETED_TRIALS += 1

    if metrics is None:
        vprint(f"{datetime.now().strftime('%H:%M:%S')}  FAILED trial={trial_idx}")
        return {
            "efficiency": 0.0,
            "fakerate": 100.0,
            "duplicaterate": 100.0,
            "runtime": 999.0,
        }

    vprint(
        f"{datetime.now().strftime('%H:%M:%S')}  END   trial={trial_idx}  "
        f"eff={metrics['efficiency']:.2f}  fake={metrics['fakerate']:.2f}  "
        f"dup={metrics['duplicaterate']:.2f}  time={metrics['runtime']:.3f}s"
    )
    return metrics


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_progress(df: pd.DataFrame, n_seed: int, outdir: Path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    trials = df.index + 1
    ax1.scatter(
        trials[:n_seed], df["inv_score"][:n_seed],
        color="red", alpha=0.7, s=50, label="Random seed",
    )
    ax1.scatter(
        trials[n_seed:], df["inv_score"][n_seed:],
        color="blue", alpha=0.7, s=50, label="Bayesian-guided",
    )
    ax1.set_xlabel("Trial")
    ax1.set_ylabel("Inverse score (higher = better)")
    ax1.set_title("Full optimization progress")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    if len(df) > n_seed:
        ax2.scatter(trials[n_seed:], df["inv_score"][n_seed:], color="blue", alpha=0.7, s=50)
        ax2.set_xlabel("Trial")
        ax2.set_ylabel("Inverse score")
        ax2.set_title("Bayesian-guided trials")
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, "No guided trials yet", ha="center", va="center", transform=ax2.transAxes)

    plt.tight_layout()
    png = outdir / "optimization_progress.png"
    plt.savefig(png, dpi=150, bbox_inches="tight")
    plt.close()
    return str(png)


def find_pareto_optimal(df: pd.DataFrame) -> pd.DataFrame:
    obj = df[["efficiency", "fakerate", "duplicaterate", "runtime"]].values.copy()
    obj[:, 0] = -obj[:, 0]  # negate efficiency for minimization

    mask = np.ones(len(obj), dtype=bool)
    for i in range(len(obj)):
        if not mask[i]:
            continue
        for j in range(len(obj)):
            if i != j and mask[j]:
                if np.all(obj[i] <= obj[j]) and np.any(obj[i] < obj[j]):
                    mask[j] = False
    return df[mask].copy()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_cli_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Bayesian optimization of ACTS CKF + seeding parameters",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--opt-vars",
        type=str,
        default="all",
        help=(
            "'all' = all 9 variables, 'seeding' = seeding only, "
            "'ckf' = CKF only, or comma-separated list"
        ),
    )
    parser.add_argument("--resume-from", type=Path, default=None)
    parser.add_argument("--n-seed-trials", type=int, default=10)
    parser.add_argument("--n-guided-trials", type=int, default=40)
    parser.add_argument("--multi-objective", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("bayesian_ckf_results"),
    )
    parser.add_argument("--events-per-trial", type=int, default=128)
    parser.add_argument(
        "--edm4hep",
        type=Path,
        default=None,
        help="Path to edm4hep.root input file",
    )
    parser.add_argument(
        "--channel",
        type=str,
        default="ttbar",
        help="ColliderML channel (used to locate edm4hep if --edm4hep not given)",
    )
    parser.add_argument("--run-id", type=int, default=0)
    parser.add_argument("--k-value", type=float, default=5.0)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    global VERBOSE, K_VALUE, EVENTS_PER_TRIAL, OPT_VARS, COMPLETED_TRIALS
    global EDM4HEP_FILE, SHIFTER_IMAGE

    args = parse_cli_args()

    VERBOSE = args.verbose
    K_VALUE = args.k_value
    EVENTS_PER_TRIAL = args.events_per_trial

    # Resolve edm4hep input
    if args.edm4hep:
        EDM4HEP_FILE = str(args.edm4hep.resolve())
    else:
        data_base = os.environ.get(
            "CCKF_DATA",
            "/global/cfs/cdirs/m4958/data/ColliderML/simulation/hard_scatter",
        )
        EDM4HEP_FILE = f"{data_base}/{args.channel}/v1/runs/{args.run_id}/edm4hep.root"

    SHIFTER_IMAGE = os.environ.get(
        "CCKF_IMAGE",
        "ghcr.io/opendatadetector/sw:0.2.2_linux-ubuntu24.04_gcc-13.3.0",
    )

    if not Path(EDM4HEP_FILE).exists():
        print(f"ERROR: edm4hep file not found: {EDM4HEP_FILE}")
        sys.exit(1)

    # Parse variable subset
    if args.opt_vars.lower() == "all":
        opt_vars = ALL_VARS
    elif args.opt_vars.lower() == "seeding":
        opt_vars = SEEDING_VARS
    elif args.opt_vars.lower() == "ckf":
        opt_vars = CKF_VARS
    else:
        opt_vars = [v.strip() for v in args.opt_vars.split(",") if v.strip()]
        bad = set(opt_vars) - set(ALL_VARS)
        if bad:
            raise ValueError(f"Unknown variable(s): {', '.join(sorted(bad))}")

    OPT_VARS = set(opt_vars)

    # Build VOCS for selected variables
    if args.multi_objective:
        vocs_subset = VOCS(
            variables={k: VOCS_FULL.variables[k] for k in opt_vars},
            objectives=VOCS_FULL.objectives,
        )
        eval_fn = evaluate_multi_objective
        gen_cls = MOBOGenerator
        ref_point = {
            "efficiency": 0.0,
            "fakerate": 100.0,
            "duplicaterate": 100.0,
            "runtime": 1000.0,
        }
    else:
        vocs_subset = VOCS(
            variables={k: VOCS_SINGLE.variables[k] for k in opt_vars},
            objectives={"score": "MINIMIZE"},
            constraints={"fail": ["LESS_THAN", 0.5]},
        )
        eval_fn = evaluate_single_objective
        gen_cls = ExpectedImprovementGenerator
        ref_point = None

    # Output directory
    if args.resume_from:
        outdir = args.resume_from.resolve()
        if not outdir.exists():
            raise FileNotFoundError(f"Resume directory {outdir} does not exist")
        print(f"Resuming from {outdir}")
    else:
        outdir = args.output_dir.resolve()
        outdir.mkdir(parents=True, exist_ok=True)

    # Build xopt objects
    evaluator = xopt.Evaluator(function=eval_fn, executor=None, max_workers=1)

    if args.multi_objective:
        generator = gen_cls(vocs=vocs_subset, n_candidates=1, reference_point=ref_point)
    else:
        generator = gen_cls(vocs=vocs_subset, n_candidates=1)

    X = None
    n_seed = args.n_seed_trials
    n_guided = args.n_guided_trials

    # Resume logic
    if args.resume_from:
        csv_file = outdir / "optimization_history.csv"
        if csv_file.exists():
            df = pd.read_csv(csv_file)
            print(f"Loaded {len(df)} previous trials from CSV")

            saved_vars = [c for c in df.columns if c in ALL_VARS]
            if set(saved_vars) != set(opt_vars):
                print(f"Variable mismatch — using saved: {saved_vars}")
                opt_vars = saved_vars
                OPT_VARS = set(opt_vars)
                if args.multi_objective:
                    vocs_subset = VOCS(
                        variables={k: VOCS_FULL.variables[k] for k in opt_vars},
                        objectives=VOCS_FULL.objectives,
                    )
                    generator = gen_cls(vocs=vocs_subset, n_candidates=1, reference_point=ref_point)
                else:
                    vocs_subset = VOCS(
                        variables={k: VOCS_SINGLE.variables[k] for k in opt_vars},
                        objectives={"score": "MINIMIZE"},
                        constraints={"fail": ["LESS_THAN", 0.5]},
                    )
                    generator = gen_cls(vocs=vocs_subset, n_candidates=1)
                evaluator = xopt.Evaluator(function=eval_fn, executor=None, max_workers=1)

            X = xopt.Xopt(evaluator=evaluator, generator=generator, vocs=vocs_subset)
            X.data = df.copy()
            X.generator.add_data(df)
        else:
            print("No CSV found, starting fresh")

    if X is None:
        X = xopt.Xopt(evaluator=evaluator, generator=generator, vocs=vocs_subset)

    try:
        COMPLETED_TRIALS = len(X.data) if hasattr(X.data, "__len__") else 0
    except Exception:
        COMPLETED_TRIALS = 0

    mode = "multi-objective" if args.multi_objective else "single-objective"
    print(f"Starting {mode} optimization")
    print(f"Variables: {opt_vars}")
    print(f"Input: {EDM4HEP_FILE}")
    print(f"Events per trial: {EVENTS_PER_TRIAL}")
    print(f"Trials — seed: {n_seed}, guided: {n_guided}")
    if COMPLETED_TRIALS > 0:
        print(f"Already completed: {COMPLETED_TRIALS}")

    # Run trials
    if n_seed > 0:
        print(f"Running {n_seed} random seed trials...")
        for _ in range(n_seed):
            X.random_evaluate(1)

    if n_guided > 0:
        print(f"Running {n_guided} Bayesian-guided trials...")
        for _ in range(n_guided):
            X.step()

    # Save results
    df = X.data
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)
    if "trial" not in df.columns:
        df["trial"] = range(1, len(df) + 1)
    if not args.multi_objective and "inv_score" not in df.columns:
        df["inv_score"] = -df["score"]

    # Round integer params
    for col in ["num_seeds_per_spm", "ckf_numMeasurementsCutOff", "ckf_nMeasurementsMin", "ckf_maxHolesAndOutliers"]:
        if col in df.columns:
            df[col] = df[col].round().astype(int)

    csv_path = outdir / "optimization_history.csv"
    df.to_csv(csv_path, index=False)

    with open(outdir / "xopt_state.json", "w") as f:
        f.write(X.model_dump_json(indent=2))

    if args.multi_objective:
        pareto = find_pareto_optimal(df)
        pareto.to_csv(outdir / "pareto_optimal.csv", index=False)
        print(f"Done. {len(df)} trials, {len(pareto)} Pareto-optimal. Results in {outdir}")
    else:
        plot_progress(df, n_seed, outdir)
        best = df.loc[df["inv_score"].idxmax()]
        print(f"Done. Best trial:\n{best.to_dict()}")
        print(f"Results in {outdir}")


if __name__ == "__main__":
    main()
