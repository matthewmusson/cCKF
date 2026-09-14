# NERSC.md — where everything lives on Perlmutter, and how to set up your own copy

Companion to `README.md`. Everything here was checked on 2026-09-14 unless
marked otherwise. `<user>` is your NERSC username; Matthew's is `mussonm`.

## 1. Logging in

NERSC requires multi-factor authentication. The workable pattern for a
scripted session is a 24-hour SSH certificate from `sshproxy`:

```bash
sshproxy -u <user>            # asks for password + OTP, writes ~/.ssh/nersc and ~/.ssh/nersc-cert.pub
ssh -i ~/.ssh/nersc -o IdentitiesOnly=yes <user>@perlmutter.nersc.gov
```

`IdentitiesOnly=yes` matters: without it ssh offers every key in your agent
first and NERSC closes the connection with "Too many authentication
failures". Longer-lived certificates exist ("scoped" keys) but need a
NERSC consulting ticket. `sshproxy` is documented at
https://docs.nersc.gov/connect/mfa/#sshproxy .

Never copy the certificate to a machine you do not control, and never let a
coding agent handle the password or OTP. Run `sshproxy` yourself; the agent
only needs the resulting key file.

## 2. Allocation

| Account | Hardware | What it was used for |
|---|---|---|
| `atlas` | CPU nodes (128 cores, 512 GB) | everything except training |
| `atlas_g` | GPU nodes (4× A100) | `train_gate.sbatch`, `train_value_*.sbatch` |

Units in Iris (https://iris.nersc.gov) are **node-hours**. Matthew's
personal share was 200 CPU node-hours and 100 GPU node-hours for the summer;
the project PI can raise it. An `salloc` bills wall-clock whether or not it
is busy, so keep interactive sessions short. Check usage with `sacct` or in
Iris.

Queues used: `debug` (30 min max, starts within minutes; one stage-1 event
or one expansion fits), `regular` (up to 12 h in practice; used for the
90-minute audit, the 3-hour regenerations, the tier-3 array), `interactive`
(`salloc -N1 -C cpu -q interactive -t 00:45:00 -A atlas`; for the incremental
build), and `-C gpu -A atlas_g` for training.

## 3. Filesystems

| Path | Quota | Lifetime | Use |
|---|---|---|---|
| `/global/homes/<x>/<user>` (`$HOME`) | 40 GB | permanent | dotfiles, nothing else |
| `/global/cfs/cdirs/atlas/<user>` | shared project quota | permanent, backed up | the git checkout, small results, the handoff bundle |
| `/pscratch/sd/<x>/<user>` (`$SCRATCH`) | 20 TB | **purged after 8 weeks without access** | builds, run outputs, parquets, caches |
| HPSS tape (`hsi`, `htar`) | large | permanent | not used; needs one interactive `hsi` login first |

`showquota` prints your numbers. The atlas project space was at 96% of space
and 98% of inodes on 2026-09-08, which is why the large intermediates stayed
on scratch and only a 37 GB bundle went to CFS. Every scratch directory
listed below was last written between 2026-08-25 and 2026-09-10; it becomes
purge-eligible eight weeks after its last access, i.e. from late October
2026. Copy what you want to keep before then, or re-generate it (the plan in
`NEXT_STEPS.md` re-generates almost all of it).

Scratch is private by default (`drwx------`). CFS directories under
`mussonm/` are group-readable for `atlas` through ACLs (`getfacl <dir>` to
confirm); if something under `cckf_handoff/` is unreadable, ask Matthew to
run `setfacl -R -m g:atlas:rX` on it.

## 4. The input data

### ColliderML

```
/global/cfs/cdirs/m4958/data/ColliderML/simulation/hard_scatter/
    ttbar/v1/runs/<0..1191>/edm4hep.root       128 events each, 5.4 GB, 1,192 runs
    jets/, zmumu/, dihiggs/, pileup_only/, ...  other channels, same layout
```

Read-only and world-readable; you do not need to be in project `m4958`. Each
run directory also holds the dataset authors' own ACTS reconstruction of the
same events (`trackstates_ckf.root`, `tracksummary_ckf.root`,
`performance_finding_{ckf,ambi}.root`, `measurements.root`,
`particles.root`, `spacepoints.root`). Those files come from their config,
not ours, but they are a useful sanity reference for what a stock CKF does
on this data.

`digi_and_reco.py` reads a run file directly: `--input-file <path>` plus
`events:` and `skip:` in the YAML choose the events.

### The 64-event working file

```
/pscratch/sd/m/mussonm/cckf/modal_backup/events/edm4hep.root     6.2 GB
```

Events 0 to 63 of run 0, extracted with `scripts/subset_edm4hep.py`. Every
number in `experiments/LOG.md` Part II and every training parquet came from
this file. Events 0 to 31 are the working set (train / val / cal split in
`configs/event_split.json`); events 32 to 63 are the sealed evaluation set.
The classical Phase-1 work in July did evaluate on 32 to 63 (see the caveat
in `docs/04_ambiguity_and_metrics.md`); nothing since August has touched
them. To move past 64 events, read the ColliderML runs directly instead of
copying more subsets.

## 5. Matthew's layout

### Scratch: `/pscratch/sd/m/mussonm/cckf/`

```
acts/{src,build,install,ccache}   the patched ACTS build (bootstrap once, then incremental)   0.9 GB install
modal_backup/                     the one-time mirror of the Modal volume (2026-08-25)
    events/edm4hep.root           the 64-event input (above)
    results/pilot_*/              32 stage-1 run directories, one event each, envelope config
    models/, weights/, optimizer/, analysis/, cache/    the Modal-era artefacts
reexpanded/                       expanded_event0000000{00..31}.parquet, 32 files, 176 GB,
                                  expanded 2026-08-25 (endcap fix) with the REVERSED step_k
reexpanded_sel/                   the same with is_ckf_selected patched in
caches_v3/, caches_v3_pure/       gate feature caches (majority label, pure-seed label)
vcache_v3/                        value cache
models_v3/                        trained checkpoints (also in cckf_handoff/models_v3)
regen_tight/, regen_fast/         8-event stage-1 runs of the tight and fast classical configs
runs_*/                           every learned-pair and classical sweep run (performance + timing copied to CFS)
tier3/{worklists,hits}/           tier-3 walker worklists and rollout hits per event
results/                          the Pareto CSVs (copied into the repo under results/pareto/)
logs/                             Slurm and memory logs
*.sbatch, *.sh, *.py              the job files, now versioned under scripts/nersc/
```

### CFS: `/global/cfs/cdirs/atlas/mussonm/`

```
cCKF/                             the git checkout that jobs run from (pull it; never scp into it)
cckf_handoff/                     the 37 GB handoff bundle, group-readable:
    examples/envelope_event4/     one complete envelope stage-1 run (events 4, 5) + expanded parquet for event 4
    examples/regen_tight/, regen_fast/   8-event tight/fast runs with their parquets under reexpanded/
    weights_v3/{maj,pure}/        the deployed gate.bin / value.bin with provenance.json
    models_v3/                    the checkpoints behind them
    results/                      Pareto CSVs and the event-4 audits
    runs/runs_*/<point>/          performance and timing files of every sweep run (no ROOT states)
    tier3_event4/{worklists,hits} one event's rollout worklist and hits
    tier3_targets/                stitched tier-3 targets
    winfail_unc_envelope/         the 32 window-failure npz arrays
ODD_v5/install/                   the Open Data Detector build (share/OpenDataDetector has xml, config, material maps)
venvs/modal/                      the Python training environment (see §6)
```

## 6. Software

Everything runs inside one container image, on login nodes and compute
nodes alike:

```bash
IMG=ghcr.io/opendatadetector/sw:0.2.2_linux-ubuntu24.04_gcc-13.3.0
shifter --image=$IMG -- <command>
```

Inside it: gcc 13, ROOT, DD4hep, podio, EDM4hep, and a stock ACTS, all from
spack. Two things to know:

- The container's system Python is 3.12 but the ACTS bindings are built for
  the spack Python 3.13. `scripts/run_cckf_nersc.sh` finds the right
  interpreter and sets `PYTHONPATH` / `LD_LIBRARY_PATH` / `ROOT_INCLUDE_PATH`
  so that our patched ACTS install shadows the stock one and ROOT can find
  the EDM4hep dictionaries. Use that script rather than reproducing its
  environment by hand.
- The training venv `venvs/modal` was created **with the container's
  Python**, so its `bin/python` is a dangling symlink outside the container.
  Run it as `shifter --image=$IMG -- $V/bin/python ...`, exactly as the
  sbatch files do. It holds everything in `requirements-train.txt` (torch,
  pyarrow, uproot, awkward, scikit-learn, wandb).

The patched ACTS is the pinned commit `4de1dcbb` with
`instrumentation.patch` and the files under `acts_patches/` applied
(`scripts/apply_cckf_integration.sh`), built by `scripts/build_cckf_nersc.sh`
into `$SCRATCH/cckf/acts`. The ODD geometry is a separate install on CFS
(`ODD_v5/install`, built 2026-08-24 from the `OpenDataDetector` repository,
tag v5); `run_cckf_nersc.sh` points at it through `ODD_INSTALL`.

## 7. Setting up your own copy

Ten steps, in order. Steps 1 to 4 are one-off.

1. **Checkout.** On a login node:
   ```bash
   cd /global/cfs/cdirs/atlas/<user>
   git clone https://github.com/matthewmusson/cCKF.git
   ```
   Keep this the only copy you run jobs from. Edit elsewhere, push, and
   `git pull` here; do not `scp` source files into it (that is how a stale
   copy silently diverges).

2. **Paths in the job files.** `scripts/nersc/*` hard-code Matthew's two
   roots. Replace them once:
   ```bash
   cd cCKF
   sed -i 's#/pscratch/sd/m/mussonm#/pscratch/sd/<x>/<user>#g; s#/global/cfs/cdirs/atlas/mussonm/cCKF#/global/cfs/cdirs/atlas/<user>/cCKF#g' scripts/nersc/*.sbatch scripts/nersc/*.sh scripts/nersc/*.py
   ```
   Leave `/global/cfs/cdirs/atlas/mussonm/ODD_v5` and `.../venvs/modal` as
   they are if you want to reuse them (both are group-readable), or point
   them at your own builds from steps 3 and 4.

3. **Training environment** (skip if you reuse `venvs/modal`):
   ```bash
   V=/global/cfs/cdirs/atlas/<user>/venvs/train
   shifter --image=$IMG -- bash -c "python3.13 -m venv $V && $V/bin/pip install -r requirements-train.txt"
   ```
   If `python3.13` is not on the container's PATH, use the full spack path
   printed by `shifter --image=$IMG -- ls -d /spack/opt/spack/linux-x86_64/python-3.13*/bin/python3`.

4. **Build the patched ACTS** into your scratch (about an hour on one node):
   ```bash
   mkdir -p $SCRATCH/cckf/logs
   sbatch scripts/nersc/bootstrap_build.sbatch
   ```
   After that, any change under `acts_patches/` is a two-minute incremental
   build from an interactive node:
   ```bash
   salloc -N1 -C cpu -q interactive -t 00:45:00 -A atlas
   ./scripts/build_cckf_nersc.sh --install
   exit
   ```
   The script rsyncs the changed headers into the ACTS source tree and
   rebuilds only the two targets that include them. The bootstrap is guarded
   by a stamp file because its CMakeLists edits are not idempotent; if you
   ever need a clean slate, delete `$SCRATCH/cckf/acts` entirely rather than
   re-running `--bootstrap` on top of a half-built tree.

5. **Input.** Either point at ColliderML directly
   (`DATA=/global/cfs/cdirs/m4958/data/ColliderML/simulation/hard_scatter/ttbar/v1/runs/0/edm4hep.root`)
   or make a smaller file with `scripts/subset_edm4hep.py` (run inside
   shifter; it needs the EDM4hep dictionaries). Copying Matthew's
   `modal_backup/events/edm4hep.root` out of his scratch also works while it
   exists.

6. **First stage-1 run** (one event, debug queue, 10 to 20 minutes):
   ```bash
   sbatch scripts/nersc/run_p1_input.sbatch cckf_envelope.yaml first_test $DATA $SCRATCH/cckf/runs
   ```
   The output directory `$SCRATCH/cckf/runs/first_test/` should contain
   `trackstates_ckf.root`, the `event*-*.csv` files and
   `performance_finding_{ckf,ambi}.root`. `docs/05_trackstates_root.md`
   shows what to look for inside.

7. **Expansion.** `scripts/nersc/reexpand.sbatch` runs `reexpand_ev.py`
   from scratch; copy `scripts/nersc/reexpand_ev.py` to `$SCRATCH/cckf/` and
   edit its repo path. An envelope event needs the whole node's memory
   (`--mem=0 -c 128`); five of the 32 events did not fit and went through
   `reexpand_event.sbatch` (chunked). Always follow with
   `scripts/audit_expansion.py` (`README.md` §4).

8. **Caches and training.** `build_caches.sbatch`, `build_vcache.sbatch`,
   then `train_gate.sbatch` / `train_value_v3.sbatch` on `atlas_g`. Each
   takes its paths as arguments; read the header.

9. **Sweeps.** `scripts/launch_sweep_nersc.sh <weights_dir> <event>` submits
   the (τ_g, τ_v) grid; `scripts/ehvi_sweep.py` runs on a login node under
   `nohup` and submits its own jobs.

10. **Keeping things.** Anything you want to survive the scratch purge goes
    to `/global/cfs/cdirs/atlas/<user>/` (mind the project quota) or to HPSS
    (`htar -cvf cckf/<name>.tar <dir>`, after one interactive `hsi` login to
    initialise your HPSS credentials).

## 8. Habits that saved time

- `squeue -u <user>` and `sacct -j <jobid> --format=JobID,State,Elapsed,MaxRSS`
  before reading logs; `MaxRSS` tells you whether a job died of memory.
- Job logs go to `$SCRATCH/cckf/logs/`; the sbatch files write a memory
  trace next to them.
- `/tmp` on a login node is not shared with compute nodes or with other
  login nodes; stage temporary files under `$SCRATCH`.
- Copy data with `scp`/`rsync` straight to NERSC from wherever it is; never
  through a third machine.
- Commit before any run longer than five minutes and put the commit hash
  in the log entry (`experiments/LOG.md`); `provenance.json` next to the
  deployed weights records the same thing for models.
