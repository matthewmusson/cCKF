# modal_apps/ — legacy Modal entry points

The project ran on Modal (GPU/CPU cloud) from July to late August 2026. Since
2026-08-25 every data product and run lives on NERSC (see the top-level
README, "Where things live"). The Modal volume `surp-acts-data` no longer
exists, so **nothing here runs as-is**; the files are kept because they hold
code that the NERSC path still cites:

| File | Still useful for |
|------|------------------|
| `modal_acts.py` | the Optuna seeding / joint MOTPE optimizer (`JOINT_OP_POINTS`, the objective functions), the pilot checks, the ODD reco driver wiring |
| `modal_build_acts.py` | the original patched-ACTS build recipe and the re-expansion driver (`expand_all_events`) that `cckf/stage1_map.py` documents against |
| `modal_train.py` | the staged cache builders, the gate ablation trainer and the F9 exporter; the training recipe now runs through `scripts/train_gate.py` / `scripts/train_value.py` on NERSC |
| `Dockerfile.modal` | the spack-based image that mirrors the ODD software container used on NERSC |

Invoke from the repository root (the image definitions use cwd-relative
paths):

    modal run modal_apps/modal_train.py::export_curves --arms A,B,C

If you revive Modal, recreate the volume and re-upload from
`$SCRATCH/cckf/modal_backup/` on NERSC, which is the mirror of what the
volume held.
