# cCKF/scripts/export_weights.py
"""Export PyTorch gate/value weights + standardization + calibrator to a .bin blob.

Usage:
    python export_weights.py \
        --checkpoint output/gate_arm_a/model.pt \
        --standardization output/gate_arm_a/standardization.npz \
        --calibration output/gate_arm_a/calibration.json \
        --output weights/gate_arm_a.bin \
        --model-type gate
"""
import argparse
import json
import struct
from pathlib import Path

import numpy as np
import torch

# The C++ feature builders (acts_patches/cckf/CckfFeatures.hpp,
# CckfBranchStopper.hpp) always produce the FULL vectors below. A model
# trained on a subset (--feature-groups / --drop-features) is padded back to
# full width here: dropped columns get first-layer weight 0, mean 0, std 1,
# so the C++ computes the feature and multiplies it by nothing. The C++ never
# changes for an ablation.
FULL_FEATURES = {
    "gate": (
        "residual_l0", "residual_l1", "chol_S_00", "chol_S_10", "chol_S_11", "chi2_inc",
        "clus_s_u", "clus_s_v", "clus_q_tot", "clus_sigma_uu", "clus_sigma_uv", "clus_sigma_vv",
        "kappa_u", "kappa_v", "q_tilde", "n_window", "eta", "state_qop", "step_k",
        "pathInX0_interval", "pitch_u", "pitch_v", "thickness", "n_hits", "n_holes", "n_seq_holes",
    ),
    "value": (
        "eta", "state_qop", "sigma2_l0", "sigma2_l1", "n_hits", "n_holes", "n_seq_holes",
        "sum_gate_logodds", "min_gate_logodds", "step_k", "x0_accumulated",
    ),
}


def pad_to_full(first_weight, mean, std, feature_names, full_names):
    """Insert zero columns for features the model was not trained on.

    Returns (first_weight, mean, std) at full width, columns in
    ``full_names`` order. Raises ValueError if ``feature_names`` contains a
    name the C++ does not build (e.g. ``window_nsigma``) or is out of order.
    """
    full = list(full_names)
    names = list(feature_names)
    unknown = [n for n in names if n not in full]
    if unknown:
        raise ValueError(
            f"checkpoint features not built by the C++ side: {unknown}; "
            "drop them at training time (--drop-features) before exporting"
        )
    pos = [full.index(n) for n in names]
    if pos != sorted(pos):
        raise ValueError("checkpoint feature order differs from the C++ order")
    n_hidden = first_weight.shape[0]
    w = np.zeros((n_hidden, len(full)), dtype=np.float32)
    m = np.zeros(len(full), dtype=np.float32)
    s = np.ones(len(full), dtype=np.float32)
    w[:, pos] = first_weight
    m[pos] = mean
    s[pos] = std
    return w, m, s


def export(checkpoint_path, standardization_path, calibration_path,
           output_path, model_type):
    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location="cpu",
                      weights_only=False)
    # If wrapped in a top-level key (e.g. from training script)
    state_dict = ckpt
    for key in ("model_state_dict", "state_dict"):
        if isinstance(ckpt, dict) and key in ckpt:
            state_dict = ckpt[key]
            break

    # Extract weight/bias pairs in order
    layer_keys = sorted(
        {k.rsplit(".", 1)[0] for k in state_dict if "weight" in k},
        key=lambda k: int(k.split(".")[0]) if k.split(".")[0].isdigit() else k,
    )
    weights = []
    biases = []
    for key in layer_keys:
        w = state_dict[f"{key}.weight"].float().numpy()
        b = state_dict[f"{key}.bias"].float().numpy()
        weights.append(w)
        biases.append(b)

    n_hidden = weights[0].shape[0]
    n_layers = len(weights) - 1  # last layer is the head

    # Load standardization first: padding needs mean/std alongside the
    # first-layer weights. "ckpt" pulls mu/sigma from the checkpoint itself,
    # which is where the v3 training scripts store them.
    if standardization_path == "ckpt":
        mean = np.asarray(ckpt["mu"], dtype=np.float32)
        std = np.asarray(ckpt["sigma"], dtype=np.float32)
    else:
        std_data = np.load(standardization_path)
        mean = std_data["mean"].astype(np.float32)
        std = std_data["std"].astype(np.float32)
    std = np.where(std > 1e-30, std, 1.0).astype(np.float32)
    assert len(mean) == weights[0].shape[1]
    assert len(std) == weights[0].shape[1]

    # Feature-subset checkpoints (ablations) are padded back to the width
    # the C++ builds. A checkpoint without feature_names is taken as full.
    full_names = FULL_FEATURES[model_type]
    feature_names = ckpt.get("feature_names") if isinstance(ckpt, dict) else None
    if feature_names is not None and list(feature_names) != list(full_names):
        weights[0], mean, std = pad_to_full(weights[0], mean, std, feature_names, full_names)
        print(f"padded {len(feature_names)} trained features to the full {len(full_names)}; "
              f"zero-weight columns: {sorted(set(full_names) - set(feature_names))}")
    n_features = weights[0].shape[1]

    if model_type == "gate":
        assert n_features == 26, f"Gate expects 26 features, got {n_features}"
        assert n_hidden == 128, f"Gate expects 128 hidden, got {n_hidden}"
        assert n_layers == 3, f"Gate expects 3 hidden layers, got {n_layers}"
    elif model_type == "value":
        # 11 = Tier-2 VALUE_FEATURES; 12 = window-conditioned Tier-3
        # VALUE_FEATURES_WINDOWED (window-conditioned tier-3 value plan,
        # Task 7). The blob header carries input_dim, so no format change is
        # needed for the extra feature -- only this width check relaxes.
        # The C++ branch stopper builds exactly 11 features (no windowed
        # path); a 12-feature windowed checkpoint must drop window_nsigma
        # (pad_to_full refuses it) or the C++ must grow first.
        assert n_features == 11, (
            f"Value expects 11 features, got {n_features}"
        )
        assert n_hidden == 128, f"Value expects 128 hidden, got {n_hidden}"
        assert n_layers == 2, f"Value expects 2 hidden layers, got {n_layers}"

    # Load calibration (Platt params). "identity" writes a no-op calibrator
    # (a=1, b=0) for models exported without a Platt fit, e.g. the value
    # function or an already-calibrated raw gate. Accepts either a flat
    # {a0,a1,b0,b1} dict or calibrate_and_audit.py's {"four_param": {...}}.
    if calibration_path == "identity":
        platt_a0, platt_a1, platt_b0, platt_b1 = 1.0, 0.0, 0.0, 0.0
    else:
        with open(calibration_path) as f:
            cal = json.load(f)
        if "four_param" in cal:
            cal = cal["four_param"]
        platt_a0 = float(cal["a0"])
        platt_a1 = float(cal["a1"])
        platt_b0 = float(cal["b0"])
        platt_b1 = float(cal["b1"])

    # Write blob
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as f:
        f.write(b"CCKF")
        f.write(struct.pack("<I", 1))  # version
        f.write(struct.pack("<I", n_features))
        f.write(struct.pack("<I", n_hidden))
        f.write(struct.pack("<I", n_layers))
        f.write(mean.tobytes())
        f.write(std.tobytes())
        f.write(struct.pack("<ffff", platt_a0, platt_a1, platt_b0, platt_b1))
        for w, b in zip(weights, biases):
            f.write(w.astype(np.float32).tobytes())
            f.write(b.astype(np.float32).tobytes())

    total_params = sum(w.size + b.size for w, b in zip(weights, biases))
    print(f"Exported {model_type}: {n_features}→{n_hidden}×{n_layers}→1, "
          f"{total_params} params, {out.stat().st_size} bytes → {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--standardization", required=True)
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-type", choices=["gate", "value"],
                        required=True)
    args = parser.parse_args()
    export(args.checkpoint, args.standardization, args.calibration,
           args.output, args.model_type)
