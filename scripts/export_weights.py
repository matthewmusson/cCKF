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

    n_features = weights[0].shape[1]
    n_hidden = weights[0].shape[0]
    n_layers = len(weights) - 1  # last layer is the head

    if model_type == "gate":
        assert n_features == 26, f"Gate expects 26 features, got {n_features}"
        assert n_hidden == 128, f"Gate expects 128 hidden, got {n_hidden}"
        assert n_layers == 3, f"Gate expects 3 hidden layers, got {n_layers}"
    elif model_type == "value":
        # 11 = Tier-2 VALUE_FEATURES; 12 = window-conditioned Tier-3
        # VALUE_FEATURES_WINDOWED (window-conditioned tier-3 value plan,
        # Task 7). The blob header carries input_dim, so no format change is
        # needed for the extra feature -- only this width check relaxes.
        assert n_features in (11, 12), (
            f"Value expects 11 (Tier 2) or 12 (windowed Tier 3) features, "
            f"got {n_features}"
        )
        assert n_hidden == 128, f"Value expects 128 hidden, got {n_hidden}"
        assert n_layers == 2, f"Value expects 2 hidden layers, got {n_layers}"

    # Load standardization. "ckpt" pulls mu/sigma from the checkpoint itself,
    # which is where the v3 training scripts store them.
    if standardization_path == "ckpt":
        mean = np.asarray(ckpt["mu"], dtype=np.float32)
        std = np.asarray(ckpt["sigma"], dtype=np.float32)
    else:
        std_data = np.load(standardization_path)
        mean = std_data["mean"].astype(np.float32)
        std = std_data["std"].astype(np.float32)
    std = np.where(std > 1e-30, std, 1.0).astype(np.float32)
    assert len(mean) == n_features
    assert len(std) == n_features

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
