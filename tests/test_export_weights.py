"""Round-trip test for the weight exporter.

Builds a small ``GateMLP`` with fixed-seed weights, exports it through
``scripts/export_weights.py`` to a binary blob, and checks the blob header.
It also regenerates the fixture files the standalone C++ test
(``tests/test_mlp_inference.cpp``) reads: a weight blob, a raw input vector,
and the PyTorch-computed reference logit for that input. These live in
``tests/fixtures/`` (checked in) rather than a tempdir, since the C++ test
runs independently of pytest and needs them on disk.
"""
import json
import struct
import tempfile
from pathlib import Path

import numpy as np
import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cckf.models import GateMLP
from scripts.export_weights import export

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


def test_roundtrip():
    torch.manual_seed(42)
    model = GateMLP(n_features=26, width=128, depth=3)
    model.eval()

    with tempfile.TemporaryDirectory() as tmp:
        # Save checkpoint
        ckpt_path = Path(tmp) / "model.pt"
        torch.save(model.state_dict(), ckpt_path)

        # Save standardization (identity: mean=0, std=1)
        std_path = Path(tmp) / "std.npz"
        np.savez(std_path,
                 mean=np.zeros(26, dtype=np.float32),
                 std=np.ones(26, dtype=np.float32))

        # Save calibration (identity Platt: a=1, b=0)
        cal_path = Path(tmp) / "cal.json"
        with open(cal_path, "w") as f:
            json.dump({"a0": 1.0, "a1": 0.0, "b0": 0.0, "b1": 0.0}, f)

        # Export
        blob_path = Path(tmp) / "gate.bin"
        export(str(ckpt_path), str(std_path), str(cal_path),
               str(blob_path), "gate")

        # Verify header
        with open(blob_path, "rb") as f:
            assert f.read(4) == b"CCKF"
            version, n_feat, n_hid, n_layers = struct.unpack("<IIII",
                                                              f.read(16))
            assert version == 1
            assert n_feat == 26
            assert n_hid == 128
            assert n_layers == 3

        # Compute reference output on a fixed, reproducible input
        gen = torch.Generator().manual_seed(1234)
        x = torch.randn(1, 26, generator=gen)
        with torch.no_grad():
            ref_logit = model(x).item()

        print(f"Reference logit for test input: {ref_logit:.6f}")

        # Persist fixtures for the standalone C++ test.
        FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
        blob_bytes = blob_path.read_bytes()
        (FIXTURES_DIR / "gate_test.bin").write_bytes(blob_bytes)
        x.numpy().astype(np.float32).flatten().tofile(
            FIXTURES_DIR / "gate_test_input.bin")
        np.array([ref_logit], dtype=np.float32).tofile(
            FIXTURES_DIR / "gate_test_expected.bin")

        print("PASS")


if __name__ == "__main__":
    test_roundtrip()
