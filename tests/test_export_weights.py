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
import pytest
import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cckf.models import GateMLP, ValueMLP
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


def _export_value_model(tmp: str, n_features: int) -> Path:
    """Export a tiny ValueMLP with the given input width, model_type='value'."""
    torch.manual_seed(0)
    model = ValueMLP(n_features=n_features, width=128, depth=2)
    model.eval()

    ckpt_path = Path(tmp) / "value_model.pt"
    torch.save(model.state_dict(), ckpt_path)

    std_path = Path(tmp) / "std.npz"
    np.savez(
        std_path,
        mean=np.zeros(n_features, dtype=np.float32),
        std=np.ones(n_features, dtype=np.float32),
    )

    blob_path = Path(tmp) / "value.bin"
    export(str(ckpt_path), str(std_path), "identity", str(blob_path), "value")
    return blob_path


@pytest.mark.parametrize("n_features", [11, 12])
def test_value_export_accepts_tier2_and_windowed_tier3_widths(tmp_path, n_features):
    """Window-conditioned tier-3 value plan, Task 7: the exporter's
    ``n_features == 11`` assert must relax to accept the windowed 12-feature
    value function too, since the blob header already carries ``input_dim``
    and needs no format change."""
    blob_path = _export_value_model(str(tmp_path), n_features)

    with open(blob_path, "rb") as f:
        assert f.read(4) == b"CCKF"
        version, n_feat, n_hid, n_layers = struct.unpack("<IIII", f.read(16))
        assert version == 1
        assert n_feat == n_features
        assert n_hid == 128
        assert n_layers == 2


def test_value_export_rejects_unexpected_width(tmp_path):
    """A value checkpoint with neither 11 nor 12 inputs is a real error
    (wrong feature vector, not a new deliberate width) and must still fail
    loudly rather than being silently accepted."""
    with pytest.raises(AssertionError):
        _export_value_model(str(tmp_path), 13)


if __name__ == "__main__":
    test_roundtrip()
