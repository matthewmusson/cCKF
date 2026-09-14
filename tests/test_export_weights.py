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


@pytest.mark.parametrize("n_features", [11])
def test_value_export_accepts_the_cpp_width(tmp_path, n_features):
    """The C++ branch stopper builds exactly 11 value features
    (CckfBranchStopper.hpp declares ``float features[11]``), so 11 is the
    only deployable width. The windowed 12-feature checkpoint is refused by
    the exporter until the C++ grows a windowed path (see
    test_windowed_value_checkpoint_refuses_export)."""
    blob_path = _export_value_model(str(tmp_path), n_features)

    with open(blob_path, "rb") as f:
        assert f.read(4) == b"CCKF"
        version, n_feat, n_hid, n_layers = struct.unpack("<IIII", f.read(16))
        assert version == 1
        assert n_feat == n_features
        assert n_hid == 128
        assert n_layers == 2


def test_value_export_rejects_unexpected_width(tmp_path):
    """A value checkpoint whose width is not 11 (and carries no
    feature_names to pad from) is a real error and must fail loudly."""
    with pytest.raises(AssertionError):
        _export_value_model(str(tmp_path), 13)
    with pytest.raises(AssertionError):
        _export_value_model(str(tmp_path), 12)


if __name__ == "__main__":
    test_roundtrip()


def _read_blob(path):
    """Parse a CCKF blob back into (n_features, mean, std, layers)."""
    with open(path, "rb") as f:
        assert f.read(4) == b"CCKF"
        _, n_feat, n_hid, n_layers = struct.unpack("<IIII", f.read(16))
        mean = np.frombuffer(f.read(4 * n_feat), dtype=np.float32)
        std = np.frombuffer(f.read(4 * n_feat), dtype=np.float32)
        f.read(16)  # platt
        layers = []
        in_dim = n_feat
        for i in range(n_layers + 1):
            out_dim = n_hid if i < n_layers else 1
            w = np.frombuffer(f.read(4 * out_dim * in_dim), dtype=np.float32).reshape(out_dim, in_dim)
            b = np.frombuffer(f.read(4 * out_dim), dtype=np.float32)
            layers.append((w, b))
            in_dim = out_dim
    return n_feat, mean, std, layers


def _forward_np(x, mean, std, layers):
    h = (x - mean) / np.where(std > 1e-30, std, 1.0)
    for i, (w, b) in enumerate(layers):
        h = h @ w.T + b
        if i < len(layers) - 1:
            h = h / (1.0 + np.exp(-h))  # SiLU
    return h


def test_feature_subset_checkpoint_is_padded_to_full_width():
    """A gate trained with --drop-features exports at the full 26 width and
    the padded blob's forward pass equals the subset model's on the kept
    columns, whatever the dropped columns contain."""
    from cckf import features
    torch.manual_seed(7)
    dropped = {"n_holes", "clus_sigma_uv", "residual_l0"}
    keep = [f for f in features.GATE_FEATURES if f not in dropped]
    model = GateMLP(n_features=len(keep), width=128, depth=3).eval()
    mu = np.random.default_rng(1).normal(size=len(keep)).astype(np.float32)
    sigma = (np.random.default_rng(2).uniform(0.5, 2.0, size=len(keep))).astype(np.float32)

    with tempfile.TemporaryDirectory() as tmp:
        ckpt_path = Path(tmp) / "gate_model.pt"
        torch.save({"state_dict": model.state_dict(), "feature_names": keep,
                    "all_feature_names": list(features.GATE_FEATURES),
                    "mu": mu, "sigma": sigma}, ckpt_path)
        cal_path = Path(tmp) / "cal.json"
        cal_path.write_text(json.dumps({"a0": 1.0, "a1": 0.0, "b0": 0.0, "b1": 0.0}))
        blob = Path(tmp) / "gate.bin"
        export(str(ckpt_path), "ckpt", str(cal_path), str(blob), "gate")

        n_feat, mean, std, layers = _read_blob(blob)
        assert n_feat == 26
        full_idx = [features.GATE_FEATURES.index(n) for n in keep]
        drop_idx = [features.GATE_FEATURES.index(n) for n in dropped]
        assert np.all(layers[0][0][:, drop_idx] == 0)
        assert np.all(mean[drop_idx] == 0) and np.all(std[drop_idx] == 1)

        x_full = np.random.default_rng(3).normal(size=(5, 26)).astype(np.float32)
        x_full[:, drop_idx] = 1e6  # garbage in the dropped columns must not matter
        ref = model(torch.from_numpy((x_full[:, full_idx] - mu) / sigma)).detach().numpy()
        got = _forward_np(x_full, mean, std, layers)[:, 0]
        np.testing.assert_allclose(got, ref, rtol=1e-4, atol=1e-4)


def test_windowed_value_checkpoint_refuses_export():
    from cckf import features
    torch.manual_seed(8)
    names = list(features.VALUE_FEATURES_WINDOWED)
    model = ValueMLP(n_features=12, width=128, depth=2)
    with tempfile.TemporaryDirectory() as tmp:
        ckpt_path = Path(tmp) / "value_model.pt"
        torch.save({"state_dict": model.state_dict(), "feature_names": names,
                    "mu": np.zeros(12, np.float32), "sigma": np.ones(12, np.float32)}, ckpt_path)
        with pytest.raises(ValueError, match="window_nsigma"):
            export(str(ckpt_path), "ckpt", "identity", str(Path(tmp) / "v.bin"), "value")
