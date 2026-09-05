"""Spot-check monotonicity of the window-conditioned value function V_φ.

Window-conditioned tier-3 value plan, Task 7 acceptance check: mean predicted
V at fixed state features should be non-increasing as the rollout window
shrinks. This samples ``--n`` random states from a cal cache, scores each
twice with only the 12th feature (``window_nsigma``) swapped between a wide
window (default 10σ) and a narrow one (default 3σ), and reports the fraction
where ``V(3) <= V(10)`` and the mean difference ``V(10) - V(3)``.

Model loading and standardisation mirror ``scripts/train_value.py`` and
``scripts/eval_value_cal.py`` exactly: the checkpoint's own ``mu``/``sigma``
are applied via ``(x - mu) / sigma``, and ``window_nsigma`` passes through
unstandardised (``mu=0``, ``sigma=1`` at that column, per
``cckf.features.NO_STANDARDIZE`` -- see :func:`score_at_window`).

Usage
-----
    python scripts/value_window_monotonicity.py MODEL CACHE_DIR --n 1000
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from cckf import models, train
from scripts.train_value import load_value_cache


def score_at_window(
    model: torch.nn.Module,
    X: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    window_idx: int,
    window_value: float,
    device: str = "cpu",
) -> np.ndarray:
    """Score ``X`` with column ``window_idx`` overridden to a constant value.

    The override happens on the *raw* feature matrix, before standardising
    with ``(x - mu) / sigma`` -- the same order ``train_value.py``'s
    ``StandardizedView`` and ``eval_value_cal.py`` use. Since
    ``window_nsigma`` is exempt from standardisation (``mu=0``, ``sigma=1``
    at its column by convention -- :data:`cckf.features.NO_STANDARDIZE`),
    the overridden raw value reaches the model unchanged.

    Returns
    -------
    numpy.ndarray
        Predicted ``V`` (sigmoid of the raw logit), shape ``(len(X),)``.
    """
    X = np.array(X, dtype=np.float32, copy=True)
    X[:, window_idx] = window_value
    Xz = train.standardize(X, mu, sigma)
    logits = train.predict_logits(model, Xz, device=device)
    return 1.0 / (1.0 + np.exp(-logits))


def monotonicity_stats(
    model: torch.nn.Module,
    X: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    window_idx: int,
    big: float = 10.0,
    small: float = 3.0,
    device: str = "cpu",
) -> dict:
    """Compare ``V`` at a wide window (``big``) vs. a narrow one (``small``).

    Returns
    -------
    dict
        ``frac_non_increasing``: fraction of sampled states with
        ``V(small) <= V(big)`` -- the plan's acceptance direction (V should
        not increase as the rollout window shrinks). ``mean_diff``:
        ``mean(V(big) - V(small))``. ``v_big_mean``/``v_small_mean``: the two
        means on their own, for context.
    """
    v_big = score_at_window(model, X, mu, sigma, window_idx, big, device)
    v_small = score_at_window(model, X, mu, sigma, window_idx, small, device)
    return {
        "frac_non_increasing": float(np.mean(v_small <= v_big)),
        "mean_diff": float(np.mean(v_big - v_small)),
        "v_big_mean": float(v_big.mean()),
        "v_small_mean": float(v_small.mean()),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("model", help="Path to a value_model.pt checkpoint.")
    ap.add_argument(
        "cache_dir",
        help=(
            "A value cache directory: either a flat 12-feature (windowed) "
            "cache, or a parent directory of nsig*/ subdirs (concatenated)."
        ),
    )
    ap.add_argument("--n", type=int, default=1000, help="States to sample.")
    ap.add_argument("--big", type=float, default=10.0, help="Wide window (sigma).")
    ap.add_argument("--small", type=float, default=3.0, help="Narrow window (sigma).")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    ckpt = torch.load(args.model, map_location="cpu", weights_only=False)
    model = models.ValueMLP(
        n_features=ckpt["n_features"], width=ckpt["width"], depth=ckpt["depth"]
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    feature_names = list(ckpt["feature_names"])
    if "window_nsigma" not in feature_names:
        raise ValueError(
            f"{args.model} was not trained on the windowed (12-feature) "
            "feature set -- 'window_nsigma' not in its feature_names "
            f"({feature_names})"
        )
    window_idx = feature_names.index("window_nsigma")

    cache = load_value_cache(args.cache_dir)
    if cache["meta"]["feature_names"] != feature_names:
        raise ValueError(
            f"cache feature_names {cache['meta']['feature_names']} != "
            f"model feature_names {feature_names}"
        )

    n_rows = cache["meta"]["n_rows"]
    n_sample = min(args.n, n_rows)
    rng = np.random.default_rng(args.seed)
    idx = np.sort(rng.choice(n_rows, size=n_sample, replace=False))
    X = np.asarray(cache["X"])[idx].astype(np.float32)

    mu = np.asarray(ckpt["mu"], dtype=np.float32)
    sigma = np.asarray(ckpt["sigma"], dtype=np.float32)
    sigma = np.where(sigma > 1e-30, sigma, 1.0).astype(np.float32)

    stats = monotonicity_stats(
        model, X, mu, sigma, window_idx, args.big, args.small, args.device
    )
    stats.update(
        {
            "n_sampled": int(n_sample),
            "big": args.big,
            "small": args.small,
            "model": str(args.model),
            "cache_dir": str(args.cache_dir),
        }
    )
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
