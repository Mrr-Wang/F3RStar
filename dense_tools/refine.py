"""Run the fixed DenseStar correction; retired research variants are archived."""
from __future__ import annotations

from pathlib import Path
import time

import numpy as np

from gluemap.dense.edge_correct import edge_correct
from .common import atomic_json, sample


def anchor_error(depth, anchors, train=False):
    a = anchors[(anchors[:, 5] > 0) == train]
    if not len(a):
        return None
    d = sample(depth, a[:, :2])
    ok = np.isfinite(d) & (d > 0)
    if not ok.any():
        return None
    error = np.abs(d[ok] / a[ok, 2] - 1)
    return {"n": int(ok.sum()), "median_abs_rel": float(np.median(error)),
            "mean_abs_rel": float(error.mean())}


def apply_variant(variant, maps, views, task: Path):
    started = time.monotonic()
    stats = {}
    if variant != "edge_anchor_direct":
        raise ValueError(f"Unsupported refinement {variant!r}; see docs/current-pipeline.md for the fixed plan")
    for idx, packed in maps.items():
        packed["depth"], stats[str(idx)] = edge_correct(
            packed["depth"], packed["rgb"], views[idx].anchors)
    statistics = {"algorithm_seconds": time.monotonic()-started, "details": stats, "uses_gt": False}
    statistics["holdout_BA_depth"] = {str(idx): anchor_error(p["depth"], views[idx].anchors)
                                      for idx, p in maps.items()}
    # Retain the actual refined depth maps to permit independent replay.
    directory = task / "refined_depth"
    directory.mkdir(parents=True, exist_ok=True)
    for idx, p in maps.items():
        np.savez_compressed(directory / f"image_{idx:05d}.npz", depth=p["depth"], conf=p["conf"],
                            mask=p.get("mask", np.ones_like(p["depth"], dtype=bool)),
                            K=views[idx].K, R=views[idx].R, C=views[idx].C)
    atomic_json(task / "algorithm.json", statistics)
    return maps, statistics
