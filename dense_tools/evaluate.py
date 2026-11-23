"""Evaluation-only metric alignment and official ETH3D scoring.

GT is read only in this module, after raw reconstruction export. Scan poses
are supplied to the unmodified official evaluator through its MLP input.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import subprocess
import time

import numpy as np
import pycolmap

from .common import ROOT, atomic_json

PROTOCOL = "eth3d_official_camera_umeyama_v2"
# Optional: build the official ETH3D multi-view evaluator and point
# ETH3D_EVAL_BIN at the binary, or place it under thirdparty/eth3d-eval/.
BINARY = Path(os.environ.get(
    "ETH3D_EVAL_BIN",
    str(ROOT / "thirdparty" / "eth3d-eval" / "ETH3DMultiViewEvaluation"),
))
TOLERANCES = (0.01, 0.02, 0.05, 0.1)


def read_xyz(path: Path) -> np.ndarray:
    with path.open("rb") as stream:
        count = None
        props = []
        vertex = False
        fmt = ""
        while True:
            raw = stream.readline()
            if not raw:
                raise ValueError(f"Incomplete PLY header: {path}")
            line = raw.decode("ascii").strip()
            if line.startswith("format "):
                fmt = line.split()[1]
            elif line.startswith("element "):
                fields = line.split()
                vertex = fields[1] == "vertex"
                if vertex:
                    count = int(fields[2])
            elif vertex and line.startswith("property "):
                props.append(line.split()[1:])
            elif line == "end_header":
                offset = stream.tell()
                break
    if count is None or count <= 0:
        raise ValueError(f"Empty PLY: {path}")
    types = {"float": "f4", "float32": "f4", "double": "f8", "float64": "f8",
             "uchar": "u1", "uint8": "u1", "int": "i4", "uint": "u4"}
    if fmt not in {"binary_little_endian", "binary_big_endian"}:
        raise ValueError("Use binary PLY for versioned scoring")
    endian = "<" if "little" in fmt else ">"
    dtype = np.dtype([(name, endian+types[kind]) for kind, name in props])
    points = np.memmap(path, dtype=dtype, mode="r", offset=offset, shape=(count,))
    return np.column_stack([points[k] for k in ("x", "y", "z")]).astype(np.float64)


def camera_centers(path: Path) -> dict[str, np.ndarray]:
    rec = pycolmap.Reconstruction(str(path))
    result = {}
    for im in rec.images.values():
        name = Path(im.name).name
        if name in result:
            raise ValueError("Ambiguous duplicate camera basename")
        result[name] = np.asarray(im.projection_center(), dtype=np.float64)
    return result


def alignment(pred: Path, gt: Path) -> dict:
    predicted, target = camera_centers(pred), camera_centers(gt)
    names = sorted(set(predicted) & set(target))
    if len(names) < 3:
        raise ValueError("At least three shared cameras are required")
    a = np.stack([predicted[n] for n in names])
    b = np.stack([target[n] for n in names])
    am, bm = a.mean(0), b.mean(0)
    x, y = a-am, b-bm
    if np.linalg.matrix_rank(x, tol=1e-9) < 2:
        raise ValueError("Degenerate camera alignment; refusing an ambiguous fit")
    u, singular, vt = np.linalg.svd(y.T @ x / len(x))
    sign = np.ones(3)
    sign[-1] = np.sign(np.linalg.det(u @ vt))
    rotation = (u * sign) @ vt
    scale = float((singular * sign).sum() / (x*x).sum() * len(x))
    translation = bm - scale * (rotation @ am)
    error = np.linalg.norm(scale * (a @ rotation.T) + translation - b, axis=1)
    return {"n_cameras": len(names), "camera_names": names, "scale": scale,
            "rotation": rotation.tolist(), "translation": translation.tolist(),
            "mean_camera_error_m": float(error.mean()),
            "median_camera_error_m": float(np.median(error)),
            "p95_camera_error_m": float(np.quantile(error, 0.95))}


def voxel_means(points: np.ndarray, voxel: float) -> np.ndarray:
    keys = np.floor(points / voxel).astype(np.int64)
    contiguous = np.ascontiguousarray(keys)
    # Each coordinate triplet is one sortable record; no lossy float hashing.
    _, inverse, counts = np.unique(contiguous.view(np.dtype((np.void, 24))).ravel(),
                                    return_inverse=True, return_counts=True)
    means = np.column_stack([np.bincount(inverse, weights=points[:, d],
                                         minlength=len(counts)) / counts for d in range(3)])
    return means.astype(np.float32)


def write_xyz(path: Path, xyz: np.ndarray) -> None:
    with path.open("wb") as stream:
        stream.write(("ply\nformat binary_little_endian 1.0\n" +
                      f"element vertex {len(xyz)}\nproperty float x\nproperty float y\nproperty float z\nend_header\n").encode())
        stream.write(xyz.astype("<f4").tobytes())


def evaluate(raw_ply: Path, pred_cameras: Path, scene: str, output: Path,
             voxel: float = 0.005, threads: int = 8) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "scores.json"
    if result_path.exists():
        return json.loads(result_path.read_text())
    if not BINARY.is_file():
        raise FileNotFoundError(f"Official evaluator not built: {BINARY}")
    gt_base = ROOT / "datasets/ETH3D/mvs"
    gt_cameras = gt_base / "multi_view_training_dslr_undistorted" / scene / "reconstruction_gt"
    mlp = gt_base / "multi_view_training_dslr_scan_eval" / scene / "dslr_scan_eval/scan_alignment.mlp"
    if not mlp.is_file():
        raise FileNotFoundError(mlp)
    started = time.monotonic()
    transform = alignment(pred_cameras, gt_cameras)
    atomic_json(output / "alignment.json", transform)
    metric_cloud = output / "metric_5mm.ply"
    if not metric_cloud.exists():
        raw = read_xyz(raw_ply)
        valid = np.isfinite(raw).all(axis=1)
        if not valid.all():
            raise ValueError("Non-finite exported reconstruction points")
        xyz = transform["scale"] * (raw @ np.asarray(transform["rotation"]).T) + transform["translation"]
        xyz = voxel_means(xyz, voxel)
        write_xyz(metric_cloud, xyz)
        atomic_json(output / "cloud.json", {"raw_count": len(raw), "metric_voxel_count": len(xyz),
                                            "metric_voxel_m": voxel, "GT_used_only_in_evaluation": True})
        del raw, xyz
    command = [str(BINARY), "--reconstruction_ply_path", str(metric_cloud),
               "--ground_truth_mlp_path", str(mlp), "--tolerances",
               ",".join(str(x) for x in TOLERANCES), "--voxel_size", "0.01"]
    env = dict(os.environ, OMP_NUM_THREADS=str(threads), OPENBLAS_NUM_THREADS="1")
    with (output / "official.log").open("w") as stream:
        proc = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, env=env)
    log = (output / "official.log").read_text()
    if proc.returncode:
        raise RuntimeError(f"Official evaluator exit {proc.returncode}: {log[-1500:]}")
    values = {}
    for label, key in [("Accuracies", "accuracy"), ("Completenesses", "completeness"),
                       ("F1-scores", "fscore")]:
        match = re.search(rf"^{label}:\s*(.+)$", log, re.MULTILINE)
        if match is None:
            raise ValueError(f"Missing official result: {label}")
        arr = [float(x) for x in match.group(1).split()]
        if len(arr) != len(TOLERANCES) or not np.isfinite(arr).all():
            raise ValueError(f"Invalid official scores: {arr}")
        values[key] = arr
    result = {"protocol": PROTOCOL, "scene": scene, "tolerances_m": TOLERANCES,
              "metrics": values, "alignment": transform, "command": command,
              "evaluation_seconds": time.monotonic()-started}
    atomic_json(result_path, result)
    return result
