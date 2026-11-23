from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import sys
import time
import uuid

import cv2
import numpy as np
import pycolmap
import torch
import yaml
from scipy.ndimage import map_coordinates

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gluemap.dense.anchors import SparseDepthAnchor
from gluemap.dense.cache import load_star_dense
from gluemap.dense.calibration import fit_view_star_scale
from gluemap.dense.fusion import fuse_depth_hypotheses, fuse_depth_hypotheses_maxconf
from gluemap.dense.ray_transport import extract_ba_cameras, map_intrinsics_to_dense

# Default output / source dirs are placeholders; pass --source / --run from CLI.
RUN = ROOT / "results" / "dense"
SOURCE = ROOT / "results"


def atomic_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{os.getpid()}.{uuid.uuid4().hex}.tmp")
    temp.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temp.replace(path)


def sample(arr: np.ndarray, xy: np.ndarray) -> np.ndarray:
    return map_coordinates(arr, [xy[:, 1], xy[:, 0]], order=1,
                           mode="constant", cval=np.nan, prefilter=False)


@dataclass
class View:
    idx: int
    name: str
    image_path: Path
    change: np.ndarray
    K: np.ndarray
    R: np.ndarray
    C: np.ndarray
    anchors: np.ndarray  # u_net, v_net, depth, weight, pid, train


def source_dir(scene: str, method: str = "infostar_q4") -> Path:
    return SOURCE / scene / method / ("infostar" if method.startswith("infostar") else method)


def load_scene(scene: str, method: str = "infostar_q4", *, source: Path | None = None):
    source = Path(source) if source is not None else source_dir(scene, method)
    config = yaml.safe_load((source / "resolved_config.yaml").read_text())
    image_root = Path(config["images_path"])
    paths = sorted(p for p in image_root.rglob("*")
                   if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"})
    name_to_idx = {p.name: i for i, p in enumerate(paths)}
    if len(name_to_idx) != len(paths):
        raise ValueError("Image basenames must be unique")
    reconstruction = pycolmap.Reconstruction(str(source / "gluemap_aba"))
    cameras = extract_ba_cameras(reconstruction, name_to_idx)
    manifest = json.loads((source / "dense_cache/manifest.json").read_text())
    changes = {}
    for entry in manifest["stars"]:
        with np.load(source / "dense_cache" / entry["path"]) as packed:
            for pos, idx in enumerate(packed["indexes"]):
                change = packed["images_change"][pos].astype(np.float64)
                if int(idx) in changes and not np.allclose(change, changes[int(idx)]):
                    raise ValueError("Inconsistent image transforms between stars")
                changes[int(idx)] = change
    # Stable holdout by point ID, shared across all observations of a track.
    weights = {}
    for pid, point in reconstruction.points3D.items():
        elements = point.track.elements
        n = len(elements)
        xyz = np.asarray(point.xyz)
        rays = []
        for observation in elements:
            im = reconstruction.images[observation.image_id]
            c = np.asarray(im.projection_center())
            ray = xyz - c
            if np.linalg.norm(ray) > 1e-8:
                rays.append(ray / np.linalg.norm(ray))
        if len(rays) >= 2:
            cosine = np.clip(np.asarray(rays) @ np.asarray(rays).T, -1, 1)
            angle = float(np.rad2deg(np.arccos(cosine.min())))
        else:
            angle = 0.0
        error = max(float(point.error), 0.0)
        weight = min(n / 4.0, 1.0) * min(angle / 5.0, 1.0) / (1.0 + error**2)
        if n < 3 or angle < 1.0 or error > 3.0:
            weight = 0.0
        weights[int(pid)] = weight
    views = {}
    for im in reconstruction.images.values():
        idx = name_to_idx[Path(im.name).name]
        camera = cameras[idx]
        change = changes[idx]
        anchors = []
        for observation in im.points2D:
            if not observation.has_point3D():
                continue
            pid = int(observation.point3D_id)
            if pid not in reconstruction.points3D:
                continue
            point = reconstruction.points3D[pid]
            z = float((camera.R @ (np.asarray(point.xyz) - camera.C))[2])
            if z <= 1e-4:
                continue
            u, v = np.asarray(observation.xy) * change[:2] + change[2:]
            train = int((pid * 2654435761) % 5 != 0)
            anchors.append([u, v, z, weights[pid], pid, train])
        views[idx] = View(idx, str(im.name), paths[idx], change,
                          map_intrinsics_to_dense(camera.K, change), camera.R,
                          camera.C, np.asarray(anchors, dtype=np.float64).reshape(-1, 6))
    return source, reconstruction, views, manifest


def read_rgb(view: View, shape: tuple[int, int]) -> np.ndarray:
    image = cv2.imread(str(view.image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(view.image_path)
    sx, sy, ox, oy = view.change
    transform = np.array([[sx, 0, ox], [0, sy, oy]], dtype=np.float64)
    return cv2.warpAffine(image, transform, (shape[1], shape[0]),
                          flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def initialize_scene(scene: str, method: str = "infostar_q4", run: Path = RUN,
                     *, source: Path | None = None) -> Path:
    output = run / "prepared" / scene
    if (output / "complete.json").exists():
        return output
    started = time.monotonic()
    source, reconstruction, views, manifest = load_scene(scene, method, source=source)
    output.mkdir(parents=True, exist_ok=True)
    scales = torch.load(source / "global_scales.pth", weights_only=False, map_location="cpu")
    per_image = {idx: [] for idx in views}
    alpha_info = []
    for entry in manifest["stars"]:
        star = int(entry["star_idx"])
        packed = load_star_dense(str(source / "dense_cache"), star)
        for pos, idx in enumerate(packed["indexes"]):
            idx = int(idx)
            if idx not in views:
                continue
            view = views[idx]
            aa = view.anchors[view.anchors[:, 5] > 0]
            sx, sy, ox, oy = view.change
            anchors = [SparseDepthAnchor((a[0]-ox)/sx, (a[1]-oy)/sy, a[2], int(a[4])) for a in aa]
            fit = fit_view_star_scale(packed["depth"][pos], packed["conf"][pos],
                                     view.change, anchors, float(scales[star]))
            depth = packed["depth"][pos].astype(np.float32) * (fit.alpha / float(scales[star]))
            confidence = packed["conf"][pos].astype(np.float32) / 255.0
            per_image[idx].append((depth, confidence))
            alpha_info.append({"star": star, "image": idx, "alpha": fit.alpha,
                               "ok": bool(fit.ok), "n_inliers": int(fit.n_inliers)})
    for idx, hypotheses in per_image.items():
        if not hypotheses:
            raise ValueError(f"No dense hypotheses for registered image {idx}")
        depths = [x[0] for x in hypotheses]
        confidences = [x[1] for x in hypotheses]
        median = fuse_depth_hypotheses(depths, confidences)
        maxconf = fuse_depth_hypotheses_maxconf(depths, confidences)
        np.savez_compressed(output / f"image_{idx:05d}.npz", depth=median.depth,
                            maxconf=maxconf.depth, conf=median.conf, mad=median.mad,
                            n=median.n_hypotheses,
                            rgb=read_rgb(views[idx], median.depth.shape))
        print(f"prepared {scene} image {idx+1}/{len(views)}", flush=True)
    atomic_json(output / "complete.json", {"scene": scene, "source": str(source),
                "method": method, "n_views": len(views), "n_stars": len(manifest["stars"]),
                "calibration": "80 percent BA track IDs; GT excluded",
                "alpha_info": alpha_info, "seconds": time.monotonic()-started})
    return output


def rays(view: View, shape: tuple[int, int]) -> np.ndarray:
    yy, xx = np.indices(shape, dtype=np.float32)
    return np.stack([(xx-view.K[0, 2])/view.K[0, 0],
                     (yy-view.K[1, 2])/view.K[1, 1], np.ones(shape)], axis=-1)


def world_points(depth: np.ndarray, view: View) -> np.ndarray:
    return (rays(view, depth.shape) * depth[..., None]) @ view.R + view.C


def write_raw_ply(path: Path, maps: dict[int, dict], views: dict[int, View]) -> int:
    # Reconstruction export uses no GT and does not voxelize in unknown units.
    path.parent.mkdir(parents=True, exist_ok=True)
    counts = []
    for idx, packed in maps.items():
        depth = packed["depth"]
        mask = np.isfinite(depth) & (depth > 0) & (packed["conf"] >= 0.1)
        if "mask" in packed:
            mask &= packed["mask"]
        if "mad" in packed:
            mask &= (packed.get("n", np.ones_like(depth)) < 2) | (packed["mad"] / np.maximum(depth, 1e-8) <= 0.15)
        counts.append((idx, mask, int(mask.sum())))
    total = sum(c for _, _, c in counts)
    if total == 0:
        raise ValueError("Empty reconstructed cloud")
    with path.open("wb") as stream:
        stream.write(("ply\nformat binary_little_endian 1.0\n" +
                      f"element vertex {total}\nproperty float x\nproperty float y\nproperty float z\nend_header\n").encode())
        for idx, mask, _ in counts:
            xyz = world_points(maps[idx]["depth"], views[idx])[mask]
            stream.write(xyz.astype("<f4").tobytes())
    return total
