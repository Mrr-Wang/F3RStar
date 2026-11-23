from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import time
import traceback

import numpy as np

from .common import ROOT, atomic_json, initialize_scene, load_scene, source_dir, write_raw_ply
from .evaluate import evaluate

DEVELOPMENT = ["scene"]  # override with --scenes
VARIANTS = ("edge_anchor_direct", "baseline", "maxconf")
ACTIVE_RUN = ROOT / "results" / "dense"


def load_maps(directory: Path):
    return {int(p.stem.split("_")[1]): dict(np.load(p))
            for p in sorted(directory.glob("image_*.npz"))}


def run_task(scene: str, variant: str, run: Path = ACTIVE_RUN, score: bool = True,
             method: str = "infostar_q4", source: Path | None = None) -> dict:
    if variant not in VARIANTS:
        raise ValueError(f"Unsupported variant {variant!r}; supported: {VARIANTS}")
    if shutil.disk_usage(ROOT).free < 5 * 2**30:
        raise RuntimeError("Disk guard: free space below 5 GiB")
    task = run / "tasks" / scene / variant
    result = task / "result.json"
    source_path = Path(source) if source is not None else source_dir(scene, method)
    if result.exists():
        previous = json.loads(result.read_text())
        if score and previous["scores"] is None:
            previous["scores"] = evaluate(task / "raw_prediction.ply", source_path / "gluemap_aba",
                                           scene, task / "evaluation")
            atomic_json(result, previous)
        return previous
    task.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    atomic_json(task / "status.json", {"status": "running", "pid": os.getpid(),
                "scene": scene, "variant": variant, "started_utc": datetime.now(timezone.utc).isoformat()})
    try:
        prepared = initialize_scene(scene, method=method, run=run, source=source_path)
        source_loaded, rec, views, manifest = load_scene(scene, method=method, source=source_path)
        maps = load_maps(prepared)
        stats = {}
        if variant == "maxconf":
            for packed in maps.values():
                packed["depth"] = packed["maxconf"]
        elif variant != "baseline":
            from .refine import apply_variant
            maps, stats = apply_variant(variant, maps, views, task)
        raw_ply = task / "raw_prediction.ply"
        if not raw_ply.exists():
            n = write_raw_ply(raw_ply, maps, views)
        else:
            n = None
        refinement_seconds = time.monotonic()-started
        atomic_json(task / "refinement.json", {"seconds_including_load_and_export": refinement_seconds,
                    "n_raw_points": n, "stats": stats, "uses_gt": False,
                    "source": str(source_loaded)})
        metrics = evaluate(raw_ply, source_loaded / "gluemap_aba", scene, task / "evaluation") if score else None
        data = {"scene": scene, "variant": variant, "status": "SUCCESS",
                "method": method, "source": str(source_loaded),
                "refinement_seconds": refinement_seconds, "wall_seconds": time.monotonic()-started,
                "scores": metrics, "stats": stats}
        atomic_json(result, data)
        atomic_json(task / "status.json", {"status": "SUCCESS", "scene": scene,
                    "variant": variant, "wall_seconds": data["wall_seconds"]})
        print(json.dumps({"scene": scene, "variant": variant,
                          "F": metrics["metrics"]["fscore"] if metrics else None}), flush=True)
        return data
    except Exception as exc:
        status = "OOM" if "out of memory" in str(exc).lower() else "ERROR"
        atomic_json(task / "status.json", {"status": status, "scene": scene,
                    "variant": variant, "error": repr(exc), "traceback": traceback.format_exc()})
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["prepare", "run", "summarize"])
    parser.add_argument("--scenes", nargs="+", default=DEVELOPMENT)
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=["edge_anchor_direct"])
    parser.add_argument("--run-dir", type=Path, default=ACTIVE_RUN)
    parser.add_argument("--method", default="infostar_q4")
    parser.add_argument("--source", type=Path, default=None,
                        help="Override reconstruction source directory containing gluemap_aba/dense_cache")
    parser.add_argument("--no-score", action="store_true")
    args = parser.parse_args()
    if args.action == "prepare":
        for scene in args.scenes:
            initialize_scene(scene, method=args.method, run=args.run_dir, source=args.source)
    elif args.action == "run":
        failures = []
        for variant in args.variants:
            for scene in args.scenes:
                try:
                    run_task(scene, variant, args.run_dir, not args.no_score,
                             method=args.method, source=args.source)
                except Exception as exc:
                    failures.append({"scene": scene, "variant": variant, "error": repr(exc)})
                    print(traceback.format_exc(), flush=True)
        summarize(args.run_dir)
        if failures:
            atomic_json(args.run_dir / "last-batch-errors.json", {"failures": failures})
            raise SystemExit(1)
    else:
        summarize(args.run_dir)


def summarize(run: Path):
    import csv
    rows = []
    for file in sorted((run / "tasks").glob("*/*/result.json")):
        result = json.loads(file.read_text())
        if not result["scores"]:
            continue
        row = {k: result[k] for k in ("scene", "variant", "refinement_seconds", "wall_seconds")}
        review_file = file.parent / "numerical-review.json"
        row["selection_eligible"] = (json.loads(review_file.read_text()).get("use_for_selection", True)
                                      if review_file.exists() else True)
        for key, values in result["scores"]["metrics"].items():
            for tol, value in zip([1, 2, 5, 10], values):
                row[f"{key}@{tol}cm_percent"] = value * 100
        rows.append(row)
    if rows:
        with (run / "results.csv").open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        groups = {}
        for variant in sorted({r["variant"] for r in rows}):
            rr = [r for r in rows if r["variant"] == variant]
            groups[variant] = {"n_scenes": len(rr), "scenes": [r["scene"] for r in rr],
                               "selection_eligible": all(r["selection_eligible"] for r in rr),
                               "mean_F2_percent": float(np.mean([r["fscore@2cm_percent"] for r in rr]))}
        atomic_json(run / "summary.json", groups)
        print(json.dumps(groups, indent=2), flush=True)


if __name__ == "__main__":
    main()
