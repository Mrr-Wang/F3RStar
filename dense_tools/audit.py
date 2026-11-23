"""Audit completed saved experiments without rerunning reconstruction or scoring."""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
from zoneinfo import ZoneInfo

import numpy as np

from .common import ROOT, RUN, atomic_json
from .report import ORDER
from .run import DEVELOPMENT


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=RUN)
    parser.add_argument("--output-dir", type=Path,
                        help="Write a new audit here without replacing historical audit records")
    args = parser.parse_args()
    run = args.run_dir
    output_dir = args.output_dir or run
    scenes = [x["scene"] for x in json.loads((run/"evaluation-set.json").read_text())["all_13"]]
    selected = json.loads((run/"extended-selection.json").read_text())["selected"]
    expected = {(s, v) for s in DEVELOPMENT for v in ORDER} | {(s, v) for s in scenes for v in selected}
    results = {}
    errors = []
    residuals = []
    cg_excluded = []
    for scene, variant in sorted(expected):
        task = run/"tasks"/scene/variant
        path = task/"result.json"
        if not path.exists():
            errors.append(f"Missing result: {scene}/{variant}")
            continue
        result = json.loads(path.read_text())
        status = json.loads((task/"status.json").read_text())
        if result["status"] != "SUCCESS" or status["status"] != "SUCCESS":
            errors.append(f"Unresolved task: {scene}/{variant}")
        scores = json.loads((task/"evaluation/scores.json").read_text())
        if scores != result["scores"]:
            errors.append(f"Result differs from official score file: {scene}/{variant}")
        if scores["protocol"] != "eth3d_official_camera_umeyama_v2" or scores["tolerances_m"] != [.01, .02, .05, .1]:
            errors.append(f"Unexpected evaluation protocol: {scene}/{variant}")
        command = scores["command"]
        mlp = command[command.index("--ground_truth_mlp_path")+1]
        if not mlp.endswith(f"/{scene}/dslr_scan_eval/scan_alignment.mlp") or command[command.index("--voxel_size")+1] != "0.01":
            errors.append(f"Unexpected official evaluator arguments: {scene}/{variant}")
        log = (task/"evaluation/official.log").read_text()
        for label, key in [("Accuracies", "accuracy"), ("Completenesses", "completeness"), ("F1-scores", "fscore")]:
            match = re.search(rf"^{label}:\s*(.+)$", log, re.MULTILINE)
            numbers = [float(x) for x in match.group(1).split()] if match else []
            if numbers != scores["metrics"][key] or len(numbers) != 4 or not all(np.isfinite(numbers)) or not all(0 <= x <= 1 for x in numbers):
                errors.append(f"Invalid official log metrics: {scene}/{variant}/{key}")
        accuracy, completeness, fscore = [np.array(scores["metrics"][k]) for k in ["accuracy", "completeness", "fscore"]]
        harmonic = 2*accuracy*completeness/np.maximum(accuracy+completeness, 1e-30)
        if not np.allclose(harmonic, fscore, atol=2e-6, rtol=0):
            errors.append(f"F-score not consistent with A/C: {scene}/{variant}")
        cloud = json.loads((task/"evaluation/cloud.json").read_text())
        if cloud["metric_voxel_m"] != .005 or not cloud["GT_used_only_in_evaluation"]:
            errors.append(f"Unexpected point sampling protocol: {scene}/{variant}")
        refinement = json.loads((task/"refinement.json").read_text())
        if refinement["uses_gt"]:
            errors.append(f"Reconstruction declares GT usage: {scene}/{variant}")
        if variant == "edge_anchor_direct":
            for image, detail in result["stats"]["details"].items():
                if detail.get("applied"):
                    value = detail["relative_linear_residual"]
                    residuals.append(value)
                    if not np.isfinite(value) or value >= 1e-7:
                        errors.append(f"Invalid direct solve: {scene}/{image}")
        if variant == "edge_anchor":
            review = json.loads((task/"numerical-review.json").read_text())
            if review["use_for_selection"]:
                errors.append(f"Unconverged CG result included in selection: {scene}")
            cg_excluded.append(scene)
        results[(scene, variant)] = result

    input_files = 0
    verified_early = []
    manifests = {}
    for scene in scenes:
        cache = run/"prepared"/scene
        initial = cache/"input-sha256.json"
        frozen = json.loads(initial.read_text()) if initial.exists() else None
        source = Path(json.loads((cache/"complete.json").read_text())["source"])
        paths = sorted(set([source/"global_scales.pth", source/"resolved_config.yaml", source/"dense_cache/manifest.json"]
                           + list((source/"gluemap_aba").glob("*.bin")) + list((source/"dense_cache").glob("*.npz"))))
        current = {str(p): {"size": p.stat().st_size, "sha256": sha256(p)} for p in paths}
        if frozen is not None:
            for name, entry in frozen.items():
                if current.get(name) != entry:
                    errors.append(f"Source differs from earlier manifest: {name}")
            verified_early.append(scene)
        manifests[scene] = {"files": current, "compared_with_earlier_manifest": frozen is not None}
        input_files += len(current)
    atomic_json(output_dir/"final-input-sha256.json", manifests)

    plan = json.loads((run/"plan.json").read_text())
    official_sources = {k: v for k, v in plan["file_hashes"].items() if k.startswith("thirdparty/")}
    for name, frozen in official_sources.items():
        if sha256(ROOT/name) != frozen:
            errors.append(f"Official evaluator source modified: {name}")
    environment = json.loads((run/"environment-and-timing-scope.json").read_text())
    if sha256(ROOT/".setup/dense-eval-build/build/ETH3DMultiViewEvaluation") != environment["official_binary_sha256"]:
        errors.append("Official evaluator binary changed")
    validation_hashes = json.loads((run/"validation-code-sha256.json").read_text())
    # Audit the immutable historical snapshot, independently of subsequent
    # maintenance of the active package. Never replace the frozen hashes.
    for name in ["common.py", "refine.py", "run.py", "evaluate.py"]:
        candidates = [(k, v) for k, v in validation_hashes.items() if Path(k).name == name]
        if len(candidates) != 1 or sha256(run/"code-snapshot-validation"/name) != candidates[0][1]:
            errors.append(f"Frozen validation snapshot changed: {name}")
    tests = (run/"final-tests.log").read_text()
    if not re.search(r"6 passed", tests) or "failed" in tests.lower():
        errors.append("Final core tests did not pass")
    gpu_tests = (run/"geometry-tests.log").read_text()
    if "2 passed" not in gpu_tests or "failed" in gpu_tests.lower():
        errors.append("Recorded GPU geometry tests did not pass")
    unexpected = [str(p) for p in (run/"tasks").glob("*/*/result.json") if (p.parent.parent.name, p.parent.name) not in expected]
    if unexpected:
        errors.append(f"Unexpected tasks: {unexpected}")
    output = {"passed": not errors, "updated_beijing": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds"),
              "expected_tasks": len(expected), "saved_official_scores": len(results),
              "public_scan_scenes": scenes, "extended_variants": selected,
              "development_configurations": len(ORDER), "selection_eligible_development_configurations": len(ORDER)-1,
              "excluded_CG_scenes": cg_excluded, "direct_solve_views_checked": len(residuals),
              "maximum_direct_relative_residual": max(residuals, default=None),
              "source_files_hashed": input_files, "earlier_source_hashes_reverified_scenes": verified_early,
              "other_scene_hash_scope": "Final preservation manifests; not evidence of a pre-run content freeze",
              "official_source_files_unchanged": len(official_sources), "validation_algorithm_hashes_unchanged": not any("Frozen validation" in e for e in errors),
              "validation_algorithm_hash_scope": "code-snapshot-validation, not the maintained active package",
              "core_tests": "6 passed on CPU", "additional_geometry_smoke": "2 passed: stereo warp on CPU, depth-basis optimizer on GPU; cold PTX compilation included in time",
              "historical_repaired_program_errors": ["office/mv_depth initial variable coverage shape mismatch"],
              "current_unresolved_errors": errors}
    atomic_json(output_dir/"completion-audit.json", output)
    print(json.dumps(output, indent=2, ensure_ascii=False))
    raise SystemExit(0 if not errors else 1)


if __name__ == "__main__":
    main()
