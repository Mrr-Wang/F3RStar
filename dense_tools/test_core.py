import re
import subprocess

import numpy as np

from .evaluate import BINARY, read_xyz, voxel_means, write_xyz
from gluemap.dense.edge_correct import edge_correct


def test_binary_ply_and_voxel_preserve_coordinates(tmp_path):
    xyz = np.array([[-0.001, 0, 0], [-0.002, 0, 0], [0.011, 0, 0]], dtype=np.float32)
    path = tmp_path / "cloud.ply"
    write_xyz(path, xyz)
    np.testing.assert_allclose(read_xyz(path), xyz)
    result = voxel_means(xyz, 0.01)
    np.testing.assert_allclose(np.sort(result[:, 0]), [-0.0015, 0.011], atol=1e-8)


def test_edge_fit_excludes_holdout_and_respects_depth_boundary():
    depth = np.ones((32, 48), np.float32)
    depth[:, 24:] = 2
    rgb = np.zeros((32, 48, 3), np.uint8)
    anchors = [[x, y, 1.08, 1, y*48+x, 1]
               for y in [3, 10, 20, 28] for x in [3, 9, 16, 21]]
    # A deliberately wrong validation anchor must never enter the fit.
    anchors.append([30, 15, 10, 1, 99999, 0])
    corrected, stats = edge_correct(depth, rgb, np.array(anchors, float))
    assert stats["applied"]
    assert stats["n_anchors"] == 16
    assert corrected[:, :24].mean() > 1.03
    np.testing.assert_allclose(corrected[:, 24:], 2, atol=1e-4)


def test_no_reliable_anchors_is_identity():
    depth = np.ones((16, 16), np.float32)
    corrected, stats = edge_correct(depth, np.zeros((16, 16, 3), np.uint8), np.empty((0, 6)))
    assert not stats["applied"]
    np.testing.assert_array_equal(corrected, depth)


def test_official_evaluator_applies_nonidentity_mlp(tmp_path):
    assert BINARY.exists(), "Build the official evaluator before running tests"
    yy, xx = np.meshgrid(np.linspace(-0.1, 0.1, 12), np.linspace(-0.1, 0.1, 12))
    scan = np.column_stack([xx.ravel(), yy.ravel(), np.ones(xx.size)])
    translation = np.array([0.3, 0.2, 0.1])
    write_xyz(tmp_path / "scan.ply", scan)
    write_xyz(tmp_path / "prediction.ply", scan+translation)
    matrix = np.eye(4)
    matrix[:3, 3] = translation
    matrix_text = "\n".join(" ".join(str(v) for v in row) for row in matrix)
    mlp = tmp_path / "scan_alignment.mlp"
    mlp.write_text('<MeshLabProject><MeshGroup><MLMesh filename="scan.ply"><MLMatrix44>' +
                   matrix_text + '</MLMatrix44></MLMesh></MeshGroup></MeshLabProject>')
    proc = subprocess.run([str(BINARY), "--reconstruction_ply_path", str(tmp_path/"prediction.ply"),
                           "--ground_truth_mlp_path", str(mlp), "--tolerances", "0.01,0.02"],
                          capture_output=True, text=True, check=True)
    match = re.search(r"^F1-scores:\s*(.+)$", proc.stdout, re.MULTILINE)
    assert match, proc.stdout
    np.testing.assert_allclose([float(x) for x in match.group(1).split()], [1, 1])
