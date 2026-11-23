"""Unit tests for P1-1.2 / P1-1.3 / P1-1.4 covisibility scoring changes."""

from __future__ import annotations

import torch

from gluemap.estimators.covisibility_extraction import CovisibilityExtraction


def _make_identity_star(B=1, N=3, H=32, W=32, depth=5.0):
    """Synthetic star: identical views, identity extrinsics, constant depth."""
    extrinsics = torch.eye(4).view(1, 1, 4, 4).expand(B, N, 4, 4).contiguous()
    # Simple pinhole: fx=fy=W, cx=W/2, cy=H/2
    K = torch.tensor(
        [[float(W), 0.0, W / 2.0], [0.0, float(W), H / 2.0], [0.0, 0.0, 1.0]]
    )
    intrinsics = K.view(1, 1, 3, 3).expand(B, N, 3, 3).contiguous()
    depth_map = torch.full((B, N, H, W, 1), depth)
    return extrinsics, intrinsics, depth_map


def test_fov_aware_scores_near_one_for_identical_views():
    """With perfect overlap, r_ij should be ~1 (not diluted by H*W)."""
    ext, K, depth = _make_identity_star()
    ce = CovisibilityExtraction(cover_min=0.01)
    world = ce._convert_from_depth_to_world_points(depth, ext, K)
    scores, _valid = ce._verify_by_reprojection(world, ext, K)
    # View 0 vs itself and identical neighbors: full support, all agree.
    assert scores.shape == (1, 3)
    assert torch.all(scores[:, 1:] > 0.95)


def test_cover_min_gates_tiny_overlap():
    """A view with almost no FOV overlap must score 0 under cover_min."""
    B, N, H, W = 1, 2, 32, 32
    ext = torch.eye(4).view(1, 1, 4, 4).expand(B, N, 4, 4).contiguous()
    # Shift camera 1 far sideways so almost nothing projects in-frame.
    ext = ext.clone()
    ext[0, 1, 0, 3] = 100.0  # huge baseline in X
    K = torch.tensor(
        [[float(W), 0.0, W / 2.0], [0.0, float(W), H / 2.0], [0.0, 0.0, 1.0]]
    ).view(1, 1, 3, 3).expand(B, N, 3, 3).contiguous()
    depth = torch.full((B, N, H, W, 1), 5.0)

    ce = CovisibilityExtraction(cover_min=0.01)
    world = ce._convert_from_depth_to_world_points(depth, ext, K)
    scores, _ = ce._verify_by_reprojection(world, ext, K)
    # Neighbor should be gated to 0 (cover << 1%).
    assert float(scores[0, 1]) == 0.0


def test_n2_returns_scores_all_matrix():
    """P1-1.4: _verify_by_reprojection_n2 returns (scores, mask, scores_all)."""
    ext, K, depth = _make_identity_star(N=4)
    ce = CovisibilityExtraction()
    world = ce._convert_from_depth_to_world_points(depth, ext, K)
    scores, mask, scores_all = ce._verify_by_reprojection_n2(world, ext, K)
    assert scores.shape == (1, 4)
    assert mask.shape[0] == 1
    assert scores_all.shape == (1, 4, 4)
    # Symmetrised with max.
    assert torch.allclose(scores_all, scores_all.transpose(1, 2))


def test_main_forwards_images_change_and_optional_scores_all():
    """main accepts images_shape_ori and can return scores_all."""
    B, N, H, W = 1, 3, 16, 16
    ext, K, depth = _make_identity_star(B=B, N=N, H=H, W=W)
    predictions = {
        "extrinsics": ext,
        "intrinsics": K,
        "depth": depth,
        "depth_conf": torch.ones(B, N, H, W),
    }
    indexes = torch.arange(N).view(1, N)
    # Pure scale, no letterbox (ETH3D-like).
    images_change = torch.tensor([[0.5, 0.5, 0.0, 0.0]]).expand(B, N, 4).float()
    images_shape_ori = torch.tensor([[H * 2, W * 2]]).expand(B, N, 2).float()

    ce = CovisibilityExtraction(return_scores_all=True)
    out = ce.main(predictions, indexes, images_change, images_shape_ori)
    assert len(out) == 7
    scores_all = out[6]
    assert scores_all.shape == (B, N, N)


def test_symmetrize_modes_max_min_geom():
    """P1-1.5: max / min / geom produce distinct, symmetric matrices."""
    asymmetric = torch.tensor(
        [[[0.9, 0.2, 0.0], [0.8, 0.9, 0.1], [0.0, 0.5, 0.9]]]
    )
    mx = CovisibilityExtraction._symmetrize_scores(asymmetric, "max")
    mn = CovisibilityExtraction._symmetrize_scores(asymmetric, "min")
    gm = CovisibilityExtraction._symmetrize_scores(asymmetric, "geom")
    assert torch.allclose(mx, mx.transpose(1, 2))
    assert torch.allclose(mn, mn.transpose(1, 2))
    assert torch.allclose(gm, gm.transpose(1, 2))
    assert abs(float(mx[0, 0, 1]) - 0.8) < 1e-6  # max(0.2, 0.8)
    assert abs(float(mn[0, 0, 1]) - 0.2) < 1e-6  # min(0.2, 0.8)
    assert abs(float(gm[0, 0, 1]) - (0.2 * 0.8) ** 0.5) < 1e-6
    # Conservative ≤ optimistic.
    assert torch.all(mn <= mx + 1e-6)


def test_depth_noise_seed_is_reproducible():
    """P1-1.6: same rng_seed → identical virtual-track 3D points."""
    ext, K, depth = _make_identity_star(H=28, W=28)  # divisible by 14
    ce_a = CovisibilityExtraction(rng_seed=7)
    ce_b = CovisibilityExtraction(rng_seed=7)
    ce_c = CovisibilityExtraction(rng_seed=99)
    world = ce_a._convert_from_depth_to_world_points(depth, ext, K)
    _, mask = ce_a._verify_by_reprojection(world, ext, K)
    out_a = ce_a._calculate_virtual_tracks(depth, ext, K, mask)
    out_b = ce_b._calculate_virtual_tracks(depth, ext, K, mask)
    out_c = ce_c._calculate_virtual_tracks(depth, ext, K, mask)
    # Compare points3d_virtual (index 1) — depth noise lands here;
    # 2D tracks can be identical under identity extrinsics.
    assert torch.allclose(out_a[1], out_b[1])
    assert not torch.allclose(out_a[1], out_c[1])


def test_virtual_track_subsample_seed_is_reproducible():
    """P1-1.6: randperm/multinomial draws are deterministic under rng_seed."""
    from gluemap.estimators.virtual_tracks import VirtualTrackPreparation

    B, N, K = 1, 4, 50
    tracks = torch.randn(B, N, K, 2)
    points = torch.randn(B, K, 3)
    valid = torch.ones(B, N, K, dtype=torch.bool)
    isneg = torch.zeros(B, N, K, dtype=torch.bool)
    valid[0, 2, 10:] = False

    def run(seed):
        d = {
            "tracks_virtual": [tracks.clone()],
            "points3d_virtual": [points.clone()],
            "valid_virtual": [valid.clone()],
            "isnegative_virtual": [isneg.clone()],
        }
        VirtualTrackPreparation(
            num_desired_tracks=20, min_num=3, rng_seed=seed
        )._subsample_virtual_tracks(d, indexes=[0])
        return d["tracks_virtual"][0].clone()

    assert torch.allclose(run(123), run(123))
    assert not torch.allclose(run(123), run(456))
