"""Maximum-spanning-tree initialization of global camera centres and scales.

Given the per-star relative-pose predictions and an initial estimate of the
global rotations, this module builds a graph over images, picks an MST that
maximises the per-edge confidence, and walks the tree to seed each image's
global centre and scale before similarity averaging refines them.

Scale is fundamentally a per-star property (each star's local reconstruction
has its own arbitrary scale), so it can only be *reconciled* between two
stars using an edge observed from both directions (i.e. both endpoint images
have their own star). With InfoStar (gluemap-execution-plan.md Sec 2) not
every image is necessarily a star center; images that are only ever a
*neighbor* in someone else's star ("leaf" images) have no local scale of
their own to reconcile. Those images can't participate in the star-to-star
MST, but once their covering star's global frame is solved, their position
follows directly from it (no separate scale needed) -- see the
``_seed_leaf_image_centers`` pass below. Whether this remains solvable
depends on ``initialize_mst_structures``'s star-only graph staying
connected; ``gluemap.estimators.info_star.connected_q_cover`` is
specifically designed to guarantee that (see its docstring).
"""

import logging

import networkx as nx
import numpy as np
import torch

from gluemap.math.geometry_weight import (
    compute_median_tri_angles,
    geom_weight,
)

logger = logging.getLogger(__name__)

# Minimum median triangulation angle (degrees) for an edge's relative scale
# to be considered reliable; below this we fall back to a unit scale ratio.
MIN_TRI_ANGLE = 1
# Geometry-weight scale default withheld until paper acceptance.
DEFAULT_GEOM_THETA0_DEG = None


def initialize_mst_structures(
    predictions_dict: dict,
    global_rotations: dict[int, np.ndarray],
    geom_theta0_deg: float = DEFAULT_GEOM_THETA0_DEG,
    use_geom_weight: bool = True,
) -> tuple[dict[int, np.ndarray], dict[int, float]]:
    """Seed global centres and scales by walking an MST over star edges.

    The function reads ``predictions_dict["indexes"]``,
    ``["points3d_virtual"]``, ``["extrinsics"]``, and ``["pose_scores"]``,
    and writes ``predictions_dict["median_tri_angle"]`` as a side effect
    (or reuses it if already present from an earlier call).
    Edges whose median triangulation angle is below ``MIN_TRI_ANGLE`` get a
    unit relative scale; otherwise the relative scale is the ratio of
    translation norms between the two directions of the edge.

    When ``use_geom_weight`` is True (P2-2.1), MST edge weights are
    ``r_ij · g_ij`` with ``g = g(θ; θ₀)`` instead of the legacy
    ``r · min(20, θ) / 20``.

    Args:
        predictions_dict: Per-star prediction dict produced upstream. Must
            contain ``"indexes"``, ``"points3d_virtual"``, ``"extrinsics"``,
            and ``"pose_scores"``. ``"median_tri_angle"`` is added in place.
        global_rotations: Mapping image-index → ``(3, 3)`` rotation matrix.
        geom_theta0_deg: Scale of the continuous geometry weight.
        use_geom_weight: If False, fall back to legacy ``min(20,θ)/20``.

    Returns:
        Tuple of:
            * ``global_centers`` keyed by image index, each ``(3,)`` float64.
            * ``global_scales`` keyed by star index.
    """
    N = max(global_rotations.keys()) + 1
    indexes = range(len(predictions_dict["indexes"]))

    rel_poses = {}
    scales = {}  # (i,j): s_j / s_i
    node_idx_to_star_idx = {}

    if "median_tri_angle" not in predictions_dict:
        compute_median_tri_angles(predictions_dict)

    for star_idx, idx in enumerate(indexes):
        node_idx_to_star_idx[predictions_dict["indexes"][idx][0]] = star_idx

    star_nodes = set(node_idx_to_star_idx.keys())

    for idx in indexes:
        poses = predictions_dict["extrinsics"][idx]
        pose_scores = predictions_dict["pose_scores"][idx]
        N_poses = poses.shape[1]
        idx_i = predictions_dict["indexes"][idx][0]
        star_idx_i = node_idx_to_star_idx[idx_i]  # idx_i is always a center
        for i in range(N_poses):
            if i == 0:
                continue
            idx_j = predictions_dict["indexes"][idx][i]
            status = (idx_j, idx_i) not in rel_poses
            score = pose_scores[0, i].item()
            if not status:
                # (idx_j, idx_i) was already recorded, which only happens
                # when idx_j is itself an active star's own center (the
                # first element of every rel_poses key is always a star
                # center) -- so this lookup is always safe here.
                star_idx_j = node_idx_to_star_idx[idx_j]
                # If either edge has a small triangulation angle, scale is
                # unreliable
                angle_current = predictions_dict["median_tri_angle"][
                    star_idx_i
                ][i - 1]
                reverse_pos = rel_poses[(idx_j, idx_i)][2]
                angle_reverse = predictions_dict["median_tri_angle"][
                    star_idx_j
                ][reverse_pos - 1]
                if (
                    angle_current < MIN_TRI_ANGLE
                    or angle_reverse < MIN_TRI_ANGLE
                ):
                    scales[(star_idx_j, star_idx_i)] = 1.0
                    scales[(star_idx_i, star_idx_j)] = 1.0
                else:
                    scales[(star_idx_j, star_idx_i)] = (
                        poses[0, i, :3, 3:].norm().item()
                        / rel_poses[(idx_j, idx_i)][0][:3, 3:].norm().item()
                    )
                    scales[(star_idx_i, star_idx_j)] = (
                        1.0 / scales[(star_idx_j, star_idx_i)]
                    )
                # Edge weight: r_ij · g_ij (P2-2.1) or legacy linear ramp.
                if use_geom_weight:
                    g = min(
                        geom_weight(float(angle_current), geom_theta0_deg),
                        geom_weight(float(angle_reverse), geom_theta0_deg),
                    )
                    score *= g
                else:
                    score *= (
                        min(20, angle_current, angle_reverse) / 20
                    )
            # Else: idx_j has no star of its own (a "leaf" image, only ever
            # a neighbor -- possible under InfoStar). No scale to reconcile;
            # kept in rel_poses for _seed_leaf_image_centers below, but
            # never added to the star-only MST graph (see star_nodes).

            rel_poses[(idx_i, idx_j)] = (poses[0, i].cpu(), idx, i, score)

    # Restrict the scale-propagation MST to edges between two star-having
    # images with a resolved scale; edges to leaf images are handled
    # separately below.
    star_edges = {
        (i, j): payload
        for (i, j), payload in rel_poses.items()
        if j in star_nodes
        and (node_idx_to_star_idx[i], node_idx_to_star_idx[j]) in scales
    }

    G = nx.Graph()
    G.add_nodes_from(star_nodes)
    for (i, j), (_pose, _idx, _i_pos, score) in star_edges.items():
        G.add_edge(i, j, weight=score)
    mst = nx.maximum_spanning_tree(G)

    global_centers = {}
    global_scales = {}
    visited = set()

    # Iterative DFS to avoid RecursionError on large graphs. Root choice is
    # an arbitrary global gauge (position/scale are only defined up to a
    # similarity transform at this stage) -- `min(star_nodes)` keeps this
    # identical to the old hardcoded `0` whenever image 0 has its own star
    # (always true pre-InfoStar).
    root = min(star_nodes) if star_nodes else None
    if root is not None:
        global_centers[root] = np.zeros((3,), dtype=np.float64)
        global_scales[node_idx_to_star_idx[root]] = 1.0
        visited.add(root)
        stack = [(root, iter(mst.neighbors(root)))]

        while stack:
            node, neighbors_iter = stack[-1]
            try:
                neighbor = next(neighbors_iter)
            except StopIteration:
                stack.pop()
                continue

            if neighbor in visited:
                continue

            visited.add(neighbor)
            idx_node = node_idx_to_star_idx[neighbor]
            idx_parent = node_idx_to_star_idx[node]
            pose, idx, i_pos, _ = rel_poses[(neighbor, node)]

            if idx_node in global_scales:
                global_scales[idx_parent] = (
                    global_scales[idx_node] * scales[(idx_node, idx_parent)]
                )
            else:
                global_scales[idx_node] = (
                    global_scales[idx_parent] * scales[(idx_parent, idx_node)]
                )

            # s_i * (c_j - c_i) = -R_j^T * t_ij
            # ==> c_i = c_j + R_j^T * t_ij / s_i
            global_centers[neighbor] = (
                global_centers[node]
                + (global_rotations[node].T @ pose[:3, 3:].numpy()).flatten()
                / global_scales[idx_node]
            )

            stack.append((neighbor, iter(mst.neighbors(neighbor))))

    _seed_leaf_image_centers(
        rel_poses, node_idx_to_star_idx, global_rotations,
        global_centers, global_scales, N,
    )

    return global_centers, global_scales


def _seed_leaf_image_centers(
    rel_poses: dict,
    node_idx_to_star_idx: dict,
    global_rotations: dict[int, np.ndarray],
    global_centers: dict[int, np.ndarray],
    global_scales: dict[int, float],
    N: int,
) -> None:
    """Seed global centres for images with no star of their own (InfoStar).

    A "leaf" image only ever appears as a *neighbor* in other stars, never
    as a center, so it has no local scale to reconcile against the MST. Its
    global position is instead read directly off of whichever covering
    star's frame has already been solved -- same propagation formula as the
    MST walk, just keyed by the covering star's image index (``i``) rather
    than the leaf's own (nonexistent) star. Picks the highest-score
    covering star when several are available (InfoStar's ``q_min`` is
    designed to make that the common case). Mutates ``global_centers`` in
    place; images with no reachable covering star are left unset (mirrors
    the pre-existing behavior for images unreachable in a disconnected
    star graph, see TestMSTInitializationDisconnected).
    """
    star_nodes = set(node_idx_to_star_idx.keys())
    leaf_images = [
        m for m in range(N) if m not in star_nodes and m not in global_centers
    ]
    if not leaf_images:
        return

    best_by_leaf: dict[int, tuple[float, int, torch.Tensor]] = {}
    for (i, j), (pose, _idx, _i_pos, score) in rel_poses.items():
        if j in star_nodes:
            continue  # handled by the star-only MST above
        if i not in global_centers:
            continue  # i's own star wasn't reachable from the MST root
        current = best_by_leaf.get(j)
        if current is None or score > current[0]:
            best_by_leaf[j] = (score, i, pose)

    n_unreachable = 0
    for m in leaf_images:
        best = best_by_leaf.get(m)
        if best is None:
            n_unreachable += 1
            continue
        _score, i, pose = best
        star_idx_i = node_idx_to_star_idx[i]
        # pose[:3, 3:] = -R_m @ scale_i * (c_m - c_i) (see
        # tests/helpers.py:build_predictions_dict / the forward direction
        # of the same identity the MST DFS above solves for c_neighbor).
        # Solving for c_m given the already-known c_i, scale_i:
        #   c_m = c_i - R_m^T @ pose[:3, 3:] / scale_i
        # Note this uses the LEAF's own rotation (global_rotations[m]), not
        # the covering star's -- unlike the DFS step above where the node
        # being solved for is also the one whose star's scale is divided
        # through, here it's the *other* way around (i is known, m is
        # solved for), so both the sign and which rotation matrix is used
        # flip relative to the DFS formula.
        global_centers[m] = global_centers[i] - (
            global_rotations[m].T @ pose[:3, 3:].numpy()
        ).flatten() / global_scales[star_idx_i]

    if n_unreachable > 0:
        logger.warning(
            "initialize_mst_structures: %d leaf image(s) had no reachable "
            "covering star and were left out of global_centers",
            n_unreachable,
        )
