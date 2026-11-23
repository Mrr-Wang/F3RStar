import logging

import numpy as np
import pyceres
import pygluemap
import torch

logger = logging.getLogger(__name__)
# Minimum angle (in degrees) for a triangle to be considered valid for
# scale estimation
MIN_TRI_ANGLE = 1


def _initialize_parameters(
    predictions_dict: dict,
    global_rotations: dict[int, np.ndarray],
    global_centers: dict[int, np.ndarray] | None,
    global_scales: list[np.ndarray] | dict[int, float] | None,
) -> tuple[int, dict[int, np.ndarray], list[np.ndarray]]:
    """
    Allocate per-image centers and per-ministar scale parameter buffers.

    Args:
        predictions_dict: Star inference output; only ``"indexes"`` is read
            here to determine the number of ministars.
        global_rotations: Per-image rotation matrices ``{image_id: R}``.
        global_centers: Optional initial per-image camera centers; missing
            images get random initialization.
        global_scales: Optional initial per-ministar scales as a list, or as
            a ``{ministar_idx: scale}`` mapping; ``None`` defaults to ones.

    Returns:
        ``(num_ministar, global_centers, global_scales)`` where the latter
        two are normalized into the canonical containers used downstream.
    """
    num_ministar = len(predictions_dict["indexes"])
    if global_centers is None:
        global_centers = {
            idx: np.random.rand(3).astype(np.float64)
            for idx in global_rotations
        }
    else:
        for idx in global_rotations:
            if idx not in global_centers:
                global_centers[idx] = np.random.rand(3).astype(np.float64)
    if global_scales is None:
        global_scales = [
            np.ones((1,)).astype(np.float64) for i in range(num_ministar)
        ]
    elif isinstance(global_scales, dict):
        temp_scales = []
        for idx_star in range(num_ministar):
            if idx_star in global_scales:
                temp_scales.append(
                    np.ones((1,)).astype(np.float64) * global_scales[idx_star]
                )
            else:
                temp_scales.append(np.ones((1,)).astype(np.float64))
        global_scales = temp_scales

    return num_ministar, global_centers, global_scales


def _add_star_edge_error(
    prob: pyceres.Problem,
    predictions_dict: dict,
    global_rotations: dict[int, np.ndarray],
    global_centers: dict[int, np.ndarray],
    global_scales: list[np.ndarray],
    num_ministar: int,
    costs: list,
    losses: list,
    geom_theta0_deg: float | None = None,
    use_geom_weight: bool = True,
    composite_edges: list | None = None,
) -> tuple[int, int, int]:
    """
    Add pairwise direction residuals for every valid star edge.

    For each ministar's center image and its neighbours, attaches a
    ``PairwiseDirectionError`` cost block linking the two camera centers
    and the ministar scale. Edges with score <= 0 or median triangulation
    angle below :data:`MIN_TRI_ANGLE` are skipped. Pins the first chosen
    center and its ministar's scale to remove the gauge.

    When ``use_geom_weight`` is True (P2-2.1), the Huber magnitude is
    ``r_ij · g_ij`` with ``g = g(θ; θ₀)``; otherwise magnitude is
    plain ``r_ij`` (legacy).

    Optional ``composite_edges`` (§5 step 3) add neighbour↔neighbour
    direction residuals (never used for gauge pinning). Composite geom
    weight uses ``min(θ_spoke_i, θ_spoke_j)``; either spoke below
    ``MIN_TRI_ANGLE`` skips that composite.

    Returns:
        ``(gauge_center_id, n_spoke, n_composite)``. Gauge id is ``-1`` if
        no spoke edge was added.
    """
    from gluemap.math.geometry_weight import geom_weight

    center = -1
    n_spoke = 0
    for idx_star in range(num_ministar):
        scores = predictions_dict["pose_scores"][idx_star][0]
        idx1 = predictions_dict["indexes"][idx_star][0]
        valid_j = torch.where(scores > 0.0)[0].tolist()

        # Now, only consider the scale of the center image
        for idx in valid_j:
            idx2 = predictions_dict["indexes"][idx_star][idx]

            if idx1 == idx2:
                continue

            angle = float(
                predictions_dict["median_tri_angle"][idx_star][idx - 1].item()
            )
            if angle < MIN_TRI_ANGLE:
                continue  # Skip near-degenerate edges

            weight = float(scores[idx].item())
            if use_geom_weight:
                weight *= geom_weight(angle, geom_theta0_deg)

            # s_i * (c_j - c_i) = -R_j^T * t_ij
            t_ij_rotated = (
                -global_rotations[idx2].T
                @ predictions_dict["extrinsics"][idx_star][0, idx, :3, 3:]
                .cpu()
                .numpy()
            )
            loss_scaled = pyceres.LossFunction(
                {"name": "huber", "params": [1e-2], "magnitude": weight}
            )

            cost = pygluemap.PairwiseDirectionError(t_ij_rotated)
            prob.add_residual_block(
                cost,
                loss_scaled,
                [
                    global_centers[idx1],
                    global_centers[idx2],
                    global_scales[idx_star],
                ],
            )

            costs.append(cost)
            losses.append(loss_scaled)
            n_spoke += 1

            if center < 0:
                # Gauge from the first spoke only — never a composite.
                prob.set_parameter_block_constant(global_centers[idx1])
                prob.set_parameter_block_constant(global_scales[idx_star])
                center = idx1

    n_composite = 0
    if composite_edges:
        for edge in composite_edges:
            if edge.i not in global_centers or edge.j not in global_centers:
                continue
            if edge.star_idx >= num_ministar:
                continue
            members = list(predictions_dict["indexes"][edge.star_idx])
            try:
                # After prune, local indices may have shifted — resolve by
                # global id. median_tri_angle[star][k] ↔ members[k+1].
                pos_i = members.index(edge.i)
                pos_j = members.index(edge.j)
            except ValueError:
                continue
            if pos_i == 0 or pos_j == 0:
                continue
            try:
                ang_i = float(
                    predictions_dict["median_tri_angle"][edge.star_idx][
                        pos_i - 1
                    ].item()
                )
                ang_j = float(
                    predictions_dict["median_tri_angle"][edge.star_idx][
                        pos_j - 1
                    ].item()
                )
            except (IndexError, KeyError, TypeError):
                continue
            if ang_i < MIN_TRI_ANGLE or ang_j < MIN_TRI_ANGLE:
                continue
            angle = min(ang_i, ang_j)
            weight = float(edge.weight)
            if use_geom_weight:
                weight *= geom_weight(angle, geom_theta0_deg)
            if weight <= 0.0:
                continue

            t_jk_rotated = -global_rotations[edge.j].T @ edge.t_ij.reshape(3, 1)
            loss_scaled = pyceres.LossFunction(
                {"name": "huber", "params": [1e-2], "magnitude": weight}
            )
            cost = pygluemap.PairwiseDirectionError(t_jk_rotated)
            prob.add_residual_block(
                cost,
                loss_scaled,
                [
                    global_centers[edge.i],
                    global_centers[edge.j],
                    global_scales[edge.star_idx],
                ],
            )
            costs.append(cost)
            losses.append(loss_scaled)
            n_composite += 1

    return center, n_spoke, n_composite


def _update_points3d(
    prob: pyceres.Problem,
    predictions_dict: dict,
    global_rotations: dict[int, np.ndarray],
    global_centers: dict[int, np.ndarray],
    global_scales: list[np.ndarray],
    num_ministar: int,
) -> None:
    """
    Apply solved similarity transforms back to predicted 3D content.

    Rescales/rotates ``world_points``, ``cam_points`` and
    ``points3d_virtual`` (when present) and rescales ministar extrinsics so
    everything is expressed in the new global frame. Mutates
    ``predictions_dict`` in place.

    Args:
        prob: Ceres problem used to detect which ministar scales were
            actually parameter blocks (others are skipped).
        predictions_dict: Star inference dict; relevant tensors are updated
            in place.
        global_rotations: Per-image rotation matrices used to rotate world
            points back into the global frame.
        global_centers: Per-image solved centers used as translations.
        global_scales: Per-ministar solved scales (length-1 numpy arrays).
        num_ministar: Number of ministars to process.
    """
    if (
        "world_points" in predictions_dict
        and len(predictions_dict["world_points"]) > 0
    ):
        # Rescale the world points
        for idx in range(num_ministar):
            idx_center = predictions_dict["indexes"][idx][0]
            predictions_dict["world_points"][idx] = (
                predictions_dict["world_points"][idx]
                @ global_rotations[idx_center]
                / global_scales[idx]
                + global_centers[idx_center]
            )
    # Since it is in camera frame, we only need to change the scale
    if (
        "cam_points" in predictions_dict
        and len(predictions_dict["cam_points"]) > 0
        and predictions_dict["cam_points"][0] is not None
    ):
        for idx in range(num_ministar):
            idx_center = predictions_dict["indexes"][idx][0]
            predictions_dict["cam_points"][idx] = (
                predictions_dict["cam_points"][idx] / global_scales[idx]
            )

    # Rescale the points and extrinsics
    if (
        "points3d_virtual" in predictions_dict
        and len(predictions_dict["points3d_virtual"]) > 0
    ):
        # Rescale the virtual points
        for idx in range(num_ministar):
            predictions_dict["points3d_virtual"][idx] = (
                predictions_dict["points3d_virtual"][idx] / global_scales[idx]
            )

    for idx_star in range(num_ministar):
        if not prob.has_parameter_block(global_scales[idx_star]):
            continue
        predictions_dict["extrinsics"][idx_star][:, :, :3, 3:] = (
            predictions_dict["extrinsics"][idx_star][:, :, :3, 3:]
            / global_scales[idx_star][0]
        )


def similarity_averaging(
    predictions_dict: dict,
    global_rotations: dict[int, np.ndarray],
    global_centers: dict[int, np.ndarray] | None = None,
    global_scales: list[np.ndarray] | dict[int, float] | None = None,
    max_num_iterations: int = 50,
    fix_scales: bool = False,
    geom_theta0_deg: float | None = None,
    use_geom_weight: bool = True,
    composite_edges: list | None = None,
    stats_out: dict | None = None,
) -> tuple[dict[int, np.ndarray], dict[int, float]]:
    """
    Solve for global per-image translations and per-ministar scales.

    Builds a Ceres problem from pairwise direction residuals over all valid
    ministar edges, fixes the gauge, solves, then back-substitutes the
    similarity transform into ``predictions_dict``.

    Args:
        predictions_dict: Star inference output; mutated in place by
            :func:`_update_points3d`.
        global_rotations: Per-image rotations (assumed already solved by
            rotation averaging).
        global_centers: Optional initial centers; missing entries are
            randomized via :func:`_initialize_parameters`.
        global_scales: Optional initial scales (list or dict).
        max_num_iterations: Max Ceres iterations.
        fix_scales: If True, freeze every ministar scale (useful when only
            translations should move).
        geom_theta0_deg / use_geom_weight: P2-2.1 continuous ``g_ij`` for
            residual magnitudes (see :func:`_add_star_edge_error`).
        composite_edges: Optional §5 step-3 composite non-spoke edges.
        stats_out: If given, filled with ``n_spoke`` / ``n_composite``.

    Returns:
        ``(global_centers, global_scales_dict)`` where ``global_scales_dict``
        maps star index → solved scalar scale (DenseStar Stage B: previously
        only centers were returned and the per-star scales were discarded
        after being divided into ``points3d_virtual`` in place).
    """
    logger.info("Performing similarity averaging...")
    num_ministar, global_centers, global_scales = _initialize_parameters(
        predictions_dict,
        global_rotations,
        global_centers,
        global_scales,
    )

    prob = pyceres.Problem()

    costs = []
    losses = []

    _gauge, n_spoke, n_composite = _add_star_edge_error(
        prob,
        predictions_dict,
        global_rotations,
        global_centers,
        global_scales,
        num_ministar,
        costs,
        losses,
        geom_theta0_deg=geom_theta0_deg,
        use_geom_weight=use_geom_weight,
        composite_edges=composite_edges,
    )
    if stats_out is not None:
        stats_out["n_spoke"] = n_spoke
        stats_out["n_composite"] = n_composite

    for idx_star in range(num_ministar):
        if not prob.has_parameter_block(global_scales[idx_star]):
            continue
        if (
            fix_scales
            or predictions_dict["median_tri_angle"][idx_star].max().item()
            < MIN_TRI_ANGLE
        ):
            prob.set_parameter_block_constant(global_scales[idx_star])
        else:
            prob.set_parameter_lower_bound(global_scales[idx_star], 0, 1e-5)
            prob.set_parameter_upper_bound(global_scales[idx_star], 0, 1e5)

    options = pyceres.SolverOptions()
    options.linear_solver_type = pyceres.LinearSolverType.SPARSE_NORMAL_CHOLESKY
    options.num_threads = 32
    options.max_num_iterations = max_num_iterations
    options.minimizer_progress_to_stdout = False

    logger.info(
        "Solving the optimization problem (spoke=%d, composite=%d)...",
        n_spoke,
        n_composite,
    )
    summary = pyceres.SolverSummary()
    pyceres.solve(options, prob, summary)

    logger.info(summary.BriefReport())

    _update_points3d(
        prob,
        predictions_dict,
        global_rotations,
        global_centers,
        global_scales,
        num_ministar,
    )

    # Flatten the length-1 Ceres parameter blocks into a plain star→float map
    # for DenseStar (and any other consumer that needs D̃ = D / s_l).
    scales_dict = {
        idx_star: float(global_scales[idx_star][0])
        for idx_star in range(num_ministar)
    }
    return global_centers, scales_dict
