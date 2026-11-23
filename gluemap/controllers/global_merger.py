import argparse
import logging
import time

import networkx as nx
import numpy as np
import torch

from gluemap.estimators.intrinsics_averaging import intrinsics_averaging
from gluemap.estimators.cross_star_consistency import (
    apply_cross_star_consistency,
    collect_composite_motion_edges,
    collect_pair_rotations,
    compute_pair_dispersions,
)
from gluemap.estimators.rotation_averaging import (
    rotation_averaging,
    rotation_averaging_pycolmap,
)
from gluemap.estimators.similarity_averaging import (
    similarity_averaging,
)
from gluemap.math.geometry import restore_identity
from gluemap.math.geometry_weight import compute_median_tri_angles
from gluemap.math.mst_initialization import initialize_mst_structures

logger = logging.getLogger(__name__)


class GlobalGluer:
    """Glue per-star predictions into a single global reconstruction.

    Runs graph refinement, intrinsics averaging, rotation averaging, MST-based
    similarity initialization and similarity averaging on the star-inference
    predictions to produce global rotations, camera centers and intrinsics.

    Most private helpers mutate ``predictions_dict`` in place — most notably
    ``pose_scores`` (suppress weak/inconsistent edges) and the per-star list
    entries (prune invisible neighbors).
    """

    def __init__(self, args: argparse.Namespace):
        self.max_rot_error = 5  # degrees

        self.valid_threshold_pose = (
            args.valid_pose_threshold
            if hasattr(args, "valid_pose_threshold")
            else 0.05
        )

        self.thres_consistency = np.deg2rad(10.0)  # degrees
        self.angle_threshold = 5.0  # degrees
        self.boost_sequential = bool(
            hasattr(args, "is_sequential") and args.is_sequential
        )
        self.use_ceres_rotation_averaging = getattr(
            args, "use_ceres_rotation_averaging", False
        )
        # §5 step 1: cross-star consistency reweight / filter (opt-in).
        self.cross_star_consistency = bool(
            getattr(args, "cross_star_consistency", False)
        )
        self.cross_star_tau_deg = float(
            getattr(args, "cross_star_tau_deg", None)
        )
        self.cross_star_hard_thresh_deg = getattr(
            args, "cross_star_hard_thresh_deg", None
        )
        if self.cross_star_hard_thresh_deg is not None:
            self.cross_star_hard_thresh_deg = float(
                self.cross_star_hard_thresh_deg
            )
        self.cross_star_mode = str(
            getattr(args, "cross_star_mode", "reweight")
        )
        # §5 step 3: composite non-spoke edges into motion averaging.
        self.composite_motion_edges = str(
            getattr(args, "composite_motion_edges", "off")
        )
        if self.composite_motion_edges not in {"off", "consistent", "all"}:
            raise ValueError(
                f"unknown composite_motion_edges="
                f"{self.composite_motion_edges!r}; "
                f"expected off|consistent|all"
            )
        # P2-2.1: continuous g_ij for MST + similarity averaging.
        self.use_geom_weight = bool(getattr(args, "use_geom_weight", False))
        self.geom_theta0_deg = float(getattr(args, "geom_theta0_deg", None))
        # Filled by _global_structure_estimation for pipeline timing.
        self.structure_timing: dict = {}

    def main(
        self,
        predictions_dict: dict,
        intrinsics_mapping: dict[int, int],
        camera_model: str,
        num_img: int,
    ) -> tuple[
        dict[int, np.ndarray],
        dict[int, np.ndarray],
        list[np.ndarray | None],
        set[tuple[int, int]],
        dict,
        dict[int, float],
    ]:
        """Run global gluing: graph refine + intrinsics/structure estimation.

        Args:
            predictions_dict: Star-inference outputs (mutated in place).
            intrinsics_mapping: ``{image_id: camera_type_idx}``.
            camera_model: COLMAP camera model name (e.g. ``"SIMPLE_PINHOLE"``).
            num_img: Total number of images in the dataset.

        Returns:
            ``(global_rotations, global_centers, global_intrinsics, valid_edges,
            predictions_dict, global_scales)``. ``global_scales`` maps star
            index → solved scalar scale (DenseStar Stage B). The returned
            ``predictions_dict`` is the same mutated object that was passed in.
        """
        self.N = num_img
        # Refine the graph structure
        predictions_dict, valid_edges = self._refine_graph_structure(
            predictions_dict
        )

        # Estimate the intrinsics
        global_intrinsics = self._estimate_intrinsics(
            predictions_dict, intrinsics_mapping, camera_model
        )

        (
            global_rotations,
            global_centers,
            global_scales,
        ) = self._global_structure_estimation(
            predictions_dict,
        )

        return (
            global_rotations,
            global_centers,
            global_intrinsics,
            valid_edges,
            predictions_dict,
            global_scales,
        )

    def _refine_graph_structure(
        self, predictions_dict: dict
    ) -> tuple[dict, set[tuple[int, int]]]:
        """Suppress weak/inconsistent edges and connect any missing components.

        Modifies ``predictions_dict`` in place (populates ``scores``, zeros
        out ``pose_scores`` for inconsistent edges, prunes invisible pairs).
        """
        predictions_dict["scores"] = {}
        for idx in range(len(predictions_dict["indexes"])):
            predictions_dict["scores"][idx] = torch.where(
                predictions_dict["vis"][idx] > 0.05,
                predictions_dict["vis"][idx],
                0.0,
            )
        # Perform two way check for filtering simple outliers
        self._filter_inconsistent_edges(predictions_dict)

        # §5 step 1: down-weight / drop spoke edges whose relative pose
        # disagrees across the stars that contain both endpoints.
        if self.cross_star_consistency:
            apply_cross_star_consistency(
                predictions_dict,
                tau_deg=self.cross_star_tau_deg,
                hard_thresh_deg=self.cross_star_hard_thresh_deg,
                mode=self.cross_star_mode,
            )

        # First, we want to collect the valid edges
        valid_edges = self._collect_valid_edges(predictions_dict)

        # Then, connect the missing edges
        self._connect_missing(valid_edges, predictions_dict)

        # Prune invisible pairs
        self._prune_invisible_pairs(predictions_dict)

        return predictions_dict, valid_edges

    def _filter_inconsistent_edges(self, predictions_dict: dict) -> None:
        """Zero out ``pose_scores`` for edges whose two directions disagree.

        Uses a two-way check on relative rotation and translation; both
        directions of an inconsistent edge are suppressed. Modifies
        ``predictions_dict["pose_scores"]`` in place.
        """
        indexes = range(len(predictions_dict["indexes"]))

        rel_poses = {}
        counter = 0
        for idx in indexes:
            poses = predictions_dict["extrinsics"][idx]
            N = poses.shape[1]
            idx_i = predictions_dict["indexes"][idx][0]
            for i in range(N):
                if i == 0:
                    continue
                idx_j = predictions_dict["indexes"][idx][i]
                if (idx_j, idx_i) in rel_poses:
                    pose = rel_poses[(idx_j, idx_i)][0]
                    # Compare the two relative poses
                    R12 = pose[:3, :3]
                    R21 = poses[0, i, :3, :3]

                    error_r = torch.acos(
                        torch.clamp(
                            ((R12 @ R21).trace() - 1) / 2,
                            -1.0 + 1e-6,
                            1.0 - 1e-6,
                        )
                    )

                    # R21 * t12_normed = -t21_normed
                    t12_normed = pose[:3, 3:] / torch.linalg.norm(pose[:3, 3:])
                    t21_normed = poses[0, i, :3, 3:] / torch.linalg.norm(
                        poses[0, i, :3, 3:]
                    )
                    error_t = torch.acos(
                        torch.clamp(
                            -torch.sum(t12_normed * (R12 @ t21_normed)),
                            -1.0 + 1e-6,
                            1.0 - 1e-6,
                        )
                    )

                    if (
                        error_r > self.thres_consistency
                        or error_t > 3 * self.thres_consistency
                    ):
                        # Inconsistent, suppress both directions
                        predictions_dict["pose_scores"][idx][0, i] = 0.0
                        idx_1 = rel_poses[(idx_j, idx_i)][1]
                        j_pos = rel_poses[(idx_j, idx_i)][2]

                        predictions_dict["pose_scores"][idx_1][0, j_pos] = 0.0
                        logger.debug(
                            f"Filtered inconsistent edge between {idx_i} and "
                            f"{idx_j}, rotation error: "
                            f"{np.rad2deg(error_r)} degrees, translation "
                            f"error: {np.rad2deg(error_t)} degrees"
                        )
                        counter += 1
                else:
                    rel_poses[(idx_i, idx_j)] = (poses[0, i].cpu(), idx, i)

        logger.info(f"Total number of inconsistent edges filtered: {counter}")

    # TODO: debug this part
    def _collect_valid_edges(
        self, predictions_dict: dict
    ) -> set[tuple[int, int]]:
        """Return edges whose ``pose_scores`` exceed the threshold."""
        # Here, the score already considers the n^2 visibility, so we can
        # just use the pose scores
        valid_edges = set()
        indexes = range(len(predictions_dict["indexes"]))
        for idx in indexes:
            valid_j = torch.where(
                predictions_dict["pose_scores"][idx][0]
                > self.valid_threshold_pose
            )[0]
            valid_edges.update(
                set(
                    [
                        (
                            predictions_dict["indexes"][idx][0],
                            predictions_dict["indexes"][idx][j],
                        )
                        for j in valid_j[1:]
                    ]
                )
            )

        return valid_edges

    def _connect_missing(
        self,
        valid_edges: set[tuple[int, int]],
        predictions_dict: dict,
    ) -> None:
        """Bump cross-component pose scores so the graph becomes connected.

        For every pair of images that fall in different connected components
        of ``valid_edges``, increases ``pose_scores`` by ``1e-2`` and adds the
        edge to ``valid_edges`` in place.
        """
        N = self.N

        # Establish tree with the valid edges
        G = nx.Graph()
        G.add_edges_from(list(valid_edges))

        # Find the connected components
        components = list(nx.connected_components(G))
        if (len(components) == 1) and (len(components[0]) == N):
            logger.info(
                f"Edge connectivity of the graph: {nx.edge_connectivity(G)}"
            )
            return

        components = [list(x) for x in components]

        image_id_to_cluster_id = {}
        for i, component in enumerate(components):
            for image_id in component:
                image_id_to_cluster_id[image_id] = i

        curr_component_num = len(components)
        for i in range(N):
            if i not in image_id_to_cluster_id:
                image_id_to_cluster_id[i] = curr_component_num
                curr_component_num += 1

        # For edges across the components, we just set all scores to be 1e-2
        for idx in range(len(predictions_dict["indexes"])):
            idx1 = predictions_dict["indexes"][idx][0]
            for i, idx_inner in enumerate(predictions_dict["indexes"][idx]):
                if i == 0:
                    continue

                if (
                    image_id_to_cluster_id[idx1]
                    != image_id_to_cluster_id[idx_inner]
                ):
                    predictions_dict["pose_scores"][idx][0, i] += 1e-2
                    logger.debug(f"{idx1} {idx_inner} cross component")
                    valid_edges.add((idx1, idx_inner))

    def _resolve_composite_edges(self, predictions_dict: dict) -> list:
        """Build §5 step-3 composite edges, or ``[]`` when mode is ``off``."""
        mode = self.composite_motion_edges
        if mode == "off":
            return []

        dispersion = predictions_dict.get("cross_star_dispersion")
        if mode == "consistent":
            if not dispersion:
                # Preview only — do not mutate pose_scores.
                pair_rots = collect_pair_rotations(predictions_dict)
                stats = compute_pair_dispersions(pair_rots, min_stars=2)
                dispersion = {
                    pair: st.dispersion_deg for pair, st in stats.items()
                }
                predictions_dict["cross_star_dispersion"] = dispersion
            return collect_composite_motion_edges(
                predictions_dict,
                mode="consistent",
                dispersion=dispersion,
                tau_deg=self.cross_star_tau_deg,
            )

        # mode == "all"
        return collect_composite_motion_edges(
            predictions_dict,
            mode="all",
            dispersion=None,
            tau_deg=self.cross_star_tau_deg,
        )

    def _global_structure_estimation(
        self,
        predictions_dict: dict,
    ) -> tuple[
        dict[int, np.ndarray], dict[int, np.ndarray], dict[int, float]
    ]:
        """Estimate global rotations and camera centers from the refined graph.

        Runs rotation averaging (filtering invalid edges twice when using the
        Ceres backend), MST-based similarity initialization, and similarity
        averaging. Falls back to identity / zero entries for any image that
        was not estimated.

        Returns:
            ``(global_rotations, global_centers, global_scales)`` covering all
            ``N`` images (scales keyed by star index).
        """
        # Double sequential edge weights (neighboring frames with index
        # diff <= 10)
        if self.boost_sequential:
            self._boost_sequential_edges(predictions_dict, boost_factor=2.0)

        composite_edges = self._resolve_composite_edges(predictions_dict)
        ra_stats: dict = {}
        sim_stats: dict = {}

        t_ra = time.perf_counter()
        if self.use_ceres_rotation_averaging:
            # Original two-pass: RA -> filter -> RA -> filter
            # RA weights = r_ij (pose_scores) only — no g_ij (P2-2.1).
            global_rotations = rotation_averaging(
                predictions_dict,
                composite_edges=composite_edges,
                stats_out=ra_stats,
            )
            self._filter_invalid_edges(predictions_dict, global_rotations)
            global_rotations = rotation_averaging(
                predictions_dict,
                global_rotations,
                composite_edges=composite_edges,
            )
            self._filter_invalid_edges(predictions_dict, global_rotations)
        else:
            global_rotations = rotation_averaging_pycolmap(
                predictions_dict,
                max_rotation_error_deg=self.max_rot_error,
                composite_edges=composite_edges,
                stats_out=ra_stats,
            )
            self._filter_invalid_edges(predictions_dict, global_rotations)
        ra_s = time.perf_counter() - t_ra

        self._prune_invisible_pairs(predictions_dict)

        # After prune, (re)compute median triangulation angles for MST +
        # similarity averaging. Must follow prune so neighbor indices align.
        compute_median_tri_angles(predictions_dict)

        # Initialize the structures by maximum spanning tree
        global_centers, global_scales = initialize_mst_structures(
            predictions_dict,
            global_rotations,
            geom_theta0_deg=self.geom_theta0_deg,
            use_geom_weight=self.use_geom_weight,
        )

        t_sim = time.perf_counter()
        global_centers, global_scales = similarity_averaging(
            predictions_dict,
            global_rotations,
            global_centers=global_centers,
            global_scales=global_scales,
            max_num_iterations=200,
            geom_theta0_deg=self.geom_theta0_deg,
            use_geom_weight=self.use_geom_weight,
            composite_edges=composite_edges,
            stats_out=sim_stats,
        )
        sim_s = time.perf_counter() - t_sim

        self.structure_timing = {
            "rotation_averaging_s": ra_s,
            "similarity_averaging_s": sim_s,
            "n_composite_edges_collected": len(composite_edges),
            "ra_n_spoke": int(ra_stats.get("n_spoke", 0)),
            "ra_n_composite": int(ra_stats.get("n_composite", 0)),
            "sim_n_spoke": int(sim_stats.get("n_spoke", 0)),
            "sim_n_composite": int(sim_stats.get("n_composite", 0)),
            "composite_motion_edges": self.composite_motion_edges,
        }
        logger.info(
            "Motion averaging timing: RA=%.2fs (spoke=%d, composite=%d), "
            "sim=%.2fs (spoke=%d, composite=%d), collected_composites=%d "
            "(mode=%s)",
            ra_s,
            self.structure_timing["ra_n_spoke"],
            self.structure_timing["ra_n_composite"],
            sim_s,
            self.structure_timing["sim_n_spoke"],
            self.structure_timing["sim_n_composite"],
            len(composite_edges),
            self.composite_motion_edges,
        )

        # Prune the edges by the global rotations
        self._mark_inconsistent_edges(
            predictions_dict, global_rotations, global_centers
        )

        # Check whether all images are estimated
        logger.info(f"Number of images: {self.N}")
        logger.info(f"Number of global rotations: {len(global_rotations)}")
        logger.info(f"Number of global centers: {len(global_centers)}")
        logger.info(f"Number of global scales: {len(global_scales)}")
        if len(global_rotations) != self.N or len(global_centers) != self.N:
            global_rotations = {
                i: (
                    global_rotations[i]
                    if i in global_rotations
                    else np.eye(3, dtype=np.float64)
                )
                for i in range(self.N)
            }
            global_centers = {
                i: (
                    global_centers[i]
                    if i in global_centers
                    else np.zeros(3, dtype=np.float64)
                )
                for i in range(self.N)
            }

        return global_rotations, global_centers, global_scales

    def _estimate_intrinsics(
        self,
        predictions_dict: dict,
        intrinsics_mapping: dict[int, int],
        camera_model: str,
    ) -> list[np.ndarray | None]:
        """Run intrinsics averaging across all star predictions."""
        indexes = range(len(predictions_dict["indexes"]))
        intrinsics_all = [
            predictions_dict["intrinsics"][idx] for idx in indexes
        ]
        members = [predictions_dict["indexes"][idx] for idx in indexes]

        global_intrinsics = intrinsics_averaging(
            intrinsics_all, members, intrinsics_mapping, camera_model
        )

        return global_intrinsics

    def _prune_invisible_pairs(self, predictions_dict: dict) -> None:
        """Drop neighbors with ``pose_scores <= 0`` from each star, in place."""
        # Metadata keys are not per-star neighbor tensors (pair-keyed dicts,
        # image maps, scalar maps). ``scores`` is a star-keyed dict of tensors
        # and is pruned below.
        _skip_keys = {
            "points3d_virtual",
            "scales",
            "star_indexes",
            "image_index_to_star_index",
            "median_tri_angle",
            "cross_star_dispersion",
            "pose_inconsistent",
        }
        indexes = range(len(predictions_dict["indexes"]))
        for idx in indexes:
            if len(predictions_dict["indexes"][idx]) == 1:
                continue
            valid_edges_curr = (
                torch.where(predictions_dict["pose_scores"][idx][0] > 0)[0]
            ).tolist()
            if (
                len(valid_edges_curr)
                == predictions_dict["pose_scores"][idx].shape[1]
            ):
                continue
            for key in predictions_dict:
                if key in _skip_keys:
                    continue
                if key == "indexes":
                    predictions_dict[key][idx] = [
                        predictions_dict[key][idx][x] for x in valid_edges_curr
                    ]
                    continue
                container = predictions_dict[key]
                # Star-keyed dicts (e.g. ``scores``) vs list-of-stars tensors.
                if isinstance(container, dict):
                    if idx not in container or container[idx] is None:
                        continue
                    item = container[idx]
                elif isinstance(container, list):
                    if container[idx] is None:
                        continue
                    item = container[idx]
                else:
                    continue
                if not torch.is_tensor(item) or item.ndim < 2:
                    continue
                predictions_dict[key][idx] = item[:, valid_edges_curr]

    def _global_relative_extrinsics(
        self,
        predictions_dict: dict,
        idx: int,
        global_rotations: dict[int, np.ndarray],
        global_centers: dict[int, np.ndarray] | None = None,
    ) -> torch.Tensor:
        """Build global extrinsics for star ``idx`` relative to its first index.

        Returns a ``(1, N, 3, 4)`` tensor.

        When ``global_centers`` is ``None``, the translation block is zero and
        only the rotation block is meaningful.
        """
        star_indexes = predictions_dict["indexes"][idx]
        rotations = torch.from_numpy(
            np.stack([global_rotations[i] for i in star_indexes])
        ).unsqueeze(0)
        if global_centers is not None:
            translations = torch.from_numpy(
                np.stack(
                    [
                        -global_rotations[i] @ global_centers[i].reshape(3, 1)
                        for i in star_indexes
                    ]
                )
            ).unsqueeze(0)
        else:
            translations = torch.zeros(
                (1, len(star_indexes), 3, 1), dtype=rotations.dtype
            )
        extrinsics = torch.cat([rotations, translations], dim=-1)
        return restore_identity(extrinsics)

    @staticmethod
    def _rotation_errors(
        rotations_global: torch.Tensor,
        rotations_local: torch.Tensor,
    ) -> torch.Tensor:
        """Per-pair geodesic rotation error in rad for ``(N, 3, 3)`` inputs."""
        diff = (
            rotations_global.double()
            @ rotations_local.transpose(-1, -2).cpu().double()
        )
        return torch.acos(
            torch.clamp(
                (torch.einsum("bii -> b", diff) - 1) / 2,
                min=-1.0 + 1e-6,
                max=1.0 - 1e-6,
            )
        )

    def _filter_invalid_edges(
        self,
        predictions_dict: dict,
        global_rotations: dict[int, np.ndarray],
    ) -> list[tuple[int, int, float, torch.Tensor]]:
        """Zero out ``pose_scores`` for edges inconsistent with global solution.

        An edge ``(i, j)`` is filtered when the relative rotation implied by
        ``global_rotations`` differs from the local rotation by more than
        ``max_rot_error`` degrees. Modifies ``predictions_dict["pose_scores"]``
        in place; returns the list of filtered ``(idx, i, score, error)``
        tuples.
        """
        indexes = range(len(predictions_dict["indexes"]))
        thres = np.deg2rad(self.max_rot_error)
        num_filtered = 0
        num_total = 0
        filtered_index = []
        for idx in indexes:
            extrinsics_global = self._global_relative_extrinsics(
                predictions_dict, idx, global_rotations
            )
            rotations_local = predictions_dict["extrinsics"][idx][0, :, :3, :3]
            errors = self._rotation_errors(
                extrinsics_global[0, :, :3, :3], rotations_local
            )
            invalid_mask = errors > thres

            invalid_mask = invalid_mask * (
                predictions_dict["pose_scores"][idx][0].cpu() > 0
            )

            num_total += rotations_local.shape[0] - 1

            if not invalid_mask.any():
                continue

            num_filtered += invalid_mask.sum().item()

            filtered_index.extend(
                [
                    (
                        idx,
                        i,
                        predictions_dict["pose_scores"][idx][0, i].item(),
                        errors[i],
                    )
                    for i in range(len(predictions_dict["indexes"][idx]))
                    if invalid_mask[i]
                ]
            )
            predictions_dict["pose_scores"][idx][0, invalid_mask] = 0.0

        if num_filtered > 0:
            logger.info(
                f"Number of filtered edges by the rotation error / total: "
                f"{num_filtered} / {num_total}"
            )
        return filtered_index

    # Filter edges by both rotation and translation consistency
    def _mark_inconsistent_edges(
        self,
        predictions_dict: dict,
        global_rotations: dict[int, np.ndarray],
        global_centers: dict[int, np.ndarray],
    ) -> None:
        """Flag edges whose rotation or translation disagrees with global pose.

        Populates ``predictions_dict["pose_inconsistent"][idx]`` with a boolean
        mask per neighbor; the mask is then consumed downstream (the scores
        themselves are not zeroed here).
        """
        indexes = range(len(predictions_dict["indexes"]))
        thres_rot = np.deg2rad(1.0)
        thres_trans = np.deg2rad(5.0)
        num_filtered = 0
        num_total = 0
        predictions_dict["pose_inconsistent"] = {}
        for idx in indexes:
            translation_local = predictions_dict["extrinsics"][idx][
                0, :, :3, 3
            ].double()

            extrinsics_global = self._global_relative_extrinsics(
                predictions_dict, idx, global_rotations, global_centers
            )
            translation_global = extrinsics_global[0, :, :3, 3]

            # Normalize
            safe_norm_global = torch.clamp(
                torch.linalg.norm(translation_global, dim=-1, keepdim=True),
                min=1e-6,
            )
            translation_global = translation_global / safe_norm_global
            safe_norm_local = torch.clamp(
                torch.linalg.norm(translation_local, dim=-1, keepdim=True),
                min=1e-6,
            )
            translation_local = translation_local / safe_norm_local

            # Compare the directions
            errors_trans = torch.acos(
                torch.clamp(
                    torch.einsum(
                        "bi,bi->b", translation_global, translation_local
                    ),
                    min=-1.0 + 1e-6,
                    max=1.0 - 1e-6,
                )
            )

            rotations_local = predictions_dict["extrinsics"][idx][0, :, :3, :3]
            errors_rot = self._rotation_errors(
                extrinsics_global[0, :, :3, :3], rotations_local
            )
            invalid_mask = (errors_rot > thres_rot) | (
                errors_trans > thres_trans
            )
            invalid_mask = invalid_mask * (
                predictions_dict["pose_scores"][idx][0].cpu() > 0
            )
            invalid_mask[0] = False  # never filter the first one

            predictions_dict["pose_inconsistent"][idx] = invalid_mask

            num_total += translation_local.shape[0] - 1

            num_filtered += invalid_mask.sum().item()

        if num_filtered > 0:
            logger.info(
                f"Number of filtered edges by the rotation error / total: "
                f"{num_filtered} / {num_total}"
            )

    def _boost_sequential_edges(
        self,
        predictions_dict: dict,
        boost_factor: float = 2.0,
    ) -> None:
        """
        Boost weights of sequential edges (neighboring frames).

        Args:
            predictions_dict: Dictionary containing pose_scores and indexes
            boost_factor: Factor to multiply the weight by (default: 2.0)
        """
        indexes = range(len(predictions_dict["indexes"]))
        seq_edges = getattr(self, "sequential_edges", set())
        num_boosted = 0

        for idx in indexes:
            center_idx = predictions_dict["indexes"][idx][0]
            for i, neighbor_idx in enumerate(predictions_dict["indexes"][idx]):
                if i == 0:
                    continue
                edge = (
                    min(center_idx, neighbor_idx),
                    max(center_idx, neighbor_idx),
                )
                if edge in seq_edges:
                    predictions_dict["pose_scores"][idx][0, i] *= boost_factor
                    num_boosted += 1

        logger.info(
            f"Boosted {num_boosted} sequential edges by factor {boost_factor}"
        )
