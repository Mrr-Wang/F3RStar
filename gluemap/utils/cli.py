import argparse
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def deep_merge(base: dict, override: dict) -> dict:
    """Deep merge two dictionaries, with override taking precedence."""
    result = deepcopy(base)
    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def load_config(config_path: str) -> dict[str, Any]:
    """Load a YAML config file with support for inheritance via ``_base_``.

    The base path is resolved relative to the config file's directory and
    loaded recursively, so chained inheritance is supported. Child values
    deep-merge over the base.
    """
    path = Path(config_path)
    with open(path) as f:
        config = yaml.safe_load(f) or {}

    if "_base_" in config:
        base_path = path.parent / config["_base_"]
        base_config = load_config(str(base_path))
        del config["_base_"]
        config = deep_merge(base_config, config)

    return config


def get_args_parser() -> argparse.ArgumentParser:
    """Build the argparse parser for the GLUEMAP demo pipeline."""
    parser = argparse.ArgumentParser("Distributed Demo Pipeline", add_help=True)

    parser.add_argument(
        "--chosen_model",
        default="pi3",
        choices=["pi3", "pi3x", "vggt", "map_anything"],
        help="which model to use for multi-view pose estimation",
    )

    parser.add_argument(
        "--path_feedforward",
        default="",
        type=str,
        help="path to the chosen feedforward model checkpoint",
    )
    parser.add_argument(
        "--path_retrieval",
        default="",
        type=str,
        help="path to the retrieval model",
    )
    parser.add_argument(
        "--path_tracker", default="", type=str, help="path to the tracker model"
    )
    parser.add_argument(
        "--path_dg", default="", type=str, help="path to the doppelganger model"
    )

    # IO
    parser.add_argument(
        "--write_path",
        default="results/",
        type=str,
        help="directory to write the results",
    )
    parser.add_argument(
        "--images_path",
        default=None,
        type=str,
        help="path to the images folder",
    )

    # Input layout
    parser.add_argument(
        "--is_sequential",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="whether the images are sequentially ordered",
    )
    parser.add_argument(
        "--sample_frequency",
        type=int,
        default=1,
        help="frequency to sample images if sequential",
    )
    parser.add_argument(
        "--is_multi_sequence",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "treat images_path as a parent dir whose subfolders matching "
            "--subfolder_regex are individual sequences"
        ),
    )
    parser.add_argument(
        "--subfolder_regex",
        type=str,
        default=None,
        help=(
            "regex matched against subfolder names under images_path when "
            "--is_multi_sequence is set (e.g. '^ios.*' for LaMAR)"
        ),
    )

    parser.add_argument(
        "--num_track_per_img",
        default=1024,
        type=int,
        help="number of tracks per image to track",
    )

    parser.add_argument(
        "--max_num_tracks",
        default=None,
        type=int,
        help=(
            "maximum number of tracks before bundle adjustment "
            "(None = unlimited)"
        ),
    )

    parser.add_argument(
        "--camera_model",
        default="SIMPLE_PINHOLE",
        type=str,
        help="camera model to use",
    )

    parser.add_argument(
        "--intrinsics_mode",
        choices=["SHARED", "PER_FOLDER", "PER_CAMERA"],
        default="SHARED",
        help=(
            "intrinsics-bucketing strategy: SHARED (one camera per unique "
            "image shape), PER_FOLDER (per dirname x shape), or PER_CAMERA "
            "(one camera per image)"
        ),
    )

    parser.add_argument(
        "--num_neighbors",
        default=100,
        type=int,
        help="number of neighbors to establish",
    )

    parser.add_argument(
        "--num_neighbors_sequential",
        default=30,
        type=int,
        help="number of neighbors to establish",
    )

    parser.add_argument(
        "--temp_path",
        default="./tmp",
        type=str,
        help="temp path, for collecting results from multiple GPUs",
    )
    parser.add_argument(
        "--save_result",
        default=True,
        type=bool,
        help="force to discard the results",
    )

    parser.add_argument(
        "--valid_pose_threshold",
        default=0.05,
        type=float,
        help=(
            "mininum threshold for valid pose. 0.05 means that if larger "
            "than 5 percent of points are covisible, it is valid"
        ),
    )

    parser.add_argument(
        "--num_workers",
        default=4,
        type=int,
        help="number of workers for data loading",
    )
    parser.add_argument(
        "--batch_size",
        default=30,
        type=int,
        help="batch size for two view inference",
    )
    parser.add_argument(
        "--dist_url",
        default="env://",
        help="url used to set up distributed training",
    )
    parser.add_argument(
        "--valid_dg_threshold",
        default=0.8,
        type=float,
        help="threshold for dg matching",
    )

    parser.add_argument(
        "--retrieval_batch_size",
        default=30,
        type=int,
        help="batch size for retrieval",
    )

    parser.add_argument(
        "--force_load",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="force load the precomputed results",
    )

    parser.add_argument(
        "--rerun_from",
        default=None,
        type=str,
        choices=["retrieval", "twoview", "star"],
        help=(
            "force rerun from a specific pipeline stage "
            "(deletes cached files from that stage onward)"
        ),
    )

    parser.add_argument(
        "--coarse_only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="only output coarse results, skip all refinement steps",
    )

    parser.add_argument(
        "--use_dummy_tracks",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "emit dummy tracks instead of running the VGGSfM tracker in "
            "star inference"
        ),
    )

    parser.add_argument(
        "--star_ratio",
        default=1.0,
        type=float,
        help=(
            "Budget fraction for the legacy weighted selector "
            "(--star_selection infostar); the default coverage selector "
            "uses the q target instead"
        ),
    )

    parser.add_argument(
        "--adaptive_stop",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Legacy weighted selector: stop ConnectedQCover as soon as the pre-star pose "
            "graph is connected and every image is covered >= "
            "info_star_q_min times, instead of (or in addition to, "
            "whichever triggers first) hitting the --star_ratio budget cap"
        ),
    )

    parser.add_argument(
        "--star_selection",
        default=None,
        choices=["infostar", "greedy"],
        help="Star selection mode (concrete default withheld until paper acceptance)",
    )

    parser.add_argument(
        "--refine_intrinsics",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Allow BA to optimize camera intrinsics; disable for fixed-intrinsics comparisons",
    )

    parser.add_argument(
        "--info_star_q_min",
        default=None,
        type=int,
        help="InfoStar minimum per-image coverage target (default withheld until paper acceptance)",
    )

    parser.add_argument(
        "--info_star_bridge_weight",
        default=0.0,
        type=float,
        help="Legacy weighted selector's bridge gain; unused by default coverage selector",
    )

    parser.add_argument(
        "--info_star_q_soft_max",
        default=None,
        type=int,
        help=(
            "Legacy weighted selector: coverage above this is not credited "
            "toward a candidate star's greedy selection gain (diminishing "
            "returns, spreads budget instead of stacking redundancy)"
        ),
    )

    parser.add_argument(
        "--posterior_star_select",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "InfoStar posterior star selection: after the main star pass, "
            "iteratively read cross_star_dispersion and run budget-capped "
            "extra stars over high-dispersion spoke edges that are still "
            "under-covered (< 2 selected stars), for up to "
            "--posterior_max_rounds rounds. No-op unless "
            "--star_ratio < 1.0 or --adaptive_stop reduced the candidate "
            "pool in the first place. Off by default (opt-in)"
        ),
    )
    parser.add_argument(
        "--posterior_dispersion_tau_deg",
        default=None,
        type=float,
        help=(
            "InfoStar posterior star selection: spoke/composite pairs with "
            "cross-star rotation dispersion above this (degrees) are "
            "targeted for an extra star"
        ),
    )
    parser.add_argument(
        "--posterior_max_extra_stars",
        default=2,
        type=int,
        help=(
            "InfoStar posterior star selection: hard cap on extra stars "
            "across ALL rounds combined (not per round)"
        ),
    )
    parser.add_argument(
        "--posterior_max_extra_cost_fraction",
        default=0.10,
        type=float,
        help=(
            "InfoStar posterior star selection: cap extra star-stage "
            "compute cost to this fraction of the *main-pass* selected "
            "stars' total (B5-calibrated) cost, shared across all rounds"
        ),
    )
    parser.add_argument(
        "--posterior_max_rounds",
        default=3,
        type=int,
        help=(
            "InfoStar posterior star selection: max iterative rounds of "
            "dispersion-recompute → select → infer. 1 recovers the legacy "
            "single-round behaviour"
        ),
    )

    parser.add_argument(
        "--save_dense_depth",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "DenseStar Stage A: during star inference, write each star's "
            "dense depth (FP16) + depth_conf (uint8) to "
            "{curr_path}/dense_cache/ as independent .npz sidecars. Never "
            "touches star_result.pth, so force_load of the pose pipeline "
            "remains valid. Off by default (opt-in)"
        ),
    )

    parser.add_argument(
        "--covis_symmetrize",
        default="max",
        type=str,
        choices=["max", "min", "geom"],
        help=(
            "P1-1.5: how to symmetrise the N×N covisibility matrix. "
            "'max' = legacy optimistic (default); 'min' = conservative; "
            "'geom' = geometric mean √(ab). Ablate on ETH3D/SMERF."
        ),
    )

    parser.add_argument(
        "--virtual_track_seed",
        default=42,
        type=int,
        help=(
            "P1-1.6: seed for the three virtual-track RNGs "
            "(depth noise, randperm coverage sample, multinomial fill). "
            "Makes virtual-track selection reproducible across runs."
        ),
    )

    parser.add_argument(
        "--cross_star_consistency",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Cross-star consistency (concrete default withheld until paper acceptance)"
        ),
    )
    parser.add_argument(
        "--cross_star_tau_deg",
        default=None,
        type=float,
        help="Cross-star consistency temperature (default withheld until paper acceptance)",
    )
    parser.add_argument(
        "--cross_star_hard_thresh_deg",
        default=None,
        type=float,
        help=(
            "§5 step 1 optional hard filter: zero scores with "
            "dispersion above this threshold (degrees); off when unset"
        ),
    )
    parser.add_argument(
        "--cross_star_mode",
        default="reweight",
        type=str,
        choices=["reweight", "filter", "both"],
        help="§5 step 1: reweight only, hard filter only, or both",
    )
    parser.add_argument(
        "--composite_motion_edges",
        default="off",
        type=str,
        choices=["off", "consistent", "all"],
        help=(
            "§5 step 3: inject composite (non-spoke) relative poses into "
            "rotation + similarity averaging. 'consistent' keeps only "
            "pairs with cross-star dispersion <= --cross_star_tau_deg; "
            "'all' adds every neighbour↔neighbour pair (ablation control). "
            "Default off"
        ),
    )

    parser.add_argument(
        "--skip_doppelgangers",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "skip the Doppelgangers two-view model and treat all pairs as "
            "valid (score=1.0)"
        ),
    )

    parser.add_argument(
        "--gt_intrinsics_path",
        default=None,
        type=str,
        help="path to a COLMAP reconstruction directory with GT intrinsics",
    )

    parser.add_argument(
        "--use_gt_intrinsics",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "use GT intrinsics from the ground truth reconstruction "
            "(requires gt_path in config)"
        ),
    )

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help=(
            "path to a YAML configuration file "
            "(CLI arguments override YAML values)"
        ),
    )

    parser.add_argument(
        "--cpu",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="force the entire pipeline to run on CPU (ignores available GPUs)",
    )

    parser.add_argument(
        "--use_geom_weight",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Geometry-weighted edges (concrete default withheld until paper acceptance)"
        ),
    )
    parser.add_argument(
        "--geom_theta0_deg",
        default=None,
        type=float,
        help="Geometry-weight scale in degrees (default withheld until paper acceptance)",
    )

    return parser


def parse_args_with_config(
    parser: argparse.ArgumentParser,
) -> argparse.Namespace:
    """Parse arguments with optional YAML config support.

    If --config is provided, YAML values are used as defaults;
    CLI arguments take precedence over YAML values.
    """
    args, _ = parser.parse_known_args()

    if args.config:
        yaml_config = load_config(args.config)
        flat_config = {
            k: v for k, v in yaml_config.items() if not isinstance(v, dict)
        }
        parser.set_defaults(**flat_config)

    args = parser.parse_args()

    if args.images_path is None:
        parser.error("the following arguments are required: --images_path")

    return args
