"""InfoStar posterior star selection orchestration (opt-in, default off).

Multi-round flow (gluemap-execution-plan.md Sec 2, "posterior star
selection" — iterative online loop):

  1. The main (adaptive-stop / budgeted) star inference pass has already
     run, producing ``predictions_dict``.
  2. Fix a total extra-star budget from the *main-pass* selected cost
     (``posterior_max_extra_cost_fraction * base_cost``) and a total
     star-count cap (``posterior_max_extra_stars``) — both apply across
     all rounds, not per round.
  3. For up to ``posterior_max_rounds`` rounds:
       a. Compute a *read-only* preview of cross-star rotation dispersion
          on the current ``predictions_dict`` (does **not** mutate
          ``pose_scores`` — that reweighting happens once, later, inside
          :class:`gluemap.controllers.global_merger.GlobalGluer`).
       b. :func:`gluemap.estimators.info_star.select_posterior_stars`
          picks extra candidate centers for high-dispersion,
          under-covered spoke edges, under the *remaining* budget.
       c. If none selected → status ``"converged"``.
       d. Otherwise run star inference for just those extras (shallow-
          copied dataset, forced non-distributed), merge into
          ``predictions_dict``, deduct budget, continue.
  4. Stop with ``"budget-limited"`` if the shared star/cost cap is
     exhausted, or ``"round-limited"`` if ``posterior_max_rounds`` is
     hit with candidates still available.
"""

from __future__ import annotations

import argparse
import copy
import logging
import time

import torch

from gluemap.controllers.star_inference import run_star_inference
from gluemap.estimators.cross_star_consistency import (
    collect_pair_rotations,
    compute_pair_dispersions,
)
from gluemap.estimators.info_star import (
    calibrated_star_cost_seconds,
    select_posterior_stars,
)

logger = logging.getLogger(__name__)

POSTERIOR_STAR_RESULT_FILENAME = "star_result_posterior.pth"


def _posterior_round_filename(round_idx: int) -> str:
    """Per-round cache name; round 0 keeps the legacy single-round name."""
    if round_idx == 0:
        return POSTERIOR_STAR_RESULT_FILENAME
    return f"star_result_posterior_round{round_idx}.pth"


def _merge_predictions(predictions_dict: dict, extra_predictions: dict) -> None:
    """Append ``extra_predictions`` list/tensor entries onto ``predictions_dict``."""
    for key, value in list(extra_predictions.items()):
        if isinstance(value, torch.Tensor):
            extra_predictions[key] = value.cpu()
        elif isinstance(value, list):
            extra_predictions[key] = [
                v.cpu() if isinstance(v, torch.Tensor) else v for v in value
            ]

    shared_keys = set(predictions_dict) & set(extra_predictions)
    mismatched = set(predictions_dict) ^ set(extra_predictions)
    for key in shared_keys:
        predictions_dict[key] = list(predictions_dict[key]) + list(
            extra_predictions[key]
        )
    if mismatched:
        logger.warning(
            "InfoStar posterior: predictions_dict key mismatch, not "
            "merged: %s",
            mismatched,
        )


def maybe_run_posterior_stars(
    args: argparse.Namespace,
    dataset,
    predictions_dict: dict,
    device: str,
    dtype: torch.dtype,
    models: dict[str, torch.nn.Module] | None,
) -> tuple[dict, dict | None]:
    """Run InfoStar multi-round posterior star selection, if enabled.

    No-ops (returns ``predictions_dict`` unchanged, ``timing=None``) unless
    ``args.posterior_star_select`` is set AND InfoStar's ``connected_q_cover``
    actually reduced the candidate pool (``dataset.info_star_result`` is not
    ``None``) — with a one-star-per-image baseline there is no reduced
    candidate pool to draw extra stars from.

    Args:
        dataset: The exact ``BaseStarDataset`` the main pass ran over; must
            still have ``query_extractors`` live on ``device`` (call this
            *before* the caller releases them).
        predictions_dict: Main pass output; extended **in place** (and
            returned) when extra stars are selected.

    Returns:
        ``(predictions_dict, timing)``. ``timing`` includes ``status``
        (``"converged"`` / ``"budget-limited"`` / ``"round-limited"``),
        ``rounds`` (per-round detail list), and aggregate counters.
    """
    if not getattr(args, "posterior_star_select", False):
        return predictions_dict, None
    if getattr(dataset, "info_star_result", None) is None:
        logger.info(
            "posterior_star_select=True but InfoStar's ConnectedQCover "
            "never ran (star_ratio=1.0 and adaptive_stop=False); every "
            "image already has its own star, nothing to add — skipping"
        )
        return predictions_dict, None
    neighbors_all = getattr(dataset, "neighbors_all", None)
    if not neighbors_all:
        logger.warning(
            "posterior_star_select=True but dataset.neighbors_all is "
            "missing/empty (unexpected) — skipping"
        )
        return predictions_dict, None

    t0 = time.perf_counter()
    tau_deg = getattr(args, "posterior_dispersion_tau_deg", None)
    if tau_deg is None:
        raise NotImplementedError(
            "F3RStar method defaults are withheld until paper acceptance."
        )
    tau_deg = float(tau_deg)
    max_extra_stars = getattr(args, "posterior_max_extra_stars", 2)
    max_extra_stars = int(max_extra_stars) if max_extra_stars is not None else None
    max_extra_cost_fraction = float(
        getattr(args, "posterior_max_extra_cost_fraction", 0.10)
    )
    max_rounds = int(getattr(args, "posterior_max_rounds", 3))
    max_rounds = max(1, max_rounds)

    main_selected = list(dataset.info_star_result.selected)
    all_selected = list(main_selected)
    edge_scores = getattr(dataset, "edge_scores", None)

    # Fixed absolute budget from the *main-pass* cost only — does not grow
    # as posterior rounds append to ``all_selected``.
    base_cost = sum(
        calibrated_star_cost_seconds(1 + len(neighbors_all.get(c, set())))
        for c in main_selected
    )
    total_cost_budget = max_extra_cost_fraction * base_cost
    remaining_cost = total_cost_budget
    remaining_stars = max_extra_stars  # None = uncapped

    rounds: list[dict] = []
    total_extra_stars = 0
    last_n_high_disp = 0
    last_n_target = 0
    status = "converged"

    for round_idx in range(max_rounds):
        if remaining_stars is not None and remaining_stars <= 0:
            status = "budget-limited"
            break
        if remaining_cost <= 0:
            status = "budget-limited"
            break

        pair_rots = collect_pair_rotations(predictions_dict)
        dispersion = {
            pair: st.dispersion_deg
            for pair, st in compute_pair_dispersions(
                pair_rots, min_stars=2
            ).items()
        }

        result = select_posterior_stars(
            neighbors_all,
            all_selected,
            dispersion,
            edge_scores=edge_scores,
            dispersion_tau_deg=tau_deg,
            max_extra_stars=remaining_stars,
            cost_fn=calibrated_star_cost_seconds,
            cost_budget_seconds=remaining_cost,
        )
        last_n_high_disp = result.n_high_dispersion_edges
        last_n_target = result.n_target_edges

        round_info: dict = {
            "round": round_idx,
            "n_extra_stars": len(result.selected),
            "n_high_dispersion_edges": result.n_high_dispersion_edges,
            "n_target_edges": result.n_target_edges,
            "selected": list(result.selected),
        }

        if not result.selected:
            status = "converged"
            rounds.append(round_info)
            break

        new_star_arrays = dataset.build_posterior_stars(result.selected)
        if not new_star_arrays:
            status = "converged"
            round_info["n_extra_stars"] = 0
            rounds.append(round_info)
            break

        extra_dataset = copy.copy(dataset)
        extra_dataset.stars = new_star_arrays

        # Only rank 0 ever reaches this code (see call site); force a solo,
        # non-distributed sub-run so `_gather_outputs` doesn't try to collect
        # from ranks that already returned. Extra stars have no prior cache
        # and never touch the DenseStar dense cache (would collide
        # star-index keys with the main pass — see gluemap.dense.cache).
        extra_args = copy.copy(args)
        extra_args.distributed = False
        extra_args.force_load = False
        extra_args.save_dense_depth = False

        extra_predictions, extra_timing = run_star_inference(
            extra_args,
            extra_dataset,
            world_size=1,
            rank=0,
            file_name=_posterior_round_filename(round_idx),
            device=device,
            dtype=dtype,
            preloaded_models=models,
        )
        _merge_predictions(predictions_dict, extra_predictions)

        round_cost = sum(
            calibrated_star_cost_seconds(1 + len(neighbors_all.get(c, set())))
            for c in result.selected
        )
        remaining_cost = max(0.0, remaining_cost - round_cost)
        if remaining_stars is not None:
            remaining_stars -= len(result.selected)
        all_selected.extend(result.selected)
        total_extra_stars += len(result.selected)

        round_info["extra_star_inference"] = extra_timing
        round_info["round_cost_seconds"] = round_cost
        round_info["remaining_cost_seconds"] = remaining_cost
        rounds.append(round_info)

        logger.info(
            "InfoStar posterior round %d/%d: ran %d extra star(s) (%.2fs) -- "
            "predictions_dict now has %d stars "
            "(remaining budget: %.1fs / %s stars)",
            round_idx + 1,
            max_rounds,
            len(new_star_arrays),
            extra_timing.get("total", 0.0),
            len(predictions_dict.get("indexes", [])),
            remaining_cost,
            remaining_stars if remaining_stars is not None else "uncapped",
        )

        # If this was the last allowed round and we still selected something,
        # report round-limited only when budget remains (otherwise budget
        # already tells the fuller story on the next-loop break).
        if round_idx == max_rounds - 1:
            still_budgeted = remaining_cost > 0 and (
                remaining_stars is None or remaining_stars > 0
            )
            status = "round-limited" if still_budgeted else "budget-limited"

    timing = {
        "n_extra_stars": total_extra_stars,
        "n_high_dispersion_edges": last_n_high_disp,
        "n_target_edges": last_n_target,
        "n_rounds": len(rounds),
        "status": status,
        "rounds": rounds,
        "base_cost_seconds": base_cost,
        "total_cost_budget_seconds": total_cost_budget,
        "remaining_cost_seconds": remaining_cost,
        "total": time.perf_counter() - t0,
    }
    logger.info(
        "InfoStar posterior multi-round: status=%s, %d extra star(s) over "
        "%d round(s) (%.2fs)",
        status,
        total_extra_stars,
        len(rounds),
        timing["total"],
    )
    return predictions_dict, timing
