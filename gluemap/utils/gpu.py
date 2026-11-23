import argparse
import datetime
import logging
import os
import pickle
import shutil
from typing import Any

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)

# Barriers here coordinate file-system handoffs between ranks, not tensor
# collectives. Inference ranks can drift far apart (per-pair cost varies, so a
# rank that draws cheap pairs finishes tens of minutes ahead), and the default
# 10-minute process-group timeout turns that drift into a watchdog abort that
# kills a run at 99% progress. Override with GLUEMAP_DIST_TIMEOUT_MIN.
_DEFAULT_DIST_TIMEOUT_MIN = 480

_CPU_GROUP: Any = None


def _dist_timeout() -> datetime.timedelta:
    """Return the collective timeout, overridable via env var."""
    try:
        minutes = int(
            os.environ.get(
                "GLUEMAP_DIST_TIMEOUT_MIN", _DEFAULT_DIST_TIMEOUT_MIN
            )
        )
    except ValueError:
        minutes = _DEFAULT_DIST_TIMEOUT_MIN
    return datetime.timedelta(minutes=max(minutes, 1))


def init_distributed_mode(args: argparse.Namespace) -> None:
    """Initialize ``torch.distributed`` from environment variables.

    Reads ``RANK``, ``WORLD_SIZE``, and ``LOCAL_RANK`` from the environment
    (the standard handshake set up by ``torchrun``/Slurm) and calls
    ``init_process_group`` with the NCCL backend. ``args`` is mutated to
    record the chosen ``rank``, ``world_size``, ``gpu``, ``dist_backend``,
    and ``distributed`` flag.

    The function bails out early — leaving ``args.distributed = False`` —
    when ``args.cpu`` is set, when ``args.nodist`` is truthy, or when the
    expected env vars are absent.

    Sets ``NCCL_NET=Socket`` and ``NCCL_IB_DISABLE=1`` so NCCL falls back to
    TCP sockets on clusters without the OFI/InfiniBand stack (e.g. CSCS
    Alps without the AWS OFI NCCL plugin).
    """
    if getattr(args, "cpu", False):
        logger.info(
            "Running on CPU (--cpu flag set), distributed mode disabled"
        )
        setup_for_distributed(is_master=True)
        args.distributed = False
        return

    nodist = args.nodist if hasattr(args, "nodist") else False
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ and not nodist:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.gpu = int(os.environ["LOCAL_RANK"])
    else:
        logger.info("Not using distributed mode")
        setup_for_distributed(is_master=True)  # hack
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = "nccl"

    # Fall back to TCP sockets when InfiniBand / OFI (libfabric) are
    # unavailable (e.g. CSCS/Alps Slingshot nodes without the AWS OFI NCCL
    # plugin). Safe for inference pipelines where high-throughput collectives
    # are not needed.
    os.environ["NCCL_NET"] = "Socket"
    os.environ["NCCL_IB_DISABLE"] = "1"

    logger.info(
        f"| distributed init (rank {args.rank}): {args.dist_url}, "
        f"gpu {args.gpu}"
    )
    timeout = _dist_timeout()
    torch.distributed.init_process_group(
        backend=args.dist_backend,
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=args.rank,
        timeout=timeout,
    )

    # Route straggler-prone barriers through gloo: a timeout there raises a
    # plain error instead of tearing down the NCCL communicator (and with it
    # the whole process) the way the NCCL watchdog does.
    global _CPU_GROUP
    try:
        _CPU_GROUP = torch.distributed.new_group(
            backend="gloo", timeout=timeout
        )
    except Exception as exc:  # pragma: no cover - gloo missing in odd builds
        logger.warning(f"gloo group unavailable ({exc}); barriers use NCCL")
        _CPU_GROUP = None

    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)


def setup_for_distributed(is_master: bool) -> None:
    """Suppress info-level logging on non-master processes."""
    if not is_master:
        logging.getLogger().setLevel(logging.WARNING)


def is_dist_avail_and_initialized() -> bool:
    """Return ``True`` only when ``torch.distributed`` is available and
    initialized."""
    if not dist.is_available():
        return False
    return dist.is_initialized()


def get_world_size() -> int:
    """Return the world size (1 outside distributed mode)."""
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank() -> int:
    """Return the current rank (0 outside distributed mode)."""
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def all_gather_object_cpu(  # type: ignore
    data: Any,
    tmpdir: None | str = None,
    rank_zero_return_only: bool = True,
    use_system_tmp: bool = False,
) -> list[Any] | None:  # pragma: no cover
    """Share arbitrary picklable data via file system caching.

    Args:
        data: any picklable object.
        tmpdir: Save path for temporary files, on a filesystem shared by all
            ranks. Required.
        rank_zero_return_only: if results should only be returned on rank 0.
        use_system_tmp: if use system tmpdir or not.

    Returns:
        list[Any]: list of data gathered from each process.
    """
    rank, world_size = get_rank(), get_world_size()
    if world_size == 1:
        return [data]

    if tmpdir is None:
        # Auto-creating one per rank would silently break multi-node runs,
        # where this handoff needs a shared filesystem path.
        raise ValueError(
            "all_gather_object_cpu requires an explicit tmpdir on a "
            "filesystem shared by all ranks"
        )

    # Scope the directory per job so two jobs sharing a temp_path cannot
    # delete each other's part_*.pkl. Derived from env rather than broadcast,
    # so ranks cannot diverge. Both parts are needed: TORCHELASTIC_RUN_ID
    # separates c10d rendezvous jobs but is the constant "none" under
    # torchrun's default static rendezvous, where only MASTER_PORT differs.
    job_id = "{}_{}".format(
        os.environ.get("TORCHELASTIC_RUN_ID") or "none",
        os.environ.get("MASTER_PORT") or "noport",
    )
    work_dir = os.path.join(tmpdir, f"gather_{job_id}")

    # Start from an empty dir: parts left behind by a crashed run would
    # otherwise be silently read back as this run's results.
    if rank == 0:
        if os.path.exists(work_dir):
            logger.warning(f"stale gather dir {work_dir}, removing it.")
            shutil.rmtree(work_dir, ignore_errors=True)
        os.makedirs(work_dir, exist_ok=True)
    synchronize()

    # encode & save
    with open(os.path.join(work_dir, f"part_{rank}.pkl"), "wb") as f:
        pickle.dump(data, f)
    synchronize()

    if rank_zero_return_only and rank != 0:
        return None

    # load & decode
    data_list = []
    for i in range(world_size):
        with open(os.path.join(work_dir, f"part_{i}.pkl"), "rb") as f:
            data_list.append(pickle.load(f))

    # remove dir
    if not rank_zero_return_only:
        synchronize()
        if rank == 0:
            shutil.rmtree(work_dir, ignore_errors=True)
        synchronize()
    elif rank == 0:
        shutil.rmtree(work_dir, ignore_errors=True)

    return data_list


def synchronize() -> None:  # pragma: no cover
    """Sync (barrier) among all processes when using distributed training."""
    if not is_dist_avail_and_initialized():
        return
    if get_world_size() == 1:
        return

    if _CPU_GROUP is not None:
        dist.barrier(group=_CPU_GROUP)
    elif torch.cuda.is_available():
        dist.barrier(device_ids=[torch.cuda.current_device()])
    else:
        dist.barrier()


def init_distributed(
    args: argparse.Namespace,
) -> tuple[int, int, torch.device, torch.dtype]:
    """Initialize distributed mode and return rank, world_size, device, and
    dtype."""
    init_distributed_mode(args)
    rank = get_rank()
    world_size = get_world_size()

    # Resolve device without loading model weights (previously called
    # load_models(keys=set()) which still pulled in chosen_model / π³ and
    # could OOM on a shared GPU before any real work started).
    if getattr(args, "cpu", False):
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32 if device.type == "cpu" else torch.bfloat16

    return rank, world_size, device, dtype
