from __future__ import annotations

import os
from typing import Optional, Tuple

import torch
import torch.distributed as dist


def infer_device_and_backend(local_rank_env: Optional[str]) -> Tuple[torch.device, str, Optional[int]]:
    """
    Infer device and backend based on availability and env-provided local rank.
    CUDA -> 'nccl', MPS/CPU -> 'gloo'.
    """
    if torch.cuda.is_available():
        local_rank = int(local_rank_env or 0)
        torch.cuda.set_device(local_rank)
        return torch.device(f"cuda:{local_rank}"), "nccl", local_rank
    if torch.backends.mps.is_available():
        return torch.device("mps"), "gloo", None
    return torch.device("cpu"), "gloo", None


def ddp_setup() -> Tuple[torch.device, int, int, int]:
    """
    Initialize process group for DDP if WORLD_SIZE>1. Returns (device, rank, world_size, local_rank).
    Uses 'gloo' for CPU/MPS and 'nccl' for CUDA per common guidance
    (see Sebastian Raschka, PyTorch in One Hour).
    """
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank_env = os.environ.get("LOCAL_RANK")

    device, backend, local_rank = infer_device_and_backend(local_rank_env)

    if world_size > 1 and not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29500")
        dist.init_process_group(backend=backend, rank=rank, world_size=world_size)

    return device, rank, world_size, (local_rank if local_rank is not None else -1)


def ddp_wrap_model(model: torch.nn.Module, device: torch.device, world_size: int, local_rank: int) -> torch.nn.Module:
    """
    Move model to device and wrap with DDP when world_size>1.
    For CUDA, pass device_ids; for CPU/MPS, use default.
    """
    model = model.to(device)
    if world_size > 1:
        if device.type == "cuda" and local_rank >= 0:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
        else:
            model = torch.nn.parallel.DistributedDataParallel(model)
    return model


def ddp_cleanup() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(rank: int) -> bool:
    return rank == 0


def shard_indices_for_rank(num_items: int, rank: int, world_size: int) -> slice:
    """Return a slice to shard a range [0, num_items) across ranks."""
    if world_size <= 1 or num_items == 0:
        return slice(0, num_items)
    per_rank = num_items // world_size
    rem = num_items % world_size
    start = rank * per_rank + min(rank, rem)
    end = start + per_rank + (1 if rank < rem else 0)
    return slice(start, end)


