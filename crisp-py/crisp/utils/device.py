"""Device management utilities for CRISP."""

import logging
from typing import List, Optional, Union

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)


def get_device(
    device: Optional[Union[str, torch.device]] = None, gpu_id: Optional[int] = None
) -> torch.device:
    """
    Get the appropriate device for computation.

    Args:
        device: Device specification (e.g., 'cuda', 'cpu', torch.device)
        gpu_id: Specific GPU ID to use

    Returns:
        torch.device object
    """
    if device is not None:
        if isinstance(device, str):
            return torch.device(device)
        return device

    if torch.cuda.is_available():
        if gpu_id is not None:
            return torch.device(f"cuda:{gpu_id}")
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def get_available_gpus() -> List[int]:
    """
    Get list of available GPU IDs.

    Returns:
        List of available GPU IDs
    """
    if not torch.cuda.is_available():
        return []

    return list(range(torch.cuda.device_count()))


def get_gpu_memory_info(device_id: int = 0) -> dict:
    """
    Get GPU memory information.

    Args:
        device_id: GPU device ID

    Returns:
        Dictionary with memory information
    """
    if not torch.cuda.is_available():
        return {"available": False}

    torch.cuda.synchronize(device_id)

    return {
        "available": True,
        "device_id": device_id,
        "allocated": torch.cuda.memory_allocated(device_id),
        "cached": torch.cuda.memory_reserved(device_id),
        "total": torch.cuda.get_device_properties(device_id).total_memory,
    }


def log_device_info(device: torch.device):
    """
    Log information about the compute device.

    Args:
        device: torch.device object
    """
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        logger.info(f"Using GPU: {props.name} (Device {device.index})")
        logger.info(f"GPU Memory: {props.total_memory / 1024**3:.2f} GB")
        logger.info(f"GPU Compute Capability: {props.major}.{props.minor}")
    elif device.type == "mps":
        logger.info("Using Apple MPS device")
    else:
        logger.info("Using CPU")


def setup_distributed(
    rank: int, world_size: int, backend: str = "nccl", init_method: str = "env://"
) -> torch.device:
    """
    Set up distributed training.

    Args:
        rank: Process rank
        world_size: Total number of processes
        backend: Distributed backend ('nccl', 'gloo')
        init_method: Method for initializing process group

    Returns:
        Device for this process
    """
    if not dist.is_initialized():
        dist.init_process_group(
            backend=backend, init_method=init_method, world_size=world_size, rank=rank
        )

    # Set device for this process
    if torch.cuda.is_available():
        torch.cuda.set_device(rank)
        device = torch.device(f"cuda:{rank}")
    else:
        device = torch.device("cpu")

    logger.info(f"Initialized process {rank}/{world_size} on {device}")

    return device


def cleanup_distributed():
    """Clean up distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()


class DeviceManager:
    """
    Manager for device allocation and memory management.

    Handles device selection, memory monitoring, and distributed setup.
    """

    def __init__(
        self,
        device: Optional[Union[str, torch.device]] = None,
        distributed: bool = False,
        rank: Optional[int] = None,
        world_size: Optional[int] = None,
    ):
        """
        Initialize device manager.

        Args:
            device: Device specification
            distributed: Whether to use distributed training
            rank: Process rank for distributed training
            world_size: Total processes for distributed training
        """
        self.distributed = distributed

        if distributed and rank is not None and world_size is not None:
            self.device = setup_distributed(rank, world_size)
            self.rank = rank
            self.world_size = world_size
        else:
            self.device = get_device(device)
            self.rank = 0
            self.world_size = 1

        log_device_info(self.device)

    def to_device(self, tensor: torch.Tensor) -> torch.Tensor:
        """Move tensor to managed device."""
        return tensor.to(self.device)

    def synchronize(self):
        """Synchronize device operations."""
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def empty_cache(self):
        """Empty GPU cache."""
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    def get_memory_info(self) -> dict:
        """Get current memory information."""
        if self.device.type == "cuda":
            return get_gpu_memory_info(self.device.index)
        return {"available": False}

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        if self.distributed:
            cleanup_distributed()


def auto_select_device() -> torch.device:
    """
    Automatically select the best available device.

    Returns:
        Selected device
    """
    if torch.cuda.is_available():
        # Select GPU with most free memory
        best_gpu = 0
        max_free_memory = 0

        for gpu_id in range(torch.cuda.device_count()):
            torch.cuda.synchronize(gpu_id)
            free_memory = torch.cuda.get_device_properties(
                gpu_id
            ).total_memory - torch.cuda.memory_allocated(gpu_id)

            if free_memory > max_free_memory:
                max_free_memory = free_memory
                best_gpu = gpu_id

        device = torch.device(f"cuda:{best_gpu}")
        logger.info(
            f"Auto-selected GPU {best_gpu} with {max_free_memory / 1024**3:.2f} GB free"
        )

    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Auto-selected MPS device")

    else:
        device = torch.device("cpu")
        logger.info("Auto-selected CPU")

    return device
