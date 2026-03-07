"""
Device compatibility utilities for Ascend NPU / CUDA GPU.

Provides a unified interface so that the rest of the codebase can call
device-agnostic helpers instead of torch.cuda.* directly.
"""

import torch


def _npu_available() -> bool:
    return hasattr(torch, "npu") and torch.npu.is_available()


def _cuda_available() -> bool:
    return torch.cuda.is_available()


# ---- device helpers ----

def current_device():
    """Return the current accelerator device index (works on both CUDA and NPU)."""
    if _npu_available():
        return torch.npu.current_device()
    return torch.cuda.current_device()


def set_device(device):
    """Set the current accelerator device."""
    if _npu_available():
        torch.npu.set_device(device)
    else:
        torch.cuda.set_device(device)


def device_count() -> int:
    """Return the number of available accelerator devices."""
    if _npu_available():
        return torch.npu.device_count()
    return torch.cuda.device_count()


def empty_cache():
    """Free unused memory on the current accelerator."""
    if _npu_available():
        torch.npu.empty_cache()
    else:
        torch.cuda.empty_cache()


def synchronize():
    """Wait for all kernels on the current accelerator to finish."""
    if _npu_available():
        torch.npu.synchronize()
    else:
        torch.cuda.synchronize()


def get_default_backend() -> str:
    """Return the default distributed backend for the current accelerator."""
    if _npu_available():
        return "hccl"
    return "nccl"
