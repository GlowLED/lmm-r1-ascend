"""
flash_attn compatibility layer for Ascend NPU and other environments where flash_attn is not available.

This module provides fallback implementations (pure PyTorch) for functions from:
- flash_attn.bert_padding: unpad_input, pad_input, index_first_axis, rearrange
- flash_attn.utils.distributed: all_gather

When flash_attn is installed, the original implementations are used for optimal performance.
When flash_attn is not available (e.g., Ascend NPU), pure PyTorch fallbacks are used.

See docs/ascend/flash_attn_compat.md for detailed documentation.
"""

import torch
import torch.distributed as dist

# ============================================================================
# Detect flash_attn availability
# ============================================================================

FLASH_ATTN_AVAILABLE = False

try:
    from flash_attn.bert_padding import (
        index_first_axis as _fa_index_first_axis,
        pad_input as _fa_pad_input,
        unpad_input as _fa_unpad_input,
    )
    from flash_attn.utils.distributed import all_gather as _fa_all_gather

    FLASH_ATTN_AVAILABLE = True
except ImportError:
    _fa_index_first_axis = None
    _fa_pad_input = None
    _fa_unpad_input = None
    _fa_all_gather = None

# rearrange always comes from einops (flash_attn re-exports it, but einops is the source)
from einops import rearrange  # noqa: F401 — re-exported


# ============================================================================
# Fallback implementations (pure PyTorch)
# ============================================================================


def _unpad_input_fallback(hidden_states: torch.Tensor, attention_mask: torch.Tensor):
    """
    Remove padding from batched sequences.

    Equivalent to flash_attn.bert_padding.unpad_input.

    Args:
        hidden_states: (batch, seqlen, ...) — input tensor with padding
        attention_mask: (batch, seqlen) — binary mask, 1 for valid tokens, 0 for padding

    Returns:
        hidden_states_unpadded: (total_nonzero, ...) — packed non-padding tokens
        indices: (total_nonzero,) — original flat indices of non-padding tokens
        cu_seqlens: (batch + 1,) — cumulative sequence lengths (on CPU, int32)
        max_seqlen_in_batch: int — max sequence length in the batch
        seqlens_in_batch: (batch,) — per-sequence lengths
    """
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)  # (batch,)
    max_seqlen_in_batch = int(seqlens_in_batch.max().item())

    # cu_seqlens: cumulative sum with a leading 0
    cu_seqlens = torch.zeros(seqlens_in_batch.shape[0] + 1, dtype=torch.int32, device="cpu")
    cu_seqlens[1:] = torch.cumsum(seqlens_in_batch.cpu(), dim=0)

    # Flat indices of non-padding positions
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()

    # Gather non-padding tokens
    # hidden_states: (batch, seqlen, ...) -> (batch * seqlen, ...) -> index -> (total, ...)
    hidden_states_flat = hidden_states.reshape(-1, *hidden_states.shape[2:])
    hidden_states_unpadded = hidden_states_flat[indices]

    return hidden_states_unpadded, indices, cu_seqlens, max_seqlen_in_batch, seqlens_in_batch


def _pad_input_fallback(hidden_states: torch.Tensor, indices: torch.Tensor, batch: int, seqlen: int):
    """
    Pad packed (unpadded) sequences back to (batch, seqlen, ...) shape.

    Equivalent to flash_attn.bert_padding.pad_input.

    Args:
        hidden_states: (total_nonzero, ...) — packed non-padding tokens
        indices: (total_nonzero,) — original flat indices from unpad_input
        batch: batch size
        seqlen: sequence length

    Returns:
        output: (batch, seqlen, ...) — padded tensor
    """
    other_dims = hidden_states.shape[1:]
    output = torch.zeros(batch * seqlen, *other_dims, dtype=hidden_states.dtype, device=hidden_states.device)
    output[indices] = hidden_states
    return output.reshape(batch, seqlen, *other_dims)


def _index_first_axis_fallback(x: torch.Tensor, indices: torch.Tensor):
    """
    Index the first axis of a tensor with the given indices.

    Equivalent to flash_attn.bert_padding.index_first_axis (which uses a CUDA kernel
    for better performance). This fallback uses simple advanced indexing.

    Args:
        x: (total, ...) — input tensor
        indices: (n,) — indices to select

    Returns:
        output: (n, ...) — selected elements
    """
    return x[indices]


def _all_gather_fallback(tensor: torch.Tensor, group=None):
    """
    All-gather tensors along dim 0 within the given process group.

    Equivalent to flash_attn.utils.distributed.all_gather.

    Args:
        tensor: local tensor to gather
        group: process group (None for default group)

    Returns:
        gathered: concatenated tensor from all ranks along dim 0
    """
    world_size = dist.get_world_size(group=group)
    if world_size == 1:
        return tensor

    # All tensors must have the same shape
    tensor = tensor.contiguous()
    gathered_list = [torch.empty_like(tensor) for _ in range(world_size)]
    dist.all_gather(gathered_list, tensor, group=group)
    return torch.cat(gathered_list, dim=0)


# ============================================================================
# Public API: use flash_attn when available, fallback otherwise
# ============================================================================

if FLASH_ATTN_AVAILABLE:
    unpad_input = _fa_unpad_input
    pad_input = _fa_pad_input
    index_first_axis = _fa_index_first_axis
    all_gather = _fa_all_gather
else:
    unpad_input = _unpad_input_fallback
    pad_input = _pad_input_fallback
    index_first_axis = _index_first_axis_fallback
    all_gather = _all_gather_fallback


__all__ = [
    "FLASH_ATTN_AVAILABLE",
    "unpad_input",
    "pad_input",
    "index_first_axis",
    "rearrange",
    "all_gather",
]
