"""Attention and masking primitives used by both model families."""

from __future__ import annotations

import math

import torch
from torch import Tensor
from torch.nn import functional as F


def causal_mask(sequence_length: int, *, device: torch.device | None = None) -> Tensor:
    """Return a ``(1, 1, T, T)`` boolean mask where True means visible."""
    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive")
    return torch.ones(
        sequence_length,
        sequence_length,
        dtype=torch.bool,
        device=device,
    ).tril().unsqueeze(0).unsqueeze(0)


def padding_mask(tokens: Tensor, pad_token_id: int) -> Tensor:
    """Return a ``(B, 1, 1, T)`` key-padding mask where True means visible."""
    if tokens.ndim != 2:
        raise ValueError("tokens must have shape (batch_size, sequence_length)")
    return tokens.ne(pad_token_id).unsqueeze(1).unsqueeze(2)


def combine_masks(*masks: Tensor | None) -> Tensor | None:
    """Combine broadcast-compatible masks using logical AND."""
    present = [mask.to(dtype=torch.bool) for mask in masks if mask is not None]
    if not present:
        return None
    combined = present[0]
    for mask in present[1:]:
        combined = combined & mask
    return combined


def scaled_dot_product_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    mask: Tensor | None = None,
    *,
    dropout_probability: float = 0.0,
    training: bool = False,
) -> tuple[Tensor, Tensor]:
    """Apply scaled dot-product attention to tensors shaped ``(B, H, T, D)``."""
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError("query, key, and value must have shape (B, H, T, D)")
    if query.size(-1) != key.size(-1):
        raise ValueError("query and key head dimensions must match")
    if key.size(-2) != value.size(-2):
        raise ValueError("key and value sequence lengths must match")

    scores = query @ key.transpose(-2, -1)
    scores = scores / math.sqrt(query.size(-1))
    if mask is not None:
        visible = mask.to(device=scores.device, dtype=torch.bool)
        scores = scores.masked_fill(~visible, float("-inf"))

    weights = F.softmax(scores, dim=-1)
    # A fully masked query has no valid probability distribution. Treat it as
    # zero context instead of allowing softmax(-inf, ...) to propagate NaNs.
    weights = torch.nan_to_num(weights, nan=0.0)
    weights = F.dropout(weights, p=dropout_probability, training=training)
    return weights @ value, weights

