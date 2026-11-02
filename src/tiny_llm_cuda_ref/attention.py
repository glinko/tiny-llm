"""Completed Week 1 Day 1 PyTorch reference implementation."""

import torch

from .basics import linear, softmax


def scaled_dot_product_attention_simple(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float | None = None,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    factor = query.shape[-1] ** -0.5 if scale is None else scale
    scores = torch.matmul(query, key.transpose(-2, -1)) * factor
    if mask is not None:
        scores = scores + mask
    return torch.matmul(softmax(scores, axis=-1), value)


class SimpleMultiHeadAttention:
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        wq: torch.Tensor,
        wk: torch.Tensor,
        wv: torch.Tensor,
        wo: torch.Tensor,
    ):
        if num_heads <= 0 or hidden_size % num_heads:
            raise ValueError("hidden_size must be divisible by a positive num_heads")
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim ** -0.5
        if any(w.shape != (hidden_size, hidden_size) for w in (wq, wk, wv, wo)):
            raise ValueError("all projection weights must be [E, E]")
        self.wq, self.wk, self.wv, self.wo = wq, wk, wv, wo

    def __call__(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch, length, dim = query.shape
        if query.shape != key.shape or query.shape != value.shape:
            raise ValueError("Day 1 assumes Q, K, V have identical shape")
        if dim != self.hidden_size:
            raise ValueError("input hidden size does not match weights")

        def heads(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
            return linear(x, w).reshape(
                batch, length, self.num_heads, self.head_dim
            ).transpose(1, 2)

        q, k, v = heads(query, self.wq), heads(key, self.wk), heads(value, self.wv)
        combined = scaled_dot_product_attention_simple(
            q, k, v, scale=self.scale, mask=mask
        )
        merged = combined.transpose(1, 2).reshape(batch, length, dim)
        return linear(merged, self.wo)
