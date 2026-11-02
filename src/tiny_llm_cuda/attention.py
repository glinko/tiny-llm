"""Week 1 Day 1 student exercises: scaled attention and multi-head attention.

Read docs/CUDA_WEEK1_DAY1.md before editing. Do not use
torch.nn.functional.scaled_dot_product_attention to solve Task 1;
it is a correctness oracle in the tests, not the learner implementation.
"""

import torch

from .basics import linear, softmax


def scaled_dot_product_attention_simple(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float | None = None,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """TODO Task 1: softmax((Q @ K^T) * scale + optional_mask) @ V.

    Q, K, V: [..., L, D]. The scale defaults to 1 / sqrt(D).
    Apply the mask by *addition* to the attention scores, if supplied.
    The function must support arbitrary leading batch dimensions.
    """
    raise NotImplementedError("Week 1 Day 1 Task 1: implement attention")


class SimpleMultiHeadAttention:
    """TODO Task 2: Q/K/V projections, per-head attention, output projection."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        wq: torch.Tensor,
        wk: torch.Tensor,
        wv: torch.Tensor,
        wo: torch.Tensor,
    ):
        # Task 2: validate hidden_size / num_heads, store weights and scale.
        raise NotImplementedError("Week 1 Day 1 Task 2: initialize multi-head attention")

    def __call__(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Task 2: [B,L,E] -> [B,H,L,D] -> [B,L,E].
        raise NotImplementedError("Week 1 Day 1 Task 2: apply multi-head attention")
