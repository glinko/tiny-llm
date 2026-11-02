"""Week 1 Day 1: basic tensor operations using PyTorch.

No torch.nn.Linear or torch.nn.functional.linear in the learner implementation:
the goal is to implement the underlying matrix multiplication.
"""

import torch


def softmax(x: torch.Tensor, axis: int) -> torch.Tensor:
    """Provided for Task 1. Optional bonus: implement a stable softmax."""
    return torch.softmax(x, dim=axis)


def linear(
    x: torch.Tensor,
    w: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """TODO Task 2: x[..., I] @ w[O, I].T, then optional bias[O]."""
    raise NotImplementedError("Week 1 Day 1 Task 2: implement linear")
