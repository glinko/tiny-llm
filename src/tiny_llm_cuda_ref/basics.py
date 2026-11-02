"""Completed Week 1 Day 1 tensor helpers."""

import torch


def softmax(x: torch.Tensor, axis: int) -> torch.Tensor:
    return torch.softmax(x, dim=axis)


def linear(
    x: torch.Tensor,
    w: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    result = torch.matmul(x, w.transpose(-2, -1))
    return result if bias is None else result + bias
