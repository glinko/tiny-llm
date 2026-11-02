"""Week 1 Day 1 CUDA lesson: run with PYTHONPATH=src python -m pytest.

By default tests target the student's TODO modules and should fail initially.
To verify the completed reference implementation first:
    TINY_LLM_CUDA_SOLUTION=ref PYTHONPATH=src python -m pytest -q tests_cuda/test_week_1_day_1.py

Do not copy the reference into the student implementation: implement the exercises.
"""

import importlib
import os

import pytest
import torch
import torch.nn.functional as F


def _load_solution():
    choice = os.environ.get("TINY_LLM_CUDA_SOLUTION", "learner")
    if choice not in {"learner", "ref"}:
        raise ValueError("TINY_LLM_CUDA_SOLUTION must be 'learner' or 'ref'")
    pkg = "tiny_llm_cuda_ref" if choice == "ref" else "tiny_llm_cuda"
    return importlib.import_module(f"{pkg}.basics"), importlib.import_module(
        f"{pkg}.attention"
    )


basics, attention = _load_solution()
CASES = [pytest.param("cpu", torch.float32, id="cpu-fp32")]
if torch.cuda.is_available():
    CASES += [
        pytest.param("cuda", torch.float32, id="cuda-fp32"),
        pytest.param("cuda", torch.float16, id="cuda-fp16"),
        pytest.param("cuda", torch.bfloat16, id="cuda-bf16"),
    ]


def _tolerance(dtype: torch.dtype) -> tuple[float, float]:
    if dtype == torch.float16:
        return 3e-2, 3e-2
    if dtype == torch.bfloat16:
        return 5e-2, 5e-2
    return 5e-4, 5e-4


@pytest.mark.parametrize("device,dtype", CASES)
def test_task_1_softmax_is_supplied(device, dtype):
    x = torch.tensor([[1.0, 0.0, -1.0]], dtype=dtype, device=device)
    torch.testing.assert_close(basics.softmax(x, axis=-1), F.softmax(x, dim=-1))


@pytest.mark.parametrize("device,dtype", CASES)
@pytest.mark.parametrize("leading", [(), (2,), (2, 3)], ids=["unbatched", "batch", "multi-batch"])
def test_task_1_attention(device, dtype, leading):
    torch.manual_seed(123)
    q = torch.randn(*leading, 4, 8, device=device, dtype=dtype)
    k = torch.randn(*leading, 4, 8, device=device, dtype=dtype)
    v = torch.randn(*leading, 4, 8, device=device, dtype=dtype)

    expected = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
    actual = attention.scaled_dot_product_attention_simple(q, k, v)
    assert actual.shape == q.shape
    atol, rtol = _tolerance(dtype)
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)


@pytest.mark.parametrize("device,dtype", CASES)
def test_task_1_attention_scale_mask(device, dtype):
    torch.manual_seed(124)
    q, k, v = (torch.randn(2, 3, 4, 8, device=device, dtype=dtype) for _ in range(3))
    mask = torch.tensor(
        [[0.0, -0.5, -1.0, -1.5], [-0.5, 0.0, -0.5, -1.0],
         [-1.0, -0.5, 0.0, -0.5], [-1.5, -1.0, -0.5, 0.0]],
        device=device, dtype=dtype,
    )
    expected = F.scaled_dot_product_attention(
        q, k, v, attn_mask=mask, scale=0.5, dropout_p=0.0
    )
    actual = attention.scaled_dot_product_attention_simple(
        q, k, v, scale=0.5, mask=mask
    )
    atol, rtol = _tolerance(dtype)
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)


@pytest.mark.parametrize("device,dtype", CASES)
@pytest.mark.parametrize("with_bias", [False, True])
def test_task_2_linear(device, dtype, with_bias):
    torch.manual_seed(125)
    x = torch.randn(2, 3, 8, device=device, dtype=dtype)
    w = torch.randn(6, 8, device=device, dtype=dtype)
    bias = torch.randn(6, device=device, dtype=dtype) if with_bias else None
    expected = F.linear(x, w, bias)
    actual = basics.linear(x, w, bias)
    atol, rtol = _tolerance(dtype)
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)


@pytest.mark.parametrize("device,dtype", CASES)
def test_task_2_simple_multi_head_attention(device, dtype):
    torch.manual_seed(126)
    b, length, emb, heads = 2, 4, 8, 2
    q = torch.randn(b, length, emb, device=device, dtype=dtype)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    wq, wk, wv, wo = (torch.randn(emb, emb, device=device, dtype=dtype) for _ in range(4))
    mask = torch.triu(
        torch.full((length, length), float("-inf"), device=device, dtype=dtype),
        diagonal=1,
    )

    def split_heads(x, w):
        return F.linear(x, w).reshape(b, length, heads, emb // heads).transpose(1, 2)

    expected_attention = F.scaled_dot_product_attention(
        split_heads(q, wq), split_heads(k, wk), split_heads(v, wv),
        attn_mask=mask, dropout_p=0.0,
    )
    expected = F.linear(
        expected_attention.transpose(1, 2).reshape(b, length, emb), wo
    )
    layer = attention.SimpleMultiHeadAttention(emb, heads, wq, wk, wv, wo)
    actual = layer(q, k, v, mask=mask)
    assert actual.shape == (b, length, emb)
    atol, rtol = _tolerance(dtype)
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
