"""Verify the local PyTorch CUDA environment before starting GPU exercises."""

from __future__ import annotations

import sys


def main() -> int:
    try:
        import torch
    except ImportError:
        print("ERROR: PyTorch is not installed. See docs/CUDA_START_HERE.md.")
        return 1

    print(f"Python: {sys.version.split()[0]}")
    print(f"PyTorch: {torch.__version__}")
    print(f"PyTorch CUDA runtime: {torch.version.cuda}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        print("ERROR: NVIDIA CUDA is unavailable. Check nvidia-smi and install a")
        print("compatible CUDA-enabled PyTorch wheel from pytorch.org/get-started/locally/")
        print("You can still run the Week 1 Day 1 CPU tests.")
        return 1

    try:
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        capability = torch.cuda.get_device_capability(0)
        print(f"Compute capability: {capability[0]}.{capability[1]}")
        a = torch.randn((64, 64), device="cuda")
        b = torch.randn((64, 64), device="cuda")
        result = a @ b
        torch.cuda.synchronize()
        assert result.isfinite().all().item(), "GPU output contains NaN/Inf"
        print("GPU matmul smoke test: PASS")
        if capability != (8, 6):
            print("NOTE: RTX 3090 normally reports compute capability 8.6.")
        return 0
    except (RuntimeError, AssertionError) as error:
        print(f"ERROR: CUDA smoke test failed: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
