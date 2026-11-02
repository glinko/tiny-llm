# CUDA / RTX 3090 port plan

Status: **planning only; no CUDA implementation or validation has been performed.**  
Target repository: `glinko/tiny-llm` (fork of `skyzh/tiny-llm`).  
Baseline: fork `main` at `17a8cc660b1547bd42dfe53b01e9c76410f4cf8c` (2026-10-31).  
Target hardware: Ubuntu Linux, NVIDIA GeForce RTX 3090, 24 GiB-class advertised VRAM (24 GB), Ampere **SM 8.6**.  
Primary purpose: keep the original **hands-on course** intact on Apple MLX, and build a parallel, testable **PyTorch + CUDA → Triton → optional CUDA C++** path for NVIDIA.

> **Scope clarification:** This is an **inference and LLM-serving** course, not a pretraining or fine-tuning course. "Training on my GPU" means completing and benchmarking the exercises locally. Model fine-tuning, cross-request prefix caching, and quantized KV are not MVP goals.

## Design principles

1. **Keep upstream MLX source, tests, docs, and Metal extension paths untouched initially.** Never replace existing `src/tiny_llm` or `src/tiny_llm_ref` in place; permit straightforward upstream sync.
2. **Maintain the learning contract.** Each CUDA lesson has (a) a student implementation with TODOs, (b) a completed CUDA reference implementation, (c) tests, and (d) a benchmark. Do not substitute a complete `transformers` forward pass for the educational work.
3. **Use PyTorch/CUDA as the readable correctness baseline, and Triton as the exercise kernel language.** CUDA C++ is an optional advanced track rather than an initial dependency.
4. **Do not attempt a global `mlx.core` compatibility shim.** Semantics differ for tensor types, `.astype`/`.to`, `mx.eval`, `mx.stream`, indexing and `Tensor.size`. Port a small set of explicit operators and data contracts.
5. **Separate inference from package installation.** macOS-only MLX dependencies and build steps must not be required on Ubuntu. Avoid breaking upstream PDM lock or GitHub Actions while establishing the CUDA environment.
6. **Use one shared checkpoint on both sides of a comparison.** The existing MLX-4bit repository weights are not bitwise/interchangeably the same as Hugging Face BF16 weights or arbitrary bitsandbytes/torchao 4-bit layouts.
7. **Tests first, optimizations second.** Preserve exact shape, mask, attention, offset, and cache invariants. Measure only after warmup and explicit CUDA synchronization.

## Proposed repository layout

```text
src/
  tiny_llm/                 # upstream MLX learner exercises: unchanged
  tiny_llm_ref/             # upstream MLX completed reference: unchanged
  extensions/               # upstream Metal learner exercises: unchanged
  extensions_ref/           # upstream Metal reference: unchanged
  tiny_llm_cuda/            # NEW CUDA learner exercises (TODOs per lesson)
    __init__.py
    basics.py
    attention.py
    positional_encoding.py
    layer_norm.py
    embedding.py
    sampler.py
    qwen3_week1.py
    kv_cache.py
    quantize.py
    week2_kernels.py
    qwen3_week2.py
    paged_kv_cache.py
    qwen3_week3.py
    batch.py
    generate.py
    agent/
  tiny_llm_cuda_ref/        # NEW complete CUDA reference; same public lesson API
    ...
  cuda_support/             # NEW internal, shared model loader/config/backend helpers
    hf_loader.py
    weight_types.py
    cuda_runtime.py
  cuda_kernels/             # NEW learner/reference Triton exercises
    ...
tests_cuda/                 # NEW GPU-free and CUDA tests; do not copy MLX tests verbatim
benches_cuda/               # NEW timing, VRAM and throughput experiments
scripts/check_cuda.py
requirements-cuda.txt      # initial platform-isolated installation recipe
docs/CUDA_PORT_PLAN.md
docs/CUDA_SETUP.md          # create during environment milestone
.github/workflows/cuda-cpu.yml  # later: GPU-free PR smoke tests
```

This structure is a starting proposal; do not scaffold all weeks before their tests and requirements exist. Aim to avoid duplicating platform-independent Week 4 agent protocol/evaluation/workspace logic where straightforward.

## Inventory of actual upstream dependencies to change

| Area | Current source | MLX-specific dependency / behavior | CUDA replacement |
| --- | --- | --- | --- |
| CLI entry | `main.py`, `batch-main.py` | `mlx_lm.load`, `mlx.core`, `mx.stream`, MX sampler | New CUDA entry points (`main-cuda.py` / `batch-main-cuda.py`) and explicit device selection |
| Model weight mapping | `qwen3_week1.py`, `qwen3_week2.py`, `qwen3_week3.py` | reads `mlx_model.args`, `mlx_model.model.layers[*]`, MLX quantized layers | small PyTorch/HF Safetensors loader with config/weights adapter |
| Basic operators | `basics.py`, `embedding.py`, `layer_norm.py`, `positional_encoding.py`, `attention.py` | MX array creation, dtypes, broadcasting, matmul, RoPE, causal mask | explicit PyTorch tensor implementation with reference tests |
| Week 1 generation | `generate.py`, `sampler.py` | `mx.eval`, `mx.argmax`, `mx.logsumexp`, MLX tokenizer wrapper | PyTorch eager decode; HF tokenizer only, explicit EOS and sampler |
| Dense/continuous cache | `kv_cache.py`, `batch.py` | mutable MX arrays, MX concatenate/eval, masking and offsets | CUDA tensors, explicit cache ownership, masking, release and rewind |
| Quantized layers | `quantize.py`, `qwen3_week2.py`, `qwen3_week3.py` | MLX 4-bit packing, `scales` + `biases`, `mx.quantized_matmul`, custom Metal | first BF16/FP32 baseline; then explicit W4 pack/unpack format and Triton matvec/matmul |
| Fused kernels | `week2_kernels.py` | C++/Metal bindings for RMSNorm, RoPE, SwiGLU, matrix prefill | Triton kernels and PyTorch reference/fallback |
| Paged KV | `paged_kv_cache.py`, `attention.py` | `paged_cache_update`, `paged_attention` native Metal | PyTorch allocator + reference gather, then Triton direct-page read/attention |
| Native build | `src/extensions*/CMakeLists.txt`, `*.metal` | MLX CMake, Metal compiler, MLX-specific nanobind | NEW CUDA/Triton modules; do **not** try to compile Metal on Linux |
| Agent generation | `agent.py`, `src/tiny_llm_ref/agent/generation.py`, `branching.py` | MLX generation + KV checkpoint tensor semantics | CUDA generation adapter and cache checkpoint clone/restore |
| Testing/CI | `tests_refsol`, `tests/utils.py`, `pyproject.toml`, `macos.yml` | `mlx.core`, MLX-4bit models, Apple GPU runner | separate CUDA tests, package group and Linux CI; keep macOS CI working |
| Profiling | `benches/*`, Metal debugging commands | MLX dispatch/synchronization and Metal tools | CUDA events, `torch.cuda.synchronize`, profiler, Nsight |

### Compatibility hazards worth testing explicitly

- **Qwen3 Q/K norm** is applied before RoPE. Preserve head dimensions, GQA grouping, RoPE pairing convention, epsilon, and dtype transitions.
- Causal masking must work for **non-square attention**, such as decoding `L=1, S>1`, and offsets during chunked prefill; blindly using `is_causal=True` may align the wrong triangle.
- `torch.Tensor.size` is a method, not MLX's element-count property: use `numel()` where intended.
- CUDA operations are asynchronous; timing and correctness checkpoints must synchronize where appropriate, not imitate `mx.eval` as an unconditional no-op.
- MLX's W4 weight packing is a custom contract; verify nibble order, group sizes, zero/bias semantics, orientation and shape before using a Triton kernel.
- PyTorch/HF attention implementations may differ in numerics and kernel dispatch. Compare float32 first, and BF16 with declared tolerances and identical checkpoint tensors.
- Respect allocation limits on the RTX 3090. Report model weights, KV cache, temporary activation memory, and peak allocated/reserved separately.
- Agent file-operation safety is independent of GPU; retain explicit approval, sandbox boundaries, command allowlists, and receipt tests.

## Sequential work packages and acceptance gates

### P0 — CUDA environment and baseline

**Tasks**
- Collect real machine data: `nvidia-smi`, driver, `python --version`, `nvcc --version` if installed, VRAM free, and `torch.cuda.get_device_capability()`. No assumption that `nvcc` is mandatory for the PyTorch-only starting point.
- Use Python 3.11 or 3.12, isolated virtualenv, an official CUDA-capable PyTorch wheel selected for the installed driver, `transformers`, `safetensors`, `pytest`, `triton` (only when kernels begin), `numpy`.
- Do not install or compile `mlx`, `mlx-lm`, `*.metal`, or the upstream Metal extensions on Linux.
- Add `scripts/check_cuda.py` with readable error messages and a minimal CUDA tensor/matmul/sync check.
- Document the NVIDIA driver / PyTorch / CUDA build matrix in `docs/CUDA_SETUP.md`; avoid permanently hardcoding a wheel URL before inspecting the host.

**Acceptance:** on the target Ubuntu RTX 3090, `check_cuda.py` reports `cuda=True`, capability `(8, 6)`, a passing GPU operation, and clear version/device information. CPU smoke import works without MLX.

### P1 — Week 1: Qwen3 forward and generation

**Tasks**
- Start from `Qwen/Qwen3-0.6B` BF16/FP32 original HF checkpoint (not the `-MLX-4bit` derivative). Let `transformers` load **only** config, tokenizer, tensors, and serve as an optional **oracle**. Learner forward must use the course's own implementation.
- Implement `linear`, `softmax`, RMSNorm, SiLU/SwiGLU, RoPE, GQA, embedding, Transformer block, LM head, causal attention and sampler in `tiny_llm_cuda_ref`; add TODO versions in `tiny_llm_cuda`.
- Capture per-operation / per-layer shapes and numerical comparisons against a trusted PyTorch/HF implementation with a fixed checkpoint and prompt.
- Implement greedy text decoding, chat template, EOS handling and basic local CLI.

**Acceptance:** GPU generates coherent output from Qwen3-0.6B; one forward pass and greedy token sequence match the HF baseline under explicitly stated tolerances/conditions; Week 1 shape, mask, dtype, CPU and CUDA tests pass. Include BF16 vs FP32 comparisons.

### P2 — Week 2, readable cache checkpoints before any custom kernels

**Tasks**
- Port `TinyKvFullCache`, cache capacity, append/fetch, rewind/release, and the checkpoint selector (`kv-cache`, `capacity-cache`).
- Demonstrate a prompt with/without cache and validate matching greedy output and attention shapes.
- Implement CUDA W4 packing/dequantization, documenting *our* packed layout. Validate reconstructed values and GEMV/GEMM against dense baseline before any optimization.
- Introduce `quantized-matvec` checkpoint with explicit fallback, without calling bitsandbytes 4-bit weights `MLX-compatible`.

**Acceptance:** cache semantics and bounded capacity pass on CPU/GPU. Quantized path correctness tested with small synthetic shapes, odd bounds, group orientation and BF16 tolerances; model-level W4 generation verified once the model weights fit.

### P3 — Week 2 performance kernels

**Tasks:** add independent, testable Triton implementations for W4 GEMV, W4 GEMM / tiled prefill, RMSNorm, RoPE, SwiGLU, and optionally tiled dense prefill attention. Preserve course checkpoint names (`simd-matmul` as CUDA/Triton equivalent, `rmsnorm`, `rope`, `swiglu`, `tiled-prefill`, `selected`), but document that NVIDIA's GPU execution model differs from Apple's SIMD-groups. Compare each kernel to the readable PyTorch reference.

**Acceptance:** each optimized checkpoint produces correct results, exposes a fallback where unsupported, and has warmup, correctness and `ms/op`, tokens/s, VRAM measurement. Do not claim improvements without measured results.

### P4 — Week 3: mini-vLLM serving

**Tasks**
- Port request scheduler, continuous batching and chunked prefill with deterministic scheduler tests. Keep state transitions and exact per-request token histories.
- Port paged KV pool: page allocation, ownership, append, free-list reuse, rewind/release, block tables, context lengths. First use an explicit gather-to-dense **correctness oracle**.
- Then write *direct* paged attention and tiled/online-softmax attention in Triton. Never label a dense-gather fallback as a completed direct-page kernel.
- Port/benchmark optional MoE and greedy speculative decoding only after the dense path is stable.

**Acceptance:** tests for interleaved requests, admission boundaries, non-square masks, page reuse, exhausted pool, rewind and mixed sequence lengths; direct-page results agree with the reference; VRAM/per-request and throughput metrics recorded.

### P5 — Week 4: coding agent

**Tasks**
- Reuse the platform-neutral tool policy, workspace, receipts, evaluation and loop logic where possible; port only MLX-coupled generation and tensor-based prefix checkpoint/branching.
- Preserve explicit approvals for writes/commands, workspace-only file tools, operator steering, receipts and restart boundaries.
- Create a disposable workspace for local tests: no credentials, no personal files.

**Acceptance:** deterministic scripted-agent tests without any GPU, plus at least one constrained RTX 3090 run with real model, approval and receipt behavior. Exact file actions must not replay after checkpoint restore.

### P6 — packaging, CI, docs and baseline report

**Tasks**
- Add CUDA-only installation and launcher instructions. Later refactor PDM dependency groups/extras when the working Linux environment and macOS lock can both be tested.
- Separate `tests_cuda` from MLX tests. Run CPU-portable tests on GitHub-hosted Ubuntu runners; GPU tests locally or on a compatible self-hosted runner. GitHub public runners are not assumed to have NVIDIA GPUs.
- Provide exercise chapter crosswalk (`MLX → PyTorch baseline → Triton kernel`).
- Add a benchmark ledger: model, context, batch, prefill TTFT, decode tokens/sec, latency percentiles, peak allocated/reserved VRAM, software versions and comparisons against stock HF PyTorch and optionally vLLM on the same device.

**Acceptance:** repeatable local setup and commands, clean Mac/MLX CI (where available), CPU smoke CI, GPU test suite on RTX 3090, benchmark report with *measured* rather than estimated numbers.

## Starter command sketch (documentation, not yet verified on host)

```bash
git clone https://github.com/glinko/tiny-llm.git
cd tiny-llm
python3.11 -m venv .venv-cuda
source .venv-cuda/bin/activate
python -m pip install -U pip
nvidia-smi

# Install a compatible PyTorch CUDA wheel selected from:
# https://pytorch.org/get-started/locally/
python -c "import torch; print(torch.__version__, torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_capability() if torch.cuda.is_available() else 'no GPU')"
```

Do **not** run upstream `pdm install` on Ubuntu unchanged: its declared dependencies include `mlx` and `mlx-lm`, and its native extension build expects Metal. Actual CUDA install/CLI commands should be committed only after the environment milestone passes.

## Initial PR backlog (one independently testable unit per PR)

1. **CUDA environment**: Linux setup, import isolation, `scripts/check_cuda.py`, GPU smoke test.
2. **HF weights + tokenizer adapter**: Qwen3-0.6B config and Safetensors, explicit loader tests, no forward via `AutoModelForCausalLM` in learner code.
3. **W1 primitive lessons**: attention, RoPE, GQA, RMSNorm, MLP, tests.
4. **W1 model + decoding**: CLI, EOS and fixed-prompt logits tests.
5. **W2 cache**: cache + bounded capacity.
6. **W2 W4 correctness**: pack/dequant, dense reference, overflow/rounding tests.
7. **W2 Triton kernels**: quantitative benchmarks per operator.
8. **W3 scheduler and paged allocator**, then direct-page attention.
9. **W4 agent adapter** with isolated workspace and deterministic tests.
10. **Benchmark/CI/docs** polishing and upstream maintenance.

## Non-goals and decisions

- No changes to original `main` just to make it import under Ubuntu.
- No attempt to compile Apple's Metal kernels as CUDA; reimplement educational algorithms for the new backend.
- No broad port of all original `tests_refsol` in one PR; migrate and cross-check lesson by lesson.
- No initial deployment service, API server, fine-tuning, LoRA, or orchestration integration; those can be later projects after inference lessons pass.
- No production claims: this is a teaching system with measurable performance experiments, not a substitute for a maintained production serving engine.
- To stay in sync with skyzh upstream, keep the CUDA track additive and avoid modifying historical lesson text/paths unless separately approved.

## Reference links

- Source: https://github.com/skyzh/tiny-llm
- User fork: https://github.com/glinko/tiny-llm
- Course: https://skyzh.github.io/tiny-llm/
- NVIDIA RTX 3090 Compute Capability: https://developer.nvidia.com/cuda/gpus
- Triton compatibility: https://github.com/triton-lang/triton
- Qwen3 (Hugging Face): https://huggingface.co/docs/transformers/en/model_doc/qwen3
- PyTorch Linux installation selector: https://pytorch.org/get-started/locally/
