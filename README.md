# torch-custom-op

Fused RMSNorm as a PyTorch custom operator: CPU and CUDA kernels registered with the dispatcher, a backward that is its own dispatcher op, fake kernels for tracing, and full `torch.compile` support with no graph breaks.

The operator is deliberately chosen to be simple. The goal of this project is integrating a custom op into PyTorch correctly, so that autograd, `opcheck` and `torch.compile` work with it.

## Install and usage

```bash
pip install torch
pip install -e . --no-build-isolation
```

```python
import torch, rmsnorm

device = "cuda" if torch.cuda.is_available() else "cpu"
x = torch.randn(4, 512, device=device)
weight = torch.ones(512, device=device)
y = torch.ops.rmsnorm.rmsnorm(x, weight, 1e-6)
```

## Integration

- `rmsnorm` and `rmsnorm_backward` are dispatcher ops with CPU and CUDA kernels.
- The CUDA kernels size the block to the row: each thread handles 8 elements, so a row of `n` elements gets `ceil(n / 8)` threads, rounded up to a multiple of 32 and capped at 1024. Row sums are reduced with warp shuffles, then across warps with shared memory.
- The kernels fall back to 256 threads per row, reading the row from memory in each pass, if the row is longer than 8192 elements, the block is too large for the kernel's register usage, `n` is not a multiple of the vector width (16 bytes), or a tensor is not 16-byte aligned.
- Autograd is registered with `register_autograd`. The backward is its own dispatcher op, so `torch.compile` can trace it.
- Both ops have fake kernels. `opcheck` passes for both.
- Supported dtypes are float32, float64, float16 and bfloat16. For float16 and bfloat16 the kernels load and store in the input dtype but compute in higher precision (float32 on CUDA, float64 on CPU).

## Benchmark

NVIDIA T4, input (8192, 4096), median of `torch.utils.benchmark.Timer`.
Reproduce with `python bench/bench.py`. `x.clone()` reads and writes the input once, so it is the reference for the forward.

**float32**

| | Forward | Forward + backward |
|---|---|---|
| `x.clone()` (reference)   | 1.15 ms     | |
| Naive eager PyTorch       | 3.90 ms     | 16.75 ms |
| `F.rms_norm`              | 1.64 ms     | 5.42 ms |
| `torch.compile` (naive)   | 1.14 ms     | 2.87 ms |
| **This op**               | **1.14 ms** | **2.81 ms** |

**float16**

| | Forward | Forward + backward |
|---|---|---|
| `x.clone()` (reference)   | 0.57 ms     | |
| Naive eager PyTorch       | 2.22 ms     | 8.73 ms |
| `F.rms_norm`              | 0.75 ms     | 2.70 ms |
| `torch.compile` (naive)   | 0.55 ms     | 1.45 ms |
| **This op**               | **0.55 ms** | **1.37 ms** |

The runtime was improved by the following measures:

- Register caching: each thread keeps its slice of the row in registers, so every input is read from memory only once.
- Two-stage reduction for `grad_weight`: the backward launches as many blocks as fit on the GPU at once. Each block sums many rows in registers and writes one row of partial sums, which are then summed into the final gradient.
- Vectorized memory access: each thread loads and stores 16 bytes at a time instead of one element. This made float16 faster and left the float32 forward unchanged, since it was already at the bandwidth limit.

## Tests

```bash
pip install pytest
python -m pytest tests
```

`tests/test_cuda.py` is skipped without a CUDA environment.
