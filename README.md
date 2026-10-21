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
- The CUDA kernel uses one block of 256 threads per row. Row sums are reduced with warp shuffles, then across warps with shared memory.
- Autograd is registered with `register_autograd`. The backward is its own dispatcher op, so `torch.compile` can trace it.
- Both ops have fake kernels. `opcheck` passes for both.

## Benchmark

NVIDIA T4, float32, input (8192, 4096), median of `torch.utils.benchmark.Timer`.
Reproduce with `python bench/bench.py`.

| | Forward | Forward + backward |
|---|---|---|
| Naive eager PyTorch | 3.91 ms | 16.80 ms |
| `F.rms_norm` | 1.65 ms | 5.42 ms |
| `torch.compile` (naive) | 1.14 ms | 2.87 ms |
| **This op** | **1.14 ms** | **3.85 ms** |

The runtime was improved by the following measures:

- Register caching: each thread keeps its slice of the row in registers, so
  every input is read from memory only once.
- Two-stage reduction for `grad_weight`: each block sums many rows in
  registers before writing, instead of writing a buffer as large as the input.


## Tests

```bash
pip install pytest
python -m pytest tests
```

`tests/test_cuda.py` is skipped without a CUDA environment.

## Limitations

- float32 and float64 only.
- The CUDA kernel always uses 256 threads per row, which may waste threads for short rows.
- The backward always computes both gradients; unneeded ones are not skipped.
