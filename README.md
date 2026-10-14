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
