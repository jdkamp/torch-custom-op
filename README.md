# torch-custom-op

Fused RMSNorm as a PyTorch custom operator: a C++ kernel registered with the dispatcher, a backward that is its own dispatcher op, fake kernels for tracing, and full `torch.compile` support with no graph breaks.

The operator is deliberately chosen to be simple. The goal of this project is integrating a custom op into PyTorch correctly, so that autograd, `opcheck` and `torch.compile` work with it.

## Install and usage

```bash
pip install torch
pip install -e . --no-build-isolation
```

```python
import torch, rmsnorm
y = torch.ops.rmsnorm.rmsnorm(x, weight, 1e-6)
```

## Integration

- `rmsnorm` and `rmsnorm_backward` are dispatcher ops with CPU kernels (float32/float64, one templated kernel).
- Autograd is registered with `register_autograd`. The backward is its own dispatcher op, so `torch.compile` can trace it.
- Both ops have fake kernels. `opcheck` passes for both, and `torch.compile(fullgraph=True)` covers forward and backward with dynamic batch sizes.

## Tests

```bash
pip install pytest
python -m pytest tests
```

## Limitations

- CPU only; no CUDA or MPS kernel yet.
- float32 and float64 only.
