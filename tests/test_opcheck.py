import pytest
import torch

import rmsnorm  # noqa: F401

EPS = 1e-6


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("shape", [(8,), (3, 8), (2, 4, 16)])
@pytest.mark.parametrize("requires_grad", [False, True])
def test_opcheck_forward(shape, dtype, requires_grad):
    x = torch.randn(shape, dtype=dtype, requires_grad=requires_grad)
    w = torch.randn(shape[-1], dtype=dtype, requires_grad=requires_grad)
    torch.library.opcheck(torch.ops.rmsnorm.rmsnorm.default, (x, w, EPS))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("shape", [(8,), (3, 8), (2, 4, 16)])
def test_opcheck_backward(shape, dtype):
    g = torch.randn(shape, dtype=dtype)
    x = torch.randn(shape, dtype=dtype)
    w = torch.randn(shape[-1], dtype=dtype)
    torch.library.opcheck(torch.ops.rmsnorm.rmsnorm_backward.default, (g, x, w, EPS))
