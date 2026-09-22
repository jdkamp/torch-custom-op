import pytest
import torch

import rmsnorm  # noqa: F401

op = torch.ops.rmsnorm.rmsnorm
EPS = 1e-6

@pytest.mark.parametrize("shape", [(8,), (3, 8), (2, 4, 16)])
def test_gradcheck(shape):
    x = torch.randn(shape, dtype=torch.float64, requires_grad=True)
    w = torch.randn(shape[-1], dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda a, b: op(a, b, EPS), (x,w))

@pytest.mark.parametrize("which", ["x", "weight"])
def test_partial_grad(which):
    x = torch.randn(3, 8, dtype=torch.float64, requires_grad=which == "x")
    w = torch.randn(8, dtype=torch.float64, requires_grad=which == "weight")
    op(x, w, EPS).sum().backward()
    assert (x.grad is not None) == (which == "x")
    assert (w.grad is not None) == (which == "weight")