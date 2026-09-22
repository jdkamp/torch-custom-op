import torch
import torch._dynamo

import rmsnorm  # noqa: F401

op = torch.ops.rmsnorm.rmsnorm
EPS = 1e-6


def fn(x, w):
    return op(x, w, EPS) * 2 + 1


def test_fullgraph_forward_and_backward():
    torch._dynamo.reset()
    x = torch.randn(4, 64, requires_grad=True)
    w = torch.randn(64, requires_grad=True)
    x_ref = x.detach().clone().requires_grad_()
    w_ref = w.detach().clone().requires_grad_()

    y = torch.compile(fn, fullgraph=True)(x, w)
    y.sum().backward()

    y_ref = fn(x_ref, w_ref)
    y_ref.sum().backward()

    torch.testing.assert_close(y, y_ref)
    torch.testing.assert_close(x.grad, x_ref.grad)
    torch.testing.assert_close(w.grad, w_ref.grad)


