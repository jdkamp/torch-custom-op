import pytest
import torch
import pytest
import torch
import torch._dynamo


import rmsnorm  # noqa: F401

op = torch.ops.rmsnorm.rmsnorm
op_backward = torch.ops.rmsnorm.rmsnorm_backward
EPS = 1e-6

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available"
)

SHAPES = [(8,), (4, 16), (3, 5, 512), (2, 4096), (0, 8), (3, 0), (5, 33), (2, 10000)]
DTYPES = [torch.float32, torch.float64]


# Forward correctness

@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", SHAPES)
def test_matches_cpu(shape, dtype):
    x = torch.randn(shape, dtype=dtype)
    w = torch.randn(shape[-1], dtype=dtype)
    expected = op(x, w, EPS)
    actual = op(x.cuda(), w.cuda(), EPS)
    assert actual.device.type == "cuda"
    torch.testing.assert_close(actual.cpu(), expected)


def test_non_contiguous_input():
    x = torch.randn(4, 16, device="cuda").t()
    w = torch.randn(4, device="cuda")
    torch.testing.assert_close(op(x, w, EPS).cpu(), op(x.cpu(), w.cpu(), EPS))


@pytest.mark.parametrize(
    "x, w, match",
    [
        (torch.randn(2, 4), torch.ones(3), "does not match"),
        (torch.randn(2, 4), torch.ones(2, 4), "must be 1-D"),
        (torch.randn(2, 4), torch.ones(4, dtype=torch.float64), "dtype mismatch"),
        (torch.ones(2, 3, dtype=torch.int64), torch.ones(3, dtype=torch.int64), "not implemented"),
    ],
)
def test_invalid_inputs(x, w, match):
    with pytest.raises(RuntimeError, match=match):
        op(x.cuda(), w.cuda(), EPS)


def test_weight_on_other_device():
    x = torch.randn(2, 4, device="cuda")
    w = torch.ones(4)
    with pytest.raises(RuntimeError, match="same device"):
        op(x, w, EPS)


# Backward correctness

@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", SHAPES)
def test_backward_matches_cpu(shape, dtype):
    g = torch.randn(shape, dtype=dtype)
    x = torch.randn(shape, dtype=dtype)
    w = torch.randn(shape[-1], dtype=dtype)
    expected_gx, expected_gw = op_backward(g, x, w, EPS)
    gx, gw = op_backward(g.cuda(), x.cuda(), w.cuda(), EPS)
    torch.testing.assert_close(gx.cpu(), expected_gx)
    torch.testing.assert_close(gw.cpu(), expected_gw)


@pytest.mark.parametrize("shape", [(8,), (3, 8), (2, 4, 16)])
def test_gradcheck(shape):
    x = torch.randn(shape, dtype=torch.float64, device="cuda", requires_grad=True)
    w = torch.randn(shape[-1], dtype=torch.float64, device="cuda", requires_grad=True)
    assert torch.autograd.gradcheck(lambda a, b: op(a, b, EPS), (x, w))


@pytest.mark.parametrize("which", ["x", "weight"])
def test_partial_grad(which):
    x = torch.randn(3, 8, dtype=torch.float64, device="cuda", requires_grad=which == "x")
    w = torch.randn(8, dtype=torch.float64, device="cuda", requires_grad=which == "weight")
    op(x, w, EPS).sum().backward()
    assert (x.grad is not None) == (which == "x")
    assert (w.grad is not None) == (which == "weight")


@pytest.mark.parametrize(
    "g, x, w, match",
    [
        (torch.randn(2, 4), torch.randn(2, 4), torch.ones(3), "does not match"),
        (torch.randn(2, 3), torch.randn(2, 4), torch.ones(4), "grad_out shape"),
        (torch.randn(2, 4), torch.randn(2, 4), torch.ones(4, dtype=torch.float64), "dtype mismatch"),
    ],
)
def test_backward_invalid_inputs(g, x, w, match):
    with pytest.raises(RuntimeError, match=match):
        op_backward(g.cuda(), x.cuda(), w.cuda(), EPS)


# opcheck

@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", [(8,), (3, 8), (2, 4, 16)])
@pytest.mark.parametrize("requires_grad", [False, True])
def test_opcheck_forward(shape, dtype, requires_grad):
    x = torch.randn(shape, dtype=dtype, device="cuda", requires_grad=requires_grad)
    w = torch.randn(shape[-1], dtype=dtype, device="cuda", requires_grad=requires_grad)
    torch.library.opcheck(op.default, (x, w, EPS))


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", [(8,), (3, 8), (2, 4, 16)])
def test_opcheck_backward(shape, dtype):
    g = torch.randn(shape, dtype=dtype, device="cuda")
    x = torch.randn(shape, dtype=dtype, device="cuda")
    w = torch.randn(shape[-1], dtype=dtype, device="cuda")
    torch.library.opcheck(op_backward.default, (g, x, w, EPS))


# torch.compile

def fn(x, w):
    return op(x, w, EPS) * 2 + 1


def test_compile_fullgraph_forward_and_backward():
    torch._dynamo.reset()
    x = torch.randn(4, 64, device="cuda", requires_grad=True)
    w = torch.randn(64, device="cuda", requires_grad=True)
    x_ref = x.detach().clone().requires_grad_()
    w_ref = w.detach().clone().requires_grad_()

    y = torch.compile(fn, fullgraph=True)(x, w)
    y.sum().backward()

    y_ref = fn(x_ref, w_ref)
    y_ref.sum().backward()

    torch.testing.assert_close(y, y_ref)
    torch.testing.assert_close(x.grad, x_ref.grad)
    torch.testing.assert_close(w.grad, w_ref.grad)
