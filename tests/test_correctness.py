import pytest
import torch
import torch.nn.functional as F

import rmsnorm  # noqa: F401

op = torch.ops.rmsnorm.rmsnorm
EPS = 1e-6

@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("shape", [(8,), (4, 16), (3, 5, 512), (2, 4096), (0, 8), (3, 0)])
def test_matches_reference(shape, dtype):
    x = torch.randn(shape, dtype=dtype)
    w = torch.randn(shape[-1], dtype=dtype)
    torch.testing.assert_close(op(x, w, EPS), F.rms_norm(x, (shape[-1],), w, EPS))

def test_non_contiguous_input():
    x = torch.randn(4, 16).t()
    w = torch.randn(4)
    torch.testing.assert_close(op(x, w, EPS), F.rms_norm(x, (4,), w, EPS))

@pytest.mark.parametrize(
    "x, w, match",
    [
        (torch.randn(2, 4), torch.ones(3), "does not match"),
        (torch.randn(2, 4), torch.ones(2, 4), "must be 1-D"),
        (torch.randn(2, 4), torch.ones(4, dtype=torch.float64), "dtype mismatch"),
        (torch.tensor(1.0), torch.ones(1), "at least 1 dimension"),
        (torch.ones(2, 3, dtype=torch.int64), torch.ones(3, dtype=torch.int64), "not implemented"),
    ],
)
def test_invalid_inputs(x, w, match):
    with pytest.raises(RuntimeError, match=match):
        op(x, w, EPS)