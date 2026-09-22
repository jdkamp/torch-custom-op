import torch


def _setup_context(ctx, inputs, output):
    x, weight, eps = inputs
    ctx.save_for_backward(x, weight)
    ctx.eps = eps

def _backward(ctx, grad_out):
    x, weight = ctx.saved_tensors
    grad_x, grad_weight = torch.ops.rmsnorm.rmsnorm_backward(
        grad_out.contiguous(), x, weight, ctx.eps
    )
    if not ctx.needs_input_grad[0]:
        grad_x = None
    if not ctx.needs_input_grad[1]:
        grad_weight = None
    return grad_x, grad_weight, None


torch.library.register_autograd(
    "rmsnorm::rmsnorm", _backward, setup_context=_setup_context
)