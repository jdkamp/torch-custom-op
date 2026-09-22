import torch

@torch.library.register_fake("rmsnorm::rmsnorm")
def _(x, weight, eps):
    torch._check(weight.dim() == 1)
    torch._check(weight.shape[0] == x.shape[-1])
    return torch.empty_like(x)

@torch.library.register_fake("rmsnorm::rmsnorm_backward")
def _(grad_out, x, weight, eps):
    return torch.empty_like(x), torch.empty_like(weight)