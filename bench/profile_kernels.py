import torch
from torch.profiler import profile, ProfilerActivity

import rmsnorm  # noqa: F401

EPS = 1e-6


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    activities = [ProfilerActivity.CPU]
    if device == "cuda":
        activities.append(ProfilerActivity.CUDA)

    rows, n = 8192, 4096
    x = torch.randn(rows, n, device=device, requires_grad=True)
    w = torch.randn(n, device=device, requires_grad=True)
    g = torch.randn(rows, n, device=device)

    for _ in range(3):  # warm-up
        torch.autograd.grad(torch.ops.rmsnorm.rmsnorm(x, w, EPS), (x, w), g)

    with profile(activities=activities) as prof:
        for _ in range(10):
            torch.autograd.grad(torch.ops.rmsnorm.rmsnorm(x, w, EPS), (x, w), g)
        if device == "cuda":
            torch.cuda.synchronize()

    sort_by = "cuda_time_total" if device == "cuda" else "cpu_time_total"
    print(prof.key_averages().table(sort_by=sort_by, row_limit=15))



if __name__ == "__main__":
    main()
