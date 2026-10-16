"""Forward and backward benchmark for the rmsnorm custom op."""
import torch
import torch.nn.functional as F
import torch.utils.benchmark as benchmark

import rmsnorm  # noqa: F401

EPS = 1e-6


def naive(x, w):
    return w * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS)


def measure(fn):
    fn()  # warm-up for the torch.compile variant this also triggers compilation
    timer = benchmark.Timer(stmt="fn()", globals={"fn": fn})
    return timer.blocked_autorange(min_run_time=1.0).median


def bench_forward(device, rows, n):
    x = torch.randn(rows, n, device=device)
    w = torch.randn(n, device=device)
    compiled_naive = torch.compile(naive)

    variants = {
        "x.clone() (ceiling)": lambda: x.clone(),
        "custom op": lambda: torch.ops.rmsnorm.rmsnorm(x, w, EPS),
        "naive eager": lambda: naive(x, w),
        "F.rms_norm": lambda: F.rms_norm(x, (n,), w, EPS),
        "torch.compile(naive)": lambda: compiled_naive(x, w),
    }

    print("forward")
    for label, fn in variants.items():
        seconds = measure(fn)
        print(f"{label:22s} {seconds * 1e3:8.3f} ms")


def bench_backward(device, rows, n):
    x = torch.randn(rows, n, device=device, requires_grad=True)
    w = torch.randn(n, device=device, requires_grad=True)
    g = torch.randn(rows, n, device=device)
    compiled_naive = torch.compile(naive)

    forwards = {
        "custom op": lambda x, w: torch.ops.rmsnorm.rmsnorm(x, w, EPS),
        "naive eager": naive,
        "F.rms_norm": lambda x, w: F.rms_norm(x, (n,), w, EPS),
        "torch.compile(naive)": compiled_naive,
    }

    print("forward + backward")
    for label, forward in forwards.items():
        def step(forward=forward):
            return torch.autograd.grad(forward(x, w), (x, w), g)

        seconds = measure(step)
        print(f"{label:22s} {seconds * 1e3:8.3f} ms")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rows, n = 8192, 4096

    name = torch.cuda.get_device_name() if device == "cuda" else "cpu"
    print(f"{name}, shape ({rows}, {n}), torch.float32")

    bench_forward(device, rows, n)
    bench_backward(device, rows, n)


if __name__ == "__main__":
    main()
