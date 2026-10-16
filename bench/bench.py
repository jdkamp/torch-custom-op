"""Forward benchmark for the rmsnorm custom op."""
import torch
import torch.nn.functional as F
import torch.utils.benchmark as benchmark

import rmsnorm  # noqa: F401

EPS = 1e-6

def naive(x, w):
    return w * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS)

def measure(fn):
    fn()
    timer = benchmark.Timer(stmt="fn()", globals={"fn": fn})
    return timer.blocked_autorange(min_run_time=1.0).median

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rows, n = 8192, 4096
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

    name = torch.cuda.get_device_name() if device == "cuda" else "cpu"
    print(f"{name}, shape ({rows}, {n}), {x.dtype}")

    for label, fn in variants.items():
        seconds = measure(fn)
        print(f"{label:22s} {seconds * 1e3:8.3f} ms")


if __name__ == "__main__":
    main()
