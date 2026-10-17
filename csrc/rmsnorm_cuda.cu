#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/library.h>

#include <algorithm>
#include <tuple>

// Sum v over the whole block. Every thread receives the result.
template <typename T>
__device__ T block_sum(T v, T* partial) {
    int lane = threadIdx.x % 32;
    int warp = threadIdx.x / 32;

    // Reduce within each warp.
    for (int offset = 16; offset > 0; offset /= 2) {
        v += __shfl_xor_sync(0xffffffff, v, offset);
    }
    if (lane == 0) {
        partial[warp] = v;
    }
    __syncthreads();

    // Reduce across warps.
    int num_warps = (blockDim.x + 31) / 32;
    v = lane < num_warps ? partial[lane] : T(0);
    for (int offset = 16; offset > 0; offset /= 2) {
        v += __shfl_xor_sync(0xffffffff, v, offset);
    }

    // partial may be reused in the next call.
    __syncthreads();

    return v;
}

template <typename T>
__global__ void rmsnorm_fwd_kernel(const T* x, const T* w, T* y, int64_t n, T eps) {
    __shared__ T partial[32];
    int64_t row = blockIdx.x;
    const T* xr = x + row * n;
    T* yr = y + row * n;

    // Each thread sums a strided slice of the row.
    T s = 0;
    for (int64_t j = threadIdx.x; j < n; j += blockDim.x) {
        s += xr[j] * xr[j];
    }

    // Reduce over the block and compute inv = 1 / rms.
    T inv = rsqrt(block_sum(s, partial) / n + eps);

    // Each thread writes its strided slice of the output.
    for (int64_t j = threadIdx.x; j < n; j += blockDim.x) {
        yr[j] = w[j] * xr[j] * inv;
    }
}

// Keeps each thread's elements in registers, so x is read only once.
// Requires n <= kElemsPerThread * blockDim.x.
template <typename T, int kElemsPerThread>
__global__ void rmsnorm_fwd_cached_kernel(const T* x, const T* w, T* y, int64_t n, T eps) {
    __shared__ T partial[32];
    int64_t row = blockIdx.x;
    const T* xr = x + row * n;
    T* yr = y + row * n;

    // Load this thread's elements once and sum their squares.
    T xv[kElemsPerThread];
    T s = 0;
    #pragma unroll
    for (int k = 0; k < kElemsPerThread; k++) {
        int64_t j = threadIdx.x + k * blockDim.x;
        xv[k] = j < n ? xr[j] : T(0);
        s += xv[k] * xv[k];
    }

    // Reduce over the block and compute inv = 1 / rms.
    T inv = rsqrt(block_sum(s, partial) / n + eps);

    // Write the output from registers.
    #pragma unroll
    for (int k = 0; k < kElemsPerThread; k++) {
        int64_t j = threadIdx.x + k * blockDim.x;
        if (j < n) {
            yr[j] = w[j] * xv[k] * inv;
        }
    }
}

template <typename T>
__global__ void rmsnorm_bwd_kernel(const T* g, const T* x, const T* w,
                                   T* gx, T* gw_rows, int64_t n, T eps) {
    __shared__ T partial[32];
    int64_t row = blockIdx.x;
    const T* gr = g + row * n;
    const T* xr = x + row * n;
    T* gxr = gx + row * n;
    T* gwr = gw_rows + row * n;

    // Pass 1: recompute inv = 1 / rms.
    T s = 0;
    for (int64_t j = threadIdx.x; j < n; j += blockDim.x) {
        s += xr[j] * xr[j];
    }
    T inv = rsqrt(block_sum(s, partial) / n + eps);

    // Pass 2: grad_weight contribution and the coupling sum.
    T d = 0;
    for (int64_t j = threadIdx.x; j < n; j += blockDim.x) {
        T xhat = xr[j] * inv;
        d += w[j] * gr[j] * xhat;
        gwr[j] = gr[j] * xhat;
    }
    T dot = block_sum(d, partial);

    // Pass 3: grad_x.
    for (int64_t j = threadIdx.x; j < n; j += blockDim.x) {
        T xhat = xr[j] * inv;
        gxr[j] = inv * (w[j] * gr[j] - xhat * dot / n);
    }
}

template <typename T, int kElemsPerThread>
__global__ void rmsnorm_bwd_cached_kernel(const T* g, const T* x, const T* w,
                                          T* gx, T* gw_rows, int64_t n, T eps) {
    __shared__ T partial[32];
    int64_t row = blockIdx.x;
    const T* gr = g + row * n;
    const T* xr = x + row * n;
    T* gxr = gx + row * n;
    T* gwr = gw_rows + row * n;

    T xv[kElemsPerThread];
    T gv[kElemsPerThread];
    T wv[kElemsPerThread];

    // Pass 1: recompute inv = 1 / rms.
    T s = 0;
    #pragma unroll
    for (int k = 0; k < kElemsPerThread; k++) {
        int64_t j = threadIdx.x + k * blockDim.x;
        bool in_row = j < n;
        xv[k] = in_row ? xr[j] : T(0);
        gv[k] = in_row ? gr[j] : T(0);
        wv[k] = in_row ? w[j] : T(0);
        s += xv[k] * xv[k];
    }
    T inv = rsqrt(block_sum(s, partial) / n + eps);

    // Pass 2: grad_weight contribution and the coupling sum.
    T d = 0;
    #pragma unroll
    for (int k = 0; k < kElemsPerThread; k++) {
        int64_t j = threadIdx.x + k * blockDim.x;
        T xhat = xv[k] * inv;
        d += wv[k] * gv[k] * xhat;
        if (j < n) {
            gwr[j] = gv[k] * xhat;
        }
    }
    T dot = block_sum(d, partial);

    // Pass 3: grad_x.
    #pragma unroll
    for (int k = 0; k < kElemsPerThread; k++) {
        int64_t j = threadIdx.x + k * blockDim.x;
        T xhat = xv[k] * inv;
        if (j < n) {
            gxr[j] = inv * (wv[k] * gv[k] - xhat * dot / n);
        }
    }
}

constexpr int kThreadsPerRow = 256;
constexpr int kElemsPerThread = 8;
constexpr int kMaxThreads = 1024;

int threads_per_row(int64_t n) {
    int64_t threads = (n + kElemsPerThread - 1) / kElemsPerThread;
    threads = (threads + 31) / 32 * 32;
    return static_cast<int>(std::clamp<int64_t>(threads, 32, kMaxThreads));
}

at::Tensor rmsnorm_cuda(const at::Tensor& x, const at::Tensor& weight, double eps) {
    TORCH_CHECK(x.device().is_cuda(), "rmsnorm: expected x on CUDA, got ", x.device());
    TORCH_CHECK(weight.device() == x.device(),
                "rmsnorm: weight must be on the same device as x, got ",
                weight.device(), " and ", x.device());
    TORCH_CHECK(x.dim() >= 1, "rmsnorm: x must have at least 1 dimension");
    TORCH_CHECK(weight.dim() == 1, "rmsnorm: weight must be 1-D, got ", weight.dim(), "-D");
    TORCH_CHECK(weight.size(0) == x.size(-1),
                "rmsnorm: weight size ", weight.size(0),
                " does not match the last dimension of x (", x.size(-1), ")");
    TORCH_CHECK(x.dtype() == weight.dtype(),
                "rmsnorm: dtype mismatch, x is ", x.dtype(), " but weight is ", weight.dtype());

    const c10::cuda::CUDAGuard device_guard(x.device());
    at::Tensor xc = x.contiguous();
    at::Tensor wc = weight.contiguous();
    at::Tensor out = at::empty_like(xc);

    const int64_t n = xc.size(-1);
    const int64_t rows = n == 0 ? 0 : xc.numel() / n;
    if (rows > 0) {
        const int threads = threads_per_row(n);
        const bool cached = n <= static_cast<int64_t>(kElemsPerThread) * threads;

        AT_DISPATCH_FLOATING_TYPES(xc.scalar_type(), "rmsnorm", [&] {
            auto stream = at::cuda::getCurrentCUDAStream();
            if (cached) {
                rmsnorm_fwd_cached_kernel<scalar_t, kElemsPerThread>
                    <<<rows, threads, 0, stream>>>(
                        xc.const_data_ptr<scalar_t>(), wc.const_data_ptr<scalar_t>(),
                        out.mutable_data_ptr<scalar_t>(), n, static_cast<scalar_t>(eps));
            } else {
                rmsnorm_fwd_kernel<scalar_t>
                    <<<rows, kThreadsPerRow, 0, stream>>>(
                        xc.const_data_ptr<scalar_t>(), wc.const_data_ptr<scalar_t>(),
                        out.mutable_data_ptr<scalar_t>(), n, static_cast<scalar_t>(eps));
            }
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        });
    }

    return out;
}

std::tuple<at::Tensor, at::Tensor> rmsnorm_backward_cuda(
    const at::Tensor& grad_out, const at::Tensor& x, const at::Tensor& weight, double eps) {
    TORCH_CHECK(x.is_cuda(), "rmsnorm_backward: expected x on CUDA, got ", x.device());
    TORCH_CHECK(grad_out.device() == x.device() && weight.device() == x.device(),
                "rmsnorm_backward: all tensors must be on the same device, got grad_out on ",
                grad_out.device(), ", x on ", x.device(), ", weight on ", weight.device());
    TORCH_CHECK(x.dim() >= 1, "rmsnorm_backward: x must have at least 1 dimension");
    TORCH_CHECK(weight.dim() == 1,
                "rmsnorm_backward: weight must be 1-D, got ", weight.dim(), "-D");
    TORCH_CHECK(weight.size(0) == x.size(-1),
                "rmsnorm_backward: weight size ", weight.size(0),
                " does not match the last dimension of x (", x.size(-1), ")");
    TORCH_CHECK(grad_out.sizes() == x.sizes(),
                "rmsnorm_backward: grad_out shape ", grad_out.sizes(),
                " does not match x shape ", x.sizes());
    TORCH_CHECK(x.dtype() == weight.dtype() && x.dtype() == grad_out.dtype(),
                "rmsnorm_backward: dtype mismatch, x is ", x.dtype(),
                ", weight is ", weight.dtype(), ", grad_out is ", grad_out.dtype());

    const c10::cuda::CUDAGuard device_guard(x.device());
    at::Tensor gc = grad_out.contiguous();
    at::Tensor xc = x.contiguous();
    at::Tensor wc = weight.contiguous();

    const int64_t n = xc.size(-1);
    const int64_t rows = n == 0 ? 0 : xc.numel() / n;

    at::Tensor grad_x = at::empty_like(xc);
    // One row of grad_weight contributions per input row, summed below.
    at::Tensor gw_rows = at::empty({rows, n}, xc.options());
    if (rows > 0) {
        const int threads = threads_per_row(n);
        const bool cached = n <= static_cast<int64_t>(kElemsPerThread) * threads;

        AT_DISPATCH_FLOATING_TYPES(xc.scalar_type(), "rmsnorm_backward", [&] {
            auto stream = at::cuda::getCurrentCUDAStream();
            if (cached) {
                rmsnorm_bwd_cached_kernel<scalar_t, kElemsPerThread>
                    <<<rows, threads, 0, stream>>>(
                        gc.const_data_ptr<scalar_t>(), xc.const_data_ptr<scalar_t>(),
                        wc.const_data_ptr<scalar_t>(), grad_x.mutable_data_ptr<scalar_t>(),
                        gw_rows.mutable_data_ptr<scalar_t>(), n, static_cast<scalar_t>(eps));
            } else {
                rmsnorm_bwd_kernel<scalar_t>
                    <<<rows, kThreadsPerRow, 0, stream>>>(
                        gc.const_data_ptr<scalar_t>(), xc.const_data_ptr<scalar_t>(),
                        wc.const_data_ptr<scalar_t>(), grad_x.mutable_data_ptr<scalar_t>(),
                        gw_rows.mutable_data_ptr<scalar_t>(), n, static_cast<scalar_t>(eps));
            }
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        });
    }
    return {grad_x, gw_rows.sum(0)};
}

TORCH_LIBRARY_IMPL(rmsnorm, CUDA, m) {
    m.impl("rmsnorm", &rmsnorm_cuda);
    m.impl("rmsnorm_backward", &rmsnorm_backward_cuda);
}
