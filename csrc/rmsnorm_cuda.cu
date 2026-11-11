#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/OpMathType.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/library.h>

#include <cstdint>
#include <cstdlib>

#include <algorithm>
#include <tuple>

constexpr int kWarpSize = 32;
constexpr int kMaxThreads = 1024;
constexpr int kMaxWarps = kMaxThreads / kWarpSize;
constexpr int kThreadsPerRow = 256;
constexpr int kElemsPerThread = 8;
constexpr int kVecBytes = 16;

// kVecBytes of consecutive elements, loaded and stored with one memory access.
template <typename T>
struct alignas(kVecBytes) Vec {
    T val[kVecBytes / sizeof(T)];
};

// block_sum reduces the per-warp sums inside a single warp.
static_assert(kMaxWarps <= kWarpSize, "per-warp sums must fit in one warp");

// Sum v over the whole block. Every thread receives the result.
// partial must hold kMaxWarps values.
template <typename T>
__device__ T block_sum(T v, T* partial) {
    int lane = threadIdx.x % kWarpSize;
    int warp = threadIdx.x / kWarpSize;

    // Reduce within each warp.
    for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
        v += __shfl_xor_sync(0xffffffff, v, offset);
    }
    if (lane == 0) {
        partial[warp] = v;
    }
    __syncthreads();

    // Reduce across warps.
    int num_warps = (blockDim.x + kWarpSize - 1) / kWarpSize;
    v = lane < num_warps ? partial[lane] : T(0);
    for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
        v += __shfl_xor_sync(0xffffffff, v, offset);
    }

    // partial may be reused in the next call.
    __syncthreads();

    return v;
}

template <typename T>
__global__ void rmsnorm_fwd_kernel(const T* x, const T* w, T* y, int64_t n, at::opmath_type<T> eps) {
    using acc_t = at::opmath_type<T>;

    __shared__ acc_t partial[kMaxWarps];
    int64_t row = blockIdx.x;
    const T* xr = x + row * n;
    T* yr = y + row * n;

    // Each thread sums a strided slice of the row.
    acc_t s = 0;
    for (int64_t j = threadIdx.x; j < n; j += blockDim.x) {
        s += static_cast<acc_t>(xr[j]) * xr[j];
    }

    // Reduce over the block and compute inv = 1 / rms.
    acc_t inv = rsqrt(block_sum(s, partial) / n + eps);

    // Each thread writes its strided slice of the output.
    for (int64_t j = threadIdx.x; j < n; j += blockDim.x) {
        yr[j] = static_cast<T>(static_cast<acc_t>(w[j]) * xr[j] * inv);
    }
}

// Keeps each thread's elements in registers, so x is read only once.
// Requires n <= kElemsPerThread * blockDim.x.
template <typename T, int kElemsPerThread>
__global__ void rmsnorm_fwd_cached_kernel(const T* x, const T* w, T* y, int64_t n, at::opmath_type<T> eps) {
    using acc_t = at::opmath_type<T>;

    __shared__ acc_t partial[kMaxWarps];
    int64_t row = blockIdx.x;
    const T* xr = x + row * n;
    T* yr = y + row * n;

    // Load this thread's elements once and sum their squares.
    acc_t xv[kElemsPerThread];
    acc_t s = 0;
    #pragma unroll
    for (int k = 0; k < kElemsPerThread; k++) {
        int64_t j = threadIdx.x + k * blockDim.x;
        xv[k] = j < n ? static_cast<acc_t>(xr[j]) : acc_t(0);
        s += xv[k] * xv[k];
    }

    // Reduce over the block and compute inv = 1 / rms.
    acc_t inv = rsqrt(block_sum(s, partial) / n + eps);

    // Write the output from registers.
    #pragma unroll
    for (int k = 0; k < kElemsPerThread; k++) {
        int64_t j = threadIdx.x + k * blockDim.x;
        if (j < n) {
            yr[j] = static_cast<T>(static_cast<acc_t>(w[j]) * xv[k] * inv);
        }
    }
}

// Like the cached kernel, but loads and stores kVecBytes at a time.
// Requires n to be a multiple of the vector size and 16-byte aligned pointers.
template <typename T, int kElemsPerThread>
__global__ void rmsnorm_fwd_vec_kernel(const T* x, const T* w, T* y, int64_t n, at::opmath_type<T> eps) {
    using acc_t = at::opmath_type<T>;
    
    constexpr int kVecSize = kVecBytes / sizeof(T);
    constexpr int kVecsPerThread = kElemsPerThread / kVecSize;
    static_assert(kElemsPerThread % kVecSize == 0, "a thread must hold whole vectors");

    __shared__ acc_t partial[kMaxWarps];
    int64_t row = blockIdx.x;
    const Vec<T>* xr = reinterpret_cast<const Vec<T>*>(x + row * n);
    const Vec<T>* wr = reinterpret_cast<const Vec<T>*>(w);
    Vec<T>* yr = reinterpret_cast<Vec<T>*>(y + row * n);

    // Load this thread's elements once and sum their squares.
    acc_t xv[kElemsPerThread];
    acc_t s = 0;
    #pragma unroll
    for (int m = 0; m < kVecsPerThread; m++) {
        int64_t v = threadIdx.x + m * blockDim.x;
        Vec<T> xvec = v * kVecSize < n ? xr[v] : Vec<T>{};
        #pragma unroll
        for (int i = 0; i < kVecSize; i++) {
            int k = m * kVecSize + i;
            xv[k] = static_cast<acc_t>(xvec.val[i]);
            s += xv[k] * xv[k];
        }

    }

    // Reduce over the block and compute inv = 1 / rms.
    acc_t inv = rsqrt(block_sum(s, partial) / n + eps);

    // Write the output from registers.
    #pragma unroll
    for (int m = 0; m < kVecsPerThread; m++) {
        int64_t v = threadIdx.x + m * blockDim.x;
        if (v * kVecSize < n) {
            Vec<T> wvec = wr[v];            // one 16-byte load
            Vec<T> yvec;
            #pragma unroll
            for (int i = 0; i < kVecSize; i++) {
                int k = m * kVecSize + i;
                yvec.val[i] = static_cast<T>(static_cast<acc_t>(wvec.val[i]) * xv[k] * inv);
            }
            yr[v] = yvec;                   // one 16-byte store
        }
    }
}

template <typename T>
__global__ void rmsnorm_bwd_kernel(const T* g, const T* x, const T* w,
                                   T* gx, at::opmath_type<T>* gw_rows, int64_t n, at::opmath_type<T> eps) {
    using acc_t = at::opmath_type<T>;

    __shared__ acc_t partial[kMaxWarps];
    int64_t row = blockIdx.x;
    const T* gr = g + row * n;
    const T* xr = x + row * n;
    T* gxr = gx + row * n;
    acc_t* gwr = gw_rows + row * n;

    // Pass 1: recompute inv = 1 / rms.
    acc_t s = 0;
    for (int64_t j = threadIdx.x; j < n; j += blockDim.x) {
        s += static_cast<acc_t>(xr[j]) * xr[j];
    }
    acc_t inv = rsqrt(block_sum(s, partial) / n + eps);

    // Pass 2: grad_weight contribution and the coupling sum.
    acc_t d = 0;
    for (int64_t j = threadIdx.x; j < n; j += blockDim.x) {
        acc_t xhat = static_cast<acc_t>(xr[j]) * inv;
        d += static_cast<acc_t>(w[j]) * gr[j] * xhat;
        gwr[j] = gr[j] * xhat;
    }
    acc_t dot = block_sum(d, partial);

    // Pass 3: grad_x.
    for (int64_t j = threadIdx.x; j < n; j += blockDim.x) {
        acc_t xhat = static_cast<acc_t>(xr[j]) * inv;
        gxr[j] = inv * (static_cast<acc_t>(w[j]) * gr[j] - xhat * dot / n);
    }
}

// Each block processes rows blockIdx.x, blockIdx.x + gridDim.x, ... and
// accumulates their grad_weight contributions in registers, writing one row
// of partial sums. Keeps x, g and w in registers, so each is read once.
// Requires n <= kElemsPerThread * blockDim.x.
template <typename T, int kElemsPerThread>
__global__ void rmsnorm_bwd_cached_kernel(const T* g, const T* x, const T* w, T* gx, at::opmath_type<T>* gw_partial,
                                          int64_t rows, int64_t n, at::opmath_type<T> eps) {
    using acc_t = at::opmath_type<T>;

    __shared__ acc_t partial[kMaxWarps];

    // w is the same for every row, so load it once per block.
    acc_t wv[kElemsPerThread];
    acc_t gw_acc[kElemsPerThread];
    #pragma unroll
    for (int k = 0; k < kElemsPerThread; k++) {
        int64_t j = threadIdx.x + k * blockDim.x;
        wv[k] = j < n ? static_cast<acc_t>(w[j]) : acc_t(0);
        gw_acc[k] = 0;
    }

    for (int64_t row = blockIdx.x; row < rows; row += gridDim.x) {
        const T* gr = g + row * n;
        const T* xr = x + row * n;
        T* gxr = gx + row * n;

        // Pass 1: load x and g once and recompute inv = 1 / rms.
        acc_t xv[kElemsPerThread];
        acc_t gv[kElemsPerThread];
        acc_t s = 0;
        #pragma unroll
        for (int k = 0; k < kElemsPerThread; k++) {
            int64_t j = threadIdx.x + k * blockDim.x;
            bool in_row = j < n;
            xv[k] = in_row ? static_cast<acc_t>(xr[j]) : acc_t(0);
            gv[k] = in_row ? static_cast<acc_t>(gr[j]) : acc_t(0);
            s += xv[k] * xv[k];
        }
        acc_t inv = rsqrt(block_sum(s, partial) / n + eps);

        // Pass 2: accumulate grad_weight and the coupling sum.
        acc_t d = 0;
        #pragma unroll
        for (int k = 0; k < kElemsPerThread; k++) {
            acc_t xhat = xv[k] * inv;
            d += wv[k] * gv[k] * xhat;
            gw_acc[k] += gv[k] * xhat;
        }
        acc_t dot = block_sum(d, partial);

        // Pass 3: grad_x, from registers.
        #pragma unroll
        for (int k = 0; k < kElemsPerThread; k++) {
            int64_t j = threadIdx.x + k * blockDim.x;
            if (j < n) {
                acc_t xhat = xv[k] * inv;
                gxr[j] = static_cast<T>(inv * (wv[k] * gv[k] - xhat * dot / n));
            }
        }
    }

    // One row of partial grad_weight sums per block.
    #pragma unroll
    for (int k = 0; k < kElemsPerThread; k++) {
        int64_t j = threadIdx.x + k * blockDim.x;
        if (j < n) {
            gw_partial[blockIdx.x * n + j] = gw_acc[k];
        }
    }
}

int threads_per_row(int64_t n) {
    int64_t threads = (n + kElemsPerThread - 1) / kElemsPerThread;
    threads = (threads + kWarpSize - 1) / kWarpSize * kWarpSize;
    return static_cast<int>(std::clamp<int64_t>(threads, kWarpSize, kMaxThreads));
}

// Largest block size the kernel can be launched with. A kernel that uses many
// registers per thread can't use the full 1024 threads.
template <typename Kernel>
int max_threads_per_block(Kernel kernel) {
    cudaFuncAttributes attr;
    C10_CUDA_CHECK(cudaFuncGetAttributes(&attr, kernel));
    return attr.maxThreadsPerBlock;
}

// Trze if p can b accessed as Vec, i.e. it is aligned to kVecBytes.
bool is_vec_aligned(const void* p) {
    return reinterpret_cast<uintptr_t>(p) % kVecBytes == 0;
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

    AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, xc.scalar_type(), "rmsnorm", [&] {
            auto stream = at::cuda::getCurrentCUDAStream();
            auto cached_kernel = rmsnorm_fwd_cached_kernel<scalar_t, kElemsPerThread>;
            auto vec_kernel = rmsnorm_fwd_vec_kernel<scalar_t, kElemsPerThread>;
            const int vec_size = kVecBytes / sizeof(scalar_t);

            // The vectorized kernel also needs whoöe vectors per row and aligned pointers.
            const bool vectorized = n <= static_cast<int64_t>(kElemsPerThread) * threads && 
                        threads <= max_threads_per_block(vec_kernel) &&
                        n % vec_size == 0 &&
                        is_vec_aligned(xc.const_data_ptr<scalar_t>()) &&
                        is_vec_aligned(wc.const_data_ptr<scalar_t>()) &&
                        is_vec_aligned(out.mutable_data_ptr<scalar_t>()) &&
                        std::getenv("RMSNORM_NO_VEC") == nullptr;

            // The cached kernel needs the whole row in registers, and a block
            // small enough for its register usage.
            const bool cached = n <= static_cast<int64_t>(kElemsPerThread) * threads &&
                                threads <= max_threads_per_block(cached_kernel);

            if (vectorized) {
                vec_kernel<<<rows, threads, 0, stream>>>(
                        xc.const_data_ptr<scalar_t>(), wc.const_data_ptr<scalar_t>(),
                        out.mutable_data_ptr<scalar_t>(), n, static_cast<at::opmath_type<scalar_t>>(eps));
            }
            else if (cached) {
                cached_kernel<<<rows, threads, 0, stream>>>(
                        xc.const_data_ptr<scalar_t>(), wc.const_data_ptr<scalar_t>(),
                        out.mutable_data_ptr<scalar_t>(), n, static_cast<at::opmath_type<scalar_t>>(eps));
            } else {
                rmsnorm_fwd_kernel<scalar_t>
                    <<<rows, kThreadsPerRow, 0, stream>>>(
                        xc.const_data_ptr<scalar_t>(), wc.const_data_ptr<scalar_t>(),
                        out.mutable_data_ptr<scalar_t>(), n, static_cast<at::opmath_type<scalar_t>>(eps));
            }
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        });
    }

    return out;
}

std::tuple<at::Tensor, at::Tensor> rmsnorm_backward_cuda(const at::Tensor& grad_out, const at::Tensor& x, 
                                                         const at::Tensor& weight, double eps) {
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
    // Partial grad_weight sums, one row per block (or per input row in the
    // non cached), summed below. Empty when there are no rows.
    const auto acc_options = xc.options().dtype(at::toOpMathType(xc.scalar_type()));
    at::Tensor gw_partial = at::empty({0, n}, acc_options);

    if (rows > 0) {
        const int threads = threads_per_row(n);

    AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, xc.scalar_type(), "rmsnorm_backward", [&] {
            auto stream = at::cuda::getCurrentCUDAStream();
            auto cached_kernel = rmsnorm_bwd_cached_kernel<scalar_t, kElemsPerThread>;

            // The cached kernel needs the whole row in registers, and a block
            // small enough for its register usage.
            const bool cached = n <= static_cast<int64_t>(kElemsPerThread) * threads &&
                                threads <= max_threads_per_block(cached_kernel);

            if (cached) {
                // As many blocks as fit on the GPU at once, given this kernel's
                // register and thread usage, each block loops over many rows.
                const int sm_count = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
                int blocks_per_sm = 0;
                C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                    &blocks_per_sm, cached_kernel, threads, 0));
                TORCH_INTERNAL_ASSERT(blocks_per_sm > 0,
                                      "rmsnorm_backward: cached kernel does not fit on an SM with ",
                                      threads, " threads");
                const int64_t blocks = std::min<int64_t>(
                    rows, static_cast<int64_t>(sm_count) * blocks_per_sm);
                gw_partial = at::empty({blocks, n}, acc_options);

                cached_kernel<<<blocks, threads, 0, stream>>>(
                    gc.const_data_ptr<scalar_t>(), xc.const_data_ptr<scalar_t>(),
                    wc.const_data_ptr<scalar_t>(), grad_x.mutable_data_ptr<scalar_t>(),
                    gw_partial.mutable_data_ptr<at::opmath_type<scalar_t>>(),
                    rows, n, static_cast<at::opmath_type<scalar_t>>(eps));
            } else {
                gw_partial = at::empty({rows, n}, acc_options);

                rmsnorm_bwd_kernel<scalar_t><<<rows, kThreadsPerRow, 0, stream>>>(
                    gc.const_data_ptr<scalar_t>(), xc.const_data_ptr<scalar_t>(),
                    wc.const_data_ptr<scalar_t>(), grad_x.mutable_data_ptr<scalar_t>(),
                    gw_partial.mutable_data_ptr<at::opmath_type<scalar_t>>(),
                    n, static_cast<at::opmath_type<scalar_t>>(eps));
            }
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        });
    }
    return {grad_x, gw_partial.sum(0).to(xc.scalar_type())};
}

TORCH_LIBRARY_IMPL(rmsnorm, CUDA, m) {
    m.impl("rmsnorm", &rmsnorm_cuda);
    m.impl("rmsnorm_backward", &rmsnorm_backward_cuda);
}
