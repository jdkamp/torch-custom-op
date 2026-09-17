#include <Python.h>
#include <ATen/ATen.h>
#include <torch/library.h>

#include <algorithm>
#include <cmath>

// Makes the .so importable as rmsnorm._C. the module itself is empty.
extern "C" {
PyObject* PyInit__C(void) {
    static struct PyModuleDef module_def = {
        PyModuleDef_HEAD_INIT, "_C", nullptr, -1, nullptr,
    };
    return PyModule_Create(&module_def);
}
}

at::Tensor myrelu_cpu(const at::Tensor& x) {
    TORCH_CHECK(x.device().is_cpu(), "myrelu: expected a CPU tensor, got ", x.device());
    TORCH_CHECK(x.dtype() == at::kFloat, "myrelu: expected float32, got ", x.dtype());

    at::Tensor xc = x.contiguous();
    at::Tensor out = at::empty_like(xc);

    const float* in = xc.const_data_ptr<float>();
    float* dst = out.mutable_data_ptr<float>();

    for (int64_t i = 0; i < xc.numel(); i++) {
        dst[i] = std::max(in[i], 0.0f);
    }
    return out;
}

template <typename T>
void rmsnorm_fwd(const T* x, const T* w, T* y, int64_t rows, int64_t n, double eps) {
    for (int64_t r = 0; r < rows; r++) {
        const T* xr = x + r * n;
        T* yr = y + r * n;

        double sum_sq = 0.0;
        for (int64_t j = 0; j < n; j++) {
            sum_sq += static_cast<double>(xr[j]) * xr[j];
        }
        const T inv_rms = static_cast<T>(1.0 / std::sqrt(sum_sq / n + eps));

        for (int64_t j = 0; j < n; j++) {
            yr[j] = w[j] * xr[j] * inv_rms;
        }
    }
}


at::Tensor rmsnorm_cpu(const at::Tensor& x, const at::Tensor& weight, double eps) {
    TORCH_CHECK(x.device().is_cpu(), "rmsnorm: expected x on CPU, got ", x.device());
    TORCH_CHECK(weight.device().is_cpu(), "rmsnorm: expected weight on CPU, got ", weight.device());
    TORCH_CHECK(x.dim() >= 1, "rmsnorm: x must have at least 1 dimension");
    TORCH_CHECK(weight.dim() == 1, "rmsnorm: weight must be 1-D, got ", weight.dim(), "-D");
    TORCH_CHECK(weight.size(0) == x.size(-1),
                "rmsnorm: weight size ", weight.size(0),
                " does not match the last dimension of x (", x.size(-1), ")");
    TORCH_CHECK(x.dtype() == weight.dtype(),
                "rmsnorm: dtype mismatch, x is ", x.dtype(), " but weight is ", weight.dtype());

    at::Tensor xc = x.contiguous();
    at::Tensor wc = weight.contiguous();
    at::Tensor out = at::empty_like(xc);

    const int64_t n = xc.size(-1);
    const int64_t rows = n == 0 ? 0 : xc.numel() / n;
    AT_DISPATCH_FLOATING_TYPES(xc.scalar_type(), "rmsnorm", [&] {
        rmsnorm_fwd<scalar_t>(xc.const_data_ptr<scalar_t>(), wc.const_data_ptr<scalar_t>(),
                              out.mutable_data_ptr<scalar_t>(), rows, n, eps);
    });

    return out;
}

TORCH_LIBRARY(rmsnorm, m) {
    m.def("myrelu(Tensor x) -> Tensor");
    m.def("rmsnorm(Tensor x, Tensor weight, float eps) -> Tensor");
}

TORCH_LIBRARY_IMPL(rmsnorm, CPU, m) {
    m.impl("myrelu", &myrelu_cpu);
    m.impl("rmsnorm", &rmsnorm_cpu);
}
