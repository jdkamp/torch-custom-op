#include <Python.h>
#include <ATen/ATen.h>
#include <torch/library.h>

#include <algorithm>

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

TORCH_LIBRARY(rmsnorm, m) {
    m.def("myrelu(Tensor x) -> Tensor");
}

TORCH_LIBRARY_IMPL(rmsnorm, CPU, m) {
    m.impl("myrelu", &myrelu_cpu);
}