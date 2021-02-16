
#include <pybind11/pybind11.h>
#include <dlpack.h>

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace py = pybind11;

void print_info(const DLTensor* tensor)
{
    printf("context : %d %d\n", tensor->ctx.device_type, tensor->ctx.device_id);
    printf("dtype : %d %d %d\n", tensor->dtype.code, tensor->dtype.bits, tensor->dtype.lanes);
    printf("ndim : %d\n", tensor->ndim);
    printf("shape : ");
    int64_t len = 1;
    for (int i=0; i<tensor->ndim; ++i) {
        len *= tensor->shape[i];
        printf("%ld ", tensor->shape[i]);
    }
    printf("\n");
    printf("strides : ");
    for (int i=0; i<tensor->ndim; ++i) {
        printf("%ld ", tensor->strides[i]);
    }
    printf("\n");
    printf("byte_offset : %lu\n", tensor->byte_offset);
}

int add(int i, int j) {
    return i + j;
}

__global__
void VecAdd(float* data, size_t length, float value)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= length)
        return;
    data[idx] += value + idx;
}

void add_gpu(py::handle capsule, float value)
{
    DLTensor* tensor = (DLTensor*) PyCapsule_GetPointer(capsule.ptr(), "dltensor");
    // print_info(tensor);
    size_t len = 1;
    for (int i=0; i<tensor->ndim; ++i) {
        len *= tensor->shape[i];
    }
    // printf("len : %ld\n", len);
    // printf("value : %f\n", value);
    VecAdd<<<(len+127)/128, 128>>>((float*)tensor->data, len, value);
}

PYBIND11_MODULE(pybind11_test, m) {
    m.doc() = R"pbdoc(
        Pybind11 example plugin
        -----------------------
        .. currentmodule:: pybind11_test
        .. autosummary::
           :toctree: _generate
           add
           subtract
    )pbdoc";

    m.def("add", &add, R"pbdoc(
        Add two numbers
        Some other explanation about the add function.
    )pbdoc");

    m.def("subtract", [](int i, int j) { return i - j; }, R"pbdoc(
        Subtract two numbers
        Some other explanation about the subtract function.
    )pbdoc");

    m.def("add_gpu", &add_gpu, R"pbdoc(
        Add value to DLPack tensor data
        Some other explanation about the add_gpu function is blowing in the wind.
    )pbdoc");


#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}

