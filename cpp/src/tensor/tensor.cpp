#include <cassert>
#include <cstddef>
#include <hip/amd_detail/amd_hip_runtime.h>
#include <hip/driver_types.h>
#include <hip/hip_runtime.h>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <numeric>
#include <string>
#include <sys/syscall.h>
#include <algorithm>
#include <stdexcept>
#include <vector>

#include "../utils/hip_utils.hpp"
#include "storage.hpp"
#include "tensor.hpp"

using namespace std;

size_t numel_from_shape(const vector<size_t>& shape) {
    return accumulate(shape.begin(), shape.end(), 1, [](size_t a, size_t b) { return a * b; });
}

vector<size_t> base_stride_from_shape(const vector<size_t>& shape) {
    vector<size_t> stride = vector<size_t>(shape.size(), 1);
    for (size_t i = 0; i < shape.size(); i++) {
        for (size_t j = i + 1; j < shape.size(); j++) {
            stride[i] *= shape[j];
        }
    }
    return stride;
}

template <typename Derived, typename T, typename D>
TensorBase<Derived, T, D>::TensorBase(const vector<size_t> &shape, const D &device) : shape(shape) {
    stride = base_stride_from_shape(shape);
    offset = 0;
    storage = make_shared<Storage<T, D>>(numel_from_shape(shape), device);
}


template <typename Derived, typename T, typename D>
size_t TensorBase<Derived, T, D>::numel() const {
    return numel_from_shape(shape);
}


template <typename Derived, typename T, typename D>
size_t TensorBase<Derived, T, D>::dim() const {
    return shape.size();
}


template <typename Derived, typename T, typename D>
bool TensorBase<Derived, T, D>::is_contiguous() const {
    return numel() != storage->size && stride == base_stride_from_shape(shape);
}


template <typename Derived, typename T, typename D>
Derived TensorBase<Derived, T, D>::copy() {
    Derived tensor(shape, storage->device);
    tensor.storage = storage;
    return tensor;
}


template <typename Derived, typename T, typename D>
Derived TensorBase<Derived, T, D>::clone() {
    Derived tensor(shape, storage->device);
    if (!is_contiguous()) { 
        tensor.write_storage(0, numel(), storage->data, storage->device);
    } else {
        throw runtime_error("view clone not implemented");
    } // TODO provide implementation for views
    return tensor;
}


template <typename Derived, typename T, typename D>
TensorBase<Derived, T, D> &TensorBase<Derived, T, D>::fill(const T &value) {
    if (!is_contiguous()) { 
        write_storage(0, this->numel(), value);
    } else {
        throw runtime_error("view fill not implemented");
    }// TODO provide implementation for views
    return *this;
}


template <typename Derived, typename T, typename D>
TensorBase<Derived, T, D> &TensorBase<Derived, T, D>::contiguous() {
    if (!is_contiguous()) { 
        // TODO
    }
    return *this;
}


template <typename Derived, typename T, typename D>
TensorBase<Derived, T, D> &TensorBase<Derived, T, D>::reshape(const vector<size_t> &new_shape) {
    assert(numel_from_shape(new_shape) == numel());
    if (!is_contiguous()) { 
        shape = new_shape;
        stride = base_stride_from_shape(shape);
        // offset already 0
        return *this;
    } else {
        throw runtime_error("view flatten not implemented");
    }// TODO provide implementation for views
}


// template <typename Derived, typename T, typename D>
// TensorBase<Derived, T, D> &TensorBase<Derived, T, D>::expand(const vector<size_t>& new_shape) {}


template <typename Derived, typename T, typename D>
TensorBase<Derived, T, D> &TensorBase<Derived, T, D>::flatten() {
    return reshape({numel()});
}


template <typename T>
string Tensor<T, CPU>::repr() const {
    return sub_repr(0, 0);
}


template <typename T>
string Tensor<T, CPU>::sub_repr(const size_t &d, const size_t &offset) const {

    string r = "[";

    if (d < this->shape.size() - 1) {
        for (size_t i = 0; i < this->shape[d]; i++) {
            r += sub_repr(d + 1, i * this->stride[d]);
            if (i < this->shape[d] - 1) {
                r += ",\n";
                for (size_t j = 0; j < d + 1; j++) {
                    r += " ";
                }
            }
        }
    } else { // d = dim
        for (size_t i = 0; i < this->shape[d]; i++) {
            r += to_string((*this->storage)[this->offset + offset + i * this->stride[d]]);
            if (i < this->shape[d] - 1) {
                r += ", ";
            }
        }
    }

    r += "]";

    return r;
}


template <typename T>
void Tensor<T, CPU>::write_storage(const size_t &offset, const size_t &n, const T &value) {
    T* start = this->storage->data + offset;
    std::fill(start, start + n, value);
}


template <typename T>
void Tensor<T, CPU>::write_storage(const size_t &offset, const size_t &n, const T* const &values, const CPU& src) {
    T* start = this->storage->data + offset;
    memcpy(start, values, n * sizeof(T));
}


template <typename T>
void Tensor<T, CPU>::write_storage(const size_t &offset, const size_t &n, const T* const &values, const GPU& src) {
    T* start = this->storage->data + offset;
    HIP_CHECK(hipSetDevice(src));
    HIP_CHECK(hipMemcpy(start, values, n*sizeof(T), hipMemcpyDeviceToHost));
}

template <typename T>
void Tensor<T, GPU>::write_storage(const size_t &offset, const size_t &n, const T &value) {
    T* start = this->storage->data + offset;
    HIP_CHECK(hipMemset(start, value, n*sizeof(T)));
}


template <typename T>
void Tensor<T, GPU>::write_storage(const size_t &offset, const size_t &n, const T* const &values, const CPU& src) {
    T *start = this->storage->data + offset;
    HIP_CHECK(hipSetDevice(this->storage->device));
    HIP_CHECK(hipMemcpy(start, values, n*sizeof(T), hipMemcpyHostToDevice));
}


template <typename T>
void Tensor<T, GPU>::write_storage(const size_t &offset, const size_t &n, const T* const &values, const GPU& src) {
    T *start = this->storage->data + offset;
    GPU device = this->storage->device;
    HIP_CHECK(hipSetDevice(device));
    if (src.id == device.id) {
        HIP_CHECK(hipMemcpy(start, values, n*sizeof(T), hipMemcpyDeviceToDevice));
    } else {
        throw runtime_error("Data copy between different GPUs not implemented yet");
        // TODO
    }
}

#define INSTANTIATE_TENSOR(T, D)                      \
    template class TensorBase<Tensor<T, D>, T, D>;    \
    template class Tensor<T, D>;

INSTANTIATE_TENSOR(bool, CPU)
INSTANTIATE_TENSOR(bool, GPU)
INSTANTIATE_TENSOR(int, CPU)
INSTANTIATE_TENSOR(int, GPU)
INSTANTIATE_TENSOR(long, CPU)
INSTANTIATE_TENSOR(long, GPU)
INSTANTIATE_TENSOR(float, CPU)
INSTANTIATE_TENSOR(float, GPU)
INSTANTIATE_TENSOR(double, CPU)
INSTANTIATE_TENSOR(double, GPU)

#undef INSTANTIATE_TENSOR
