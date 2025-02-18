#include <hip/amd_detail/amd_hip_runtime.h>
#include <hip/driver_types.h>
#include <hip/hip_runtime.h>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <numeric>
#include <string>
#include <sys/syscall.h>
#include <algorithm>
#include <stdexcept>

#include "../utils/hip_utils.hpp"
#include "tensor.hpp"

using namespace std;

template <typename Derived, typename T, typename D>
TensorBase<Derived, T, D>::TensorBase(const vector<size_t> &shape, const D &device) : shape(shape) {
    strides = vector<size_t>(shape.size(), 1);
    offsets = vector<size_t>(shape.size(), 0);
    size_t numel = 1;
    for (size_t i = 0; i < shape.size(); i++) {
        this->shape[i] = shape[i];
        numel *= shape[i];
        for (size_t j = i + 1; j < shape.size(); j++) {
            strides[i] *= shape[j];
        }
    }

    storage = make_shared<Storage<T, D>>(numel, device);
}


template <typename Derived, typename T, typename D>
size_t TensorBase<Derived, T, D>::numel() const {
    return accumulate(shape.begin(), shape.end(), 1, [](size_t a, size_t b) { return a * b; });
}


template <typename Derived, typename T, typename D>
size_t TensorBase<Derived, T, D>::dim() const {
    return shape.size();
}


template <typename Derived, typename T, typename D>
bool TensorBase<Derived, T, D>::is_view() const {
    return numel() != storage.get()->get_size();
}


template <typename T>
string Tensor<T, CPU>::repr() const {
    return sub_repr(0, 0);
}


template <typename T>
string Tensor<T, CPU>::sub_repr(const size_t d, const size_t offset) const {

    string r = "[";

    if (d < this->shape.size() - 1) {
        for (size_t i = 0; i < this->shape[d]; i++) {
            r += sub_repr(d + 1, i * this->strides[d]);
            if (i < this->shape[d] - 1) {
                r += ",\n";
                for (size_t j = 0; j < d + 1; j++) {
                    r += " ";
                }
            }
        }
    } else { // d = dim
        for (size_t i = 0; i < this->shape[d]; i++) {
            r += to_string((*this->storage)[offset + i * this->strides[d]]);
            if (i < this->shape[d] - 1) {
                r += ", ";
            }
        }
    }

    r += "]";

    return r;
}


template <typename T>
void Tensor<T, CPU>::write_storage(const size_t offset, const size_t n, const T value) {
    T* start = this->storage.get()->get_data() + offset;
    std::fill(start, start + n, value);
}


template <typename T>
void Tensor<T, CPU>::write_storage(const size_t offset, const size_t n, const T* values, const Device& src) {
    T* start = this->storage.get()->get_data() + offset;
    if (typeid(&src) == typeid(CPU)) {
        for (size_t i = 0; i < n; i++) {
            start[i] = values[i];
        }
    } else {
        HIP_CHECK(hipSetDevice(src));
        HIP_CHECK(hipMemcpy(start, values, n*sizeof(T), hipMemcpyDeviceToHost));
    }
}


template <typename T>
void Tensor<T, GPU>::write_storage(const size_t offset, const size_t n, const T value) {
    T* start = this->storage.get()->get_data() + offset;
    HIP_CHECK(hipMemset(start, value, n*sizeof(T)));
}


template <typename T>
void Tensor<T, GPU>::write_storage(const size_t offset, const size_t n, const T* values, const Device& src) {
    T *start = this->storage.get()->get_data() + offset;
    GPU device = this->storage.get()->get_device();
    HIP_CHECK(hipSetDevice(device));
    if (typeid(&src) == typeid(CPU)) {
        HIP_CHECK(hipMemcpy(start, values, n*sizeof(T), hipMemcpyHostToDevice));
    } else {
        if (src.get_id() == device.get_id()) {
            HIP_CHECK(hipMemcpy(start, values, n*sizeof(T), hipMemcpyDeviceToDevice));
        } else {
            throw runtime_error("Data copy between different GPUs not implemented yet");
            // TODO
        }
    }
}


template <typename Derived, typename T, typename D>
TensorBase<Derived, T, D> &TensorBase<Derived, T, D>::fill(const T value) {
    if (!is_view()) { 
        write_storage(0, this->numel(), value);
    } else {
        throw runtime_error("view fill not implemented");
    }// TODO provide implementation for views
    return *this;
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
