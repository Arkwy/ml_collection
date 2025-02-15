#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <numeric>
#include <sys/syscall.h>
#include <iostream>

#include "tensor.hpp"

using namespace std;


template <typename T, typename D> TensorBase<T, D>::TensorBase(const vector<size_t> shape, const D device) : shape(shape) {
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

template <typename T> string Tensor<T, CPU>::repr() const { return sub_repr(0, 0); }

template <typename T> string Tensor<T, CPU>::sub_repr(const size_t d, const size_t offset) const {

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


template <typename T, typename D> Tensor<T, D> empty(const vector<size_t> shape, const D device) {
    Tensor<T, D> tensor(shape, device);
    return tensor;
}


template <typename T> Tensor<T, CPU> full(const vector<size_t> shape, const T value, const CPU device) {
    Tensor<T, CPU> tensor(shape, device);

    // TODO

    return tensor;
}


template <typename T> Tensor<T, GPU> full(const vector<size_t> shape, const T value, const GPU device) {
    Tensor<T, GPU> tensor(shape, device);

    // TODO

    return tensor;
}



template <typename T, typename D> Tensor<T, D> zeros(const vector<size_t> shape, const D device) {
    return full<T, D>(shape, (T) 0, device);
}



template <typename T> Tensor<T, CPU> Tensor<T, CPU>::copy() {
    Tensor<T, CPU> tensor(this->shape);

    // TODO
    
    return tensor;
}

template <typename T, typename D> size_t TensorBase<T, D>::numel() const{
    return accumulate(shape.begin(), shape.end(), 1, [](size_t a, size_t b) {
        return a * b;
    });;
}

template <typename T, typename D> size_t TensorBase<T, D>::dim() const{
    return shape.size();
}


// template Tensor<float, CPU> full(vector<size_t>, float, CPU);

template class TensorBase<bool, CPU>;
template class TensorBase<bool, GPU>;
template class TensorBase<int, CPU>;
template class TensorBase<int, GPU>;
template class TensorBase<long, CPU>;
template class TensorBase<long, GPU>;
template class TensorBase<float, CPU>;
template class TensorBase<float, GPU>;
template class TensorBase<double, CPU>;
template class TensorBase<double, GPU>;


template class Tensor<bool, CPU>;
template class Tensor<bool, GPU>;
template class Tensor<int, CPU>;
template class Tensor<int, GPU>;
template class Tensor<long, CPU>;
template class Tensor<long, GPU>;
template class Tensor<float, CPU>;
template class Tensor<float, GPU>;
template class Tensor<double, CPU>;
template class Tensor<double, GPU>;
