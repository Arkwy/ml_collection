#ifndef TENSOR_H
#define TENSOR_H


#include <algorithm>
#include <array>
#include <cstddef>
#include <memory>
#include <numeric>
#include <variant>

#include <hip/amd_detail/amd_hip_runtime.h>
#include <hip/driver_types.h>
#include <hip/hip_runtime.h>

#include "../utils/hip_utils.hpp"
#include "device.hpp"
#include "storage.hpp"


using namespace std;


template <size_t N>
size_t numel_from_shape(const array<size_t, N> &shape) {
    return std::accumulate(shape.begin(), shape.end(), 1, [](size_t a, size_t b) { return a * b; });
}


template <size_t N>
array<size_t, N> base_stride_from_shape(const array<size_t, N> &shape) {
    array<size_t, N> stride;
    for (size_t i = 0; i < shape.size(); i++) {
        stride[i] = 1;
        for (size_t j = i + 1; j < shape.size(); j++) {
            stride[i] *= shape[j];
        }
    }
    return stride;
}


struct Slice {
    int start, end, step;
};
struct FullSlice {};
struct Expansion {};
struct NewDim {};


using TensorIndexer = variant<int, FullSlice, Slice, Expansion, NewDim>;


template <typename Derived, typename T, typename D, size_t N>
class TensorBase {
    using This = TensorBase<Derived, T, D, N>;


  protected:
    shared_ptr<Storage<T, D>> storage;
    array<size_t, N> shape;
    array<size_t, N> stride;
    size_t offset;


    TensorBase(const array<size_t, N> &shape, const D &device = D()) : shape(shape) {
        stride = base_stride_from_shape(shape);
        offset = 0;
        storage = make_shared<Storage<T, D>>(numel_from_shape(shape), device);
    }


    virtual void write_storage(const size_t &offset, const size_t &n, const T &value) = 0;
    virtual void write_storage(const size_t &offset, const size_t &n, const T *const &values, const CPU &src) = 0;
    virtual void write_storage(const size_t &offset, const size_t &n, const T *const &values, const GPU &src) = 0;


  public:
    size_t numel() const { return numel_from_shape(shape); }


    bool is_contiguous() const { return numel() != storage->size && stride == base_stride_from_shape(shape); }


    const shared_ptr<Storage<T, D>> get_storage() const { return storage; }


    const D &get_device() const { return storage->device; }


    const array<size_t, N> &get_shape() const { return shape; }


    const array<size_t, N> &get_stride() const { return stride; }


    const size_t &get_offset() const { return offset; }


    static Derived full(const array<size_t, N> &shape, const T &value, const D &device = D()) {
        Derived tensor(shape, device);
        tensor.fill(value);
        return tensor;
    }


    static Derived zeros(const array<size_t, N> &shape, const D &device = D()) {
        return This::full(shape, (T)0, device);
    }


    static Derived ones(const array<size_t, N> &shape, const D &device = D()) {
        return This::full(shape, (T)1, device);
    }


    static Derived empty(const array<size_t, N> &shape, const D &device = D()) { return Derived(shape, device); }


    static Derived arange(const size_t &n, const D &device = D())
        requires(!std::same_as<T, bool>)
    {
        Derived tensor({n}, device);
        vector<T> range(n);
        for (int i = 0; i < n; i++) {
            range[i] = i;
        }
        tensor.write_storage(0, n, range.data(), CPU());
        return tensor;
    }


    Derived clone() {
        Derived tensor(shape, storage->device);
        if (!is_contiguous()) {
            tensor.write_storage(0, numel(), storage->data, storage->device);
        } else {
            throw runtime_error("view clone not implemented");
        } // TODO provide implementation for views
        return tensor;
    }


    Derived copy() {
        Derived tensor(shape, storage->device);
        tensor.storage = storage;
        return tensor;
    }


    This &fill(const T &value) {
        if (!is_contiguous()) {
            write_storage(0, this->numel(), value);
        } else {
            throw runtime_error("view fill not implemented");
        } // TODO provide implementation for views
        return *this;
    }


    // This &contiguous();


    // This &fill(const T *&values);


    // This &reshape(const array<size_t, N> &new_shape) {
    //     assert(numel_from_shape(new_shape) == numel());
    //     if (!is_contiguous()) {
    //         shape = new_shape;
    //         stride = base_stride_from_shape(shape);
    //         // offset already 0
    //         return *this;
    //     } else {
    //         throw runtime_error("view flatten not implemented");
    //     } // TODO provide implementation for views
    // }


    // template <size_t M>
    // Derived reshape(const array<size_t, M> &new_shape);


    // This &flatten() { return reshape({numel()}); }


    // This &operator[](const initializer_list<TensorIndexer> &slices);
};



template <typename T, typename D, size_t N>
class Tensor : public TensorBase<Tensor<T, D, N>, T, D, N> {};



template <typename T, size_t N>
class Tensor<T, CPU, N> : public TensorBase<Tensor<T, CPU, N>, T, CPU, N> {
    using Base = TensorBase<Tensor<T, CPU, N>, T, CPU, N>;
    friend class TensorBase<Tensor<T, CPU, N>, T, CPU, N>;


  private:
    string sub_repr(const size_t &d, const size_t &offset) const {
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


    void write_storage(const size_t &offset, const size_t &n, const T &value) override {
        T *start = this->storage->data + offset;
        std::fill(start, start + n, value);
    }


    void write_storage(const size_t &offset, const size_t &n, const T *const &values, const CPU &src) override {
        T *start = this->storage->data + offset;
        memcpy(start, values, n * sizeof(T));
    }


    void write_storage(const size_t &offset, const size_t &n, const T *const &values, const GPU &src) override {
        T *start = this->storage->data + offset;
        HIP_CHECK(hipSetDevice(src));
        HIP_CHECK(hipMemcpy(start, values, n * sizeof(T), hipMemcpyDeviceToHost));
    }


  public:
    Tensor(const array<size_t, N> &shape, const CPU &device = CPU()) : Base(shape, device) {}
    string repr() const { return sub_repr(0, 0); }
};



template <typename T, size_t N>
class Tensor<T, GPU, N> : public TensorBase<Tensor<T, GPU, N>, T, GPU, N> {
    using Base = TensorBase<Tensor<T, GPU, N>, T, GPU, N>;
    friend class TensorBase<Tensor<T, GPU, N>, T, GPU, N>;


  private:
    void write_storage(const size_t &offset, const size_t &n, const T &value) override {
        T *start = this->storage->data + offset;
        HIP_CHECK(hipMemset(start, value, n * sizeof(T)));
    }


    void write_storage(const size_t &offset, const size_t &n, const T *const &values, const CPU &src) override {
        T *start = this->storage->data + offset;
        HIP_CHECK(hipSetDevice(this->storage->device));
        HIP_CHECK(hipMemcpy(start, values, n * sizeof(T), hipMemcpyHostToDevice));
    }


    void write_storage(const size_t &offset, const size_t &n, const T *const &values, const GPU &src) override {
        T *start = this->storage->data + offset;
        GPU device = this->storage->device;
        HIP_CHECK(hipSetDevice(device));
        if (src.id == device.id) {
            HIP_CHECK(hipMemcpy(start, values, n * sizeof(T), hipMemcpyDeviceToDevice));
        } else {
            throw runtime_error("Data copy between different GPUs not implemented yet");
            // TODO
        }
    }


  public:
    Tensor(const array<size_t, N> &shape, const GPU &device = GPU()) : Base(shape, device) {}
};



#endif
