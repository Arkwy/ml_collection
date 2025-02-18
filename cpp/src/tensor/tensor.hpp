#ifndef TENSOR_H
#define TENSOR_H

#include <memory>
#include <vector>

#include "device.hpp"
#include "storage.hpp"

using namespace std;

template <typename Derived, typename T, typename D>
class TensorBase {
    using This = TensorBase<Derived, T, D>;
  protected:
    shared_ptr<Storage<T, D>> storage;
    vector<size_t> shape;
    vector<size_t> strides;
    vector<size_t> offsets;

    virtual void write_storage(const size_t offset, const size_t n, const T value) = 0;
    virtual void write_storage(const size_t offset, const size_t n, const T* values, const Device& src) = 0;

  public:
    TensorBase(const vector<size_t> &shape, const D &device = D());
    size_t numel() const;
    size_t dim() const;
    bool is_view() const;
    static Derived full(const vector<size_t> &shape, const T value, const D &device = D()) {
        Derived tensor(shape, device);
        tensor.fill(value);
        return tensor;
    }
    This& fill(const T value);
};



template <typename T, typename D>
class Tensor : public TensorBase<Tensor<T, D>, T, D> {};



template <typename T>
class Tensor<T, CPU> : public TensorBase<Tensor<T, CPU>, T, CPU> {
    using Base = TensorBase<Tensor<T, CPU>, T, CPU>;

  private:
    string sub_repr(const size_t d, const size_t offset) const;
    void write_storage(const size_t offset, const size_t size, const T value) override;
    void write_storage(const size_t offset, const size_t size, const T* values, const Device& src) override;

  public:
    Tensor(const vector<size_t> &shape, const CPU &device = CPU()) : Base(shape, device) {}
    string repr() const;
};



template <typename T>
class Tensor<T, GPU> : public TensorBase<Tensor<T, GPU>, T, GPU> {
    using Base = TensorBase<Tensor<T, GPU>, T, GPU>;

  private:
    void write_storage(const size_t offset, const size_t n, const T value) override;
    void write_storage(const size_t offset, const size_t n, const T* values, const Device& src) override;

  public:
    Tensor(const vector<size_t> &shape, const GPU &device = GPU()) : Base(shape, device) {}
};

#endif
