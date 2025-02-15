#ifndef TENSOR_H
#define TENSOR_H

#include <memory>
#include <vector>

#include "storage.hpp"
#include "device.hpp"

using namespace std;

template <typename T, typename D>
class TensorBase {
  protected:
    shared_ptr<Storage<T, D>> storage;
    vector<size_t> shape;
    vector<size_t> strides;
    vector<size_t> offsets;

  public:
    TensorBase(const vector<size_t> shape, const D device = D());
    size_t numel() const;
    size_t dim() const;
};

template <typename T, typename D> class Tensor : public TensorBase<T, D> {
  public:
    Tensor(const vector<size_t> shape, const D device = D());
};

template <typename T>
class Tensor<T, CPU>: public TensorBase<T, CPU> {
  private:
    string sub_repr(const size_t d, const size_t offset) const;

  public:
    Tensor(const vector<size_t> shape, const CPU = CPU());
    Tensor<T, CPU> copy();
    string repr() const;
};


template <typename T>
class Tensor<T, GPU>:  public TensorBase<T, GPU> {

  public:
    Tensor(const vector<size_t> shape, const GPU = GPU());
    Tensor<T, GPU> copy();
};

template <typename T, typename D>
Tensor<T, D> empty(const vector<size_t> shape, const D device = D());

template <typename T, typename D>
Tensor<T, D> full(const vector<size_t> shape, const T value, const D device = D());

template <typename T, typename D>
Tensor<T, D> zeros(const vector<size_t> shape, const D device = D());

#endif
