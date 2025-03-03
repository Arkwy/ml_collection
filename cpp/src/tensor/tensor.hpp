#ifndef TENSOR_H
#define TENSOR_H

#include <cstddef>
#include <memory>
#include <optional>
#include <variant>
#include <vector>

#include "device.hpp"
#include "storage.hpp"

using namespace std;


struct Slice {
	int start, end, step;
};
struct FullSlice {};
struct Expansion {};
struct NewDim {};

using TensorIndexer = variant<int, FullSlice, Slice, Expansion, NewDim>;

template <typename T, typename D>
class Tensor;

template <typename Derived, typename T, typename D>
class TensorBase {
	using This = TensorBase<Derived, T, D>;

  protected:
	shared_ptr<Storage<T, D>> storage;
	vector<size_t> shape;
	vector<size_t> stride;
	size_t offset;
	mutable optional<size_t> numel_;

	TensorBase(const vector<size_t> &shape, const D &device = D());

	virtual void write_storage(const size_t &offset, const size_t &n, const T &value) = 0;
	virtual void write_storage(const size_t &offset, const size_t &n, const T *const &values, const CPU &src) = 0;
	virtual void write_storage(const size_t &offset, const size_t &n, const T *const &values, const GPU &src) = 0;
	virtual This &fill(const T &value) = 0;
    virtual This &fill(const T* const &values) = 0;

  public:
	const size_t &numel() const;
	const size_t dim() const;
	bool is_contiguous() const;
	const shared_ptr<Storage<T, D>> get_storage() const {
		return storage;
	}
	const D &get_device() const {
		return storage->device;
	}
	const vector<size_t> &get_shape() const {
		return shape;
	}
	const vector<size_t> &get_stride() const {
		return stride;
	}
	const size_t &get_offset() const {
		return offset;
	}
	static Derived full(const vector<size_t> &shape, const T &value, const D &device = D()) {
		Derived tensor(shape, device);
		tensor.fill(value);
		return tensor;
	}
	static Derived zeros(const vector<size_t> &shape, const D &device = D()) {
		return This::full(shape, (T)0, device);
	}
	static Derived ones(const vector<size_t> &shape, const D &device = D()) {
		return This::full(shape, (T)1, device);
	}
	static Derived empty(const vector<size_t> &shape, const D &device = D()) {
		return Derived(shape, device);
	}
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
	Derived clone() const; // deep copy
	virtual This &contiguous() = 0;
    virtual Tensor<T, CPU> to(const CPU& device) const = 0;
    virtual Tensor<T, GPU> to(const GPU& device) const = 0;
	This &reshape(const vector<size_t> &new_shape);
	This &permute(const vector<size_t> &permutations);
	This &expand(const vector<size_t> &expanded_shape);
	This &flatten();
	This &operator[](const initializer_list<TensorIndexer> &slices);
};



template <typename T, typename D>
class Tensor : public TensorBase<Tensor<T, D>, T, D> {};



template <typename T>
class Tensor<T, CPU> : public TensorBase<Tensor<T, CPU>, T, CPU> {
	using Base = TensorBase<Tensor<T, CPU>, T, CPU>;
	using This = Tensor<T, CPU>;
	friend class TensorBase<Tensor<T, CPU>, T, CPU>;
	friend class Tensor<T, GPU>;

  private:
	string sub_repr(const size_t &d, const size_t &offset) const;
	void write_storage(const size_t &offset, const size_t &n, const T &value) override;
	void write_storage(const size_t &offset, const size_t &n, const T *const &values, const CPU &src) override;
	void write_storage(const size_t &offset, const size_t &n, const T *const &values, const GPU &src) override;
	This &fill(const T &value) override;
	This &fill(const T* const &values) override;

  public:
	Tensor(const vector<size_t> &shape, const CPU &device = CPU()) : Base(shape, device) {
	}
	This &contiguous() override;
    Tensor<T, CPU> to(const CPU& device) const override;
    Tensor<T, GPU> to(const GPU& device) const override;
	string repr() const;
};



template <typename T>
class Tensor<T, GPU> : public TensorBase<Tensor<T, GPU>, T, GPU> {
	using Base = TensorBase<Tensor<T, GPU>, T, GPU>;
	using This = Tensor<T, GPU>;
	friend class TensorBase<Tensor<T, GPU>, T, GPU>;
	friend class Tensor<T, CPU>;

  private:
	void write_storage(const size_t &offset, const size_t &n, const T &value) override;
	void write_storage(const size_t &offset, const size_t &n, const T *const &values, const CPU &src) override;
	void write_storage(const size_t &offset, const size_t &n, const T *const &values, const GPU &src) override;
	This &fill(const T &value) override;
	This &fill(const T* const &values) override;

  public:
	Tensor(const vector<size_t> &shape, const GPU &device = GPU()) : Base(shape, device) {
	}
	This &contiguous() override;
    Tensor<T, CPU> to(const CPU& device) const override;
    Tensor<T, GPU> to(const GPU& device) const override;
};


#endif
