#include "tensor.hpp"

#include <hip/amd_detail/amd_hip_runtime.h>
#include <hip/amd_detail/amd_warp_functions.h>
#include <hip/driver_types.h>
#include <hip/hip_runtime.h>
#include <sys/syscall.h>

#include <algorithm>
#include <cassert>
#include <coroutine>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "../utils/hip_utils.hpp"
#include "storage.hpp"
#include "tensor.hip"

using namespace std;

// ###################################### UTILS ###########################################

size_t numel_from_shape(const vector<size_t> &shape) {
	return accumulate(shape.begin(), shape.end(), 1, [](size_t a, size_t b) { return a * b; });
}

vector<size_t> base_stride_from_shape(const vector<size_t> &shape) {
	vector<size_t> stride = vector<size_t>(shape.size(), 1);
	for (size_t i = 0; i < shape.size(); i++) {
		for (size_t j = i + 1; j < shape.size(); j++) {
			stride[i] *= shape[j];
		}
	}
	return stride;
}

// reimplement generator from c++23 as hipcc cannot use it as of now
template <typename T>
struct Generator {
	struct promise_type {
		T current_value;

		auto get_return_object() {
			return Generator{std::coroutine_handle<promise_type>::from_promise(*this)};
		}
		std::suspend_always initial_suspend() {
			return {};
		}
		std::suspend_always final_suspend() noexcept {
			return {};
		}
		std::suspend_always yield_value(T value) {
			current_value = value;
			return {};
		}
		void return_void() {
		}
		void unhandled_exception() {
			std::terminate();
		}
	};

	std::coroutine_handle<promise_type> handle;
	explicit Generator(std::coroutine_handle<promise_type> h) : handle(h) {
	}
	~Generator() {
		if (handle) handle.destroy();
	}

	Generator(const Generator &) = delete;	// Disable copying
	Generator(Generator &&other) noexcept : handle(other.handle) {
		other.handle = nullptr;
	}

	struct iterator {
		std::coroutine_handle<promise_type> handle;

		iterator(std::coroutine_handle<promise_type> h) : handle(h) {
			if (handle) handle.resume();  // Start coroutine
		}

		bool operator!=(std::default_sentinel_t) const {
			return !handle.done();
		}

		iterator &operator++() {
			handle.resume();
			return *this;
		}

		T operator*() const {
			return handle.promise().current_value;
		}
	};

	iterator begin() {
		return iterator{handle};
	}
	std::default_sentinel_t end() {
		return {};
	}
};

Generator<pair<size_t, size_t>> indexes(
	const vector<size_t> &shape, const vector<size_t> &stride, const size_t &offset
) {
	size_t dim = shape.size();
	size_t numel = numel_from_shape(shape);
	vector<size_t> shape_index(dim, 0);
	size_t old_storage_index = offset;

	for (size_t i = 0; i < numel - 1; i++) {
		co_yield pair<size_t, size_t>(i, old_storage_index);
		for (size_t d = dim; d-- > 0;) {
			old_storage_index += stride[d];
			if (++shape_index[d] < shape[d]) {
				break;
			}
			shape_index[d] = 0;
			old_storage_index -= shape[d] * stride[d];
		}
	}
	co_yield pair<size_t, size_t>(numel - 1, old_storage_index);
}


// ###################################### BASE ###########################################

template <typename Derived, typename T, typename D>
TensorBase<Derived, T, D>::TensorBase(const vector<size_t> &shape, const D &device) : shape(shape) {
	stride = base_stride_from_shape(shape);
	offset = 0;
	storage = make_shared<Storage<T, D>>(numel_from_shape(shape), device);
}

template <typename Derived, typename T, typename D>
const size_t &TensorBase<Derived, T, D>::numel() const {
	if (!numel_) {
		numel_ = numel_from_shape(shape);
	}
	return *numel_;
}

template <typename Derived, typename T, typename D>
const size_t TensorBase<Derived, T, D>::dim() const {
	return shape.size();
}

template <typename Derived, typename T, typename D>
bool TensorBase<Derived, T, D>::is_contiguous() const {
	return numel() == storage->size && stride == base_stride_from_shape(shape);
}


template <typename Derived, typename T, typename D>
Derived TensorBase<Derived, T, D>::clone() const {
	Derived tensor(static_cast<const Derived&>(*this));
	if (!this->is_contiguous()) {
		return tensor.contiguous();
	}
	tensor.storage = make_shared<Storage<T, D>>(tensor.numel(), tensor.get_device());
	tensor.write_storage(0, tensor.numel(), this->storage->data, this->get_device());
	return tensor;
}

template <typename Derived, typename T, typename D>
TensorBase<Derived, T, D> &TensorBase<Derived, T, D>::flatten() {
	return reshape({numel()});
}

template <typename Derived, typename T, typename D>
TensorBase<Derived, T, D> &TensorBase<Derived, T, D>::permute(const vector<size_t> &permutations) {
	// permutation is purely reordering shape and stride
	assert(permutations.size() == this->dim());
	vector<size_t> new_stride(this->dim());
	vector<size_t> new_shape(this->dim());
	for (size_t i = 0; i < this->dim(); i++) {
		new_stride[i] = this->stride[permutations[i]];
		new_shape[i] = this->shape[permutations[i]];
	}
	this->stride = new_stride;
	this->shape = new_shape;
	return *this;
}


template <typename Derived, typename T, typename D>
TensorBase<Derived, T, D> &TensorBase<Derived, T, D>::expand(const vector<size_t> &expanded_shape) {
	// expanded dim -> stride[dim] = 0
	// expandable dims: shape[dim] = 1 or additional dim(s) placed before all others
	size_t size_diff = expanded_shape.size() - shape.size();
	assert(size_diff >= 0);
	vector<size_t> new_stride(expanded_shape.size(), 0);
	for (size_t i = 0; i < shape.size(); i++) {
		assert(expanded_shape[i + size_diff] == shape[i] || shape[i] == 1);
		if (expanded_shape[i + size_diff] == shape[i]) {
			new_stride[i + size_diff] = stride[i];
		}
	}
	stride = new_stride;
	shape = expanded_shape;
	return *this;
}


template <typename Derived, typename T, typename D>
TensorBase<Derived, T, D> &TensorBase<Derived, T, D>::reshape(const vector<size_t> &new_shape) {
	// reshape can only be done for contiguous tensors (maybe not only but torch seems to do it that way)
	assert(numel_from_shape(new_shape) == numel());
	if (shape != new_shape) {
		if (is_contiguous()) {
			shape = new_shape;
			stride = base_stride_from_shape(shape);
		} else {
			this->contiguous();
			this->reshape(new_shape);
		}
	}
	return *this;
}



// ###################################### CPU ###########################################


template <typename T>
Tensor<T, CPU> &Tensor<T, CPU>::fill(const T *&values) {
	// TODO
	return *this;
}

template <typename T>
Tensor<T, CPU> &Tensor<T, CPU>::fill(const T &value) {
	if (this->numel() == this->storage->size) {
		write_storage(0, this->numel(), value);
	} else {
	}  // TODO provide implementation for views
	return *this;
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
			r += sub_repr(d + 1, offset + i * this->stride[d]);
			if (i < this->shape[d] - 1) {
				r += ",\n";
				for (size_t j = 0; j < d + 1; j++) {
					r += " ";
				}
			}
		}
	} else {  // d = dim
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
	T *start = this->storage->data + offset;
	std::fill(start, start + n, value);
}

template <typename T>
void Tensor<T, CPU>::write_storage(const size_t &offset, const size_t &n, const T *const &values, const CPU &src) {
	T *start = this->storage->data + offset;
	memcpy(start, values, n * sizeof(T));
}

template <typename T>
void Tensor<T, CPU>::write_storage(const size_t &offset, const size_t &n, const T *const &values, const GPU &src) {
	T *start = this->storage->data + offset;
	HIP_CHECK(hipSetDevice(src));
	HIP_CHECK(hipMemcpy(start, values, n * sizeof(T), hipMemcpyDeviceToHost));
}


template <typename T>
Tensor<T, CPU> &Tensor<T, CPU>::contiguous() {
	if (!this->is_contiguous()) {
		shared_ptr<Storage<T, CPU>> new_storage = make_shared<Storage<T, CPU>>(this->numel(), this->storage->device);

		for (pair<size_t, size_t> idxs : indexes(this->shape, this->stride, this->offset)) {
			new_storage->data[idxs.first] = this->storage->data[idxs.second];
		}

		this->storage = new_storage;
		this->stride = base_stride_from_shape(this->shape);
		this->offset = 0;
	}
	return *this;
}

template <typename T>
Tensor<T, CPU> Tensor<T, CPU>::to(const CPU &device) const {
    return *this;
}

template <typename T>
Tensor<T, GPU> Tensor<T, CPU>::to(const GPU &device) const {
	Tensor<T, GPU> new_tensor(this->shape, device);
    const T* data;
	if (!this->is_contiguous()) {
        Tensor<T, CPU> clone = this->clone();
        data = clone.get_storage()->data;
	} else {
        data = this->get_storage()->data;
	}
    new_tensor.write_storage(0, new_tensor.numel(), data, this->get_device());
    return new_tensor;
}

// ###################################### GPU ###########################################

template <typename T>
Tensor<T, GPU> &Tensor<T, GPU>::fill(const T *&values) {
	// TODO
	return *this;
}


template <typename T>
Tensor<T, GPU> &Tensor<T, GPU>::fill(const T &value) {
	if (this->numel() == this->storage->size) {
		write_storage(0, this->numel(), value);
	} else {
	}  // TODO provide implementation for views
	return *this;
}

template <typename T>
void Tensor<T, GPU>::write_storage(const size_t &offset, const size_t &n, const T &value) {
	T *start = this->storage->data + offset;
	HIP_CHECK(hipMemset(start, value, n * sizeof(T)));
}

template <typename T>
void Tensor<T, GPU>::write_storage(const size_t &offset, const size_t &n, const T *const &values, const CPU &src) {
	T *start = this->storage->data + offset;
	HIP_CHECK(hipSetDevice(this->storage->device));
	HIP_CHECK(hipMemcpy(start, values, n * sizeof(T), hipMemcpyHostToDevice));
}

template <typename T>
void Tensor<T, GPU>::write_storage(const size_t &offset, const size_t &n, const T *const &values, const GPU &src) {
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

template <typename T>
Tensor<T, GPU> &Tensor<T, GPU>::contiguous() {
	if (!this->is_contiguous()) {
		shared_ptr<Storage<T, GPU>> new_storage = make_shared<Storage<T, GPU>>(this->numel(), this->storage->device);
		vector<size_t> new_stride = base_stride_from_shape(this->shape);

		HIP_CHECK(hipSetDevice(this->get_device().id));

		size_t dim = this->dim();
        size_t size_bytes = dim * sizeof(size_t);
        size_t* device_shape, *device_src_stride, *device_dest_stride;
        HIP_CHECK(hipMalloc(&device_shape, size_bytes));
        HIP_CHECK(hipMalloc(&device_src_stride, size_bytes));
        HIP_CHECK(hipMalloc(&device_dest_stride, size_bytes));

        HIP_CHECK(hipMemcpy(device_shape, this->shape.data(), size_bytes ,hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(device_src_stride, this->stride.data(), size_bytes ,hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(device_dest_stride, new_stride.data(), size_bytes ,hipMemcpyHostToDevice));
        
		const HIPTensor<T>
			src{this->storage->data, dim, this->numel(), device_shape, device_src_stride, this->offset};
		const HIPTensor<T> dest{new_storage->data, dim, this->numel(), device_shape, device_dest_stride, 0};
        hipPointerAttribute_t attributes;
        HIP_CHECK(hipPointerGetAttributes(&attributes, src.data));
        HIP_CHECK(hipPointerGetAttributes(&attributes, dest.data));


        hipDeviceProp_t prop;
        HIP_CHECK(hipGetDeviceProperties(&prop, this->get_device().id));
        int max_threads_per_block = prop.maxThreadsPerBlock;
        int warp_size = prop.warpSize;

        int threads_per_block = min(max_threads_per_block, ((this->numel() + warp_size - 1) / warp_size) * warp_size);
        dim3 block_dim(threads_per_block, 1, 1);
        dim3 grid_dim((this->numel() + threads_per_block - 1) / threads_per_block, 1, 1);

		write_to_contiguous<T><<<grid_dim, block_dim>>>(src, dest);
        HIP_CHECK(hipDeviceSynchronize());

        HIP_CHECK(hipFree(device_shape));
        HIP_CHECK(hipFree(device_src_stride));
        HIP_CHECK(hipFree(device_dest_stride));

		this->storage = new_storage;
		this->stride = new_stride;
		this->offset = 0;
	}
	return *this;
}

template <typename T>
Tensor<T, CPU> Tensor<T, GPU>::to(const CPU &device) const {
	Tensor<T, CPU> new_tensor(this->shape, device);
    const T* data;
	if (!this->is_contiguous()) {
        Tensor<T, GPU> clone = this->clone();
        data = clone.get_storage()->data;
	} else {
        data = this->get_storage()->data;
	}
    new_tensor.write_storage(0, new_tensor.numel(), data, this->get_device());
    return new_tensor;
}

template <typename T>
Tensor<T, GPU> Tensor<T, GPU>::to(const GPU &device) const {
	if (device.id == this->get_device().id) {
        return *this;
	}
	Tensor<T, GPU> new_tensor(this->shape, device);
    const T* data;
	if (!this->is_contiguous()) {
        Tensor<T, GPU> clone = this->clone();
        data = clone.get_storage()->data;
	} else {
        data = this->get_storage()->data;
	}
    new_tensor.write_storage(0, new_tensor.numel(), data, this->get_device());
    return new_tensor;
}

#define INSTANTIATE_TENSOR(T, D)                   \
	template class TensorBase<Tensor<T, D>, T, D>; \
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
