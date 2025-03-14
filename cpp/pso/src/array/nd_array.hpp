#ifndef ND_ARRAY_H
#define ND_ARRAY_H

#include <hip/amd_detail/amd_hip_runtime.h>
#include <hip/driver_types.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <cstring>
#include <iomanip>
#include <ios>
#include <memory>
#include <sstream>
#include <stdexcept>

#include "synced_array.hpp"
#include "utils.hpp"

template <typename T, size_t N, size_t... M>
struct NDArrayBase {
    using This = NDArrayBase<T, N, M...>;
    constexpr static const size_t dim = 1 + sizeof...(M);
    constexpr static const size_t size = mul<N, M...>::value;

    const T* const get_device() const { return synced_array->get_device(offset, size); }

    T* const get_mut_device() const { return synced_array->get_mut_device(offset, size); }

    const T* const get_host() const { return synced_array->get_host(offset, size); }

    T* const get_mut_host() const { return synced_array->get_mut_host(offset, size); }

    size_t device_id() const { return synced_array->device_id; }

    void fill(const T& value) const {
        T* data_ptr = this->get_mut_host();
        std::fill(data_ptr, data_ptr + this->size, value);
    }

    void device_copy(const This& other) const {
        if (this->device_id() != other.device_id()) {
            throw std::runtime_error("Data copy between different GPUs not implemented yet.");
        }
        T* const this_data_ptr = this->get_mut_device();
        const T* const other_data_ptr = other.get_device();
        HIP_CHECK(hipSetDevice(this->device_id()));
        HIP_CHECK(hipMemcpy(this_data_ptr, other_data_ptr, this->size * sizeof(T), hipMemcpyDeviceToDevice));
    }

    void host_copy(const This& other) const {
        T* this_data_ptr = this->get_mut_host();
        T* other_data_ptr = other.get_host();
        memcpy(this_data_ptr, other_data_ptr, this->size * sizeof(T));
    }

  protected:
    const std::shared_ptr<SyncedArray<T>> synced_array;
    const bool is_view = false;
    const size_t offset = 0;

    NDArrayBase(const size_t& device_id = 0) : synced_array(std::make_shared<SyncedArray<T>>(size, device_id)) {}

    NDArrayBase(const std::array<T, size>& data, const size_t& device_id = 0) : NDArrayBase(device_id) {
        memcpy(synced_array->get_mut_host(offset, size), data.data(), size * sizeof(T));
    }

    NDArrayBase(const std::shared_ptr<SyncedArray<T>>& synced_array, const size_t& offset)
        : synced_array(synced_array), is_view(true), offset(offset) {}
};



template <typename T, size_t N, size_t... M>
struct NDArray : public NDArrayBase<T, N, M...> {
    using Base = NDArrayBase<T, N, M...>;

    NDArray(const size_t& device_id = 0) : Base(device_id) {}

    NDArray(const std::array<T, Base::size>& data, const size_t& device_id = 0) : Base(data, device_id) {}

    NDArray(const std::shared_ptr<SyncedArray<T>>& synced_array, const size_t& offset) : Base(synced_array, offset) {}

    NDArray<T, M...> get_index(const size_t& index) const {
        assert(index < N);
        return NDArray<T, M...>(this->synced_array, this->offset + index * mul<M...>::value);
    }

    NDArray<T, M...> operator[](const size_t& index) const { return this->get_index(index); }

    std::string repr(const size_t& offset = 0) const {
        std::ostringstream oss;
        oss << "[" << this->get_index(0).repr(offset + 1);
        for (size_t i = 1; i < N; i++) {
            oss << std::endl << std::setw(offset + 1) << " " << this->get_index(i).repr(offset + 1);
        }
        oss << "]";
        return oss.str();
    }
};

template <typename T, size_t N>
struct NDArray<T, N> : public NDArrayBase<T, N> {
    using Base = NDArrayBase<T, N>;

    NDArray(const size_t& device_id = 0) : Base(device_id) {}

    NDArray(const std::array<T, Base::size>& data, const size_t& device_id = 0) : Base(data, device_id) {}

    NDArray(const std::shared_ptr<SyncedArray<T>>& synced_array, const size_t& offset) : Base(synced_array, offset) {}

    T get_index(const size_t& index) const {
        assert(index < N);
        return this->get_host()[this->offset + index];
    }

    T& get_index(const size_t& index) {
        assert(index < N);
        return this->get_mut_host()[index];
    }

    T operator[](const size_t& index) const { return this->get_index(index); }

    T& operator[](const size_t& index) { return this->get_index(index); }

    std::string repr(const size_t& offset = 0) const {
        const T* const arr = this->get_host();
        std::ostringstream oss;
        oss << "[";
        for (size_t i = 0; i < N; i++) {
            oss << std::setw(9) << std::scientific << std::setprecision(2) << arr[i];
            if (i != N - 1) oss << ", ";
        }
        oss << "]";
        return oss.str();
    }
};



template <typename T, size_t N, size_t... M>
std::ostream& operator<<(std::ostream& os, const NDArray<T, N, M...>& array) {
    return os << array.repr();
}

#endif
