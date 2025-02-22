#ifndef STORAGE_H
#define STORAGE_H

#include <cstddef>
#include <hip/hip_runtime.h>

#include "../utils/hip_utils.hpp"
#include "device.hpp"


template <typename T, typename D> struct Storage {
    const size_t size;
    T* const data;
    const D device;
private:
    Storage();
};



template <typename T> struct Storage<T, CPU> {
    const CPU device;
    const size_t size;
    T* const data;

    Storage(const size_t& size, const CPU& device = CPU()) : device(device), size(size), data(init_data(size)) {}
    Storage(Storage<T, CPU>& storage) = delete;
    ~Storage() {
        delete[] this->data;
    }
    T operator[](const int& index) {
        assert(index < this->size);

        int i = index;

        if (index < 0) {
            i += this->size;
            assert(i >= 0);
        }

        return this->data[i];
    }

private:
    T* const init_data(const size_t& size) const {
        return new T[size];
    }
};



template <typename T> struct Storage<T, GPU> {
    const GPU device;
    const size_t size;
    T* const data;

    Storage(const size_t& size, const GPU& device = GPU()) : device(device), size(size), data(init_data(size, device)) {}
    Storage(Storage<T, GPU>& storage) = delete;
    ~Storage() {
        HIP_CHECK(hipSetDevice(this->device));
        HIP_CHECK(hipFree(this->data));
    };
private:
    T* const init_data(const size_t& size, const GPU& device) const {
        T* data = nullptr;
        HIP_CHECK(hipSetDevice(device));
        HIP_CHECK(hipMalloc(&data, size * sizeof(T)));
        return data;
    }
};

#endif
