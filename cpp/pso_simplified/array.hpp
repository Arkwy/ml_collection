#ifndef DEVICE_ARRAY_H
#define DEVICE_ARRAY_H

#include <hip/driver_types.h>
#include <hip/hip_runtime.h>
#include <sys/types.h>

#include "../utils/hip_utils.hpp"

template <typename T>
struct HostArray {
    const uint size;
    const bool pinned = true;
    T* const data;

    HostArray(const uint& size, const bool& pinned)
        : size(size), pinned(pinned), data(alloc_data()) {}


    ~HostArray() {
        if (pinned) {
            HIP_CHECK_NOEXCEPT(hipHostFree(data));
        } else {
            delete[] data;
        }
    }

    HostArray(HostArray& other) = delete;

  private:
    T* alloc_data() {
        if (pinned) {
            T* data = nullptr;
            HIP_CHECK(hipHostMalloc(data, size * sizeof(T)));
            return data;
        } else {
            return new T[size];
        }
    }
};


template <typename T>
struct DeviceArray {
    const uint size;
    const uint device_id = 0;
    T* const data;

    DeviceArray(const uint& size, const uint& device_id = 0)
        : size(size), device_id(device_id), data(alloc_device(device_id)) {}


    ~DeviceArray() {
        HIP_CHECK_NOEXCEPT(hipFree(data));
    }

    DeviceArray(DeviceArray& other) = delete;


    T* alloc_device(const uint& device_id = 0) const {
        T* data;
        HIP_CHECK(hipSetDevice(device_id));
        HIP_CHECK(hipMalloc(&data, size * sizeof(T)));
        return data;
    }
};


template <typename T>
void arrcpy(const DeviceArray<T>& dst, const HostArray<T>& src) {
    assert(dst.size == src.size);
    HIP_CHECK(hipSetDevice(dst.device_id));
    HIP_CHECK(hipMemcpy(dst.data, src.data, dst.size * sizeof(T), hipMemcpyHostToDevice));
}


template <typename T>
void arrcpy(const HostArray<T>& dst, const DeviceArray<T>& src) {
    assert(dst.size == src.size);
    HIP_CHECK(hipSetDevice(src.device_id));
    HIP_CHECK(hipMemcpy(dst.data, src.data, dst.size * sizeof(T), hipMemcpyDeviceToHost));
}


template <typename T>
void arrcpy(const DeviceArray<T>& dst, const DeviceArray<T>& src) {
    assert(dst.size == src.size);
    assert(dst.device_id == src.device_id && "Inter devices copy not supported yet.");
    HIP_CHECK(hipSetDevice(src.device_id));
    HIP_CHECK(hipMemcpy(dst.data, src.data, dst.size * sizeof(T), hipMemcpyDeviceToDevice));
}


#endif
