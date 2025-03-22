#ifndef DEVICE_ARRAY_H
#define DEVICE_ARRAY_H

#include <hip/hip_runtime.h>
#include <sys/types.h>

#include "../../utils/hip_utils.hpp"
#include "../../utils/logger.hpp"


template <typename T>
struct DeviceArray {
    const uint size;
    const uint device_id = 0;
    T* const data;

    DeviceArray(const uint& size, const uint& device_id = 0)
        : size(size), device_id(device_id), data(alloc_device(device_id)) {}


    ~DeviceArray() {
        hipError_t status = hipFree(data);
        if (status != hipSuccess) {
            LOG(LOG_LEVEL_ERROR,
                "Error: HIP reports %s during the destruction of DeviceArray.",
                hipGetErrorString(status));
        }
    }

    DeviceArray(DeviceArray& other) = delete;


    T* alloc_device(const uint& device_id = 0) const {
        T* data;
        HIP_CHECK(hipSetDevice(device_id));
        HIP_CHECK(hipMalloc(&data, size * sizeof(T)));
        return data;
    }
};

#endif
