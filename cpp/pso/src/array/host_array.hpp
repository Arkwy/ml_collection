#ifndef HOST_ARRAY_H
#define HOST_ARRAY_H

#include <hip/amd_detail/amd_hip_runtime.h>
#include <hip/hip_runtime.h>
#include <sys/types.h>

#include <cstddef>

#include "../utils/logger.hpp"
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
            hipError_t status(hipHostFree(data));
            if (status != hipSuccess) {
                LOG(LOG_LEVEL_ERROR,
                    "Error: HIP reports %s during the destruction of HostArray.",
                    hipGetErrorString(status));
            }
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

#endif
