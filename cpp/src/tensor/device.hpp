#ifndef DEVICE_H
#define DEVICE_H

#include <cstddef>
#include <hip/hip_runtime.h>
#include <stdexcept>

struct GPU {
    const size_t id;
    GPU(const size_t& id = 0) : id(check_id(id)){}
    operator int() const {
        return id;
    }

  private:
    size_t check_id(const size_t &id) {
        int device_count;
        HIP_CHECK(hipGetDeviceCount(&device_count));
        if (id >= device_count) {
            throw std::invalid_argument("Device id greater than the number of available GPUs."); 
        }

        return id;
    }

};

struct CPU {};

#endif
