#include <hip/amd_detail/amd_hip_runtime.h>
#include <stdexcept>

#include "device.hpp"
#include "../utils/hip_utils.hpp"

using namespace std;

GPU::GPU(const size_t id) {

    int device_count;
    HIP_CHECK(hipGetDeviceCount(&device_count));
    if (id >= device_count) {
        if (id == 0) {
            // TODO
        }
        throw invalid_argument("Device id greater than the number of available GPUs."); 
    }

    this->id = id;
}

Device::operator int() const {
    return id;
}

int Device::get_id() const {
    return id;
}

CPU::CPU() {
    id = 0;
}
