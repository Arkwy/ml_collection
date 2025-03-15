#ifndef SYNCED_ARRAY_H
#define SYNCED_ARRAY_H

// #include <hip/amd_detail/amd_hip_runtime.h>
// #include <hip/driver_types.h>
#include <hip/amd_detail/amd_hip_runtime.h>
#include <hip/hip_runtime.h>
#include <sys/types.h>

#include <algorithm>
#include <cstddef>

#include "../utils/hip_utils.hpp"
#include "../utils/logger.hpp"


//// bitmask based DesyncStatus, may avoid useless sync but bigger overhead
// struct DesyncStatus {
//     std::vector<bool> device_modified;
//     std::vector<bool> host_modified;

//     DesyncStatus(size_t size) : device_modified(size_, false), host_modified(size_, false) {}

//     void mark_device_modified(size_t start, size_t end) {
//         std::fill(device_modified.begin() + start, device_modified.begin() + end, true);
//     }

//     void mark_host_modified(size_t start, size_t end) {
//         std::fill(host_modified.begin() + start, host_modified.begin() + end, true);
//     }

//     bool has_overlap(size_t start, size_t end, bool check_device) const {
//         const auto& modified = check_device ? device_modified : host_modified;
//         return std::any_of(modified.begin() + start, modified.begin() + end, [](bool v) { return v; });
//     }

//     void clear() {
//         std::fill(device_modified.begin(), device_modified.end(), false);
//         std::fill(host_modified.begin(), host_modified.end(), false);
//     }
// };

struct DesyncStatus {
    bool device_side = false;
    bool host_side = false;
    size_t start = 0;
    size_t end = 0;

    DesyncStatus operator+(const DesyncStatus& other) const {
        if (!(device_side || host_side)) return other;
        DesyncStatus new_status{
            device_side || other.device_side,
            host_side || other.host_side,
            std::min(start, other.start),
            std::max(end, other.end)
        };
        assert(!(new_status.device_side && new_status.host_side));
        return new_status;
    }

    void operator+=(const DesyncStatus& other) { *this = *this + other; }

    bool overlaps_section(const size_t& start_, const size_t& end_) const {
        return (device_side || host_side) && (start_ < end) && (end_ > start);
    }

    size_t size() const { return end - start; }
};

template <typename T>
struct DualArray {
    const size_t size;
    const size_t device_id = 0;
    const bool pinned = true;

    DualArray(const size_t& size, const size_t& device_id = 0, const bool& pinned = true)
        : size(size), device_id(device_id), pinned(pinned), device_data(alloc_device(device_id)), host_data(alloc_host()) {}


    ~DualArray() {
        if (pinned) {
            hipError_t h_status = hipHostFree(host_data);
            if (h_status != hipSuccess) {
                LOG(LOG_LEVEL_ERROR,
                    "Error: HIP reports %s during the destruction of SyncedArray (double free ?).",
                    hipGetErrorString(h_status));
            }
        } else {
            delete[] host_data;
        }
        hipError_t status = hipFree(device_data);
        if (status != hipSuccess) {
            LOG(LOG_LEVEL_ERROR,
                "Error: HIP reports %s during the destruction of SyncedArray (double free ?).",
                hipGetErrorString(status));
        }
    }

    DualArray(DualArray& other) = delete;

    const T* const get_device(const size_t& start, const size_t& size) const {
        if (desync_status.host_side && desync_status.overlaps_section(start, start + size)) {
            sync_with_host();
        }
        return device_data + start;
    }

    T* const get_mut_device(const size_t& start, const size_t& size) const {
        if (desync_status.host_side) {
            sync_with_host();
        }
        desync_status += DesyncStatus{true, false, start, start + size};
        return device_data + start;
    }

    const T* const get_host(const size_t& start, const size_t& size) const {
        if (desync_status.device_side && desync_status.overlaps_section(start, start + size)) {
            sync_with_device();
        }
        return host_data + start;
    }

    T* const get_mut_host(const size_t& start, const size_t& size) const {
        if (desync_status.device_side) {
            sync_with_device();
        }
        desync_status += DesyncStatus{false, true, start, start + size};
        return host_data + start;
    }

  private:
    T* const device_data;
    T* const host_data;
    mutable DesyncStatus desync_status;

    T* alloc_device(const size_t& device_id = 0) const {
        T* data;
        HIP_CHECK(hipSetDevice(device_id));
        HIP_CHECK(hipMalloc(&data, size * sizeof(T)));
        return data;
    }

    T* alloc_host() const {
        T* data;
        HIP_CHECK(hipHostMalloc(&data, size * sizeof(T)));
        return data;
    }

    void sync_with_device() const {
        if (desync_status.device_side) {
            LOG(LOG_LEVEL_INFO, "Sync with device on %p.", this);
            assert(!desync_status.host_side);
            HIP_CHECK(hipSetDevice(device_id));
            HIP_CHECK(hipDeviceSynchronize());
            HIP_CHECK(hipMemcpy(
                host_data + desync_status.start,
                device_data + desync_status.start,
                desync_status.size() * sizeof(T),
                hipMemcpyDeviceToHost
            ));
            desync_status = DesyncStatus{false, false, 0, 0};
        }
    }

    void sync_with_host() const {
        if (desync_status.host_side) {
            LOG(LOG_LEVEL_INFO, "Sync with host on %p.", this);
            assert(!desync_status.device_side);
            HIP_CHECK(hipSetDevice(device_id));
            HIP_CHECK(hipDeviceSynchronize());
            HIP_CHECK(hipMemcpy(
                device_data + desync_status.start,
                host_data + desync_status.start,
                desync_status.size() * sizeof(T),
                hipMemcpyHostToDevice
            ));
            desync_status = DesyncStatus{false, false, 0, 0};
        }
    }
};

#endif
