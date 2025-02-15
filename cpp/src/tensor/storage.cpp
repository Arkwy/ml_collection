#include <hip/amd_detail/amd_hip_runtime.h>
#include <hip/hip_runtime.h>

#include "../utils/hip_utils.hpp"
#include "device.hpp"
#include "storage.hpp"

using namespace std;

template <typename T> Storage<T, CPU>::Storage(const size_t size, const CPU device) : size(size), device(device) {
    data = new T[size];
}

template <typename T> Storage<T, GPU>::Storage(const size_t size, const GPU device) : size(size), device(device) {
    HIP_CHECK(hipSetDevice(device));
    HIP_CHECK(hipMalloc(&data, size * sizeof(T)));
}

template <typename T> Storage<T, CPU>::~Storage() {
    delete[] data;
}
template <typename T> Storage<T, GPU>::~Storage() {
    HIP_CHECK(hipSetDevice(device));
    HIP_CHECK(hipFree(data));
}

template <typename T, typename D> D Storage<T, D>::get_device() const {
    return device;
}

template <typename T, typename D> size_t Storage<T, D>::get_size() const {
    return size;
}


template <typename T> T Storage<T, CPU>::operator[](const int index) {
    assert(index < size);

    int i = index;

    if (index < 0) {
        i += size;
        assert(i >= 0);
    }

    return data[i];
    
}
// template <typename T> void Storage<T>::copy_data(Storage<T>& dst) const {
    

// }


template class Storage<bool, CPU>;
template class Storage<bool, GPU>;
template class Storage<int, CPU>;
template class Storage<int, GPU>;
template class Storage<long, CPU>;
template class Storage<long, GPU>;
template class Storage<float, CPU>;
template class Storage<float, GPU>;
template class Storage<double, CPU>;
template class Storage<double, GPU>;
