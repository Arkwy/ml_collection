#include <hip/amd_detail/amd_hip_runtime.h>
#include <hip/hip_runtime.h>

#include "../utils/hip_utils.hpp"
#include "device.hpp"
#include "storage.hpp"

using namespace std;

template <typename T> Storage<T, CPU>::Storage(const size_t& size, const CPU& device) {
    this->size = size;
    this->device = device;
    this->data = new T[this->size];
}


template <typename T> Storage<T, GPU>::Storage(const size_t& size, const GPU& device) {
    this->size = size;
    this->device = device;
    HIP_CHECK(hipSetDevice(this->device));
    HIP_CHECK(hipMalloc(&this->data, this->size * sizeof(T)));
}


template <typename T> Storage<T, CPU>::~Storage() {
    delete[] this->data;
}


template <typename T> Storage<T, GPU>::~Storage() {
    HIP_CHECK(hipSetDevice(this->device));
    HIP_CHECK(hipFree(this->data));
}


template <typename T, typename D> const D& StorageBase<T, D>::get_device() const {
    return device;
}


// template <typename T, typename D>  const T* const & StorageBase<T, D>::get_data() const {
//     return data;
// }

template <typename T, typename D>  T* const & StorageBase<T, D>::get_data() {
    return data;
}

template <typename T, typename D> const size_t& StorageBase<T, D>::get_size() const {
    return size;
}


template <typename T> T Storage<T, CPU>::operator[](const int& index) {
    assert(index < this->size);

    int i = index;

    if (index < 0) {
        i += this->size;
        assert(i >= 0);
    }

    return this->data[i];
    
}


#define INSTANTIATE_STORAGE(T, D)        \
    template class StorageBase<T, D>;    \
    template class Storage<T, D>;

INSTANTIATE_STORAGE(bool, CPU)
INSTANTIATE_STORAGE(bool, GPU)
INSTANTIATE_STORAGE(int, CPU)
INSTANTIATE_STORAGE(int, GPU)
INSTANTIATE_STORAGE(long, CPU)
INSTANTIATE_STORAGE(long, GPU)
INSTANTIATE_STORAGE(float, CPU)
INSTANTIATE_STORAGE(float, GPU)
INSTANTIATE_STORAGE(double, CPU)
INSTANTIATE_STORAGE(double, GPU)

#undef INSTANTIATE_STORAGE
