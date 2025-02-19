#ifndef STORAGE_H
#define STORAGE_H

#include <cstddef>

#include "device.hpp"


template <typename T, typename D> class StorageBase {
  protected:
    size_t size;
    T *data;
    D device;

  public:
    const size_t& get_size() const;
    const D &get_device() const;
    const T* const &get_data() const; // when called by const storages, i.e storage is retrieved from get_storage() of a tensor
    T* const &get_data(); // when called by one of the owning tensor
};

template <typename T, typename D> class Storage : public StorageBase<T, D> {};

template <typename T> class Storage<T, CPU> : public StorageBase<T, CPU> {
  public:
    Storage(const Storage<T,CPU>&) = delete;
    Storage(const size_t& size, const CPU& device = CPU());
    ~Storage();
    T operator[](const int& index);
};

template <typename T> class Storage<T, GPU> : public StorageBase<T, GPU> {
  public:
    Storage(const Storage<T,GPU>&) = delete;
    Storage(const size_t& size, const GPU& device = GPU());
    ~Storage();
};

#endif
