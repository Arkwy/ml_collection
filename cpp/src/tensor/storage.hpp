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
    size_t get_size() const;
    D get_device() const;
    T *get_data() const;
};

template <typename T, typename D> class Storage : public StorageBase<T, D> {};

template <typename T> class Storage<T, CPU> : public StorageBase<T, CPU> {
  public:
    Storage(const size_t size, const CPU& device = CPU());
    ~Storage();
    T operator[](const int index);
};

template <typename T> class Storage<T, GPU> : public StorageBase<T, GPU> {
  public:
    Storage(const size_t size, const GPU& device = GPU());
    ~Storage();
};

#endif
