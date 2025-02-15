#ifndef STORAGE_H
#define STORAGE_H

#include <cstddef>

#include "device.hpp"



// TODO use StorageBase and inherit from it

template <typename T, typename D> class Storage {
  private:
    size_t size;
    T *data;
    D device;

  public:
    size_t get_size() const;
    D get_device() const;
    // Storage<T> to(const Device device) const;
    // void copy_data_to(Storage<T>& dst) const;
    // void copy_sub_data_to(Storage<T>& dst, size_t n, size_t src_offset, size_t dst_offset);
};

template <typename T> class Storage<T, CPU> {
  private:
    size_t size;
    T *data;
    CPU device;

  public:
    Storage(const size_t size, const CPU device = CPU());
    ~Storage();
    T operator[](const int index);
};

template <typename T> class Storage<T, GPU> {
  private:
    size_t size;
    T *data;
    GPU device;

  public:
    Storage(const size_t size, const GPU device = GPU());
    ~Storage();
};

#endif
