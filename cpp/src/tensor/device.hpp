#ifndef DEVICE_H
#define DEVICE_H

#include <cstddef>

class Device {
  protected:
    int id;

  public:
    int get_id() const;
    operator int() const;
};

class GPU : public Device {
  public:
    GPU(const size_t id = 0);
};

class CPU : public Device {
  public:
    CPU();
};

#endif
