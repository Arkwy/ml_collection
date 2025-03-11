#ifndef HIP_TIMER_H
#define HIP_TIMER_H

#include <chrono>
#include <hip/amd_detail/amd_hip_runtime.h>
#include <hip/hip_runtime.h>

class HIPTimer {
  private:
    hipEvent_t hip_start, hip_stop;
    std::chrono::time_point<std::chrono::high_resolution_clock> cpu_start, cpu_stop;
    float hip_last, hip_total, cpu_last, cpu_total;
    uint laps;
    bool running;

  public:
    HIPTimer();
    ~HIPTimer() noexcept;
    void reset();
    void start();
    void stop();
    void status();
};

#endif
