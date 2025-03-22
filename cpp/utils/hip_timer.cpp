#include "hip_timer.hpp"

#include <hip/amd_detail/amd_hip_runtime.h>

#include <iostream>

#include "hip_utils.hpp"
#include "logger.hpp"

using namespace std::literals;

HIPTimer::HIPTimer() {
    HIP_CHECK(hipEventCreate(&hip_start));
    HIP_CHECK(hipEventCreate(&hip_stop));
    reset();
}

void HIPTimer::reset() {
    hip_last = hip_total = cpu_last = cpu_total = 0.0;
    laps = 0;
    running = false;
}

void HIPTimer::start() {
    if (running) {
        LOG(LOG_LEVEL_WARNING, "Timer already started.");
        return;
    }
    HIP_CHECK(hipEventRecord(hip_start, NULL));
    cpu_start = std::chrono::high_resolution_clock::now();
    running = true;
}

void HIPTimer::stop() {
    if (!running) {
        LOG(LOG_LEVEL_WARNING, "Timer not started.");
        return;
    }
    HIP_CHECK(hipEventRecord(hip_stop, NULL));
    HIP_CHECK(hipEventSynchronize(hip_stop));  // Wait for completion

    HIP_CHECK(hipDeviceSynchronize());  // Ensure kernel is finished before stopping the
    // timer TODO: check redundancy with last line
    cpu_stop = std::chrono::high_resolution_clock::now();

    // Compute time
    HIP_CHECK(hipEventElapsedTime(&hip_last, hip_start, hip_stop));
    hip_total += hip_last;

    cpu_last = (cpu_stop - cpu_start) / 1.ms;
    cpu_total += cpu_last;

    laps += 1;
    running = false;
}

void HIPTimer::status() {
    std::cout << "Timer status";
    if (running) {
        std::cout << " (running, status from last finished lap)";
    }
    std::cout << ":" << std::endl;
    std::cout << "  Laps: " << laps << std::endl;
    std::cout << "  Last GPU time: " << hip_last << " ms" << std::endl;
    std::cout << "  Total GPU time: " << hip_total << " ms" << std::endl;
    std::cout << "  Last CPU time: " << cpu_last << " ms" << std::endl;
    std::cout << "  Total CPU time: " << cpu_total << " ms" << std::endl;
}

HIPTimer::~HIPTimer() noexcept {
    hipError_t status_start = hipEventDestroy(hip_start);
    hipError_t status_stop = hipEventDestroy(hip_stop);
    if (status_start != hipSuccess || status_stop != hipSuccess) {
        LOG(LOG_LEVEL_ERROR, "HIP faild destroying timer event.")
    }
}
