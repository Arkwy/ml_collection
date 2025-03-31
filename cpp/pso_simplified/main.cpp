#include <iostream>

#define WARP_SIZE 32
#define MAX_THREADS_PER_BLOCK 1024

#include <cstddef>
#include "pso.hpp"

struct MyEvalFunction {
    constexpr static const uint32_t dim = 2;
    static __device__ float eval(const float* point) {
        float p0 = point[0] - 0.5;
        return (p0 * p0 + point[1] * point[1]); // + point[2] * point[2]);
    }
};


int main() {
    int supports_coop;
    HIP_CHECK(hipDeviceGetAttribute(&supports_coop, hipDeviceAttributeCooperativeLaunch, 0));
    if (!supports_coop) {
        std::cout << "Error: Cooperative kernels not supported on this GPU!" << std::endl;
        return -1;
    }

    const uint number_of_particles = 100;

    const float momentum = .2;
    const float cognitive_coefficient = .2;
    const float social_coefficient = .6;

    PSO<MyEvalFunction> pso(number_of_particles, momentum, cognitive_coefficient, social_coefficient);

    pso.run(100);
    auto [best_fitness, best_point] = pso.get_best();

    std::cout << "(" << best_point[0] << ", " << best_point[1] << ") -> " << best_fitness << std::endl;

    return 0;
}
