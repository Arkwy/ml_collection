#include <iostream>

#define WARP_SIZE 32
#define MAX_THREADS_PER_BLOCK 1024

#include "pso.hpp"

struct MyEvalFunction {
    constexpr static const uint32_t dim = 3;
    static __device__ float eval(const float* point) {
        printf("%d -> %f, %f, %f\n", this_grid().thread_rank(), point[0], point[1], point[2]);
        return point[0] * point[0] + point[1] * point[1] + point[2] * point[2];
    }
};


int main() {
    int supports_coop;
    HIP_CHECK(hipDeviceGetAttribute(&supports_coop, hipDeviceAttributeCooperativeLaunch, 0));
    if (!supports_coop) {
        std::cout << "Error: Cooperative kernels not supported on this GPU!" << std::endl;
        return -1;
    }

    const uint number_of_particles = 5;
    const float momentum = 0.5;
    const float cognitive_coefficient = 0.2;
    const float social_coefficient = 0.6;

    PSO<MyEvalFunction> pso(number_of_particles, momentum, cognitive_coefficient, social_coefficient);
    pso.run(10);


    // TODO auto [best_fitness, best_point] = pso.get_best();

    return 0;
}
