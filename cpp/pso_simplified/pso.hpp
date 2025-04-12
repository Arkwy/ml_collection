#ifndef PSO_H
#define PSO_H

// Ensure some macro constants are defined
#ifdef WARP_SIZE
static_assert(std::is_integral<decltype(WARP_SIZE)>::value, "`WARP_SIZE` must be an integral type.");
static_assert(WARP_SIZE >= 1, "`WARP_SIZE` must be >= 1.");

#ifdef MAX_THREADS_PER_BLOCK
static_assert(
    std::is_integral<decltype(MAX_THREADS_PER_BLOCK)>::value, "`MAX_THREADS_PER_BLOCK` must be an integral type."
);
static_assert(MAX_THREADS_PER_BLOCK >= 1, "`MAX_THREADS_PER_BLOCK` must be >= 1.");


#include <hip/hip_runtime.h>

#include <algorithm>
#include <bit>
#include <concepts>
#include <limits>
#include <type_traits>

#include "../utils/hip_utils.hpp"
#include "array.hpp"
#include "eval_function.hpp"
#include "pso_O0.hip"  // nearly no optimization
// #include "pso_O1.hip" // use of shared memory TODO
// #include "pso_O2.hip" // no bank conflits TODO
// #include "pso_O3.hip" // avoidance of warp divergence TODO
// #include "pso_O4.hip" // loop unrolling TODO
// #include "pso_O5.hip" // use of warp level communication TODO

template <EvalFunction EF>
struct PSO {
    constexpr static const uint dim = EF::dim;

    const uint n;
    const float momentum;
    const float cognitive_coefficient;
    const float social_coefficient;


    PSO(const uint& n, const float& momentum, const float& cognitive_coefficient, const float& social_coefficient)
        : n(n),
          momentum(momentum),
          cognitive_coefficient(cognitive_coefficient),
          social_coefficient(social_coefficient),
          position(n * EF::dim),
          velocity(n * EF::dim),
          fitness(n),
          best_fitness(n),
          best_position(n * EF::dim),
          best_known_fitness(n),
          best_known_position(n * EF::dim),
          r_p(n),
          r_g(n),
          device_state_ptr(alloc_device_state()) {}


    ~PSO() { HIP_CHECK_NOEXCEPT(hipFree(device_state_ptr)); }


    void run(const uint& iterations) {
        HIP_CHECK(hipMemcpy(&(device_state_ptr->iterations), &iterations, sizeof(uint), hipMemcpyHostToDevice));
        void* kernel_args[] = {&device_state_ptr};

        dim3 grid_dim((n + MAX_THREADS_PER_BLOCK - 1) / MAX_THREADS_PER_BLOCK);
        dim3 block_dim(std::min(n, (uint)MAX_THREADS_PER_BLOCK));

        HIP_CHECK(hipLaunchCooperativeKernel(pso_kernel<EF>, grid_dim, block_dim, kernel_args, 0, hipStreamDefault));
        HIP_CHECK(hipDeviceSynchronize());
    }


    std::pair<float, std::array<float, EF::dim>> get_best() {
        DeviceArray<uint> idxs_buffer(std::bit_floor(n));
        DeviceArray<float> d_best_fitness(1);
        DeviceArray<float> d_best_position(EF::dim);


        dim3 grid_dim((n + MAX_THREADS_PER_BLOCK - 1) / MAX_THREADS_PER_BLOCK);
        dim3 block_dim(std::min(n, (uint)MAX_THREADS_PER_BLOCK));

        void* kernel_args[] = {
            &device_state_ptr,
            (void*)&(d_best_fitness.data),
            (void*)&(d_best_position.data),
            (void*)&(idxs_buffer.data)
        };

        HIP_CHECK(
            hipLaunchCooperativeKernel(best_position_and_fitness, grid_dim, block_dim, kernel_args, 0, hipStreamDefault)
        );

        float h_best_fitness;
        std::array<float, EF::dim> h_best_position;

        HIP_CHECK(hipDeviceSynchronize());

        HIP_CHECK(hipMemcpy(&h_best_fitness, d_best_fitness.data, sizeof(float), hipMemcpyDeviceToHost));
        HIP_CHECK(
            hipMemcpy(h_best_position.data(), d_best_position.data, EF::dim * sizeof(float), hipMemcpyDeviceToHost)
        );

        return std::pair<float, std::array<float, EF::dim>>(h_best_fitness, h_best_position);
    }


  private:
    DeviceArray<float> position;
    DeviceArray<float> velocity;
    DeviceArray<float> fitness;
    DeviceArray<float> best_fitness;
    DeviceArray<float> best_position;
    DeviceArray<float> best_known_fitness;
    DeviceArray<float> best_known_position;
    DeviceArray<float> r_p;
    DeviceArray<float> r_g;
    State* device_state_ptr;


    State* alloc_device_state() {
        State* device_state_ptr = nullptr;
        HIP_CHECK(hipMalloc(&device_state_ptr, sizeof(State)));

        State state{
            0,
            false,
            0,
            dim,
            n,
            momentum,
            cognitive_coefficient,
            social_coefficient,
            position.data,
            velocity.data,
            fitness.data,
            best_fitness.data,
            best_position.data,
            best_known_fitness.data,
            best_known_position.data,
            r_p.data,
            r_g.data,
        };
        HIP_CHECK(hipMemcpy(device_state_ptr, &state, sizeof(State), hipMemcpyHostToDevice));

        return device_state_ptr;
    }
};


#else
static_assert(false, "`MAX_THREADS_PER_BLOCK` not defined");
#endif

#else
static_assert(false, "`WARP_SIZE` not defined");
#endif

#endif
