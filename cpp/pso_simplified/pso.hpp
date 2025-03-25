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
#include <concepts>
#include <type_traits>

#include "../utils/hip_utils.hpp"
#include "eval_function.hpp"
#include "pso.hip"



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
          position(alloc_device_array(n * EF::dim)),
          velocity(alloc_device_array(n * EF::dim)),
          fitness(alloc_device_array(n)),
          best_fitness(alloc_device_array(n)),
          best_position(alloc_device_array(n * EF::dim)),
          best_known_fitness(alloc_device_array(n)),
          best_known_position(alloc_device_array(n * EF::dim)),
          r_p(alloc_device_array(n)),
          r_g(alloc_device_array(n)),
          device_state_ptr(alloc_device_state()) {}


    ~PSO() {
        HIP_CHECK_NOEXCEPT(hipFree(position));
        HIP_CHECK_NOEXCEPT(hipFree(velocity));
        HIP_CHECK_NOEXCEPT(hipFree(fitness));
        HIP_CHECK_NOEXCEPT(hipFree(best_fitness));
        HIP_CHECK_NOEXCEPT(hipFree(best_position));
        HIP_CHECK_NOEXCEPT(hipFree(best_known_fitness));
        HIP_CHECK_NOEXCEPT(hipFree(best_known_position));
        HIP_CHECK_NOEXCEPT(hipFree(r_p));
        HIP_CHECK_NOEXCEPT(hipFree(r_g));
        HIP_CHECK_NOEXCEPT(hipFree(device_state_ptr));
    }

    void run(const uint& iterations) {
        HIP_CHECK(hipMemcpy(&(device_state_ptr->iterations), &iterations, sizeof(uint), hipMemcpyHostToDevice));
        void* kernel_args[] = {(void*) &device_state_ptr};

        dim3 grid_dim((n + MAX_THREADS_PER_BLOCK - 1) / MAX_THREADS_PER_BLOCK);
        dim3 block_dim(std::min(n, (uint)MAX_THREADS_PER_BLOCK));

        HIP_CHECK(hipLaunchCooperativeKernel(
            pso_kernel<EF>,
            grid_dim,
            block_dim,
            kernel_args,
            0,
            hipStreamDefault
        ));
        HIP_CHECK(hipDeviceSynchronize());
    }


  private:
    float* const position;
    float* const velocity;
    float* const fitness;
    float* const best_fitness;
    float* const best_position;
    float* const best_known_fitness;
    float* const best_known_position;
    float* const r_p;
    float* const r_g;
    State* device_state_ptr;


    float* alloc_device_array(const size_t& size) {
        float* arr = nullptr;
        HIP_CHECK(hipMalloc(&arr, size * sizeof(float)));
        return arr;
    }

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
            position,
            velocity,
            fitness,
            best_fitness,
            best_position,
            best_known_fitness,
            best_known_position,
            r_p,
            r_g
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
