#ifndef PSO_H
#define PSO_H

#include <hip/amd_detail/amd_hip_runtime.h>

#include <limits>

#include "../array/nd_array.hpp"
#include "../ops/reduce/1d.hpp"
#include "eval_function.hpp"
#include "particles.hpp"
#include "topology.hpp"


template <uint N, EvalFunctionType EF, TopologyType Tp>
struct PSO {
    constexpr static const uint D = EF::DefinitionSpace::dim;
    const EF eval_function;
    const float momentum;
    const float cognitive_coefficient;
    const float social_coefficient;

    const Particles<N, D> particles;

    PSO(const EF& eval_function,
        const float& weight,
        const float& cognitive_coefficient,
        const float& social_coefficient)
        : eval_function(eval_function),
          momentum(weight),
          cognitive_coefficient(cognitive_coefficient),
          social_coefficient(social_coefficient),
          particles() {
        eval_function.definition_space.sample(particles.position);
        eval_function.definition_space.sample(particles.velocity);
        init_fitness();
    }

    void step() const {
        // TODO large D improvement and MultiThreaded/Custom support
        update_velocity();
        update_position();
        update_fitness();
    }

    void run(const uint& iterations) const {
        for (uint i = 0; i < iterations; i++) {
            step();
        }
    }

    std::pair<NDArray<float, D>, float> best() const {
        uint best_fitness_idx = reduce_1d<ArgMin<float>>(particles.best_known_fitness);
        return std::pair<NDArray<float, D>, float>(
            particles.best_known_position[best_fitness_idx],
            particles.best_known_fitness[best_fitness_idx]
        );
    }


  private:
    void init_fitness() const {
        EF::eval_points(particles);

        particles.best_position.device_copy(particles.position);
        particles.best_fitness.device_copy(particles.fitness);

        particles.best_known_fitness.fill(std::numeric_limits<float>::infinity());

        Tp::template share_fitness<N, D>(
            particles.fitness,
            particles.position,
            particles.best_known_fitness,
            particles.best_known_position
        );
    }


    void update_fitness() const {
        uint device_id = particles.position.device_id();
        HIP_CHECK(hipSetDevice(device_id));

        hipDeviceProp_t props;
        HIP_CHECK(hipGetDeviceProperties(&props, device_id));
        uint max_threads_per_block = props.maxThreadsPerBlock;

        EF::eval_points(particles);

        update_pb<<<(N + max_threads_per_block - 1) / max_threads_per_block, std::min(max_threads_per_block, N)>>>(
            particles.fitness.get_device(),
            particles.position.get_device(),
            particles.best_fitness.get_mut_device(),
            particles.best_position.get_mut_device()
        );

        Tp::template share_fitness<N, D>(
            particles.fitness,
            particles.position,
            particles.best_known_fitness,
            particles.best_known_position
        );
    }


    __global__ static void update_pb(
        const float* const fitness, const float* const position, float* const best_fitness, float* const best_position
    ) {
        uint idx = threadIdx.x + blockDim.x * blockIdx.x;

        if (idx >= N) return;  // no sync nor shared memory, can return early safely

        if (fitness[idx] < best_fitness[idx]) {
            best_fitness[idx] = fitness[idx];
#pragma unroll
            for (uint i = 0; i < D; i++) {
                uint coord_idx = i + D * idx;
                best_position[coord_idx] = position[coord_idx];
            }
        }
    }

    void update_velocity() const {
        // TODO optimise for large D

        uint device_id = particles.position.device_id();
        HIP_CHECK(hipSetDevice(device_id));

        hipDeviceProp_t props;
        HIP_CHECK(hipGetDeviceProperties(&props, device_id));
        uint max_threads_per_block = props.maxThreadsPerBlock;

        DeviceArray<float> r_p(N, device_id); // TODO stop reallocating each step
        DeviceArray<float> r_g(N, device_id);

        rocrand_cpp::default_random_engine engine;
        rocrand_cpp::uniform_real_distribution distribution;

        distribution(engine, r_p.data, N * sizeof(float));
        distribution(engine, r_g.data, N * sizeof(float));

        update_velocity_kernel<<<
            (N + max_threads_per_block - 1) / max_threads_per_block,
            std::min(max_threads_per_block, N)>>>(
            particles.position.get_device(),
            particles.velocity.get_mut_device(),
            momentum,
            particles.best_position.get_device(),
            r_p.data,
            cognitive_coefficient,
            particles.best_known_position.get_device(),
            r_g.data,
            social_coefficient
        );
    }

    __global__ static void update_velocity_kernel(
        const float* const position,
        float* const velocity,
        const float momentum,
        const float* const best_position,
        const float* const r_p,
        const float cognitive_coefficient,
        const float* const best_known_position,
        const float* const r_g,
        const float social_coefficient
    ) {
        uint idx = threadIdx.x + blockDim.x * blockIdx.x;

        if (idx >= N) return;  // no sync nor shared memory, can return early safely

#pragma unroll
        for (uint i = 0; i < D; i++) {
            uint coord_idx = i + D * idx;
            float coord_i = position[coord_idx];
            velocity[coord_idx] = (momentum * velocity[coord_idx]) +
                                  (cognitive_coefficient * r_p[idx] * (best_position[coord_idx] - coord_i)) +
                                  (social_coefficient * r_g[idx] * (best_known_position[coord_idx] - coord_i));
        }
    }

    void update_position() const {
        // TODO optimise for large D

        uint device_id = particles.position.device_id();
        HIP_CHECK(hipSetDevice(device_id));

        hipDeviceProp_t props;
        HIP_CHECK(hipGetDeviceProperties(&props, device_id));
        uint max_threads_per_block = props.maxThreadsPerBlock;

        update_position_kernel<<<
            (N + max_threads_per_block - 1) / max_threads_per_block,
            std::min(max_threads_per_block, N)>>>(particles.position.get_mut_device(), particles.velocity.get_device());

        eval_function.definition_space.bound(particles.position); // make sure particles don't leave definition space
    }

    __global__ static void update_position_kernel(float* const position, const float* const velocity) {
        uint idx = threadIdx.x + blockDim.x * blockIdx.x;

        if (idx >= N) return;  // no sync nor shared memory, can return early safely

#pragma unroll
        for (uint i = 0; i < D; i++) {
            uint coord_idx = i + D * idx;
            position[coord_idx] += velocity[coord_idx];
        }
    }
};

#endif
