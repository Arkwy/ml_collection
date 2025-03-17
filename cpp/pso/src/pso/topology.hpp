#ifndef PSO_TOPOLOGY_H
#define PSO_TOPOLOGY_H

#include <hip/amd_detail/amd_hip_runtime.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <stdexcept>

#include "../array/nd_array.hpp"
#include "../ops/reduce/1d.hpp"

enum class TopologyCategory {
    GLOBAL,
    STAR,
    RING,
};

template <TopologyCategory T>
struct Topology {
    constexpr static const TopologyCategory category = T;

    template <uint N, uint D>
    static void share_fitness(
        const NDArray<float, N>& fitness,
        const NDArray<float, N, D>& position,
        const NDArray<float, N>& best_known_fitness,
        const NDArray<float, N, D>& best_known_position
    );
};

template <>
struct Topology<TopologyCategory::GLOBAL> {
    template <uint N, uint D>
    static void share_fitness(
        const NDArray<float, N>& fitness,
        const NDArray<float, N, D>& position,
        const NDArray<float, N>& best_known_fitness,
        const NDArray<float, N, D>& best_known_position
    ) {
        uint best_fitness_idx = reduce_1d<ArgMin<float>, N>(fitness);
        if (fitness[best_fitness_idx] < best_known_fitness[0]) {   // best_known_fitness is the same for every particle
            best_known_fitness.fill(fitness[best_fitness_idx]);    // TODO use GPU fill when implemeted
            best_known_position.fill(position[best_fitness_idx]);  // TODO same
        }
    }
};


template <>
struct Topology<TopologyCategory::STAR> {
    template <uint N, uint D>
    static void share_fitness(
        const NDArray<float, N>& fitness,
        const NDArray<float, N, D>& position,
        const NDArray<float, N>& best_known_fitness,
        const NDArray<float, N, D>& best_known_position
    ) {
        assert(best_known_fitness.device_id() == best_known_position.device_id());
        HIP_CHECK(hipSetDevice(best_known_fitness.device_id()));
        hipDeviceProp_t props;
        HIP_CHECK(hipGetDeviceProperties(&props, best_known_position.device_id()));
        uint max_threads_per_block = props.maxThreadsPerBlock;

        uint best_step_fitness_idx = reduce_1d<ArgMin<float>, N>(fitness);

        // TODO try implementing faster kernel for cases where D is large (use PointEvaluationMode to decide which mode
        // to use ?)
        share_fitness_kernel<N, D>
            <<<(N + max_threads_per_block - 1) / max_threads_per_block, std::min(max_threads_per_block, N)>>>(
                best_step_fitness_idx,
                fitness.get_device(),
                position.get_device(),
                best_known_fitness.get_mut_device(),
                best_known_position.get_mut_device()
            );
    }

  private:
    template <uint N, uint D>
    __global__ static void share_fitness_kernel(
        const uint best_fitness_idx,
        const float* const fitness,
        const float* const position,
        float* const best_known_fitness,
        float* const best_known_position
    ) {
        uint idx = threadIdx.x + blockDim.x * blockIdx.x;

        if (idx >= N) return;  // no sync nor shared memory, can return early safely

        // load data read multiple times in registers
        const float f = fitness[idx];
        const float f0 = fitness[0];
        const float bkf = best_known_fitness[idx];

        if (idx == 0) {
            if (fitness[best_fitness_idx] < bkf) {
                best_known_fitness[idx] = fitness[best_fitness_idx];
#pragma unroll
                for (uint i = 0; i < D; i++) {
                    best_known_position[i] = position[best_fitness_idx * D + i];
                }
            }
        } else {
            if (f < bkf || f0 < bkf) {
                if (f0 < f) {
                    best_known_fitness[idx] = f0;
#pragma unroll
                    for (uint i = 0; i < D; i++) {
                        best_known_position[i + idx * D] = position[i];
                    }
                } else {
                    best_known_fitness[idx] = f;
#pragma unroll
                    for (uint i = 0; i < D; i++) {
                        best_known_position[i + idx * D] = position[i + idx * D];
                    }
                }
            }
        }
    }
};


template <>
struct Topology<TopologyCategory::RING> {
    template <uint N, uint D>
    static void share_fitness(
        const NDArray<float, N>& fitness,
        const NDArray<float, N, D>& position,
        const NDArray<float, N>& best_known_fitness,
        const NDArray<float, N, D>& best_known_position
    ) {
        assert(best_known_fitness.device_id() == best_known_position.device_id());
        HIP_CHECK(hipSetDevice(best_known_fitness.device_id()));
        hipDeviceProp_t props;
        HIP_CHECK(hipGetDeviceProperties(&props, best_known_position.device_id()));
        uint max_threads_per_block = props.maxThreadsPerBlock;

        // TODO try implementing faster kernel for cases where D is large (use PointEvaluationMode to decide which mode
        // to use ?)
        share_fitness_kernel<N, D>
            <<<(N + max_threads_per_block - 1) / max_threads_per_block, std::min(max_threads_per_block, N)>>>(
                fitness.get_device(),
                position.get_device(),
                best_known_fitness.get_mut_device(),
                best_known_position.get_mut_device()
            );
    }

  private:
    template <uint N, uint D>
    __global__ static void share_fitness_kernel(
        const float* const fitness,
        const float* const position,
        float* const best_known_fitness,
        float* const best_known_position
    ) {
        uint idx = threadIdx.x + blockDim.x * blockIdx.x;

        if (idx >= N) return;  // no sync nor shared memory, can return early safely

        uint prev_idx = idx > 0 ? idx - 1 : N - 1;
        uint next_idx = idx < N - 1 ? idx + 1 : 0;

        // load data read multiple times in registers
        const float p_f = fitness[prev_idx];
        const float f = fitness[idx];
        const float n_f = fitness[next_idx];
        const float bkf = best_known_fitness[idx];

        if (f < bkf || p_f < bkf || n_f < bkf) {
            if (f <= p_f && f <= n_f) { // current has best fitness
                best_known_fitness[idx] = f;
#pragma unroll
                for (uint i = 0; i < D; i++) {
                    best_known_position[i + idx * D] = position[i + idx * D];
                }
            } else if (p_f <= f && p_f <= n_f) { // previous has best fitness
                best_known_fitness[idx] = p_f;
#pragma unroll
                for (uint i = 0; i < D; i++) {
                    best_known_position[i + idx * D] = position[i + prev_idx * D];
                }
            } else if (n_f <= f && n_f <= p_f) { // next has best fitness
                best_known_fitness[idx] = n_f;
#pragma unroll
                for (uint i = 0; i < D; i++) {
                    best_known_position[i + idx * D] = position[i + next_idx * D];
                }
            } else { // should never happen
                assert(false);
            }
        }
    }
};

inline const char* topology_to_string(TopologyCategory topology_type) {
    switch (topology_type) {
        case TopologyCategory::GLOBAL:
            return "GLOBAl";
            break;
        default:
            return "UNKNOWN";
    }
}


template <typename T>
concept TopologyType = requires {
    // This checks if `T` is or inherits from any specialization of `Topology<Tp>`
    []<TopologyCategory Tp>(Topology<Tp>*) {}(std::declval<T*>());
};


#endif
