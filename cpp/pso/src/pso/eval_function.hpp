#ifndef EVAL_FUNCTION_H
#define EVAL_FUNCTION_H

#include <hip/amd_detail/amd_hip_runtime.h>

#include <cstdio>
#include <type_traits>

#include "particles.hpp"
#include "space.hpp"

enum class PointEvaluationMode { SingleThreaded, MultiThreaded, Custom };


template <typename T>
concept SpaceType = requires {
    // This checks if `T` inherits from any specialization of `Space<D>`
    []<uint D>(Space<D>*) {}(std::declval<T*>());
};


template <PointEvaluationMode PEM>
concept is_single_threaded = std::integral_constant<bool, PEM == PointEvaluationMode::SingleThreaded>::value;


template <PointEvaluationMode PEM>
concept is_multi_threaded = std::integral_constant<bool, PEM == PointEvaluationMode::MultiThreaded>::value;


template <SpaceType S, PointEvaluationMode PEM = PointEvaluationMode::SingleThreaded, typename Derived = void>
struct EvalFunction {
    using DefinitionSpace = S;
    const DefinitionSpace definition_space;  // PSO get access to space through EvalFunction


    EvalFunction(const DefinitionSpace& space) : definition_space(space) {}


    __device__ static float eval_point(const float* const point);


    __device__ static void eval_point(const float* const point, uint dim, float& result);


    __global__ static void eval_points(const float* const points, float* const result, const int N)
        requires is_single_threaded<PEM>
    {
        int point_idx = threadIdx.x + blockDim.x * blockIdx.x;
        if (point_idx < N) {
            if constexpr (std::is_void_v<Derived>) {
                result[point_idx] = eval_point(points + point_idx * DefinitionSpace::dim);
            } else {
                result[point_idx] = Derived::eval_point(points + point_idx * DefinitionSpace::dim);
            }
        }
    }


    __global__ static void eval_points(const float* const points, float* const result, const int N)
        requires is_multi_threaded<PEM>
    {
        int point_idx = blockIdx.x;
        int dim_idx = threadIdx.x;

        assert(point_idx < N);
        
        if constexpr (std::is_void_v<Derived>) {
            eval_point(points + point_idx * DefinitionSpace::dim, dim_idx, result[point_idx]);
        } else {
            Derived::eval_point(points + point_idx * DefinitionSpace::dim, dim_idx, result[point_idx]);
        }
    }


    template <uint N>
    static void eval_points(const Particles<N, DefinitionSpace::dim>& particles)
        requires is_single_threaded<PEM>
    {
        uint device_id = particles.position.device_id();
        HIP_CHECK(hipSetDevice(device_id));

        hipDeviceProp_t props;
        HIP_CHECK(hipGetDeviceProperties(&props, device_id));
        uint max_threads_per_block = props.maxThreadsPerBlock;

        eval_points<<<(N + max_threads_per_block - 1) / max_threads_per_block, std::min(max_threads_per_block, N)>>>(
            particles.position.get_device(),
            particles.fitness.get_mut_device(),
            N
        );
    }


    template <uint N>
    static void eval_points(const Particles<N, DefinitionSpace::dim>& particles)
        requires is_multi_threaded<PEM>
    {
        uint device_id = particles.position.device_id();

        hipDeviceProp_t props;
        HIP_CHECK(hipGetDeviceProperties(&props, device_id));

        uint max_threads_per_block = props.maxThreadsPerBlock;

        if (DefinitionSpace::dim > max_threads_per_block) {
            throw std::runtime_error("Spatial dimensions limit exceeded. Please use `PointEvaluationMode::Custom`");
        }

        uint warp_size = props.warpSize;
        uint threads_per_block = warp_size * ((DefinitionSpace::dim + warp_size - 1) / warp_size);

        // TODO try finding how to have better grid and block dim while keeping eval_point definition easy
        eval_points<<<N, threads_per_block>>>(particles.position.get_device(), particles.fitness.get_mut_device(), N);
    }
};


template <typename T>
concept EvalFunctionType = requires {
    // This checks if `T` is or inherits from any specialization of `EvalFunction<PEM, S, D>`
    []<typename S, PointEvaluationMode PEM, typename D>(EvalFunction<S, PEM, D>*) {}(std::declval<T*>());
};

#endif
